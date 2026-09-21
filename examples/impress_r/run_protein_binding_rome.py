import copy
import csv
import os
import shutil
import asyncio
import tempfile
from typing import Dict, Any, Optional, List

from rhapsody.backends import DragonExecutionBackendV3
from concurrent.futures import ProcessPoolExecutor
from rhapsody.backends import ConcurrentExecutionBackend

from impress import PipelineSetup
from impress import ImpressManager
from impress.pipelines.protein_binding import ProteinBindingPipeline

import rome

# The ProteinMPNN trainer ships with this example (mpnn.py, beside the inference
# mpnn_wrapper.py), not with the framework. Import it whether this file is run as
# a script from inside the example dir or imported as examples.impress_r.*.
try:
    from examples.impress_r.mpnn import (
        ProteinMPNNConfig,
        ProteinMPNNTrainer,
        percentile_sampler,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpnn import ProteinMPNNConfig, ProteinMPNNTrainer, percentile_sampler

# The dauparas/ProteinMPNN checkout IMPRESS runs (the same `-mpnn` path). Set
# ROME_MPNN_REPO to it for a real training round.
MPNN_REPO = os.environ.get(
    "ROME_MPNN_REPO",
    f"/work/nvme/bdyk/{os.environ.get('USER', 'user')}/ProteinMPNN",
)


async def adaptive_criteria(current_score: float, previous_score: float) -> bool:
    """
    Determine if protein quality has degraded requiring pipeline migration.

    Compares current and previous protein scores to decide if a protein
    should be moved to a new pipeline for optimization.

    Args:
        current_score: Current protein structure quality score
        previous_score: Previous protein structure quality score

    Returns:
        True if quality has degraded (score increased), False otherwise
    """
    return current_score > previous_score


def make_adaptive_decision(rome_manager: rome.Manager, stage_dir: str):
    """IMPRESS's adaptive_decision with the two ROME hooks folded in.

    ``rome_manager`` and ``stage_dir`` are captured because ``adaptive_fn`` has a
    fixed one-argument signature. Everything below the two hooks is IMPRESS's own
    migration logic, unchanged.
    """
    os.makedirs(stage_dir, exist_ok=True)

    async def adaptive_decision(pipeline: ProteinBindingPipeline) -> Optional[Dict[str, Any]]:
        """
        Adaptive function for AlphaFold protein structure optimization.

        Evaluates protein scores and creates child pipelines for proteins with
        degraded quality. Implements adaptive optimization strategy by moving
        underperforming proteins to new pipeline instances.
        """
        MAX_SUB_PIPELINES: int = 3
        sub_iter_seqs: Dict[str, str] = {}

        # Read current scores from CSV
        file_name = f'af_stats_{pipeline.name}_pass_{pipeline.passes}.csv'
        accepted = 0
        with open(file_name) as fd:
            for row in csv.DictReader(fd):          # ID, avg_plddt, ptm, avg_pae
                protein = row['ID'].split('.')[0]
                # IMPRESS's own migration criterion is the interface pAE.
                pipeline.current_scores[protein] = float(row['avg_pae'])

                # -- ROME HOOK 1: contribute this design to the corpus ------
                # The prediction at output_path_af/{protein}.pdb is keyed by
                # pipeline (not pass) and is deleted on migration, so copy it
                # aside before recording it — otherwise the corpus would point at
                # a file whose contents change under it.
                src = os.path.join(pipeline.output_path_af, f'{protein}.pdb')
                if not os.path.exists(src):
                    continue
                staged = os.path.join(
                    stage_dir, f'{pipeline.name}_pass{pipeline.passes}_{protein}.pdb'
                )
                shutil.copyfile(src, staged)
                ranked = pipeline.iter_seqs.get(protein) or []
                sequence = ranked[pipeline.seq_rank][0] if len(ranked) > pipeline.seq_rank else ''
                uid = rome_manager.add_training_data(
                    path=staged,
                    sequence=sequence,
                    backbone_id=protein,               # cluster on the target
                    pLDDT=float(row['avg_plddt']),
                    pTM=float(row['ptm']),
                    pAE=float(row['avg_pae']),
                    score=float(row['avg_plddt']),
                )
                accepted += uid is not None

        # -- ROME HOOK 2: collect the improved model -----------------------
        # get_current_model() returns the latest *versioned* checkpoint
        # ({model_name}_v{n}.pt, kept per round). With publish_into_repo the round
        # also copies it onto the checkout's fixed weights pointer, so the next
        # MPNN pass picks it up with no wrapper change; this just reports what
        # ROME currently has (the basename shows the version).
        weights = rome_manager.get_current_model()
        pipeline.logger.pipeline_log(
            f'ROME: corpus {rome_manager.data.total_count} (+{accepted} this pass) | '
            f'{rome_manager.get_training_status().name}'
            + (f' | model {os.path.basename(weights)}' if weights else '')
        )

        # ---------------------------------------------------------------------
        # IMPRESS's own migration logic below — unchanged.
        # ---------------------------------------------------------------------

        # First pass — just save current scores as previous
        if not pipeline.previous_scores:
            pipeline.logger.pipeline_log('Saving current scores as previous and returning')
            pipeline.previous_scores = copy.deepcopy(pipeline.current_scores)
            return

        # Identify proteins that got worse
        sub_iter_seqs = {}
        for protein, curr_score in pipeline.current_scores.items():
            if protein not in pipeline.iter_seqs:
                continue

            decision = await adaptive_criteria(curr_score, pipeline.previous_scores[protein])
            pipeline.logger.pipeline_log(f'Adaptive descision: {decision}')

            if decision:
                sub_iter_seqs[protein] = pipeline.iter_seqs.pop(protein)

        # Spawn a new pipeline for bad proteins
        if sub_iter_seqs and pipeline.sub_order < MAX_SUB_PIPELINES:
            new_name: str = f"{pipeline.name}_sub{pipeline.sub_order + 1}"

            pipeline.set_up_new_pipeline_dirs(new_name)

            # Copy PDB files for bad proteins
            for protein in sub_iter_seqs:
                src = f'{pipeline.output_path_af}/{protein}.pdb'
                dst = f'{pipeline.base_path}/{new_name}_in/{protein}.pdb'
                shutil.copyfile(src, dst)

            # Build a request for a new pipeline
            new_config = {
                'name': new_name,
                'type': type(pipeline),
                # children carry the same ROME hooks
                'adaptive_fn': make_adaptive_decision(rome_manager, stage_dir),
                'config': {
                    'is_child': True,
                    'start_pass': pipeline.passes,
                    'passes': pipeline.passes,
                    'iter_seqs': sub_iter_seqs,
                    'seq_rank': pipeline.seq_rank + 1,
                    'sub_order': pipeline.sub_order + 1,
                    'previous_scores': copy.deepcopy(pipeline.previous_scores),
                }
            }

            # Submit the request
            pipeline.submit_child_pipeline_request(new_config)

            pipeline.finalize(sub_iter_seqs)

            if not pipeline.fasta_list_2:
                pipeline.kill_parent = True
        else:
            pipeline.previous_scores = copy.deepcopy(pipeline.current_scores)

    return adaptive_decision


async def _make_backend():
    """An execution backend that runs tasks in their own processes.

    ROME must *not* run its training rounds in the manager's own process: a
    fine-tune loads ProteinMPNN onto the GPU, and an in-process (local) backend
    would leave that CUDA context and the model resident in the long-lived
    campaign driver for the whole run. A process-based backend runs each round
    in a task process that exits when the round finishes, so the VRAM is
    released. On Delta that is Dragon; ``ROME_BACKEND=concurrent`` selects the
    ProcessPoolExecutor backend for a laptop/login-node smoke test.
    """
    if os.environ.get("ROME_BACKEND", "dragon").lower() == "concurrent":
        return await ConcurrentExecutionBackend(ProcessPoolExecutor())
    return await DragonExecutionBackendV3()


def _build_trainer(checkpoint_dir: str):
    """ProteinMPNN trainer by default; a dummy sleeper for a wiring smoke test.

    ROME_TRAINER=dummy runs without torch, a GPU, or the checkout — use it to
    prove the campaign wiring end to end before turning the real fine-tune on.
    """
    want = os.environ.get('ROME_TRAINER', 'mpnn').lower()
    if want == 'mpnn' and os.path.isdir(MPNN_REPO):
        return ProteinMPNNTrainer(ProteinMPNNConfig(
            mpnn_repo=MPNN_REPO,
            initial_weights=os.path.join(MPNN_REPO, 'vanilla_model_weights', 'v_48_020.pt'),
            model_name='v_48_020',
            # fine-tune the weights IMPRESS runs and publish back into the repo,
            # so the next MPNN pass picks them up with no wrapper change
            publish_into_repo=True,
        ), gpus=1)

    if want == 'mpnn':
        print(f"[ROME] ROME_MPNN_REPO={MPNN_REPO!r} not found; "
              "falling back to the dummy trainer (set ROME_MPNN_REPO or "
              "ROME_TRAINER=dummy to silence this).")
    from rome.dummy import DummyTrainer

    return DummyTrainer(train_seconds=1.0, gpus=0)


async def impress_protein_bind() -> None:
    """
    Execute protein binding analysis with adaptive optimization.

    Creates and manages multiple ProteinBindingPipeline instances with
    adaptive optimization capabilities. Each pipeline can spawn child
    pipelines based on protein quality degradation.
    """
    workdir = tempfile.mkdtemp(prefix='impress_r_')

    # ROME gets its OWN process-based backend, so its training rounds run as
    # tasks in their own processes rather than inside this driver — otherwise a
    # fine-tune's GPU allocation would stay resident in the campaign driver for
    # the whole run (see _make_backend). It builds its engine on this backend at
    # start() and shuts it down at stop(), so its training runs independently of
    # IMPRESS's tasks. A round fires once min_samples designs have accumulated.
    rome_backend = await _make_backend()
    rome_manager = rome.Manager(
        backend=rome_backend,
        data_config=rome.DataConfig(
            # The campaign contributes ~one scored design per pipeline per pass,
            # so the corpus grows slowly; percentile_sampler needs no score
            # calibration (see docs/impress.md).
            min_samples=int(os.environ.get('ROME_MIN_SAMPLES', 4)),
            sample_func=percentile_sampler(0.33),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=_build_trainer(os.path.join(workdir, 'checkpoints')),
            checkpoint_dir=os.path.join(workdir, 'checkpoints'),
            poll_interval=1.0,
            # On the Dragon backend a finished round's result can be delivered
            # late while a task runs; publish from disk after this grace.
            result_fallback_seconds=float(os.environ.get('ROME_FALLBACK', 60)),
        ),
    )
    await rome_manager.start()

    backend = await _make_backend()
    manager: ImpressManager = ImpressManager(execution_backend=backend)

    adaptive_fn = make_adaptive_decision(rome_manager, os.path.join(workdir, 'designs'))
    pipeline_setups: List[PipelineSetup] = [
        PipelineSetup(
            name='p1',
            type=ProteinBindingPipeline,
            adaptive_fn=adaptive_fn
        )
    ]

    try:
        await manager.start(pipeline_setups=pipeline_setups)
        print('\nROME:', rome_manager.report())
    finally:
        await manager.flow.shutdown()
        await rome_manager.stop()


if __name__ == "__main__":
    asyncio.run(impress_protein_bind())
