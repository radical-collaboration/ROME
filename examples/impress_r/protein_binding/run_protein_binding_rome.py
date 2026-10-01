"""Protein binding pipeline with ROME-A adaptive ProteinMPNN fine-tuning.

Light wrapper around IMPRESS's run_protein_binding.py. All scientific logic
(MPNN, Boltz, pLDDT extraction, pipeline migration) stays in IMPRESS. This
file only adds the two ROME hooks to adaptive_decision:

  Hook 1: stage the Boltz best model and add it to the ROME corpus
  Hook 2: log the current training status and latest model path

Usage::

    dragon -s run_protein_binding_rome.py
"""

import asyncio
import csv
import importlib
import os
import shutil
import tempfile
from typing import Any, Callable, List

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend
from concurrent.futures import ProcessPoolExecutor
from rhapsody.backends import ConcurrentExecutionBackend

from impress import ImpressManager, PipelineSetup

import rome


def _impress_pb_dir() -> str:
    """Locate IMPRESS's protein_binding examples directory."""
    impress_root = os.environ.get("IMPRESS_DIR")
    if not impress_root:
        _here = os.path.dirname(os.path.abspath(__file__))
        impress_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", "IMPRESS"))
    return os.path.join(impress_root, "examples", "protein_binding")


# ProteinBindingPipeline and adaptive_decision live in IMPRESS's examples dir.
try:
    from impress.pipelines.protein_binding import ProteinBindingPipeline
except ImportError:
    import sys
    _pb_dir = _impress_pb_dir()
    if _pb_dir not in sys.path:
        sys.path.insert(0, _pb_dir)
    from protein_binding import ProteinBindingPipeline  # type: ignore[import]

try:
    from examples.impress_r.protein_binding.run_protein_binding import (
        adaptive_decision as _base_adaptive_decision,
    )
except (ModuleNotFoundError, ImportError):
    import sys
    _pb_dir = _impress_pb_dir()
    if _pb_dir not in sys.path:
        sys.path.insert(0, _pb_dir)
    from run_protein_binding import adaptive_decision as _base_adaptive_decision  # type: ignore[import]

# ProteinMPNN trainer ships with this ROME example (not with the framework).
try:
    from examples.impress_r.protein_binding.mpnn_trainer import (
        ProteinMPNNConfig,
        ProteinMPNNTrainer,
        percentile_sampler,
    )
except ModuleNotFoundError:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer, percentile_sampler  # noqa: F401

# ProteinMPNN checkout — same env var as IMPRESS's delta_gpu_run.sh uses.
MPNN_REPO = os.environ.get("MPNN_PATH", "")


def _load_reward_fn() -> Callable:
    """Load reward function from ROME_REWARD_FN (``module:function``).

    Default: ``mpnn_trainer:pb_reward_fn``.  Override to swap in a
    campaign-specific function without touching this file::

        export ROME_REWARD_FN=my_module:my_reward_fn
    """
    spec = os.environ.get("ROME_REWARD_FN", "mpnn_trainer:pb_reward_fn")
    mod_name, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


# ── Backend ───────────────────────────────────────────────────────────────────

async def _make_backend():
    if os.environ.get("ROME_BACKEND", "dragon").lower() == "concurrent":
        return await ConcurrentExecutionBackend.create(ProcessPoolExecutor())
    return await DragonExecutionBackend()


# ── Trainer ───────────────────────────────────────────────────────────────────

def _build_trainer(checkpoint_dir: str) -> Any:
    """ProteinMPNN fine-tuner. Requires MPNN_PATH to point to a dauparas/ProteinMPNN checkout."""
    if not MPNN_REPO or not os.path.isdir(MPNN_REPO):
        raise RuntimeError(
            f"MPNN_PATH={MPNN_REPO!r} not found. "
            "Set MPNN_PATH to a ProteinMPNN checkout before running."
        )
    return ProteinMPNNTrainer(ProteinMPNNConfig(
        mpnn_repo=MPNN_REPO,
        initial_weights=os.path.join(MPNN_REPO, 'vanilla_model_weights', 'v_48_020.pt'),
        model_name='v_48_020',
        publish_into_repo=True,
    ), gpus=1)


# ── Main ──────────────────────────────────────────────────────────────────────

async def impress_protein_bind_rome() -> None:
    workdir = tempfile.mkdtemp(prefix='impress_r_pb_')
    stage_dir = os.path.join(workdir, 'designs')
    os.makedirs(stage_dir, exist_ok=True)

    # ROME uses its own backend so training rounds run in isolated task
    # processes (GPU memory released when the round finishes).
    rome_backend = await _make_backend()
    output_dir = os.environ.get(
        "IMPRESS_OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "campaign_outputs"),
    )
    os.makedirs(output_dir, exist_ok=True)

    rome_manager = rome.Manager(
        backend=rome_backend,
        data_config=rome.DataConfig(
            min_samples=int(os.environ.get('ROME_MIN_SAMPLES', 4)),
            sample_func=percentile_sampler(0.33, on_summary=print),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=_build_trainer(os.path.join(workdir, 'checkpoints')),
            checkpoint_dir=os.path.join(output_dir, 'rome_checkpoints'),
            poll_interval=1.0,
            result_fallback_seconds=float(os.environ.get('ROME_FALLBACK', 120)),
            train_kwargs={"reward_fn": _load_reward_fn()},
        ),
    )
    await rome_manager.start()

    @rome_manager.trainer.on_checkpoint
    def _log_checkpoint(checkpoint_path: str, version: int) -> None:
        print(
            f"[ROME] v{version} published — "
            f"corpus {rome_manager.data.total_count} designs → "
            f"{os.path.basename(checkpoint_path)}",
            flush=True,
        )

    backend = await _make_backend()
    flow = await WorkflowEngine.create(backend=backend)
    manager: ImpressManager = ImpressManager(flow)

    async def adaptive_decision(pipeline: ProteinBindingPipeline) -> None:
        # ROME hook 1: stage this pass's structures and add to corpus.
        # PDB files are already written at output_path_af by s4 (Boltz).
        file_name = os.path.join(
            pipeline.output_base_path,
            f'af_stats_{pipeline.name}_pass_{pipeline.passes}.csv',
        )
        accepted = 0
        with open(file_name) as fd:
            for row in csv.DictReader(fd):
                protein = row['ID'].split('.')[0]
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
                    backbone_id=protein,
                    pLDDT=float(row['avg_plddt']),
                    pTM=float(row['ptm']),
                    pAE=float(row['avg_pae']),
                    score=float(row['avg_plddt']),
                )
                accepted += uid is not None

        # Delegate IMPRESS migration logic (CSV re-read, scoring, child spawn).
        # Pass adaptive_decision as _adaptive_fn so child pipelines carry the
        # ROME hooks too.
        await _base_adaptive_decision(pipeline, _adaptive_fn=adaptive_decision)

        # ROME hook 2: log training status.
        weights = rome_manager.get_current_model()
        pipeline.logger.pipeline_log(
            f'ROME: corpus {rome_manager.data.total_count} (+{accepted} this pass) | '
            f'{rome_manager.get_training_status().name}'
            + (f' | model {os.path.basename(weights)}' if weights else '')
        )

    _pb_fallback = _impress_pb_dir()
    scripts_dir = os.environ.get("IMPRESS_SCRIPTS_DIR", _pb_fallback)
    input_base_dir = os.environ.get("IMPRESS_BASE_DIR", scripts_dir)
    output_base_dir = output_dir  # already resolved and mkdir'd above

    max_passes = int(os.environ.get("IMPRESS_MAX_PASSES", "10"))
    n_pipelines = int(os.environ.get("IMPRESS_N_PIPELINES", "1"))

    pipeline_setups: List[PipelineSetup] = [
        PipelineSetup(
            name=f'p{i}',
            type=ProteinBindingPipeline,
            config={
                "base_path": scripts_dir,
                "input_base_path": input_base_dir,
                "output_base_path": output_base_dir,
                "max_passes": max_passes,
            },
            adaptive_fn=adaptive_decision,
        )
        for i in range(1, n_pipelines + 1)
    ]

    try:
        await manager.start(pipeline_setups=pipeline_setups)
        print('\nROME:', rome_manager.report())
    finally:
        await flow.shutdown()
        await rome_manager.stop()


if __name__ == "__main__":
    asyncio.run(impress_protein_bind_rome())
