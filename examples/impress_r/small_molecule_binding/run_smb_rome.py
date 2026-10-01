"""Small-molecule binding pipeline with ROME-A adaptive LigandMPNN fine-tuning.

This script is a thin wrapper around ``run_small_molecule_binding.py``.  The
scientific pipeline is completely unchanged — this file only adds the two ROME
hooks to ``adaptive_decision``:

  Hook 1 (fold branch): stage the AF2 best model and add it to the ROME corpus
  Hook 2 (fold branch): log the current training status and latest model path

The designed chains (A) and ligand context (HETATM) are handled automatically
by LigandMPNN — no wrapper changes there.

**Reward function.**  The per-sample training reward is loaded from
``ROME_REWARD_FN`` (default ``ligandmpnn_trainer:smb_reward_fn``).  Point
this env var at ``your_module:your_fn`` to swap in a custom reward without
touching this file.  The default implementation is documented in
``ligandmpnn_trainer.smb_reward_fn``.

Usage::

    dragon -s run_smb_rome.py
"""

import asyncio
import importlib
import os
import shutil
import tempfile
from typing import Any, Callable, List, Optional

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend

from impress import ImpressManager, PipelineSetup

def _impress_smb_dir() -> str:
    """Locate IMPRESS's small_molecule_binding directory.

    Preferred: set IMPRESS_DIR to the IMPRESS checkout root.
    Fallback: assume IMPRESS lives as a sibling directory to ROME.
    """
    import sys as _sys
    impress_root = os.environ.get("IMPRESS_DIR")
    if not impress_root:
        # ROME/examples/impress_r/small_molecule_binding/ → go up 4 levels → parent of ROME
        _here = os.path.dirname(os.path.abspath(__file__))
        impress_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", "IMPRESS"))
    return os.path.join(impress_root, "examples", "small_molecule_binding")


try:
    from examples.small_molecule_binding.run_small_molecule_binding import (
        PROD, adaptive_decision as _base_adaptive_decision,
    )
    from examples.small_molecule_binding.small_molecule_binding import SmallMoleculeBindingPipeline
except ModuleNotFoundError:
    import sys
    _smb_dir = _impress_smb_dir()
    if _smb_dir not in sys.path:
        sys.path.insert(0, _smb_dir)
    from run_small_molecule_binding import (
        PROD, adaptive_decision as _base_adaptive_decision,
    )
    from small_molecule_binding import SmallMoleculeBindingPipeline

try:
    from examples.impress_r.small_molecule_binding.ligandmpnn_trainer import (
        LigandMPNNConfig,
        LigandMPNNTrainer,
        percentile_sampler,
    )
except ModuleNotFoundError:
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from ligandmpnn_trainer import LigandMPNNConfig, LigandMPNNTrainer, percentile_sampler  # noqa: F401 (re-exported)

import rome

import rhapsody, logging
rhapsody.enable_logging(level=logging.INFO)


# ── Configuration ─────────────────────────────────────────────────────────────

N_PIPELINES = int(os.environ.get("IMPRESS_N_PIPELINES", PROD.n_pipelines))
MAX_PASSES  = int(os.environ.get("ROME_MAX_PASSES", 10))

#: LigandMPNN checkout — same directory mpnn.sh and mpnn_run.py use.
MPNN_DIR = os.environ.get("MPNN_DIR", "")


# ── Reward function loader ────────────────────────────────────────────────────

def _load_reward_fn() -> Callable:
    """Load reward function from ``ROME_REWARD_FN`` (``module:function``)."""
    spec = os.environ.get("ROME_REWARD_FN", "ligandmpnn_trainer:smb_reward_fn")
    mod_name, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


# ── Trainer builder ───────────────────────────────────────────────────────────

def _build_trainer(checkpoint_dir: str) -> Any:
    """LigandMPNN fine-tuner; falls back to dummy if ROME_TRAINER=dummy or MPNN_DIR missing."""
    rome_trainer = os.environ.get("ROME_TRAINER", "mpnn").lower()
    if rome_trainer != "dummy" and MPNN_DIR and os.path.isdir(MPNN_DIR):
        return LigandMPNNTrainer(LigandMPNNConfig(mpnn_dir=MPNN_DIR), gpus=1)

    if rome_trainer != "dummy":
        print(
            f"[ROME] MPNN_DIR={MPNN_DIR!r} not found; falling back to DummyTrainer "
            "(set MPNN_DIR to enable real fine-tuning, or export ROME_TRAINER=dummy to silence this)."
        )
    from rome.dummy import DummyTrainer
    return DummyTrainer(train_seconds=1.0, gpus=0)


# ── Main ──────────────────────────────────────────────────────────────────────

async def impress_smallmol_bind_rome() -> None:
    workdir = tempfile.mkdtemp(prefix="impress_smb_rome_")
    stage_dir = os.path.join(workdir, "designs")
    os.makedirs(stage_dir, exist_ok=True)

    backend = await DragonExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)

    rome_manager = rome.Manager(
        asyncflow=flow,
        data_config=rome.DataConfig(
            min_samples=int(os.environ.get("ROME_MIN_SAMPLES", 8)),
            sample_func=percentile_sampler(0.33, on_summary=print),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=_build_trainer(os.path.join(workdir, "checkpoints")),
            checkpoint_dir=os.path.join(
                os.environ.get("IMPRESS_OUTPUT_DIR", workdir), "rome_checkpoints"
            ),
            poll_interval=5.0,
            result_fallback_seconds=float(os.environ.get("ROME_FALLBACK", 60)),
            train_kwargs={"reward_fn": _load_reward_fn()},
            max_consecutive_failures=int(os.environ.get("ROME_MAX_FAILURES", 3)),
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

    manager: ImpressManager = ImpressManager(flow)

    async def adaptive_decision(pipeline: SmallMoleculeBindingPipeline) -> None:
        step    = pipeline.state.get("last_analysis_step")
        metrics = pipeline.state.get("last_analysis_metrics", {})

        # Cache metrics that are overwritten by later analysis steps.
        # At 'fastrelax' the dict has 'interact'; at 'interface' it has 'max_sc'.
        # By 'fold', both are gone — we read back from pipeline.state.
        if step == "fastrelax":
            pipeline.state["_rome_interact"] = metrics.get("interact")
        elif step == "interface":
            pipeline.state["_rome_max_sc"] = metrics.get("max_sc")

        # Delegate all decision logic to the base adaptive_decision.
        await _base_adaptive_decision(pipeline)

        # ROME hooks — only at the fold step (after AF2).
        if step != "fold":
            return

        plddt   = metrics.get("best_complex_plddt", -1.0)
        interact = pipeline.state.get("_rome_interact")
        max_sc  = pipeline.state.get("_rome_max_sc")

        # Stage the best AF2 model for training.
        best_model = pipeline.state.get("best_fold_model")
        accepted = 0
        skip_note = ""
        if best_model and os.path.exists(best_model):
            stem   = os.path.splitext(os.path.basename(best_model))[0]
            staged = os.path.join(
                stage_dir,
                f"{pipeline.name}_pass{pipeline.passes}_{stem}.pdb",
            )
            shutil.copyfile(best_model, staged)
            uid = rome_manager.add_training_data(
                path=staged,
                plddt=float(plddt),
                interaction_energy=float(interact) if interact is not None else None,
                max_sc=float(max_sc)    if max_sc   is not None else None,
                score=float(plddt),
            )
            accepted = 1 if uid is not None else 0
            if not accepted:
                skip_note = " — filtered by ROME (see [ROME-DATA] log)"
        else:
            skip_note = f" — no AF2 model (best_fold_model={best_model!r})"

        weights = rome_manager.get_current_model()
        pipeline.logger.pipeline_log(
            f"ROME: corpus {rome_manager.data.total_count} (+{accepted}{skip_note} this pass) | "
            f"{rome_manager.get_training_status().name}"
            + (f" | model {os.path.basename(weights)}" if weights else "")
        )

    # __file__ may resolve to SLURM's spool copy; use IMPRESS_DIR env var instead.
    _impress_dir = os.environ.get("IMPRESS_DIR") or os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "IMPRESS")
    )
    smb_dir = os.path.join(_impress_dir, "examples", "small_molecule_binding")
    output_dir   = os.environ.get(
        "IMPRESS_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "campaign_outputs")
    )
    os.makedirs(output_dir, exist_ok=True)

    pipeline_setups: List[PipelineSetup] = [
        PipelineSetup(
            name=f"p{str(i)}",
            type=SmallMoleculeBindingPipeline,
            adaptive_fn=adaptive_decision,
            kwargs={
                "base_path":                 output_dir,
                "scripts_path":              os.path.join(os.path.abspath(smb_dir), "scripts"),
                "input_dir":                 os.path.join(os.path.abspath(smb_dir), f"p{i}_in"),
                "backbone_max_ca_deviation": PROD.backbone_max_ca_deviation,
                "backbone_min_ss_fraction":  PROD.backbone_min_ss_fraction,
                "fastrelax_max_fa_rep":      PROD.fastrelax_max_fa_rep,
                "fastrelax_max_total_score": PROD.fastrelax_max_score,
                "fastrelax_max_interact":    PROD.fastrelax_max_interact,
                "interface_min_sc":          PROD.interface_min_sc,
                "fold_min_plddt":            PROD.fold_min_plddt,
                "fold_min_ligand_iptm":      PROD.fold_min_ligand_iptm,
                "diffusion_batch_size":      PROD.diffusion_batch_size,
                "num_refine_cycles":         PROD.num_refine_cycles,
                "mpnn_ensemble_size":        PROD.mpnn_ensemble_size,
                "rfd3_partial_t":            PROD.rfd3_partial_t,
                "max_tasks":                 PROD.max_tasks,
            }
        )
        for i in range(1, N_PIPELINES + 1)
    ]

    try:
        await manager.start(pipeline_setups=pipeline_setups)
        print("\nROME:", rome_manager.report())
    finally:
        await rome_manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(impress_smallmol_bind_rome())
