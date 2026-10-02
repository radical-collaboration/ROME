"""Small-molecule binding with an online surrogate for the diffusion step.

A sibling of ``small_molecule_binding/run_smb_rome.py`` that closes the loop
around a different stage. That workflow fine-tunes LigandMPNN -- **stage 2**,
sequence design. This one leaves every model in the pipeline alone and instead
learns **stage 1's parameters**: the ``select_buried``/``select_exposed``
partition of the target ligand's heavy atoms, which is the RFD3 ``rasa_bin``
conditioning signal and has been hand-authored and frozen in every campaign to
date.

The scientific pipeline is unchanged. What this file adds to
``adaptive_decision`` is three hooks rather than the sibling's two:

  Hook 1 (fold): stage the fold complex and add the pass to the ROME corpus
  Hook 2 (fold): pair the *action* this pass ran under with the ``ligand_iptm``
                 it produced, and append that pair to the replay buffer
  Hook 3 (fold): if a newer surrogate has been published, propose a partition
                 and install it as this pipeline's base spec

Hook 2 is the one that makes the loop a loop. The surrogate regresses the
partition we wrote against the reward it produced -- the action is ours, the
reward is downstream of it -- rather than regressing the designed binder's own
burial, which would only measure compliance with our previous instruction.

Two things this runner does that the sibling does not
-----------------------------------------------------

**It stages its own input directories.** The sibling points ``input_dir`` at
``$IMPRESS_DIR/examples/small_molecule_binding/p{i}_in``, inside the IMPRESS git
checkout. This workflow *writes* to that directory, so it copies ``p1_in`` into
the campaign output tree and points the pipelines there. That also fixes a
second problem for free: only ``p1_in`` is populated in the checkout, while
``PROD.n_pipelines`` is 4.

**Each pipeline gets its own seed.** Proposal is deterministic, so without this
every pipeline would receive an identical partition and the campaign's parallel
structure would collapse to a single hypothesis per round. Distinct seeds also
grow the ``(action, reward)`` corpus at rate N instead of 1.

Usage::

    dragon -s run_interface_targeting_rome.py
"""

import asyncio
import importlib
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List

from radical.asyncflow import WorkflowEngine
from rhapsody.backends import DragonExecutionBackend

from impress import ImpressManager, PipelineSetup


def _impress_smb_dir() -> str:
    """Locate IMPRESS's small_molecule_binding directory.

    Preferred: set IMPRESS_DIR to the IMPRESS checkout root.
    Fallback: assume IMPRESS lives as a sibling directory to ROME.
    """
    impress_root = os.environ.get("IMPRESS_DIR")
    if not impress_root:
        # ROME/examples/impress_r/interface_targeting/ → up 4 → parent of ROME
        _here = os.path.dirname(os.path.abspath(__file__))
        impress_root = os.path.normpath(
            os.path.join(_here, "..", "..", "..", "..", "IMPRESS")
        )
    return os.path.join(impress_root, "examples", "small_molecule_binding")


try:
    from examples.small_molecule_binding.run_small_molecule_binding import (
        PROD, adaptive_decision as _base_adaptive_decision,
    )
    from examples.small_molecule_binding.small_molecule_binding import (
        SmallMoleculeBindingPipeline,
    )
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
    from examples.impress_r.interface_targeting import itgt_propose, itgt_spec
    from examples.impress_r.interface_targeting.itgt_trainer import (
        ItgtSurrogateConfig,
        ItgtSurrogateTrainer,
        itgt_reward_fn,
        percentile_sampler,
    )
except ModuleNotFoundError:
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import itgt_propose  # noqa: F401 (re-exported)
    import itgt_spec  # noqa: F401 (re-exported)
    from itgt_trainer import (  # noqa: F401 (re-exported)
        ItgtSurrogateConfig,
        ItgtSurrogateTrainer,
        itgt_reward_fn,
        percentile_sampler,
    )

import rome

import logging

import rhapsody

rhapsody.enable_logging(level=logging.INFO)


# ── Configuration ─────────────────────────────────────────────────────────────

N_PIPELINES = int(os.environ.get("IMPRESS_N_PIPELINES", PROD.n_pipelines))
MAX_PASSES = int(os.environ.get("ROME_MAX_PASSES", 10))

#: The interface-targeting checkout providing the models. No weights ship with
#: it, so every campaign trains its surrogate from its own data.
ITGT_DIR = os.environ.get("ITGT_DIR", "")

#: ``reward_ridge`` (the action→outcome surrogate), ``burial_prior`` or
#: ``trunk``. See the warning in ``itgt_trainer``'s module docstring before
#: choosing either of the latter two.
ITGT_COMPONENT = os.environ.get("ROME_ITGT_COMPONENT", "reward_ridge")

#: Candidate partitions ranked per proposal.
ITGT_CANDIDATES = int(os.environ.get("ROME_ITGT_CANDIDATES", 64))

#: Half-width of the selection-size band around the template's own selection.
ITGT_SIZE_BAND = int(os.environ.get("ROME_ITGT_SIZE_BAND", 4))


# ── Reward function loader ────────────────────────────────────────────────────

def _load_reward_fn() -> Callable:
    """Load reward function from ``ROME_REWARD_FN`` (``module:function``)."""
    spec = os.environ.get("ROME_REWARD_FN", "itgt_trainer:itgt_reward_fn")
    mod_name, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


# ── Input staging ─────────────────────────────────────────────────────────────

def _stage_inputs(source: str, root: str, n: int) -> List[str]:
    """Copy the pipeline inputs out of the IMPRESS checkout and snapshot them.

    This workflow rewrites each pipeline's base design spec every round, so it
    must not operate on a tracked file in someone else's repository. Copies are
    made once; an existing staged directory is reused so a restarted campaign
    keeps its history. Every pipeline is seeded from the same source because
    only ``p1_in`` is populated upstream.
    """
    os.makedirs(root, exist_ok=True)
    staged: List[str] = []
    for index in range(1, n + 1):
        target = os.path.join(root, f"p{index}_in")
        if not os.path.isdir(target):
            shutil.copytree(source, target)
        # Idempotent: the pristine template is captured on first sight only, so
        # every round patches from the hand-authored spec rather than from the
        # previous round's output.
        itgt_spec.snapshot_template(target)
        staged.append(target)
    return staged


# ── Trainer builder ───────────────────────────────────────────────────────────

def _build_trainer(replay_buffer: str, reference_pdb: str, ligand_key: str) -> Any:
    """The surrogate trainer, or a dummy when itgt is unavailable.

    A missing ``ITGT_DIR`` degrades the whole workflow to "the hand-authored
    partition, unchanged" -- which is exactly the sibling's behaviour and a
    legitimate campaign -- rather than failing at startup.
    """
    requested = os.environ.get("ROME_TRAINER", "itgt").lower()
    if requested != "dummy" and itgt_propose.itgt_available(ITGT_DIR or None):
        return ItgtSurrogateTrainer(
            ItgtSurrogateConfig(
                itgt_dir=ITGT_DIR or None,
                component=ITGT_COMPONENT,
                replay_buffer=replay_buffer,
                reference_pdb=reference_pdb,
                ligand_key=ligand_key,
                size_band=ITGT_SIZE_BAND,
                n_candidates=ITGT_CANDIDATES,
            ),
            gpus=0,
        )

    if requested != "dummy":
        print(
            f"[ROME] itgt not importable (ITGT_DIR={ITGT_DIR!r}); falling back to "
            "DummyTrainer. The base spec will not be modified — set ITGT_DIR to "
            "enable the surrogate, or export ROME_TRAINER=dummy to silence this."
        )
    from rome.dummy import DummyTrainer

    return DummyTrainer(train_seconds=1.0, gpus=0)


# ── Main ──────────────────────────────────────────────────────────────────────

async def impress_interface_targeting_rome() -> None:
    workdir = tempfile.mkdtemp(prefix="impress_itgt_rome_")
    stage_dir = os.path.join(workdir, "designs")
    os.makedirs(stage_dir, exist_ok=True)

    smb_dir = _impress_smb_dir()
    output_dir = os.environ.get(
        "IMPRESS_OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "campaign_outputs"),
    )
    os.makedirs(output_dir, exist_ok=True)

    input_dirs = _stage_inputs(
        os.path.join(os.path.abspath(smb_dir), "p1_in"),
        os.path.join(output_dir, "rome_inputs"),
        N_PIPELINES,
    )
    replay_buffer = os.path.join(output_dir, "itgt_replay.jsonl")

    # Resolve the reference structure and ligand key from the spec the pipeline
    # itself reads, so the partition is expressed in the atom names RFD3 is
    # handed rather than in some other file's.
    base_spec_path = itgt_spec.spec_path(input_dirs[0])
    base_spec = itgt_spec.load_spec(base_spec_path)
    ligand_key = itgt_spec.ligand_key(base_spec)
    reference_pdb = itgt_spec.reference_pdb(base_spec_path, base_spec)
    print(
        f"[ROME] interface targeting: ligand {ligand_key} in "
        f"{os.path.basename(reference_pdb)}; "
        f"template partition "
        f"{ {k: len(v) for k, v in itgt_spec.template_partition(base_spec).items()} }"
    )

    # Give the campaign's v0 action an identity before any reward arrives, so
    # the first (action, reward) pair has an action attached. Without this the
    # first round regresses on nothing.
    seeded = itgt_propose.seed_template_action(
        replay_buffer, input_dirs, itgt_dir=ITGT_DIR or None
    )
    print(f"[ROME] seeded {seeded} template action(s) into {replay_buffer}")

    backend = await DragonExecutionBackend()
    flow = await WorkflowEngine.create(backend=backend)

    rome_manager = rome.Manager(
        asyncflow=flow,
        data_config=rome.DataConfig(
            min_samples=int(os.environ.get("ROME_MIN_SAMPLES", 8)),
            # fraction=1.0: the surrogate fits the whole observed reward range.
            # Selecting the best third of a reward regression's own training set
            # truncates the response variable and leaves the model unable to
            # distinguish a bad partition from an unobserved one.
            sample_func=percentile_sampler(1.0, on_summary=print),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=_build_trainer(replay_buffer, reference_pdb, ligand_key),
            checkpoint_dir=os.path.join(output_dir, "rome_checkpoints"),
            poll_interval=5.0,
            result_fallback_seconds=float(os.environ.get("ROME_FALLBACK", 60)),
            train_kwargs={"reward_fn": _load_reward_fn()},
            max_consecutive_failures=int(os.environ.get("ROME_MAX_FAILURES", 3)),
        ),
    )
    await rome_manager.start()

    #: Latest published surrogate, as ``(artifact path, version)``. Set by the
    #: checkpoint hook, consumed at each pipeline's own pass boundary.
    published: Dict[str, Any] = {"artifact": None, "version": 0}

    @rome_manager.trainer.on_checkpoint
    def _note_checkpoint(checkpoint_path: str, version: int) -> None:
        """Record the new surrogate. Deliberately does no work.

        ROME calls checkpoint callbacks inline on the manager's event loop,
        inside a bare ``except Exception: traceback.print_exc()``. Proposing
        here would put seconds of SASA and bond perception on the loop that
        submits every IMPRESS task, and would reduce any failure to a printed
        traceback while no spec ever changed. The proposal happens in
        ``adaptive_decision`` instead, where it is both off this loop and
        visible in the pipeline log.
        """
        published["artifact"] = checkpoint_path
        published["version"] = version
        print(
            f"[ROME] surrogate v{version} published — "
            f"corpus {rome_manager.data.total_count} passes → "
            f"{os.path.basename(checkpoint_path)}",
            flush=True,
        )

    manager: ImpressManager = ImpressManager(flow)

    async def adaptive_decision(pipeline: SmallMoleculeBindingPipeline) -> None:
        step = pipeline.state.get("last_analysis_step")
        metrics = pipeline.state.get("last_analysis_metrics", {})

        # Cache metrics that later analysis steps overwrite. At 'fastrelax' the
        # dict has 'interact'; at 'interface' it has 'max_sc'. By 'fold' both are
        # gone, so they are read back from pipeline.state.
        if step == "fastrelax":
            pipeline.state["_rome_interact"] = metrics.get("interact")
        elif step == "interface":
            pipeline.state["_rome_max_sc"] = metrics.get("max_sc")

        # Delegate every decision to the base adaptive_decision, unchanged.
        await _base_adaptive_decision(pipeline)

        if step != "fold":
            return

        # The pipeline records its own resolved input directory, which is the
        # staged copy this workflow writes to. Read it from the pipeline rather
        # than threading it through kwargs, which the pipeline would ignore.
        input_dir = getattr(pipeline, "pipeline_inputs", "") or ""
        seed = _pipeline_seed(pipeline.name)
        plddt = metrics.get("best_complex_plddt", -1.0)
        ligand_iptm = metrics.get("best_ligand_iptm")
        interact = pipeline.state.get("_rome_interact")
        max_sc = pipeline.state.get("_rome_max_sc")

        # ── Hook 1: contribute the pass to the ROME corpus ────────────────────
        # best_fold_model is set on any fold that produced a confidence file,
        # not only a passing one, so failures arrive here too. That is wanted: a
        # reward regression needs its negatives, and filtering to successes
        # truncates exactly the range it learns from.
        best_model = pipeline.state.get("best_fold_model")
        accepted = 0
        skip_note = ""
        if best_model and os.path.exists(best_model):
            stem = os.path.splitext(os.path.basename(best_model))[0]
            staged = os.path.join(
                stage_dir, f"{pipeline.name}_pass{pipeline.passes}_{stem}.pdb"
            )
            shutil.copyfile(best_model, staged)
            uid = rome_manager.add_training_data(
                path=staged,
                plddt=float(plddt),
                ligand_iptm=float(ligand_iptm) if ligand_iptm is not None else None,
                interaction_energy=float(interact) if interact is not None else None,
                max_sc=float(max_sc) if max_sc is not None else None,
                action_uid=str(pipeline.state.get("_itgt_action_uid", "")),
                score=itgt_reward_fn({"ligand_iptm": ligand_iptm}),
            )
            accepted = 1 if uid is not None else 0
            if not accepted:
                skip_note = " — filtered by ROME (see [ROME-DATA] log)"
        else:
            skip_note = f" — no fold model (best_fold_model={best_model!r})"

        # ── Hook 2: pair this pass's action with the reward it produced ───────
        # The action is whatever partition the spec held when this pass ran, so
        # it is read from the pipeline's own input dir rather than from the
        # currently-published model -- a round may have published mid-pass.
        if input_dir and ligand_iptm is not None:
            try:
                action = itgt_propose.installed_action(input_dir)
                itgt_propose.record_action(
                    replay_buffer,
                    action,
                    {
                        "ligand_iptm": float(ligand_iptm),
                        "plddt": float(plddt),
                    },
                    group=f"{pipeline.name}|pass{pipeline.passes}",
                    source=f"{pipeline.name}|v{pipeline.state.get('_itgt_applied_version', 0)}",
                    logger=pipeline.logger.pipeline_log,
                )
            except Exception as exc:  # noqa: BLE001 - never break the pipeline
                pipeline.logger.pipeline_log(f"ROME: could not record action: {exc}")

        # ── Hook 3: install a newer surrogate's proposal for the next pass ────
        # Safe here and nowhere else: this pipeline's next RFD3 task has not
        # been submitted, so the write provably precedes the read within one
        # sequential pass. It also avoids straddling the pipeline's two separate
        # reads of the base spec during a guided step.
        artifact = published.get("artifact")
        version = int(published.get("version", 0))
        applied = int(pipeline.state.get("_itgt_applied_version", 0))
        if artifact and version > applied and input_dir:
            ok = itgt_propose.apply_to_pipeline(
                input_dir,
                artifact,
                version,
                itgt_dir=ITGT_DIR or None,
                n_candidates=ITGT_CANDIDATES,
                size_band=ITGT_SIZE_BAND,
                seed=seed + version,
                logger=pipeline.logger.pipeline_log,
            )
            if ok:
                pipeline.state["_itgt_applied_version"] = version
                try:
                    pipeline.state["_itgt_action_uid"] = _action_uid(input_dir)
                except Exception:  # noqa: BLE001
                    pass

        weights = rome_manager.get_current_model()
        pipeline.logger.pipeline_log(
            f"ROME: corpus {rome_manager.data.total_count} "
            f"(+{accepted}{skip_note} this pass) | "
            f"{rome_manager.get_training_status().name} | "
            f"surrogate v{pipeline.state.get('_itgt_applied_version', 0)}"
            + (f" | {os.path.basename(weights)}" if weights else "")
        )

    pipeline_setups: List[PipelineSetup] = [
        PipelineSetup(
            name=f"p{str(i)}",
            type=SmallMoleculeBindingPipeline,
            adaptive_fn=adaptive_decision,
            kwargs={
                "base_path":                 output_dir,
                "scripts_path":              os.path.join(os.path.abspath(smb_dir), "scripts"),
                # The staged copy, never the IMPRESS checkout: this workflow
                # writes here every round.
                "input_dir":                 input_dirs[i - 1],
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
            },
        )
        for i in range(1, N_PIPELINES + 1)
    ]

    try:
        await manager.start(pipeline_setups=pipeline_setups)
        print("\nROME:", rome_manager.report())
    finally:
        await rome_manager.stop()
        await flow.shutdown()


def _pipeline_seed(name: str) -> int:
    """A stable per-pipeline seed, so siblings explore different partitions.

    Derived from the pipeline's own name (``p1``, ``p2``, ...) rather than from
    enumeration order, so a pipeline keeps its seed across a restart.
    """
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return 1000 * (int(digits) if digits else 1)


def _action_uid(input_dir: str) -> str:
    """Provenance id of the partition currently installed in ``input_dir``."""
    from itgt.common.provenance import selection_uid

    return selection_uid(itgt_propose.installed_action(input_dir))


if __name__ == "__main__":
    asyncio.run(impress_interface_targeting_rome())
