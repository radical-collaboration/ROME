"""Turning a published surrogate into the next pass's diffusion parameters.

This is the *apply* half of the loop, and the piece the sibling workflow has no
equivalent of: ``small_molecule_binding`` publishes LigandMPNN weights that the
next pass picks up by path, whereas here the model's output has to be written
into a configuration file the pipeline reads.

Where this runs, and why not on checkpoint publish
--------------------------------------------------

Not from ``trainer.on_checkpoint``. ``rome/trainer.py`` calls checkpoint
callbacks **inline on the manager's event loop**, inside
``except Exception: traceback.print_exc()``. Hanging proposal there would put
seconds of SASA, bond perception and model evaluation on the loop that submits
every IMPRESS task, and would reduce a ``FeatureMismatch`` or a non-itgt
checkpoint directory to a printed traceback while every log still looked
healthy and no spec ever changed.

Instead the runner stores ``(artifact, version)`` on publish and calls
:func:`apply_to_pipeline` from ``adaptive_decision`` at the pipeline's own fold
step, after the base decision returns. At that instant the pipeline's next RFD3
task has not been submitted, so the write provably precedes the read within one
sequential pass. That also closes a hole atomicity does not: the pipeline reads
its base spec **twice** per guided step -- once for ``partial.input``, once for
the selections -- so a replace landing between the two reads could mix versions.

Every entry point here returns a value and logs; none raises into the pipeline.
A failed proposal must cost one pass's improvement, not the campaign.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

try:  # the sibling module, importable either as a package or as a bare file
    from examples.impress_r.interface_targeting import itgt_spec
except ModuleNotFoundError:  # pragma: no cover - exercised by the runner
    import itgt_spec


def _pooled_features():
    """The feature pooling used when the model was fit.

    Imported from the training wrapper rather than reimplemented: a model is
    only valid on the features it was fit on, so one definition has to serve
    both sides. Dual-path for the same reason as ``itgt_spec`` above -- this
    workflow is imported both as a package (tests, docs) and as bare files (the
    runner, from inside its own directory).
    """
    try:
        from examples.impress_r.interface_targeting.itgt_train_wrapper import (
            pooled_features,
        )
    except ModuleNotFoundError:  # pragma: no cover - exercised by the runner
        from itgt_train_wrapper import pooled_features

    return pooled_features

#: Artifact kinds this module knows how to propose from.
REWARD_KIND = "RewardRidge"
PER_ATOM_KINDS = ("Baseline", "PerAtomHead")

__all__ = [
    "itgt_available",
    "config_from_artifact",
    "artifact_kind",
    "propose_groups",
    "apply_to_pipeline",
    "record_action",
    "seed_template_action",
    "template_action",
    "installed_action",
]

Logger = Callable[[str], None]


def _log(logger: Optional[Logger], message: str) -> None:
    (logger or print)(f"[itgt] {message}")


# ── availability ──────────────────────────────────────────────────────────────


def itgt_available(itgt_dir: Optional[str] = None) -> bool:
    """Is ``itgt`` importable, putting ``itgt_dir`` on the path if needed?

    Called before anything else so a campaign without itgt degrades to "the
    hand-authored partition, unchanged" rather than to a crash.
    """
    try:
        import itgt  # noqa: F401, PLC0415

        return True
    except ImportError:
        pass
    if itgt_dir and os.path.isdir(itgt_dir):
        import sys

        if itgt_dir not in sys.path:
            sys.path.insert(0, itgt_dir)
        try:
            import itgt  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False
    return False


# ── reading the artifact ──────────────────────────────────────────────────────


def artifact_kind(path: str) -> Optional[str]:
    """The artifact's ``kind``, or ``None`` if this is not an itgt artifact.

    A ``DummyTrainer`` round publishes a plain directory or a JSON checkpoint,
    and ``TrainerConfig.checkpoint_dir`` is handed over as-is, so "is this even
    mine?" has to be answerable without raising.
    """
    if not path or not os.path.isdir(path):
        return None
    if not os.path.exists(os.path.join(path, "meta.json")):
        return None
    try:
        from itgt.common import artifacts

        return artifacts.read_meta(path).kind
    except Exception:  # noqa: BLE001 - an unreadable artifact is simply not ours
        return None


def config_from_artifact(path: str):
    """The ``DatasetConfig`` the artifact was fit under.

    Read back out of the artifact rather than reconstructed. Both of itgt's
    savers record it in ``hyperparameters["dataset"]``, and
    ``artifacts.load`` compares ``feature_names`` and ``feature_config``
    element-by-element and *raises* on any difference -- so rebuilding a config
    on this side and hoping it matches is how a campaign dies in round three.
    """
    from itgt.common.artifacts import read_meta
    from itgt.common.datasets import DatasetConfig

    meta = read_meta(path)
    stored = (meta.hyperparameters or {}).get("dataset")
    return DatasetConfig.from_dict(stored) if stored else DatasetConfig()


def _load_reward_model(path: str, dataset_config) -> Tuple[Any, Any, Any, float]:
    """``(coef, centre, scale, intercept)`` for a :data:`REWARD_KIND` artifact.

    Loaded with the feature guard armed: this is the one place a featurization
    change between training and proposing would otherwise score the wrong
    columns and look perfectly healthy doing it.
    """
    from itgt.common import artifacts
    from itgt.common.features import feature_names

    artifact = artifacts.load(
        path,
        expect_features=feature_names(dataset_config.features),
        expect_feature_config=dataset_config.features.to_dict(),
    )
    state = artifact.state
    return (state["coef"], state["centre"], state["scale"],
            float(state["intercept"]))


# ── proposing ─────────────────────────────────────────────────────────────────


def _reward_proposal(
    artifact: str,
    reference_pdb: str,
    ligand_key: str,
    *,
    n_candidates: int,
    min_atoms: int,
    max_atoms: int,
    mark_fringe_partial: bool,
    seed: int,
    logger: Optional[Logger] = None,
):
    """Score random candidate partitions with the reward model, keep the best.

    Candidates come from ``itgt.method1_random`` -- connected surface patches
    under the same well-formedness rules every method shares. A per-atom scorer
    would return exactly one greedy answer; a reward model over *sets* needs a
    set of candidates to rank, and random surface growth is itgt's only
    label-free source of them.
    """
    import numpy as np
    from itgt.method1_random.sampler import SamplerConfig, sample_targets

    pooled_features = _pooled_features()

    dataset_config = config_from_artifact(artifact)
    coef, centre, scale, intercept = _load_reward_model(artifact, dataset_config)

    from itgt.common.datasets import featurize_ligand

    ligand = featurize_ligand(reference_pdb, ligand_key, dataset_config)
    if ligand is None:
        raise ValueError(f"could not featurise {ligand_key!r} in {reference_pdb}")

    candidates = sample_targets(
        reference_pdb,
        ligand_key,
        n=n_candidates,
        config=SamplerConfig(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            mark_fringe_partial=mark_fringe_partial,
            seed=seed,
        ),
    )
    if not candidates:
        raise ValueError("the sampler produced no candidate partitions")

    best, best_score = None, -np.inf
    for target in candidates:
        features = pooled_features(ligand, target.flat_ligand_atoms())
        if features is None:
            continue
        score = float(((features - centre) / scale) @ coef + intercept)
        if score > best_score:
            best, best_score = target, score
    if best is None:
        raise ValueError("no candidate could be featurised against the reference")

    _log(
        logger,
        f"ranked {len(candidates)} candidates; best predicted reward "
        f"{best_score:.4f} over {best.n_selected()} selected atoms",
    )
    return best, {"predicted_reward": best_score, "n_candidates": len(candidates)}


def _per_atom_proposal(
    artifact: str,
    reference_pdb: str,
    ligand_key: str,
    *,
    min_atoms: int,
    max_atoms: int,
    mark_fringe_partial: bool,
    seed: int,
    logger: Optional[Logger] = None,
):
    """Greedy proposal from itgt's per-atom scorer (the ``burial_prior`` path).

    ``infer.propose`` returns exactly one target when given only a per-atom
    scorer -- by design, since a per-atom score has one answer and padding the
    list with duplicates would misrepresent how much exploration happened.
    """
    from itgt.common.postprocess import PostprocessConfig
    from itgt.method2_hotspots.infer import propose

    targets, report = propose(
        reference_pdb,
        ligand_key,
        trunk=artifact,
        n=1,
        dataset_config=config_from_artifact(artifact),
        postprocess_config=PostprocessConfig(
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            mark_fringe_partial=mark_fringe_partial,
        ),
        seed=seed,
    )
    if not targets:
        raise ValueError("propose returned no targets")
    _log(logger, f"per-atom proposal: {report.get('mode')}")
    return targets[0], report


def propose_groups(
    artifact: str,
    reference_pdb: str,
    ligand_key: str,
    *,
    n_candidates: int = 64,
    min_atoms: int = 4,
    max_atoms: int = 14,
    mark_fringe_partial: bool = False,
    seed: int = 0,
    logger: Optional[Logger] = None,
):
    """``(burial groups, target, report)`` for the next pass.

    Dispatches on the artifact's own ``kind``, so the proposal path cannot
    disagree with what was actually trained.
    """
    kind = artifact_kind(artifact)
    if kind == REWARD_KIND:
        target, report = _reward_proposal(
            artifact, reference_pdb, ligand_key,
            n_candidates=n_candidates, min_atoms=min_atoms, max_atoms=max_atoms,
            mark_fringe_partial=mark_fringe_partial, seed=seed, logger=logger,
        )
    elif kind in PER_ATOM_KINDS:
        target, report = _per_atom_proposal(
            artifact, reference_pdb, ligand_key,
            min_atoms=min_atoms, max_atoms=max_atoms,
            mark_fringe_partial=mark_fringe_partial, seed=seed, logger=logger,
        )
    else:
        raise ValueError(f"cannot propose from artifact kind {kind!r} at {artifact}")

    component = next(iter(target.burial or {}), None)
    if component is None:
        raise ValueError("proposed target carries no burial partition")
    return target.burial_groups(component), target, report


# ── applying ──────────────────────────────────────────────────────────────────


def apply_to_pipeline(
    input_dir: str,
    artifact: str,
    version: int,
    *,
    itgt_dir: Optional[str] = None,
    n_candidates: int = 64,
    min_atoms: Optional[int] = None,
    max_atoms: Optional[int] = None,
    size_band: int = 4,
    mark_fringe_partial: bool = False,
    seed: int = 0,
    spec_name: str = itgt_spec.DEFAULT_SPEC_NAME,
    logger: Optional[Logger] = None,
) -> bool:
    """Propose a partition and install it as this pipeline's base spec.

    Returns ``True`` when the spec was rewritten. Returns ``False`` -- having
    logged why -- for every failure mode, so a pipeline keeps running on the
    partition it already has. The caller treats ``False`` as "try again next
    pass", never as an error.

    Patches from the pristine template snapshot, not from the previous round's
    output, so round *n*'s partition is a function of the model and the original
    template and proposals cannot compound.
    """
    try:
        if not itgt_available(itgt_dir):
            _log(logger, "itgt is not importable; leaving the base spec unchanged")
            return False

        path = itgt_spec.spec_path(input_dir, spec_name)
        if not os.path.exists(path):
            _log(logger, f"no base spec at {path}; nothing to patch")
            return False

        template = itgt_spec.load_template(input_dir, spec_name)
        entry = itgt_spec.entry_name(template)
        ligand = itgt_spec.ligand_key(template, entry)
        reference = itgt_spec.reference_pdb(path, template)
        if not os.path.exists(reference):
            _log(logger, f"reference structure missing: {reference}")
            return False

        # Bind the size band to the template's own selection rather than to
        # library defaults. Mean-pooled features have a small-patch bias -- the
        # average chemistry of four well-chosen atoms beats that of fourteen --
        # so a floor of `min_atoms` is not a safety net but the answer the
        # argmax collapses onto, and a 4-atom buried patch is not an interface.
        # Banding around the hand-authored 14 keeps size a variable the campaign
        # explores deliberately instead of one the objective quietly minimises.
        current = itgt_spec.template_partition(template, entry, ligand)
        anchor = len(current.get("buried", ())) or 12
        if min_atoms is None:
            min_atoms = max(4, anchor - size_band)
        if max_atoms is None:
            max_atoms = max(min_atoms, anchor + size_band)

        groups, target, report = propose_groups(
            artifact, reference, ligand,
            n_candidates=n_candidates, min_atoms=min_atoms, max_atoms=max_atoms,
            mark_fringe_partial=mark_fringe_partial, seed=seed, logger=logger,
        )

        present = itgt_spec.ligand_atom_names(reference, ligand)
        ok, missing = itgt_spec.coverage_ok(groups, present)
        if not ok:
            # The pipeline's own guard checks only two of the three selection
            # fields while copying all three through, so a slip in
            # select_partially_buried would escalate from a clean fallback to an
            # RFD3 validation error mid-run. Catch it here instead.
            _log(logger, f"proposal names atoms absent from the reference: {sorted(missing)}")
            return False

        spec = itgt_spec.patch_spec(template, groups, entry, ligand)
        _validate_spec(spec[entry], logger)
        itgt_spec.write_spec_atomic(spec, input_dir, version, spec_name)

        sizes = {level: len(names) for level, names in groups.items()}
        _log(
            logger,
            f"v{version} -> {os.path.basename(input_dir)}: {sizes} "
            f"({report.get('predicted_reward', 'n/a')} predicted)",
        )
        return True
    except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
        _log(logger, f"proposal failed ({type(exc).__name__}: {exc}); spec unchanged")
        return False


def _validate_spec(entry_body: Dict[str, Any], logger: Optional[Logger]) -> None:
    """Run itgt's own RFD3 spec validator when it is importable.

    It enforces rASA disjointness, rejects empty selections and encodes the
    ``partial_t``/``length`` exclusivity, so it is strictly better than
    re-deriving those rules here. Best-effort: it imports gemmi, and a campaign
    that can still train should not be stopped by a missing validator.
    """
    try:
        from itgt.common.emit.rfd3 import validate_spec

        validate_spec(entry_body)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - surface it, do not swallow silently
        _log(logger, f"WARNING: spec validator objected: {exc}")


# ── the (action, reward) pairing ──────────────────────────────────────────────


def _action_from(spec: Dict[str, Any], spec_path: str, name: str):
    """Invert one design spec into the ``InterfaceTarget`` it encodes.

    ``itgt.common.corpus.target_from_spec`` is itgt's spec inverter -- what
    makes the existing hand-authored specs usable as training data. Here it
    serves the same purpose for our own writes: it recovers the *action* a pass
    ran under, so a reward can be attributed to it.
    """
    from itgt.common.corpus import target_from_spec

    entry = itgt_spec.entry_name(spec)
    reference = itgt_spec.reference_pdb(spec_path, spec)
    return target_from_spec(spec[entry], name, reference)


def installed_action(input_dir: str, spec_name: str = itgt_spec.DEFAULT_SPEC_NAME):
    """The partition **currently installed** in this pipeline's base spec.

    This is the action a pass actually ran under, and the one its reward must be
    attributed to. Distinct from :func:`template_action`, which always returns
    the pristine v0 partition: reading the snapshot here would credit every
    pass's reward to the hand-authored partition and the surrogate would learn
    nothing from its own proposals.
    """
    path = itgt_spec.spec_path(input_dir, spec_name)
    return _action_from(
        itgt_spec.load_spec(path), path, f"installed_{os.path.basename(input_dir)}"
    )


def template_action(input_dir: str, spec_name: str = itgt_spec.DEFAULT_SPEC_NAME):
    """The hand-authored partition as an ``InterfaceTarget``.

    Patching with this would be a no-op, which is exactly why it is useful for a
    different purpose: it gives the campaign's v0 *action* an identity, so the
    first reward observation has something to be attributed to. Without it the
    first ``(action, reward)`` pair has no action and round 1 regresses on
    nothing.
    """
    path = itgt_spec.spec_path(input_dir, spec_name)
    return _action_from(
        itgt_spec.load_template(input_dir, spec_name),
        path,
        f"v0_{os.path.basename(input_dir)}",
    )


def record_action(
    buffer_path: str,
    target: Any,
    metrics: Dict[str, float],
    *,
    group: str = "",
    source: str = "",
    logger: Optional[Logger] = None,
) -> bool:
    """Append an ``(action, reward)`` pair to the replay buffer.

    ``group`` is the leave-one-group-out split key and should identify the pass
    that produced the reward -- the corpus's own default groups by structure and
    ligand, which on a single-target campaign is one group for everything and a
    validation estimate of nothing.

    Returns ``True`` when the pair was new; the buffer dedupes on
    ``InterfaceTarget.key()``, so re-proposing an already-scored partition
    updates its metrics rather than counting it twice as evidence.
    """
    try:
        from itgt.common.corpus import CorpusRecord, ReplayBuffer
        from itgt.common.provenance import selection_uid

        buffer = ReplayBuffer(buffer_path)
        added = buffer.add(
            CorpusRecord(
                target=target,
                metrics=dict(metrics),
                source=source,
                group=group,
                uid=selection_uid(target),
            )
        )
        buffer.flush()
        _log(
            logger,
            f"replay buffer: {'added' if added else 'updated'} "
            f"{dict(metrics)} (now {len(buffer.corpus.records)} pairs)",
        )
        return added
    except Exception as exc:  # noqa: BLE001
        _log(logger, f"could not record action ({type(exc).__name__}: {exc})")
        return False


def seed_template_action(
    buffer_path: str,
    input_dirs: Sequence[str],
    *,
    itgt_dir: Optional[str] = None,
    logger: Optional[Logger] = None,
) -> int:
    """Register each pipeline's v0 action, with no reward yet.

    Called once at startup. The reward lands at that pipeline's first fold step,
    which updates the record in place rather than adding a second one.
    """
    if not itgt_available(itgt_dir):
        return 0
    seeded = 0
    for input_dir in input_dirs:
        try:
            target = template_action(input_dir)
        except Exception as exc:  # noqa: BLE001
            _log(logger, f"could not invert the template in {input_dir}: {exc}")
            continue
        if record_action(
            buffer_path, target, {}, group=f"{os.path.basename(input_dir)}|v0",
            source="template", logger=logger,
        ):
            seeded += 1
    return seeded
