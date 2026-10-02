"""Compute-side training round for the interface-target surrogate.

Run as its own process by :meth:`ItgtSurrogateTrainer.as_command`::

    python itgt_train_wrapper.py --job job.json

This is the only file in the workflow that imports ``itgt``, numpy,
scikit-learn or torch, and every such import is inside a function so that
importing this module costs nothing and reading it statically (griffe, for the
API reference) needs none of them installed.

Three obligations, all part of the ROME contract
------------------------------------------------

1. **Atomic artifact publication.** The artifact is built in a scratch
   directory and moved into place, so a reader never sees a half-written
   ``meta.json``.
2. **``train_complete`` written last**, after the artifact is on disk. The
   training manager polls for that marker because on Dragon a task can run to
   completion and never resolve its future -- and with publish-in-place the
   artifact path may already exist from the previous round, so its presence
   proves nothing.
3. **Durable error reporting.** Failures print ``TRAINING_ERROR:`` with a
   traceback to stderr *and* append to ``job["error_log"]``, because a round's
   stderr can be lost and ``/tmp`` gets cleaned.

What each component fits
------------------------

``reward_ridge`` (default)
    A ridge regression from a partition to the ``ligand_iptm`` it produced.
    Features are the reference ligand's own per-atom feature rows pooled over
    the selected mask, plus the selected fraction -- about 51 numbers describing
    "the average chemistry of this patch". Fit over the whole replay buffer,
    reported with leave-one-group-out, and stored as plain arrays so the
    proposer can load and evaluate it with numpy alone.

``burial_prior`` / ``trunk``
    itgt's per-atom scorers on ``burial_fraction`` labels from the designed
    complexes. **Self-confirming** -- see the module docstring of
    ``itgt_trainer`` -- and kept for cold start and ablation only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Written into ``output_dir`` as the round's final action.
#: Mirrors ``rome.trainer.TRAIN_COMPLETE_MARKER``; duplicated as a literal so
#: the wrapper does not import ROME on the compute side.
TRAIN_COMPLETE_MARKER = "train_complete"

#: Artifact kind for the reward model. Deliberately *not* one of itgt's
#: method-2 kinds, so ``itgt.method2_hotspots.infer.load`` refuses it rather
#: than mistaking it for a per-atom scorer.
REWARD_KIND = "RewardRidge"

METHOD = "rome_interface_targeting"

__all__ = ["run_round", "main", "TRAIN_COMPLETE_MARKER", "REWARD_KIND"]


# ── itgt plumbing ─────────────────────────────────────────────────────────────


def _prepare_itgt(config: Dict[str, Any]) -> None:
    """Put the itgt checkout on ``sys.path`` when it is not installed."""
    itgt_dir = config.get("itgt_dir")
    if itgt_dir and itgt_dir not in sys.path:
        try:
            import itgt  # noqa: F401, PLC0415
        except ImportError:
            sys.path.insert(0, itgt_dir)


def _dataset_config(config: Dict[str, Any]):
    """The single ``DatasetConfig`` a round uses.

    Built once and recorded into the artifact, so the proposer reads it back
    instead of reconstructing it. ``itgt.method2_hotspots.infer.load`` always
    checks ``feature_names`` and ``feature_config`` against what the caller
    would feed the model, and raises ``FeatureMismatch`` on any difference, so
    "reconstruct the same config on both sides" is a standing invitation to a
    mid-campaign crash.
    """
    from itgt.common.datasets import DatasetConfig

    return DatasetConfig(label=config.get("label", "burial_fraction"))


def _featurize_reference(config: Dict[str, Any], dataset_config):
    """The apo reference ligand every action is expressed against."""
    from itgt.common.datasets import featurize_ligand

    reference = config["reference_pdb"]
    ligand_key = config.get("ligand_key")
    ligand = featurize_ligand(reference, ligand_key, dataset_config)
    if ligand is None:
        raise ValueError(
            f"could not featurise ligand {ligand_key!r} in {reference}; "
            "check the ligand key matches the HETATM resname"
        )
    return ligand


# ── reward_ridge ──────────────────────────────────────────────────────────────


def pooled_features(ligand, selected_names: Sequence[str]):
    """Pool the reference ligand's feature rows over one selection.

    ``mean(x[selected])`` plus the selected fraction. The per-atom matrix
    already carries ``rsasa`` and ``circular_variance`` as columns, so the mean
    is a description of the patch's average chemistry *and* its average
    exposure -- which is what a linear model needs in order to say anything
    about a patch it has never seen.

    Returns ``None`` when the selection names nothing present in the reference,
    which is how a stale replay-buffer entry is skipped rather than silently
    contributing a zero row.
    """
    import numpy as np

    index = {name: i for i, name in enumerate(ligand.atom_names)}
    rows = [index[name] for name in selected_names if name in index]
    if not rows:
        return None
    x = np.asarray(ligand.x, dtype=float)
    pooled = x[rows].mean(axis=0)
    fraction = len(rows) / float(len(ligand.atom_names))
    return np.concatenate([pooled, [fraction]])


def _buffer_examples(config: Dict[str, Any], ligand) -> Tuple[Any, Any, List[str], int]:
    """``(X, y, groups, n_skipped)`` from the replay buffer."""
    import numpy as np
    from itgt.common.corpus import Corpus

    path = config["replay_buffer"]
    corpus = Corpus.load(path)
    reward_field = config.get("reward_field", "ligand_iptm")

    xs, ys, groups, skipped = [], [], [], 0
    for record in corpus.records:
        value = record.metrics.get(reward_field)
        if value is None:
            skipped += 1
            continue
        names = record.target.flat_ligand_atoms()
        features = pooled_features(ligand, names)
        if features is None:
            skipped += 1
            continue
        xs.append(features)
        ys.append(float(value))
        groups.append(record.group or record.uid or "ungrouped")
    if not xs:
        raise ValueError(
            f"no usable (action, {reward_field}) pairs in {path}: "
            f"{len(corpus.records)} records, {skipped} skipped. "
            "The runner records an action at each pass's fold step; if this is "
            "empty the pairing never happened."
        )
    return np.asarray(xs), np.asarray(ys), groups, skipped


def _fit_ridge(x, y, alpha: float):
    """Centred, scaled ridge returning ``(coef, intercept, centre, scale)``.

    Standardising here rather than relying on a pipeline keeps the stored
    artifact a handful of plain arrays, which is what lets the proposer evaluate
    it with numpy and no scikit-learn.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0  # constant columns carry no signal; do not blow up
    z = (x - centre) / scale
    model = Ridge(alpha=alpha)
    model.fit(z, y)
    return (
        np.asarray(model.coef_, dtype=float),
        float(model.intercept_),
        centre,
        scale,
    )


def _logo_report(x, y, groups: Sequence[str], alpha: float) -> Dict[str, Any]:
    """Leave-one-group-out predictions, and whether they beat the mean.

    The honest question at this corpus size is not "what is the training error"
    but "does knowing the partition help at all". ``r2`` below zero means it
    does not: predicting the campaign's mean reward would do better, and the
    proposal should not be trusted yet.
    """
    import numpy as np

    unique = sorted(set(groups))
    if len(unique) < 3:
        return {"validated": False, "n_groups": len(unique),
                "reason": "need >= 3 groups for a leave-one-group-out estimate"}

    predictions, truths = [], []
    for held in unique:
        mask = np.array([g != held for g in groups])
        if mask.sum() < 2 or (~mask).sum() == 0:
            continue
        coef, intercept, centre, scale = _fit_ridge(x[mask], y[mask], alpha)
        z = (x[~mask] - centre) / scale
        predictions.extend(z @ coef + intercept)
        truths.extend(y[~mask])
    if len(predictions) < 3:
        return {"validated": False, "n_groups": len(unique),
                "reason": "too few held-out predictions"}

    predictions = np.asarray(predictions)
    truths = np.asarray(truths)
    residual = float(((truths - predictions) ** 2).sum())
    total = float(((truths - truths.mean()) ** 2).sum())
    report: Dict[str, Any] = {
        "validated": True,
        "n_groups": len(unique),
        "n_heldout": int(len(predictions)),
        "logo_rmse": float(np.sqrt(((truths - predictions) ** 2).mean())),
        "logo_r2": (1.0 - residual / total) if total > 0 else None,
        "baseline_rmse": float(np.sqrt(((truths - truths.mean()) ** 2).mean())),
    }
    if len(predictions) >= 4 and truths.std() > 0 and predictions.std() > 0:
        report["logo_pearson"] = float(np.corrcoef(truths, predictions)[0, 1])
    return report


def _train_reward_ridge(job: Dict[str, Any], scratch: str) -> Dict[str, Any]:
    """Fit the reward model and write it as an itgt artifact."""
    from itgt.common import artifacts
    from itgt.common.features import feature_names

    config = job["config"]
    dataset_config = _dataset_config(config)
    ligand = _featurize_reference(config, dataset_config)
    x, y, groups, skipped = _buffer_examples(config, ligand)

    alpha = float(config.get("ridge_alpha", 1.0))
    report = _logo_report(x, y, groups, alpha)
    coef, intercept, centre, scale = _fit_ridge(x, y, alpha)

    report.update(
        {
            "component": "reward_ridge",
            "n_pairs": int(len(y)),
            "n_skipped": int(skipped),
            "n_features": int(x.shape[1]),
            "reward_field": config.get("reward_field", "ligand_iptm"),
            "reward_mean": float(y.mean()),
            "reward_std": float(y.std()),
            "ridge_alpha": alpha,
        }
    )

    artifacts.save(
        scratch,
        {
            "coef": coef,
            "intercept": intercept,
            "centre": centre,
            "scale": scale,
        },
        artifacts.ArtifactMeta(
            method=METHOD,
            kind=REWARD_KIND,
            scorer=config.get("reward_field", "ligand_iptm"),
            feature_names=feature_names(dataset_config.features),
            feature_config=dataset_config.features.to_dict(),
            corpus={"n_pairs": int(len(y)), "n_groups": report.get("n_groups")},
            hyperparameters={
                "ridge": {"alpha": alpha},
                "dataset": dataset_config.to_dict(),
                "pooling": "mean_over_selected_plus_fraction",
                "reference_pdb": config.get("reference_pdb"),
                "ligand_key": config.get("ligand_key"),
            },
            extra={"report": report},
        ),
    )
    return report


# ── burial_prior / trunk ──────────────────────────────────────────────────────


def _complex_examples(job: Dict[str, Any], dataset_config) -> Tuple[List[Any], Dict[str, Any]]:
    """Featurize the staged fold complexes into itgt training examples.

    ``featurize_complex`` returns ``None`` rather than raising -- for an
    undetectable ligand, a size outside ``[min_ligand_atoms, max_ligand_atoms]``
    or an empty binder side -- so the skips are counted and an all-skipped round
    is made to fail loudly instead of training on nothing.
    """
    from itgt.common.datasets import featurize_complex

    config = job["config"]
    ligand_key = config.get("ligand_key")
    examples, skipped = [], []
    for design in job.get("designs", []):
        path = design.get("path")
        if not path or not os.path.exists(path):
            skipped.append((path, "missing file"))
            continue
        example = featurize_complex(path, ligand_key, dataset_config)
        if example is None:
            skipped.append((path, "featurize_complex returned None"))
            continue
        examples.append(example)
    stats = {"n_examples": len(examples), "n_skipped": len(skipped),
             "skipped": skipped[:10]}
    if not examples:
        raise ValueError(
            f"every staged complex was skipped ({len(skipped)} of them); "
            f"first reasons: {skipped[:3]}. Boltz assigns the ligand its own "
            "atom names and resname, so check the ligand key resolves in the "
            "fold output, not just in the reference."
        )
    return examples, stats


def _train_per_atom(job: Dict[str, Any], scratch: str) -> Dict[str, Any]:
    """Fit itgt's per-atom scorer -- the self-confirming prior. See the warning
    in ``itgt_trainer``'s module docstring before reading its numbers."""
    config = job["config"]
    dataset_config = _dataset_config(config)
    examples, stats = _complex_examples(job, dataset_config)

    if config.get("component") == "trunk":
        from itgt.method2_hotspots.train import TrainConfig, train_trunk

        _, report = train_trunk(
            examples,
            dataset_config=dataset_config,
            train_config=TrainConfig(
                epochs=int(config.get("epochs", 60)),
                device=config.get("device", "cpu"),
            ),
            out=scratch,
        )
    else:
        from itgt.method2_hotspots.baseline import BaselineConfig, train_baseline

        _, report = train_baseline(
            examples,
            dataset_config=dataset_config,
            config=BaselineConfig(seed=int(config.get("seed", 0))),
            out=scratch,
        )
        # train_baseline computes n_groups and never splits on it: its
        # `train_mse` is in-sample and a 200-estimator GBT drives it to ~0 by
        # construction. Say so in the report rather than let a reader infer
        # generalisation from it.
        report["validated"] = False
        report["train_mse_is_in_sample"] = True

    report.update(stats)
    return report


# ── round driver ──────────────────────────────────────────────────────────────


def run_round(job: Dict[str, Any]) -> str:
    """Train one round and return the artifact path.

    Writes ``train_complete`` into ``output_dir`` as its final action.
    """
    config = job["config"]
    output_dir = job["output_dir"]
    artifact = job["artifact"]
    os.makedirs(output_dir, exist_ok=True)

    _prepare_itgt(config)
    from itgt.common.nn import set_seed

    set_seed(int(config.get("seed", 0)))

    scratch = f"{artifact}.tmp"
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch, exist_ok=True)

    component = config.get("component", "reward_ridge")
    try:
        if component == "reward_ridge":
            report = _train_reward_ridge(job, scratch)
        elif component in ("burial_prior", "trunk"):
            report = _train_per_atom(job, scratch)
        else:
            raise ValueError(f"unknown component: {component!r}")

        # Publish atomically: a reader sees the previous artifact or this one.
        if os.path.isdir(artifact):
            shutil.rmtree(artifact)
        os.replace(scratch, artifact)

        report["artifact"] = artifact
        report["model_version"] = job.get("model_version")
        if job.get("report"):
            with open(job["report"], "w") as fh:
                json.dump(report, fh, indent=2, default=str)

        if report.get("validated") is False:
            print(
                f"[itgt] WARNING: round produced no held-out estimate "
                f"({report.get('reason', 'see report')}). The published model is "
                f"unvalidated; treat its proposals as exploration, not as a result.",
                flush=True,
            )
        elif report.get("logo_r2") is not None and report["logo_r2"] < 0:
            print(
                f"[itgt] WARNING: leave-one-group-out r2={report['logo_r2']:.3f} < 0 "
                f"-- predicting the campaign mean beats this model. The partition "
                f"may not be driving {report.get('reward_field')} yet.",
                flush=True,
            )
        print(f"[itgt] round complete: {json.dumps(report, default=str)}", flush=True)
        return artifact
    finally:
        if os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)
        _release_memory()
        # Last action, after the artifact is on disk: the manager polls for this.
        with open(os.path.join(output_dir, TRAIN_COMPLETE_MARKER), "w") as fh:
            fh.write(artifact)


def _release_memory() -> None:
    """Drop references and free VRAM, when torch was involved at all."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not fail a good round
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="path to the job spec JSON")
    args = parser.parse_args(argv)

    with open(args.job) as fh:
        job = json.load(fh)

    try:
        print(run_round(job), flush=True)
        return 0
    except Exception:  # noqa: BLE001 - the round's only chance to report
        text = traceback.format_exc()
        print(f"TRAINING_ERROR:\n{text}", file=sys.stderr, flush=True)
        log = job.get("error_log") or os.path.join(
            job.get("output_dir", "."), "train_error.log"
        )
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
            with open(log, "a") as fh:
                fh.write(f"\n=== {datetime.datetime.now().isoformat()} ===\n{text}")
        except OSError:
            pass  # stderr already has it
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
