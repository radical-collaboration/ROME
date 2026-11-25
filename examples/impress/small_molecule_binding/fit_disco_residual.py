"""Fit the project's first `residual.json`, on a benchmark that cannot be mistaken for affinity.

Every piece of Model C has been reviewed and tested against fixtures, and until this module ran
there was **no `residual.json` anywhere in four checkouts** -- so the artifact contract, the
name-checked projection, `botorch_gp`'s `train_Yvar` and the calibration band had never once met
real data. This is [J14]'s rehearsal: the disco islands benchmark, 8,200 real designs over 41
real M-CSA targets, target `log_motif_rmsd`.

**It claims nothing about affinity, and that is enforced rather than promised.** The artifact's
`target` is `log_motif_rmsd`, so `Residual.require_target` refuses it under `dg_corrected`; its
`featuriser` is `disco-fold-1`, which is deliberately **not** in `asteer.features.FEATURISERS`,
so `Residual.project` refuses a `chain-metrics-1` vector against this basis by name. Two
refusals, two mechanisms, both tested.

**The prior is fitted on a different family of targets than the folds it is corrected on.** The
benchmark splits itself -- 8 targets are `ame_rfd3_first8`, 33 are `mcsa_mod8-*` -- and the prior
takes the 8 while every reported number comes from the 33. Fitting the prior on all 41 and then
holding targets out one at a time would leak each fold's own target into the prior it is measured
against, and *read and never written* is the one property C's residual has.

**A training point is one `(target, rfd3 backbone)` cell**: 410 of them, 20 folds each (4 seeds x
5 chai1 models), target `mean ln(motif_rmsd)`, and `train_Yvar` from that mean's own standard
error. Not one point per design -- 8,200 points is an exact GP nobody can afford -- and not one
per target, because 41 points is one per group and then `logo_r2` and `loo_r2` are the same
number by construction, which is the conflation J7 existed to remove.

**The feature list is derived by a declared rule and not typed.** J14 named
`aggregate_score, iptm, ptm`; measured, `iptm` is 0.000000 on all 8,200 rows and
`aggregate_score` is **exactly `0.2 x ptm`** -- `a == float32(0.2) * p` is bit-exact on all
8,200 rows -- so that list is one column with a constant and a duplicate beside it:
`af2_rank1_plddt` again, in a second corpus. The rule
below drops both mechanically, and `test_disco_residual.py` re-measures the degeneracies in the
data so neither can come back silently.

Run it:

    python -m examples.impress.small_molecule_binding.fit_disco_residual \\
        --benchmark-data ../discontinuous_scaffolds/rfd3-islands-validation/benchmark_data \\
        --island-counts ../disco-surrogates/tests/data/specs/island_counts.csv \\
        --out examples/impress/small_molecule_binding/artifacts/disco_rehearsal
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["build_corpus", "choose_features", "fit_prior", "logo_folds", "logo_table",
           "load_corpus", "write_corpus", "gate_verdict", "run", "main",
           "TARGET", "UNITS", "FEATURISER", "PRIOR_FAMILY", "KEYS", "DROPPED_BY_NAME"]

#: What this residual predicts, and in what. **Not** a free energy, and the artifact says so in
#: the one field `require_target` reads.
TARGET = "log_motif_rmsd"
UNITS = "ln(angstrom)"

#: Deliberately absent from `asteer.features.FEATURISERS`. A real Model C round embeds with
#: `chain-metrics-1`, and `Residual.project`'s first check refuses that vector against this
#: basis *by name* -- which is how a rehearsal stays unusable by accident rather than by
#: convention. Registering it would make it loadable, which is the opposite of the point.
FEATURISER = "disco-fold-1"

#: The target family the prior is fitted on; everything else is corpus for the GP and the folds.
PRIOR_FAMILY = "ame_rfd3_first8"

#: Per-cell replication: 4 sequence seeds x 5 chai1 models.
FOLDS_PER_CELL = 20

#: The column rule's three thresholds, declared here rather than inline.
#: `RHO_BAR` is J1's bar, reused rather than re-chosen.
RHO_BAR = 0.2
DUPLICATE_PEARSON = 0.99
COLLINEAR_PEARSON = 0.9

#: Not features: the keys, and the quantities derived from the target.
KEYS = ("model_name", "rfd3_model_idx", "family", "island_count", "n_folds", "y", "y_sd", "noise")

#: Dropped by name, with the measurement that justifies it, because the generic rule would keep
#: whichever of the pair sorts first and the honest name is the one the fold tool emits.
#: `aggregate_score` is `0.2 * ptm` to Pearson 1.000000 over all 8,200 rows -- a composite that
#: in this corpus is one of its own inputs, rescaled. `test_disco_residual.py` re-measures the
#: relation, so if the data ever changes the test fails rather than the artifact quietly
#: carrying one column twice.
DROPPED_BY_NAME = {
    "aggregate_score": "exactly 0.2 x ptm (Pearson 1.000000 over 8,200 rows): one column twice",
}

Fitter = Callable[[List[Dict[str, Any]], Dict[str, Any]], Any]


# -- the corpus --------------------------------------------------------------


def _rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


#: `has_inter_chain_clashes` is the string `False` on all 8,200 rows. Read as a number it is a
#: constant and the rule drops it for the reason that is true; left unparsed it would be dropped
#: as *missing on every cell*, which is a different and wrong statement about the data.
_BOOLEAN = {"true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0}


def _float(text: str) -> Optional[float]:
    if isinstance(text, str) and text.strip() in _BOOLEAN:
        return _BOOLEAN[text.strip()]
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    bar = _mean(values)
    return math.sqrt(sum((v - bar) ** 2 for v in values) / (len(values) - 1))


def build_corpus(data_dir: str, island_counts: str) -> List[Dict[str, Any]]:
    """One row per `(target, rfd3 backbone)` cell, from the three benchmark CSVs.

    The fold-stage rows are aggregated to the cell; `campaign_analysis_backbone.csv` joins
    **1:1** on the same key, which is what makes the backbone columns exact rather than
    averaged, and the sequence-stage rows are the mean over the cell's 4 seeds.

    A cell whose backbone row is missing is dropped and named in the raised error rather than
    carried with holes -- the join is 1:1 by construction, so a miss means the corpus changed.
    """
    folds = _rows(os.path.join(data_dir, "campaign_analysis.csv"))
    backbone = _rows(os.path.join(data_dir, "campaign_analysis_backbone.csv"))
    sequence = _rows(os.path.join(data_dir, "campaign_analysis_sequence.csv"))
    islands = {r["INPUT_ID"]: int(r["RESIDUE_ISLAND_COUNT"]) for r in _rows(island_counts)}

    def key(row: Dict[str, str]) -> Tuple[str, int]:
        return (row["model_name"], int(row["rfd3_model_idx"]))

    by_cell: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in folds:
        by_cell.setdefault(key(row), []).append(row)
    bb = {key(r): r for r in backbone}
    seq: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in sequence:
        seq.setdefault(key(row), []).append(row)

    fold_numeric = [c for c in folds[0] if c not in
                    ("run_dir", "experiment", "model_name", "rfd3_model_idx", "seed",
                     "chai1_model_idx", "motif_rmsd")]
    bb_numeric = [c for c in backbone[0] if c not in
                  ("run_dir", "experiment", "model_name", "rfd3_model_idx")
                  and _float(backbone[0][c]) is not None]
    seq_numeric = ("overall_confidence", "ligand_confidence", "seq_rec")

    corpus: List[Dict[str, Any]] = []
    for cell, members in sorted(by_cell.items()):
        name, idx = cell
        if cell not in bb:
            raise ValueError(
                f"cell {name}/{idx} has {len(members)} fold rows and no backbone row. The join "
                "is 1:1 on this corpus (410 cells, 410 backbone rows); a miss means the "
                "benchmark changed and the feature columns would be half absent"
            )
        if name not in islands:
            raise ValueError(f"no island count for target {name!r} in the stratifier")
        y = [math.log(float(r["motif_rmsd"])) for r in members]
        row: Dict[str, Any] = {
            "model_name": name,
            "rfd3_model_idx": idx,
            "family": PRIOR_FAMILY if members[0]["experiment"].startswith(PRIOR_FAMILY)
            else "mcsa_mod8",
            "island_count": islands[name],
            "n_folds": len(y),
            "y": _mean(y),
            "y_sd": _sd(y),
            # `train_Yvar` is the cell mean's own standard error. Known noise, not inferred:
            # within a cell the 20 folds differ by seed alone, so sd/sqrt(n) is the whole story.
            "noise": _sd(y) / math.sqrt(len(y)) if len(y) > 1 else 0.0,
        }
        for column in fold_numeric:
            values = [v for v in (_float(r[column]) for r in members) if v is not None]
            row[column] = _mean(values) if len(values) == len(members) else ""
        for column in bb_numeric:
            row[column] = _float(bb[cell][column]) if _float(bb[cell][column]) is not None else ""
        for column in seq_numeric:
            values = [v for v in (_float(r[column]) for r in seq.get(cell, []))
                      if v is not None]
            row[column] = _mean(values) if values else ""
        corpus.append(row)
    return corpus


def write_corpus(corpus: List[Dict[str, Any]], path: str) -> None:
    """The 410 cells, committed beside the artifact.

    The raw benchmark lives in a sibling checkout that no CI will ever have, so the evidence
    behind the artifact travels with it -- and the test that re-derives the LOGO table reads
    this, which is what makes it hermetic.
    """
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(corpus[0]))
        writer.writeheader()
        writer.writerows(corpus)


def load_corpus(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _rows(path):
        parsed: Dict[str, Any] = {}
        for column, text in row.items():
            if column == "model_name" or column == "family":
                parsed[column] = text
            elif column in ("rfd3_model_idx", "island_count", "n_folds"):
                parsed[column] = int(text)
            else:
                value = _float(text)
                parsed[column] = value if value is not None else ""
        out.append(parsed)
    return out


# -- the column rule ---------------------------------------------------------


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman's rho with midranks, stdlib. `None` when either side is constant."""
    def ranks(values: Sequence[float]) -> Optional[List[float]]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = mid
            i = j + 1
        return None if len(set(out)) < 2 else out

    ra, rb = ranks(a), ranks(b)
    if ra is None or rb is None:
        return None
    return _pearson(ra, rb)


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    abar, bbar = _mean(a), _mean(b)
    num = sum((x - abar) * (y - bbar) for x, y in zip(a, b))
    da = math.sqrt(sum((x - abar) ** 2 for x in a))
    db = math.sqrt(sum((y - bbar) ** 2 for y in b))
    if da <= 0.0 or db <= 0.0:
        return None
    return num / (da * db)


def _by_target(corpus: List[Dict[str, Any]], column: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for row in corpus:
        out.setdefault(row["model_name"], []).append(row[column])
    return out


def choose_features(corpus: List[Dict[str, Any]]) -> Tuple[Tuple[str, ...], List[Dict[str, Any]]]:
    """The declared rule, in order, with every verdict recorded.

    1. drop the keys and anything non-numeric;
    2. drop constants;
    3. drop a column missing on any cell -- *a feature the corpus cannot supply for every row is
       not a feature*, and imputing one here would invent the value the GP then fits;
    4. drop a linear duplicate (|Pearson| >= 0.99), plus the by-name drops above;
    5. keep a column only if it clears |rho| >= 0.2 on **one of two axes** -- mean within-target
       Spearman, or between-target Spearman of the column's target mean against the target's
       mean. Two axes because the measurement says they are different questions: within a
       target only `ptm` carries anything (-0.599) and every backbone column is |rho| <= 0.12,
       which replicates disco's headline and J1's; between targets those same columns are the
       strongest thing in the data, because they describe how hard the *target* is rather than
       how good the design is;
    6. of a surviving pair with |Pearson| > 0.9, drop the one with the smaller max |rho|.

    Returns the tuple **in corpus-column order**, which is the order the artifact records and
    `project` name-checks against.
    """
    candidates = [c for c in corpus[0] if c not in KEYS]
    verdicts: List[Dict[str, Any]] = []
    target_means = {name: _mean(values) for name, values in _by_target(corpus, "y").items()}
    alive: List[str] = []
    stats: Dict[str, Dict[str, Any]] = {}

    for column in candidates:
        values = [row[column] for row in corpus]
        record: Dict[str, Any] = {"column": column, "rho_within": None, "rho_between": None}
        if any(v == "" for v in values):
            record["verdict"] = f"dropped: missing on {sum(1 for v in values if v == '')} cells"
            verdicts.append(record)
            continue
        record["distinct"] = len(set(values))
        if record["distinct"] <= 1:
            record["verdict"] = f"dropped: constant at {values[0]}"
            verdicts.append(record)
            continue
        if column in DROPPED_BY_NAME:
            record["verdict"] = f"dropped by name: {DROPPED_BY_NAME[column]}"
            verdicts.append(record)
            continue
        within = [r for r in (_spearman(v, _by_target(corpus, "y")[name])
                              for name, v in _by_target(corpus, column).items())
                  if r is not None]
        record["rho_within"] = _mean(within) if within else None
        per_target = {name: _mean(v) for name, v in _by_target(corpus, column).items()}
        names = sorted(per_target)
        record["rho_between"] = _spearman([per_target[n] for n in names],
                                          [target_means[n] for n in names])
        best = max(abs(record["rho_within"] or 0.0), abs(record["rho_between"] or 0.0))
        record["max_abs_rho"] = best
        if best < RHO_BAR:
            record["verdict"] = f"dropped: max |rho| {best:.3f} under the bar of {RHO_BAR}"
            verdicts.append(record)
            continue
        record["verdict"] = "kept"
        verdicts.append(record)
        alive.append(column)
        stats[column] = record

    # 4 and 6 in one pass over the survivors, strongest first so the weaker of a pair goes.
    ordered = sorted(alive, key=lambda c: -stats[c]["max_abs_rho"])
    kept: List[str] = []
    for column in ordered:
        clash = None
        for already in kept:
            r = _pearson([row[column] for row in corpus], [row[already] for row in corpus])
            if r is not None and abs(r) > COLLINEAR_PEARSON:
                clash = (already, r)
                break
        if clash is None:
            kept.append(column)
            continue
        already, r = clash
        kind = "duplicate" if abs(r) >= DUPLICATE_PEARSON else "collinear"
        stats[column]["verdict"] = (
            f"dropped: {kind} with {already} (Pearson {r:+.4f}), which carries the larger "
            f"max |rho| ({stats[already]['max_abs_rho']:.3f} against "
            f"{stats[column]['max_abs_rho']:.3f})")
    features = tuple(c for c in candidates if c in kept)
    return features, verdicts


# -- the prior ---------------------------------------------------------------


def fit_prior(corpus: List[Dict[str, Any]], features: Sequence[str]) -> Dict[str, Any]:
    """A PCA rotation plus a linear fit on it, fitted on `PRIOR_FAMILY` alone.

    **Every component is kept.** The basis is a rotation and not a reduction, because a
    rehearsal that dropped components would conflate two questions -- does the prior fit, and
    does the basis lose the signal -- and only the first is being asked. The eigenvector sign is
    pinned (largest-magnitude entry positive) so a re-fit reproduces the artifact byte for byte.
    """
    import numpy as np  # noqa: PLC0415

    train = [row for row in corpus if row["family"] == PRIOR_FAMILY]
    if len(train) <= len(features):
        raise ValueError(
            f"{len(train)} cells in family {PRIOR_FAMILY!r} for {len(features)} features: an "
            "OLS fit here is interpolation, not a prior")
    x = np.array([[float(row[c]) for c in features] for row in train], dtype=float)
    centre = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    if float(scale.min()) <= 0.0:
        raise ValueError(
            f"a feature is constant within family {PRIOR_FAMILY!r}: "
            f"{[c for c, s in zip(features, scale) if s <= 0.0]}. The rule cleared it on the "
            "whole corpus, so the prior's family cannot scale it and the projection would "
            "divide by zero")
    z = (x - centre) / scale
    _, vectors = np.linalg.eigh(np.cov(z, rowvar=False))
    components = vectors.T[::-1]          # rows, strongest first
    for i, row in enumerate(components):  # pin the sign so the fit is reproducible
        if row[int(np.argmax(np.abs(row)))] < 0:
            components[i] = -row
    basis = z @ components.T
    design = np.column_stack([np.ones(len(basis)), basis])
    beta, *_ = np.linalg.lstsq(design, np.array([row["y"] for row in train]), rcond=None)
    return {
        "version": 1,
        "features": list(features),
        "featuriser": FEATURISER,
        "target": TARGET,
        "units": UNITS,
        "centre": [float(v) for v in centre],
        "scale": [float(v) for v in scale],
        "components": [[float(v) for v in row] for row in components],
        "intercept": float(beta[0]),
        "coef": [float(v) for v in beta[1:]],
        "notes": (
            f"J14's rehearsal, not an affinity model. Fitted 2026-11-24 on the "
            f"{len(train)} {PRIOR_FAMILY} cells of the disco islands benchmark "
            f"({len({r['model_name'] for r in train})} M-CSA targets); corrected and validated "
            f"on the mcsa_mod8 family, which shares no target with it. Predicts "
            f"{TARGET} in {UNITS} and is refused under any other property name by "
            f"require_target. Its featuriser {FEATURISER!r} is deliberately not in "
            "asteer.features.FEATURISERS, so project() refuses a real Model C vector against "
            "this basis by name."
        ),
    }


# -- the folds ---------------------------------------------------------------


def logo_folds(examples: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[int], List[int]]]:
    """One fold per group, in `affdelta_train_wrapper.evaluate`'s shape.

    Not a second split rule: `run()` asserts that `_r2` over the pooled columns this produces
    equals `evaluate`'s own `logo_r2` to 1e-9, which is the check item 50 wanted and the reason
    a per-fold table can be reported at all -- `evaluate` returns the pooled number and throws
    the per-fold predictions, and the held-out sigma, away.
    """
    groups = sorted({str(e["group"]) for e in examples if e.get("group")})
    folds = []
    for held in groups:
        test = [i for i, e in enumerate(examples) if str(e.get("group")) == held]
        train = [i for i, e in enumerate(examples) if str(e.get("group")) != held and e.get("group")]
        if len(train) < 2 or not test:
            continue
        folds.append((held, train, test))
    return folds


def logo_table(examples: Sequence[Dict[str, Any]], job: Dict[str, Any], fit: Fitter,
               islands: Dict[str, int]) -> Dict[str, Any]:
    """Per held-out group: the error, the bias, and the calibration of its own sigma.

    The calibration is the number that matters and the one `evaluate` cannot report: it computes
    a mean squared z-score over the **leave-one-out** folds, where a cell's own target is in the
    training set, and the gate's band is read against generalisation off the group.
    """
    rows: List[Dict[str, Any]] = []
    pooled_truth: List[float] = []
    pooled_pred: List[float] = []
    pooled_sigma: List[float] = []
    for held, train_at, test_at in logo_folds(examples):
        posterior = fit([examples[i] for i in train_at], job)
        mean, sigma = posterior([examples[i]["z"] for i in test_at])
        truth = [examples[i]["residual"] for i in test_at]
        rows.append({
            "group": held,
            "island_count": islands.get(held),
            "n": len(test_at),
            # The vectors, not only their summaries. 10 cells x 33 folds of three numbers is
            # 15 KB and it makes **every** stratification re-derivable from the committed
            # report -- including the one this episode's result turns on, which was not
            # foreseen when the table was designed. A summary can only answer the question
            # whoever wrote it had.
            "truth": list(truth),
            "pred": list(mean),
            "sigma": list(sigma),
            "truth_mean": _mean(truth),
            "pred_mean": _mean(mean),
            "bias": _mean([p - t for t, p in zip(truth, mean)]),
            "rmse": math.sqrt(_mean([(t - p) ** 2 for t, p in zip(truth, mean)])),
            "sigma_mean": _mean(sigma),
            "calibration": _mean([((t - p) / max(s, 1e-6)) ** 2
                                  for t, p, s in zip(truth, mean, sigma)]),
        })
        pooled_truth.extend(truth)
        pooled_pred.extend(mean)
        pooled_sigma.extend(sigma)
    return {
        "folds": rows,
        "n_groups": len(rows),
        "n": len(pooled_truth),
        "r2": _r2(pooled_truth, pooled_pred),
        "rmse": math.sqrt(_mean([(t - p) ** 2 for t, p in zip(pooled_truth, pooled_pred)])),
        "rmse_predict_mean": _sd(pooled_truth),
        "calibration": _mean([((t - p) / max(s, 1e-6)) ** 2
                              for t, p, s in zip(pooled_truth, pooled_pred, pooled_sigma)]),
        "sigma_mean": _mean(pooled_sigma),
        "by_island_count": _by_island(rows),
    }


def _by_island(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The stratification J14 asked for: the folds grouped by island count.

    Reported rather than used. The benchmark's own question was whether discontinuity costs
    accuracy, and a residual whose error tracks island count would be a residual that needs the
    count as a feature -- which is a measurement, not a correction to apply here.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(int(row["island_count"]), []).append(row)
    return [{"island_count": count,
             "n_groups": len(members),
             "n": sum(r["n"] for r in members),
             "rmse": math.sqrt(_mean([r["rmse"] ** 2 for r in members])),
             "bias": _mean([r["bias"] for r in members]),
             "calibration": _mean([r["calibration"] for r in members])}
            for count, members in sorted(out.items())]


def _r2(truth: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (  # noqa: PLC0415
        _r2 as c_r2,
    )

    return c_r2(truth, pred)


# -- the variant that is measured and not adopted ----------------------------


def gp_with_learned_noise(train: List[Dict[str, Any]], job: Dict[str, Any]) -> Any:
    """`botorch_gp` with disco's one difference, measured here and **changed nowhere**.

    `disco-surrogates/src/disco_surrogates/latent/model.py:198-218` fits the same construction
    with `learn_additional_noise=True`, and records why: within a target the objective is
    determined up to seed noise, so the known `sd^2/n` is the whole story; *between* targets two
    targets with the same features genuinely differ, so there is irreducible model error on top
    of the measurement error. Leave-one-target-out on its own 41-target corpus,
    `False -> True`: rho +0.321 -> +0.551, RMSE 0.414 -> 0.270.

    Model C's `botorch_gp` passes `train_Yvar` and nothing else, which is `False`. This function
    exists so J14 can report the size of that gap **on C's own code path** rather than assert it
    from disco's README. Adopting it is J15, which owns C's sigma.
    """
    import torch  # noqa: PLC0415
    from botorch.fit import fit_gpytorch_mll  # noqa: PLC0415
    from botorch.models import SingleTaskGP  # noqa: PLC0415
    from botorch.models.transforms.outcome import Standardize  # noqa: PLC0415
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood  # noqa: PLC0415
    from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415

    from examples.impress.small_molecule_binding.affdelta_train_wrapper import GP  # noqa: PLC0415

    torch.manual_seed(int(job.get("seed", 0)))
    x = torch.tensor([e["z"] for e in train], dtype=torch.double)
    y = torch.tensor([[e["residual"]] for e in train], dtype=torch.double)
    yvar = torch.tensor([[max(e["noise"] ** 2, 1e-8)] for e in train], dtype=torch.double)
    model = SingleTaskGP(
        x, y,
        likelihood=FixedNoiseGaussianLikelihood(noise=yvar.squeeze(-1),
                                                learn_additional_noise=True),
        outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return GP(model, x, y, yvar)


# -- what the gate makes of it -----------------------------------------------


def gate_verdict(table: Dict[str, Any], metrics: Optional[Dict[str, Any]],
                 corpus_size: int) -> Dict[str, Any]:
    """Hand the first real residual's numbers to `asteer.promotion.decide`.

    **Two verdicts, and the first one is the honest one.** As fitted, the manifest names
    featuriser `disco-fold-1`, which is not registered -- so the gate refuses it at
    `featuriser_unknown` with `INSUFFICIENT_EVIDENCE`, which is the **third** mechanism keeping
    this rehearsal out of Model C and the one nobody had to write.

    The second, `on_the_numbers_alone`, is a manifest that **does not describe this fit**: its
    featuriser and target are swapped for Model C's real ones so the gate runs past the
    identity rules and names the rule the *numbers* meet. It exists to answer one question --
    what would the gate say if these numbers came from a real round -- and it is reported under
    a name that cannot be mistaken for a verdict about this artifact.
    """
    import datetime  # noqa: PLC0415

    from asteer.contracts import ModelManifest  # noqa: PLC0415
    from asteer.promotion import decide  # noqa: PLC0415

    common = dict(
        version=1,
        trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corpus_size=corpus_size,
        git_sha="",
        nll=(metrics or {}).get("nll"),
        calibration=table["calibration"],
        logo_r2=table["r2"],
        holdout_n=table["n"],
        n_holdout_groups=table["n_groups"],
        min_holdout_n=8,
    )
    out: Dict[str, Any] = {}
    for name, overrides in (
        ("as_fitted", {"model_id": "disco_rehearsal", "featuriser": FEATURISER,
                       "trained_on": TARGET}),
        ("on_the_numbers_alone", {"model_id": "aff_delta", "featuriser": "chain-metrics-1",
                                  "trained_on": "dg_corrected"}),
    ):
        decision = decide(ModelManifest(**{**common, **overrides}))
        out[name] = {"outcome": decision.outcome.name, "rule": decision.rule,
                     "reason": decision.reason}
    return out


# -- the run -----------------------------------------------------------------


def examples_for(corpus: Sequence[Dict[str, Any]], residual: Any,
                 features: Sequence[str]) -> List[Dict[str, Any]]:
    """Corpus rows into `evaluate`'s examples, through the **name-checked** projection.

    Both names are passed on purpose. `run_round` would resolve them from
    `FEATURISER_FOR_MODEL[model_id]`, and this rehearsal has no model id in that map by
    design -- so the names are handed over here instead, and a permuted feature list is refused
    by the name of the first column that moved rather than silently projected (item 74).
    """
    out = []
    for row in corpus:
        x = [float(row[c]) for c in features]
        z = residual.project(x, featuriser=FEATURISER, columns=tuple(features))
        out.append({"x": x, "y": float(row["y"]), "z": z, "noise": float(row["noise"]),
                    "group": row["model_name"], "residual": float(row["y"]) - residual.predict(z)})
    return out


def run(data_dir: str, island_counts: str, out_dir: str, *, corpus_csv: str = "",
        min_labels: int = 8, seed: int = 0, variant: bool = True,
        loo: bool = True) -> Dict[str, Any]:
    """Fit the artifact, correct it on the other family, and write the report.

    **`evaluate` is called and `run_round` is not.** `run_round` publishes a version into a
    `ModelRegistry` and copies the residual beside the weights; a version whose `trained_on` is
    `log_motif_rmsd` has no business in Model C's registry even once, and *"it was only a
    rehearsal"* is not a property a registry can read. Everything this episode exercises is
    inside `evaluate`: J7's corrected LOGO, `train_Yvar`, `Standardize`, and the sigma the
    promotion gate reads.
    """
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (  # noqa: PLC0415
        Residual, botorch_gp, evaluate,
    )

    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    if corpus_csv:
        corpus = load_corpus(corpus_csv)
    else:
        corpus = build_corpus(data_dir, island_counts)
        write_corpus(corpus, os.path.join(out_dir, "corpus.csv"))
    features, verdicts = choose_features(corpus)
    payload = fit_prior(corpus, features)
    residual_path = os.path.join(out_dir, "residual.json")
    with open(residual_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    residual = Residual.load(residual_path)
    residual.require_target(TARGET, where=residual_path)

    held = [row for row in corpus if row["family"] != PRIOR_FAMILY]
    examples = examples_for(held, residual, features)
    islands = {row["model_name"]: int(row["island_count"]) for row in held}
    job = {"min_labels": min_labels, "seed": seed}

    prior_truth = [e["y"] for e in examples]
    prior_pred = [e["y"] - e["residual"] for e in examples]
    in_family = examples_for([r for r in corpus if r["family"] == PRIOR_FAMILY],
                             residual, features)
    report: Dict[str, Any] = {
        "episode": "J14",
        "target": TARGET,
        "units": UNITS,
        "featuriser": FEATURISER,
        "claims_about_affinity": "none",
        "corpus": {
            "cells": len(corpus),
            "folds_per_cell": FOLDS_PER_CELL,
            "targets": len({r["model_name"] for r in corpus}),
            "prior_family": PRIOR_FAMILY,
            "prior_cells": len(corpus) - len(held),
            "prior_targets": len({r["model_name"] for r in corpus
                                  if r["family"] == PRIOR_FAMILY}),
            "held_cells": len(held),
            "held_targets": len({r["model_name"] for r in held}),
            "noise_median": sorted(r["noise"] for r in corpus)[len(corpus) // 2],
            "within_cell_sd_median": sorted(r["y_sd"] for r in corpus)[len(corpus) // 2],
        },
        "features": list(features),
        "feature_verdicts": verdicts,
        "prior": {
            "components": len(payload["components"]),
            "r2_in_family": _r2([e["y"] for e in in_family],
                                [e["y"] - e["residual"] for e in in_family]),
            "r2_out_of_family": _r2(prior_truth, prior_pred),
            "rmse_out_of_family": math.sqrt(_mean([(t - p) ** 2
                                                   for t, p in zip(prior_truth, prior_pred)])),
            "rmse_predict_mean": _sd(prior_truth),
            "residual_sd": _sd([e["residual"] for e in examples]),
        },
    }

    table = logo_table(examples, job, botorch_gp, islands)
    report["logo"] = table
    if variant:
        report["logo_learn_additional_noise"] = logo_table(examples, job,
                                                           gp_with_learned_noise, islands)
    if loo:
        loo_started = time.time()
        metrics = evaluate(list(examples), job, residual, botorch_gp)
        report["evaluate"] = metrics
        report["evaluate_seconds"] = time.time() - loo_started
        # Both grouped numbers, since J15: `evaluate` now keeps the sigma its grouped fold
        # computes, so its calibration is comparable to this table's as well as its r2.
        for mine, theirs in (("r2", "logo_r2"), ("calibration", "calibration")):
            if metrics[theirs] is None or table[mine] is None:
                continue
            drift = abs(metrics[theirs] - table[mine])
            if drift > 1e-9:
                raise ValueError(
                    f"this module's fold table and evaluate's {theirs} disagree by "
                    f"{drift:.2e} ({table[mine]} against {metrics[theirs]}). They are the "
                    "same split over the same examples, so a disagreement is two split "
                    "rules -- item 50's shape, which this project has already paid for once")
    report["gate"] = gate_verdict(table, report.get("evaluate"), len(examples))
    report["seconds"] = time.time() - started
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import warnings  # noqa: PLC0415

    # botorch asks for inputs in the unit cube. These are PCA coordinates of standardised
    # features -- centred on zero and spanning several standard deviations by construction --
    # so the warning is a correct description of a deliberate choice, 33 folds deep. Silenced
    # here, in the CLI, and never inside `run`, so a caller that wants it gets it.
    # By message: botorch's `InputDataWarning` descends from `Warning`, not `UserWarning`.
    warnings.filterwarnings("ignore", message=".*unit cube.*")

    here = os.path.dirname(os.path.abspath(__file__))
    # The five checkouts are siblings under one parent; both defaults assume that layout and
    # both are overridable, because the benchmark belongs to neither this repo nor its CI.
    siblings = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-data", default=os.path.join(
        siblings, "discontinuous_scaffolds", "rfd3-islands-validation", "benchmark_data"))
    parser.add_argument("--island-counts", default=os.path.join(
        siblings, "disco-surrogates", "tests", "data", "specs", "island_counts.csv"))
    parser.add_argument("--out", default=os.path.join(here, "artifacts", "disco_rehearsal"))
    parser.add_argument("--corpus", default="", help="skip the benchmark and read these cells")
    parser.add_argument("--no-loo", action="store_true",
                        help="skip evaluate's leave-one-out, which is n refits")
    parser.add_argument("--no-variant", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.benchmark_data, args.island_counts, args.out, corpus_csv=args.corpus,
                 variant=not args.no_variant, loo=not args.no_loo)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("feature_verdicts", "logo", "logo_learn_additional_noise")},
                     indent=2, sort_keys=True))
    print("features:", ", ".join(report["features"]))
    for name in ("logo", "logo_learn_additional_noise"):
        if name in report:
            t = report[name]
            print(f"{name}: groups {t['n_groups']} n {t['n']} r2 {t['r2']:+.3f} "
                  f"rmse {t['rmse']:.3f} (predict-mean {t['rmse_predict_mean']:.3f}) "
                  f"calibration {t['calibration']:.1f} sigma {t['sigma_mean']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
