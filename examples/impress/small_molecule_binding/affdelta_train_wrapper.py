"""Model C's recalibration round: read a residual, correct it, report an honest sigma.

**The residual is read and never written.** It carries its own version, its PCA basis, its
centre and scale, and the linear prior fitted offline on public affinity data. This round
projects the in-house labels into *that* basis, subtracts the prior, and fits a GP to what is
left. Re-fitting the projection on ten points would move the basis the prior was trained in,
and the prior would then be evaluated somewhere it never saw -- which is the refit this stage
exists to avoid, wearing the name of a recalibration.

**Known observation noise, not a learned one.** Every ATM label carries its own `sigma`, and a
GP told to infer a single homoscedastic noise from ten points will infer it from whichever
points it fits first. `train_Yvar` takes the measurement's own uncertainty, which is what makes
the posterior's sigma mean something.

**σ is the output, not a diagnostic.** Item 14 ranks on `dg_corrected + k·sigma_dg`, so a
collapsed σ turns a lower confidence bound into a posterior mean and the ranking silently stops
accounting for uncertainty. impress-a's `calibration_in_range` gate reads this number with a
default band of 0.1 to 5.0, and a round reports the held-out σ so the gate has something true
to read.

**Below `min_labels` there is no GP.** The round publishes a version whose held-out metrics are
`None`, which the promotion gate reads as INSUFFICIENT_EVIDENCE. A GP fitted on four points
would answer every query with the prior and a confident-looking σ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

TRAIN_COMPLETE_MARKER = "train_complete"

__all__ = ["run_round", "main", "Residual", "evaluate", "botorch_gp",
           "TRAIN_COMPLETE_MARKER"]

Posterior = Callable[[Sequence[Sequence[float]]], Tuple[List[float], List[float]]]
Fitter = Callable[[List[Dict[str, Any]], Dict[str, Any]], Posterior]


# -- the thing that is read, not fitted --------------------------------------


class Residual:
    """The offline prior: a PCA basis plus a linear residual, both fixed.

    Loaded from JSON so the artifact is inspectable and diffable, and so a round can record
    which `version` it read without the registry having to understand the model.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self.version = int(data["version"])
        self.components = [list(map(float, row)) for row in data["components"]]
        self.centre = [float(v) for v in data["centre"]]
        self.scale = [float(v) or 1.0 for v in data["scale"]]
        self.coef = [float(v) for v in data["coef"]]
        self.intercept = float(data["intercept"])
        self.notes = str(data.get("notes", ""))
        if len(self.components) != len(self.coef):
            raise ValueError(
                f"residual v{self.version}: {len(self.components)} components but "
                f"{len(self.coef)} coefficients -- the basis and the prior disagree, which "
                "would project into one space and predict in another"
            )

    @classmethod
    def load(cls, path: str) -> "Residual":
        with open(path) as fh:
            return cls(json.load(fh))

    def project(self, x: Sequence[float]) -> List[float]:
        """Into the residual's own basis. Never re-fitted here; see the module docstring."""
        if len(x) != len(self.centre):
            raise ValueError(
                f"residual v{self.version} was fitted on {len(self.centre)}-dim embeddings "
                f"and got {len(x)}: a round cannot reduce an embedding it was not built for"
            )
        centred = [(float(v) - c) / s for v, c, s in zip(x, self.centre, self.scale)]
        return [sum(w * z for w, z in zip(row, centred)) for row in self.components]

    def predict(self, z: Sequence[float]) -> float:
        return self.intercept + sum(c * v for c, v in zip(self.coef, z))


# -- the fit (the only heavy import in this example's C half) ----------------


def botorch_gp(train: List[Dict[str, Any]], job: Dict[str, Any]) -> Posterior:
    """A Matérn-5/2 GP with **known** per-point noise, on the residual's basis.

    botorch's `SingleTaskGP` defaults to a scaled Matérn-5/2 with ARD, which is the kernel the
    specification names; `train_Yvar` is what makes the noise the measurement's own rather
    than something inferred from ten points. `Standardize` keeps the outcome scaling out of
    the kernel's hyperparameters.

    The same construction is in use elsewhere in this group -- `protein-surrogate/sitegp` and
    `disco-surrogates` both fit botorch GPs with known heteroscedastic noise -- which is why
    this is a call rather than a hand-rolled kernel. A third implementation of a GP would be
    the divergence CON-8 exists to prevent.
    """
    import torch  # noqa: PLC0415
    from botorch.fit import fit_gpytorch_mll  # noqa: PLC0415
    from botorch.models import SingleTaskGP  # noqa: PLC0415
    from botorch.models.transforms.outcome import Standardize  # noqa: PLC0415
    from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415

    torch.manual_seed(int(job.get("seed", 0)))
    x = torch.tensor([e["z"] for e in train], dtype=torch.double)
    y = torch.tensor([[e["residual"]] for e in train], dtype=torch.double)
    yvar = torch.tensor([[e["noise"] ** 2] for e in train], dtype=torch.double)
    model = SingleTaskGP(x, y, train_Yvar=yvar, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()

    def posterior(rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        with torch.no_grad():
            post = model.posterior(torch.tensor(list(rows), dtype=torch.double))
            mean = post.mean.squeeze(-1).tolist()
            sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).tolist()
        return list(mean), list(sigma)

    return posterior


# -- the evaluation ----------------------------------------------------------


def _calibration(y: Sequence[float], mean: Sequence[float], sigma: Sequence[float]) -> float:
    """Mean squared z-score; 1.0 is calibrated, above 1 overconfident."""
    return sum(((t - m) / max(float(s), 1e-6)) ** 2 for t, m, s in zip(y, mean, sigma)) / max(
        len(y), 1)


def _nll(y: Sequence[float], mean: Sequence[float], sigma: Sequence[float]) -> float:
    total = 0.0
    for truth, mu, sd in zip(y, mean, sigma):
        var = max(float(sd) ** 2, 1e-12)
        total += 0.5 * (math.log(2.0 * math.pi * var) + (truth - mu) ** 2 / var)
    return total / max(len(y), 1)


def evaluate(examples: List[Dict[str, Any]], job: Dict[str, Any], residual: Residual,
             fit: Fitter) -> Dict[str, Any]:
    """Leave-one-out over 10–20 labels, which at this size is the only honest split.

    A 20% holdout of twelve points is two points, and two points cannot calibrate anything.
    Leave-one-out refits n times, which costs seconds for a GP and buys every label a turn as
    held-out evidence.
    """
    report: Dict[str, Any] = {
        "nll": None, "calibration": None, "logo_r2": None, "holdout_n": 0,
        "sigma_dg_mean": None, "n_labels": len(examples),
        "residual_version": residual.version, "validated": False, "reason": "",
    }
    if len(examples) < int(job["min_labels"]):
        report["reason"] = (
            f"{len(examples)} labels, below the declared floor of {job['min_labels']}. A GP "
            "fitted here would answer every query with the prior and a confident-looking "
            "sigma, which is worse than reporting nothing."
        )
        return report

    means: List[float] = []
    sigmas: List[float] = []
    truths: List[float] = []
    for i in range(len(examples)):
        rest = examples[:i] + examples[i + 1:]
        mean, sigma = fit(rest, job)([examples[i]["z"]])
        means.append(mean[0])
        sigmas.append(sigma[0])
        truths.append(examples[i]["residual"])
    report["nll"] = _nll(truths, means, sigmas)
    report["calibration"] = _calibration(truths, means, sigmas)
    report["holdout_n"] = len(truths)
    report["sigma_dg_mean"] = sum(sigmas) / len(sigmas)
    report["validated"] = True

    groups = sorted({str(e["group"]) for e in examples if e.get("group")})
    if len(groups) >= 3 and len(groups) == len({e.get("group") for e in examples}):
        bar = sum(truths) / len(truths)
        total = sum((t - bar) ** 2 for t in truths)
        if total > 0:
            report["logo_r2"] = 1.0 - sum(
                (t - m) ** 2 for t, m in zip(truths, means)) / total
    return report


# -- the round ---------------------------------------------------------------


def run_round(job: Dict[str, Any], *, fit: Optional[Fitter] = None) -> str:
    started = time.process_time()
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    residual = Residual.load(job["residual_path"])

    examples = []
    for example in job["examples"]:
        z = residual.project(example["x"])
        examples.append({**example, "z": z,
                         "residual": example["y"] - residual.predict(z)})
    report = evaluate(examples, job, residual, fit or botorch_gp)
    elapsed = time.process_time() - started

    version = _publish(job, report, elapsed)
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump({**report, "version": version, "train_seconds_cpu": elapsed}, fh, indent=2)
    with open(os.path.join(output_dir, TRAIN_COMPLETE_MARKER), "w") as fh:
        fh.write(str(version))
    return output_dir


def _publish(job: Dict[str, Any], report: Dict[str, Any], elapsed: float) -> int:
    import datetime  # noqa: PLC0415

    from asteer.contracts import ModelManifest  # noqa: PLC0415
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    registry = ModelRegistry(job["registry_root"])
    existing = registry.versions(job["model_id"])
    version = (max(existing) + 1) if existing else 0
    registry.publish(ModelManifest(
        model_id=job["model_id"],
        version=version,
        trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corpus_size=int(job["corpus_size"]),
        git_sha=str(job.get("git_sha") or ""),
        parent_version=job.get("parent_version"),
        nll=report["nll"],
        calibration=report["calibration"],
        logo_r2=report["logo_r2"],
        holdout_n=report["holdout_n"],
        min_holdout_n=int(job["min_labels"]),
        min_effect_size=float(job["min_effect_size"]),
        train_seconds_cpu=elapsed,
        promoted=False,
        notes=" ".join(filter(None, [
            str(job.get("notes") or ""),
            # Which prior this version corrects. Without it a `v0` is an unlabelled starting
            # point rather than "public residual R, zero in-house labels".
            f"residual=v{report['residual_version']}",
            f"labels={report['n_labels']}",
            report["reason"],
        ])).strip(),
    ))
    return version


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job) as fh:
        run_round(json.load(fh))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
