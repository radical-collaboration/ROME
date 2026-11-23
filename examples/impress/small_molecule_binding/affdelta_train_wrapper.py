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

__all__ = ["run_round", "main", "Residual", "evaluate", "botorch_gp", "GP", "load_state",
           "STATE_NAME", "RESIDUAL_NAME", "TRAIN_COMPLETE_MARKER"]

#: This version's weights and the prior they correct, both inside the version directory.
#: The residual is **copied** rather than referenced: a path recorded in a manifest can be
#: moved or refitted, and then "v3 corrects public residual R" stops being checkable.
STATE_NAME = "gp.pt"
RESIDUAL_NAME = "residual.json"

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


class GP:
    """A fitted `SingleTaskGP`, callable as a `Posterior` and writable to a version directory.

    A class rather than the closure this used to return, because a closure is a posterior and
    nothing else: the fit it holds could not be written anywhere, so every round's model died
    with the process that fitted it (backlog item 52).

    What is saved is the training data **and** the fitted hyperparameters. A GP is defined by
    both -- re-fitting from the data at load time would re-run the marginal-likelihood
    optimisation and could land somewhere else, which is a different model wearing the same
    version number.
    """

    def __init__(self, model: Any, x: Any, y: Any, yvar: Any) -> None:
        self.model = model
        self.x, self.y, self.yvar = x, y, yvar

    def __call__(self, rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            post = self.model.posterior(torch.tensor(list(rows), dtype=torch.double))
            mean = post.mean.squeeze(-1).tolist()
            sigma = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).tolist()
        return list(mean), list(sigma)

    def save(self, version_dir: str) -> str:
        import torch  # noqa: PLC0415

        torch.save({"x": self.x, "y": self.y, "yvar": self.yvar,
                    "state_dict": self.model.state_dict()},
                   os.path.join(version_dir, STATE_NAME))
        return STATE_NAME


def _single_task_gp(x: Any, y: Any, yvar: Any):
    """The construction, in one place so a load rebuilds exactly what a fit built."""
    from botorch.models import SingleTaskGP  # noqa: PLC0415
    from botorch.models.transforms.outcome import Standardize  # noqa: PLC0415

    return SingleTaskGP(x, y, train_Yvar=yvar, outcome_transform=Standardize(m=1))


def load_state(version_dir: str, manifest: Any = None) -> GP:
    """A version directory in, a posterior out, with **no refit**.

    Raises when the directory holds no weights, which `streams.load_promoted` turns into an
    `Unavailable` naming the directory. A loader that refitted from the saved data instead
    would re-run the hyperparameter optimisation and could serve a different model under the
    same version number.
    """
    import torch  # noqa: PLC0415

    path = os.path.join(version_dir, STATE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {STATE_NAME} in {version_dir}: this version was published with a manifest "
            "and no weights, so there is nothing to serve"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = _single_task_gp(state["x"], state["y"], state["yvar"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return GP(model, state["x"], state["y"], state["yvar"])


def botorch_gp(train: List[Dict[str, Any]], job: Dict[str, Any]) -> GP:
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
    from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: PLC0415

    torch.manual_seed(int(job.get("seed", 0)))
    x = torch.tensor([e["z"] for e in train], dtype=torch.double)
    y = torch.tensor([[e["residual"]] for e in train], dtype=torch.double)
    yvar = torch.tensor([[e["noise"] ** 2] for e in train], dtype=torch.double)
    model = _single_task_gp(x, y, yvar)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return GP(model, x, y, yvar)


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


def _r2(truth: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    """1 - SSE/SST, or None when the truth has no variance to explain.

    None rather than 0.0 or 1.0: a corpus whose labels are all the same value has no R-squared,
    and inventing one for it is how a degenerate fold reads as a result.
    """
    if len(truth) < 2:
        return None
    bar = sum(truth) / len(truth)
    total = sum((t - bar) ** 2 for t in truth)
    if total <= 0.0:
        return None
    return 1.0 - sum((t - p) ** 2 for t, p in zip(truth, pred)) / total


def evaluate(examples: List[Dict[str, Any]], job: Dict[str, Any], residual: Residual,
             fit: Fitter) -> Dict[str, Any]:
    """Leave-one-out over 10-20 labels, which at this size is the only honest split -- and,
    separately, a real leave-one-GROUP-out when the groups allow one.

    A 20% holdout of twelve points is two points, and two points cannot calibrate anything.
    Leave-one-out refits n times, which costs seconds for a GP and buys every label a turn as
    held-out evidence. That is `loo_r2`, and it is reported always.

    **`logo_r2` used to be that same number under a grouped name** (item 73). The guard was
    `len(groups) >= 3 and len(groups) == len({e["group"] ...})` -- which asks whether the
    groups are *distinct*, the one condition under which leave-one-out and leave-one-group-out
    coincide -- and then it published the leave-one-*out* residuals anyway. No fold held a
    group out, no test caught it, and it became load-bearing the moment the promotion gate
    landed, because the gate reads `logo_r2` to decide whether a version generalises off its
    own scaffold cluster.

    So the two are now two numbers. `logo_r2` refits once per held-out group, in the shape
    `preorg_train_wrapper.evaluate` already used for Model B, and is **`None` with a reason**
    when the groups cannot support one -- which the gate reads as `INSUFFICIENT_EVIDENCE`
    rather than as a pass.
    """
    report: Dict[str, Any] = {
        "nll": None, "calibration": None, "loo_r2": None, "logo_r2": None,
        "n_holdout_groups": None, "holdout_n": 0,
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

    # Leave-one-OUT, named as such. It is what the n refits above already computed, and it is
    # a real number worth reporting -- it just is not a grouped one.
    report["loo_r2"] = _r2(truths, means)

    # Leave-one-GROUP-out: one refit per group, each predicting only the group it held out.
    # Ungrouped records are excluded rather than pooled into a nameless group, for the reason
    # `preorg.py::_group_of` gives: a group nobody assigned is not a group.
    grouped = [e for e in examples if e.get("group")]
    groups = sorted({str(e["group"]) for e in grouped})
    report["n_holdout_groups"] = len(groups)
    if len(groups) < 3:
        report["reason"] = (
            f"{len(groups)} scaffold clusters among {len(grouped)} grouped labels: a "
            "leave-one-group-out estimate needs at least three, and is near-meaningless "
            "below about four. loo_r2, nll and calibration are reported; logo_r2 is not "
            "invented."
        )
        return report

    pooled_truth: List[float] = []
    pooled_pred: List[float] = []
    for held in groups:
        inner_train = [e for e in grouped if str(e["group"]) != held]
        inner_test = [e for e in grouped if str(e["group"]) == held]
        if len(inner_train) < 2 or not inner_test:
            continue
        inner_mean, _ = fit(inner_train, job)([e["z"] for e in inner_test])
        pooled_pred.extend(inner_mean)
        pooled_truth.extend(e["residual"] for e in inner_test)
    if len(pooled_truth) >= 3:
        report["logo_r2"] = _r2(pooled_truth, pooled_pred)
    if report["logo_r2"] is None:
        report["reason"] = (
            f"{len(groups)} clusters, and no grouped fold produced one: every split left "
            "fewer than two training labels, or the held-out residuals had no variance. "
            "logo_r2 is not invented."
        )
    return report


# -- the round ---------------------------------------------------------------


def run_round(job: Dict[str, Any], *, fit: Optional[Fitter] = None) -> str:
    """Recalibrate, evaluate leave-one-out, fit on everything, write the weights.

    **The evaluated model and the published model are two different fits.** The metrics come
    from n leave-one-out refits; the weights come from a final fit on all the labels, because
    the version that gets served should have seen every measurement that was paid for.

    **The weights land before the manifest**, so no reader can see a published version that
    has none, and the residual is **copied** in beside them rather than referenced.
    """
    started = time.process_time()
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    residual = Residual.load(job["residual_path"])

    examples = []
    for example in job["examples"]:
        z = residual.project(example["x"])
        examples.append({**example, "z": z,
                         "residual": example["y"] - residual.predict(z)})
    fitter = fit or botorch_gp
    report = evaluate(examples, job, residual, fitter)

    weights = "none"
    if report["validated"]:
        final = fitter(examples, job)
        saver = getattr(final, "save", None)
        if saver is not None:
            version_dir = _version_dir(job)
            os.makedirs(version_dir, exist_ok=True)
            with open(job["residual_path"]) as src, \
                    open(os.path.join(version_dir, RESIDUAL_NAME), "w") as dst:
                dst.write(src.read())
            weights = saver(version_dir)
    report["weights"] = weights
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
    from asteer.features import FEATURISER_FOR_MODEL  # noqa: PLC0415
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    registry = ModelRegistry(job["registry_root"])
    version = _next_version(job)
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
        # The gate reads the HOLDOUT group count, not the corpus one (asteer J5, measured in
        # J2: an honest clustering of this data is mostly singletons, so a corpus-wide count
        # clears a bar of four for free).
        n_holdout_groups=report["n_holdout_groups"],
        # Resolved here rather than carried in the job, so a driver cannot name a featuriser
        # the model does not use. This wrapper runs inside the task and already imports from
        # asteer; the driver, which is constructed in the campaign process, still imports
        # nothing (CON-4).
        featuriser=FEATURISER_FOR_MODEL.get(str(job["model_id"])),
        trained_on=str(job.get("property_name") or ""),
        min_holdout_n=int(job["min_labels"]),
        min_effect_size=float(job["min_effect_size"]),
        train_seconds_cpu=elapsed,
        promoted=False,
        notes=" ".join(filter(None, [
            str(job.get("notes") or ""),
            # Leave-one-OUT, reported beside the grouped one and never as it (item 73).
            f"loo_r2={report['loo_r2']}" if report["loo_r2"] is not None else "",
            # Which prior this version corrects. Without it a `v0` is an unlabelled starting
            # point rather than "public residual R, zero in-house labels".
            f"residual=v{report['residual_version']}",
            f"labels={report['n_labels']}",
            # What a stream will find in this version's directory.
            f"weights={report.get('weights', 'none')}",
            report["reason"],
        ])).strip(),
    ))
    return version


def _registry(job: Dict[str, Any]):
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    return ModelRegistry(job["registry_root"])


def _next_version(job: Dict[str, Any]) -> int:
    existing = _registry(job).versions(job["model_id"])
    return (max(existing) + 1) if existing else 0


def _version_dir(job: Dict[str, Any]) -> str:
    return str(_registry(job).version_dir(job["model_id"], _next_version(job)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job) as fh:
        run_round(json.load(fh))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
