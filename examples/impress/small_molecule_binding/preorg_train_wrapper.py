"""Model B's round: the half that owns torch, and the half that owns the honesty.

Two things happen here and they are deliberately separable. **Fitting** is a five-member deep
ensemble of small MLPs with a heteroscedastic Gaussian head -- torch, and the only torch in
this example. **Evaluating** is a scaffold-cluster split, a held-out NLL, a calibration number
and a leave-one-group-out `r2` -- numpy and the standard library, and testable on a machine
with no torch at all.

The split matters because the evaluation is where a surrogate lies. A model that memorised a
near-duplicate of a held-out design scores well and reports a generalisation it does not have,
and nothing in the numbers shows it. So `fit` is an argument with a default rather than a
hard-wired call: a test drives a whole round -- split, metrics, manifest, completion marker --
with a deterministic stand-in, and what is exercised is every line that decides what gets
reported.

**`train_complete` is written last**, after the manifest is on disk, because the training
manager polls for that marker rather than for the checkpoint: on Dragon a task can run to
completion and never resolve its future.

**Insufficient evidence is a result, not a failure.** `LabelRecord` carries no scaffold
cluster -- the cluster is a property of the design, which lives in impress-a's tree -- so a
corpus can arrive with nothing to group by. When it does, this writes a manifest with
`nll`, `calibration` and `logo_r2` all `None`, which the promotion gate reads as
`INSUFFICIENT_EVIDENCE` rather than as a pass. The alternative, a random split, is the single
most common way a surrogate's reported accuracy is wrong.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Mirrors `rome.trainer.TRAIN_COMPLETE_MARKER`; a literal so this file imports nothing from
#: ROME and can be run as a bare script on an allocation.
TRAIN_COMPLETE_MARKER = "train_complete"

__all__ = ["run_round", "main", "scaffold_split", "evaluate", "Ensemble", "Bundle",
           "load_state", "preorg_predictor", "STATE_NAME", "TRAIN_COMPLETE_MARKER",
           "DERIVED_PROPERTY", "SCALE_FLOOR", "SERVED_PROPERTIES"]

#: Where a published version keeps its weights. `ModelRegistry.version_dir` reserves the
#: directory and deliberately prescribes no format -- *"a trainer writes its weights in here
#: in whatever shape it likes"* -- so the name is the trainer's to declare and the stream's
#: loader to read.
#: The one metric Model B reports that is derived rather than measured (decision L2).
#:
#: **Spelled here and in `preorg.py`, with a test asserting the two agree** -- the same shape
#: item 66 settled for `PROPERTY_FOR_METRIC`. The driver must import nothing heavy (CON-4) and
#: this module owns torch, so the wrapper cannot import the driver and the driver cannot import
#: the wrapper. A test is what keeps one string one string.
DERIVED_PROPERTY = "induced_fit_risk"

#: The three quantities a served version must hold, because `preorg_score/spec.yaml` puts all
#: four of its metrics in `metrics_reported` and a missing one is an integrity failure by the
#: toolkit's own rule.
#:
#: A round may name fewer -- a corpus with only one label type is a real situation and fitting
#: it is honest -- but the result is **not servable**, and `preorg_predictor` says so by name
#: rather than serving three keys and omitting the fourth. `sigmoid(z)` over one quantity is a
#: perfectly good number and a different one from the judgement over three, and giving it the
#: same name is how a metric quietly changes meaning between versions.
SERVED_PROPERTIES: tuple = ("preorg_rmsd", "rotamer_entropy", "ligand_rmsf")

#: The smallest standard deviation a property's corpus may have before the z-score stops
#: meaning anything. A corpus whose labels are all equal has no scale, and dividing by zero
#: would make every z infinite rather than undefined -- so the floor is what keeps the risk a
#: number, and it is large enough that a degenerate corpus produces a risk near 0.5 rather
#: than a confident 0 or 1.
SCALE_FLOOR = 1e-6

STATE_NAME = "ensemble.pt"

Predictor = Callable[[Sequence[Sequence[float]]], Tuple[List[float], List[float]]]
Fitter = Callable[[List[Dict[str, Any]], Dict[str, Any]], Predictor]


# -- the split ---------------------------------------------------------------


def scaffold_split(examples: Sequence[Dict[str, Any]], fraction: float,
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Whole scaffold clusters to one side or the other, largest-first.

    **The boundary never splits a cluster.** A random split leaves near-duplicates of held-out
    designs in the training set, so the model scores well by having memorised a neighbour. That
    is invisible in the reported numbers, which is what makes it worth code rather than a note.

    Largest-first overshoots the fraction less than filling smallest-first, and is
    deterministic given the same clusters. Returns `([], [])` when any example is ungrouped:
    the caller reports insufficient evidence rather than falling back to a random split.

    This rule is also implemented in AuCaM's `campaigns/asd_campaign/acquisition.py` over a
    different record type. Two implementations of one rule is a known duplication and is
    tracked; the rule is short and its failure is silent, so neither side guesses.
    """
    if any(not e.get("group") for e in examples):
        return [], []
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        clusters.setdefault(str(example["group"]), []).append(example)
    target = len(examples) * fraction
    holdout: List[Dict[str, Any]] = []
    for name in sorted(clusters, key=lambda k: (-len(clusters[k]), k)):
        if len(holdout) + len(clusters[name]) > target and holdout:
            continue
        holdout.extend(clusters[name])
        if len(holdout) >= target:
            break
    held = {id(e) for e in holdout}
    train = [e for e in examples if id(e) not in held]
    if not train or not holdout:
        return [], []
    return train, holdout


# -- the metrics -------------------------------------------------------------


def _gaussian_nll(y: Sequence[float], mean: Sequence[float], sigma: Sequence[float]) -> float:
    """Mean negative log likelihood of a heteroscedastic Gaussian, in nats."""
    total = 0.0
    for truth, mu, sd in zip(y, mean, sigma):
        var = max(float(sd) ** 2, 1e-12)
        total += 0.5 * (math.log(2.0 * math.pi * var) + (truth - mu) ** 2 / var)
    return total / max(len(y), 1)


def _calibration(y: Sequence[float], mean: Sequence[float], sigma: Sequence[float]) -> float:
    """Mean squared z-score. **1.0 is calibrated**, above 1 overconfident, below 1 timid.

    Chosen over a coverage fraction because at 4-8 held-out points a coverage number can only
    take a handful of values and reads as precise when it is not.
    """
    total = 0.0
    for truth, mu, sd in zip(y, mean, sigma):
        total += ((truth - mu) / max(float(sd), 1e-6)) ** 2
    return total / max(len(y), 1)


def _r2(y: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    if len(y) < 2:
        return None
    bar = sum(y) / len(y)
    total = sum((t - bar) ** 2 for t in y)
    if total <= 0.0:
        return None
    residual = sum((t - p) ** 2 for t, p in zip(y, pred))
    return 1.0 - residual / total


def evaluate(examples: List[Dict[str, Any]], job: Dict[str, Any], fit: Fitter,
             ) -> Dict[str, Any]:
    """Held-out NLL and calibration, plus leave-one-group-out `r2`, or why neither exists.

    `logo_r2` is reported **even when negative**: below zero means predicting the campaign's
    mean beats the model, which is exactly the finding that catches a self-confirming
    objective, and omitting it would turn the one honest number into a missing one.

    The **group count rides beside the value** (CON-13). At 20-40 samples with a 20% holdout a
    round is judged on 4-8 points, and `logo_r2` is near-meaningless below about four groups,
    so a gate that saw only the number would read noise as evidence.
    """
    report: Dict[str, Any] = {
        "nll": None, "calibration": None, "logo_r2": None,
        "holdout_n": 0, "n_groups": len({e.get("group") for e in examples if e.get("group")}),
        # Groups the grouped fold ACTUALLY held out, which is not the corpus count: a group
        # whose removal leaves fewer than two training rows is skipped below, and J2 measured
        # a clustering of this data that is 2,487 singletons out of 2,508 groups. The gate
        # reads this one (asteer J5, rule 3).
        "n_holdout_groups": 0,
        "validated": False, "reason": "",
    }
    train, holdout = scaffold_split(examples, float(job["holdout_fraction"]))
    if not holdout:
        report["reason"] = (
            "no scaffold-cluster split was possible: at least one record carries no "
            f"metadata['scaffold'], or one cluster holds the whole corpus of {len(examples)}. "
            "A random split would report a generalisation this model has not shown."
        )
        return report
    if len(holdout) < int(job["min_holdout_n"]):
        report["reason"] = (
            f"{len(holdout)} held-out points, below the declared floor of "
            f"{job['min_holdout_n']} (CON-13). The split happened; the evidence is too thin "
            "to read, which is a different thing from the model being bad."
        )
        report["holdout_n"] = len(holdout)
        return report

    predict = fit(train, job)
    mean, sigma = predict([e["x"] for e in holdout])
    truth = [e["y"] for e in holdout]
    report["nll"] = _gaussian_nll(truth, mean, sigma)
    report["calibration"] = _calibration(truth, mean, sigma)
    report["holdout_n"] = len(holdout)

    groups = sorted({str(e["group"]) for e in examples})
    if len(groups) >= 3:
        pooled_truth: List[float] = []
        pooled_pred: List[float] = []
        held_out = 0
        for held in groups:
            inner_train = [e for e in examples if str(e["group"]) != held]
            inner_test = [e for e in examples if str(e["group"]) == held]
            if len(inner_train) < 2 or not inner_test:
                continue
            inner_mean, _ = fit(inner_train, job)([e["x"] for e in inner_test])
            pooled_pred.extend(inner_mean)
            pooled_truth.extend(e["y"] for e in inner_test)
            held_out += 1
        report["n_holdout_groups"] = held_out
        if len(pooled_truth) >= 3:
            report["logo_r2"] = _r2(pooled_truth, pooled_pred)
    report["validated"] = report["logo_r2"] is not None
    if not report["validated"]:
        report["reason"] = (
            f"{len(groups)} scaffold clusters: a leave-one-group-out estimate needs at least "
            "three, and is near-meaningless below about four. Held-out NLL and calibration "
            "are reported; logo_r2 is not invented."
        )
    return report


# -- the fit (the only torch in this example) --------------------------------


def _nets(n_features: int, job: Dict[str, Any]):
    """The architecture, built in one place so a load rebuilds exactly what a fit built.

    Two copies of a layer stack is how a checkpoint comes to load into the wrong shape and
    still produce finite numbers -- `torch.load` would raise on a size mismatch, but a
    changed activation or a changed output width would not.
    """
    from torch import nn  # noqa: PLC0415

    members = []
    for _ in range(int(job["n_members"])):
        layers: List[Any] = []
        width = n_features
        for hidden in job["hidden"]:
            layers += [nn.Linear(width, int(hidden)), nn.SiLU()]
            width = int(hidden)
        members.append(nn.Sequential(*layers, nn.Linear(width, 2)))
    return members


class Ensemble:
    """Five nets, callable as a `Predictor` and writable to a version directory.

    A class rather than the closure this used to return, because a closure is a `Predictor`
    and nothing else: the weights it holds cannot be written anywhere, so every round's model
    died with the process that fitted it (backlog item 52).
    """

    def __init__(self, members: List[Any], job: Dict[str, Any], n_features: int) -> None:
        self.members = members
        self.hidden = [int(h) for h in job["hidden"]]
        self.n_members = int(job["n_members"])
        self.n_features = int(n_features)
        # **What this ensemble actually regressed on, and what produced its inputs.**
        # Both travel with the weights rather than being read off a module constant at serve
        # time (J8). The property because `preorg_predictor` used to take it from
        # `preorg.DEFAULT_PROPERTY`, so an ensemble fitted on `rotamer_entropy` would have
        # served its numbers under the key `preorg_rmsd`; the featuriser because a predictor
        # that embeds with one and was fitted with another is wrong in a way that produces
        # finite, plausible numbers.
        self.property_name = str(job.get("property_name") or "")
        self.featuriser = str(job.get("featuriser") or "")

    def __call__(self, rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            grid = torch.tensor(list(rows), dtype=torch.float32)
            means, variances = [], []
            for net in self.members:
                out = net(grid)
                means.append(out[:, 0])
                variances.append(out[:, 1].clamp(-6.0, 4.0).exp())
            stacked = torch.stack(means)
            mu = stacked.mean(0)
            # Total variance = mean aleatoric + spread of the means (epistemic).
            var = torch.stack(variances).mean(0) + stacked.var(0, unbiased=False)
            return mu.tolist(), var.clamp(min=1e-12).sqrt().tolist()

    def save(self, version_dir: str) -> str:
        import torch  # noqa: PLC0415

        path = os.path.join(version_dir, STATE_NAME)
        torch.save({"hidden": self.hidden, "n_members": self.n_members,
                    "n_features": self.n_features,
                    "property_name": self.property_name,
                    "featuriser": self.featuriser,
                    "members": [m.state_dict() for m in self.members]}, path)
        return STATE_NAME


def load_state(version_dir: str, manifest: Any = None) -> Bundle:
    """The `build` a stream hands `surrogate_stream`: a version directory in, a `Bundle` out.

    Raises when the directory holds no weights, which `streams.load_promoted` turns into an
    `Unavailable` naming the directory. A loader that fell back to an untrained ensemble would
    serve a confident-looking sigma from no evidence at all.

    **A pre-J12 single-property state file is refused by name**, not silently adapted. Its
    numbers are fine; what it does not carry is the corpus scale, and without that
    `induced_fit_risk` cannot be derived -- so serving it would mean either omitting a metric
    `metrics_reported` requires or inventing the scale here, and the second is how a risk
    acquires a unit nobody wrote down.
    """
    import torch  # noqa: PLC0415

    path = os.path.join(version_dir, STATE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {STATE_NAME} in {version_dir}: this version was published with a manifest "
            "and no weights, so there is nothing to serve"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "ensembles" not in state:
        raise ValueError(
            f"{version_dir} holds a single-property ensemble, written before J12. It carries "
            "no corpus scale, so `induced_fit_risk` cannot be derived from it -- and "
            f"`preorg_score` requires all four metrics. Refit the round; it now fits "
            "every property it names in one pass."
        )
    ensembles: Dict[str, Any] = {}
    for prop, saved in state["ensembles"].items():
        job = {"n_members": saved["n_members"], "hidden": saved["hidden"],
               "property_name": prop, "featuriser": state.get("featuriser", "")}
        members = _nets(int(saved["n_features"]), job)
        for net, weights in zip(members, saved["members"]):
            net.load_state_dict(weights)
            net.eval()
        ensembles[prop] = Ensemble(members, job, int(saved["n_features"]))
    return Bundle(ensembles, state.get("scale") or {}, state.get("featuriser", ""))


def preorg_predictor(version_dir: str, manifest: Any = None):
    """The `build` a stream is handed: a version directory in, `features -> values` out.

    **The featuriser is `asteer.features.embed`, and that is the whole point of this function.**
    Model B trains on `LabelRecord.metadata["embedding"]` and impress-a's `preorg_score` client
    sends `{"structure", "pocket_selection"}`; until 2026-11-22 the only thing that turned one
    into the other lived in a test harness, where only the serving side could reach it and the
    corpus side was written by hand to match. Going through `embed` here means a corpus row and
    a score request agree because they share a function rather than because two copies were
    kept in step.

    **Which featuriser is named, and it is Model B's own** (asteer J4). `embed` no longer has
    a default: a single module-level featuriser is what let a corpus row and a request disagree
    while both looked fine, and `FEATURISER_FOR_MODEL` is now the one place that correspondence
    is written. This asks for it by model id rather than by name, so the day Model B's
    featuriser changes, this call follows without being edited.

    **And both are checked against the weights rather than assumed** (J8). Two separate
    defects, both of which produce finite, plausible numbers:

    - the served key used to be `preorg.DEFAULT_PROPERTY`, a module constant, so an ensemble
      fitted on `rotamer_entropy` would have served its numbers under `preorg_rmsd` -- which
      goes live the moment J12 makes one round fit more than one property;
    - a predictor that embeds with `pocket-geom-1` while the weights were fitted on
      `standin-1` feeds the model nine real columns it has never seen. Nothing in the
      arithmetic complains; the numbers are simply wrong.

    A payload `embed` refuses raises out of the stream, which `streams.serve_requests` turns
    into a failed response for that request alone. That is the right blast radius: one client
    sent something unfeaturisable, and the other two surrogates keep serving.
    """
    from asteer.features import embed, featuriser_for_model  # noqa: PLC0415

    ensemble = load_state(version_dir, manifest)

    # The model id from the manifest when there is one, and otherwise from the registry's own
    # layout, `<root>/<model_id>/v<n>`. **Not `from streams import MODEL_B`**: that resolves
    # only when the example directory is on `sys.path`, which is true when the wrapper runs
    # as a script and false everywhere else -- and nothing exercised this function until J8's
    # tests, so it had never been found.
    model_id = getattr(manifest, "model_id", "") or os.path.basename(
        os.path.dirname(os.path.abspath(version_dir))
    )
    featuriser = featuriser_for_model(model_id)

    # The properties this version regresses on, READ FROM THE WEIGHTS. Three regressions
    # sharing a corpus, not one multi-output head -- which ones a version holds is a fact
    # about the fit, not about the driver's import.
    props = ensemble.properties
    if not props:
        raise ValueError(
            f"{version_dir} holds a bundle that names no property. Serving it would mean "
            "choosing a key for it here, which is the defect J8 removed. Refit the round."
        )
    unservable = [p for p in SERVED_PROPERTIES if p not in props]
    if unservable:
        raise ValueError(
            f"{version_dir} holds {props} and `preorg_score` requires "
            f"{list(SERVED_PROPERTIES)} plus the derived {DERIVED_PROPERTY!r} in "
            f"`metrics_reported`, so it is missing {unservable}. Serving three keys and "
            "omitting the fourth is an integrity failure by the toolkit's own rule. The round "
            "that produced this fitted what it named; name all three and refit."
        )
    missing_scale = [p for p in props if p not in ensemble.scale]
    if missing_scale:
        raise ValueError(
            f"{version_dir} carries no corpus scale for {missing_scale}, so "
            f"{DERIVED_PROPERTY!r} cannot be z-scored against it. The risk is a judgement "
            "over three quantities with different units; without their scale it is a number "
            "with a unit nobody wrote down (J12, decision L2)."
        )
    if not ensemble.featuriser:
        raise ValueError(
            f"{version_dir} holds an ensemble that does not name the featuriser its inputs "
            "came from, so there is no way to tell whether it matches the one this stream "
            "embeds with. Written before J8; refit the round."
        )
    if ensemble.featuriser != featuriser.name:
        raise ValueError(
            f"{version_dir} was fitted with featuriser {ensemble.featuriser!r} and "
            f"{model_id} now embeds with {featuriser.name!r} ({featuriser.width} columns). "
            "Serving it would feed the model inputs it has never seen, and the arithmetic "
            "would not complain. Refit the round against the current featuriser, or promote "
            "a version that was."
        )

    def predict(features):
        row = embed(features, featuriser=featuriser)
        means, sigmas = ensemble([row])
        value = {prop: means[prop][0] for prop in props}
        sigma = {prop: sigmas[prop][0] for prop in props}
        # **The risk is derived here and never measured** (decision L2), and here is the
        # stream rather than the agent because the scale it z-scores against is a fact about
        # the fit. `PreorgScoreAgent` only passes values through; it has no corpus.
        value[DERIVED_PROPERTY], sigma[DERIVED_PROPERTY] = ensemble.risk(value, sigma)
        return value, sigma

    return predict


# -- the fit (the only torch in this example) --------------------------------


def torch_ensemble(train: List[Dict[str, Any]], job: Dict[str, Any]) -> Ensemble:
    """A deep ensemble with a heteroscedastic Gaussian head.

    Two uncertainties, and they are different things. The head's `log_var` is **aleatoric** --
    the noise in the measurement itself, which short MD genuinely has. The spread across
    members is **epistemic** -- what the model does not know because nobody measured there. A
    single net reports only the first and reports it confidently, which is the failure this
    model exists to avoid at 20-40 samples.

    `log_var` is clamped. An unclamped NLL on a handful of points drives the variance to zero
    on whatever it fits first and the loss follows it down, which looks like convergence.
    """
    import torch  # noqa: PLC0415

    # The fitted ensemble carries its own provenance, so a predictor never has to assume it.
    job = {**job, "featuriser": _featuriser_for(job)}

    x = torch.tensor([e["x"] for e in train], dtype=torch.float32)
    y = torch.tensor([e["y"] for e in train], dtype=torch.float32).unsqueeze(-1)
    members = _nets(int(x.shape[1]), job)
    for member, net in enumerate(members):
        torch.manual_seed(int(job["seed"]) + member)
        for layer in net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        optimiser = torch.optim.Adam(net.parameters(), lr=float(job["learning_rate"]))
        for _ in range(int(job["epochs"])):
            optimiser.zero_grad()
            out = net(x)
            mean, log_var = out[:, :1], out[:, 1:].clamp(-6.0, 4.0)
            loss = 0.5 * (log_var + (y - mean) ** 2 / log_var.exp()).mean()
            loss.backward()
            optimiser.step()
        net.eval()
    return Ensemble(members, job, int(x.shape[1]))


# -- the round ---------------------------------------------------------------


def _featuriser_for(job: Dict[str, Any]) -> str:
    """The featuriser this model embeds with, by model id.

    Resolved inside the task rather than carried from the driver: the driver is constructed
    in the campaign process and imports nothing from asteer (CON-4), and a driver that could
    name a featuriser could name one the model does not use. One helper so the fit and the
    manifest cannot disagree about it -- which is the class of defect J4 exists for.
    """
    from asteer.features import FEATURISER_FOR_MODEL  # noqa: PLC0415

    return FEATURISER_FOR_MODEL.get(str(job.get("model_id") or ""), "")


class Bundle:
    """One round's worth of Model B: an `Ensemble` per property, and the corpus scale.

    **One artifact, not three**, because a version is one promotion decision: three files in a
    version directory would let a reader see two properties served from a version whose third
    never fitted, and `metrics_reported` would fail with the version looking complete.

    The **scale is what makes `induced_fit_risk` derivable at all** (J12). The risk is a
    judgement over three quantities with different units, so it needs each one's corpus mean
    and standard deviation to z-score against -- and those are a fact about the *fit*, not
    about the request, so they travel with the weights. `PreorgScoreAgent` has no access to
    them, which is exactly why decision L2's risk is derived in the stream and not in the
    agent: a risk without a scale is a number with a unit nobody wrote down.
    """

    #: The three weights of the risk combination. **Declared, not fitted**, and a tuple rather
    #: than a parameter so that changing them is a diff somebody reads. There is no
    #: `induced_fit_risk` ground truth anywhere, so a fitted weight would be evidence that is
    #: not there, dressed as a measurement.
    RISK_WEIGHTS: Dict[str, float] = {}

    def __init__(self, ensembles: Dict[str, Any], scale: Dict[str, Any],
                 featuriser: str = "") -> None:
        self.ensembles = dict(ensembles)
        #: property -> (mean, sd) over that property's training rows. `sd` is floored, because
        #: a corpus whose labels are all equal has no scale and a zero divisor would make
        #: every z-score infinite rather than undefined.
        self.scale = {k: (float(v[0]), max(float(v[1]), SCALE_FLOOR))
                      for k, v in scale.items()}
        self.featuriser = str(featuriser)

    @property
    def properties(self) -> List[str]:
        """In the order the round named them, which is the order the metrics are reported."""
        return list(self.ensembles)

    def __call__(self, rows: Sequence[Sequence[float]]
                 ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        means: Dict[str, List[float]] = {}
        sigmas: Dict[str, List[float]] = {}
        for prop, ensemble in self.ensembles.items():
            mean, sigma = ensemble(rows)
            means[prop], sigmas[prop] = list(mean), list(sigma)
        return means, sigmas

    def risk(self, means: Dict[str, float], sigmas: Dict[str, float]) -> Tuple[float, float]:
        """`sigmoid(mean(z))` over the three z-scores, and its propagated sigma.

        **All three quantities are minimised**, so a prediction above the corpus mean is a
        pocket that reorganises more than the corpus typically does -- which is what the risk
        is about. A positive mean z therefore means higher risk, and `sigmoid` is what bounds
        it into the [0, 1] the acquisition rule compares against a ceiling.

        The sigma is the sigmoid's derivative times the combined z sigma, which is the
        ordinary first-order propagation and is *declared* as such: it is exact for a linear
        map and an approximation here, and the alternative -- reporting no sigma -- is the one
        thing `acquisition.rank` refuses a candidate for.
        """
        absent = [p for p in SERVED_PROPERTIES if p not in self.ensembles]
        if absent:
            raise ValueError(
                f"{DERIVED_PROPERTY} is the judgement over {list(SERVED_PROPERTIES)} and this "
                f"bundle holds {self.properties}, missing {absent}. A sigmoid over fewer "
                "quantities is a perfectly good number and a different one; giving it this "
                "name is how a metric changes meaning between versions"
            )
        weights = self.RISK_WEIGHTS or {p: 1.0 / len(self.ensembles) for p in self.ensembles}
        total = sum(weights.get(p, 0.0) for p in self.ensembles) or 1.0
        z_mean = 0.0
        z_var = 0.0
        for prop in self.ensembles:
            centre, spread = self.scale[prop]
            w = weights.get(prop, 0.0) / total
            z_mean += w * (float(means[prop]) - centre) / spread
            z_var += (w * float(sigmas[prop]) / spread) ** 2
        value = 1.0 / (1.0 + math.exp(-z_mean))
        return value, value * (1.0 - value) * math.sqrt(z_var)

    def save(self, version_dir: str) -> str:
        import torch  # noqa: PLC0415

        path = os.path.join(version_dir, STATE_NAME)
        torch.save({
            # Keyed by property, so a reader can see which ones a version holds without
            # loading torch modules, and so a missing one is missing by name.
            "ensembles": {
                prop: {"hidden": e.hidden, "n_members": e.n_members,
                       "n_features": e.n_features,
                       "members": [m.state_dict() for m in e.members]}
                for prop, e in self.ensembles.items()
            },
            "scale": {k: list(v) for k, v in self.scale.items()},
            "featuriser": self.featuriser,
        }, path)
        return STATE_NAME


def run_round(job: Dict[str, Any], *, fit: Optional[Fitter] = None) -> str:
    """Evaluate held out, fit on everything, write the weights, publish unpromoted.

    **The evaluated model and the published model are two different fits, on purpose.** The
    metrics come from a scaffold-cluster holdout and from leave-one-group-out, because a
    number read off the data a model was fitted on is not a measurement. The weights come
    from a final fit on the whole corpus, because throwing away a fifth of twenty labels in
    the model that actually gets served would be paying for honesty twice.

    **The weights land before the manifest.** `ModelRegistry.promoted` is what a stream
    reads, and a version is only visible to `versions()` once its manifest exists -- so
    writing the weights first means no reader can ever see a published version that has none.
    """
    started = time.process_time()
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    fitter = fit or torch_ensemble
    properties = [str(p) for p in job.get("properties") or []]
    if not properties:
        raise ValueError(
            f"{job.get('model_id')}: the job names no properties. A round that fitted "
            "whatever happened to be in the corpus would publish a version nobody can say "
            "what it predicts"
        )

    by_property = {prop: [e for e in job["examples"]
                          if str(e.get("property_name") or "") == prop]
                   for prop in properties}
    missing = [p for p, rows in by_property.items() if not rows]
    if missing:
        raise ValueError(
            f"{job.get('model_id')}: the job names {properties} and carries no example for "
            f"{missing}. The driver's `validate` checks this against the corpus; a job that "
            "reaches here without them was built by something else"
        )

    # **One report per property, and the manifest carries the worst of each.** A version is one
    # promotion decision over three regressions, so promoting on the best of them would serve
    # two keys on a third's strength -- and `preorg_score`'s clients read each key as its own
    # quantity. The per-property numbers are in `metrics.json` and in the notes; the gate reads
    # the conservative ones.
    reports = {prop: evaluate(rows, job, fitter) for prop, rows in by_property.items()}
    report = _worst_of(reports, properties)

    weights = "none"
    fitted: Dict[str, Any] = {}
    scale: Dict[str, Any] = {}
    for prop, rows in by_property.items():
        if len(rows) < 2:
            continue
        fitted[prop] = fitter(rows, job)
        ys = [float(e["y"]) for e in rows]
        centre = sum(ys) / len(ys)
        spread = math.sqrt(sum((y - centre) ** 2 for y in ys) / len(ys))
        scale[prop] = (centre, spread)
    if len(fitted) == len(properties) and all(hasattr(m, "save") for m in fitted.values()):
        bundle = Bundle(fitted, scale, featuriser=_featuriser_name(job))
        os.makedirs(_version_dir(job), exist_ok=True)
        weights = bundle.save(_version_dir(job))
    elif fitted:
        # Partial is not a version. Two properties served and a third absent would fail
        # `metrics_reported` -- an integrity failure by the toolkit's own rule -- while the
        # manifest looked complete, so the round publishes metrics and no weights instead.
        report["reason"] = (
            (report.get("reason") or "")
            + f" Fitted {sorted(fitted)} of {properties}; a partial bundle is not published, "
            "because `metrics_reported` requires all four keys and a version missing one "
            "would be an integrity failure wearing a complete manifest."
        ).strip()
        report["validated"] = False
    report["weights"] = weights
    report["per_property"] = reports
    elapsed = time.process_time() - started

    version = _publish(job, report, elapsed)
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump({**report, "version": version,
                   "train_seconds_cpu": elapsed}, fh, indent=2)
    # Last, and after the manifest is on disk. The manager polls for this.
    with open(os.path.join(output_dir, TRAIN_COMPLETE_MARKER), "w") as fh:
        fh.write(str(version))
    return output_dir


def _worst_of(reports: Dict[str, Dict[str, Any]], order: Sequence[str]) -> Dict[str, Any]:
    """One report from several: the **most conservative** value of each field.

    Worst and not mean, for the reason the loop above gives: a mean would let a strong
    regression carry a weak one past the gate, and the three keys are read separately by
    clients who have no way to know which one the number came from. So the NLL is the highest,
    the calibration the furthest from 1, `logo_r2` the lowest, the counts the smallest, and
    `validated` is an AND -- a version is validated when every property it serves is.
    """
    out: Dict[str, Any] = dict(reports[order[0]])
    for prop in order[1:]:
        other = reports[prop]
        for key, pick in (("nll", max), ("logo_r2", min),
                          ("holdout_n", min), ("n_groups", min), ("n_holdout_groups", min)):
            a, b = out.get(key), other.get(key)
            out[key] = b if a is None else (a if b is None else pick(a, b))
        # Calibration is a mean squared z-score, so 1.0 is right and either side is wrong:
        # the worst is the one furthest from it, which neither max nor min expresses.
        a, b = out.get("calibration"), other.get("calibration")
        if a is None or (b is not None and abs(b - 1.0) > abs(a - 1.0)):
            out["calibration"] = b
        out["validated"] = bool(out.get("validated")) and bool(other.get("validated"))
    reasons = [f"{p}: {reports[p]['reason']}" for p in order if reports[p].get("reason")]
    out["reason"] = " | ".join(reasons)
    return out


def _featuriser_name(job: Dict[str, Any]) -> str:
    from asteer.features import FEATURISER_FOR_MODEL  # noqa: PLC0415

    return str(FEATURISER_FOR_MODEL.get(str(job.get("model_id") or "")) or "")


def _registry(job: Dict[str, Any]):
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    return ModelRegistry(job["registry_root"])


def _next_version(job: Dict[str, Any]) -> int:
    existing = _registry(job).versions(job["model_id"])
    return (max(existing) + 1) if existing else 0


def _version_dir(job: Dict[str, Any]) -> str:
    return str(_registry(job).version_dir(job["model_id"], _next_version(job)))


def _publish(job: Dict[str, Any], report: Dict[str, Any], elapsed: float) -> int:
    """Write `v{N}` through `asteer.registry`, **unpromoted**.

    Only the controller flips the promoted pointer (CON-6). A trainer that promoted its own
    version would be the only reader of its own evidence.
    """
    import datetime  # noqa: PLC0415

    from asteer.contracts import ModelManifest  # noqa: PLC0415
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    registry = ModelRegistry(job["registry_root"])
    version = _next_version(job)
    manifest = ModelManifest(
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
        # The HOLDOUT group count, not the corpus one. J2 measured why: an honest clustering
        # of this data is 2,487 singletons out of 2,508 groups, so a corpus-wide count clears
        # the gate's bar of four for free while the grouped fold may average over one.
        n_holdout_groups=report["n_holdout_groups"],
        featuriser=_featuriser_for(job),
        # **Every property this round fitted**, comma-joined. The gate splits it and requires
        # each one to be a property this model may serve (asteer J12): a version that fitted
        # two of its own and one of somebody else's is not promotable on the strength of the
        # two, because `preorg_score`'s clients read each key as its own quantity.
        trained_on=",".join(str(p) for p in job["properties"]),
        min_holdout_n=int(job["min_holdout_n"]),
        min_effect_size=float(job["min_effect_size"]),
        # CPU wall clock, as what it is. Never converted into a GPU-hour (backlog item 48).
        train_seconds_cpu=elapsed,
        promoted=False,
        notes=" ".join(filter(None, [
            str(job.get("notes") or ""),
            f"properties={'+'.join(str(p) for p in job['properties'])}",
            # Both counts, because they answer different questions and J2 showed how far
            # apart they can be: how many clusters the corpus has, and how many the grouped
            # fold actually held out.
            f"groups={report['n_groups']}",
            f"holdout_groups={report['n_holdout_groups']}",
            # What a stream will find in this version's directory. "none" is the honest
            # answer for a round that produced no model, and is what item 52 was about.
            f"weights={report.get('weights', 'none')}",
            report["reason"],
        ])).strip(),
    )
    registry.publish(manifest)
    return version


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="path to the job JSON")
    args = parser.parse_args(argv)
    with open(args.job) as fh:
        job = json.load(fh)
    run_round(job)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
