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

__all__ = ["run_round", "main", "scaffold_split", "evaluate", "Ensemble", "load_state",
           "preorg_predictor", "STATE_NAME", "TRAIN_COMPLETE_MARKER"]

#: Where a published version keeps its weights. `ModelRegistry.version_dir` reserves the
#: directory and deliberately prescribes no format -- *"a trainer writes its weights in here
#: in whatever shape it likes"* -- so the name is the trainer's to declare and the stream's
#: loader to read.
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


def load_state(version_dir: str, manifest: Any = None) -> Ensemble:
    """The `build` a stream hands `surrogate_stream`: a version directory in, a predictor out.

    Raises when the directory holds no weights, which `streams.load_promoted` turns into an
    `Unavailable` naming the directory. A loader that fell back to an untrained ensemble would
    serve a confident-looking sigma from no evidence at all.
    """
    import torch  # noqa: PLC0415

    path = os.path.join(version_dir, STATE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {STATE_NAME} in {version_dir}: this version was published with a manifest "
            "and no weights, so there is nothing to serve"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    job = {"n_members": state["n_members"], "hidden": state["hidden"],
           # Written by `save` since J8. A state file from before then carries neither, and
           # `preorg_predictor` refuses it by name rather than guessing -- an ensemble that
           # cannot say what it fitted is one whose numbers cannot be served under any key.
           "property_name": state.get("property_name", ""),
           "featuriser": state.get("featuriser", "")}
    members = _nets(int(state["n_features"]), job)
    for net, weights in zip(members, state["members"]):
        net.load_state_dict(weights)
        net.eval()
    return Ensemble(members, job, int(state["n_features"]))


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

    # The property this version regresses on, READ FROM THE WEIGHTS. One per round by design
    # -- the three B predicts are three regressions sharing a corpus, not one multi-output
    # head -- so which one it was is a fact about the fit, not about the driver's import.
    prop = ensemble.property_name
    if not prop:
        raise ValueError(
            f"{version_dir} holds an ensemble that does not name the property it fitted. It "
            "was written before J8, and serving it would mean choosing a key for it here, "
            "which is the defect J8 removed. Refit the round."
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
        (mean,), (sigma,) = ensemble([embed(features, featuriser=featuriser)])
        return {prop: mean}, {prop: sigma}

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
    examples = list(job["examples"])
    fitter = fit or torch_ensemble
    report = evaluate(examples, job, fitter)

    weights = "none"
    if len(examples) >= 2:
        final = fitter(examples, job)
        saver = getattr(final, "save", None)
        if saver is not None:
            os.makedirs(_version_dir(job), exist_ok=True)
            weights = saver(_version_dir(job))
    report["weights"] = weights
    elapsed = time.process_time() - started

    version = _publish(job, report, elapsed)
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump({**report, "version": version,
                   "train_seconds_cpu": elapsed}, fh, indent=2)
    # Last, and after the manifest is on disk. The manager polls for this.
    with open(os.path.join(output_dir, TRAIN_COMPLETE_MARKER), "w") as fh:
        fh.write(str(version))
    return output_dir


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
        trained_on=str(job["property_name"]),
        min_holdout_n=int(job["min_holdout_n"]),
        min_effect_size=float(job["min_effect_size"]),
        # CPU wall clock, as what it is. Never converted into a GPU-hour (backlog item 48).
        train_seconds_cpu=elapsed,
        promoted=False,
        notes=" ".join(filter(None, [
            str(job.get("notes") or ""),
            f"property={job['property_name']}",
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
