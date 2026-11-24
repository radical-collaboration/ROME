"""Model B's round, with the ensemble replaced by a stand-in.

**No torch here, deliberately.** `run_round` takes its fitter as an argument, so every line
that decides *what gets reported* -- the scaffold split, the held-out NLL, the calibration
number, the leave-one-group-out `r2`, the manifest and the completion marker -- is exercised
on a machine with no scientific stack. The torch ensemble is one function and is covered by
its own skipping test.

That split is the point rather than a convenience. A surrogate lies in its evaluation, not in
its forward pass.

Skipped without `asteer`: a sibling checkout rather than a declared ROME dependency.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from asteer.registry import ModelRegistry  # noqa: E402

from examples.impress.small_molecule_binding.preorg import (  # noqa: E402
    PocketPreorgConfig,
    PocketPreorgTrainer,
)
from examples.impress.small_molecule_binding.preorg_train_wrapper import (  # noqa: E402
    TRAIN_COMPLETE_MARKER,
    evaluate,
    run_round,
    scaffold_split,
)


# -- a corpus, and two stand-in fitters --------------------------------------


def _row(n: int, *, group: str, x0: float, featured: bool = True,
         prop: str = "preorg_rmsd") -> dict:
    """One corpus row in the shape `corpus.record_to_row` produces."""
    metadata = {"scaffold": group}
    if featured:
        # **The width comes from the registry**, not from a literal pair. asteer's J9 moved
        # `pocket_preorg` from `standin-1`'s two columns to `pocket-geom-1`'s nine, and a
        # corpus fitted at one width and served at another is the defect J8 closed on the
        # property name and this closes on the shape.
        from asteer.features import featuriser_for_model

        width = featuriser_for_model("pocket_preorg").width
        metadata["embedding"] = [x0] + [1.0] * (width - 1)
    return {
        "uid": f"lab-{n:03d}", "label_id": f"lab-{n:03d}", "node_id": f"n{n:06d}",
        "property_name": prop, "value": 2.0 * x0, "sigma": 0.05, "metadata": metadata,
    }


#: The example directory, for a subprocess that must import the wrapper by name.
RUNNER_DIR = Path(__file__).resolve().parents[2] / "examples" / "impress" / "small_molecule_binding"


def _seeded(x0: float = 1.0) -> list:
    """One serving vector of Model B's declared width."""
    from asteer.features import featuriser_for_model

    return [x0] + [1.0] * (featuriser_for_model("pocket_preorg").width - 1)


def _corpus(n_groups: int = 5, per_group: int = 4, prop: str = "preorg_rmsd") -> list:
    rows, n = [], 0
    for g in range(n_groups):
        for k in range(per_group):
            n += 1
            rows.append(_row(n, group=f"sc{g}", x0=float(g) + 0.25 * k, prop=prop))
    return rows


def _corpus_all(n_groups: int = 5, per_group: int = 4) -> list:
    """One corpus carrying all three of Model B's labels for every design.

    The same `x` per design under three property names, which is what a real `md_label` run
    produces: one MD measurement yields `preorg_rmsd`, `rotamer_entropy` and `ligand_rmsf`
    from the same trajectory.
    """
    rows = []
    for i, prop in enumerate(("preorg_rmsd", "rotamer_entropy", "ligand_rmsf")):
        for row in _corpus(n_groups, per_group, prop=prop):
            # Distinct ids per property, and a scale that differs between them, so a risk
            # computed from one property's scale against another's is visibly wrong.
            rows.append({**row, "uid": f"{row['uid']}-{prop}",
                         "label_id": f"{row['label_id']}-{prop}",
                         "value": row["value"] * (i + 1)})
    return rows


def _linear_fit(train, job):
    """Least squares on the first feature, in pure python. Recovers `y = 2*x0` exactly."""
    xs = [e["x"][0] for e in train]
    ys = [e["y"] for e in train]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1e-12
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    residual = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    spread = max((sum(r * r for r in residual) / n) ** 0.5, 1e-3)

    def predict(rows):
        mean = [slope * row[0] + intercept for row in rows]
        return mean, [spread] * len(mean)
    return predict


def _upside_down_fit(train, job):
    """Predicts the corpus mean reflected through itself: a model worse than the mean."""
    ys = [e["y"] for e in train]
    centre = sum(ys) / len(ys)

    def predict(rows):
        return [centre * 3.0] * len(rows), [0.5] * len(rows)
    return predict


def _job(tmp_path, corpus, **over) -> dict:
    """A round over ONE property unless a test names more.

    The config's own default is all three (J12), which is what a campaign wants; most tests
    here are about the mechanics of a round -- the split, the manifest, the marker -- and one
    regression is enough to see them. `_corpus_all` and the four-metric tests below cover the
    multi-property shape.
    """
    over.setdefault("properties", ("preorg_rmsd",))
    config = PocketPreorgConfig(registry_root=str(tmp_path / "models"), **over)
    trainer = PocketPreorgTrainer(config)
    out = str(tmp_path / "round0")
    os.makedirs(out, exist_ok=True)
    job, _ = trainer._build_job(corpus, out)
    return job


# -- the split ---------------------------------------------------------------


def test_the_boundary_never_splits_a_cluster():
    """A random split leaves near-duplicates of held-out designs in the training set, so the
    model scores well by memorising a neighbour -- and nothing in the numbers shows it."""
    examples = [{"group": g, "x": [0.0], "y": 0.0} for g in list("aaabbcc")]
    train, holdout = scaffold_split(examples, 0.3)
    train_groups = {e["group"] for e in train}
    holdout_groups = {e["group"] for e in holdout}
    assert train_groups and holdout_groups
    assert not (train_groups & holdout_groups), "a cluster appeared on both sides"


def test_an_ungrouped_corpus_refuses_to_split_rather_than_splitting_randomly():
    examples = [{"group": "a", "x": [0.0], "y": 0.0}, {"group": None, "x": [0.0], "y": 0.0}]
    assert scaffold_split(examples, 0.5) == ([], [])


def test_the_split_is_asteer_s_rule_and_not_a_second_copy_of_it():
    """Item 50, closed on this side. One rule, two repos, and its failure is silent.

    The same rule was implemented here over corpus rows and in AuCaM's `acquisition.py` over
    `Candidate` objects. What this asserts is not that the arithmetic is *correct* -- asteer's
    own tests do that, including against both copies this replaced -- but that this function
    is a thin wrapper over it, so the two cannot drift.
    """
    from asteer.grouping import holdout_indices

    examples = [{"group": g, "x": [0.0], "y": 0.0} for g in list("aaaabbbccd")]
    train, holdout = scaffold_split(examples, 0.3)
    train_at, holdout_at = holdout_indices([e["group"] for e in examples], 0.3)

    assert [examples[i] for i in train_at] == train
    assert [examples[i] for i in holdout_at] == holdout


def test_the_metadata_keys_this_driver_reads_are_the_ones_asteer_writes():
    """Two strings in two distributions, kept equal by a test -- `TRAIN_COMPLETE_MARKER`'s
    shape, and for the same reason: the driver imports nothing heavy at module scope, so it
    mirrors the constant instead of importing it. A mismatch would make every round report
    insufficient evidence with nothing wrong anywhere that a reader could see."""
    from asteer.features import EMBEDDING_KEY
    from asteer.grouping import SCAFFOLD_KEY

    from examples.impress.small_molecule_binding import preorg

    assert preorg.GROUP_KEY == SCAFFOLD_KEY
    assert preorg.FEATURE_KEY == EMBEDDING_KEY


def test_a_round_over_grouped_labels_returns_a_verdict_at_last(tmp_path):
    """**The first B round in this project's history that is not `INSUFFICIENT_EVIDENCE`.**

    Nothing assigned `metadata["scaffold"]` until J20 (item 71), so `_group_of` returned
    `None` for every row, `scaffold_split` returned `([], [])`, and the gate read *"holdout
    groups unknown"*. Model B could not be validated, whatever the labels cost.

    Six clusters of eight designs: the holdout is one whole cluster of 8, which clears the
    gate's `min_holdout_n` of 8, and the leave-one-group-out averages over six groups, which
    clears `min_holdout_groups` of 4. The verdict may be `HOLD` -- the stand-in fitter is
    exact, so its sigma is a floor and the calibration number says so -- and `HOLD` is a
    verdict: *"we looked and it is not promotable"* is what CON-13 exists to distinguish from
    *"we could not tell"*.
    """
    from asteer.promotion import PromotionOutcome, decide

    job = _job(tmp_path, _corpus(n_groups=6, per_group=8))
    run_round(job, fit=_linear_fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)

    assert manifest.holdout_n >= 8
    assert manifest.n_holdout_groups >= 4
    assert manifest.logo_r2 is not None, "a grouped score, from groups that exist"

    decision = decide(manifest)
    assert decision.outcome is not PromotionOutcome.INSUFFICIENT_EVIDENCE, (
        f"still no verdict: {decision.rule} -- {decision.reason}"
    )
    assert decision.outcome in (PromotionOutcome.PROMOTE, PromotionOutcome.HOLD)


# -- what reaches the manifest -----------------------------------------------


def test_held_out_metrics_reach_the_manifest(tmp_path):
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)

    registry = ModelRegistry(job["registry_root"])
    manifest = registry.read_manifest("pocket_preorg", 0)
    assert manifest.nll is not None
    assert manifest.calibration is not None
    assert manifest.logo_r2 is not None
    assert manifest.holdout_n >= job["min_holdout_n"]
    assert manifest.logo_r2 > 0.5, "a fitter that recovers y=2*x0 should beat the mean"


def test_a_negative_logo_r2_is_reported_rather_than_omitted(tmp_path):
    """Below zero means predicting the campaign's mean beats the model. That is the number
    that catches a self-confirming objective, so omitting it loses the only honest signal."""
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_upside_down_fit)

    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert manifest.logo_r2 is not None, "a bad model must still report its number"
    assert manifest.logo_r2 < 0.0


def test_the_group_count_rides_beside_the_value(tmp_path):
    """CON-13: 20-40 samples with a 20% holdout is judged on 4-8 points, and `logo_r2` is
    near-meaningless below about four groups. A gate seeing only the number reads noise."""
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert "groups=5" in manifest.notes


# -- insufficient evidence ---------------------------------------------------


def test_too_few_held_out_points_is_insufficient_evidence_not_a_pass(tmp_path):
    job = _job(tmp_path, _corpus(), min_holdout_n=50)
    run_round(job, fit=_linear_fit)

    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert manifest.nll is None and manifest.calibration is None and manifest.logo_r2 is None, (
        "the gate reads a None held-out metric as INSUFFICIENT_EVIDENCE; a number here would "
        "be read as a pass"
    )
    assert manifest.min_holdout_n == 50
    assert "below the declared floor" in manifest.notes


def test_a_corpus_with_no_scaffold_clusters_reports_why(tmp_path):
    """`LabelRecord` has no scaffold field -- the cluster is a property of the design -- so a
    corpus can arrive with nothing to group by. That is a result, not a crash."""
    corpus = _corpus()
    for row in corpus:
        row["metadata"].pop("scaffold")
    job = _job(tmp_path, corpus)
    report = evaluate(job["examples"], job, _linear_fit)

    assert report["nll"] is None and report["logo_r2"] is None
    assert "random split" in report["reason"]


def test_fewer_than_three_groups_reports_nll_but_invents_no_logo_r2(tmp_path):
    job = _job(tmp_path, _corpus(n_groups=2, per_group=6))
    report = evaluate(job["examples"], job, _linear_fit)
    assert report["nll"] is not None, "a held-out split happened, so NLL is real"
    assert report["logo_r2"] is None
    assert "at least" in report["reason"]


# -- publish is not promote --------------------------------------------------


def test_publishing_a_version_leaves_the_promoted_pointer_alone(tmp_path):
    """CON-6: only the controller promotes. A trainer that promoted its own version would be
    the only reader of its own evidence."""
    job = _job(tmp_path, _corpus())
    registry = ModelRegistry(job["registry_root"])
    assert registry.promoted("pocket_preorg") is None

    run_round(job, fit=_linear_fit)

    assert registry.promoted("pocket_preorg") is None
    assert registry.read_manifest("pocket_preorg", 0).promoted is False


def test_a_second_round_publishes_the_next_version(tmp_path):
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    run_round(job, fit=_linear_fit)
    assert ModelRegistry(job["registry_root"]).versions("pocket_preorg") == [0, 1]


# -- cost, as what it is -----------------------------------------------------


def test_the_round_records_cpu_wall_clock_and_calls_it_that(tmp_path):
    """Item 48. A real measurement of a real thing, never converted into a GPU-hour."""
    job = _job(tmp_path, _corpus())
    out = run_round(job, fit=_linear_fit)

    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert manifest.train_seconds_cpu is not None and manifest.train_seconds_cpu >= 0.0
    assert manifest.train_seconds_cpu < 60.0, (
        "the claim this supports is that the round is seconds rather than hours, so a "
        "one-GPU-hour budget is not what will bind"
    )
    metrics = json.loads(open(os.path.join(out, "metrics.json")).read())
    assert "train_seconds_cpu" in metrics
    assert "gpu" not in json.dumps(metrics).lower(), (
        "no GPU figure may appear here: the conversion factor is the thing item 48 refuses"
    )


# -- the completion contract -------------------------------------------------


def test_the_marker_is_written_last_and_names_the_version(tmp_path):
    """On Dragon a task can run to completion and never resolve its future, so the manager
    polls for this marker rather than for the checkpoint."""
    job = _job(tmp_path, _corpus())
    out = run_round(job, fit=_linear_fit)
    marker = os.path.join(out, TRAIN_COMPLETE_MARKER)
    assert os.path.exists(marker)
    assert open(marker).read().strip() == "0"
    assert os.path.getmtime(marker) >= os.path.getmtime(os.path.join(out, "metrics.json"))


# -- the driver's refusals ---------------------------------------------------


def test_a_corpus_with_no_target_property_is_refused_on_the_manager(tmp_path):
    """`validate` runs before a task is scheduled, so an unusable corpus fails on the manager
    instead of burning an allocation."""
    trainer = PocketPreorgTrainer(PocketPreorgConfig(registry_root=str(tmp_path)))
    corpus = [_row(1, group="a", x0=1.0, prop="dg_abfe")]
    with pytest.raises(ValueError, match="property_name"):
        trainer.validate(corpus)


def test_a_corpus_with_no_feature_vector_is_refused(tmp_path):
    """Training on an empty feature set produces a model that predicts the corpus mean and
    reports a plausible NLL for doing it."""
    trainer = PocketPreorgTrainer(PocketPreorgConfig(registry_root=str(tmp_path)))
    corpus = [_row(n, group="a", x0=1.0, featured=False, prop=prop)
              for n, prop in enumerate(PocketPreorgConfig().properties)]
    with pytest.raises(ValueError, match="feature set"):
        trainer.validate(corpus)


def test_a_one_member_ensemble_is_refused(tmp_path):
    """The spread across members IS the epistemic term; one member has no spread."""
    with pytest.raises(ValueError, match="n_members"):
        PocketPreorgConfig(n_members=1)


def test_as_command_writes_a_job_and_names_the_wrapper(tmp_path):
    config = PocketPreorgConfig(registry_root=str(tmp_path / "models"))
    trainer = PocketPreorgTrainer(config)
    out = str(tmp_path / "round0")
    os.makedirs(out)
    command, artifact = trainer.as_command(_corpus(), out)
    assert artifact == os.path.abspath(out)
    assert "preorg_train_wrapper.py" in command
    written = json.loads(open(os.path.join(out, "job.json")).read())
    assert written["model_id"] == "pocket_preorg"
    assert len(written["examples"]) == 20


# -- CON-4 -------------------------------------------------------------------


def test_importing_the_driver_pulls_in_no_torch():
    """The driver is constructed in the campaign process. A transitive torch import there
    keeps a CUDA context resident for the whole run."""
    code = (
        "import sys; "
        "import examples.impress.small_molecule_binding.preorg as p; "
        "print(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'torch', 'numpy', 'sklearn', 'scipy'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"the driver pulled in {out.stdout.strip()}"


# -- the one test that needs torch -------------------------------------------


def test_the_real_ensemble_reports_two_uncertainties(tmp_path):
    """Aleatoric from the head's clamped `log_var`, epistemic from the spread across members.

    Asserted as a property rather than against fixed numbers: a five-member ensemble on
    twenty points is not a thing to pin values on.
    """
    # `importorskip("torch")` is NOT enough in this suite: `tests/conftest.py` installs a stub
    # `torch` module carrying only `no_grad` when the real one is absent, so the bare name
    # imports and the skip never fires. Ask for a submodule the stub does not have.
    pytest.importorskip("torch.nn", reason="the real torch; conftest stubs the bare name")
    from examples.impress.small_molecule_binding.preorg_train_wrapper import torch_ensemble

    job = _job(tmp_path, _corpus(), epochs=40)
    predict = torch_ensemble(job["examples"], job)
    mean, sigma = predict([_seeded(0.0), _seeded(4.0)])
    assert len(mean) == len(sigma) == 2
    assert all(s > 0.0 for s in sigma), "a zero sigma would make the NLL infinite"


# -- the weights (item 52) ---------------------------------------------------


class _Recording:
    """`_linear_fit`, with the size of every training set it was handed recorded."""

    def __init__(self):
        self.sizes: list[int] = []

    def __call__(self, train, job):
        self.sizes.append(len(train))
        return _linear_fit(train, job)


def test_the_published_model_is_fitted_on_everything_the_evaluated_one_is_not(tmp_path):
    """Two fits, on purpose. The metrics come from a scaffold-cluster holdout and from
    leave-one-group-out, because a number read off the data a model was fitted on is not a
    measurement. The weights come from a fit on the whole corpus, because throwing away a
    fifth of twenty labels in the model that actually gets served pays for honesty twice."""
    fit = _Recording()
    corpus = _corpus()
    run_round(_job(tmp_path, corpus), fit=fit)
    assert fit.sizes, "the fitter ran"
    assert fit.sizes[-1] == len(corpus), (
        f"the last fit saw {fit.sizes[-1]} of {len(corpus)} examples; the published weights "
        "must be the all-data fit, not the holdout's"
    )
    assert min(fit.sizes[:-1]) < len(corpus), "the evaluated fits held something out"


def test_a_fitter_that_cannot_be_saved_publishes_no_weights_and_says_so(tmp_path):
    """A stub returns a closure, which holds weights nowhere. The manifest says `none`
    rather than leaving a stream to discover it."""
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert "weights=none" in manifest.notes
    assert os.listdir(os.path.join(job["registry_root"], "pocket_preorg", "v0")) == [
        "manifest.json"]


def test_loading_a_version_that_has_no_weights_refuses(tmp_path):
    from examples.impress.small_molecule_binding.preorg_train_wrapper import load_state

    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    version_dir = os.path.join(job["registry_root"], "pocket_preorg", "v0")
    with pytest.raises(FileNotFoundError, match="nothing to serve"):
        load_state(version_dir)


def test_an_unloadable_promoted_version_reaches_a_stream_as_an_outage(tmp_path):
    """The seam into EII.5: `streams.load_promoted` turns the refusal above into an
    `Unavailable` naming the directory, rather than a crashed stream or a plausible number."""
    from asteer.contracts import PromotionDecision, PromotionOutcome

    from examples.impress.small_molecule_binding.preorg_train_wrapper import load_state
    from examples.impress.small_molecule_binding.streams import Unavailable, load_promoted

    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    ModelRegistry(job["registry_root"]).promote(PromotionDecision(
        decision_id="d0", model_id="pocket_preorg", to_version=0,
        outcome=PromotionOutcome.PROMOTE, decided_at="2026-11-21T00:00:00Z"))
    served = load_promoted(None, None, model_id="pocket_preorg",
                           registry_root=job["registry_root"], build=load_state)
    assert isinstance(served, Unavailable) and "did not load" in served.reason


def test_the_real_ensemble_is_written_and_loads_back(tmp_path):
    pytest.importorskip("torch.nn", reason="the wrapper's own dependency")
    from examples.impress.small_molecule_binding.preorg_train_wrapper import load_state

    job = _job(tmp_path, _corpus(), epochs=40)
    run_round(job)
    version_dir = os.path.join(job["registry_root"], "pocket_preorg", "v0")
    assert sorted(os.listdir(version_dir)) == ["ensemble.pt", "manifest.json"]
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert "weights=ensemble.pt" in manifest.notes
    loaded = load_state(version_dir, manifest)
    means, sigmas = loaded([_seeded()])
    prop = job["properties"][0]
    assert abs(means[prop][0] - 2.0) < 1.0, "the corpus is y = 2*x0"
    assert sigmas[prop][0] > 0.0, "an ensemble that reports zero spread is reporting nothing"


# -- J8: a predictor reads its provenance instead of assuming it -----------------------


def _fitted_version(tmp_path, *, properties=None, **over):
    """One real round, written to v0, with the bundle on disk.

    **All three properties by default, because that is what a servable version is** (J12):
    `preorg_score` requires all four of its metrics, so a bundle holding fewer is fittable and
    not servable. The corpus carries every property the round names, because a round filters
    on them -- which is the whole of item 66 and the reason a mismatched fixture produces a
    version with a manifest and no weights rather than a visible failure.
    """
    pytest.importorskip("torch.nn", reason="the wrapper's own dependency")
    properties = tuple(properties or ("preorg_rmsd", "rotamer_entropy", "ligand_rmsf"))
    corpus = (_corpus_all() if len(properties) == 3
              else [r for p in properties for r in _corpus(prop=p)])
    job = _job(tmp_path, corpus, epochs=40, properties=properties, **over)
    run_round(job)
    return job, os.path.join(job["registry_root"], "pocket_preorg", "v0")


def test_the_ensemble_records_the_property_and_the_featuriser_it_was_fitted_with(tmp_path):
    from examples.impress.small_molecule_binding.preorg_train_wrapper import load_state

    from asteer.features import FEATURISER_FOR_MODEL

    job, version_dir = _fitted_version(tmp_path)
    loaded = load_state(version_dir)
    assert loaded.properties == list(job["properties"])
    assert loaded.featuriser == FEATURISER_FOR_MODEL["pocket_preorg"]


def test_the_served_keys_come_from_the_weights_and_not_from_a_module_constant(tmp_path):
    """**The defect J8 removed, now over three keys instead of one.**

    `preorg_predictor` read `preorg.DEFAULT_PROPERTY`, a module constant, so an ensemble
    fitted on `rotamer_entropy` would have served its numbers under the key `preorg_rmsd` --
    and a client reading a rotamer entropy as a pocket RMSD gets a finite, plausible,
    completely wrong number. Which properties a version holds is a fact about the fit.
    """
    from examples.impress.small_molecule_binding import preorg
    from examples.impress.small_molecule_binding.preorg_train_wrapper import preorg_predictor

    _job_, version_dir = _fitted_version(tmp_path)
    predict = preorg_predictor(version_dir)
    # A vector rather than a path: this test is about which *keys* the values are served
    # under, and since asteer's J9 Model B's featuriser reads the file a payload names.
    values, sigmas = predict({"embedding": _seeded()})

    assert sorted(values) == sorted(preorg.DEFAULT_PROPERTIES + (preorg.DERIVED_PROPERTY,))
    assert sorted(sigmas) == sorted(values)
    assert preorg.DEFAULT_PROPERTY == "preorg_rmsd", (
        "the module constant is unchanged; nothing reads it to choose a key any more"
    )


def test_a_version_holding_fewer_than_three_properties_is_refused_by_name(tmp_path):
    """Fittable and **not servable**, and the refusal names both halves of why.

    Serving three keys and omitting the fourth is an integrity failure by the toolkit's own
    rule, and `sigmoid(z)` over one quantity is a different number wearing the name of the
    judgement over three.
    """
    from examples.impress.small_molecule_binding.preorg_train_wrapper import preorg_predictor

    job, version_dir = _fitted_version(tmp_path, properties=("rotamer_entropy",))
    assert "weights=ensemble.pt" in ModelRegistry(
        job["registry_root"]).read_manifest("pocket_preorg", 0).notes, (
        "the round fitted what it named and published it; it is the serving that refuses"
    )
    with pytest.raises(ValueError) as caught:
        preorg_predictor(version_dir)
    message = str(caught.value)
    assert "preorg_rmsd" in message and "ligand_rmsf" in message
    assert "metrics_reported" in message


def test_a_version_fitted_with_another_featuriser_is_refused_by_name(tmp_path):
    """A predictor that embeds with one featuriser and was fitted with another serves a
    model inputs it has never seen, and nothing in the arithmetic complains."""
    import torch

    from examples.impress.small_molecule_binding.preorg_train_wrapper import (
        STATE_NAME, preorg_predictor,
    )

    _job_, version_dir = _fitted_version(tmp_path)
    path = os.path.join(version_dir, STATE_NAME)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["featuriser"] = "encoder-activations-9"
    torch.save(state, path)

    with pytest.raises(ValueError) as caught:
        preorg_predictor(version_dir)
    assert "encoder-activations-9" in str(caught.value)
    assert "never seen" in str(caught.value)


def test_a_version_that_does_not_name_its_featuriser_is_refused_rather_than_guessed_at(
    tmp_path,
):
    """J8's refusal, on the field that still lives at the top of the state file."""
    import torch

    from examples.impress.small_molecule_binding.preorg_train_wrapper import (
        STATE_NAME, preorg_predictor,
    )

    _job_, version_dir = _fitted_version(tmp_path)
    path = os.path.join(version_dir, STATE_NAME)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["featuriser"] = ""
    torch.save(state, path)

    with pytest.raises(ValueError, match="before J8"):
        preorg_predictor(version_dir)


def test_a_single_property_state_file_from_before_j12_is_refused_by_name(tmp_path):
    """It carries no corpus scale, so the risk cannot be z-scored against anything.

    Refused rather than adapted: the alternative is inventing the scale at serve time, which
    is how a risk acquires a unit nobody wrote down.
    """
    import torch

    from examples.impress.small_molecule_binding.preorg_train_wrapper import (
        STATE_NAME, load_state,
    )

    _job_, version_dir = _fitted_version(tmp_path)
    path = os.path.join(version_dir, STATE_NAME)
    state = torch.load(path, map_location="cpu", weights_only=False)
    # The pre-J12 shape: one ensemble at the top level, no `scale`.
    one = next(iter(state["ensembles"].values()))
    torch.save({**one, "property_name": "preorg_rmsd",
                "featuriser": state["featuriser"]}, path)

    with pytest.raises(ValueError) as caught:
        load_state(version_dir)
    assert "before J12" in str(caught.value)
    assert "corpus scale" in str(caught.value)


def test_the_manifest_carries_what_the_promotion_gate_reads(tmp_path):
    """All three were prose or absent: a module constant, an unrecorded target, and a
    corpus-wide group count in a sentence."""
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)

    assert manifest.trained_on == ",".join(job["properties"])
    assert manifest.featuriser
    assert manifest.n_holdout_groups is not None

    from asteer.promotion import decide

    for rule in ("featuriser_unnamed", "target_unnamed", "holdout_groups_unknown"):
        assert decide(manifest).rule != rule


def test_the_holdout_group_count_is_not_the_corpus_group_count(tmp_path):
    """asteer J5 rule 3 reads the holdout count, and J2 measured why.

    A group whose removal leaves fewer than two training rows is skipped by the fold, so the
    number of groups the grouped score actually averages over can be smaller than the number
    of clusters in the corpus. Reporting the corpus count as if it were the holdout count is
    how a bar of four gets cleared for free.
    """
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_linear_fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    notes = manifest.notes

    assert "groups=" in notes and "holdout_groups=" in notes, "both counts are reported"
    assert manifest.n_holdout_groups <= int(notes.split("groups=")[1].split()[0]), (
        "the fold cannot hold out more groups than the corpus has"
    )


# -- J12: all four metrics, and the risk derived in the stream ---------------


def test_the_derived_property_name_is_one_string_in_two_modules():
    """The driver imports nothing heavy (CON-4) and the wrapper owns torch, so neither can
    import the other. A test is what keeps one string one string -- the shape item 66 settled
    for `PROPERTY_FOR_METRIC`."""
    from examples.impress.small_molecule_binding import preorg
    from examples.impress.small_molecule_binding import preorg_train_wrapper as w

    assert preorg.DERIVED_PROPERTY == w.DERIVED_PROPERTY
    assert tuple(preorg.DEFAULT_PROPERTIES) == tuple(w.SERVED_PROPERTIES)
    assert preorg.DERIVED_PROPERTY not in preorg.DEFAULT_PROPERTIES, (
        "a version fitted on its own judgement would be fitted on nothing"
    )


def test_a_round_that_named_the_derived_property_is_refused():
    """Decision L2: there is no `induced_fit_risk` ground truth anywhere."""
    from examples.impress.small_molecule_binding.preorg import DERIVED_PROPERTY

    trainer = PocketPreorgTrainer(PocketPreorgConfig(properties=(DERIVED_PROPERTY,)))
    with pytest.raises(ValueError, match="DERIVED"):
        trainer.validate(_corpus())


def test_a_round_naming_a_property_the_corpus_lacks_is_refused_by_that_name():
    """Not by the first missing one: a round that fitted two of three and published would
    leave `metrics_reported` failing on the third with the manifest looking complete."""
    trainer = PocketPreorgTrainer(PocketPreorgConfig())
    with pytest.raises(ValueError) as caught:
        trainer.validate(_corpus(prop="preorg_rmsd"))
    assert "rotamer_entropy" in str(caught.value)
    assert "ligand_rmsf" in str(caught.value)


def test_one_round_fits_every_property_and_writes_one_artifact(tmp_path):
    """One artifact, not three: a version is one promotion decision."""
    job, version_dir = _fitted_version(tmp_path)
    assert sorted(os.listdir(version_dir)) == ["ensemble.pt", "manifest.json"]

    from examples.impress.small_molecule_binding.preorg_train_wrapper import load_state

    bundle = load_state(version_dir)
    assert bundle.properties == list(job["properties"])
    assert sorted(bundle.scale) == sorted(job["properties"])


def test_the_contract_is_satisfied_three_means_three_sigmas_and_a_bounded_risk(tmp_path):
    """`preorg_score/spec.yaml` puts all four metrics in `metrics_reported`."""
    from examples.impress.small_molecule_binding.preorg import (
        DEFAULT_PROPERTIES, DERIVED_PROPERTY,
    )
    from examples.impress.small_molecule_binding.preorg_train_wrapper import preorg_predictor

    _job_, version_dir = _fitted_version(tmp_path)
    values, sigmas = preorg_predictor(version_dir)({"embedding": _seeded()})

    for prop in DEFAULT_PROPERTIES:
        assert prop in values and prop in sigmas
        assert sigmas[prop] > 0.0
    assert 0.0 <= values[DERIVED_PROPERTY] <= 1.0, values[DERIVED_PROPERTY]
    assert sigmas[DERIVED_PROPERTY] >= 0.0


def test_the_risk_rises_with_every_prediction_and_is_half_at_the_corpus_mean(tmp_path):
    """The definition, checked rather than restated.

    All three quantities are minimised, so a prediction at the corpus mean is an average
    pocket -- z = 0 on each, mean z = 0, `sigmoid(0) = 0.5`. Above the mean on every one is a
    pocket that reorganises more than the corpus typically does, and the risk goes up.
    """
    from examples.impress.small_molecule_binding.preorg import DEFAULT_PROPERTIES
    from examples.impress.small_molecule_binding.preorg_train_wrapper import Bundle

    scale = {p: (10.0, 2.0) for p in DEFAULT_PROPERTIES}
    bundle = Bundle({p: object() for p in DEFAULT_PROPERTIES}, scale)

    at_mean = {p: 10.0 for p in DEFAULT_PROPERTIES}
    tiny = {p: 1e-9 for p in DEFAULT_PROPERTIES}
    assert bundle.risk(at_mean, tiny)[0] == pytest.approx(0.5)

    worse = {p: 14.0 for p in DEFAULT_PROPERTIES}      # z = +2 on each
    better = {p: 6.0 for p in DEFAULT_PROPERTIES}      # z = -2 on each
    assert bundle.risk(worse, tiny)[0] > 0.5
    assert bundle.risk(better, tiny)[0] < 0.5
    assert bundle.risk(worse, tiny)[0] == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0)), abs=1e-9), "sigmoid(mean z), weights (1,1,1)/3"


def test_the_risk_sigma_is_propagated_and_not_invented(tmp_path):
    """Zero spread gives zero risk sigma; more spread gives more, and acquisition refuses a
    candidate with none at all."""
    from examples.impress.small_molecule_binding.preorg import DEFAULT_PROPERTIES
    from examples.impress.small_molecule_binding.preorg_train_wrapper import Bundle

    bundle = Bundle({p: object() for p in DEFAULT_PROPERTIES},
                    {p: (10.0, 2.0) for p in DEFAULT_PROPERTIES})
    means = {p: 10.0 for p in DEFAULT_PROPERTIES}
    assert bundle.risk(means, {p: 0.0 for p in DEFAULT_PROPERTIES})[1] == 0.0
    small = bundle.risk(means, {p: 0.1 for p in DEFAULT_PROPERTIES})[1]
    large = bundle.risk(means, {p: 1.0 for p in DEFAULT_PROPERTIES})[1]
    assert 0.0 < small < large


def test_a_bundle_missing_a_property_refuses_to_name_its_sigmoid_the_risk():
    from examples.impress.small_molecule_binding.preorg_train_wrapper import Bundle

    bundle = Bundle({"preorg_rmsd": object()}, {"preorg_rmsd": (1.0, 1.0)})
    with pytest.raises(ValueError, match="judgement over"):
        bundle.risk({"preorg_rmsd": 1.0}, {"preorg_rmsd": 0.1})


def test_a_degenerate_corpus_gives_a_risk_near_a_half_rather_than_a_confident_one():
    """A corpus whose labels are all equal has no scale, and the floor is what keeps the
    risk a number instead of making every z infinite."""
    from examples.impress.small_molecule_binding.preorg import DEFAULT_PROPERTIES
    from examples.impress.small_molecule_binding.preorg_train_wrapper import Bundle

    bundle = Bundle({p: object() for p in DEFAULT_PROPERTIES},
                    {p: (5.0, 0.0) for p in DEFAULT_PROPERTIES})
    value, _ = bundle.risk({p: 5.0 for p in DEFAULT_PROPERTIES},
                           {p: 0.0 for p in DEFAULT_PROPERTIES})
    assert value == pytest.approx(0.5)


def test_the_scale_travels_with_the_weights_so_a_fresh_process_derives_the_same_risk(tmp_path):
    """The reason the risk is derived in the stream and not in the agent (decision L2).

    The agent has no corpus; the scale is a fact about the fit. A subprocess that loads the
    version and nothing else must reach the same number, or the risk depends on who asked.
    """
    import subprocess

    _job_, version_dir = _fitted_version(tmp_path)
    from examples.impress.small_molecule_binding.preorg_train_wrapper import preorg_predictor

    here = preorg_predictor(version_dir)({"embedding": _seeded()})[0]["induced_fit_risk"]

    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(RUNNER_DIR)!r})\n"
        "from preorg_train_wrapper import preorg_predictor\n"
        f"v, _ = preorg_predictor({version_dir!r})({{'embedding': {_seeded()!r}}})\n"
        "print(json.dumps(v['induced_fit_risk']))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip()) == pytest.approx(here)


def test_the_manifest_names_every_property_the_round_fitted(tmp_path):
    """And the gate requires every one of them to be a property this model may serve."""
    from asteer.promotion import decide

    job, _ = _fitted_version(tmp_path)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("pocket_preorg", 0)
    assert manifest.trained_on == "preorg_rmsd,rotamer_entropy,ligand_rmsf"
    assert decide(manifest).rule != "substitute_target"

    import dataclasses

    # The evidence rules run first and this fixture's holdout is small, so they are cleared
    # here to reach the target rule -- which is the one under test. The order is J5's and
    # deliberate: a stand-in or a substitute target is named before an accuracy number.
    intruder = dataclasses.replace(
        manifest, trained_on="preorg_rmsd,dg_corrected",
        holdout_n=8, n_holdout_groups=4, logo_r2=0.5, calibration=1.0, nll=1.0,
    )
    assert decide(intruder).rule == "substitute_target", (
        "two of its own and one of somebody else's is not promotable on the strength of two"
    )
