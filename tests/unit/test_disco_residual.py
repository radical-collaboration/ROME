"""The project's first `residual.json`, and the two refusals that keep it out of Model C.

**No torch here, for the reason `test_preorg_trainer.py` gives**: everything that decides what
the artifact *means* -- its four J11 fields, the feature list the declared rule re-derives, the
grouped fold table, and the two name checks that make the rehearsal unusable as an affinity
model -- is exercised with a stand-in fitter on a machine with no scientific stack. The GP is
one call and `test_affdelta_trainer.py` already covers it.

**Hermetic by committed evidence.** The raw benchmark (8,200 designs over 41 M-CSA targets) lives
in a sibling checkout that no CI will have, so the 410 cells it reduces to are committed beside
the artifact and these tests read those. The one test that re-derives the cells from the raw CSVs
skips when the sibling is absent, which is the only thing committed data cannot prove.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from examples.impress.small_molecule_binding.affdelta_train_wrapper import (  # noqa: E402
    Residual,
    evaluate,
)
from examples.impress.small_molecule_binding.fit_disco_residual import (  # noqa: E402
    DROPPED_BY_NAME,
    FEATURISER,
    PRIOR_FAMILY,
    TARGET,
    build_corpus,
    choose_features,
    examples_for,
    load_corpus,
    logo_table,
)

ARTIFACTS = (Path(__file__).resolve().parents[2]
             / "examples/impress/small_molecule_binding/artifacts/disco_rehearsal")
BENCHMARK = (Path(__file__).resolve().parents[3]
             / "discontinuous_scaffolds/rfd3-islands-validation/benchmark_data")
ISLANDS = (Path(__file__).resolve().parents[3]
           / "disco-surrogates/tests/data/specs/island_counts.csv")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(str(ARTIFACTS / "corpus.csv"))


@pytest.fixture(scope="module")
def residual():
    return Residual.load(str(ARTIFACTS / "residual.json"))


@pytest.fixture(scope="module")
def report():
    with open(ARTIFACTS / "report.json") as fh:
        return json.load(fh)


def _knn(k: int = 5):
    """A deterministic stand-in posterior: the mean of the k nearest training residuals.

    A real predictor rather than a constant, so `r2` is a number worth comparing, and pure
    stdlib so the fold machinery is testable with no scientific stack. Its sigma is the spread
    of those k, which is enough for `evaluate` to compute an nll and a calibration from.
    """
    def fit(train, job):
        rows = [(e["z"], e["residual"]) for e in train]

        def posterior(queries):
            means, sigmas = [], []
            for q in queries:
                near = sorted(rows, key=lambda r: sum((a - b) ** 2
                                                      for a, b in zip(r[0], q)))[:k]
                values = [r[1] for r in near]
                bar = sum(values) / len(values)
                var = sum((v - bar) ** 2 for v in values) / max(len(values) - 1, 1)
                means.append(bar)
                sigmas.append(max(math.sqrt(var), 1e-3))
            return means, sigmas
        return posterior
    return fit


# -- the artifact ------------------------------------------------------------


def test_the_artifact_carries_J11s_four_fields_and_round_trips(residual, tmp_path):
    """**The first `residual.json` in this project's history**, and J11's fields are what make
    it checkable: what its columns are, what produced them, what it predicts, in what units.

    `Residual.__init__` raises on a missing one rather than defaulting, so loading it at all is
    most of this test; the round trip is what proves the file is the whole state.
    """
    assert residual.target == TARGET == "log_motif_rmsd"
    assert residual.units == "ln(angstrom)"
    assert residual.featuriser == FEATURISER
    assert len(residual.features) == len(residual.centre) == len(residual.scale)
    assert len(residual.components) == len(residual.coef)
    assert "not an affinity model" in residual.notes, (
        "the notes are the only field a human reads before serving it")

    again = tmp_path / "residual.json"
    with open(ARTIFACTS / "residual.json") as src:
        payload = json.load(src)
    again.write_text(json.dumps(payload), encoding="utf-8")
    twin = Residual.load(str(again))
    assert twin.features == residual.features
    assert twin.components == residual.components
    assert twin.coef == residual.coef and twin.intercept == residual.intercept
    probe = [float(i) for i in range(len(residual.features))]
    assert twin.predict(twin.project(probe)) == residual.predict(residual.project(probe))


def test_the_feature_list_is_the_one_the_declared_rule_rederives(corpus, residual):
    """The list is **derived, not typed**, which is the whole answer to the trap below.

    J14 named `aggregate_score, iptm, ptm`. Measured, that is one column with a constant and an
    exact duplicate beside it, so a typed list would have put one column in twice -- and the
    artifact names its columns in order, which is what `project` checks.
    """
    features, verdicts = choose_features(corpus)
    assert features == residual.features
    assert len(features) == len({*features}), "a column would be named twice"
    kept = [v["column"] for v in verdicts if v["verdict"] == "kept"]
    assert set(features) <= set(kept)
    assert features == tuple(c for c in corpus[0] if c in features), "not in corpus order"


def test_the_degeneracies_that_forced_the_rule_are_still_in_the_data(corpus):
    """Re-measured rather than trusted, so none of the three can come back silently.

    If the benchmark is ever refreshed and `iptm` starts varying, or `aggregate_score` stops
    being `ptm` rescaled, this test fails and the feature rule gets looked at -- which is the
    difference between a measurement and a comment.
    """
    assert {row["iptm"] for row in corpus} == {0.0}, "iptm was 0.000000 on all 8,200 rows"
    # A cell is the mean of 20 float32 values written out as decimal, so the ratio spreads by
    # 7.1e-9 here. The **bit-exact** form of the same claim is asserted on the raw rows, in the
    # test below that has them -- this is the hermetic shadow of it.
    ratios = [row["aggregate_score"] / row["ptm"] for row in corpus]
    assert max(ratios) - min(ratios) < 1e-7 and abs(_mean(ratios) - 0.2) < 1e-7, (
        "aggregate_score was 0.2 x ptm: one column twice")
    assert "aggregate_score" in DROPPED_BY_NAME
    missing = sum(1 for row in corpus if row["n_clashing.ligand_min_distance"] == "")
    assert missing == 10, "the two ligand columns were incomplete, which is why they are out"


def test_the_prior_is_fitted_on_a_family_that_shares_no_target_with_the_folds(corpus, report):
    """Decision 1, asserted: a prior fitted on a fold's own target would leak into the number
    it is measured against, and *read and never written* is the one property C's residual has.
    """
    prior = {row["model_name"] for row in corpus if row["family"] == PRIOR_FAMILY}
    held = {row["model_name"] for row in corpus if row["family"] != PRIOR_FAMILY}
    assert not (prior & held)
    assert len(prior) == 8 and len(held) == 33
    assert report["corpus"]["prior_targets"] == 8
    assert report["corpus"]["held_targets"] == 33
    assert report["logo"]["n_groups"] == 33


# -- the two refusals --------------------------------------------------------


def test_the_rehearsal_is_refused_as_an_affinity_model(residual):
    """Refusal one: by target. `log_motif_rmsd` is not a free energy and cannot be served as one.

    This is the mechanism J14 rested its whole safety argument on -- *"the gate would refuse to
    promote it under C's property name even if someone tried"* -- so it is asserted rather than
    repeated.
    """
    with pytest.raises(ValueError, match="log_motif_rmsd"):
        residual.require_target("dg_corrected", where="the rehearsal")
    residual.require_target(TARGET)


def test_a_real_model_c_vector_is_refused_against_this_basis_by_name(residual):
    """Refusal two: by featuriser, which is why `disco-fold-1` is deliberately unregistered.

    A real Model C round embeds with `chain-metrics-1`. Ten of its columns against eight of
    these would be refused by width anyway -- the point is that it is refused as *the wrong
    featuriser*, naming the cause, which is the ordering item 74 bought.
    """
    from asteer.features import featuriser_for_model

    real = featuriser_for_model("aff_delta")
    with pytest.raises(ValueError, match=re.escape(FEATURISER)):
        residual.project([0.0] * real.width, featuriser=real.name, columns=real.columns)

    permuted = (residual.features[1], residual.features[0], *residual.features[2:])
    with pytest.raises(ValueError, match="position 0"):
        residual.project([0.0] * len(residual.features), featuriser=FEATURISER,
                         columns=permuted)


def test_the_rehearsal_never_publishes_a_version(residual):
    """Decision 2: the module calls `evaluate` and not `run_round`.

    `run_round` publishes into a `ModelRegistry` and copies the residual beside the weights. A
    version whose `trained_on` is `log_motif_rmsd` has no business in Model C's registry even
    once, and *"it was only a rehearsal"* is not a property a registry can read. Asserted
    against the source because the alternative is asserting it in a comment.
    """
    import ast

    from examples.impress.small_molecule_binding import fit_disco_residual

    tree = ast.parse(Path(fit_disco_residual.__file__).read_text(encoding="utf-8"))
    # Every identifier the module *executes*. Not a text search: the docstrings say
    # "`run_round` is not called" in so many words, and a grep would read that as the call.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            called.add(node.id)
        elif isinstance(node, ast.Attribute):
            called.add(node.attr)
        elif isinstance(node, ast.alias):
            called.add(node.asname or node.name.rsplit(".", 1)[-1])

    assert "run_round" not in called
    assert "ModelRegistry" not in called
    assert "evaluate" in called, "the rehearsal is pointless if it evaluates nothing"


# -- the table ---------------------------------------------------------------


def test_the_logo_table_covers_every_genuine_group_exactly_once(corpus, residual, report):
    """The pass criterion: a LOGO table over every genuine group, re-derived here.

    33 folds and 330 cells, each target held out once with its island count beside it. The
    stand-in fitter makes the *numbers* this test's own; the structure is the artifact's, and
    the committed report is checked against it fold for fold.
    """
    held = [row for row in corpus if row["family"] != PRIOR_FAMILY]
    examples = examples_for(held, residual, residual.features)
    islands = {row["model_name"]: row["island_count"] for row in held}

    table = logo_table(examples, {"seed": 0}, _knn(), islands)

    assert table["n_groups"] == 33
    assert table["n"] == 330 == sum(fold["n"] for fold in table["folds"])
    assert [fold["group"] for fold in table["folds"]] == sorted(islands)
    assert all(fold["n"] == 10 for fold in table["folds"]), "10 rfd3 backbones per target"
    assert all(fold["island_count"] == islands[fold["group"]] for fold in table["folds"])
    assert sum(row["n"] for row in table["by_island_count"]) == 330
    assert [row["island_count"] for row in table["by_island_count"]] \
        == sorted({islands[g] for g in islands})

    committed = report["logo"]
    assert [f["group"] for f in committed["folds"]] == [f["group"] for f in table["folds"]]
    assert [f["island_count"] for f in committed["folds"]] \
        == [f["island_count"] for f in table["folds"]]
    assert [f["n"] for f in committed["folds"]] == [f["n"] for f in table["folds"]]
    # The truth side is the artifact's, not the fitter's: these are the residuals of the
    # committed prior, so they must match to the last bit whatever fits them.
    for mine, theirs in zip(table["folds"], committed["folds"]):
        assert mine["truth_mean"] == pytest.approx(theirs["truth_mean"], abs=1e-9)


def test_the_fold_table_and_evaluates_own_logo_agree_on_the_same_split(corpus, residual):
    """Item 50's check, applied to this module: one split rule, cross-checked not re-written.

    `evaluate` returns the pooled `logo_r2` and throws away the per-fold predictions and the
    held-out sigma, which is why `logo_table` exists at all. Two implementations of a split
    over two record types is the defect item 50 closed, so the two are asserted equal here and
    `run()` raises if they ever drift at fit time.
    """
    held = [row for row in corpus if row["family"] != PRIOR_FAMILY]
    examples = examples_for(held, residual, residual.features)
    islands = {row["model_name"]: row["island_count"] for row in held}
    job = {"min_labels": 8, "seed": 0}

    table = logo_table(examples, job, _knn(), islands)
    metrics = evaluate(list(examples), job, residual, _knn())

    assert metrics["n_holdout_groups"] == table["n_groups"] == 33
    assert metrics["logo_r2"] == pytest.approx(table["r2"], abs=1e-9)
    assert metrics["holdout_n"] == 330, "loo_r2 is over the same cells, one at a time"
    assert metrics["loo_r2"] != metrics["logo_r2"], (
        "J7: holding out a whole target is a different and harder question than holding out "
        "one of its cells, and the two numbers being equal is the defect item 73 was")


def test_the_committed_folds_add_up_to_the_committed_pooled_numbers(report):
    """Every pooled number re-derived from the vectors beside it, so the report is internally
    checkable and not a transcript of a run nobody can re-add.

    And the stratification is re-derivable here too, which is the reason the vectors are
    committed rather than their summaries -- the first reading taken off the summaries was
    **wrong**. The two single-island targets carry errors three to five times every other
    fold's, so dropping them looks like it should rescue the pooled `r2`; it does not, because
    they inflate the spread to be explained faster than they inflate the error. `r2` goes from
    -0.111 to -0.350. The model has no skill on a held-out target in either stratification, and
    a summary-only report would have supported the comfortable reading of that.
    """
    folds = report["logo"]["folds"]
    truth = [t for fold in folds for t in fold["truth"]]
    pred = [p for fold in folds for p in fold["pred"]]
    sigma = [s for fold in folds for s in fold["sigma"]]
    assert len(truth) == len(pred) == len(sigma) == 330

    bar = _mean(truth)
    sse = sum((t - p) ** 2 for t, p in zip(truth, pred))
    sst = sum((t - bar) ** 2 for t in truth)
    assert 1.0 - sse / sst == pytest.approx(report["logo"]["r2"], abs=1e-9)
    assert math.sqrt(sse / len(truth)) == pytest.approx(report["logo"]["rmse"], abs=1e-9)
    assert _mean([((t - p) / s) ** 2 for t, p, s in zip(truth, pred, sigma)]) \
        == pytest.approx(report["logo"]["calibration"], abs=1e-9)

    single = [fold for fold in folds if fold["island_count"] == 1]
    rest = [fold for fold in folds if fold["island_count"] != 1]
    assert len(single) == 2, "the benchmark holds exactly two contiguous-motif targets"
    assert _mean([f["rmse"] for f in single]) > 3 * _mean([f["rmse"] for f in rest]), (
        "the error is concentrated in the one class the prior's family never contained")

    rest_truth = [t for fold in rest for t in fold["truth"]]
    rest_pred = [p for fold in rest for p in fold["pred"]]
    rest_bar = _mean(rest_truth)
    rest_r2 = 1.0 - (sum((t - p) ** 2 for t, p in zip(rest_truth, rest_pred))
                     / sum((t - rest_bar) ** 2 for t in rest_truth))
    assert report["logo"]["r2"] < 0.0
    assert rest_r2 < report["logo"]["r2"], (
        "dropping the two outlier targets must not be reported as a rescue: they inflate the "
        "spread faster than the error, so r2 gets worse, and the plan doc says so")


def test_the_gate_refuses_the_rehearsal_before_it_reads_a_single_number(report):
    """Refusal three, which nobody had to write: `asteer.promotion` refuses an unregistered
    featuriser at `featuriser_unknown`, with INSUFFICIENT_EVIDENCE, before accuracy is reached.

    So leaving `disco-fold-1` out of the registry buys all three: `require_target` by target,
    `project` by featuriser, and the gate by identity ahead of merit. Re-derived from the
    committed table rather than read off the report, so the verdict is a computation and not a
    transcript.
    """
    from examples.impress.small_molecule_binding.fit_disco_residual import gate_verdict

    fresh = gate_verdict(report["logo"], report.get("evaluate"), report["logo"]["n"])

    assert fresh == report["gate"]
    assert fresh["as_fitted"]["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert fresh["as_fitted"]["rule"] == "featuriser_unknown"
    # The counterfactual manifest, which describes no fit: it asks only which rule these
    # numbers meet. Pinned to what was measured -- a change here means the model changed, and
    # the plan doc's result is the place that records by how much.
    assert fresh["on_the_numbers_alone"]["outcome"] == "HOLD"
    assert fresh["on_the_numbers_alone"]["rule"] == "logo_r2_not_positive"
    assert report["logo"]["r2"] <= 0.0


def test_the_report_records_what_the_prior_is_worth(report):
    """The result this episode exists to produce, asserted as a shape rather than a value.

    The numbers are in the plan doc; what a test can keep honest is that the report carries
    the comparison at all -- the prior's fit on its own family, its fit off it, and the
    `rmse` of simply predicting the mean, without which an `r2` is unreadable.
    """
    assert report["target"] == TARGET
    assert report["claims_about_affinity"] == "none"
    for key in ("r2_in_family", "r2_out_of_family", "rmse_out_of_family", "rmse_predict_mean"):
        assert isinstance(report["prior"][key], float)
    for key in ("r2", "rmse", "rmse_predict_mean", "calibration", "sigma_mean"):
        assert key in report["logo"]
    assert report["corpus"]["cells"] == 410
    assert report["corpus"]["folds_per_cell"] == 20


# -- the one thing committed data cannot prove -------------------------------


@pytest.mark.skipif(not BENCHMARK.exists() or not ISLANDS.exists(),
                    reason="the disco benchmark is a sibling checkout, not a ROME dependency")
def test_the_committed_cells_are_what_the_benchmark_reduces_to(corpus):
    """8,200 designs in, 410 cells out, re-derived from the raw CSVs when they are present.

    Committed data can prove the artifact is self-consistent; only this can prove it is the
    benchmark's. Skipped where the sibling checkout is absent, which is everywhere but a
    laptop that has both.
    """
    rebuilt = build_corpus(str(BENCHMARK), str(ISLANDS))

    assert len(rebuilt) == len(corpus) == 410
    for fresh, committed in zip(rebuilt, corpus):
        assert fresh["model_name"] == committed["model_name"]
        assert fresh["rfd3_model_idx"] == committed["rfd3_model_idx"]
        assert fresh["island_count"] == committed["island_count"]
        assert fresh["family"] == committed["family"]
        assert fresh["y"] == pytest.approx(committed["y"], abs=1e-9)
        assert fresh["noise"] == pytest.approx(committed["noise"], abs=1e-9)
    assert choose_features(rebuilt)[0] == choose_features(corpus)[0]

    # The duplicate, in its strongest form: `aggregate_score` is not merely correlated with
    # `ptm` at Pearson 1.000000, it **is** `float32(0.2) * ptm` on every one of the 8,200 raw
    # rows, bit for bit. That is the claim `DROPPED_BY_NAME` makes and this is where it can be
    # checked, because a cell mean cannot carry it.
    import csv
    import struct

    def f32(text):
        return struct.unpack("f", struct.pack("f", float(text)))[0]

    with open(BENCHMARK / "campaign_analysis.csv", newline="") as fh:
        raw = list(csv.DictReader(fh))
    assert len(raw) == 8200
    assert all(f32(row["aggregate_score"]) == f32(f32(0.2) * f32(row["ptm"])) for row in raw)
    assert {f32(row["iptm"]) for row in raw} == {0.0}


def _mean(values):
    return sum(values) / len(values)
