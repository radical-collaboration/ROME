"""Model C's recalibration round, with the GP replaced by a stand-in.

Same split as B's tests and for the same reason: the fitter is an argument, so everything that
decides *what gets reported* -- reading the residual, projecting into its basis, leave-one-out,
the sigma that item 14 ranks on, the manifest -- runs with no torch and no botorch. The real
botorch GP has one test of its own, skipped where it is not installed.

Skipped without `asteer`: a sibling checkout rather than a declared ROME dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from asteer.registry import ModelRegistry  # noqa: E402

from examples.impress.small_molecule_binding.affdelta import (  # noqa: E402
    AffDeltaConfig,
    AffDeltaTrainer,
)
from examples.impress.small_molecule_binding.affdelta_train_wrapper import (  # noqa: E402
    TRAIN_COMPLETE_MARKER,
    Residual,
    run_round,
)

#: Model C's columns, from the registry rather than written out here -- the same discipline
#: J9 applied to Model B's fixtures. The residual's basis and these fixtures' vectors are both
#: this wide, so a change to the featuriser moves them together instead of leaving the corpus
#: and the artifact to disagree in a way only `project` would notice.
def _columns():
    from asteer.features import featuriser_for_model

    return list(featuriser_for_model("aff_delta").columns)


COLUMNS = _columns()
WIDTH = len(COLUMNS)


def _basis(n: int) -> list:
    """`n` unit vectors over `WIDTH` dimensions: the first two columns, kept as they are."""
    return [[1.0 if i == k else 0.0 for i in range(WIDTH)] for k in range(n)]


#: The offline prior: Model C's embedding reduced to 2, with a linear residual on top. The
#: arithmetic is unchanged from before J11 -- the prior is `-8 + x0` and the basis keeps the
#: first two columns -- and what J11 added is the four fields that make the artifact checkable:
#: `features` (so a permutation is refused by name, item 74), `featuriser` (so a `standin-1`
#: vector is refused as the wrong featuriser rather than as a width), `target` (so a residual
#: fitted on one quantity is never served under another's name) and `units`.
RESIDUAL = {
    "version": 3,
    "components": _basis(2),
    "centre": [0.0] * WIDTH,
    "scale": [1.0] * WIDTH,
    "coef": [1.0, 0.0],
    "intercept": -8.0,
    "features": COLUMNS,
    "featuriser": "chain-metrics-1",
    "target": "dg_corrected",
    "units": "kcal/mol",
    "notes": "fitted offline on public affinity data",
}


def _residual_file(tmp_path) -> str:
    path = tmp_path / "residual_v3.json"
    path.write_text(json.dumps(RESIDUAL, indent=2))
    return str(path)


def _row(n: int, x0: float, *, group: str | None = None, sigma: float | None = 0.4,
         prop: str = "dg_corrected", bump: float = 0.0) -> dict:
    # `dg_corrected`, not `dg_abfe` (item 66, fixed by J6). A `LabelRecord` carries the
    # PROPERTY a measurement supersedes, and this trainer filters on `property_name`. These
    # fixtures wrote the metric name, which is exactly why the defect survived every test
    # here: the corpus and the filter were wrong together.
    metadata: dict = {"embedding": [x0, 0.5] + [0.0] * (WIDTH - 2)}
    if group:
        metadata["scaffold"] = group
    return {
        "uid": f"lab-{n:03d}", "label_id": f"lab-{n:03d}", "node_id": f"n{n:06d}",
        "property_name": prop,
        # The prior says -8 + x0. The truth drifts away from it with x0, so the residual is a
        # real function of the basis rather than a constant -- a constant residual makes every
        # leave-one-out prediction exact and a calibration number meaningless. The drift is
        # ~2 kcal/mol across the corpus, which is the scale a real ABFE residual has; a toy
        # residual of a few hundredths produces a posterior sigma BELOW impress-a's
        # `calibration_in_range` floor, and the gate is right to refuse it.
        # `bump` is a per-GROUP offset the basis cannot explain, used only by the
        # leave-one-group-out tests: without one, the residual is exactly linear in x0, the
        # stub fits it perfectly, and every fold scores 1.0 -- so a grouped fold and a
        # leave-one-out fold would agree by construction and item 73's test could not fail.
        "value": -8.0 + x0 - 0.3 - 0.5 * x0 + bump,
        "sigma": sigma, "metadata": metadata,
    }


def _corpus(n: int = 12) -> list:
    return [_row(i + 1, 0.4 * i, group=f"sc{i % 4}") for i in range(n)]


def _stub_gp(train, job):
    """Least squares on the first component plus a residual-spread sigma. Deterministic."""
    xs = [e["z"][0] for e in train]
    ys = [e["residual"] for e in train]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    denom = sum((x - mx) ** 2 for x in xs) or 1e-12
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    spread = max((sum((y - (slope * x + intercept)) ** 2
                      for x, y in zip(xs, ys)) / k) ** 0.5, 0.25)

    def posterior(rows):
        return [slope * r[0] + intercept for r in rows], [spread] * len(rows)
    return posterior


def _job(tmp_path, corpus, **over) -> dict:
    config = AffDeltaConfig(registry_root=str(tmp_path / "models"),
                            residual_path=_residual_file(tmp_path), **over)
    trainer = AffDeltaTrainer(config)
    out = str(tmp_path / "round0")
    os.makedirs(out, exist_ok=True)
    job, _ = trainer._build_job(corpus, out)
    return job


# -- recalibration is not a refit --------------------------------------------


def test_the_residual_is_read_and_left_exactly_as_it_was(tmp_path):
    """Including its PCA basis. Re-fitting a projection on ten points moves the basis the
    prior was trained in, and the prior is then evaluated somewhere it never saw."""
    job = _job(tmp_path, _corpus())
    before = hashlib.sha256(Path(job["residual_path"]).read_bytes()).hexdigest()
    run_round(job, fit=_stub_gp)
    after = hashlib.sha256(Path(job["residual_path"]).read_bytes()).hexdigest()
    assert before == after, "the round rewrote the residual it was supposed to correct"


def test_the_manifest_names_the_residual_version_it_read(tmp_path):
    """So a campaign's `v0` is 'public residual v3, zero in-house labels' rather than an
    unlabelled starting point."""
    job = _job(tmp_path, _corpus())
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert "residual=v3" in manifest.notes
    assert "labels=12" in manifest.notes


def test_a_basis_that_does_not_match_the_embedding_is_refused(tmp_path):
    residual = Residual(RESIDUAL)
    with pytest.raises(ValueError, match="cannot reduce an embedding"):
        residual.project([1.0, 2.0])


def test_a_residual_whose_basis_and_prior_disagree_is_refused():
    broken = {**RESIDUAL, "coef": [1.0]}
    with pytest.raises(ValueError, match="basis and the prior disagree"):
        Residual(broken)


# -- insufficient evidence ---------------------------------------------------


def test_fewer_than_five_labels_is_insufficient_evidence_not_a_fitted_gp(tmp_path):
    job = _job(tmp_path, _corpus(4))
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert manifest.nll is None and manifest.calibration is None
    assert manifest.holdout_n == 0
    assert "below the declared floor" in manifest.notes


def test_a_round_at_ten_labels_does_fit(tmp_path):
    job = _job(tmp_path, _corpus(10))
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert manifest.holdout_n == 10, "leave-one-out gives every label a turn as evidence"
    assert manifest.nll is not None and manifest.calibration is not None


# -- sigma is the output, not a diagnostic -----------------------------------


def test_the_reported_sigma_passes_impress_as_calibration_gate(tmp_path):
    """Item 14 ranks on `dg_corrected + k·sigma_dg`, so a collapsed sigma turns a lower
    confidence bound into a posterior mean. impress-a's gate reads 0.1 <= sigma <= 5.0."""
    pytest.importorskip("impress_a", reason="impress-a is a sibling checkout")
    from impress_a.core.qc import GateOutcome
    from impress_a.tools import gates as impress_gates

    job = _job(tmp_path, _corpus())
    out = run_round(job, fit=_stub_gp)
    metrics = json.loads(Path(out, "metrics.json").read_text())

    result = impress_gates.get("calibration_in_range")(
        {"metrics": {"sigma_dg": metrics["sigma_dg_mean"]}}, {})
    assert result.outcome is GateOutcome.PASS, result.detail


def test_a_label_reporting_no_sigma_gets_the_declared_default_not_zero(tmp_path):
    """A GP told the observation noise is zero interpolates every label exactly and reports
    a sigma of ~0 between them."""
    corpus = [_row(i + 1, 0.4 * i, sigma=None) for i in range(6)]
    job = _job(tmp_path, corpus, default_sigma=0.7)
    assert all(e["noise"] == 0.7 for e in job["examples"])

    with pytest.raises(ValueError, match="default_sigma"):
        AffDeltaConfig(default_sigma=0.0)


# -- the dg_lcb seam ---------------------------------------------------------


def test_the_posterior_this_produces_ranks_the_way_aucam_ranks(tmp_path):
    """The seam item 14 reads. Built from C's own output rather than from invented numbers,
    because the thing worth testing is that the two sides agree about the sign."""
    aucam = Path("/home/mason/exdrive/rad/campaign_manager/campaigns/asd_campaign")
    if not (aucam / "acquisition.py").exists():
        pytest.skip("the campaign_manager checkout is not beside us")
    sys.path.insert(0, str(aucam))
    from acquisition import DEFAULT_LCB_K, Candidate, lower_confidence_bound, rank

    job = _job(tmp_path, _corpus())
    out = run_round(job, fit=_stub_gp)
    metrics = json.loads(Path(out, "metrics.json").read_text())
    sigma = metrics["sigma_dg_mean"]

    candidates = [Candidate(node_id=f"n{i}", dg_corrected=-9.0 + 0.4 * i,
                            sigma_dg=sigma, scaffold="sc0") for i in range(4)]
    ordered = rank(candidates)
    expected = sorted(candidates,
                      key=lambda c: lower_confidence_bound(c.dg_corrected, sigma,
                                                           DEFAULT_LCB_K))
    assert [c.node_id for c in ordered] == [c.node_id for c in expected]
    assert ordered[0].node_id == "n0", "the most negative dG ranks first; ΔG is minimised"


# -- cost --------------------------------------------------------------------


def test_a_round_takes_seconds_and_records_it_as_cpu_wall_clock(tmp_path):
    job = _job(tmp_path, _corpus(10))
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert manifest.train_seconds_cpu is not None
    assert manifest.train_seconds_cpu < 10.0, (
        "C retrains every cycle, so seconds per round is the property that makes that possible"
    )


def test_publishing_leaves_the_promoted_pointer_alone(tmp_path):
    job = _job(tmp_path, _corpus())
    registry = ModelRegistry(job["registry_root"])
    run_round(job, fit=_stub_gp)
    assert registry.promoted("aff_delta") is None
    assert registry.read_manifest("aff_delta", 0).promoted is False


def test_the_marker_is_written_last(tmp_path):
    job = _job(tmp_path, _corpus())
    out = run_round(job, fit=_stub_gp)
    marker = Path(out, TRAIN_COMPLETE_MARKER)
    assert marker.exists() and marker.read_text().strip() == "0"
    assert marker.stat().st_mtime >= Path(out, "metrics.json").stat().st_mtime


# -- the driver's refusals ---------------------------------------------------


def test_a_round_with_no_residual_is_refused_on_the_manager(tmp_path):
    trainer = AffDeltaTrainer(AffDeltaConfig(registry_root=str(tmp_path)))
    with pytest.raises(ValueError, match="residual_path"):
        trainer.validate(_corpus())


def test_a_missing_residual_file_is_refused(tmp_path):
    trainer = AffDeltaTrainer(
        AffDeltaConfig(registry_root=str(tmp_path), residual_path=str(tmp_path / "nope.json")))
    with pytest.raises(ValueError, match="does not exist"):
        trainer.validate(_corpus())


def test_a_corpus_with_no_abfe_labels_is_refused(tmp_path):
    trainer = AffDeltaTrainer(AffDeltaConfig(registry_root=str(tmp_path),
                                             residual_path=_residual_file(tmp_path)))
    with pytest.raises(ValueError, match="dg_corrected"):
        trainer.validate([_row(1, 0.0, prop="preorg_rmsd")])


def test_as_command_writes_a_job_naming_the_wrapper(tmp_path):
    config = AffDeltaConfig(registry_root=str(tmp_path / "models"),
                            residual_path=_residual_file(tmp_path))
    trainer = AffDeltaTrainer(config)
    out = str(tmp_path / "round0")
    os.makedirs(out)
    command, artifact = trainer.as_command(_corpus(), out)
    assert "affdelta_train_wrapper.py" in command and artifact == os.path.abspath(out)
    assert json.loads(Path(out, "job.json").read_text())["model_id"] == "aff_delta"


# -- CON-4 -------------------------------------------------------------------


def test_importing_the_driver_pulls_in_nothing_heavy():
    code = (
        "import sys; "
        "import examples.impress.small_molecule_binding.affdelta as a; "
        "print(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'torch', 'botorch', 'gpytorch', 'numpy', 'sklearn', 'scipy'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"the driver pulled in {out.stdout.strip()}"


# -- the one test that needs the real GP -------------------------------------


def test_the_real_botorch_gp_returns_a_posterior_with_a_usable_sigma(tmp_path):
    """Matérn-5/2 with known per-point noise, which is the specified architecture.

    Asserted as properties rather than against fixed numbers: a GP on twelve points is not a
    thing to pin values on, and the claim worth making is that the sigma is finite, positive
    and inside the band item 14's ranking needs.
    """
    pytest.importorskip("botorch", reason="botorch is the wrapper's dependency")
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import botorch_gp

    job = _job(tmp_path, _corpus())
    residual = Residual.load(job["residual_path"])
    train = [{**e, "z": residual.project(e["x"]),
              "residual": e["y"] - residual.predict(residual.project(e["x"]))}
             for e in job["examples"]]
    mean, sigma = botorch_gp(train, job)([train[0]["z"], [9.9, 0.5]])
    assert len(mean) == len(sigma) == 2
    assert all(s > 0.0 for s in sigma)
    assert sigma[1] > sigma[0], "a point far outside the labels must be less certain"


# -- the weights (item 52) ---------------------------------------------------


class _RecordingGP:
    """`_stub_gp`, with the size of every training set it was handed recorded."""

    def __init__(self):
        self.sizes: list[int] = []

    def __call__(self, train, job):
        self.sizes.append(len(train))
        return _stub_gp(train, job)


def test_the_published_gp_saw_every_label_the_evaluated_ones_did_not(tmp_path):
    """Leave-one-out measures; the final fit on all the labels is what gets served. A
    version that had seen eleven of twelve measurements would be throwing away one that
    cost an allocation.

    **Three kinds of fit, in order, since J7 made `logo_r2` a real grouped fold**: n
    leave-one-out refits for `loo_r2`, nll and calibration; then one refit per held-out
    group; then the single all-data fit that gets written. The middle block is what the old
    code did not do -- it published the leave-one-*out* residuals under the grouped name.
    """
    fit = _RecordingGP()
    corpus = _corpus()
    run_round(_job(tmp_path, corpus), fit=fit)

    n = len(corpus)
    assert fit.sizes[-1] == n, "the published fit saw every label"
    assert fit.sizes[:n] == [n - 1] * n, "leave-one-out, n times"

    grouped = fit.sizes[n:-1]
    groups = {row["metadata"]["scaffold"] for row in corpus
              if row["metadata"].get("scaffold")}
    assert len(grouped) == len(groups), (
        "one refit per held-out group -- the fold the old code never ran"
    )
    for size in grouped:
        assert 0 < size < n, "a grouped fold trains on strictly fewer than all the labels"


def test_a_round_below_the_floor_publishes_no_weights(tmp_path):
    """No GP was fitted, so there is nothing to write, and the manifest says so rather than
    leaving a stream to find an empty directory."""
    job = _job(tmp_path, _corpus(3))
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert "weights=none" in manifest.notes
    assert manifest.has_heldout_metrics is False


def test_loading_a_version_that_has_no_weights_refuses(tmp_path):
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import load_state

    job = _job(tmp_path, _corpus(3))
    run_round(job, fit=_stub_gp)
    with pytest.raises(FileNotFoundError, match="nothing to serve"):
        load_state(os.path.join(job["registry_root"], "aff_delta", "v0"))


def test_the_real_gp_is_written_with_the_residual_it_corrected(tmp_path):
    """The residual is **copied** in, not referenced. A path recorded in a manifest can be
    moved or refitted, and then 'v0 corrects public residual R' stops being checkable."""
    pytest.importorskip("botorch", reason="botorch is the wrapper's dependency")
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (
        RESIDUAL_NAME,
        load_state,
    )

    job = _job(tmp_path, _corpus())
    run_round(job)
    version_dir = os.path.join(job["registry_root"], "aff_delta", "v0")
    assert sorted(os.listdir(version_dir)) == ["gp.pt", "manifest.json", "residual.json"]
    assert json.loads(Path(version_dir, RESIDUAL_NAME).read_text())["version"] == 3

    loaded = load_state(version_dir)
    mean, sigma = loaded([[1.0, 0.5]])
    assert sigma[0] > 0.0
    fresh = botorch_gp_on(job, _corpus())
    assert abs(mean[0] - fresh[0][0]) < 1e-6, "a load is not a refit; it reproduces this fit"


def botorch_gp_on(job, corpus):
    """The same fit again, to show the saved one is the one that comes back."""
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (
        Residual,
        botorch_gp,
    )

    residual = Residual.load(job["residual_path"])
    rows = []
    for example in job["examples"]:
        z = residual.project(example["x"])
        rows.append({**example, "z": z, "residual": example["y"] - residual.predict(z)})
    return botorch_gp(rows, job)([[1.0, 0.5]])


# -- item 73: `logo_r2` was a leave-one-OUT R-squared under a grouped name ---------------


def _grouped_corpus(groups: int = 3, per_group: int = 4) -> list:
    """`groups` clusters of `per_group` labels, with a real within-cluster offset.

    The offset matters: if every cluster sat on the same line, holding one out would change
    nothing and the grouped fold would agree with leave-one-out by accident -- which is the
    one way this test could pass while the defect was still there.
    """
    rows = []
    bumps = [0.0, 1.6, -1.6, 2.4, -2.4, 0.8]
    n = 0
    for g in range(groups):
        for i in range(per_group):
            n += 1
            rows.append(_row(n, 0.4 * i + 3.0 * g, group=f"sc{g}",
                             bump=bumps[g % len(bumps)]))
    return rows


def test_the_grouped_score_is_not_the_leave_one_out_score(tmp_path):
    """**The regression nobody wrote.** Three groups of four.

    The old guard was `len(groups) >= 3 and len(groups) == len({e.get("group") ...})` --
    both sides are *sets*, so it is true whenever every record carries a group, and it then
    published the leave-one-out residuals under the key `logo_r2`. Nothing held a group out.

    With three clusters of four that are genuinely offset from one another, the two numbers
    are different, and this asserts that they are reported as two numbers.
    """
    job = _job(tmp_path, _grouped_corpus())
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)

    assert manifest.logo_r2 is not None, "three groups is enough for a grouped fold"
    assert manifest.n_holdout_groups == 3
    assert "loo_r2=" in manifest.notes, "the leave-one-out number is still reported"

    loo = float(manifest.notes.split("loo_r2=")[1].split()[0])
    assert loo != pytest.approx(manifest.logo_r2), (
        "the grouped score equals the leave-one-out score, which is the defect: holding out "
        "a whole cluster is strictly harder than holding out one of its members"
    )
    assert manifest.logo_r2 < loo, (
        "holding out a whole offset cluster should score worse than holding out one point "
        "with its three neighbours left in the training set"
    )


def test_too_few_groups_publishes_no_grouped_score_and_says_why(tmp_path):
    """`None` with a reason, which the promotion gate reads as INSUFFICIENT_EVIDENCE.

    Two clusters cannot support a leave-one-group-out, and the old code would have published
    a number for them -- because its guard asked whether the groups were distinct, not
    whether there were enough of them to hold one out.
    """
    job = _job(tmp_path, _grouped_corpus(groups=2, per_group=6))
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)

    assert manifest.logo_r2 is None
    assert manifest.n_holdout_groups == 2
    assert "2 scaffold clusters" in manifest.notes
    assert "not invented" in manifest.notes
    assert "loo_r2=" in manifest.notes, "the honest number is still reported"


def test_an_ungrouped_record_is_excluded_rather_than_pooled(tmp_path):
    """A group nobody assigned is not a group, which is `preorg.py::_group_of`'s rule.

    Pooling the ungrouped into one nameless cluster would invent a held-out group out of
    whatever records happened to lack a label.
    """
    corpus = _grouped_corpus() + [_row(99, 9.9), _row(100, 10.3)]
    job = _job(tmp_path, corpus)
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)
    assert manifest.n_holdout_groups == 3, "the two ungrouped rows are not a fourth group"


def test_the_manifest_carries_what_the_gate_reads(tmp_path):
    """asteer's promotion gate reads the featuriser, the target and the HOLDOUT group count.

    All three were prose or absent before J4/J5/J7: the featuriser was a module constant
    stamped on every vector, the target was not recorded at all, and the group count was a
    sentence in `notes`.
    """
    job = _job(tmp_path, _grouped_corpus())
    run_round(job, fit=_stub_gp)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("aff_delta", 0)

    assert manifest.trained_on == "dg_corrected"
    assert manifest.featuriser, "the version must name the featuriser its vectors came from"
    assert manifest.n_holdout_groups is not None

    from asteer.promotion import decide

    decision = decide(manifest)
    assert decision.evidence["trained_on"] == "dg_corrected"
    assert decision.rule != "target_unnamed"
    assert decision.rule != "featuriser_unnamed"
    assert decision.rule != "holdout_groups_unknown"


# -- J11: a residual that knows its own columns, and a C that can serve ------


def _residual_variant(tmp_path, **over):
    path = tmp_path / "residual_variant.json"
    path.write_text(json.dumps({**RESIDUAL, **over}, indent=2))
    return str(path)


def test_a_residual_missing_any_of_the_four_new_fields_is_refused_by_name(tmp_path):
    """Required, not optional, for the reason `preorg_predictor` gives about an ensemble that
    does not name its featuriser: a check that can be skipped is a check that will be.

    Nothing is lost by requiring them -- no real residual exists yet, J14 fits the first.
    """
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import Residual

    for field in ("features", "featuriser", "target", "units"):
        data = {k: v for k, v in RESIDUAL.items() if k != field}
        with pytest.raises(ValueError, match=field):
            Residual(data)
        assert "J11" in _message(Residual, data), "the refusal must say how to fix it"


def _message(cls, data):
    try:
        cls(data)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected a refusal")


def test_a_permuted_vector_is_refused_by_the_name_of_the_column_that_moved(tmp_path):
    """**Item 74.** Ten columns give ten factorial ways to be wrong and one to be right, and
    `project` used to check only a width -- so a correctly-sized vector in the wrong order
    projected without complaint and predicted nonsense."""
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import Residual

    residual = Residual(RESIDUAL)
    swapped = [COLUMNS[1], COLUMNS[0], *COLUMNS[2:]]
    with pytest.raises(ValueError) as caught:
        residual.project([0.0] * WIDTH, featuriser="chain-metrics-1", columns=swapped)
    message = str(caught.value)
    assert COLUMNS[0] in message and COLUMNS[1] in message
    assert "position 0" in message
    # And the right order still projects.
    assert residual.project([1.0, 0.5] + [0.0] * (WIDTH - 2),
                            featuriser="chain-metrics-1", columns=COLUMNS) == [1.0, 0.5]


def test_a_standin_vector_is_refused_as_the_wrong_featuriser_and_not_as_a_width():
    """The refusal names the **cause**, not the symptom.

    A width refusal would send a reader to look at a basis that is perfectly fine. The
    stand-in is what is wrong, and it is what produced Model B's logo_r2 of -0.19 on job
    24414064.
    """
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import Residual

    residual = Residual(RESIDUAL)
    with pytest.raises(ValueError) as caught:
        residual.project([0.37, 0.52], featuriser="standin-1")
    message = str(caught.value)
    assert "standin-1" in message and "chain-metrics-1" in message
    assert "dim embeddings" not in message, (
        "the width refusal must not fire first: it sends a reader to look at a basis that is "
        f"perfectly fine. Got: {message}"
    )


def test_a_residual_fitted_on_another_target_refuses_against_this_property(tmp_path):
    """A residual fitted on `interaction_energy` would hand a client a Rosetta score under
    the name of a free energy -- finite, plausible, and about something else."""
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import Residual

    residual = Residual({**RESIDUAL, "target": "interaction_energy"})
    with pytest.raises(ValueError, match="interaction_energy"):
        residual.require_target("dg_corrected")
    residual.require_target("interaction_energy")  # its own target is fine


def test_a_round_refuses_a_residual_whose_target_is_not_the_property_it_fits(tmp_path):
    """Checked once in `run_round`, before any example is projected."""
    job = _job(tmp_path, _corpus())
    job["residual_path"] = _residual_variant(tmp_path, target="interaction_energy")
    with pytest.raises(ValueError, match="fitted on target"):
        run_round(job)


def test_model_c_serves_a_mean_and_a_sigma_for_the_first_time(tmp_path):
    """**The first time Model C serves anything.**

    `affdelta_build` raised `AssertionError("unreachable")` under a docstring describing this
    composition. The number is the prior plus the correction, on the residual's own basis:

        dg_corrected = residual.predict(z) + posterior_mean(z),  z = residual.project(x)

    and the sigma is the posterior's alone, because the prior is fixed and contributes no
    uncertainty to add in quadrature.
    """
    pytest.importorskip("botorch", reason="botorch is the wrapper's dependency")
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (
        Residual,
        affdelta_predictor,
        load_state,
    )
    from asteer.registry import ModelRegistry

    job = _job(tmp_path, _corpus())
    run_round(job)
    registry = ModelRegistry(job["registry_root"])
    manifest = registry.read_manifest("aff_delta", 0)
    version_dir = registry.version_dir("aff_delta", 0)

    predict = affdelta_predictor(version_dir, manifest)
    x = [1.0, 0.5] + [0.0] * (WIDTH - 2)
    values, sigmas = predict({"embedding": list(x)})

    assert list(values) == ["dg_corrected"] and list(sigmas) == ["dg_corrected"]
    assert sigmas["dg_corrected"] > 0.0, "acquisition refuses a candidate with no sigma"

    # The composition, re-derived here rather than restated: prior plus correction.
    residual = Residual.load(os.path.join(version_dir, "residual.json"))
    z = residual.project(x, featuriser="chain-metrics-1", columns=COLUMNS)
    (correction,), (sigma,) = load_state(version_dir)([z])
    assert values["dg_corrected"] == pytest.approx(residual.predict(z) + correction)
    assert sigmas["dg_corrected"] == pytest.approx(sigma)
    # And it is NOT the bare posterior: the prior is -8 + x0, so the two differ by about that.
    assert abs(values["dg_corrected"] - correction) > 1.0


def test_the_envelope_refuses_a_version_whose_residual_was_not_copied_in(tmp_path):
    """Weights without the prior would serve a residual wearing the name of a free energy."""
    pytest.importorskip("botorch", reason="botorch is the wrapper's dependency")
    from examples.impress.small_molecule_binding.affdelta_train_wrapper import (
        RESIDUAL_NAME,
        affdelta_predictor,
    )
    from asteer.registry import ModelRegistry

    job = _job(tmp_path, _corpus())
    run_round(job)
    registry = ModelRegistry(job["registry_root"])
    version_dir = registry.version_dir("aff_delta", 0)
    os.remove(os.path.join(version_dir, RESIDUAL_NAME))
    with pytest.raises(ValueError, match="correction to a prior"):
        affdelta_predictor(version_dir, registry.read_manifest("aff_delta", 0))
