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

#: The offline prior: a 4-dim embedding reduced to 2, with a linear residual on top.
RESIDUAL = {
    "version": 3,
    "components": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    "centre": [0.0, 0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0, 1.0],
    "coef": [1.0, 0.0],
    "intercept": -8.0,
    "notes": "fitted offline on public affinity data",
}


def _residual_file(tmp_path) -> str:
    path = tmp_path / "residual_v3.json"
    path.write_text(json.dumps(RESIDUAL, indent=2))
    return str(path)


def _row(n: int, x0: float, *, group: str | None = None, sigma: float | None = 0.4,
         prop: str = "dg_abfe") -> dict:
    metadata: dict = {"embedding": [x0, 0.5, 0.0, 0.0]}
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
        "value": -8.0 + x0 - 0.3 - 0.5 * x0,
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
    with pytest.raises(ValueError, match="dg_abfe"):
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
