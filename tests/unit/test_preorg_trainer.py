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
import os
import subprocess
import sys

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
        metadata["embedding"] = [x0, 1.0]
    return {
        "uid": f"lab-{n:03d}", "label_id": f"lab-{n:03d}", "node_id": f"n{n:06d}",
        "property_name": prop, "value": 2.0 * x0, "sigma": 0.05, "metadata": metadata,
    }


def _corpus(n_groups: int = 5, per_group: int = 4) -> list:
    rows, n = [], 0
    for g in range(n_groups):
        for k in range(per_group):
            n += 1
            rows.append(_row(n, group=f"sc{g}", x0=float(g) + 0.25 * k))
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
    corpus = [_row(1, group="a", x0=1.0, featured=False)]
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
    mean, sigma = predict([[0.0, 1.0], [4.0, 1.0]])
    assert len(mean) == len(sigma) == 2
    assert all(s > 0.0 for s in sigma), "a zero sigma would make the NLL infinite"
