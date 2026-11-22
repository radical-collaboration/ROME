"""Model A's reward-head round, with `itgt`'s head replaced by a stand-in.

Same split as B's and C's tests and for the same reason: the fitter is an argument, so
everything that decides *what gets banked and what gets reported* -- the action/outcome
pairing, the replay buffer, the leave-one-group-out fold, the disabled-head round, the
manifest -- runs with no torch and no `itgt`.

**And the stand-in is not the evidence.** EII.3 measured the hazard directly: a stub fitter
agreed with a gate that the real one failed. So the real `itgt` head has a test of its own at
the bottom of this file, on a synthetic complex whose right answer is known by construction,
skipped where `itgt` is not beside us.

Skipped without `asteer`: a sibling checkout rather than a declared ROME dependency.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from asteer.registry import ModelRegistry  # noqa: E402

from examples.impress.small_molecule_binding.liganchor import (  # noqa: E402
    ANCHOR_OWN_METRICS,
    LigAnchorConfig,
    LigAnchorTrainer,
)
from examples.impress.small_molecule_binding.liganchor_train_wrapper import (  # noqa: E402
    TRAIN_COMPLETE_MARKER,
    ReplayBuffer,
    run_round,
)

ITGT = Path("/home/mason/exdrive/rad/interface-targeting")
IMPRESS_A = Path("/home/mason/exdrive/rad/IMPRESS/agent/src/impress_a/tools")


# -- fixtures ----------------------------------------------------------------


def _action(atoms, structure: str = "/tmp/lig.pdb", resname: str = "LIG") -> dict:
    return {"structure_path": structure, "resname": resname,
            "engaged_atoms": list(atoms)}


def _record(n: int, atoms, value: float, *, group: str = "sc0",
            prop: str = "dg_abfe", action: dict | None = None) -> dict:
    """One `LabelRecord` row as `corpus.record_to_row` hands it over."""
    metadata: dict = {"scaffold": group}
    if action is not False:
        metadata["action"] = action if action else _action(atoms)
    return {
        "uid": f"lab-{n:03d}", "label_id": f"lab-{n:03d}", "node_id": f"nd-{n:03d}",
        "property_name": prop, "value": value, "sigma": 0.4,
        "produced_at": "2026-11-21T00:00:00Z", "metadata": metadata,
    }


#: The ligand's atoms, and the three that are worth anchoring. The outcome is a function of
#: how many of those three an anchor set holds -- an action -> outcome relation, which is the
#: only kind model A is allowed to learn.
ATOMS = [f"C{i}" for i in range(1, 9)]
GOOD = {"C1", "C2", "C3"}


def _corpus(n: int = 16, groups: int = 2) -> list[dict]:
    rows = []
    for i in range(n):
        atoms = [ATOMS[i % 8], ATOMS[(i + 1) % 8], ATOMS[(i + 3) % 8]]
        hits = len(set(atoms) & GOOD)
        rows.append(_record(i, atoms, -7.0 - 0.8 * hits + 0.01 * i,
                            group=f"sc{i % groups}"))
    return rows


def _config(tmp_path, **over) -> LigAnchorConfig:
    defaults = dict(registry_root=str(tmp_path / "models"),
                    buffer_path=str(tmp_path / "buffer.jsonl"))
    defaults.update(over)
    return LigAnchorConfig(**defaults)


def _job(tmp_path, records=None, **over) -> dict:
    trainer = LigAnchorTrainer(_config(tmp_path, **over))
    job, _ = trainer._build_job(records if records is not None else _corpus(),
                                str(tmp_path / "round"))
    return job


class _StubHead:
    """Per-atom credit from the training pairs, averaged over an anchor set.

    Crude on purpose and genuinely a learner: it has to read the outcomes to rank, so a
    broken pairing shows up as a flat score rather than as an exception.
    """

    def __init__(self):
        self.calls: list[int] = []
        self.trained: list[set] = []

    def __call__(self, train, job):
        self.calls.append(len(train))
        self.trained.append({row["action_id"] for row in train})
        credit: dict = {}
        for row in train:
            signed = int(job["direction"]) * float(row["outcome_value"])
            for atom in row["action"]["engaged_atoms"]:
                credit.setdefault(atom, []).append(signed)

        def score(rows):
            out = []
            for row in rows:
                atoms = row["action"]["engaged_atoms"]
                values = [v for a in atoms for v in credit.get(a, [])]
                out.append(sum(values) / len(values) if values else 0.0)
            return out

        return score


# -- action, not artifact ----------------------------------------------------


@pytest.mark.parametrize("metric", sorted(ANCHOR_OWN_METRICS))
def test_a_target_read_off_the_anchor_set_is_refused(metric):
    with pytest.raises(ValueError, match="compliance with our previous instruction"):
        LigAnchorConfig(property_name=metric)


def test_a_target_that_is_a_key_of_the_action_is_refused_without_being_listed(tmp_path):
    """The general form of the same rule: whatever the action carries, it is not the target."""
    action = _action(["C1", "C2"])
    action["pocket_volume"] = 412.0
    rows = [_record(i, ["C1", "C2"], -8.0, prop="pocket_volume", action=action)
            for i in range(4)]
    trainer = LigAnchorTrainer(_config(tmp_path, property_name="pocket_volume"))
    with pytest.raises(ValueError, match="compliance with our previous instruction"):
        trainer.validate(rows)


def test_the_refused_targets_are_what_impress_as_anchor_tool_actually_emits():
    """The seam. `ANCHOR_OWN_METRICS` is a copy of a list that lives in another repo, so it
    is worth exactly as much as a test that it is still the same list."""
    source = IMPRESS_A / "asteer_design_agents.py"
    if not source.exists():
        pytest.skip("the IMPRESS checkout is not beside us")
    text = source.read_text()
    block = re.search(r'"anchor_spec": spec\.as_artifact\(\).*?"metrics": \{(.*?)\}',
                      text, re.S)
    assert block, "the anchor tool's metrics block moved; this seam needs re-anchoring"
    emitted = set(re.findall(r'"(\w+)":', block.group(1)))
    assert emitted == set(ANCHOR_OWN_METRICS), (
        f"impress-a's anchor tool emits {sorted(emitted)}; model A refuses "
        f"{sorted(ANCHOR_OWN_METRICS)} as regression targets. A metric computed from the "
        "anchor set and not on that list is a self-regression waiting to be configured."
    )


# -- the rest of the driver's refusals ---------------------------------------


def test_an_unsigned_direction_is_refused():
    with pytest.raises(ValueError, match="which way is better"):
        LigAnchorConfig(direction=0)


def test_a_min_pairs_floor_below_two_is_refused():
    with pytest.raises(ValueError, match="min_pairs"):
        LigAnchorConfig(min_pairs=1)


def test_the_scorer_names_the_campaign_and_is_never_itgts_proxy():
    """`itgt` refuses to compare a proxy-fit artifact against a campaign-scored one, which
    only works if a campaign round says which it is."""
    assert LigAnchorConfig().resolved_scorer() == "asd:dg_abfe"
    assert LigAnchorConfig().resolved_scorer() != "proxy"


def test_a_round_with_no_replay_buffer_is_refused_on_the_manager(tmp_path):
    trainer = LigAnchorTrainer(LigAnchorConfig(registry_root=str(tmp_path)))
    with pytest.raises(ValueError, match="buffer_path"):
        trainer.validate(_corpus())


def test_an_enabled_head_with_no_frozen_encoder_is_refused(tmp_path):
    trainer = LigAnchorTrainer(_config(tmp_path, head_enabled=True))
    with pytest.raises(ValueError, match="encoder_path"):
        trainer.validate(_corpus())


def test_an_encoder_path_that_does_not_exist_is_refused(tmp_path):
    trainer = LigAnchorTrainer(_config(tmp_path, head_enabled=True,
                                       encoder_path=str(tmp_path / "nope")))
    with pytest.raises(ValueError, match="does not exist"):
        trainer.validate(_corpus())


def test_an_outcome_with_no_action_trains_nothing_and_is_refused(tmp_path):
    rows = [_record(i, [], -8.0, action=False) for i in range(4)]
    trainer = LigAnchorTrainer(_config(tmp_path))
    with pytest.raises(ValueError, match="no action"):
        trainer.validate(rows)


# -- the job -----------------------------------------------------------------


def test_the_job_pairs_an_action_with_its_downstream_outcome(tmp_path):
    job = _job(tmp_path)
    assert len(job["pairs"]) == 16
    first = job["pairs"][0]
    assert first["action"]["engaged_atoms"] and first["outcome_value"] < 0
    assert first["action_id"] == first["node_id"], "one anchor set per node by default"
    assert first["outcome_label_id"] == "lab-000", "the pairing survives as a reference"
    assert {p["group"] for p in job["pairs"]} == {"sc0", "sc1"}


def test_as_command_writes_a_job_and_names_the_wrapper(tmp_path):
    out = tmp_path / "round"
    out.mkdir()
    trainer = LigAnchorTrainer(_config(tmp_path))
    command, artifact = trainer.as_command(_corpus(), str(out))
    assert "liganchor_train_wrapper.py" in command
    assert artifact == str(out)
    assert json.loads((out / "job.json").read_text())["pairs"]


# -- the buffer, which is the one thing that resumes -------------------------


def test_the_buffer_folds_by_action_id_so_an_outcome_supersedes_a_banked_action(tmp_path):
    buffer = ReplayBuffer(str(tmp_path / "buffer.jsonl"))
    buffer.append([{"action_id": "a1", "action": _action(["C1"]), "outcome_value": None}])
    buffer.append([{"action_id": "a1", "action": _action(["C1"]), "outcome_value": -8.4}])
    rows = buffer.load()
    assert len(rows) == 1 and rows[0]["outcome_value"] == -8.4


def test_a_torn_final_line_costs_one_pair_and_not_the_buffer(tmp_path):
    path = tmp_path / "buffer.jsonl"
    buffer = ReplayBuffer(str(path))
    buffer.append([{"action_id": f"a{i}", "action": _action(["C1"]), "outcome_value": -8.0}
                   for i in range(5)])
    with open(path, "a") as fh:
        fh.write('{"action_id": "a5", "action": {"struct')
    assert len(buffer.load()) == 5, "an interrupted append must not cost four earlier rounds"


def test_a_restarted_trainer_refits_from_the_whole_buffer(tmp_path):
    """R8's one genuine resume: round 2 fits on round 1's pairs as well as its own."""
    fit = _StubHead()
    first = _corpus(6)
    run_round(_job(tmp_path, first, head_enabled=False), fit=fit)
    assert fit.calls == [], "a banking round must not fit anything"

    second = [_record(100 + i, [ATOMS[i % 8], ATOMS[(i + 2) % 8]], -7.4 - 0.1 * i,
                      group=f"sc{i % 2}") for i in range(10)]
    out = run_round(_job(tmp_path, second, head_enabled=True), fit=fit)
    metrics = json.loads(Path(out, "metrics.json").read_text())
    assert metrics["buffer_size"] == 16
    assert metrics["banked_this_round"] == 10
    assert fit.calls, "round 2 fits"
    # Each leave-one-group-out fold trains on the complement, not on the whole buffer, so
    # the thing to check is that round 1's pairs are in those complements at all.
    banked_first = {p["action_id"] for p in _job(tmp_path, first)["pairs"]}
    seen = set().union(*fit.trained)
    assert banked_first <= seen, (
        f"round 2 never saw {sorted(banked_first - seen)}; a buffer that only held round 2 "
        "would make round 2 a round 1 with a longer history"
    )


def test_re_banking_the_same_round_does_not_double_the_buffer(tmp_path):
    fit = _StubHead()
    rows = _corpus(10)
    run_round(_job(tmp_path, rows, head_enabled=False), fit=fit)
    out = run_round(_job(tmp_path, rows, head_enabled=False), fit=fit)
    metrics = json.loads(Path(out, "metrics.json").read_text())
    assert metrics["buffer_size"] == 10 and metrics["banked_this_round"] == 0


# -- the round ---------------------------------------------------------------


def test_a_disabled_head_banks_and_publishes_nothing_steerable(tmp_path):
    """Round 1, source spec 7.3. No held-out metrics is what the promotion gate reads as
    INSUFFICIENT_EVIDENCE, so the banking round cannot become the promoted pointer."""
    fit = _StubHead()
    job = _job(tmp_path, head_enabled=False)
    out = run_round(job, fit=fit)
    assert fit.calls == []

    registry = ModelRegistry(job["registry_root"])
    manifest = registry.read_manifest("lig_anchor", 0)
    assert manifest.has_heldout_metrics is False
    assert manifest.promoted is False
    assert registry.promoted("lig_anchor") is None
    assert "head=disabled" in manifest.notes
    assert json.loads(Path(out, "metrics.json").read_text())["buffer_size"] == 16


def test_fewer_resolved_pairs_than_the_floor_is_insufficient_evidence(tmp_path):
    fit = _StubHead()
    job = _job(tmp_path, _corpus(6), head_enabled=True)
    run_round(job, fit=fit)
    manifest = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0)
    assert manifest.has_heldout_metrics is False
    assert "below the declared floor" in manifest.notes
    assert fit.calls == [], "nothing is fitted below the floor"


def test_one_group_is_not_a_fold(tmp_path):
    """Leave-one-group-out over a single ligand has no folds, and a random split would leak."""
    fit = _StubHead()
    job = _job(tmp_path, _corpus(12, groups=1), head_enabled=True)
    out = run_round(job, fit=fit)
    report = json.loads(Path(out, "metrics.json").read_text())
    assert report["validated"] is False and report["logo_r2"] is None
    assert "leak" in report["reason"]
    assert fit.calls == []


def test_a_validated_round_reports_logo_r2_and_ranking_accuracy(tmp_path):
    fit = _StubHead()
    job = _job(tmp_path, head_enabled=True)
    out = run_round(job, fit=fit)
    report = json.loads(Path(out, "metrics.json").read_text())
    assert report["validated"] is True
    assert report["logo_r2"] is not None
    assert 0.0 <= report["rank_accuracy"] <= 1.0
    assert report["holdout_n"] == 16, "every pair gets a turn as held-out evidence"
    manifest = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0)
    assert manifest.logo_r2 == report["logo_r2"]
    assert manifest.min_holdout_n == job["min_pairs"]


def test_a_negative_logo_r2_is_reported_rather_than_withheld(tmp_path):
    """`logo_r2 < 0` means predicting the group mean beats the model, which is the number
    that catches a self-confirming objective."""

    def backwards(train, job):
        return lambda rows: [float(r["outcome_value"]) for r in rows]

    job = _job(tmp_path, head_enabled=True)
    out = run_round(job, fit=backwards)
    report = json.loads(Path(out, "metrics.json").read_text())
    assert report["logo_r2"] is not None and report["logo_r2"] < 0


def test_the_round_records_cpu_wall_clock_and_never_names_a_gpu(tmp_path):
    job = _job(tmp_path, head_enabled=True)
    out = run_round(job, fit=_StubHead())
    manifest = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0)
    assert manifest.train_seconds_cpu is not None and manifest.train_seconds_cpu >= 0.0
    assert "gpu" not in manifest.notes.lower(), (
        "item 48: a CPU wall clock is a measurement of a real thing and is never converted "
        "into a GPU-hour"
    )
    assert "train_seconds_cpu" in json.loads(Path(out, "metrics.json").read_text())


def test_the_manifest_does_not_claim_a_frozen_encoder(tmp_path):
    """The one gap the adaptation leaves. `itgt`'s `train_head` trains the MIL head's own
    encoder end to end and has no hook to freeze a pretrained one, so the manifest says what
    happened. When the hook lands upstream this test is what forces the note to change."""
    job = _job(tmp_path, head_enabled=True)
    run_round(job, fit=_StubHead())
    notes = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0).notes
    assert "encoder=trained_end_to_end" in notes
    assert "frozen" not in notes


def test_the_marker_is_written_last(tmp_path):
    job = _job(tmp_path, head_enabled=False)
    out = run_round(job, fit=_StubHead())
    marker = Path(out, TRAIN_COMPLETE_MARKER)
    assert marker.exists() and marker.read_text().strip() == "0"
    assert marker.stat().st_mtime >= Path(out, "metrics.json").stat().st_mtime


# -- the real head -----------------------------------------------------------


def test_the_real_itgt_head_ranks_a_corpus_it_was_fitted_on(tmp_path):
    """`itgt`'s MIL head on a synthetic complex whose interface is known by construction.

    In-sample rather than held-out on purpose: across ligands, eight bags transfer at about
    chance, and a test that asserted otherwise would be asserting a result rather than that
    the machinery runs. What this proves is that the corpus this wrapper assembles is one
    `train_head` can actually learn from -- which is the whole claim of the reuse verdict.
    """
    if not (ITGT / "itgt" / "method2_hotspots" / "train.py").exists():
        pytest.skip("the interface-targeting checkout is not beside us")
    pytest.importorskip("torch.nn", reason="itgt's [ml] extra")
    pytest.importorskip("gemmi", reason="itgt reads structures with gemmi")
    sys.path.insert(0, str(ITGT))
    sys.path.insert(0, str(ITGT / "examples"))
    from synthetic_ligand import write

    from examples.impress.small_molecule_binding.liganchor_train_wrapper import itgt_head

    structure = str(tmp_path / "lig.pdb")
    truth = write(structure, n_ring=10, seed=0)
    ring, interface = truth["ring_names"], set(truth["interface_ligand_atoms"])

    import random
    rng = random.Random(7)
    pairs = []
    for i in range(8):
        atoms = sorted(rng.sample(ring, 4))
        hits = len(set(atoms) & interface)
        pairs.append({
            "action_id": f"act-{i}", "node_id": f"nd-{i}", "group": "lig",
            "action": _action(atoms, structure=structure),
            "outcome_value": -7.0 - 0.8 * hits + 0.01 * i,
        })

    job = {"property_name": "dg_abfe", "direction": -1, "epochs": 150, "seed": 0,
           "scorer": "asd:dg_abfe", "encoder_path": ""}
    scores = itgt_head(pairs, job)(pairs)
    assert len(scores) == len(pairs)

    correct = total = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = -pairs[i]["outcome_value"], -pairs[j]["outcome_value"]
            if a == b:
                continue
            total += 1
            correct += int((scores[i] - scores[j]) * (a - b) > 0)
    assert correct / total > 0.7, (
        f"the head ranked {correct}/{total} of the pairs it was fitted on correctly; below "
        "this the corpus is not one `train_head` can learn from at all"
    )


# -- the weights (item 52) ---------------------------------------------------


def test_a_banking_round_publishes_no_weights_because_it_fitted_nothing(tmp_path):
    job = _job(tmp_path, head_enabled=False)
    run_round(job, fit=_StubHead())
    manifest = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0)
    assert "weights=none" in manifest.notes
    assert os.listdir(os.path.join(job["registry_root"], "lig_anchor", "v0")) == [
        "manifest.json"]


def test_the_published_head_is_fitted_on_every_banked_pair(tmp_path):
    """The folds measure; the final fit on the whole buffer is what gets served."""
    fit = _StubHead()
    job = _job(tmp_path, head_enabled=True)
    run_round(job, fit=fit)
    assert fit.calls[-1] == 16, (
        f"the last fit saw {fit.calls[-1]} of 16 pairs; the published head must be the "
        "all-buffer fit, not a fold's"
    )
    assert all(n < 16 for n in fit.calls[:-1]), "the folds held a group out"


def test_the_final_fit_is_told_where_to_write_its_artifact(tmp_path):
    """`itgt`'s `train_head` writes the artifact itself, recording the scorer, the scoring
    config, the feature columns and the corpus fingerprint. Reconstructing that meta here
    would be a second copy of itgt's provenance, drifting one field at a time."""
    seen: list = []

    def recording(train, job):
        seen.append(job.get("artifact_out"))
        return _StubHead()(train, job)

    job = _job(tmp_path, head_enabled=True)
    run_round(job, fit=recording)
    assert seen[-1] and seen[-1].endswith(os.path.join("lig_anchor", "v0", "head"))
    assert all(o is None for o in seen[:-1]), "a fold must not write the published artifact"


def test_loading_a_version_that_has_no_head_refuses(tmp_path):
    from examples.impress.small_molecule_binding.liganchor_train_wrapper import load_state

    job = _job(tmp_path, head_enabled=False)
    run_round(job, fit=_StubHead())
    with pytest.raises(FileNotFoundError, match="nothing to serve"):
        load_state(os.path.join(job["registry_root"], "lig_anchor", "v0"))


def test_the_real_head_is_written_as_an_itgt_artifact_and_loads_back(tmp_path):
    if not (ITGT / "itgt" / "method2_hotspots" / "train.py").exists():
        pytest.skip("the interface-targeting checkout is not beside us")
    pytest.importorskip("torch.nn", reason="itgt's [ml] extra")
    pytest.importorskip("gemmi", reason="itgt reads structures with gemmi")
    sys.path.insert(0, str(ITGT))
    sys.path.insert(0, str(ITGT / "examples"))
    from synthetic_ligand import write

    from examples.impress.small_molecule_binding.liganchor_train_wrapper import load_state

    import random
    rng = random.Random(7)
    rows = []
    for group, kwargs in (("g1", {"n_ring": 10, "seed": 0}),
                          ("g2", {"n_ring": 8, "radius": 3.4, "seed": 1})):
        structure = str(tmp_path / f"{group}.pdb")
        truth = write(structure, **kwargs)
        ring, interface = truth["ring_names"], set(truth["interface_ligand_atoms"])
        for k in range(8):
            atoms = sorted(rng.sample(ring, 4))
            hits = len(set(atoms) & interface)
            rows.append(_record(len(rows), atoms, -7.0 - 0.8 * hits + 0.01 * k,
                                group=group,
                                action=_action(atoms, structure=structure)))

    job = _job(tmp_path, rows, head_enabled=True, epochs=10)
    run_round(job)
    version_dir = os.path.join(job["registry_root"], "lig_anchor", "v0")
    assert sorted(os.listdir(version_dir)) == ["head", "manifest.json"]
    manifest = ModelRegistry(job["registry_root"]).read_manifest("lig_anchor", 0)
    assert "weights=head/" in manifest.notes

    meta = json.loads(Path(version_dir, "head", "meta.json").read_text())
    assert meta["kind"] == "MILHead"
    assert meta["scorer"] == "asd:dg_abfe", (
        "itgt refuses to compare a proxy-fit artifact against a campaign-scored one, which "
        "only works if a campaign round names itself"
    )
    assert type(load_state(version_dir, manifest)).__name__ == "MILHead"


def test_a_proxy_fit_head_is_refused_at_load(tmp_path):
    """A model fitted on `itgt`'s synthetic quality function must not steer a campaign.
    itgt's own guard is `evaluate.compare_to_baseline`; serving one would route around it."""
    from examples.impress.small_molecule_binding.liganchor_train_wrapper import load_state

    version_dir = tmp_path / "lig_anchor" / "v0" / "head"
    version_dir.mkdir(parents=True)
    (version_dir / "meta.json").write_text(json.dumps(
        {"method": "hotspots", "kind": "MILHead", "scorer": "proxy", "format": "torch"}))
    (version_dir / "state.pt").write_bytes(b"")
    if not (ITGT / "itgt").exists():
        pytest.skip("the interface-targeting checkout is not beside us")
    sys.path.insert(0, str(ITGT))
    with pytest.raises(ValueError, match="proxy labels"):
        load_state(str(version_dir.parent))
