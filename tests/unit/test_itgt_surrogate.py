"""Unit tests for the interface-targeting surrogate workflow.

Everything here runs with **no itgt, no torch, no scikit-learn and no IMPRESS**.
That is the point of the module split under
``examples/impress_r/interface_targeting/``: the spec-patching surface is where
most of the correctness lives, so it is kept in pure standard library and
tested without a scientific stack.

The fixtures are built inline rather than read out of the IMPRESS checkout, so
these tests pass on a machine that has never seen IMPRESS. The shapes they
encode were taken from the live
``examples/small_molecule_binding/p1_in/ALR_binder_design.json`` and its
reference PDB, including the detail that makes the fixed-column parsing
necessary: the ligand's resname is literally ``A:R``.
"""
from __future__ import annotations

import json
import os

import pytest

from examples.impress_r.interface_targeting import itgt_spec


# ── fixtures ──────────────────────────────────────────────────────────────────

#: Atom names of the reference ligand, in file order.
LIGAND_ATOMS = (
    ["S1", "S2"]
    + [f"O{i}" for i in range(3, 11)]
    + ["N11", "N12"]
    + [f"C{i}" for i in range(13, 31)]
)

BURIED = ["C18", "C19", "C23", "C25", "C26", "C27", "C29", "C30",
          "N12", "O3", "O5", "O7", "O8", "S1"]
EXPOSED = [n for n in LIGAND_ATOMS if n not in BURIED]


def _template() -> dict:
    """The live base spec's shape: one entry named ``partial``, two rASA bins."""
    return {
        "partial": {
            "input": "./input_pdbs/scaffold-with-ALR.pdb",
            "ligand": "A:R",
            "length": "70-170",
            "select_exposed": {"A:R": ",".join(EXPOSED)},
            "select_buried": {"A:R": ",".join(BURIED)},
        }
    }


def _pdb_text(names=LIGAND_ATOMS, resname="A:R") -> str:
    """Fixed-column HETATM records, resname in columns 18-20."""
    lines = []
    for serial, name in enumerate(names, start=1124):
        line = (
            f"HETATM{serial:>5} {name:<4}"
            f"{'':1}{resname:<3} B 386"
            f"{0.0:>12.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.fixture
def input_dir(tmp_path):
    """A staged pipeline input directory with a spec and its reference PDB."""
    pdb_dir = tmp_path / "input_pdbs"
    pdb_dir.mkdir()
    (pdb_dir / "scaffold-with-ALR.pdb").write_text(_pdb_text())
    (tmp_path / itgt_spec.DEFAULT_SPEC_NAME).write_text(json.dumps(_template(), indent=4))
    return str(tmp_path)


# ── reading the spec ──────────────────────────────────────────────────────────

def test_entry_name_requires_the_entry_the_pipeline_reads():
    """``_write_guided_rfd3_json`` indexes ``base['partial']`` unconditionally,
    so a spec without it would silently take the unguided fallback forever."""
    assert itgt_spec.entry_name(_template()) == "partial"
    with pytest.raises(KeyError, match="partial"):
        itgt_spec.entry_name({"buried": {}})


def test_reference_pdb_resolves_relative_to_the_spec_directory(input_dir):
    """Mirrors the pipeline's ``_resolve_reference_pdb_path``: diverging here
    would featurize a different structure than RFD3 is handed."""
    path = itgt_spec.spec_path(input_dir)
    resolved = itgt_spec.reference_pdb(path, itgt_spec.load_spec(path))
    assert os.path.exists(resolved)
    assert os.path.realpath(resolved) == os.path.realpath(
        os.path.join(input_dir, "input_pdbs", "scaffold-with-ALR.pdb")
    )


def test_reference_pdb_passes_absolute_paths_through(tmp_path):
    spec = _template()
    spec["partial"]["input"] = "/somewhere/else/ligand.pdb"
    assert itgt_spec.reference_pdb(str(tmp_path / "spec.json"), spec) == (
        "/somewhere/else/ligand.pdb"
    )


def test_template_partition_inverts_the_spec():
    groups = itgt_spec.template_partition(_template())
    assert set(groups) == {"buried", "exposed"}
    assert groups["buried"] == BURIED
    assert groups["exposed"] == EXPOSED
    # The hand-authored partition covers the whole ligand -- no atom is left
    # outside the three rASA bins.
    assert len(groups["buried"]) + len(groups["exposed"]) == len(LIGAND_ATOMS)


def test_ligand_atom_names_parses_a_colon_bearing_resname(input_dir):
    """``A:R`` is why this is fixed-column parsing and not a whitespace split."""
    reference = os.path.join(input_dir, "input_pdbs", "scaffold-with-ALR.pdb")
    assert itgt_spec.ligand_atom_names(reference, "A:R") == LIGAND_ATOMS
    assert itgt_spec.ligand_atom_names(reference, "LIG") == []


# ── patching ──────────────────────────────────────────────────────────────────

def test_patching_with_the_templates_own_partition_is_a_round_trip():
    """The decisive test: if this is not identity, nothing downstream can be
    trusted, because every round's spec is produced by this function."""
    template = _template()
    groups = itgt_spec.template_partition(template)
    assert itgt_spec.patch_spec(template, groups) == template


def test_patch_spec_preserves_everything_it_does_not_own():
    template = _template()
    patched = itgt_spec.patch_spec(
        template, {"buried": ["S1", "S2"], "exposed": LIGAND_ATOMS[2:]}
    )
    assert list(patched) == ["partial"]
    body, original = patched["partial"], template["partial"]
    for field in ("input", "ligand", "length"):
        assert body[field] == original[field]
    # select_fixed_atoms is absent upstream and must stay absent: its *keys*
    # leak into LigandMPNN's fixed-residue list, so inventing it would corrupt
    # stages 2 and 3 rather than stage 1.
    assert "select_fixed_atoms" not in body


def test_patch_spec_does_not_mutate_the_template():
    template = _template()
    before = json.dumps(template, sort_keys=True)
    itgt_spec.patch_spec(template, {"buried": ["S1"], "exposed": LIGAND_ATOMS[1:]})
    assert json.dumps(template, sort_keys=True) == before


def test_patch_spec_keys_selections_by_the_templates_ligand_value():
    """The selection key is the entry's ``ligand`` field. It coincides with the
    resname on this target, but they are different concepts and the spec is the
    authority."""
    template = _template()
    template["partial"]["ligand"] = "X:Q"
    patched = itgt_spec.patch_spec(template, {"buried": ["S1"], "exposed": ["S2"]})
    assert patched["partial"]["select_buried"] == {"X:Q": "S1"}


def test_an_absent_level_removes_its_field_rather_than_writing_it_empty():
    """RFD3 raises on a selection matching no atoms, so a two-bin proposal must
    produce a two-bin spec -- not a third field with an empty string."""
    patched = itgt_spec.patch_spec(
        _template(), {"buried": ["S1"], "exposed": LIGAND_ATOMS[1:]}
    )
    assert "select_partially_buried" not in patched["partial"]


def test_three_bin_proposals_write_the_partial_field():
    patched = itgt_spec.patch_spec(
        _template(),
        {"buried": ["S1"], "partial": ["S2"], "exposed": LIGAND_ATOMS[2:]},
    )
    assert patched["partial"]["select_partially_buried"] == {"A:R": "S2"}


def test_patch_spec_rejects_degenerate_input():
    with pytest.raises(ValueError, match="no selections"):
        itgt_spec.patch_spec(_template(), {})
    with pytest.raises(ValueError, match="unknown burial level"):
        itgt_spec.patch_spec(_template(), {"very_buried": ["S1"]})


# ── the coverage guard ────────────────────────────────────────────────────────

def test_coverage_checks_all_three_fields_including_partial():
    """The pipeline's own guard validates only ``select_exposed`` and
    ``select_buried`` while copying ``select_partially_buried`` through
    unchecked, so a slip there escalates a clean fallback into a mid-run RFD3
    validation error. This check is what closes that gap."""
    ok, missing = itgt_spec.coverage_ok(
        {"buried": ["S1"], "partial": ["NOPE"], "exposed": ["S2"]}, LIGAND_ATOMS
    )
    assert not ok
    assert missing == {"NOPE"}

    ok, missing = itgt_spec.coverage_ok({"buried": BURIED, "exposed": EXPOSED}, LIGAND_ATOMS)
    assert ok and not missing


# ── staging and history ───────────────────────────────────────────────────────

def test_snapshot_is_idempotent_so_rounds_cannot_compound(input_dir):
    """Round *n* patches from the pristine template, never from round *n-1*'s
    output. A second snapshot call after a patch must not capture the patch."""
    first = itgt_spec.snapshot_template(input_dir)
    patched = itgt_spec.patch_spec(
        itgt_spec.load_template(input_dir), {"buried": ["S1"], "exposed": LIGAND_ATOMS[1:]}
    )
    itgt_spec.write_spec_atomic(patched, input_dir, version=1)

    assert itgt_spec.snapshot_template(input_dir) == first
    assert itgt_spec.template_partition(itgt_spec.load_template(input_dir))["buried"] == BURIED
    # ...while the live spec carries the patch.
    live = itgt_spec.load_spec(itgt_spec.spec_path(input_dir))
    assert itgt_spec.template_partition(live)["buried"] == ["S1"]


def test_write_spec_atomic_keeps_a_per_version_copy(input_dir):
    """``scripts/validate_run.py`` check 3 asserts guided JSONs match *the
    current* base spec, so once the base moves, earlier rounds only stay
    auditable through this history."""
    itgt_spec.snapshot_template(input_dir)
    spec = itgt_spec.patch_spec(
        itgt_spec.load_template(input_dir), {"buried": ["S1"], "exposed": LIGAND_ATOMS[1:]}
    )
    itgt_spec.write_spec_atomic(spec, input_dir, version=3)
    history = os.path.join(input_dir, itgt_spec.ITGT_DIR_NAME, "history", "v3.json")
    assert json.loads(open(history).read()) == spec


def test_load_template_falls_back_to_the_live_spec(input_dir):
    """Before the first snapshot there is nothing else to patch from."""
    assert itgt_spec.load_template(input_dir) == _template()


# ── reward, admission, sampling ───────────────────────────────────────────────

def test_reward_is_none_safe():
    """``best_ligand_iptm`` is absent when the fold produced no confidence file
    and ``None`` when the JSON carried no such key. Neither may raise, and both
    are usable evidence: that partition produced nothing measurable."""
    from examples.impress_r.interface_targeting.itgt_trainer import itgt_reward_fn

    assert itgt_reward_fn({"ligand_iptm": 0.75}) == pytest.approx(0.75)
    assert itgt_reward_fn({"ligand_iptm": None}) == 0.0
    assert itgt_reward_fn({}) == 0.0
    assert itgt_reward_fn({"ligand_iptm": "not a number"}) == 0.0


def test_corpus_filter_admits_failures_by_default():
    """The opposite of the sibling's filter, deliberately: a reward regression
    needs its negatives, and filtering to good designs truncates the response
    variable it learns from."""
    from examples.impress_r.interface_targeting.itgt_trainer import itgt_corpus_filter

    admit = itgt_corpus_filter()
    assert admit({"ligand_iptm": 0.1})
    assert admit({"ligand_iptm": None})

    strict = itgt_corpus_filter(min_ligand_iptm=0.5)
    assert not strict({"ligand_iptm": 0.1})
    assert strict({"ligand_iptm": 0.9})

    needs_reward = itgt_corpus_filter(require_reward=True)
    assert not needs_reward({"ligand_iptm": None})


def test_sampler_keeps_the_whole_corpus_by_default():
    """``fraction`` defaults to 1.0 here, unlike the sibling's 0.33: selecting
    the best third of a reward regression's own training data is a bias."""
    from examples.impress_r.interface_targeting.itgt_trainer import percentile_sampler

    records = [{"uid": str(i), "ligand_iptm": i / 10.0} for i in range(10)]
    assert len(percentile_sampler()(records)) == 10


def test_sampler_ranks_by_the_requested_direction():
    from examples.impress_r.interface_targeting.itgt_trainer import percentile_sampler

    records = [{"uid": str(i), "ligand_iptm": i / 10.0} for i in range(10)]
    best = percentile_sampler(0.3, min_shard=1)(records)
    assert [r["uid"] for r in best] == ["9", "8", "7"]


def test_sampler_passes_records_through_when_no_field_is_numeric():
    from examples.impress_r.interface_targeting.itgt_trainer import percentile_sampler

    records = [{"uid": "a"}, {"uid": "b"}]
    assert percentile_sampler(0.5, min_shard=1)(records) == records


# ── the task's own guards ─────────────────────────────────────────────────────

def test_config_rejects_impossible_settings():
    from examples.impress_r.interface_targeting.itgt_trainer import ItgtSurrogateConfig

    with pytest.raises(ValueError, match="component"):
        ItgtSurrogateConfig(component="telekinesis").validate()
    with pytest.raises(ValueError, match="size_band"):
        ItgtSurrogateConfig(
            component="burial_prior", size_band=-1
        ).validate()
    with pytest.raises(ValueError, match="min_atoms <= max_atoms"):
        ItgtSurrogateConfig(
            component="burial_prior", min_atoms=20, max_atoms=4
        ).validate()
    # reward_ridge cannot run without somewhere to read its pairs from.
    with pytest.raises(ValueError, match="replay_buffer"):
        ItgtSurrogateConfig(component="reward_ridge").validate()


def _trainer(tmp_path, **overrides):
    from examples.impress_r.interface_targeting.itgt_trainer import (
        ItgtSurrogateConfig,
        ItgtSurrogateTrainer,
    )

    buffer = tmp_path / "replay.jsonl"
    buffer.write_text("")
    config = dict(
        component="reward_ridge",
        replay_buffer=str(buffer),
        reference_pdb=str(tmp_path / "ref.pdb"),
        ligand_key="A:R",
    )
    config.update(overrides)
    return ItgtSurrogateTrainer(ItgtSurrogateConfig(**config))


def test_validate_rejects_a_corpus_without_a_structure_path(tmp_path):
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="path"):
        trainer.validate([{"ligand_iptm": 0.5}])


def test_validate_rejects_a_corpus_with_no_reward_at_all(tmp_path):
    """A round that regresses on ``ligand_iptm`` and has none of it would fit
    nothing, so it fails on the manager instead of after a queue wait."""
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="ligand_iptm"):
        trainer.validate([{"path": "/tmp/a.pdb", "ligand_iptm": None}])


def test_validate_accepts_a_mixed_corpus(tmp_path):
    """Failures alongside successes is the normal, wanted case."""
    _trainer(tmp_path).validate(
        [
            {"path": "/tmp/a.pdb", "ligand_iptm": 0.8},
            {"path": "/tmp/b.pdb", "ligand_iptm": None},
        ]
    )


def test_as_command_declares_the_artifact_and_writes_a_job(tmp_path):
    """An executable task's return value is the backend's execution result, not
    a path, so the artifact location has to be declared up front."""
    trainer = _trainer(tmp_path)
    output_dir = str(tmp_path / "round")
    command, artifact = trainer.as_command(
        [{"path": "/tmp/a.pdb", "ligand_iptm": 0.8, "uid": "u1"}],
        output_dir,
        model_version=2,
    )

    assert artifact == os.path.join(output_dir, "artifact")
    assert "--job" in command and command.endswith("job.json")

    job = json.loads(open(os.path.join(output_dir, "job.json")).read())
    assert job["artifact"] == artifact
    assert job["model_version"] == 2
    assert job["designs"][0]["ligand_iptm"] == pytest.approx(0.8)
    assert os.path.exists(job["manifest"])


def test_the_round_command_writes_the_completion_marker(tmp_path):
    """``train_complete`` is how the training manager detects a finished round
    when Dragon fails to deliver a task's result, so the wrapper must name it."""
    from examples.impress_r.interface_targeting import itgt_train_wrapper

    assert itgt_train_wrapper.TRAIN_COMPLETE_MARKER == "train_complete"

    from rome.trainer import TRAIN_COMPLETE_MARKER

    assert itgt_train_wrapper.TRAIN_COMPLETE_MARKER == TRAIN_COMPLETE_MARKER


def test_train_func_is_an_in_process_escape_hatch(tmp_path):
    seen = {}

    def fake_round(job):
        seen.update(job)
        return job["artifact"]

    trainer = _trainer(tmp_path, train_func=fake_round)
    output_dir = str(tmp_path / "round")
    assert trainer.as_command([{"path": "/tmp/a.pdb"}], output_dir) is None
    artifact = trainer.train([{"path": "/tmp/a.pdb", "ligand_iptm": 0.5}], output_dir)
    assert artifact == os.path.join(output_dir, "artifact")
    assert seen["config"]["component"] == "reward_ridge"


# ── the apply step's failure modes ────────────────────────────────────────────

def test_a_non_itgt_checkpoint_is_not_mistaken_for_an_artifact(tmp_path):
    """``DummyTrainer`` publishes a plain directory and ROME hands the
    checkpoint path over as-is, so "is this even mine?" must be answerable
    without raising."""
    from examples.impress_r.interface_targeting import itgt_propose

    dummy = tmp_path / "dummy_v1"
    dummy.mkdir()
    (dummy / "checkpoint.json").write_text("{}")

    assert itgt_propose.artifact_kind(str(dummy)) is None
    assert itgt_propose.artifact_kind(str(tmp_path / "missing")) is None
    assert itgt_propose.artifact_kind("") is None


def test_apply_returns_false_instead_of_raising(input_dir, tmp_path):
    """A failed proposal must cost one pass's improvement, never the campaign."""
    from examples.impress_r.interface_targeting import itgt_propose

    dummy = tmp_path / "dummy_v1"
    dummy.mkdir()

    before = open(itgt_spec.spec_path(input_dir)).read()
    assert itgt_propose.apply_to_pipeline(
        input_dir, str(dummy), 1, logger=lambda _m: None
    ) is False
    # ...and the spec it could not improve is left exactly as it was.
    assert open(itgt_spec.spec_path(input_dir)).read() == before


def test_apply_reports_a_missing_spec_without_raising(tmp_path):
    from examples.impress_r.interface_targeting import itgt_propose

    messages = []
    assert itgt_propose.apply_to_pipeline(
        str(tmp_path), str(tmp_path), 1, logger=messages.append
    ) is False
    assert messages


# ── grouping, which the staged filename is load-bearing for ───────────────────

def test_the_staged_filename_groups_per_pipeline_and_pass():
    """itgt's ``design_stem`` strips ``_model_N`` and a trailing ``_N``. Every
    Boltz fold output is called ``boltz_input_model_0.pdb``, which would collapse
    the whole campaign into one group and make leave-one-group-out an estimate
    of nothing. The staged name is what restores a real split, so it is part of
    the contract rather than cosmetic."""
    pytest.importorskip("itgt")
    from itgt.common.datasets import design_stem

    assert design_stem("boltz_input_model_0.pdb") == "boltz_input"
    assert design_stem("p1_pass3_boltz_input_model_0.pdb") == "p1_pass3_boltz_input"
    assert design_stem("p1_pass3_boltz_input_model_7.pdb") == design_stem(
        "p1_pass3_boltz_input_model_0.pdb"
    )
    assert design_stem("p2_pass3_boltz_input_model_0.pdb") != design_stem(
        "p1_pass3_boltz_input_model_0.pdb"
    )
