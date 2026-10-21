"""The IMPRESS-R hook wiring, tested offline (no MPNN, AlphaFold, or Dragon).

``examples/impress_r/protein_binding_rome.py`` adds two calls to IMPRESS's
``adaptive_decision``. The pipeline itself needs GPUs and the science
executables, but the hook logic — read the score CSV, copy the prediction aside,
pull the ranked sequence, contribute to ROME — is plain Python and is exactly
where a wiring bug would hide. This drives it with a fake pipeline and a manager
double, on the *first pass* (IMPRESS's migration branch just saves scores and
returns), so it exercises the hooks without the migration machinery.
"""

import csv
import os
from types import SimpleNamespace

import pytest

# IMPRESS's protein_binding module reads os.environ['USER'] at import time.
os.environ.setdefault("USER", "tester")

pytest.importorskip("impress")
pytest.importorskip("rhapsody")

try:
    from examples.impress_r.protein_binding.run_protein_binding_rome import make_adaptive_decision  # noqa: E402
except Exception as exc:  # the runner's deps are cluster-managed; skip if absent
    pytest.skip(f"run_protein_binding_rome not importable here: {exc}",
                allow_module_level=True)


class _ManagerDouble:
    """Records add_training_data calls; enough surface for the hooks."""

    def __init__(self):
        self.added = []
        self.data = SimpleNamespace(total_count=0)
        self._status = SimpleNamespace(name="NOT_ENOUGH_DATA")

    def add_training_data(self, **record):
        self.added.append(record)
        self.data.total_count = len(self.added)
        return f"uid{len(self.added)}"

    def get_current_model(self):
        return None

    def get_training_status(self):
        return self._status


def _fake_pipeline(tmp_path, rows):
    af_out = tmp_path / "af_out"
    af_out.mkdir()
    iter_seqs = {}
    for protein, _plddt, _ptm, _pae in rows:
        (af_out / f"{protein}.pdb").write_text(
            f"ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n"
        )
        # iter_seqs maps design -> ranked [seq, score] pairs; seq_rank picks one.
        iter_seqs[protein] = [[f"SEQ_{protein}_rank0", 1.0], [f"SEQ_{protein}_rank1", 2.0]]

    log = SimpleNamespace(pipeline_log=lambda *a, **k: None)
    return SimpleNamespace(
        name="p1", passes=1, output_path_af=str(af_out), iter_seqs=iter_seqs,
        seq_rank=0, current_scores={}, previous_scores={}, base_path=str(tmp_path),
        sub_order=0, fasta_list_2=[], logger=log,
    )


def _write_stats(tmp_path, name, passes, rows):
    path = tmp_path / f"af_stats_{name}_pass_{passes}.csv"
    with open(path, "w", newline="") as fd:
        w = csv.writer(fd)
        w.writerow(["ID", "avg_plddt", "ptm", "avg_pae"])
        for protein, plddt, ptm, pae in rows:
            w.writerow([f"{protein}.pdb", plddt, ptm, pae])
    return path


@pytest.mark.asyncio
async def test_hook_contributes_each_scored_design(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                      # adaptive_decision opens a relative CSV
    rows = [("2ejy", 95.0, 0.90, 3.5), ("2pdz", 96.0, 0.88, 4.1)]
    _write_stats(tmp_path, "p1", 1, rows)

    manager = _ManagerDouble()
    fn = make_adaptive_decision(manager, str(tmp_path / "stage"))
    await fn(_fake_pipeline(tmp_path, rows))

    assert len(manager.added) == 2
    by_backbone = {r["backbone_id"]: r for r in manager.added}
    assert set(by_backbone) == {"2ejy", "2pdz"}

    rec = by_backbone["2ejy"]
    assert rec["pLDDT"] == 95.0 and rec["pTM"] == 0.90 and rec["pAE"] == 3.5
    assert rec["sequence"] == "SEQ_2ejy_rank0"       # seq_rank 0 of the ranked list
    assert os.path.isfile(rec["path"])               # the prediction was staged


@pytest.mark.asyncio
async def test_hook_copies_the_prediction_aside(tmp_path, monkeypatch):
    """The staged file must be independent of the overwritten source."""
    monkeypatch.chdir(tmp_path)
    rows = [("2ejy", 95.0, 0.90, 3.5)]
    _write_stats(tmp_path, "p1", 1, rows)

    manager = _ManagerDouble()
    pipeline = _fake_pipeline(tmp_path, rows)
    fn = make_adaptive_decision(manager, str(tmp_path / "stage"))
    await fn(pipeline)

    staged = manager.added[0]["path"]
    assert os.path.dirname(staged) == str(tmp_path / "stage")
    assert staged != os.path.join(pipeline.output_path_af, "2ejy.pdb")

    # Overwrite the source: the staged copy must not change with it.
    before = open(staged).read()
    open(os.path.join(pipeline.output_path_af, "2ejy.pdb"), "w").write("OVERWRITTEN")
    assert open(staged).read() == before


@pytest.mark.asyncio
async def test_first_pass_saves_scores_and_does_not_migrate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [("2ejy", 95.0, 0.90, 3.5)]
    _write_stats(tmp_path, "p1", 1, rows)

    manager = _ManagerDouble()
    pipeline = _fake_pipeline(tmp_path, rows)
    fn = make_adaptive_decision(manager, str(tmp_path / "stage"))
    await fn(pipeline)

    # IMPRESS's first-pass branch: current scores become previous, no child.
    assert pipeline.previous_scores == {"2ejy": 3.5}   # the interface pAE
    assert "2ejy" in pipeline.iter_seqs                 # nothing migrated out


@pytest.mark.asyncio
async def test_hook_skips_a_design_with_no_prediction_file(tmp_path, monkeypatch):
    """A CSV row whose PDB is missing is skipped, not crashed on."""
    monkeypatch.chdir(tmp_path)
    rows = [("2ejy", 95.0, 0.90, 3.5), ("gone", 80.0, 0.70, 6.0)]
    _write_stats(tmp_path, "p1", 1, rows)

    manager = _ManagerDouble()
    pipeline = _fake_pipeline(tmp_path, rows)
    os.remove(os.path.join(pipeline.output_path_af, "gone.pdb"))
    fn = make_adaptive_decision(manager, str(tmp_path / "stage"))
    await fn(pipeline)

    assert [r["backbone_id"] for r in manager.added] == ["2ejy"]
