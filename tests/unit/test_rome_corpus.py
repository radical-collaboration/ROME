"""The ASD corpus hook: what a trainer is owed, what it takes, and what it leaves owed.

Everything here runs with **no torch, no scikit-learn and no IMPRESS**, on a tmp directory.
The hook is `asteer` plus the standard library by design, and the last test asserts that
rather than trusting it (CON-4).

Skipped, not failed, without `asteer`: it is a sibling checkout rather than a declared
dependency of ROME today. `examples/impress_r/delta_env_setup.sh` installs it alongside the
other three on Delta.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("asteer", reason="asteer is a sibling checkout, not a ROME dependency")

from asteer.contracts import LabelKind, LabelRecord  # noqa: E402

from examples.impress.small_molecule_binding.corpus import (  # noqa: E402
    CorpusBatch,
    LabelCorpus,
    record_to_row,
)

B = "pocket_preorg"
C = "aff_delta"


def _record(n: int, value: float, *, kind: LabelKind = LabelKind.MD) -> LabelRecord:
    return LabelRecord(
        label_id=f"lab-{n:03d}",
        kind=kind,
        design_campaign_id="asd-smb",
        node_id=f"n{n:06d}",
        property_name="preorg_rmsd",
        value=value,
        produced_at="2026-11-21T00:00:00+00:00",
        units="angstrom",
        sigma=0.05,
        protocol="md/30ns",
        source_job_id="24155612",
    )


@pytest.fixture
def store(tmp_path):
    """Nine records whose values rank cleanly, so a third is unambiguous."""
    from asteer.labels import LabelQueue

    queue = LabelQueue(tmp_path)
    for n, value in enumerate([0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.5], start=1):
        queue.write(_record(n, value))
    return tmp_path


# -- what a round takes, and what it leaves ----------------------------------


def test_a_rank_sampled_round_marks_only_what_it_took(store):
    """The whole reason this reads `asteer.labels` instead of a watermark.

    A sampler selects a non-contiguous subset. A high-water mark advanced past it would
    retire the records it skipped having never trained on them -- E0.8's defect, one repo
    away. The set marks three ids and leaves six owed.
    """
    corpus = LabelCorpus(store, B, fraction=1 / 3, rank_by={"value": "low"},
                         on_summary=None)
    batch = corpus.take()
    assert len(batch) == 3
    assert [row["value"] for row in batch.rows] == [0.1, 0.2, 0.3], "ranked low-first"

    corpus.commit(batch)
    still_owed = {rec.label_id for _, rec in corpus.owed()}
    # Values are 0.9 0.1 0.7 0.3 0.8 0.2 0.6 0.4 0.5 for lab-001..009, so the best third
    # by `value: low` is 002, 006, 004 -- deliberately non-contiguous, which is the whole
    # point: no watermark value describes this batch.
    assert still_owed == {f"lab-{n:03d}" for n in (1, 3, 5, 7, 8, 9)}, (
        f"six records should still be owed, got {sorted(still_owed)}"
    )
    assert set(batch.ids) == {"lab-002", "lab-006", "lab-004"}
    assert set(batch.ids).isdisjoint(still_owed)
    assert {rec.label_id for _, rec in batch.left_owed} == still_owed


def test_taking_without_committing_loses_nothing(store):
    """A trainer can die between `take()` and `commit()`, and a restart must not pay for it.

    This is why they are two calls. `take()` marks nothing, so the same records are offered
    again -- which is also `LabelQueue.unconsumed`'s documented promise: *"called again
    before `mark_consumed()`, it returns the same tail. That is the point."*
    """
    corpus = LabelCorpus(store, B, on_summary=None)
    first = corpus.take()
    assert len(first) == 9

    # The round dies here. Nothing was committed.
    again = LabelCorpus(store, B, on_summary=None).take()
    assert again.ids == first.ids

    corpus.commit(first)
    assert LabelCorpus(store, B, on_summary=None).take().ids == ()


def test_committing_the_same_batch_twice_is_a_no_op(store):
    """`Consumed.mark` unions, so a replayed commit cannot double-count.

    Worth asserting here rather than trusting: re-consuming one record re-weights one
    measurement in a corpus of twenty to forty, which is a change to the model nobody
    would see in a log.
    """
    corpus = LabelCorpus(store, B, on_summary=None)
    batch = corpus.take()
    assert corpus.commit(batch) == 9
    assert corpus.commit(batch) == 9


# -- the audit ---------------------------------------------------------------


def test_every_round_reports_what_it_admitted(store):
    """A filter that admits everything and one that admits nothing look identical from a
    trainer's side, and both have happened upstream -- 83% from the shipped defaults, 6%
    from three clauses at their 33rd percentile, because the scores correlate.

    The summary is built by the hook rather than delegated to `percentile_sampler`'s own
    `on_summary`, which does not fire at all when it finds no numeric field to rank on --
    precisely the admits-everything case.
    """
    seen = []
    corpus = LabelCorpus(store, B, fraction=1 / 3, rank_by={"value": "low"},
                         on_summary=seen.append)
    batch = corpus.take()

    assert len(seen) == 1, "exactly one summary per round"
    assert seen[0] == batch.summary
    assert batch.summary["owed"] == 9
    assert batch.summary["selected"] == 3
    assert batch.summary["left_owed"] == 6
    assert batch.summary["admitted"] == pytest.approx(1 / 3)
    assert batch.summary["ranked_by"] == {"value": "low"}


def test_the_unfiltered_round_is_audited_too(store):
    """The admits-everything case has to be as visible as the admits-nothing one."""
    seen = []
    LabelCorpus(store, B, on_summary=seen.append).take()
    assert seen[0]["admitted"] == 1.0
    assert seen[0]["selected"] == seen[0]["owed"] == 9
    assert seen[0]["ranked_by"] is None


def test_an_empty_round_reports_rather_than_dividing_by_zero(tmp_path):
    seen = []
    batch = LabelCorpus(tmp_path, B, on_summary=seen.append).take()
    assert len(batch) == 0
    assert seen[0]["owed"] == 0 and seen[0]["admitted"] == 0.0


# -- the refusals ------------------------------------------------------------


def test_a_subset_without_a_ranking_direction_is_refused(tmp_path):
    """There is no safe default direction, so the hook will not guess one.

    `preorg_rmsd` wants low and a reward wants high. A sampler ranking the wrong way returns
    exactly the records the trainer should have skipped, and nothing downstream can tell:
    the batch is well-formed, the model trains, and the corpus is biased.
    """
    with pytest.raises(ValueError, match="rank_by"):
        LabelCorpus(tmp_path, B, fraction=0.33)

    with pytest.raises(ValueError, match="fraction"):
        LabelCorpus(tmp_path, B, fraction=0.0, rank_by={"value": "low"})


def test_a_batch_cannot_be_committed_against_another_consumer(store):
    """Marking B's batch against C would retire records C never saw."""
    b_batch = LabelCorpus(store, B, on_summary=None).take()
    with pytest.raises(ValueError, match=C):
        LabelCorpus(store, C, on_summary=None).commit(b_batch)


# -- multiplicity ------------------------------------------------------------


def test_two_consumers_keep_their_own_sets(store):
    """B and C read the same store at their own rates, which is why this is a set per
    consumer rather than one mark on the stream."""
    b = LabelCorpus(store, B, on_summary=None)
    c = LabelCorpus(store, C, fraction=1 / 3, rank_by={"value": "low"}, on_summary=None)

    b.commit(b.take())
    assert len(b.take()) == 0, "B has taken everything"

    c_batch = c.take()
    assert len(c_batch) == 3, "C is unaffected by B's progress"
    c.commit(c_batch)
    assert len(c.owed()) == 6, "C is owed the six its sampler skipped"
    assert len(c.take()) == 2, "and its next third of those six is two"
    assert len(b.take()) == 0


# -- the contract between the record and the row -----------------------------


def test_a_row_carries_the_id_as_its_uid_and_the_seq_as_data(store):
    """`uid` is the label id because that is what `Consumed` keys on and what the sampler
    breaks rank ties with. A tie broken on an unstable key reorders the batch between two
    runs over identical data; the store's `seq` is filename order and renumbers."""
    row = record_to_row(7, _record(7, 0.4))
    assert row["uid"] == row["label_id"] == "lab-007"
    assert row["seq"] == 7
    assert row["value"] == 0.4 and row["sigma"] == 0.05
    assert row["kind"] == "md"


def test_the_batch_rows_are_derived_and_the_pairs_are_the_source_of_truth(store):
    """`mark_consumed` takes the pairs `unconsumed` returned, deliberately: *"so a caller
    cannot mark a record it never received."* The flat rows are for the trainer."""
    batch = LabelCorpus(store, B, on_summary=None).take()
    assert isinstance(batch, CorpusBatch)
    assert {rec.label_id for _, rec in batch.selected} == set(batch.ids)
    assert [row["uid"] for row in batch.rows] == list(batch.ids)


# -- CON-4 -------------------------------------------------------------------


def test_importing_the_hook_pulls_in_no_scientific_stack():
    """CON-4, in a cold interpreter because an earlier test here may have imported anything.

    The hook runs inside a Dragon task. A transitive torch import there leaves a CUDA
    context resident in the driver for the whole campaign, and the import that does it is
    never the one anybody wrote on purpose.
    """
    code = (
        "import sys; "
        "import examples.impress.small_molecule_binding.corpus as c; "
        "print(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'torch', 'numpy', 'sklearn', 'scipy', 'pandas'}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"the hook pulled in {out.stdout.strip()}"
