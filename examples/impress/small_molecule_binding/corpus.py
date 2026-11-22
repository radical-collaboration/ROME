"""The corpus hook for the ASD surrogates: owed labels in, a training batch out.

Three trainers (A `LigAnchor`, B `PocketPreorg`, C `AffDelta`) read one label store at three
different rates. This module is the only place that decides what each of them has not yet
seen, and the only place that records what it took.

**It takes the corpus from `asteer.labels`, not from `DataManager`'s watermark, and that is a
decision rather than a preference.** ROME's consumption model is a prefix high-water mark,
which can say *"taken everything up to n"* and nothing else. The selection policy here is
rank-based, and **a sampler selects a non-contiguous subset**: advance a mark past a sampled
batch and every record the sampler skipped is retired having never been trained on; refuse to
advance it and the next round re-selects the same records. There is no correct mark value for
a sampled batch, which makes a mark the wrong *structure*, not a coarser one.
`asteer.labels.Consumed` keeps a set of ids per consumer instead, and that is what this reads.

`DataManager` is untouched and is correct for a prefix consumer. This project is not one.

**Taking and committing are two calls, because a trainer can die between them.** `take()`
reads and selects; `commit()` marks. A round that crashes after `take()` leaves every record
still owed, which is the behaviour a restart needs -- 20 to 40 physics labels is a corpus
where losing one is a measurable loss, and `Consumed.mark()` only ever unions, so a replayed
commit is harmless.

**Every round reports what it admitted, on every path.** The upstream filter's defaults
admitted 83% of a real campaign -- IMPRESS's own keep/drop thresholds applied downstream of
themselves -- and three clauses each at their 33rd percentile admit 6%, not 33%, because the
scores correlate. From a trainer's side a filter that silently admits everything and one that
silently admits nothing look identical, so the summary is not optional and is not delegated:
`percentile_sampler`'s own `on_summary` does not fire when it finds no numeric field to rank
on, which is precisely the silent-everything case. The summary here is built by this module
and always present.

**`fraction` defaults to 1.0, and the episode plan's `0.33` is declined with a reason.** That
number came from filtering *designs*, where the corpus is large and self-generated. These
records are measured physics: C's GP is *recalibrated* by 10-20 ATM labels rather than fitted,
and B's MD labels cost GPU-hours each. Selecting the best third of a regression's own targets
is a bias, not a filter -- the sibling sampler says so in its own docstring, which is why its
`fraction` also defaults to 1.0 and why it tells the reader to lower it only where the sampler
is the only place a reward enters at all. Each trainer passes its own fraction; this module
refuses to guess one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from asteer.labels import LabelQueue

__all__ = [
    "CorpusBatch",
    "LabelCorpus",
    "log_summary",
    "record_to_row",
]

LOG = logging.getLogger(__name__)


def log_summary(summary: Dict[str, Any]) -> None:
    """The default audit sink: one line per round, at INFO.

    A print would be wrong inside a Dragon task and a silent default would be worse than
    either, so the default is the logger and the summary also rides on the batch.
    """
    LOG.info(
        "corpus round for %s: %d owed, %d selected, %d left owed, ranked_by=%s",
        summary.get("consumer"), summary.get("owed"), summary.get("selected"),
        summary.get("left_owed"), summary.get("ranked_by"),
    )


def record_to_row(seq: int, record: Any) -> Dict[str, Any]:
    """One `LabelRecord` as a flat dict the sampler and a trainer can both read.

    `uid` is the label id because that is what `Consumed` keys on and what the sampler uses
    to break rank ties -- a tie broken on an unstable key would reorder the batch between two
    runs over identical data.

    `seq` rides along for ordering and for debugging, and is deliberately NOT the identity:
    the store's sequence numbers are filename order and a set of ids survives a renumbering.
    """
    return {
        "uid": record.label_id,
        "seq": seq,
        "label_id": record.label_id,
        "kind": getattr(record.kind, "value", str(record.kind)),
        "design_campaign_id": record.design_campaign_id,
        "node_id": record.node_id,
        "property_name": record.property_name,
        "value": float(record.value),
        "units": record.units,
        "sigma": None if record.sigma is None else float(record.sigma),
        "replicates": int(record.replicates),
        "protocol": record.protocol,
        "produced_at": record.produced_at,
        "source_job_id": record.source_job_id,
        "metadata": dict(record.metadata),
    }


@dataclass(frozen=True)
class CorpusBatch:
    """One round's selection, and everything needed to commit or to audit it.

    `selected` holds the `(seq, LabelRecord)` pairs exactly as `LabelQueue.unconsumed()`
    returned them, because `mark_consumed()` takes that shape on purpose -- *"so a caller
    cannot mark a record it never received"*. `rows` is the same thing flattened for a
    trainer and for the sampler; it is derived, never the source of truth for a commit.
    """

    consumer: str
    selected: tuple = ()
    #: What this round saw and did NOT select. Stays owed; nothing marks it.
    left_owed: tuple = ()
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> tuple:
        return tuple(record_to_row(seq, rec) for seq, rec in self.selected)

    @property
    def ids(self) -> tuple:
        return tuple(rec.label_id for _, rec in self.selected)

    def __len__(self) -> int:
        return len(self.selected)


class LabelCorpus:
    """One consumer's view of one label store: what it is owed, and what it has taken."""

    def __init__(
        self,
        root: str | Path,
        consumer: str,
        *,
        fraction: float = 1.0,
        rank_by: Optional[Dict[str, str]] = None,
        min_shard: int = 1,
        on_summary: Optional[Callable[[Dict[str, Any]], None]] = log_summary,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if fraction < 1.0 and not rank_by:
            # The direction is per-property and there is no safe default: `preorg_rmsd` wants
            # low and a reward wants high, and a sampler ranking the wrong way returns exactly
            # the records the trainer should not have. Refusing is cheaper than discovering it
            # in a model.
            raise ValueError(
                f"{consumer}: fraction={fraction} selects a subset, so `rank_by` must say "
                "which field to rank on and in which direction ({'value': 'low'} or 'high'). "
                "There is no safe default: the direction is a property of the label, not of "
                "the corpus, and ranking the wrong way returns the records the trainer "
                "should have skipped."
            )
        self.queue = LabelQueue(root)
        self.consumer = consumer
        self.fraction = fraction
        self.rank_by = dict(rank_by) if rank_by else None
        self.min_shard = min_shard
        self.on_summary = on_summary

    # -- read ---------------------------------------------------------------

    def owed(self) -> List[Any]:
        """Every `(seq, LabelRecord)` this consumer has not taken, oldest first."""
        return list(self.queue.unconsumed(self.consumer))

    def take(self) -> CorpusBatch:
        """Select this round's batch. Marks nothing -- see `commit`."""
        pairs = self.owed()
        by_uid = {rec.label_id: (seq, rec) for seq, rec in pairs}
        rows = [record_to_row(seq, rec) for seq, rec in pairs]
        kept = rows if self.fraction >= 1.0 else self._sample(rows)
        chosen = {row["uid"] for row in kept}
        selected = [by_uid[uid] for uid in (row["uid"] for row in kept)]
        summary = {
            "consumer": self.consumer,
            "owed": len(pairs),
            "selected": len(selected),
            "left_owed": len(pairs) - len(selected),
            "fraction": self.fraction,
            "ranked_by": self.rank_by,
            # Built here, not taken from the sampler: `percentile_sampler` skips its own
            # `on_summary` when no field it ranks on carries a number, which is the
            # admits-everything case this audit exists to make visible.
            "admitted": (len(selected) / len(pairs)) if pairs else 0.0,
        }
        if self.on_summary is not None:
            self.on_summary(summary)
        return CorpusBatch(
            consumer=self.consumer,
            selected=tuple(selected),
            left_owed=tuple((seq, rec) for seq, rec in pairs
                            if rec.label_id not in chosen),
            summary=summary,
        )

    def _sample(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Imported here rather than at module scope for two reasons: the sibling module pulls
        # in `rome`, and this hook is otherwise `asteer` plus the standard library, so a
        # reader on a machine with neither can still import it; and it keeps the CON-4
        # surface of the module to what a cold interpreter actually loads.
        from examples.impress_r.interface_targeting.itgt_trainer import (  # noqa: PLC0415
            percentile_sampler,
        )

        # One implementation of the ranking rule, not a second one that agrees today. The
        # fields are ours; the arithmetic is the sibling's and is tested there.
        sampler = percentile_sampler(
            self.fraction, rank_by=self.rank_by, min_shard=self.min_shard,
        )
        return list(sampler(rows))

    # -- commit -------------------------------------------------------------

    def commit(self, batch: CorpusBatch) -> int:
        """Mark exactly this batch's ids as taken. Returns the consumer's new total.

        Separate from `take()` so a round that fails between them leaves every record owed.
        `Consumed.mark` unions, so committing the same batch twice is a no-op rather than a
        double count.
        """
        if batch.consumer != self.consumer:
            raise ValueError(
                f"batch belongs to {batch.consumer!r}, not {self.consumer!r}: marking it here "
                "would retire another trainer's records against this consumer's set"
            )
        return self.queue.mark_consumed(self.consumer, list(batch.selected))
