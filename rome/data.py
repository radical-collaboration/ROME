"""Data Manager — collects scored outputs and builds them into a dataset.

The host workflow already produces scored things: completions with rewards,
sequences with pLDDT/pTM/pAE, designs with a simulation score. ROME's data
manager is where those land. Any task, on any node, calls :meth:`DataManager.add`
and the record becomes part of the training corpus; the training manager reads
the corpus back out as a dataset when it is time to train.

Records are stored one key per record (see :mod:`rome.utils`) so concurrent
producers never clobber each other, and the corpus is monotonic: adding is the
only thing the host workflow does. Filtering, deduplication and sampling are
applied on the way *out*, when a dataset is built.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from rome._logging import get_logger
from rome.utils import MODEL_VERSION_KEY, Namespace

_log = get_logger(__name__)

RECORD_NS = "record"
"""Namespace holding one key per accepted record."""

META_NS = "meta"
"""Namespace holding bookkeeping counters (consumed watermark, etc)."""

CONSUMED_KEY = "consumed"
"""Key under :data:`META_NS` counting records already folded into a round."""


@dataclass
class DataConfig:
    """Knobs for how the data manager turns raw scored outputs into a dataset.

    Parameters
    ----------
    min_samples : int
        How many *unconsumed* records must accumulate before the training
        manager considers a round possible. This is the "training starts
        automatically once enough data accumulates" threshold.
    max_records : Optional[int]
        Soft cap on corpus size. When exceeded, the oldest records are evicted
        on the next :meth:`DataManager.add`. ``None`` (the default) keeps
        everything.

        .. warning::

           On Dragon, leave this unset unless you have measured that you need
           it. Eviction is the only thing that pops from the corpus dictionary,
           and a Dragon ``DDict.keys()`` scan **silently returns a truncated
           list** when another client pops concurrently — one measured scan
           returned 39 of 400 keys and reported success. With no evictions the
           corpus is never popped, so scans are exact and the training shard is
           complete. See ``docs/dragon.md``.
    score_key : str
        Field of a record holding its scalar quality score. Used by the
        built-in ``top_k`` sampler and by ``min_score``.
    min_score : Optional[float]
        Reject records scoring below this on the way in. ``None`` accepts all.
    filter_func : Optional[Callable[[dict], bool]]
        Arbitrary admission predicate, applied after ``min_score``. Return
        ``False`` to drop the record. This is where IMPRESS-R plugs in its
        pLDDT/pTM/pAE thresholds.
    dedup_key : Optional[Callable[[dict], Any]]
        Returns a hashable identity for a record. Records whose identity was
        already seen are dropped. ``None`` disables deduplication.
    sample_func : Optional[Callable[[List[dict]], List[dict]]]
        Builds the training shard from the corpus. Overrides ``sampling``.
    sampling : str
        Built-in shard builder: ``'all'`` (default), ``'top_k'`` (best
        ``shard_size`` by ``score_key``) or ``'recent'`` (newest
        ``shard_size``).
    shard_size : Optional[int]
        Number of records the built-in samplers draw. ``None`` means no limit.
    consume_on_train : bool
        When ``True`` (default) records handed to a training round count as
        consumed, so the next round waits for ``min_samples`` *new* records.
        The corpus itself is never deleted.
    """

    min_samples: int = 32
    max_records: Optional[int] = None
    score_key: str = "score"
    min_score: Optional[float] = None
    filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    dedup_key: Optional[Callable[[Dict[str, Any]], Any]] = None
    sample_func: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
    sampling: str = "all"
    shard_size: Optional[int] = None
    consume_on_train: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Extra fields stamped onto every record (run id, campaign name, ...)."""


class DataManager:
    """Gathers scored outputs from the host workflow and creates a dataset.

    The public surface is deliberately tiny — the slide deck's promise is that
    adopting ROME costs a few API calls:

    * :meth:`add` / :meth:`add_batch` — call from anywhere in the workflow
    * :meth:`get_dataset` — what the training task trains on
    * :meth:`ready_to_train` / :meth:`unconsumed_count` — what the training
      manager polls

    Parameters
    ----------
    ddict : Namespace
        Shared state. Every ROME component in a run must be handed the same
        one (``Manager`` does this for you).
    config : DataConfig, optional
        Defaults are used when omitted.
    """

    def __init__(self, ddict: Namespace, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.ddict = ddict
        self._records = ddict.namespace(RECORD_NS)
        self._meta = ddict.namespace(META_NS)
        # Dedup identities seen by *this* process. The authoritative check is
        # the corpus scan in _is_duplicate; this is just a fast path.
        self._seen: set = set()

    # -- writing ------------------------------------------------------------

    def add(
        self,
        score: Optional[float] = None,
        *,
        uid: Optional[str] = None,
        **fields: Any,
    ) -> Optional[str]:
        """Add one scored output to the corpus.

        Everything is keyword-driven so the same call works for any domain::

            rome.data.add(prompt=p, completion=c, score=reward)
            rome.data.add(sequence=seq, pdb_path=pdb, score=plddt, pTM=ptm)

        Parameters
        ----------
        score : float, optional
            The record's scalar quality. Stored under ``config.score_key``.
        uid : str, optional
            Caller-supplied identity. Generated when omitted.
        **fields
            Arbitrary payload, stored verbatim.

        Returns
        -------
        Optional[str]
            The record uid, or ``None`` when the record was filtered out.
        """
        record = dict(self.config.metadata)
        record.update(fields)
        if score is not None:
            record[self.config.score_key] = score
        record["uid"] = uid or uuid.uuid4().hex
        record.setdefault("added_at", time.time())
        record.setdefault("model_version", self.model_version)

        if not self._accepts(record):
            _log.info("rejected design %s (filtered: %s)",
                      record["uid"][:8], self._reject_reason(record))
            return None
        if self._is_duplicate(record):
            _log.info("rejected design %s (duplicate)", record["uid"][:8])
            return None

        self._records[record["uid"]] = record
        self._meta.increment("total")
        self._evict_if_needed()
        _log.info("received design %s%s — corpus %d (%d unconsumed)",
                  record["uid"][:8], self._score_note(record),
                  self.total_count, self.unconsumed_count)
        return record["uid"]

    def add_batch(self, records: Iterable[Dict[str, Any]]) -> List[str]:
        """Add many records at once; returns the uids that were accepted."""
        added = []
        for record in records:
            payload = dict(record)
            uid = payload.pop("uid", None)
            score = payload.pop(self.config.score_key, None)
            result = self.add(score=score, uid=uid, **payload)
            if result is not None:
                added.append(result)
        return added

    def _accepts(self, record: Dict[str, Any]) -> bool:
        cfg = self.config
        if cfg.min_score is not None:
            score = record.get(cfg.score_key)
            if score is None or score < cfg.min_score:
                return False
        if cfg.filter_func is not None and not cfg.filter_func(record):
            return False
        return True

    def _reject_reason(self, record: Dict[str, Any]) -> str:
        """Why ``_accepts`` turned a record away — for the DEBUG log line."""
        cfg = self.config
        if cfg.min_score is not None:
            score = record.get(cfg.score_key)
            if score is None:
                return f"no {cfg.score_key}"
            if score < cfg.min_score:
                return f"{cfg.score_key}={score} < {cfg.min_score}"
        return "filter_func"

    def _score_note(self, record: Dict[str, Any]) -> str:
        """`` (score=…)`` for the INFO line, or empty when unscored."""
        score = record.get(self.config.score_key)
        if score is None:
            return ""
        note = f"{score:g}" if isinstance(score, float) else str(score)
        return f" ({self.config.score_key}={note})"

    def _is_duplicate(self, record: Dict[str, Any]) -> bool:
        if self.config.dedup_key is None:
            return False
        identity = self.config.dedup_key(record)
        if identity in self._seen:
            return True
        for existing in self._records.values():
            if self.config.dedup_key(existing) == identity:
                self._seen.add(identity)
                return True
        self._seen.add(identity)
        return False

    def _evict_if_needed(self) -> None:
        cap = self.config.max_records
        if cap is None:
            return
        records = self.get_records()
        overflow = len(records) - cap
        if overflow <= 0:
            return
        for record in records[:overflow]:
            self._records.pop(record["uid"], None)

    # -- reading ------------------------------------------------------------

    def get_records(self) -> List[Dict[str, Any]]:
        """Every record in the corpus, oldest first."""
        records = [r for r in self._records.values() if isinstance(r, dict)]
        records.sort(key=lambda r: (r.get("added_at", 0.0), str(r.get("uid", ""))))
        return records

    def get_dataset(self, as_hf_dataset: bool = False) -> Any:
        """Build the training dataset from the corpus.

        Applies ``config.sample_func`` when set, otherwise the built-in
        ``config.sampling`` strategy.

        Parameters
        ----------
        as_hf_dataset : bool
            Return a ``datasets.Dataset`` instead of a list of dicts. Requires
            ``datasets`` to be installed; the LLM trainer needs it, the MPNN
            trainer does not, which is why it is opt-in.
        """
        records = self._sample(self.get_records())
        if not as_hf_dataset:
            return records
        from datasets import Dataset

        return Dataset.from_list(records)

    def _sample(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cfg = self.config
        if cfg.sample_func is not None:
            return list(cfg.sample_func(records))
        size = cfg.shard_size
        if cfg.sampling == "all":
            return records if size is None else records[-size:]
        if cfg.sampling == "recent":
            return records if size is None else records[-size:]
        if cfg.sampling == "top_k":
            ranked = sorted(
                records,
                key=lambda r: r.get(cfg.score_key, float("-inf")),
                reverse=True,
            )
            return ranked if size is None else ranked[:size]
        raise ValueError(
            f"unknown sampling strategy {cfg.sampling!r}; "
            "expected 'all', 'recent' or 'top_k', or set sample_func"
        )

    # -- accounting ---------------------------------------------------------

    @property
    def total_count(self) -> int:
        """Number of records currently in the corpus."""
        return len(self._records.keys())

    @property
    def consumed_count(self) -> int:
        """Number of records already folded into a completed training round."""
        return int(self._meta.get(CONSUMED_KEY, 0))

    @property
    def unconsumed_count(self) -> int:
        """Records added since the last training round consumed the corpus."""
        return max(0, self.total_count - self.consumed_count)

    @property
    def model_version(self) -> int:
        """Version of the model currently published by the training manager.

        Stamped onto each record so a later analysis can tell which model
        generated it — the "produced_under_version" idea from IMPRESS-R.
        """
        return int(self.ddict.get(MODEL_VERSION_KEY, 0))

    def ready_to_train(self) -> bool:
        """Whether enough fresh data has accumulated for a training round."""
        return self.unconsumed_count >= self.config.min_samples

    def mark_consumed(self, count: Optional[int] = None) -> int:
        """Record that ``count`` records were used for training.

        Called by the training manager when a round completes. Defaults to the
        whole current corpus. No-op when ``config.consume_on_train`` is off, so
        a workflow that wants to retrain on everything every round can.
        """
        if not self.config.consume_on_train:
            return self.consumed_count
        total = self.total_count if count is None else self.consumed_count + count
        self._meta[CONSUMED_KEY] = min(total, self.total_count)
        return self.consumed_count

    def clear(self) -> int:
        """Drop the entire corpus. Returns the number of records removed."""
        removed = self._records.clear()
        self._meta[CONSUMED_KEY] = 0
        self._meta["total"] = 0
        self._seen.clear()
        return removed

    def __len__(self) -> int:
        return self.total_count

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<DataManager records={self.total_count} "
            f"unconsumed={self.unconsumed_count} "
            f"min_samples={self.config.min_samples}>"
        )
