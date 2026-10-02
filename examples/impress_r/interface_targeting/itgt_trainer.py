"""Driver-side ``TrainTask`` for the online interface-target surrogate.

This is the half of the trainer that runs on the ROME manager: it validates the
corpus, writes a job spec, and hands a shell command to the training manager.
It imports **no torch, no scikit-learn and no itgt** -- those live in
``itgt_train_wrapper.py``, which runs as its own process on the allocation. That
split is what lets this module be imported (and read statically by griffe for
the API reference) on a machine with none of them installed, and it is the same
split ``small_molecule_binding/ligandmpnn_trainer.py`` uses.

What is being learned
---------------------

Not ProteinMPNN, and not LigandMPNN. The model here parameterizes **stage 1**,
RFD3 backbone diffusion, by choosing the ``select_buried``/``select_exposed``
partition of the target ligand's heavy atoms -- the RFD3 ``rasa_bin``
conditioning signal, which in every campaign to date has been hand-authored and
frozen.

The default component regresses **action to outcome**:

* **action** -- the partition this workflow wrote into the spec, in reference
  atom names
* **outcome** -- the ``ligand_iptm`` of the pass that action produced

Those two are causally clean: the action is ours, the reward is downstream of
it. The pairs accumulate in an ``itgt.common.corpus.ReplayBuffer``, which is
what itgt built for online training and which dedupes on
``InterfaceTarget.key()`` so re-proposing a scored target updates it rather than
double-counting it as evidence.

.. warning::

   ``component="burial_prior"`` fits itgt's per-atom scorer on
   ``burial_fraction`` labels taken from the designed complexes. **That target
   is self-confirming** and is offered only as a cold-start prior and an
   ablation. The binder buried those atoms *because the previous round's
   ``select_buried`` told RFD3 to bury them*, so the label measures compliance
   with our own conditioning rather than quality; ``ligand_iptm`` never enters
   the objective. Fit it on one ligand and it reduces further, to a per-atom
   mean over admitted designs. Do not read a low ``train_mse`` from it as
   evidence of anything -- ``itgt.method2_hotspots.baseline.train_baseline``
   performs no held-out validation at all.

Why the corpus and the training set differ
------------------------------------------

The dataset ROME hands :meth:`ItgtSurrogateTrainer.as_command` is the *fresh
delta* -- it exists to trigger rounds once ``min_samples`` new designs land, and
to be logged. The regression fits on the **whole** replay buffer, because
shrinking a reward regression to the best third of the corpus throws away
exactly the dynamic range it needs. Failed folds belong in it too: the pipeline
sets ``best_fold_model`` on any fold that produced a confidence file, not only
on passing ones, so negatives come free and are informative about which
partitions not to propose.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rome.train.base import TrainTask

#: Record fields a round cannot run without.
REQUIRED_FIELDS: Tuple[str, ...] = ("path",)

#: The reward field the surrogate regresses on.
REWARD_FIELD = "ligand_iptm"

#: Components this trainer can fit.
COMPONENTS: Tuple[str, ...] = ("reward_ridge", "burial_prior", "trunk")

#: Default ranking for the corpus sampler, when one is used at all.
DEFAULT_RANK_BY: Dict[str, str] = {"ligand_iptm": "high", "plddt": "high"}

__all__ = [
    "ItgtSurrogateConfig",
    "ItgtSurrogateTrainer",
    "itgt_reward_fn",
    "itgt_corpus_filter",
    "percentile_sampler",
    "COMPONENTS",
    "REQUIRED_FIELDS",
    "REWARD_FIELD",
    "DEFAULT_RANK_BY",
]


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class ItgtSurrogateConfig:
    """How the surrogate is fit, and where its inputs live.

    ``itgt_dir`` is the ``interface-targeting`` checkout. There are **no trained
    itgt artifacts anywhere in that repo** -- every weight is written to a path
    the caller chooses -- so the first round of a campaign always trains from
    scratch, and a missing ``itgt_dir`` must degrade to a no-op rather than a
    crash.
    """

    itgt_dir: Optional[str] = None
    component: str = "reward_ridge"

    #: Replay buffer of ``(InterfaceTarget, metrics)`` pairs -- the real
    #: training set for ``reward_ridge``. Written by the runner at each pass's
    #: fold step, where the action and its reward are both in hand.
    replay_buffer: Optional[str] = None

    #: Reference structure and ligand key the partition is expressed against.
    #: Resolved from the pipeline's own base spec, so the atom names match what
    #: RFD3 is handed.
    reference_pdb: Optional[str] = None
    ligand_key: Optional[str] = None

    #: Selection geometry. Both bounds default to ``None`` so the proposer
    #: derives them from the template's own selection size at runtime, banded by
    #: ``size_band``. Pinning them here overrides that. The band exists because
    #: mean-pooled features have a small-patch bias -- the average chemistry of
    #: four well-chosen atoms beats that of fourteen -- so an unbanded argmax
    #: collapses onto the floor, and a 4-atom buried patch is not an interface.
    min_atoms: Optional[int] = None
    max_atoms: Optional[int] = None
    size_band: int = 4

    #: Whether the rim of the selected patch becomes its own ``partial`` bin.
    #: **Science-critical, not cosmetic.** At 14 selected atoms on a 30-atom
    #: ligand with a 5 A contact radius, the fringe covers essentially the
    #: remainder, ``exposed`` empties, ``burial_groups`` omits it, and RFD3 is
    #: silently told "nothing on this ligand is exposed" -- a radically
    #: different conditioning signal from the template's 16 exposed atoms.
    #: False keeps the two-bin shape the hand-authored spec uses.
    mark_fringe_partial: bool = False

    #: Candidate partitions drawn per proposal (method-1 random surface patches).
    n_candidates: int = 64

    #: Ridge regularisation. The corpus is O(10)-O(100) pairs at ~52 features,
    #: so this is doing real work, not decoration.
    ridge_alpha: float = 1.0

    #: ``burial_prior``/``trunk`` only.
    label: str = "burial_fraction"
    epochs: int = 60
    device: str = "cpu"
    seed: int = 0

    #: Override the wrapper script; defaults to the sibling file.
    train_script: Optional[str] = None
    #: In-process escape hatch, for tests.
    train_func: Optional[Callable[..., str]] = None

    def validate(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(
                f"component must be one of {COMPONENTS}, got {self.component!r}"
            )
        if self.itgt_dir and not os.path.isdir(self.itgt_dir):
            raise ValueError(f"itgt_dir does not exist: {self.itgt_dir}")
        if self.min_atoms is not None and self.min_atoms <= 0:
            raise ValueError(f"min_atoms must be positive, got {self.min_atoms}")
        if (
            self.min_atoms is not None
            and self.max_atoms is not None
            and self.min_atoms > self.max_atoms
        ):
            raise ValueError(
                f"need min_atoms <= max_atoms, got {self.min_atoms}/{self.max_atoms}"
            )
        if self.size_band < 0:
            raise ValueError(f"size_band must be >= 0, got {self.size_band}")
        if self.component == "reward_ridge" and not self.replay_buffer:
            raise ValueError("component='reward_ridge' needs replay_buffer set")
        if self.component == "reward_ridge" and not self.reference_pdb:
            raise ValueError("component='reward_ridge' needs reference_pdb set")

    def resolved_train_script(self) -> str:
        """Path of the compute-side wrapper."""
        if self.train_script:
            return self.train_script
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "itgt_train_wrapper.py")

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("train_func", None)
        return out


# ── The task ──────────────────────────────────────────────────────────────────


class ItgtSurrogateTrainer(TrainTask):
    """Fit the interface-target surrogate for one round.

    Runs as an **executable task** by default (:meth:`as_command`), so the
    round's process exits when it ends and releases its memory. :meth:`train`
    stays as the in-process path for tests and for ``train_func``.
    """

    wants_hf_dataset = False

    def __init__(
        self,
        config: Optional[ItgtSurrogateConfig] = None,
        *,
        gpus: Optional[int] = None,
        nodes: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self.config = config or ItgtSurrogateConfig()
        self.config.validate()
        # The default component is a ridge over ~52 features: CPU, seconds, no
        # GPU to ask for. `trunk` is a 3-layer MPNN at hidden 96 and is still
        # comfortable on CPU, so a GPU is opt-in rather than assumed.
        super().__init__(
            gpus=0 if gpus is None else gpus,
            nodes=1 if nodes is None else nodes,
            name=name or "itgt_surrogate",
        )

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, dataset: Any) -> None:
        """Reject a corpus that cannot train, before a task is scheduled.

        Runs on the manager, so an unusable corpus fails immediately instead of
        after a queue wait.
        """
        super().validate(dataset)
        records = list(dataset)
        missing = [f for f in REQUIRED_FIELDS if not any(f in r for r in records)]
        if missing:
            have = sorted({k for r in records for k in r})
            raise ValueError(
                f"{self.name} needs field(s) {missing}; the corpus has {have}. "
                "Add them via add_training_data(path=..., ligand_iptm=...)."
            )
        if self.config.component == "reward_ridge":
            if not _any_reward(records):
                raise ValueError(
                    f"{self.name} regresses on {REWARD_FIELD!r} and no record has "
                    f"a numeric one, so the round would fit nothing. Check that the "
                    f"fold step is reporting best_ligand_iptm."
                )
            buf = self.config.replay_buffer
            if buf and not os.path.exists(buf):
                raise ValueError(
                    f"replay buffer not found: {buf}. The runner seeds it with the "
                    "template's own partition at startup, so its absence means the "
                    "v0 action was never recorded."
                )

    # ── job spec ──────────────────────────────────────────────────────────────

    def write_manifest(self, records: Sequence[Dict[str, Any]], output_dir: str) -> str:
        """Write the round's corpus delta as JSON Lines, for audit.

        JSON Lines rather than the sibling's parquet: this corpus is O(10)
        records of scalars per round, and a pandas/pyarrow import on the driver
        buys nothing here.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "corpus_delta.jsonl")
        with open(path, "w") as fh:
            for record in records:
                fh.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        return path

    def _build_job(
        self, records: Sequence[Dict[str, Any]], output_dir: str, **kwargs: Any
    ) -> Tuple[Dict[str, Any], str]:
        """Assemble the job dict and the artifact directory it will write."""
        os.makedirs(output_dir, exist_ok=True)
        artifact = os.path.join(output_dir, "artifact")
        job: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "artifact": artifact,
            "output_dir": output_dir,
            "manifest": self.write_manifest(records, output_dir),
            "model_version": kwargs.get("model_version"),
            "report": os.path.join(output_dir, "report.json"),
            "error_log": os.path.join(output_dir, "train_error.log"),
            "designs": [
                {
                    "uid": r.get("uid", ""),
                    "path": r.get("path"),
                    REWARD_FIELD: _as_float(r.get(REWARD_FIELD)),
                    "plddt": _as_float(r.get("plddt")),
                    "action_uid": r.get("action_uid", ""),
                }
                for r in records
            ],
        }
        reward_fn = kwargs.get("reward_fn")
        if reward_fn is not None:
            # cloudpickle when available: a reward function defined in the
            # runner's __main__ is not picklable by the stdlib pickler, and the
            # wrapper is a separate process.
            try:
                import cloudpickle as _cp

                dumps = _cp.dumps
            except ImportError:
                dumps = pickle.dumps
            job["reward_fn_pickle"] = base64.b64encode(dumps(reward_fn)).decode()
        return job, artifact

    def as_command(
        self, dataset: Any, output_dir: str, **kwargs: Any
    ) -> Optional[Tuple[str, str]]:
        """Run the round as its own process.

        Returns ``(command, artifact_dir)``. The artifact path has to be
        declared because an executable task's return value is the backend's
        execution result, not a path -- and the command must write
        ``train_complete`` into ``output_dir`` as its final action, which is how
        the training manager detects completion when Dragon fails to deliver a
        task's result.
        """
        if self.config.train_func is not None:
            return None
        records = list(dataset)
        job, artifact = self._build_job(records, output_dir, **kwargs)
        job_path = os.path.join(output_dir, "job.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=2)
        script = self.config.resolved_train_script()
        return f"{sys.executable} {script} --job {job_path}", artifact

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """In-process round: the ``train_func`` hook, or the wrapper imported."""
        records = list(dataset)
        job, artifact = self._build_job(records, output_dir, **kwargs)
        if self.config.train_func is not None:
            return self.config.train_func(job) or artifact

        # Imported rather than subprocessed, so this path is usable from a test
        # without a shell. The wrapper owns every heavy import.
        sys.path.insert(0, os.path.dirname(self.config.resolved_train_script()))
        from itgt_train_wrapper import run_round  # noqa: PLC0415

        return run_round(job)


# ── Reward, admission and sampling ────────────────────────────────────────────


def itgt_reward_fn(record: Dict[str, Any]) -> float:
    """Per-design reward: the fold stage's ``ligand_iptm``.

    Interface confidence, not whole-complex confidence -- the question this
    surrogate optimizes is whether the *interface* came out right, and pLDDT can
    be high for a confidently-folded binder that misses the ligand entirely.

    ``ligand_iptm`` is absent whenever the fold step produced no confidence file
    and ``None`` whenever the confidence JSON carried no such key, so this
    returns ``0.0`` for both rather than raising. A record scoring ``0.0`` is
    still usable evidence for the regression: it says that partition produced
    nothing measurable.

    Override with ``ROME_REWARD_FN=your_module:your_fn``.
    """
    value = record.get(REWARD_FIELD)
    return float(value) if _is_number(value) else 0.0


def itgt_corpus_filter(
    min_ligand_iptm: Optional[float] = None,
    require_reward: bool = False,
) -> Callable[[Dict[str, Any]], bool]:
    """Admission gate for the surrogate's corpus.

    Both knobs default **off**, which is deliberate and the opposite of the
    sibling's filter. A reward regression needs its failures: filtering to good
    designs truncates the response variable and leaves the model unable to tell
    a bad partition from an unobserved one. Set ``min_ligand_iptm`` only to drop
    records you believe are measurement artefacts, not merely poor results.
    """

    def _passes(record: Dict[str, Any]) -> bool:
        value = record.get(REWARD_FIELD)
        if require_reward and not _is_number(value):
            return False
        if min_ligand_iptm is not None:
            return _is_number(value) and float(value) >= min_ligand_iptm
        return True

    return _passes


def percentile_sampler(
    fraction: float = 1.0,
    *,
    rank_by: Optional[Dict[str, str]] = None,
    min_shard: int = 1,
    on_summary: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Select the best ``fraction`` of the corpus by average rank.

    Non-parametric: ``rank_by`` maps field names to ``'high'`` or ``'low'`` and
    ranks are averaged across fields, so each contributes equally regardless of
    scale. Mirrors the sibling's sampler, with one difference that matters --
    ``fraction`` defaults to ``1.0`` here. Selecting the best third of a reward
    regression's own training data is a bias, not a filter; the surrogate wants
    the whole observed range. Lower it only for ``burial_prior``, where the
    sampler is the only place the reward enters at all.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    directions = dict(rank_by if rank_by is not None else DEFAULT_RANK_BY)
    for key, direction in directions.items():
        if direction not in ("high", "low"):
            raise ValueError(f"rank_by[{key!r}] must be 'high' or 'low', got {direction!r}")

    def _sample(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = list(records)
        if not records:
            return records
        usable = {
            key: direction
            for key, direction in directions.items()
            if any(_is_number(r.get(key)) for r in records)
        }
        if not usable:
            return records
        total = {id(r): 0.0 for r in records}
        for key, direction in usable.items():
            ordered = sorted(
                records,
                key=lambda r: (
                    not _is_number(r.get(key)),
                    -float(r[key])
                    if _is_number(r.get(key)) and direction == "high"
                    else (float(r[key]) if _is_number(r.get(key)) else 0.0),
                ),
            )
            for position, record in enumerate(ordered):
                total[id(record)] += position
        ranked = sorted(records, key=lambda r: (total[id(r)], str(r.get("uid", ""))))
        keep = max(min(min_shard, len(records)), int(round(len(records) * fraction)))
        selected = ranked[:keep]
        if on_summary is not None:
            cutoffs = {}
            for key, direction in usable.items():
                values = [float(r[key]) for r in selected if _is_number(r.get(key))]
                if values:
                    cutoffs[key] = min(values) if direction == "high" else max(values)
            on_summary(
                {
                    "corpus": len(records),
                    "selected": len(selected),
                    "ranked_by": usable,
                    "cutoffs": cutoffs,
                }
            )
        return selected

    return _sample


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _as_float(value: Any) -> Optional[float]:
    return float(value) if _is_number(value) else None


def _any_reward(records: Sequence[Dict[str, Any]]) -> bool:
    return any(_is_number(r.get(REWARD_FIELD)) for r in records)


def _jsonable(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop anything that will not survive ``json.dumps``."""
    out: Dict[str, Any] = {}
    for key, value in record.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            out[key] = repr(value)
        else:
            out[key] = value
    return out
