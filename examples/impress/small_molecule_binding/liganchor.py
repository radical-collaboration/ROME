"""Model A, `LigAnchor`: the driver half, which imports nothing heavy.

A scores a ligand's heavy atoms for how worth anchoring they are, and its **online reward
head** is what turns a campaign's own results back into that score. The head is the part
this module schedules.

**It regresses action to outcome, and that is the whole point.** The target is what happened
*downstream* of an anchor set -- a physics label, an assay, B's or C's reading of the design
the anchor set produced -- never a property of the anchor set itself. Regressing on the
artifact's own properties *"would only measure compliance with our previous instruction"*,
which is the central lesson of the interface-targeting precedent this trainer adapts. The
refusal is mechanical rather than documentary: `ANCHOR_OWN_METRICS` names what the
anchor-emitting tool computes from its own output, and a target that appears as a key inside
the action payload is refused by the same rule without anybody having to list it.

**Round 1 banks and does not steer** (source spec 7.3). With `head_enabled=False` a round
appends its pairs to the replay buffer and publishes a version with no held-out metrics,
which the promotion gate reads as INSUFFICIENT_EVIDENCE -- so a banking round cannot become
the promoted pointer, and the head switches on at round 2 as a separate arm against the
round-1 baseline. That is a mechanism, not a convention: nothing has to remember not to
promote it.

**The replay buffer is the one thing here that genuinely resumes (R8).** It is a path, not a
directory under this round's output, because a buffer that lives in the round's output dies
with the round. Each round re-fits from the *whole* buffer; `buffer_path` is required for
exactly that reason.

**No heavy import here.** `liganchor_train_wrapper.py` owns torch and `itgt` (CON-4).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rome.train.base import TrainTask

__all__ = ["ANCHOR_OWN_METRICS", "DEFAULT_PROPERTY", "LigAnchorConfig", "LigAnchorTrainer"]

#: The downstream outcome an anchor set is judged by. A label, not a property of the anchor.
DEFAULT_PROPERTY = "dg_abfe"
ACTION_KEY = "action"
GROUP_KEY = "scaffold"

#: What the anchor-emitting tool computes **from the anchor set itself**, measured from
#: `impress_a.tools.asteer_design_agents`'s `ligand_anchor_score` metrics block. A head
#: regressed on one of these learns to reproduce the selection rule that produced it.
ANCHOR_OWN_METRICS = frozenset({"anchor_entropy", "n_engaged_atoms"})

_SELF_REGRESSION = (
    "regressing on the artifact's own properties would only measure compliance with our "
    "previous instruction; model A's target must be a downstream outcome"
)


@dataclass
class LigAnchorConfig:
    """A round's settings. The frozen encoder and the buffer are read, not produced here."""

    model_id: str = "lig_anchor"
    registry_root: str = "models"
    #: The pretrained geometric encoder, frozen. Trained on contact labels, read from a path;
    #: this round fits the pooling and the head over it and never updates it.
    encoder_path: str = ""
    #: The replay buffer, outside any round's output directory. Required: see the module
    #: docstring on R8.
    buffer_path: str = ""
    property_name: str = DEFAULT_PROPERTY
    #: +1 if a larger outcome is better, -1 if smaller is (dG is -1). The head ranks, so it
    #: needs to be told which way, and a wrong sign trains a head to prefer the worst designs.
    direction: int = -1
    action_key: str = ACTION_KEY
    group_key: str = GROUP_KEY
    #: False banks the round's pairs and publishes nothing steerable. Round 2 flips it.
    head_enabled: bool = False
    #: Below this many resolved pairs a round reports insufficient evidence.
    min_pairs: int = 8
    min_effect_size: float = 0.1
    #: `itgt.method2_hotspots.train.TrainConfig`'s own default. Overriding an upstream
    #: default with a number nobody measured is how a hyperparameter becomes folklore.
    epochs: int = 60
    seed: int = 0
    scorer: str = ""
    train_script: Optional[str] = None
    train_func: Optional[Callable[[Dict[str, Any]], str]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.property_name in ANCHOR_OWN_METRICS:
            raise ValueError(
                f"property_name={self.property_name!r} is a metric the anchor tool computes "
                f"from its own anchor set: {_SELF_REGRESSION}"
            )
        if self.direction not in (-1, 1):
            raise ValueError(
                f"direction must be -1 or +1, got {self.direction}: a ranking head has to be "
                "told which way is better, and an unsigned one ranks by coincidence"
            )
        if self.min_pairs < 2:
            raise ValueError(
                f"min_pairs must be >= 2, got {self.min_pairs}: a ranking loss needs a pair, "
                "and one bag has nothing to be ranked against"
            )

    def resolved_scorer(self) -> str:
        """The scorer name recorded on the artifact.

        `itgt` reserves `"proxy"` for its synthetic labels and refuses to compare a
        proxy-fit artifact against a campaign-scored one. A campaign round must therefore
        name itself, and the default names the property it actually regressed.
        """
        return self.scorer or f"asd:{self.property_name}"

    def resolved_train_script(self) -> str:
        if self.train_script:
            return self.train_script
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "liganchor_train_wrapper.py")


class LigAnchorTrainer(TrainTask):
    """One reward-head round: bank the pairs, and fit the whole buffer if the head is on."""

    def __init__(self, config: Optional[LigAnchorConfig] = None, **kwargs: Any) -> None:
        self.config = config or LigAnchorConfig()
        kwargs.setdefault("name", self.config.model_id)
        kwargs.setdefault("gpus", 0)
        super().__init__(**kwargs)

    def validate(self, dataset: Any) -> None:
        """Refuse on the manager what cannot train, before a task is scheduled."""
        super().validate(dataset)
        config = self.config
        if not config.buffer_path:
            raise ValueError(
                f"{self.name}: no `buffer_path`. The replay buffer is re-fitted whole every "
                "round (R8), so it cannot live under a round's output directory -- a buffer "
                "that dies with the round makes round 2 a round 1 with a longer history"
            )
        if config.head_enabled and not config.encoder_path:
            raise ValueError(
                f"{self.name}: head_enabled with no `encoder_path`. The head is fitted over a "
                "frozen pretrained encoder; without one this round would train the geometry "
                "too, on the handful of campaign labels the head itself is short of"
            )
        if config.head_enabled and not os.path.exists(config.encoder_path):
            raise ValueError(
                f"{self.name}: encoder_path does not exist: {config.encoder_path}"
            )
        records = list(dataset)
        usable = [r for r in records
                  if r.get("property_name") == config.property_name
                  and _action_of(r, config.action_key)]
        if not usable:
            raise ValueError(
                f"{self.name}: none of {len(records)} records carries both "
                f"property_name={config.property_name!r} and "
                f"metadata[{config.action_key!r}] -- a record with no action is an outcome "
                "with nothing to attribute it to, and model A regresses the attribution"
            )
        for record in usable:
            action = _action_of(record, config.action_key)
            if config.property_name in action:
                raise ValueError(
                    f"{self.name}: property_name={config.property_name!r} is also a key of "
                    f"the action payload on {record.get('node_id')!r}: {_SELF_REGRESSION}"
                )

    def as_command(self, dataset: Any, output_dir: str,
                   **kwargs: Any) -> Optional[Tuple[str, str]]:
        if self.config.train_func is not None:
            return None
        job, artifact = self._build_job(list(dataset), output_dir, **kwargs)
        job_path = os.path.join(output_dir, "job.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=2)
        return f"{sys.executable} {self.config.resolved_train_script()} --job {job_path}", artifact

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        job, artifact = self._build_job(list(dataset), output_dir, **kwargs)
        if self.config.train_func is not None:
            return self.config.train_func(job) or artifact
        sys.path.insert(0, os.path.dirname(self.config.resolved_train_script()))
        from liganchor_train_wrapper import run_round  # noqa: PLC0415

        return run_round(job)

    def _build_job(self, records: Sequence[Dict[str, Any]], output_dir: str,
                   **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        config = self.config
        pairs: List[Dict[str, Any]] = []
        for record in records:
            if record.get("property_name") != config.property_name:
                continue
            action = _action_of(record, config.action_key)
            if not action:
                continue
            if config.property_name in action:
                raise ValueError(
                    f"{self.name}: property_name={config.property_name!r} is also a key of "
                    f"the action payload on {record.get('node_id')!r}: {_SELF_REGRESSION}"
                )
            metadata = record.get("metadata") or {}
            node_id = record.get("node_id")
            pairs.append({
                # One anchor set per design node is the natural identity; a campaign that
                # proposes several says so by carrying its own id.
                "action_id": str(metadata.get("action_id") or node_id),
                "node_id": node_id,
                "outcome_label_id": record.get("label_id"),
                # Cross-ligand quality is not comparable, so the group is what the head
                # z-scores and ranks within. The structure is the fallback key.
                "group": str(metadata.get(config.group_key)
                             or action.get("structure_path") or "") or None,
                "action": action,
                "outcome_value": float(record["value"]),
                "proposed_at": str(metadata.get("proposed_at") or record.get("produced_at") or ""),
            })
        job = {
            "model_id": config.model_id,
            "registry_root": os.path.abspath(config.registry_root),
            "buffer_path": os.path.abspath(config.buffer_path),
            "encoder_path": os.path.abspath(config.encoder_path) if config.encoder_path else "",
            "output_dir": os.path.abspath(output_dir),
            "property_name": config.property_name,
            "direction": int(config.direction),
            "head_enabled": bool(config.head_enabled),
            "min_pairs": config.min_pairs,
            "min_effect_size": config.min_effect_size,
            "epochs": config.epochs,
            "seed": config.seed,
            "scorer": config.resolved_scorer(),
            "notes": config.notes,
            "corpus_size": len(records),
            "pairs": pairs,
            "parent_version": kwargs.get("parent_version"),
            "git_sha": kwargs.get("git_sha", ""),
        }
        job.update(config.extra)
        return job, os.path.abspath(output_dir)


def _action_of(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    raw = (record.get("metadata") or {}).get(key)
    return dict(raw) if isinstance(raw, dict) and raw else {}
