"""Model B, `PocketPreorg`: the driver half, which imports nothing heavy.

B predicts apo-state preorganisation from a designed holo complex -- apo→holo pocket Cα RMSD,
rotamer entropy, ligand RMSF -- each with an uncertainty, and it is the **FK potential**, so it
is the highest-throughput component in the system: called once per particle per resampling
event, roughly 45 times per trajectory under Tier 2. A five-member deep ensemble with a
heteroscedastic Gaussian NLL, over 20-40 samples a round.

**This module holds no heavy import and that is CON-4, not style.** `preorg_train_wrapper.py`
owns torch and the ensemble; this side builds a job description and hands it over. A driver that
imports torch keeps a CUDA context resident in the campaign process for the whole run.

**Publishing is not promoting.** A finished round writes `v{N}` through `asteer.registry` with
its held-out metrics and `promoted=False`. Only the controller flips the pointer (CON-6), and
the controller is phase III. A trainer that promoted its own model would be grading its own
homework with the registry as the mark sheet.

**`logo_r2` is reported even when it is negative**, because negative is the finding: it means
predicting the campaign's mean beats the model, which is the number that catches a
self-confirming objective. And the **group count is recorded beside it**, because a five-member
ensemble on 20-40 samples with a 20% holdout is judged on roughly 4-8 points, where `logo_r2`
is near-meaningless below about four groups (CON-13). A manifest with too few groups says
*insufficient evidence*; it does not say *pass*.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rome.train.base import TrainTask

__all__ = [
    "DEFAULT_PROPERTY",
    "PocketPreorgConfig",
    "PocketPreorgTrainer",
]

#: The label B is trained against. One property per round: the three B predicts are three
#: regressions sharing a corpus, not one multi-output head, so a round names which it is.
DEFAULT_PROPERTY = "preorg_rmsd"

#: Where a record carries its feature vector and its scaffold cluster. Both live under the
#: `LabelRecord.metadata` dict rather than in the contract, because the embedding comes from a
#: co-folding model and the cluster from impress-a's planner -- neither is a property of the
#: measurement, which is what `LabelRecord` is.
FEATURE_KEY = "embedding"
GROUP_KEY = "scaffold"


@dataclass
class PocketPreorgConfig:
    """Everything a round needs that is not the corpus.

    `min_holdout_n` and `min_effect_size` are the version's declared floors and ride on the
    manifest, because they are what the promotion gate compares against and a floor kept
    anywhere else can be edited after the fact (CON-13).
    """

    model_id: str = "pocket_preorg"
    registry_root: str = "models"
    property_name: str = DEFAULT_PROPERTY
    feature_key: str = FEATURE_KEY
    group_key: str = GROUP_KEY
    n_members: int = 5
    hidden: Tuple[int, ...] = (64, 64)
    epochs: int = 200
    learning_rate: float = 1e-2
    holdout_fraction: float = 0.2
    min_holdout_n: int = 4
    min_effect_size: float = 0.1
    seed: int = 0
    #: Path to the wrapper. Resolved beside this file, so a checkout works with no env var.
    train_script: Optional[str] = None
    #: In-process hook, as `itgt` does it: a test drives a whole round without a shell.
    train_func: Optional[Callable[[Dict[str, Any]], str]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_members < 2:
            raise ValueError(
                f"n_members must be >= 2, got {self.n_members}: the ensemble spread IS the "
                "epistemic term, and one member has no spread -- a single net would report a "
                "confident sigma from the aleatoric head alone, which is the uncertainty this "
                "model exists to avoid reporting"
            )
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError(
                f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}"
            )

    def resolved_train_script(self) -> str:
        if self.train_script:
            return self.train_script
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "preorg_train_wrapper.py")


class PocketPreorgTrainer(TrainTask):
    """One round of B, as an executable task by default.

    The command form is preferred for the same reason the base class gives: the subprocess
    exits when the round ends, releasing its VRAM. The in-process form exists so a test can
    drive a round without a shell, and so a laptop run needs no scheduler.
    """

    def __init__(self, config: Optional[PocketPreorgConfig] = None, **kwargs: Any) -> None:
        self.config = config or PocketPreorgConfig()
        kwargs.setdefault("name", self.config.model_id)
        super().__init__(**kwargs)

    # -- what the manager calls -------------------------------------------

    def validate(self, dataset: Any) -> None:
        """Refuse a corpus that cannot train, on the manager, before a task is scheduled.

        The base class refuses an empty dataset. This adds the two that look like data and
        are not: a corpus where no record carries the property this round regresses, and one
        where no record carries a feature vector. Either would train a model that predicts a
        constant and report it as a model.
        """
        super().validate(dataset)
        records = list(dataset)
        with_target = [r for r in records if r.get("property_name") == self.config.property_name]
        if not with_target:
            raise ValueError(
                f"{self.name}: no record in a corpus of {len(records)} carries "
                f"property_name={self.config.property_name!r}. A round with no targets would "
                "fit nothing and publish a version saying so in metrics nobody reads"
            )
        featured = [r for r in with_target if _features_of(r, self.config.feature_key)]
        if not featured:
            raise ValueError(
                f"{self.name}: {len(with_target)} records carry the target and none carries "
                f"metadata[{self.config.feature_key!r}]. Training on an empty feature set "
                "produces a model that predicts the corpus mean and reports a plausible NLL "
                "for doing it"
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
        """In-process round: the `train_func` hook, or the wrapper imported.

        Imported rather than subprocessed so this path is usable from a test without a shell.
        The wrapper owns every heavy import, so importing it here is also where torch would
        enter -- which is why it happens inside this method and not at module scope.
        """
        job, artifact = self._build_job(list(dataset), output_dir, **kwargs)
        if self.config.train_func is not None:
            return self.config.train_func(job) or artifact
        sys.path.insert(0, os.path.dirname(self.config.resolved_train_script()))
        from preorg_train_wrapper import run_round  # noqa: PLC0415

        return run_round(job)

    # -- the job ------------------------------------------------------------

    def _build_job(self, records: Sequence[Dict[str, Any]], output_dir: str,
                   **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        """Everything the wrapper needs, as JSON, and nothing it can get wrong silently.

        The examples are extracted HERE rather than in the wrapper, so the driver's `validate`
        and the wrapper's input are the same selection. Two selections would be two chances
        for a round to train on a different corpus than the one that was checked.
        """
        config = self.config
        examples: List[Dict[str, Any]] = []
        for record in records:
            if record.get("property_name") != config.property_name:
                continue
            features = _features_of(record, config.feature_key)
            if not features:
                continue
            examples.append({
                "uid": record.get("uid") or record.get("label_id"),
                "node_id": record.get("node_id"),
                "group": _group_of(record, config.group_key),
                "x": features,
                "y": float(record["value"]),
                "sigma": record.get("sigma"),
            })
        job = {
            "model_id": config.model_id,
            "registry_root": os.path.abspath(config.registry_root),
            "output_dir": os.path.abspath(output_dir),
            "property_name": config.property_name,
            "n_members": config.n_members,
            "hidden": list(config.hidden),
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "holdout_fraction": config.holdout_fraction,
            "min_holdout_n": config.min_holdout_n,
            "min_effect_size": config.min_effect_size,
            "seed": config.seed,
            "notes": config.notes,
            "corpus_size": len(records),
            "examples": examples,
            "parent_version": kwargs.get("parent_version"),
            "git_sha": kwargs.get("git_sha", ""),
        }
        job.update(config.extra)
        return job, os.path.abspath(output_dir)


def _features_of(record: Dict[str, Any], key: str) -> List[float]:
    raw = (record.get("metadata") or {}).get(key)
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    try:
        return [float(v) for v in raw]
    except (TypeError, ValueError):
        return []


def _group_of(record: Dict[str, Any], key: str) -> Optional[str]:
    """The scaffold cluster, or `None` -- never a stand-in.

    `LabelRecord` has no scaffold field: the cluster is a property of the design, which lives
    in impress-a's tree, so it can only arrive through `metadata`. When it does not, this
    returns `None` and the round reports **insufficient evidence** rather than inventing a
    grouping. Treating every ungrouped record as one cluster puts them all on one side of the
    boundary; treating each as its own makes a random split wearing a cluster's name, which is
    the single most common way a surrogate's reported accuracy is wrong and is invisible in
    the numbers.
    """
    value = (record.get("metadata") or {}).get(key)
    return str(value) if value else None
