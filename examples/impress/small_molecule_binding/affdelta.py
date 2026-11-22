"""Model C, `AffDelta`: the driver half, which imports nothing heavy.

C corrects a co-folding model's own affinity prediction rather than replacing it, and reports a
calibrated σ beside the correction. **It is what `dg_lcb` is computed from** — item 14 ranks on
`dg_corrected + k·sigma_dg` under B's risk threshold — so the σ is not decoration: a collapsed
one turns a lower confidence bound into a posterior mean and the ranking silently stops
accounting for uncertainty.

**Two stages, and only the second happens here.** A residual fitted offline on public affinity
data (PDBbind / BindingDB paired with co-folding predictions) is an **input** this round reads;
the 10–20 in-house ATM labels *recalibrate* the GP against that prior rather than fit one from
scratch. 60–150 in-house labels do not fit in a single 4×A100 allocation, which is what forces
the split (§5.5).

**The residual artifact is read, never rewritten — including its PCA basis.** The projection
that reduces a frozen embedding comes with the residual, because re-fitting a projection on ten
points *is* the refit this stage exists to avoid: it would quietly move the basis the prior was
trained in, and the prior would then be evaluated somewhere it never saw. The manifest records
which residual version the round read, so a campaign's `v0` is *"public residual R, zero
in-house labels"* rather than an unlabelled starting point.

**No heavy import here.** `affdelta_train_wrapper.py` owns numpy and the GP (CON-4).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rome.train.base import TrainTask

__all__ = ["DEFAULT_PROPERTY", "AffDeltaConfig", "AffDeltaTrainer"]

#: The in-house label C recalibrates against. ATM alchemical transfer, 10–20 a round.
DEFAULT_PROPERTY = "dg_abfe"
FEATURE_KEY = "embedding"
GROUP_KEY = "scaffold"


@dataclass
class AffDeltaConfig:
    """A round's settings. The residual's own basis is NOT among them — it is read."""

    model_id: str = "aff_delta"
    registry_root: str = "models"
    #: The offline residual: its own artifact, with its own version and its own provenance.
    residual_path: str = ""
    property_name: str = DEFAULT_PROPERTY
    feature_key: str = FEATURE_KEY
    group_key: str = GROUP_KEY
    #: Below this, a round reports insufficient evidence rather than a fitted GP. Five is the
    #: floor the episode declares; 10–20 is what an allocation actually buys.
    min_labels: int = 5
    #: Known per-point observation noise when a label reports none. A GP told the noise is
    #: zero will interpolate every label exactly and report a σ of ~0 between them.
    default_sigma: float = 0.5
    min_effect_size: float = 0.1
    seed: int = 0
    train_script: Optional[str] = None
    train_func: Optional[Callable[[Dict[str, Any]], str]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_labels < 2:
            raise ValueError(
                f"min_labels must be >= 2, got {self.min_labels}: a GP on one point reports "
                "the prior everywhere and a σ that says nothing about the chemotype"
            )
        if self.default_sigma <= 0.0:
            raise ValueError(
                f"default_sigma must be > 0, got {self.default_sigma}: zero observation noise "
                "makes the posterior interpolate every label exactly, and item 14 then ranks "
                "on a lower confidence bound that is a posterior mean"
            )

    def resolved_train_script(self) -> str:
        if self.train_script:
            return self.train_script
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "affdelta_train_wrapper.py")


class AffDeltaTrainer(TrainTask):
    """One recalibration round. Seconds, so C can retrain every cycle."""

    def __init__(self, config: Optional[AffDeltaConfig] = None, **kwargs: Any) -> None:
        self.config = config or AffDeltaConfig()
        kwargs.setdefault("name", self.config.model_id)
        kwargs.setdefault("gpus", 0)
        super().__init__(**kwargs)

    def validate(self, dataset: Any) -> None:
        """Refuse on the manager what cannot recalibrate, before a task is scheduled."""
        super().validate(dataset)
        if not self.config.residual_path:
            raise ValueError(
                f"{self.name}: no `residual_path`. This stage recalibrates a residual fitted "
                "offline on public affinity data; with no prior to correct there is nothing "
                "to recalibrate and the round would be a from-scratch fit on 10-20 points"
            )
        if not os.path.exists(self.config.residual_path):
            raise ValueError(
                f"{self.name}: residual_path does not exist: {self.config.residual_path}"
            )
        records = list(dataset)
        usable = [r for r in records
                  if r.get("property_name") == self.config.property_name
                  and _features_of(r, self.config.feature_key)]
        if not usable:
            raise ValueError(
                f"{self.name}: none of {len(records)} records carries both "
                f"property_name={self.config.property_name!r} and "
                f"metadata[{self.config.feature_key!r}]"
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
        from affdelta_train_wrapper import run_round  # noqa: PLC0415

        return run_round(job)

    def _build_job(self, records: Sequence[Dict[str, Any]], output_dir: str,
                   **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        config = self.config
        examples: List[Dict[str, Any]] = []
        for record in records:
            if record.get("property_name") != config.property_name:
                continue
            features = _features_of(record, config.feature_key)
            if not features:
                continue
            sigma = record.get("sigma")
            examples.append({
                "uid": record.get("uid") or record.get("label_id"),
                "node_id": record.get("node_id"),
                "group": str((record.get("metadata") or {}).get(config.group_key) or "") or None,
                "x": features,
                "y": float(record["value"]),
                # A label that reported no sigma gets the declared default, not zero.
                "noise": float(sigma) if isinstance(sigma, (int, float)) and sigma > 0
                         else float(config.default_sigma),
            })
        job = {
            "model_id": config.model_id,
            "registry_root": os.path.abspath(config.registry_root),
            "residual_path": os.path.abspath(config.residual_path),
            "output_dir": os.path.abspath(output_dir),
            "property_name": config.property_name,
            "min_labels": config.min_labels,
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
