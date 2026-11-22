"""Model A's reward-head round: bank the pairs, re-fit the whole buffer, report what held out.

**The corpus is the only thing this had to build.** `itgt/method2_hotspots/train.py` says so
in its own module docstring -- *"When milestone 1d lands, `train_head` changes not at all --
only the corpus handed to it does"* -- and this project's campaign is that milestone's label
source. So `itgt_head` assembles an `itgt.common.corpus.Corpus` out of banked
(anchor set -> outcome) pairs and calls `train_head`. It does not reimplement the ranking
loss, the within-group z-scoring, the annealed attention auxiliary or the MIL pooling; a
second copy of any of them is the divergence CON-8 exists to prevent.

**The scorer name is a guard, not a label.** `itgt` reserves `"proxy"` for its synthetic
quality function and `evaluate.compare_to_baseline` refuses to compare a proxy-fit artifact
against a campaign-scored one. A round here records the campaign's own scorer, so the two
cannot be silently mixed in a table.

**One gap the adaptation does not close, named rather than papered over.** The specification
asks for an online reward head over a **frozen** pretrained encoder. `train_head` builds a
`MILHead`, whose encoder it trains end to end; there is no hook to load pretrained encoder
weights or to freeze them. itgt's own answer to the same problem is the annealed auxiliary
loss that pins attention to the geometric contact labels -- but it is annealed *to zero*, so
after annealing the encoder is unconstrained, and the two are not equivalent. `encoder_path`
is therefore read for its `EncoderConfig` and its feature columns, which is a real check
(a head built on a reordered feature list still runs and still returns finite numbers), and
the manifest records `encoder=trained_end_to_end` rather than claiming a freeze that did not
happen. The hook belongs in `itgt`, by the same placement rule that put every adapter in the
repo that breaks without it.

**The buffer is append-only and folded on read.** A round appends; a read folds by
`action_id` with last-wins, so an outcome arriving for an action banked unresolved supersedes
it without rewriting history. A torn final line -- the only corruption an append can produce
-- is skipped rather than fatal, because a round that cannot read its buffer at all has lost
every previous round.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

TRAIN_COMPLETE_MARKER = "train_complete"

__all__ = ["ReplayBuffer", "run_round", "main", "evaluate", "itgt_head", "load_state",
           "STATE_NAME", "TRAIN_COMPLETE_MARKER"]

#: Where a published version keeps its head, as an `itgt` artifact directory rather than a
#: bare state dict. `itgt.common.artifacts` records the feature columns a model was fit on
#: and **raises** at load time when the caller would feed it a different list -- a head
#: loaded against reordered columns still runs and still returns finite numbers, so that
#: check is the whole reason to use their format rather than `torch.save`.
STATE_NAME = "head"

#: Quality per bag, in the order the bags were handed over. Comparable within a group only.
Scorer = Callable[[Sequence[Dict[str, Any]]], List[float]]
Fitter = Callable[[List[Dict[str, Any]], Dict[str, Any]], Scorer]


# -- the thing that resumes --------------------------------------------------


class ReplayBuffer:
    """Every (anchor set -> outcome) pair this campaign has banked, as one JSONL file.

    R8's one genuine resume. The head is re-fitted from the whole buffer each round rather
    than updated incrementally: at 8-40 pairs an incremental update is dominated by which
    order the pairs arrived in, and a refit of this size costs seconds.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def append(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Append rows verbatim. Returns how many are new identities, for the audit line."""
        if not rows:
            return 0
        known = {row["action_id"] for row in self.load()}
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return len({row["action_id"] for row in rows} - known)

    def load(self) -> List[Dict[str, Any]]:
        """The folded buffer: one row per `action_id`, last write wins, order preserved."""
        if not os.path.exists(self.path):
            return []
        folded: Dict[str, Dict[str, Any]] = {}
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line is what an interrupted append leaves behind. Losing
                    # one pair is a loss; refusing to read the other thirty-nine is a worse
                    # one, and the next round re-banks it anyway.
                    continue
                if isinstance(row, dict) and row.get("action_id"):
                    folded[str(row["action_id"])] = row
        return list(folded.values())


# -- the fit (the only heavy import in this example's A half) ----------------


def itgt_head(train: List[Dict[str, Any]], job: Dict[str, Any]) -> Scorer:
    """`itgt`'s gated-attention MIL head, fitted on the campaign's own pairs.

    The two things this hands over are the corpus and the scoring config. The composite
    names the **outcome** property with its direction, so the quality `train_head` ranks is
    the downstream result rather than anything read off the anchor set.
    """
    import torch  # noqa: PLC0415
    from itgt.common.nn.batch import collate_graphs  # noqa: PLC0415
    from itgt.common.scoring.base import CompositeConfig  # noqa: PLC0415
    from itgt.method2_hotspots import data as m2data  # noqa: PLC0415
    from itgt.method2_hotspots.train import TrainConfig, train_head  # noqa: PLC0415

    prop = str(job["property_name"])
    scoring = CompositeConfig(weights={prop: 1.0},
                              directions={prop: int(job["direction"])},
                              normalize="zscore")
    dataset_config = _dataset_config(job)
    train_config = TrainConfig(epochs=int(job["epochs"]), seed=int(job["seed"]), device="cpu")
    # `out` is `train_head`'s own artifact write, which records the scorer, the scoring
    # config, the feature columns and the corpus fingerprint. Reconstructing that meta here
    # would be a second copy of itgt's provenance, drifting one field at a time. Only the
    # final all-data fit names an `out`; the leave-one-group-out folds do not.
    model, _ = train_head(_corpus(train, job), scoring, str(job["scorer"]),
                          dataset_config=dataset_config, train_config=train_config,
                          out=job.get("artifact_out") or None)
    model.eval()

    def score(rows: Sequence[Dict[str, Any]]) -> List[float]:
        bags, report = m2data.bags_from_corpus(_corpus(list(rows), job), scoring, dataset_config)
        if len(bags) != len(rows):
            raise ValueError(
                f"{len(rows)} held-out pairs produced {len(bags)} bags: {report}. A silently "
                "dropped bag would shift every later score onto the wrong pair"
            )
        with torch.no_grad():
            tensors = collate_graphs(bags)
            quality, _ = model(
                tensors["x"], tensors["edge_index"], tensors["edge_attr"],
                tensors["mask"].to(torch.float32), tensors["batch"], tensors["n_graphs"],
            )
        return [float(v) for v in quality]

    return score


def _dataset_config(job: Dict[str, Any]):
    """The featuriser the frozen encoder was pretrained under, where one is named.

    A head built on a reordered feature list still runs and still returns finite numbers,
    which is why `itgt.common.artifacts` checks columns at load time and why this reads the
    encoder artifact rather than assuming the defaults still match it.
    """
    from itgt.common.datasets import DatasetConfig  # noqa: PLC0415

    path = str(job.get("encoder_path") or "")
    if not path:
        return DatasetConfig()
    from itgt.common import artifacts  # noqa: PLC0415

    meta = artifacts.read_meta(path)
    config = DatasetConfig()
    if meta.feature_config:
        config.features = type(config.features)(**{
            k: v for k, v in meta.feature_config.items()
            if k in type(config.features).__dataclass_fields__
        })
    return config


def _corpus(rows: Sequence[Dict[str, Any]], job: Dict[str, Any]):
    """Banked pairs as an `itgt` corpus: one record per action, metrics = the outcome."""
    from itgt.common.corpus import Corpus, CorpusRecord  # noqa: PLC0415
    from itgt.common.selection import InterfaceTarget  # noqa: PLC0415

    prop = str(job["property_name"])
    records = []
    for row in rows:
        action = row["action"]
        records.append(CorpusRecord(
            target=InterfaceTarget(
                structure_path=str(action["structure_path"]),
                ligand_key=str(action["resname"]),
                ligand_atoms=tuple(action["engaged_atoms"]),
                name=str(row.get("node_id") or row["action_id"]),
            ),
            metrics={prop: float(row["outcome_value"])},
            source=str(row.get("outcome_label_id") or ""),
            group=str(row.get("group") or ""),
            uid=str(row["action_id"]),
        ))
    return Corpus(records)


def load_state(version_dir: str, manifest: Any = None):
    """A version directory in, a bag scorer out, through `itgt`'s own loader.

    Raises when the directory holds no head, which `streams.load_promoted` turns into an
    `Unavailable` naming the directory.
    """
    path = os.path.join(version_dir, STATE_NAME)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no {STATE_NAME}/ in {version_dir}: this version was published with a manifest "
            "and no head, so there is nothing to serve"
        )
    from itgt.common import artifacts  # noqa: PLC0415

    # The cheapest check first, and before any weights are read. `itgt` reserves "proxy" for
    # its synthetic quality function and `evaluate.compare_to_baseline` refuses to compare a
    # proxy-fit artifact against a campaign-scored one; serving one into a campaign would
    # route around that refusal entirely.
    meta = artifacts.read_meta(path)
    if meta.is_proxy:
        raise ValueError(
            f"{path} was fit on itgt's proxy labels (scorer={meta.scorer!r}); a campaign "
            "must not be steered by a model that never saw a campaign label"
        )
    from itgt.method2_hotspots import infer  # noqa: PLC0415

    model, _ = infer.load(path)
    return model


# -- the evaluation ----------------------------------------------------------


def _zscore(values: Sequence[float]) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    return [0.0 if sd <= 0 else (v - mean) / sd for v in values]


def evaluate(pairs: List[Dict[str, Any]], job: Dict[str, Any],
             fit: Fitter) -> Dict[str, Any]:
    """Leave-one-group-out over the whole buffer, scored as ranking first and R² second.

    Groups, not pairs, are the split unit: two anchor sets on the same ligand are not
    independent evidence, and a random split over them reports a number about memorisation.
    A group of one is held out of nothing -- there is no within-group pair to rank -- so it
    trains and never tests, which is `itgt.common.corpus.leave_one_group_out`'s own rule.
    """
    report: Dict[str, Any] = {
        "nll": None, "calibration": None, "logo_r2": None, "holdout_n": 0,
        "rank_accuracy": None, "n_pairs": len(pairs), "n_groups": 0,
        "validated": False, "reason": "",
    }
    resolved = [p for p in pairs if p.get("outcome_value") is not None]
    report["n_pairs"] = len(resolved)
    if len(resolved) < int(job["min_pairs"]):
        report["reason"] = (
            f"{len(resolved)} resolved pairs, below the declared floor of {job['min_pairs']}. "
            "A ranking head fitted here would rank on whichever pair it saw first."
        )
        return report

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for pair in resolved:
        by_group.setdefault(str(pair.get("group") or ""), []).append(pair)
    report["n_groups"] = len(by_group)
    foldable = [g for g, rows in by_group.items() if len(rows) >= 2]
    if len(foldable) < 1 or len(by_group) < 2:
        report["reason"] = (
            f"{len(by_group)} group(s), {len(foldable)} of them with a within-group pair: "
            "leave-one-group-out has no fold here. Reporting a random split instead would "
            "leak, which is what grouping exists to prevent."
        )
        return report

    correct = total = 0
    predictions: List[float] = []
    truths: List[float] = []
    for held in foldable:
        rest = [p for g, rows in by_group.items() if g != held for p in rows]
        if not rest:
            continue
        rows = by_group[held]
        scores = fit(rest, job)(rows)
        truth = _zscore([int(job["direction"]) * float(p["outcome_value"]) for p in rows])
        predictions.extend(scores)
        truths.extend(truth)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if truth[i] == truth[j]:
                    continue
                total += 1
                correct += int((scores[i] - scores[j]) * (truth[i] - truth[j]) > 0)

    report["holdout_n"] = len(truths)
    report["rank_accuracy"] = (correct / total) if total else None
    if len(truths) >= 2:
        bar = statistics.fmean(truths)
        denominator = sum((t - bar) ** 2 for t in truths)
        if denominator > 0:
            report["logo_r2"] = 1.0 - sum(
                (t - p) ** 2 for t, p in zip(truths, predictions)) / denominator
    report["validated"] = report["logo_r2"] is not None
    if not report["validated"]:
        report["reason"] = (
            "every held-out outcome in the fold was identical, so there is no variance to "
            "explain and an R² would be a division by zero dressed as a result"
        )
    return report


# -- the round ---------------------------------------------------------------


def run_round(job: Dict[str, Any], *, fit: Optional[Fitter] = None) -> str:
    started = time.process_time()
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Banking is the round's durable job and happens first: a fit that crashes must not cost
    # the campaign the pairs it was handed.
    buffer = ReplayBuffer(job["buffer_path"])
    banked = buffer.append(job.get("pairs") or [])
    pairs = buffer.load()

    weights = "none"
    if job.get("head_enabled"):
        fitter = fit or itgt_head
        report = evaluate(pairs, job, fitter)
        if report["validated"]:
            # The evaluated model and the published model are two different fits: the folds
            # measure, the final fit on every banked pair is what gets served.
            version_dir = _version_dir(job)
            os.makedirs(version_dir, exist_ok=True)
            fitter([p for p in pairs if p.get("outcome_value") is not None],
                   {**job, "artifact_out": os.path.join(version_dir, STATE_NAME)})
            if os.path.isdir(os.path.join(version_dir, STATE_NAME)):
                weights = STATE_NAME + "/"
    else:
        report = {
            "nll": None, "calibration": None, "logo_r2": None, "holdout_n": 0,
            "rank_accuracy": None, "n_pairs": len(pairs), "n_groups": 0, "validated": False,
            # Round 1 (source spec 7.3). No held-out metrics means the promotion gate reads
            # INSUFFICIENT_EVIDENCE, so this version cannot become the promoted pointer and
            # cannot steer -- which is the mechanism, not a convention anyone has to keep.
            "reason": "head disabled: this round banks pairs and publishes nothing steerable",
        }
    report["weights"] = weights
    elapsed = time.process_time() - started

    version = _publish(job, report, elapsed)
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump({**report, "version": version, "train_seconds_cpu": elapsed,
                   "head_enabled": bool(job.get("head_enabled")),
                   "banked_this_round": banked, "buffer_size": len(pairs)}, fh, indent=2)
    with open(os.path.join(output_dir, TRAIN_COMPLETE_MARKER), "w") as fh:
        fh.write(str(version))
    return output_dir


def _publish(job: Dict[str, Any], report: Dict[str, Any], elapsed: float) -> int:
    import datetime  # noqa: PLC0415

    from asteer.contracts import ModelManifest  # noqa: PLC0415
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    registry = ModelRegistry(job["registry_root"])
    version = _next_version(job)
    accuracy = report.get("rank_accuracy")
    registry.publish(ModelManifest(
        model_id=job["model_id"],
        version=version,
        trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corpus_size=int(job["corpus_size"]),
        git_sha=str(job.get("git_sha") or ""),
        parent_version=job.get("parent_version"),
        nll=report["nll"],
        calibration=report["calibration"],
        logo_r2=report["logo_r2"],
        holdout_n=report["holdout_n"],
        min_holdout_n=int(job["min_pairs"]),
        min_effect_size=float(job["min_effect_size"]),
        train_seconds_cpu=elapsed,
        promoted=False,
        notes=" ".join(filter(None, [
            str(job.get("notes") or ""),
            f"head={'enabled' if job.get('head_enabled') else 'disabled'}",
            f"scorer={job['scorer']}",
            f"pairs={report['n_pairs']}",
            f"groups={report['n_groups']}",
            # Not "frozen": see the module docstring on the one gap this adaptation leaves.
            "encoder=trained_end_to_end" if job.get("head_enabled") else "",
            # What a stream will find in this version's directory.
            f"weights={report.get('weights', 'none')}",
            "" if accuracy is None else f"rank_accuracy={accuracy:.3f}",
            report["reason"],
        ])).strip(),
    ))
    return version


def _registry(job: Dict[str, Any]):
    from asteer.registry import ModelRegistry  # noqa: PLC0415

    return ModelRegistry(job["registry_root"])


def _next_version(job: Dict[str, Any]) -> int:
    existing = _registry(job).versions(job["model_id"])
    return (max(existing) + 1) if existing else 0


def _version_dir(job: Dict[str, Any]) -> str:
    return str(_registry(job).version_dir(job["model_id"], _next_version(job)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job) as fh:
        run_round(json.load(fh))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
