"""Phase II's entry point: three ASD surrogates, trained from the label store and served.

A sibling of ``impress_r/interface_targeting/run_interface_targeting_rome.py``, and it is
deliberately a *different kind* of runner. That one drives an IMPRESS campaign and learns one
stage's parameters from the campaign's own outputs. This one drives **nothing**: it is the
surrogate half of the system, and the thing it consumes -- measured physics labels -- is
produced by a design campaign running beside it, through ``asteer.labels_workflow``.

So the loop here has no pipelines and no ``adaptive_decision``. It has a pump::

    LabelQueue  ->  LabelCorpus.take()  ->  Manager.add_training_batch()  ->  commit()
                         (per consumer)        (ROME fires a round at min_samples)

and the three things that pump feeds: three trainers, three model registries, three streams.

Four decisions this file makes, each of which is a decision rather than a default
-------------------------------------------------------------------------------

**One engine, three managers (CON-1).** ``rome.Manager`` holds one trainer, and there are
three models, so there are three managers. Every one of them is handed *this* engine, which
is why ``rome/manager.py:205`` -- the branch that builds an engine when nobody handed one
over -- never runs here. It is also the only place a resource policy can attach, which is the
next decision.

Three managers means three shared dictionaries rather than one, and that is forced:
``manager.py:140`` namespaces every manager under a fixed ``rome|`` prefix, so three managers
over one dictionary would collide on ``model_path`` and ``model_version`` -- B's published
checkpoint would arrive as C's.

**The application declares a quantity; the middleware chooses the device.** No line in this
file names a GPU, a host or a node. Each trainer declares ``gpus`` -- a count -- and the
demand handed to the policy is *derived* from that count (:func:`_demand`), so the number is
written once. Placement is the middleware's: ``ResourcePolicy`` in ``node`` mode picks the
node and the devices and writes them into the task description as ``placement``.

**The node inventory has one source, and it is the backend.** ``NodeShape(cpus=…, gpus=…)``
hand-written in a runner is the same fact as the submit script's ``--gpus-per-node`` and
AuCaM's ``capacity``, stated a third time with nothing checking the three agree -- and a pool
claiming devices the job did not get does not fail, it places onto devices that are not
there. So :func:`node_specs` reads ``backend.topology()``, which rhapsody fills from the
Dragon runtime (``backends/execution/dragon.py:326``), and **narrows the GPU count by
``CUDA_VISIBLE_DEVICES``** because that probe is NVML's and NVML does not honour it. A
backend that reports no topology gets **no policy**, with the reason printed: inventing one
is the failure this is avoiding.

**One stream serves; two refuse by name.** ``streams.RESIDENT_FROM`` already says B is
resident from M2 and A and C from M4, and the reason shows up here as a missing
``Predict``: B has ``preorg_train_wrapper.preorg_predictor`` (features in, mean and sigma
out), while A's head scores *bags* comparably within a group only, and C's posterior lives on
the residual's PCA basis and has to be composed with the prior it corrects. Neither has a
single-request envelope yet. So all three streams are configured and A's and C's ``build``
*raises a named refusal*, which ``streams.load_promoted`` turns into an ``Unavailable`` the
client reports as ``StreamUnavailable``. It is a mechanism and not a comment: it cannot fire
until somebody promotes a version for A or C, which is the exact moment a guess would
otherwise be served.

What this file is NOT
---------------------

It is not a measurement of the placement chain. ``radical.asyncflow`` 0.7.0 -- the released
line every sibling repository pins -- has no ``policies`` module and no ``policy`` keyword on
``WorkflowEngine.create``; both live on the unmerged ``feature/policy-resource``. So
:func:`resource_policy` reports which of the two is missing and the engine is built without
one, which is the released middleware's own behaviour: every ready task goes to the backend
at once and the backend chooses the node. Running against the branch is EII.6's job.

It also does not place the streams, and cannot. A stream is an asyncflow *service*
(``rome/stream.py:166``) and services bypass the policy (``workflow_manager.py:1856-1859``),
so a resident stream's demand is read by nobody -- backlog item 58, stated here rather than
worked around, because the workaround costs a ``dragon.infrastructure.policy`` import in this
process and this process is the one CON-4 keeps clean.

Usage::

    dragon -s run_asd_surrogates_rome.py          # on an allocation
    ASD_RUN_SECONDS=60 dragon -s run_asd_surrogates_rome.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# The example's own modules, importable either as a package from ROME's root or bare from
# this directory -- the sibling runner's pattern, for the same reason: an allocation runs
# `dragon -s run_asd_surrogates_rome.py` from inside this directory.
try:
    from examples.impress.small_molecule_binding import streams
    from examples.impress.small_molecule_binding.affdelta import (
        AffDeltaConfig, AffDeltaTrainer,
    )
    from examples.impress.small_molecule_binding.corpus import LabelCorpus
    from examples.impress.small_molecule_binding.liganchor import (
        LigAnchorConfig, LigAnchorTrainer,
    )
    from examples.impress.small_molecule_binding.preorg import (
        PocketPreorgConfig, PocketPreorgTrainer,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the allocation, not by tests
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import streams  # noqa: F401
    from affdelta import AffDeltaConfig, AffDeltaTrainer
    from corpus import LabelCorpus
    from liganchor import LigAnchorConfig, LigAnchorTrainer
    from preorg import PocketPreorgConfig, PocketPreorgTrainer

import rome

#: Item 36's naming contract, read rather than restated. A `LabelRecord` carries the
#: *property* a measurement supersedes (`dg_corrected`), not the metric the tool reported
#: (`dg_abfe`). Reading the mapping here means the runner cannot be a fourth place the
#: correspondence is written -- and it stays right even though J6 has since fixed the two
#: drivers that defaulted to the metric, because passing the property explicitly is what a
#: runner should do whatever the defaults are. See backlog item 66.
from asteer.labels_workflow import PROPERTY_FOR_METRIC  # noqa: E402

ABFE_PROPERTY = PROPERTY_FOR_METRIC["dg_abfe"][0]
PREORG_PROPERTY = PROPERTY_FOR_METRIC["preorg_rmsd"][0]

__all__ = [
    "Settings", "Surrogate",
    "node_specs", "resource_policy", "visible_gpus",
    "anchor_build", "affdelta_build", "preorg_build",
    "build_surrogates", "pump", "asd_surrogates_rome", "main",
]


# ── settings ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """Everything the run needs that is not the allocation. One source: the environment."""

    output_dir: str
    label_root: str
    registry_root: str
    checkpoint_root: str
    #: Residual artifact C reads and never rewrites; its PCA basis comes with it.
    residual_path: str = ""
    #: A's frozen encoder, and A's replay buffer, which must outlive a round (R8).
    encoder_path: str = ""
    buffer_path: str = ""
    min_samples: Dict[str, int] = field(default_factory=dict)
    trainer_gpus: Dict[str, int] = field(default_factory=dict)
    #: A stream task's GPUs. Zero by default because the featuriser Model B is pointed at is
    #: still a declared stand-in of width 2
    #: (`asteer.features.FEATURISERS["standin-1"].width`) and a five-member MLP over two
    #: inputs does not want a device. A real featuriser changes this number, not this comment.
    stream_gpus: int = 0
    poll_interval: float = 5.0
    max_rounds: Optional[int] = None
    run_seconds: float = 0.0
    #: Touch this file to wind the service down. A resident surrogate service needs a way to
    #: stop that is not SIGKILL, because SIGKILL loses an in-flight round's output directory.
    stop_file: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        here = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(
            os.environ.get("ASD_OUTPUT_DIR", os.path.join(here, "campaign_outputs"))
        )
        return cls(
            output_dir=output_dir,
            label_root=os.environ.get("ASD_LABEL_ROOT", os.path.join(output_dir, "labels")),
            registry_root=os.environ.get("ASD_REGISTRY_ROOT",
                                         os.path.join(output_dir, "models")),
            checkpoint_root=os.environ.get("ASD_CHECKPOINT_ROOT",
                                           os.path.join(output_dir, "rome_checkpoints")),
            residual_path=os.environ.get("ASD_RESIDUAL", ""),
            encoder_path=os.environ.get("ASD_ENCODER", ""),
            buffer_path=os.environ.get("ASD_BUFFER",
                                       os.path.join(output_dir, "liganchor_replay.jsonl")),
            min_samples={
                streams.MODEL_B: int(os.environ.get("ASD_MIN_SAMPLES_B", 20)),
                streams.MODEL_C: int(os.environ.get("ASD_MIN_SAMPLES_C", 5)),
                streams.MODEL_A: int(os.environ.get("ASD_MIN_SAMPLES_A", 8)),
            },
            trainer_gpus={
                streams.MODEL_B: int(os.environ.get("ASD_TRAIN_GPUS_B", 1)),
                # A GP recalibrated by ten to twenty points is seconds of CPU; asking for a
                # device would hold one for the whole round and use none of it.
                streams.MODEL_C: int(os.environ.get("ASD_TRAIN_GPUS_C", 0)),
                streams.MODEL_A: int(os.environ.get("ASD_TRAIN_GPUS_A", 1)),
            },
            stream_gpus=int(os.environ.get("ASD_STREAM_GPUS", 0)),
            poll_interval=float(os.environ.get("ASD_POLL_INTERVAL", 5.0)),
            max_rounds=(int(os.environ["ASD_MAX_ROUNDS"])
                        if os.environ.get("ASD_MAX_ROUNDS") else None),
            run_seconds=float(os.environ.get("ASD_RUN_SECONDS", 0.0)),
            stop_file=os.environ.get("ASD_STOP_FILE",
                                     os.path.join(output_dir, "STOP")),
        )


# ── the node inventory, derived once, from the allocation ─────────────────────


def visible_gpus(probed: int, visible: Optional[str] = None) -> int:
    """Narrow a probed GPU count by ``CUDA_VISIBLE_DEVICES``.

    rhapsody's topology fills ``gpus`` from a pynvml probe of the local node, and **NVML
    counts the devices the machine has, not the devices this job was given**: it does not
    honour ``CUDA_VISIBLE_DEVICES``. A pool built from the raw count on a shared node claims
    devices the job did not ask for, and the symptom is not an admission failure -- it is a
    task placed onto a device that is not there.

    An empty variable means *no devices*, which is CUDA's own reading of it, and an unset one
    means *no restriction*. A malformed entry is ignored rather than guessed at, and cannot
    widen the count.
    """
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return int(probed)
    allowed = [token for token in visible.split(",") if token.strip() != ""]
    return min(int(probed), len(allowed))


def node_specs(backend: Any) -> List[Dict[str, Any]]:
    """The allocation's nodes as ``{"name", "cpus", "gpus"}``, or ``[]`` when unknown.

    Two things worth knowing about what comes back, both from rhapsody's own docstring:

    * ``cores`` is **the Dragon pool's workers per node**, not the node's physical cores. For
      admission that is the right unit -- a task occupies a worker -- but it is not the
      number the submit script asks SLURM for, and the two differ by construction (Dragon's
      pool is ``max(1, num_cpus // 2)``, lowered to the cores the job's affinity allows).
    * ``gpus`` is a probe of the *local* node, assuming homogeneity. :func:`visible_gpus`
      bounds it; nothing can make it per-node without a per-node probe.

    ``[]`` is "topology unknown" and is returned unchanged rather than filled in, because
    every way of filling it in restates the inventory somewhere else.
    """
    try:
        topology = backend.topology()
    except Exception as exc:  # noqa: BLE001 -- an unreportable topology is not a crash
        print(f"[ASD] backend.topology() raised {type(exc).__name__}: {exc}", flush=True)
        return []
    specs: List[Dict[str, Any]] = []
    for node in topology or []:
        specs.append({
            "name": node.get("id") or None,
            "cpus": float(node.get("cores") or 0),
            "gpus": visible_gpus(int(node.get("gpus") or 0)),
        })
    return [spec for spec in specs if spec["cpus"] > 0]


def resource_policy(backend: Any, *, mode: str = "node") -> Tuple[Any, str]:
    """``(policy, why)``: a ``ResourcePolicy`` over the allocation, or ``None`` and the reason.

    Three reasons it comes back ``None``, and each is reported rather than inferred:

    1. ``radical.asyncflow.policies`` does not exist -- the released 0.7.0 line.
    2. It exists but ``WorkflowEngine.create`` takes no ``policy``; passing it would be a
       ``TypeError`` several frames from the cause.
    3. The backend reports no topology, so there is no inventory to build a pool from.

    Without a policy the engine behaves as the released middleware does: every ready task
    goes to the backend at once and the backend picks the node. That is a legitimate run --
    it is what every campaign to date has done -- and it is not the same run, so it is said
    out loud.
    """
    try:
        from radical.asyncflow.policies import (  # noqa: PLC0415
            NodeShape, Pool, ResourcePolicy,
        )
    except ImportError:
        return None, ("radical.asyncflow has no `policies` module: the released 0.7.0 line "
                      "carries no ready-queue policy. ResourcePolicy is on the unmerged "
                      "feature/policy-resource branch; set ASYNCFLOW_REF to measure it")
    import inspect  # noqa: PLC0415

    from radical.asyncflow import WorkflowEngine  # noqa: PLC0415
    if "policy" not in inspect.signature(WorkflowEngine.create).parameters:
        return None, ("radical.asyncflow has a `policies` module and "
                      "WorkflowEngine.create takes no `policy`: this install is between the "
                      "two, and passing one would raise TypeError inside the factory")

    specs = node_specs(backend)
    if not specs:
        return None, (f"{type(backend).__name__}.topology() reported no nodes, so there is "
                      "no inventory to build a Pool from. A hand-written NodeShape here "
                      "would be the node count's third spelling (item 59) and a Pool that "
                      "claims devices the job did not get places onto devices that are not "
                      "there")
    nodes = [NodeShape(cpus=spec["cpus"], gpus=spec["gpus"], name=spec["name"])
             for spec in specs]
    gpus = sum(node.gpus for node in nodes)
    return ResourcePolicy(Pool(nodes, mode=mode)), (
        f"ResourcePolicy over {len(nodes)} node(s), {gpus} GPU(s), mode={mode!r}, derived "
        f"from {type(backend).__name__}.topology()"
    )


# ── what each stream is handed, and what two of them refuse ───────────────────


def preorg_build(version_dir: str, manifest: Any = None) -> streams.Predict:
    """Model B: the featuriser and the ensemble, both from the wrapper.

    Imported inside the call, not at module scope, because the wrapper owns torch (CON-4) and
    this module is constructed in the driver process. The wrapper's own heavy imports are
    function-local too, so this would be safe either way -- it is written this way so that
    staying safe does not depend on the wrapper staying disciplined.
    """
    from preorg_train_wrapper import preorg_predictor  # noqa: PLC0415

    return preorg_predictor(version_dir, manifest)


def _no_envelope(model_id: str, what: str) -> None:
    raise NotImplementedError(
        f"{model_id} has a published model and no single-request `Predict` envelope: {what}. "
        f"`streams.RESIDENT_FROM[{model_id!r}]` already says this surrogate is resident from "
        "M4; this refusal is the same statement as a mechanism, so a promoted version cannot "
        "be served as a number nobody can account for. `streams.load_promoted` turns it into "
        "an `Unavailable` and the client into `StreamUnavailable`."
    )


def anchor_build(version_dir: str, manifest: Any = None) -> streams.Predict:
    """Model A: refuses, by name. See :func:`_no_envelope`."""
    _no_envelope(streams.MODEL_A, (
        "A's head is a `Scorer` over bags of atoms whose output is comparable *within a "
        "group only* (`liganchor_train_wrapper.py:57`), so a single request's score is not a "
        "quantity, and it reports no sigma at all"
    ))
    raise AssertionError("unreachable")  # pragma: no cover


def affdelta_build(version_dir: str, manifest: Any = None) -> streams.Predict:
    """Model C: refuses, by name. See :func:`_no_envelope`."""
    _no_envelope(streams.MODEL_C, (
        "C's GP lives on the residual's PCA basis and predicts a *correction*, so serving "
        "`dg_corrected` means composing `Residual.predict(project(x))` with the posterior "
        "and reading the residual artifact at load time -- a composition, not an adapter"
    ))
    raise AssertionError("unreachable")  # pragma: no cover


# ── one model's half of the loop ──────────────────────────────────────────────


@dataclass
class Surrogate:
    """One model: its manager, the corpus it consumes, and what it has published."""

    model_id: str
    manager: Any
    corpus: LabelCorpus
    recorder: streams.checkpoint_recorder


def _demand(task: Any, cpus: float = 1.0) -> Dict[str, Any]:
    """``task_description["demand"]``, derived from the trainer's own declared quantity.

    Written here and nowhere else, because ``TrainTask.gpus`` is already the number: a demand
    with its own ``gpus`` literal would be the same quantity stated twice, and the two would
    disagree the first time one of them was tuned. The policy reads ``demand`` (item 57); the
    backend ignores the key when there is no policy.
    """
    return {"demand": {"cpus": float(cpus), "gpus": int(getattr(task, "gpus", 0) or 0)}}


def _trainers(settings: Settings) -> Dict[str, Any]:
    """The three drivers. No heavy import happens here; each config names its wrapper."""
    gpus = settings.trainer_gpus
    return {
        streams.MODEL_B: PocketPreorgTrainer(
            PocketPreorgConfig(
                model_id=streams.MODEL_B,
                registry_root=settings.registry_root,
                property_name=PREORG_PROPERTY,
            ),
            gpus=gpus[streams.MODEL_B],
        ),
        streams.MODEL_C: AffDeltaTrainer(
            AffDeltaConfig(
                model_id=streams.MODEL_C,
                registry_root=settings.registry_root,
                residual_path=settings.residual_path,
                # Not the default: `AffDeltaConfig.property_name` defaults to the *metric*
                # ATM reports, and the label store holds the property it supersedes.
                property_name=ABFE_PROPERTY,
            ),
            gpus=gpus[streams.MODEL_C],
        ),
        streams.MODEL_A: LigAnchorTrainer(
            LigAnchorConfig(
                model_id=streams.MODEL_A,
                registry_root=settings.registry_root,
                encoder_path=settings.encoder_path,
                buffer_path=settings.buffer_path,
                property_name=ABFE_PROPERTY,
                # Round 1 banks and does not steer (source spec 7.3); the head switches on
                # as a separate arm, which is a mechanism and not a convention.
                head_enabled=False,
            ),
            gpus=gpus[streams.MODEL_A],
        ),
    }


def _stream_build(model_id: str) -> Callable[..., streams.Predict]:
    return {streams.MODEL_B: preorg_build,
            streams.MODEL_C: affdelta_build,
            streams.MODEL_A: anchor_build}[model_id]


def build_surrogates(flow: Any, settings: Settings) -> List[Surrogate]:
    """Three managers over one engine, in the order their labels arrive.

    ``flow`` is required rather than optional: a manager handed ``None`` builds its own engine
    at ``start`` (``manager.py:205``), and three engines over one allocation is three
    schedulers competing for the same nodes with no policy between them.
    """
    if flow is None:
        raise ValueError(
            "build_surrogates needs the host engine. Passing None would have each manager "
            "build its own at start(), which is CON-1's whole objection -- and the resource "
            "policy attaches to an engine, so three engines is also three inventories"
        )
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs(settings.registry_root, exist_ok=True)

    built: List[Surrogate] = []
    for model_id, trainer in _trainers(settings).items():
        recorder = streams.checkpoint_recorder()
        manager = rome.Manager(
            asyncflow=flow,
            data_config=rome.DataConfig(
                min_samples=settings.min_samples[model_id],
                # Selection already happened upstream, in `LabelCorpus`: a rank-based
                # sampler over a non-contiguous subset is why this project does not use
                # DataManager's watermark. So 'all' here is not a weaker filter, it is the
                # absence of a second one.
                sampling="all",
                # The only way a row is offered twice is a crash between `take` and
                # `commit`, which the take/commit split exists to allow. Keying on the
                # label id makes the replay a no-op instead of a duplicated label.
                dedup_key=_label_id,
            ),
            trainer_config=rome.TrainerConfig(
                trainer=trainer,
                checkpoint_dir=os.path.join(settings.checkpoint_root, model_id),
                poll_interval=settings.poll_interval,
                max_rounds=settings.max_rounds,
                task_description=_demand(trainer),
                on_checkpoint=recorder,
                max_consecutive_failures=int(os.environ.get("ASD_MAX_FAILURES", 3)),
            ),
            stream_configs=[
                streams.surrogate_stream(
                    model_id, _stream_build(model_id),
                    registry_root=settings.registry_root,
                    num_gpus=settings.stream_gpus,
                )
            ],
        )
        built.append(Surrogate(
            model_id=model_id,
            manager=manager,
            # One consumer per model over one store, each with its own `Consumed` set, and
            # `fraction=1.0`: these are measured physics labels, and selecting the best
            # third of a regression's own targets is a bias rather than a filter.
            corpus=LabelCorpus(settings.label_root, model_id, fraction=1.0),
            recorder=recorder,
        ))
    return built


def _label_id(record: Dict[str, Any]) -> Any:
    """A corpus row's identity. Module-level so it survives being handed to a task."""
    return record.get("uid") or record.get("label_id")


def pump(surrogate: Surrogate) -> Dict[str, Any]:
    """Move whatever this model is owed from the label store into its ROME corpus.

    ``take`` then ``add`` then ``commit``, in that order and never collapsed: a crash between
    the second and the third leaves every record still owed, which is what a restart needs
    when a round's worth of labels cost GPU-hours each. ``Consumed.mark`` only unions, so the
    replayed commit is a no-op.
    """
    batch = surrogate.corpus.take()
    if not batch:
        return {"model_id": surrogate.model_id, "owed": 0, "added": 0}
    accepted = surrogate.manager.add_training_batch(list(batch.rows))
    taken = surrogate.corpus.commit(batch)
    return {"model_id": surrogate.model_id, "owed": len(batch),
            "added": len(accepted), "consumed_total": taken}


def report(surrogates: List[Surrogate]) -> str:
    lines = []
    for s in surrogates:
        manager = s.manager
        lines.append(
            f"  {s.model_id}: corpus {manager.data.total_count}, "
            f"{manager.trainer.rounds_completed} round(s), "
            f"published v{s.recorder.version}"
            + (f" -> {os.path.basename(s.recorder.artifact)}" if s.recorder.artifact else "")
        )
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────


async def asd_surrogates_rome(settings: Optional[Settings] = None) -> List[Surrogate]:
    """Bring the three surrogates up over one engine, pump labels into them, wind down."""
    import logging  # noqa: PLC0415

    import rhapsody  # noqa: PLC0415
    from radical.asyncflow import WorkflowEngine  # noqa: PLC0415
    from rhapsody.backends import DragonExecutionBackend  # noqa: PLC0415

    rhapsody.enable_logging(level=logging.INFO)
    settings = settings or Settings.from_env()

    backend = await DragonExecutionBackend()
    policy, why = resource_policy(backend)
    print(f"[ASD] placement: {why}", flush=True)

    # The one engine, built once, with the policy if there is one. CON-1 lives on this line:
    # every manager below is handed `flow`, so none of them builds a second.
    kwargs: Dict[str, Any] = {"backend": backend}
    if policy is not None:
        kwargs["policy"] = policy
    flow = await WorkflowEngine.create(**kwargs)

    surrogates = build_surrogates(flow, settings)
    for surrogate in surrogates:
        await surrogate.manager.start()
    print(f"[ASD] {len(surrogates)} surrogate(s) up; labels from {settings.label_root}",
          flush=True)

    deadline = time.monotonic() + settings.run_seconds if settings.run_seconds else None
    try:
        while True:
            for surrogate in surrogates:
                moved = pump(surrogate)
                if moved["owed"]:
                    print(f"[ASD] {moved['model_id']}: +{moved['added']} of "
                          f"{moved['owed']} owed", flush=True)
            if settings.stop_file and os.path.exists(settings.stop_file):
                print(f"[ASD] {settings.stop_file} exists; winding down", flush=True)
                break
            if deadline is not None and time.monotonic() >= deadline:
                print(f"[ASD] {settings.run_seconds}s elapsed; winding down", flush=True)
                break
            await asyncio.sleep(settings.poll_interval)
        print("[ASD] done:\n" + report(surrogates), flush=True)
    finally:
        for surrogate in surrogates:
            await surrogate.manager.stop()
        # The engine is this file's, so this file shuts it down -- no manager owns it, which
        # is the other half of handing every one of them the same `flow`.
        await flow.shutdown()
    return surrogates


def main() -> int:
    asyncio.run(asd_surrogates_rome())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
