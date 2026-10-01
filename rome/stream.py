"""Stream Manager — persistent asynchronous inference and reward streams.

Inference and reward are not one-shot calls in ROME; they are long-lived
tasks that sit on their nodes and keep processing requests for the whole
campaign. The stream manager owns them: it submits them to asyncflow as
services, feeds them requests from anywhere in the host workflow, collects
their outputs, and — the part that closes ROME's loop — hot-swaps their weights
whenever the training manager publishes a newer checkpoint.

The workflow supplies the actual inference and reward code. ROME supplies the
loop around it::

    await rome.stream.start(StreamConfig(
        name="generate",
        kind=StreamKind.INFERENCE,
        load_func=lambda path, ctx: load_my_model(path),
        process_func=lambda prompts, ctx: ctx.model.generate(prompts),
        num_streams=4,
        num_gpus=1,
    ))

A workflow that would rather own its own loop passes ``stream_func`` instead
and gets the same context object, including the reload signal.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from dragon.data.ddict import DDict
from dragon.native.event import Event

from rome._logging import get_logger
from rome.utils import (
    MODEL_PATH_KEY,
    MODEL_VERSION_KEY,
    Namespace,
    resource_description,
    submit_task,
)

log = get_logger(__name__)

REQUEST_NS = "req"
"""Sub-namespace holding one key per pending request, inside a group's DDict."""
OUTPUT_NS = "out"
"""Sub-namespace holding one key per finished result."""
STATUS_NS = "status"
"""Sub-namespace holding one key per replica's status."""

DEFAULT_STREAM_DDICT_KWARGS: Dict[str, Any] = {
    "managers_per_node": 1,
    "n_nodes": 1,
    "total_mem": 256 * 1024 ** 2,
}
"""Defaults for a stream group's DDict.

Much smaller than the manager's, because a group only ever holds requests in
flight and results not yet collected -- it does not accumulate. Override per
group via :attr:`StreamConfig.ddict_kwargs`.
"""


class StreamStatus(Enum):
    """Status of a single inference/reward stream task."""

    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    RELOADING = 3
    STOPPING = 4
    STOPPED = 5
    FAILED = 6


class StreamKind(str, Enum):
    """What a stream is for.

    Both kinds run the same loop; the distinction is for routing and reporting.
    It matters because reward streams are the ones whose outputs feed the data
    manager.
    """

    INFERENCE = "inference"
    REWARD = "reward"


@dataclass
class StreamConfig:
    """Configuration for a group of identical stream tasks.

    Parameters
    ----------
    name : str
        Identifies the stream group. Requests are routed by name, so a workflow
        can run several independent groups side by side.
    kind : StreamKind
        ``INFERENCE`` or ``REWARD``.
    process_func : Callable
        The workflow's per-batch work: ``(inputs, ctx) -> results``, where
        ``inputs`` is a list of payloads and ``results`` is a list of the same
        length. May be sync or async. A plain ``(inputs)`` function is accepted
        too, so existing code needs no signature change.
    load_func : Optional[Callable]
        ``(model_path, ctx) -> model``, called at startup and on every reload.
        The result becomes ``ctx.model``. Omit for streams that need no model
        (most reward functions).
    stream_func : Optional[Callable]
        Full override: ``(ctx) -> None``, replacing the managed loop. The
        implementation must honour ``ctx.should_stop()`` and call
        ``ctx.maybe_reload()`` between batches. ``process_func`` is ignored.
    model_path : Optional[str]
        Initial checkpoint. When ``None`` the stream starts from whatever the
        training manager has already published, if anything.
    num_streams : int
        How many identical tasks to run for this group.
    num_gpus, num_nodes : int
        Per-task resources, turned into an asyncflow ``task_description``.
    task_description : Optional[dict]
        Backend-specific resource request, passed through to asyncflow
        verbatim. Overrides ``num_gpus``/``num_nodes`` when set.
    as_service : bool
        Submit as an asyncflow service task (default). Streams outlive
        individual workflow stages, which is exactly what services are for.
    batch_size : int
        Maximum requests handed to ``process_func`` in one call.
    poll_interval : float
        Seconds to idle when the request queue is empty.
    auto_reload : bool
        Watch the published model version and reload without being asked —
        "the model is reloaded when the Stream Manager sees that a new model
        checkpoint is available". Turn off to reload only on an explicit
        :meth:`Stream.reload_model`.
    drain_on_stop : bool
        Finish already-queued requests before stopping.
    on_output : Optional[Callable[[dict], None]]
        Called with each output record as it is produced. The manager uses this
        to pipe reward-stream results into the data manager.
    load_kwargs : dict
        Extra keyword arguments for ``load_func``.
    ddict : DDict, optional
        Dictionary backing this group's request and result queues. When
        ``None`` the stream manager allocates one and destroys it on stop.
    ddict_kwargs : dict, optional
        Passed to ``DDict(...)`` when the manager allocates the group's
        dictionary. Defaults to :data:`DEFAULT_STREAM_DDICT_KWARGS`.
    """

    name: str = "default"
    kind: StreamKind = StreamKind.INFERENCE
    process_func: Optional[Callable] = None
    load_func: Optional[Callable] = None
    stream_func: Optional[Callable] = None
    model_path: Optional[str] = None
    num_streams: int = 1
    num_gpus: int = 1
    num_nodes: int = 1
    task_description: Optional[Dict[str, Any]] = None
    as_service: bool = True
    batch_size: int = 1
    poll_interval: float = 0.1
    auto_reload: bool = True
    drain_on_stop: bool = True
    on_output: Optional[Callable[[Dict[str, Any]], None]] = None
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    ddict: Any = None
    ddict_kwargs: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.stream_func is None and self.process_func is None:
            raise ValueError(
                "StreamConfig needs either process_func (managed loop) or "
                "stream_func (self-driven loop)"
            )
        if self.num_streams < 1:
            raise ValueError("StreamConfig.num_streams must be >= 1")

    def describe_resources(self) -> Dict[str, Any]:
        if self.task_description is not None:
            return self.task_description
        return resource_description(gpus=self.num_gpus, nodes=self.num_nodes)


class StreamContext:
    """Handle given to workflow code running inside a stream task.

    Exposes the loaded model, the reload/stop signals and the shared DDict, so
    a self-driven ``stream_func`` has everything the managed loop has.
    """

    def __init__(self, task: "StreamTask"):
        self._task = task
        self.model: Any = None

    @property
    def config(self) -> StreamConfig:
        return self._task.config

    @property
    def index(self) -> int:
        """Which replica of the group this is (``0..num_streams-1``)."""
        return self._task.index

    @property
    def ddict(self) -> Namespace:
        """ROME's shared state — the same view the managers use."""
        return self._task.root

    @property
    def stream_ddict(self) -> Namespace:
        """This stream group's own slice of the shared state."""
        return self._task.ddict

    @property
    def model_path(self) -> Optional[str]:
        return self._task.model_path

    @property
    def model_version(self) -> int:
        return self._task.local_version

    def should_stop(self) -> bool:
        return self._task.stop_event.is_set()

    def maybe_reload(self) -> bool:
        """Reload weights if a newer checkpoint is available; returns whether it did."""
        return self._task.maybe_reload(self)

    def next_requests(self, limit: Optional[int] = None) -> List[Tuple[str, Any]]:
        """Claim up to ``limit`` pending requests as ``(request_id, payload)``."""
        return self._task.claim(limit)

    def emit(self, request_id: str, result: Any) -> Dict[str, Any]:
        """Publish one result back to the workflow."""
        return self._task.emit(request_id, result)


class StreamTask:
    """One persistent inference or reward stream task.

    Holds the asyncflow future for the running task plus its control signals.
    Status lives in the DDict rather than on the object, so a manager on
    another node can read it.
    """

    def __init__(self, index: int, config: StreamConfig, ddict: Namespace,
                 root: Namespace, task_fut: Any = None):
        self.index = index
        self.config = config
        self.ddict = ddict
        """This stream group's slice of the DDict (requests, results, status)."""
        self.root = root
        """ROME's root namespace, where the published checkpoint lives."""
        self.task_fut = task_fut
        self.reload_event = Event()
        self.stop_event = Event()
        self.local_version = 0
        self.model_path: Optional[str] = config.model_path
        self._requests = ddict.namespace(REQUEST_NS, index)
        self._outputs = ddict.namespace(OUTPUT_NS)
        self._status_key = f"{STATUS_NS}|{index}"
        self.status = StreamStatus.NOT_STARTED

    # -- crossing a process boundary ----------------------------------------

    #: Attributes that belong to the *driver* and must never be pickled into a
    #: task process. ``task_fut`` is the killer: the task body closes over this
    #: object by reference, and a multi-process backend pickles that body from
    #: its dispatcher thread *after* :meth:`StreamManager.start` has assigned
    #: the submission future here. An ``_asyncio.Future`` cannot be pickled, so
    #: the dispatcher raises, dies, and every task queued behind it -- ROME's
    #: or the host workflow's -- silently never runs. Dropping it here makes the
    #: body picklable whenever the backend gets round to it.
    _DRIVER_ONLY = ("task_fut",)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        for name in self._DRIVER_ONLY:
            state[name] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -- status -------------------------------------------------------------

    @property
    def status(self) -> StreamStatus:
        value = self.ddict.get(self._status_key, StreamStatus.NOT_STARTED.name)
        if isinstance(value, StreamStatus):
            return value
        try:
            return StreamStatus[value]
        except KeyError:  # pragma: no cover - defensive
            return StreamStatus.NOT_STARTED

    @status.setter
    def status(self, value: StreamStatus) -> None:
        # Stored by name: enum members do not survive every serializer, names do.
        self.ddict[self._status_key] = value.name

    # -- queues -------------------------------------------------------------

    @property
    def pending(self) -> int:
        return len(self._requests.keys())

    def submit(self, request_id: str, payload: Any) -> None:
        self._requests[request_id] = payload

    def claim(self, limit: Optional[int] = None) -> List[Tuple[str, Any]]:
        """Pop pending requests for this task.

        Popping is what makes a request owned: each key is removed by exactly
        one claimer, so replicas never double-process a request.
        """
        limit = limit or self.config.batch_size
        return self._requests.drain(limit=limit)

    def emit(self, request_id: str, result: Any) -> Dict[str, Any]:
        record = {
            "request_id": request_id,
            "result": result,
            "stream": self.config.name,
            "stream_index": self.index,
            "kind": self.config.kind.value,
            "model_version": self.local_version,
            "completed_at": time.time(),
        }
        self._outputs[request_id] = record
        if self.config.on_output is not None:
            try:
                self.config.on_output(record)
            except Exception:  # noqa: BLE001 - a bad hook must not kill the stream
                traceback.print_exc()
        return record

    # -- weights ------------------------------------------------------------

    def published_version(self) -> int:
        return int(self.root.get(MODEL_VERSION_KEY, 0))

    def needs_reload(self) -> bool:
        if self.reload_event.is_set():
            return True
        if not self.config.auto_reload:
            return False
        return self.published_version() > self.local_version

    def maybe_reload(self, ctx: "StreamContext") -> bool:
        """Load the newest published checkpoint into this task, if there is one."""
        if not self.needs_reload():
            return False
        path = self.root.get(MODEL_PATH_KEY) or self.model_path
        version = self.published_version()
        self.status = StreamStatus.RELOADING
        try:
            if self.config.load_func is not None and path is not None:
                ctx.model = _call_user(
                    self.config.load_func, path, ctx, **self.config.load_kwargs
                )
            self.model_path = path
            self.local_version = max(version, self.local_version)
            log.info("%s[%d] reloaded weights -> v%d (%s)",
                     self.config.name, self.index, self.local_version,
                     os.path.basename(path) if path else "no checkpoint")
            # Cleared last, so a waiter in reload_model() only unblocks once the
            # new weights are actually in memory.
            self.reload_event.clear()
            self.status = StreamStatus.RUNNING
            return True
        except Exception:
            # Leave status=FAILED so waiters in reload_model() don't incorrectly
            # assume the reload succeeded.
            self.status = StreamStatus.FAILED
            self.reload_event.clear()
            raise

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<StreamTask {self.config.name}[{self.index}] "
            f"status={self.status.name} v={self.local_version}>"
        )


class Stream:
    """Stream manager: owns every inference and reward stream in a run.

    Each stream group gets **its own DDict** for requests and results, rather
    than sharing the manager's. A dictionary has no server-side prefix query, so
    a scan lists every key in it and filters client-side; sharing one dictionary
    would make the cost of a replica's poll grow with the corpus, coupling two
    rates that have nothing to do with each other. Per group, a scan covers only
    that group's in-flight work, which does not accumulate.

    The manager's own dictionary is still used, read-only from here, for the
    published checkpoint — that is what the streams reload from.

    Parameters
    ----------
    ddict : Namespace
        ROME's shared state, holding the published checkpoint.
    asyncflow : WorkflowEngine
        Engine the stream tasks are submitted to, as services.
    """

    def __init__(self, ddict: Namespace, asyncflow: Any):
        self.ddict = ddict
        self.asyncflow = asyncflow
        self.stream_tasks: List[StreamTask] = []
        self._configs: Dict[str, StreamConfig] = {}
        self._round_robin: Dict[str, int] = {}
        #: group name -> Namespace over that group's dictionary.
        self._group_ns: Dict[str, Namespace] = {}
        #: Groups whose dictionary this manager allocated, and must destroy.
        self._owned: Dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    async def start(self, config: StreamConfig) -> List[StreamTask]:
        """Start ``config.num_streams`` persistent tasks for one stream group.

        Call it once per group — typically an inference group and a reward
        group — against the same manager.
        """
        config.validate()
        self._configs[config.name] = config
        self._round_robin.setdefault(config.name, 0)
        ns = self._open_group(config)
        description = config.describe_resources()

        started: List[StreamTask] = []
        for i in range(config.num_streams):
            task = StreamTask(index=i, config=config, ddict=ns, root=self.ddict)
            task.status = StreamStatus.STARTING

            # Captures the StreamTask and nothing else. A Dragon execution
            # backend runs a task body in a *different process*, so the entry
            # point has to pickle; capturing ``self`` would drag the workflow
            # engine along, which does not pickle -- and the submission then
            # hangs rather than raising. See docs/delta.md.
            async def stream_entry(_task=task):
                await run_stream_task(_task)

            stream_entry.__name__ = f"rome_stream_{config.name}_{i}"
            task.task_fut = submit_task(
                self.asyncflow,
                stream_entry,
                task_description=description,
                service=config.as_service,
            )
            self.stream_tasks.append(task)
            started.append(task)
        log.info("started %s stream '%s' (%d replica%s, %d gpu%s each)",
                 config.kind.value, config.name, config.num_streams,
                 "" if config.num_streams == 1 else "s", config.num_gpus,
                 "" if config.num_gpus == 1 else "s")
        return started

    def _open_group(self, config: StreamConfig) -> Namespace:
        """Get (or allocate) the dictionary backing one stream group."""
        existing = self._group_ns.get(config.name)
        if existing is not None:
            return existing
        if config.ddict is not None:
            group_ddict = config.ddict
        else:
            group_ddict = DDict(
                **{**DEFAULT_STREAM_DDICT_KWARGS, **(config.ddict_kwargs or {})}
            )
            self._owned[config.name] = group_ddict
        # No key prefix: the dictionary is dedicated to this group, so its
        # sub-namespaces (req/out/status) are the only things in it.
        ns = Namespace(group_ddict, "")
        self._group_ns[config.name] = ns
        return ns

    def _namespace(self, name: str) -> Namespace:
        """The namespace for a started group."""
        try:
            return self._group_ns[name]
        except KeyError:
            raise RuntimeError(f"stream {name!r} has not been started") from None

    # -- request/response ---------------------------------------------------

    def submit(self, payload: Any, stream: Optional[str] = None,
               request_id: Optional[str] = None) -> str:
        """Queue one request and return its id.

        Requests round-robin across the group's replicas, so every task gets an
        even share without needing a shared claim protocol.
        """
        name = self._resolve_name(stream)
        tasks = self.tasks_for(name)
        if not tasks:
            raise RuntimeError(f"stream {name!r} has not been started")
        rid = request_id or uuid.uuid4().hex
        index = self._round_robin[name] % len(tasks)
        self._round_robin[name] = index + 1
        tasks[index].submit(rid, payload)
        return rid

    def submit_batch(self, payloads: List[Any], stream: Optional[str] = None) -> List[str]:
        """Queue many requests; returns their ids in order."""
        return [self.submit(p, stream=stream) for p in payloads]

    def get_outputs(self, stream: Optional[str] = None,
                    consume: bool = True) -> List[Dict[str, Any]]:
        """Collect finished results, oldest first.

        Parameters
        ----------
        stream : str, optional
            Which group to read. Defaults to the only started group.
        consume : bool
            Remove results as they are read (default). Pass ``False`` to peek
            without draining — useful for progress displays.
        """
        ns = self._namespace(self._resolve_name(stream)).namespace(OUTPUT_NS)
        pairs = ns.drain() if consume else ns.items()
        records = [v for _, v in pairs if isinstance(v, dict)]
        records.sort(key=lambda r: r.get("completed_at", 0.0))
        return records

    async def get_output(self, request_id: str, stream: Optional[str] = None,
                         timeout: Optional[float] = None,
                         poll_interval: float = 0.05) -> Optional[Dict[str, Any]]:
        """Await one specific request's result.

        Returns ``None`` on timeout rather than raising, so a caller polling
        several requests can keep going.
        """
        ns = self._namespace(self._resolve_name(stream)).namespace(OUTPUT_NS)
        deadline = None if timeout is None else time.time() + timeout
        while True:
            record = ns.pop(request_id, None)
            if record is not None:
                return record
            if deadline is not None and time.time() >= deadline:
                return None
            await asyncio.sleep(poll_interval)

    def pending(self, stream: Optional[str] = None) -> int:
        """How many requests are queued and not yet claimed."""
        return sum(t.pending for t in self.tasks_for(self._resolve_name(stream)))

    # -- weights ------------------------------------------------------------

    def signal_reload(self, model_path: Optional[str] = None,
                      stream: Optional[str] = None) -> List[StreamTask]:
        """Raise the reload flag on the targeted tasks without waiting.

        Each task swaps weights between batches on its own schedule, so this
        never interrupts an in-flight inference call.
        """
        if model_path is not None:
            self.ddict[MODEL_PATH_KEY] = model_path
        targets = self.stream_tasks if stream is None else self.tasks_for(stream)
        for task in targets:
            if model_path is not None:
                task.config.model_path = model_path
            task.reload_event.set()
        return targets

    async def reload_model(self, model_path: Optional[str] = None,
                           stream: Optional[str] = None,
                           wait_for_reload: bool = True,
                           timeout: float = 300.0) -> None:
        """Tell the streams to pick up new weights.

        Called explicitly by the workflow, or implicitly by :meth:`on_checkpoint`
        when the training manager publishes. ``model_path=None`` means "reload
        whatever is currently published".
        """
        targets = self.signal_reload(model_path, stream)
        if not wait_for_reload:
            return
        deadline = time.time() + timeout
        for task in targets:
            while task.reload_event.is_set() and time.time() < deadline:
                if task.status in (StreamStatus.STOPPED, StreamStatus.FAILED):
                    break
                await asyncio.sleep(0.05)

    def on_checkpoint(self, checkpoint_path: str, version: int) -> None:
        """Training-manager hook: a new checkpoint just landed.

        Registered by :class:`~rome.manager.Manager`, so publishing a checkpoint
        and hot-swapping the streams are the same event. Non-blocking by design.
        """
        self.signal_reload(checkpoint_path)

    # -- status / shutdown --------------------------------------------------

    def get_status(self, stream: Optional[str] = None) -> List[StreamStatus]:
        """Status of every task, or of one group's tasks."""
        targets = self.stream_tasks if stream is None else self.tasks_for(stream)
        return [task.status for task in targets]

    def report(self) -> Dict[str, Any]:
        """Per-group summary for dashboards and logs."""
        out: Dict[str, Any] = {}
        for name, config in self._configs.items():
            tasks = self.tasks_for(name)
            out[name] = {
                "kind": config.kind.value,
                "tasks": len(tasks),
                "status": [t.status.name for t in tasks],
                "pending": sum(t.pending for t in tasks),
                # The *published* version, read from shared state — authoritative
                # and cross-process. A replica's own local_version lives in its
                # task process and does not come back to the driver, so reading
                # that would report 0 on a multi-process backend even while the
                # replica is serving the latest checkpoint.
                "model_version": int(self.ddict.get(MODEL_VERSION_KEY, 0)),
            }
        return out

    async def stop(self, stream: Optional[str] = None, wait_for_stop: bool = True,
                   timeout: float = 60.0) -> None:
        """Signal the stream tasks to stop, optionally waiting for them to finish.

        Stopping does not release the group dictionaries — results drained on
        the way down stay readable, which is the whole point of
        ``drain_on_stop``. Call :meth:`close` once those have been collected.
        """
        targets = self.stream_tasks if stream is None else self.tasks_for(stream)
        for task in targets:
            task.stop_event.set()
        if not wait_for_stop:
            return
        deadline = time.time() + timeout
        for task in targets:
            while time.time() < deadline:
                if task.status in (StreamStatus.STOPPED, StreamStatus.FAILED,
                                   StreamStatus.NOT_STARTED):
                    break
                await asyncio.sleep(0.05)

    def close(self, stream: Optional[str] = None) -> None:
        """Release the dictionaries this manager allocated for its groups.

        Separate from :meth:`stop` so a workflow can collect what the streams
        drained on the way down before their storage goes away. Idempotent.
        """
        for name in (list(self._owned) if stream is None else [stream]):
            self._release_group(name)

    def _release_group(self, name: str) -> None:
        """Destroy a group's dictionary, if this manager allocated it.

        A dictionary the workflow supplied is left alone — it may well outlive
        the stream, and it is not ours to free.
        """
        group_ddict = self._owned.pop(name, None)
        self._group_ns.pop(name, None)
        if group_ddict is None:
            return
        destroy = getattr(group_ddict, "destroy", None)
        if destroy is None:
            return
        try:
            destroy()
        except Exception:  # noqa: BLE001 - teardown must not mask a real error
            traceback.print_exc()

    # -- helpers ------------------------------------------------------------

    def tasks_for(self, name: str) -> List[StreamTask]:
        return [t for t in self.stream_tasks if t.config.name == name]

    def _resolve_name(self, stream: Optional[str]) -> str:
        if stream is not None:
            return stream
        if len(self._configs) == 1:
            return next(iter(self._configs))
        if not self._configs:
            raise RuntimeError("no streams have been started")
        raise ValueError(
            f"several streams are running ({', '.join(sorted(self._configs))}); "
            "pass stream=<name>"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Stream groups={sorted(self._configs)} tasks={len(self.stream_tasks)}>"


# ---------------------------------------------------------------------------
# The stream task body
#
# Module-level, and taking only a StreamTask, on purpose. A Dragon execution
# backend runs the body in another process, so everything it closes over has to
# survive pickling. A StreamTask does: its DDict views and its Dragon events are
# transportable, and Dragon re-attaches the receiving process automatically. A
# bound method would also carry the Stream manager and therefore the workflow
# engine, which does not pickle -- and the symptom is a task that never starts
# rather than an error.
# ---------------------------------------------------------------------------

async def run_stream_task(task: "StreamTask") -> None:
    """The persistent loop: load a model, then process until told to stop."""
    ctx = StreamContext(task)
    config = task.config
    try:
        initial_path = config.model_path or task.root.get(MODEL_PATH_KEY)
        if config.load_func is not None and initial_path is not None:
            ctx.model = _call_user(
                config.load_func, initial_path, ctx, **config.load_kwargs
            )
            task.model_path = initial_path
        task.local_version = int(task.root.get(MODEL_VERSION_KEY, 0))
        task.status = StreamStatus.RUNNING

        if config.stream_func is not None:
            await _call_user_async(config.stream_func, ctx)
        else:
            await _managed_loop(task, ctx)
    except asyncio.CancelledError:
        task.status = StreamStatus.STOPPED
        raise
    except Exception:  # noqa: BLE001 - status carries the failure
        task.status = StreamStatus.FAILED
        traceback.print_exc()
        raise
    else:
        task.status = StreamStatus.STOPPED


async def _managed_loop(task: "StreamTask", ctx: StreamContext) -> None:
    config = task.config
    while not task.stop_event.is_set():
        task.maybe_reload(ctx)
        batch = task.claim(config.batch_size)
        if not batch:
            await asyncio.sleep(config.poll_interval)
            continue
        await _process_batch(task, ctx, batch)

    if config.drain_on_stop:
        task.status = StreamStatus.STOPPING
        while True:
            batch = task.claim(config.batch_size)
            if not batch:
                break
            await _process_batch(task, ctx, batch)


async def _process_batch(task: "StreamTask", ctx: StreamContext,
                         batch: List[Tuple[str, Any]]) -> None:
    """Run the workflow's code over one batch and publish the results.

    A failure is reported per-request rather than killing the stream: one
    malformed payload should not take down a task the rest of the campaign
    depends on.
    """
    request_ids = [rid for rid, _ in batch]
    payloads = [payload for _, payload in batch]
    try:
        results = await _call_user_async(task.config.process_func, payloads, ctx)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        for rid in request_ids:
            task.emit(rid, {"error": repr(exc)})
        return

    for rid, result in zip(request_ids, _align(results, len(request_ids))):
        task.emit(rid, result)


# ---------------------------------------------------------------------------
# Calling workflow-supplied code
# ---------------------------------------------------------------------------

def _accepts_context(func: Callable) -> bool:
    """Whether ``func`` wants the ROME context as its second positional argument.

    Existing inference and reward code usually has the shape ``f(inputs)``.
    Adopting ROME should not force a signature change, so the context is only
    passed when there is somewhere to put it.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # builtins and other C callables
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
        return True
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _user_args(func: Callable, arg: Any, ctx: Any) -> Tuple[Any, ...]:
    return (arg, ctx) if (ctx is not None and _accepts_context(func)) else (arg,)


async def _call_user_async(func: Callable, arg: Any, ctx: Any = None, **kwargs: Any) -> Any:
    """Call workflow code from the stream loop.

    A blocking function is pushed to a worker thread so one slow inference call
    cannot stall the other streams sharing this event loop.
    """
    args = _user_args(func, arg, ctx)
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _call_user(func: Callable, arg: Any, ctx: Any = None, **kwargs: Any) -> Any:
    """Call workflow code from a synchronous point (model loading)."""
    result = func(*_user_args(func, arg, ctx), **kwargs)
    if inspect.isawaitable(result):
        raise TypeError(
            f"{getattr(func, '__name__', func)!r} is async, but load_func is "
            "called synchronously; make it a regular function"
        )
    return result


def _align(results: Any, expected: int) -> List[Any]:
    """Coerce whatever the workflow's function returned into ``expected`` results.

    A batch function returns a list; single-item code usually returns a bare
    value. Both are accepted so ``batch_size=1`` streams can reuse ordinary
    scalar-returning reward functions unchanged.
    """
    if isinstance(results, (list, tuple)):
        results = list(results)
        if len(results) == expected:
            return results
        if expected == 1:
            return [results]
        raise ValueError(
            f"stream function returned {len(results)} results for {expected} requests"
        )
    if expected == 1:
        return [results]
    raise ValueError(f"stream function returned a single result for {expected} requests")
