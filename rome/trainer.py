"""Training Manager — creates and schedules training tasks, publishes checkpoints.

Responsibilities, straight off the design slide:

* create and schedule training tasks on HPC — every round is submitted to the
  ``radical.asyncflow`` engine the host workflow handed to ROME, so the
  training task gets its own nodes and GPUs like any other workflow task;
* publish updated checkpoints back to the workflow, so the stream manager and
  the host workflow pick up the improved model mid-campaign;
* answer whether training is *possible*, *running*, or *finished*.

Training starts automatically once enough data accumulates (the data manager's
``min_samples`` threshold) or is triggered manually by the workflow awaiting
:meth:`Trainer.train`.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dragon.native.event import Event

from rome._logging import get_logger
from rome.data import DataManager
from rome.train.base import FunctionTrainer, TrainTask
from rome.utils import (
    MODEL_PATH_KEY,
    MODEL_VERSION_KEY,
    Namespace,
    resource_description,
    submit_task,
)


log = get_logger(__name__)
#: A round publishing a checkpoint *creates a model*, so it logs under the
#: ``[ROME-MODEL]`` tag (matching IMPRESS's green "checkpoint" component) rather
#: than ``[ROME-TRAINER]``.
model_log = get_logger("rome.model")


def _has_files(root: str) -> bool:
    """Whether a round has written anything into its output directory.

    The directory itself is created by the driver before the round is
    submitted, so only a file inside it means the task body actually ran.
    """
    for _dirpath, _dirnames, filenames in os.walk(root):
        if filenames:
            return True
    return False


TRAIN_COMPLETE_MARKER = "train_complete"
"""Filename an executable trainer's wrapper writes into a round's ``output_dir``.

Written as the wrapper's *final* action, so the driver can detect completion even
when the backend never delivers the task's result. It must appear only when the
round has finished writing its checkpoint — unlike the checkpoint itself, which
for ``publish_into_repo`` is a stable path that already exists from the previous
round (or the initial weights). Kept in sync with the wrapper scripts; see
:meth:`rome.train.base.TrainTask.as_command`.
"""


def _round_output_ready(path: str) -> bool:
    """Whether a round's output is on disk — a checkpoint file or a filled dir."""
    return os.path.isfile(path) or _has_files(path)


def _command_body(command: str):
    """Wrap a shell command as an asyncflow executable-task body.

    An executable task's body returns the command line to run; asyncflow places
    it on the backend and runs it as a subprocess. Defined at module scope, and
    closing over only the ``command`` string, so it pickles cleanly to a
    multi-process backend.
    """
    async def run_command():
        return command

    run_command.__name__ = "rome_train_command"
    return run_command


class TrainerStatus(Enum):
    """Status of the ROME training manager.

    ``NOT_ENOUGH_DATA`` and ``WAITING`` are both idle, but they answer different
    questions: the first says a round is *not currently possible*, the second
    says it is possible and about to start.
    """

    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4
    NOT_ENOUGH_DATA = 5
    TRAINING_COMPLETE = 6
    WAITING = 7
    FAILED = 8

    @property
    def is_idle(self) -> bool:
        return self in (
            TrainerStatus.NOT_ENOUGH_DATA,
            TrainerStatus.WAITING,
            TrainerStatus.TRAINING_COMPLETE,
        )

    @property
    def is_terminal(self) -> bool:
        return self in (TrainerStatus.STOPPED, TrainerStatus.FAILED)


@dataclass
class TrainerConfig:
    """Configuration for the training manager.

    Parameters
    ----------
    trainer : TrainTask or Callable
        The training algorithm. A bare callable
        ``(dataset, output_dir, **kwargs) -> checkpoint_path`` is wrapped in a
        :class:`~rome.train.base.FunctionTrainer` automatically.
    checkpoint_dir : str
        Root for published checkpoints; rounds land in
        ``<checkpoint_dir>/<trainer name>/v<version>``.
    auto_train : bool
        Poll the data manager and fire a round as soon as enough data has
        accumulated. Turn off to drive training purely by hand.
    max_rounds : Optional[int]
        Stop after this many completed rounds. ``None`` runs until stopped.
    poll_interval : float
        Seconds between data-threshold checks in the auto-train loop.
    result_fallback_seconds : Optional[float]
        Grace period before a round whose checkpoint is already on disk is
        treated as finished even though the execution backend never delivered
        its result. ``None`` disables the fallback and waits on the backend
        forever.

        This exists because of a real backend defect, not as a convenience.
        Dragon pre-registers a running task's result key, so *reading* that key
        blocks rather than raising ``KeyError``. rhapsody's monitor sweeps its
        outstanding tasks in order, so a task that never completes -- which is
        exactly what a ROME inference stream is -- blocks the sweep on its own
        key forever, and every result behind it, including a finished training
        round, is never delivered. Confirmed by direct measurement::

            manager(0)['0-0'] -> STILL BLOCKED after 20s   # the stream service
            manager(0)['0-1'] -> returned in 0.12s         # the finished round

        The round itself is fine: it runs, and its checkpoint is on disk. Only
        the notification is lost, so the checkpoint is the sounder signal.
    task_description : Optional[dict]
        Backend-specific resource request forwarded to asyncflow. When
        ``None``, one is derived from the trainer's ``gpus``/``nodes``.
    train_kwargs : dict
        Extra keyword arguments forwarded to every ``TrainTask.train`` call.
    on_checkpoint : Optional[Callable[[str, int], None]]
        Called with ``(checkpoint_path, version)`` after each successful round.
        The manager registers the stream manager's hot-swap hook here.
    stop_on_failure : bool
        Move to ``FAILED`` and end the auto-train loop when a round raises.
        When ``False`` (default) the failure is recorded and polling continues.
    max_consecutive_failures : Optional[int]
        Stop the auto-train loop after this many consecutive failed rounds.
        ``None`` (default) retries indefinitely. Resets to zero on a
        successful round.
    """

    trainer: Any = None
    checkpoint_dir: str = "./rome_checkpoints"
    auto_train: bool = True
    max_rounds: Optional[int] = None
    poll_interval: float = 5.0
    result_fallback_seconds: Optional[float] = 60.0
    task_description: Optional[Dict[str, Any]] = None
    train_kwargs: Dict[str, Any] = field(default_factory=dict)
    on_checkpoint: Optional[Callable[[str, int], None]] = None
    stop_on_failure: bool = False
    max_consecutive_failures: Optional[int] = None


class Trainer:
    """Training manager in the ROME framework.

    Parameters
    ----------
    ddict : Namespace
        Shared state — the same DDict view every other ROME component uses.
    data : DataManager
        Corpus the training rounds draw from.
    asyncflow : WorkflowEngine
        Engine training tasks are submitted to.
    config : TrainerConfig, optional
        May also be supplied later to :meth:`start`.
    """

    def __init__(
        self,
        ddict: Namespace,
        data: DataManager,
        asyncflow: Any,
        config: Optional[TrainerConfig] = None,
    ):
        self.ddict = ddict
        self.data = data
        self.asyncflow = asyncflow
        self.config = config or TrainerConfig()
        self.stop_event = Event()
        self.listener_fut: Optional[asyncio.Task] = None

        self._status = TrainerStatus.NOT_STARTED
        self._rounds_completed = 0
        self._last_error: Optional[str] = None
        self._in_flight = False
        #: Future for the round currently in flight, for diagnostics only.
        self._round_fut = None
        self._extra_callbacks: List[Callable[[str, int], None]] = []

    # -- lifecycle ----------------------------------------------------------

    async def start(self, config: Optional[TrainerConfig] = None) -> Optional[asyncio.Task]:
        """Start the training manager.

        With ``auto_train`` on this spawns the polling loop that fires a round
        whenever the corpus crosses ``min_samples``. With it off the manager
        simply becomes available for manual :meth:`train` calls.

        The poll loop is a plain asyncio task in the manager's own process —
        only the training rounds themselves go to the execution backend.
        """
        if config is not None:
            self.config = config
        if self.config.trainer is None:
            raise ValueError("TrainerConfig.trainer must be set before start()")

        self._status = TrainerStatus.STARTING
        self.stop_event.clear()
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Settle on a real status before returning: a caller that checks
        # get_training_status() immediately after start() should see whether a
        # round is possible, not that the loop has not been scheduled yet.
        self._status = self._idle_status()
        if not self.config.auto_train:
            return None

        self.listener_fut = asyncio.create_task(self._trainer_listener())
        return self.listener_fut

    async def _trainer_listener(self) -> None:
        """Poll the corpus and fire a training round when one becomes possible."""
        self._status = self._idle_status()
        consecutive_failures = 0
        while not self.stop_event.is_set():
            if self._rounds_exhausted():
                self._status = TrainerStatus.TRAINING_COMPLETE
                return
            if self._in_flight or not self.data.ready_to_train():
                self._status = self._idle_status()
                await asyncio.sleep(self.config.poll_interval)
                continue
            try:
                await self._run_round()
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced via status
                self._record_failure(exc)
                consecutive_failures += 1
                cap = self.config.max_consecutive_failures
                if self.config.stop_on_failure or (cap is not None and consecutive_failures >= cap):
                    log.error(
                        "stopping auto-train loop after %d consecutive failure%s",
                        consecutive_failures,
                        "" if consecutive_failures == 1 else "s",
                    )
                    return
                await asyncio.sleep(self.config.poll_interval)
        self._status = TrainerStatus.STOPPED

    async def stop(self, wait_for_stop: bool = True, timeout: float = 300.0) -> None:
        """Stop the training manager.

        An in-flight round is *not* cancelled — killing a half-finished
        fine-tune would leave a torn checkpoint — so this waits it out instead.
        """
        self._status = TrainerStatus.STOPPING
        self.stop_event.set()
        if not wait_for_stop:
            return
        deadline = time.time() + timeout
        while self._in_flight and time.time() < deadline:
            await asyncio.sleep(0.05)
        if self.listener_fut is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.listener_fut),
                    timeout=max(0.0, deadline - time.time()) or 0.1,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        if self._status is not TrainerStatus.FAILED:
            self._status = TrainerStatus.STOPPED

    # -- training -----------------------------------------------------------

    async def train(self, force: bool = False, **kwargs: Any) -> Optional[str]:
        """Run one training round now — the workflow's manual trigger.

        Returns the path to the new checkpoint, or ``None`` when the round was
        skipped because the corpus was not ready (pass ``force=True`` to train
        anyway) or another round is already in flight.

        Failures propagate: a manual call is expected to surface its own
        errors, unlike the auto-train loop which records them in
        :attr:`status`.
        """
        if not force and not self.data.ready_to_train():
            self._status = TrainerStatus.NOT_ENOUGH_DATA
            return None
        return await self._run_round(**kwargs)

    async def _run_round(self, **kwargs: Any) -> Optional[str]:
        if self._in_flight:
            return None
        self._in_flight = True
        try:
            return await self._execute_round(**kwargs)
        finally:
            self._in_flight = False

    async def _execute_round(self, **kwargs: Any) -> str:
        """Build the dataset, submit the training task, publish the checkpoint."""
        task = self.train_task
        self._status = TrainerStatus.RUNNING

        dataset = self.data.get_dataset(as_hf_dataset=task.wants_hf_dataset)
        task.validate(dataset)
        sample_count = self._dataset_size(dataset)

        version = self.model_version + 1
        output_dir = task.prepare_output_dir(
            self.config.checkpoint_dir, version, task.name
        )

        call_kwargs = dict(self.config.train_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("model_version", version)

        log.info("submitting training round %d (%s designs, trainer %s) -> v%d",
                 self._rounds_completed + 1,
                 sample_count if sample_count >= 0 else "?", task.name, version)
        checkpoint = await self._submit_round(task, dataset, output_dir, call_kwargs)
        self._publish(checkpoint or output_dir, version, sample_count)
        return checkpoint or output_dir

    async def _submit_round(self, task, dataset, output_dir, call_kwargs) -> Any:
        """Hand one training round to asyncflow.

        A trainer that implements :meth:`~rome.train.base.TrainTask.as_command`
        is submitted as an *executable* task — a shell command that runs the
        round in its own process, the way IMPRESS submits its wrapper scripts.
        Otherwise the round is a *function* task: ``TrainTask.train`` is
        synchronous and blocking on purpose, so the body runs it in a thread and
        awaits that. Resource requirements come from the ``TrainTask`` itself,
        which is what keeps "adding a new training algorithm requires just one
        task" true.
        """
        description = self.config.task_description
        if description is None:
            description = resource_description(gpus=task.gpus, nodes=task.nodes)

        plan = task.as_command(dataset, output_dir, **call_kwargs)
        if plan is not None:
            command, checkpoint = plan
            log.info("dispatching training round as a command: %s", command)
            self._round_fut = submit_task(
                self.asyncflow, _command_body(command),
                task_description=description, executable=True,
            )
            # An executable task's result is not the checkpoint path, so we wait
            # for completion and then report where the command wrote it. The
            # wrapper writes a per-round completion marker into output_dir as its
            # last action, so the marker appearing proves the round finished —
            # which lets _await_round detect completion the moment it appears,
            # without waiting on a future the backend may never resolve (see
            # docs/dragon.md). The checkpoint itself can't be the signal: with
            # publish_into_repo it is a stable path that already exists from the
            # previous round or the initial weights.
            marker = os.path.join(output_dir, TRAIN_COMPLETE_MARKER)
            await self._await_round(self._round_fut, marker, authoritative=True)
            return checkpoint

        async def train_entry():
            return await asyncio.to_thread(task.train, dataset, output_dir, **call_kwargs)

        train_entry.__name__ = f"rome_train_{task.name}"
        # Kept only so a stalled round can be inspected: a round that never
        # returns leaves no other handle on the submission.
        self._round_fut = submit_task(
            self.asyncflow, train_entry, task_description=description
        )
        return await self._await_round(self._round_fut, output_dir)

    async def _await_round(self, fut: Any, done_path: str,
                           *, authoritative: bool = False) -> Any:
        """Wait for a round, believing the disk when the backend goes quiet.

        The round finishes as soon as *either* its future resolves *or* its
        output appears on disk — because on Dragon a task can run to completion
        and never resolve its future: a running service blocks rhapsody's monitor
        from delivering the result (see ``docs/dragon.md``). So we cannot wait on
        the future alone.

        ``done_path`` is the per-round completion marker for an executable round
        (see :data:`TRAIN_COMPLETE_MARKER`) or the output directory for a function
        round. When ``authoritative`` (the executable case) the wrapper writes the
        marker only after the checkpoint is safely on disk, so its existence *is*
        completion: we poll for it briskly and publish the instant it appears,
        rather than waiting out ``result_fallback_seconds``. For a function round
        the future's return value is the real checkpoint path, so we give the
        backend the full grace to deliver it before falling back to the output
        directory. A failed round still surfaces its exception, because a body
        that raised never writes its output.
        """
        grace = self.config.result_fallback_seconds
        if grace is None:
            return await fut

        # Poll the disk this often. Brisk for an authoritative checkpoint file so
        # completion is caught within seconds; for a function round there is
        # nothing to gain from polling before the grace elapses, so wait it out.
        poll = min(2.0, grace) if authoritative else grace
        waited = 0.0
        while True:
            try:
                # Awaiting the future is what actually *drives* the task; the
                # shield keeps the timeout from cancelling the round.
                return await asyncio.wait_for(asyncio.shield(fut), timeout=poll)
            except asyncio.TimeoutError:
                waited += poll
                if _round_output_ready(done_path) and (authoritative or waited >= grace):
                    log.warning(
                        "round %s: the execution backend has not delivered a "
                        "result after %.0fs, but the checkpoint is on disk — "
                        "publishing from disk. The task finished; its future "
                        "never resolved, which on Dragon means a running service "
                        "is blocking result delivery (see docs/dragon.md).",
                        done_path, waited,
                    )
                    return done_path
                # Nothing on disk yet and the future is still pending. Say so
                # every `grace` seconds rather than looping silently — a silent
                # re-wait is exactly what a hang looks like. This is normal while
                # a round is still running; it only means trouble if the output
                # stays absent long past when the round should have finished,
                # which points at the round not being scheduled at all (e.g. its
                # execution slot held by a long-lived stream service on a small
                # allocation — see docs/dragon.md and test_task_capacity_dragon).
                if waited >= grace and (waited % grace) < poll:
                    log.info(
                        "round %s: still waiting after %.0fs (future pending, no "
                        "output on disk yet)", done_path, waited,
                    )

    def _publish(self, checkpoint: str, version: int, sample_count: int) -> None:
        """Make a finished checkpoint visible to the rest of the workflow.

        Order matters: the path is written *before* the version is bumped, so a
        stream task that notices the new version always finds a valid path
        behind it.
        """
        self.ddict[MODEL_PATH_KEY] = checkpoint
        self.ddict[MODEL_VERSION_KEY] = version
        self.ddict["last_trained_at"] = time.time()
        self.ddict["last_train_samples"] = sample_count
        self.data.mark_consumed()
        self._rounds_completed += 1
        model_log.info("published v%d (%s designs) -> %s",
                       version, sample_count if sample_count >= 0 else "?",
                       checkpoint)
        self._status = (
            TrainerStatus.TRAINING_COMPLETE
            if self._rounds_exhausted()
            else self._idle_status()
        )
        for callback in self._checkpoint_callbacks():
            try:
                callback(checkpoint, version)
            except Exception:  # noqa: BLE001 - a bad hook must not lose a checkpoint
                traceback.print_exc()

    def _checkpoint_callbacks(self) -> List[Callable[[str, int], None]]:
        callbacks = list(self._extra_callbacks)
        if self.config.on_checkpoint is not None:
            callbacks.append(self.config.on_checkpoint)
        return callbacks

    def on_checkpoint(self, callback: Callable[[str, int], None]) -> Callable:
        """Register an extra ``(checkpoint_path, version)`` callback."""
        self._extra_callbacks.append(callback)
        return callback

    # -- status -------------------------------------------------------------

    @property
    def status(self) -> TrainerStatus:
        """Whether training is possible, running, or finished."""
        if self._in_flight or self._status is TrainerStatus.RUNNING:
            return TrainerStatus.RUNNING
        if self._status.is_terminal or self._status is TrainerStatus.TRAINING_COMPLETE:
            return self._status
        if self._status in (
            TrainerStatus.NOT_STARTED,
            TrainerStatus.STARTING,
            TrainerStatus.STOPPING,
        ):
            return self._status
        return self._idle_status()

    def get_status(self) -> TrainerStatus:
        """Alias for :attr:`status` — the API-call form from the design slide."""
        return self.status

    def _idle_status(self) -> TrainerStatus:
        return (
            TrainerStatus.WAITING
            if self.data.ready_to_train()
            else TrainerStatus.NOT_ENOUGH_DATA
        )

    def get_current_model(self) -> Optional[str]:
        """Path to the newest published checkpoint, or ``None`` before round 1."""
        return self.ddict.get(MODEL_PATH_KEY)

    @property
    def model_version(self) -> int:
        """Version of the newest published checkpoint (0 = never trained)."""
        return int(self.ddict.get(MODEL_VERSION_KEY, 0))

    @property
    def rounds_completed(self) -> int:
        return self._rounds_completed

    @property
    def last_error(self) -> Optional[str]:
        """Traceback of the most recent failed round, if any."""
        return self._last_error

    def report(self) -> Dict[str, Any]:
        """One-call summary for dashboards and logs."""
        return {
            "status": self.status.name,
            "model_version": self.model_version,
            "model_path": self.get_current_model(),
            "rounds_completed": self._rounds_completed,
            "corpus_size": self.data.total_count,
            "unconsumed": self.data.unconsumed_count,
            "last_error": self._last_error,
        }

    # -- helpers ------------------------------------------------------------

    @property
    def train_task(self) -> TrainTask:
        """The configured trainer, wrapping a bare callable if needed."""
        trainer = self.config.trainer
        if trainer is None:
            raise ValueError("TrainerConfig.trainer must be set")
        if isinstance(trainer, TrainTask):
            return trainer
        if callable(trainer):
            wrapped = FunctionTrainer(trainer)
            self.config.trainer = wrapped
            return wrapped
        raise TypeError(
            f"TrainerConfig.trainer must be a TrainTask or callable, "
            f"got {type(trainer).__name__}"
        )

    def _rounds_exhausted(self) -> bool:
        cap = self.config.max_rounds
        return cap is not None and self._rounds_completed >= cap

    @staticmethod
    def _dataset_size(dataset: Any) -> int:
        try:
            return len(dataset)
        except TypeError:
            return -1

    def _record_failure(self, exc: BaseException) -> None:
        self._last_error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        log.error("training round failed: %s: %s\n%s",
                  type(exc).__name__, exc, self._last_error.rstrip())
        self._status = (
            TrainerStatus.FAILED if self.config.stop_on_failure else self._idle_status()
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<Trainer status={self.status.name} version={self.model_version} "
            f"rounds={self._rounds_completed}>"
        )
