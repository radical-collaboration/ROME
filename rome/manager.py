"""ROME Manager — the single object a host workflow talks to.

ROME adds model improvement to a workflow that already exists. The manager is
the seam: it owns the three components from the design deck and wires them to
each other so the workflow only ever sees a handful of API calls.

    Data Manager    gathers scored outputs -> training dataset
    Training Manager schedules training tasks -> publishes checkpoints
    Stream Manager  runs inference/reward streams -> hot-swaps checkpoints

The wiring that closes the loop is one line in :meth:`Manager.start`: the
training manager's checkpoint callback is the stream manager's reload hook, so
a finished training round *is* the event that swaps the streams onto the new
model. Nothing in the host workflow has to orchestrate that.

Typical adoption::

    rome = Manager(asyncflow=flow, data_config=DataConfig(min_samples=64),
                   trainer_config=TrainerConfig(trainer=my_trainer))
    await rome.start()

    ...                                   # the workflow runs unchanged
    rome.add_training_data(sequence=seq, score=plddt)   # from anywhere
    if rome.get_current_model():                        # improved weights
        mpnn.load(rome.get_current_model())

    await rome.stop()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dragon.data.ddict import DDict
from dragon.native.event import Event
from radical.asyncflow import WorkflowEngine

from rome._logging import get_logger
from rome.data import DataConfig, DataManager
from rome.stream import Stream, StreamConfig, StreamKind, StreamStatus
from rome.trainer import Trainer, TrainerConfig, TrainerStatus
from rome.utils import MODEL_PATH_KEY, MODEL_VERSION_KEY, Namespace

log = get_logger("rome.manager")

ROME_NS = "rome"
"""Prefix isolating ROME's keys.

A DDict shared with the host workflow therefore never collides with the
workflow's own state.
"""

DEFAULT_DDICT_KWARGS: Dict[str, Any] = {
    "managers_per_node": 1,
    "n_nodes": 1,
    "total_mem": 1024 ** 3,
}
"""Defaults for a DDict ROME allocates itself.

Dragon's own default ``total_mem`` is 3 MiB; a campaign corpus of a few thousand
records plus the stream queues goes past that, and the failure mode is an
allocation error deep in a manager rather than anything ROME can explain, so
ask for room.
"""


class Manager:
    """Manager class for handling tasks and data in the ROME framework.

    Parameters
    ----------
    asyncflow : WorkflowEngine, optional
        The engine ROME submits its tasks to. Pass the host workflow's — for
        an IMPRESS campaign that is ``impress_manager.flow``, available once the
        manager has started — and ROME's rounds and streams are scheduled
        against the same allocation as the campaign's own tasks.

        Leave it ``None`` and ROME builds its own engine at :meth:`start` and
        shuts it down at :meth:`stop`. That is the right choice when ROME
        should manage its own tasks independently of the host, or when the host
        creates its engine internally and does not hand one out.
    backend : optional
        Execution backend for the engine ROME builds for itself. Ignored when
        ``asyncflow`` is given. When both are omitted, asyncflow falls back to a
        **local, in-process** backend — fine for tests and CPU work, but wrong
        for a GPU fine-tune: a training round then runs inside this driver
        process, so the model's CUDA context stays resident for the whole
        campaign instead of being freed when the round ends. Pass a
        process-based backend (``DragonExecutionBackend`` on Delta, or a
        ``ConcurrentExecutionBackend(ProcessPoolExecutor())``) so each round runs
        in a task process that releases its VRAM on exit. See
        ``examples/impress_r/run_protein_binding_rome.py``.
    data_config : DataConfig, optional
        How scored outputs become a dataset, and how much data a training round
        needs.
    trainer_config : TrainerConfig, optional
        Which training algorithm to run and when. Required before training can
        start; may also be passed to :meth:`start`.
    stream_configs : list[StreamConfig], optional
        Inference and reward streams to bring up at :meth:`start`. Streams can
        equally be started later via ``manager.stream.start(...)``, or omitted
        entirely by workflows that only want the data + training halves.
    ddict : DDict, optional
        Share the host workflow's dictionary instead of allocating one. ROME
        namespaces all of its keys, so sharing is safe, and passing the DDict
        object into a task is all it takes to reach it from another node —
        Dragon attaches the receiving process automatically.
    ddict_kwargs : dict, optional
        Passed to ``DDict(...)`` when ROME allocates its own. Worth setting
        on a real campaign: Dragon's default ``total_mem`` is 3 MiB, which a
        corpus of a few thousand records will exhaust, so ROME asks for
        :data:`DEFAULT_DDICT_KWARGS` instead.
    auto_reload_streams : bool
        Wire published checkpoints to stream reloads (default). Turn off if the
        workflow would rather decide when inference switches models.
    """

    def __init__(
        self,
        asyncflow: Optional[WorkflowEngine] = None,
        *,
        backend: Any = None,
        data_config: Optional[DataConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        stream_configs: Optional[List[StreamConfig]] = None,
        ddict: Optional[DDict] = None,
        ddict_kwargs: Optional[Dict[str, Any]] = None,
        auto_reload_streams: bool = True,
    ):
        self.asyncflow = asyncflow
        self.backend = backend
        #: True when ROME built the engine and is therefore responsible for
        #: shutting it down. An engine handed in belongs to the host workflow.
        self._owns_asyncflow = False
        if ddict is not None:
            self.manager_ddict = ddict
        else:
            self.manager_ddict = DDict(**{**DEFAULT_DDICT_KWARGS, **(ddict_kwargs or {})})
        self.ns = Namespace(self.manager_ddict, f"{ROME_NS}|")
        self.stop_event = Event()
        self.auto_reload_streams = auto_reload_streams
        self._stream_configs = list(stream_configs or [])
        self._started = False

        self.data = DataManager(self.ns, data_config)
        self.trainer = Trainer(self.ns, self.data, asyncflow, trainer_config)
        self.stream = Stream(self.ns, asyncflow)

    # -- lifecycle ----------------------------------------------------------

    async def start(
        self,
        trainer_config: Optional[TrainerConfig] = None,
        stream_configs: Optional[List[StreamConfig]] = None,
    ) -> "Manager":
        """Start the ROME manager: bring up the streams, then the trainer.

        Order matters. Streams come up first so they are already draining
        requests by the time the first checkpoint lands, and the checkpoint hook
        is registered before the trainer can fire a round — otherwise a very
        fast first round could publish weights nobody is listening for.
        """
        if trainer_config is not None:
            self.trainer.config = trainer_config
        if stream_configs is not None:
            self._stream_configs = list(stream_configs)

        await self._ensure_asyncflow()
        self.stop_event.clear()

        if self.auto_reload_streams:
            self.trainer.on_checkpoint(self.stream.on_checkpoint)

        for config in self._stream_configs:
            if config.kind is StreamKind.REWARD and config.on_output is None:
                # A reward stream's whole point is to score things for training,
                # so its results go straight into the corpus unless the workflow
                # has said otherwise.
                config.on_output = self._reward_output_to_corpus
            await self.stream.start(config)

        if self.trainer.config.trainer is not None:
            await self.trainer.start()

        self._started = True
        log.info("started — %d stream group%s, min_samples=%d, trainer %s",
                 len(self._stream_configs),
                 "" if len(self._stream_configs) == 1 else "s",
                 self.data.config.min_samples,
                 type(self.trainer.config.trainer).__name__
                 if self.trainer.config.trainer is not None else "none")
        return self

    async def _ensure_asyncflow(self) -> None:
        """Build an engine if the workflow did not hand one over.

        Deferred to :meth:`start` rather than done in ``__init__`` because
        ``WorkflowEngine.create`` is a coroutine, and because a host that means
        to share its own engine may not have started it yet when ROME is
        constructed.
        """
        if self.asyncflow is not None:
            return
        self.asyncflow = await WorkflowEngine.create(backend=self.backend)
        self._owns_asyncflow = True
        # The sub-managers were constructed before the engine existed.
        self.trainer.asyncflow = self.asyncflow
        self.stream.asyncflow = self.asyncflow

    async def stop(self, wait: bool = True, timeout: float = 300.0) -> None:
        """Stop the ROME manager.

        Trainer first: an in-flight round is waited out rather than cancelled,
        and letting the streams keep serving while it finishes means the final
        checkpoint still reaches them.
        """
        self.stop_event.set()
        log.info("stopping — corpus %d, %d round%s completed, model v%d",
                 self.data.total_count, self.trainer.rounds_completed,
                 "" if self.trainer.rounds_completed == 1 else "s",
                 self.model_version)
        await self.trainer.stop(wait_for_stop=wait, timeout=timeout)
        await self.stream.stop(wait_for_stop=wait, timeout=timeout)
        # Each stream group owns a dictionary; the manager's teardown is where
        # they are released, after the streams have finished draining.
        self.stream.close()
        if self._owns_asyncflow and self.asyncflow is not None:
            # Only an engine ROME built is ROME's to shut down; the host
            # workflow's is still running its own tasks.
            await self.asyncflow.shutdown()
            self.asyncflow = None
            self._owns_asyncflow = False
        self._started = False

    async def __aenter__(self) -> "Manager":
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # -- data API -----------------------------------------------------------

    def add_training_data(self, prompt: Any = None, completion: Any = None,
                          score: Optional[float] = None, **fields: Any) -> Optional[str]:
        """Add training data to the manager, from anywhere in the workflow.

        The ``prompt``/``completion``/``score`` triple is the LLM-shaped
        convenience form; any other domain passes keywords instead::

            rome.add_training_data(sequence=seq, pdb_path=pdb, score=plddt)

        Returns the record uid, or ``None`` when the data manager's filters
        rejected the sample.
        """
        if prompt is not None:
            fields["prompt"] = prompt
        if completion is not None:
            fields["completion"] = completion
        return self.data.add(score=score, **fields)

    def add_training_batch(self, records: List[Dict[str, Any]]) -> List[str]:
        """Add many scored outputs at once; returns the accepted uids."""
        return self.data.add_batch(records)

    def get_training_dataset(self, as_hf_dataset: bool = False) -> Any:
        """The dataset the next training round would train on."""
        return self.data.get_dataset(as_hf_dataset=as_hf_dataset)

    def _reward_output_to_corpus(self, record: Dict[str, Any]) -> None:
        """Default wiring from a reward stream into the corpus.

        A reward stream's ``process_func`` may return a bare score or a dict of
        fields; both become a corpus record. Anything else is left alone —
        ROME should not guess at a shape it does not recognise.
        """
        result = record.get("result")
        if isinstance(result, dict):
            if "error" in result and len(result) == 1:
                return
            self.data.add_batch([result])
        elif isinstance(result, (int, float)):
            self.data.add(score=float(result), request_id=record.get("request_id"))

    # -- training API -------------------------------------------------------

    async def start_training(self, force: bool = True, **kwargs: Any) -> Optional[str]:
        """Trigger a training round by hand; returns the new checkpoint path."""
        return await self.trainer.train(force=force, **kwargs)

    def get_training_status(self) -> TrainerStatus:
        """Get the status of the training tasks."""
        return self.trainer.status

    def get_current_model(self) -> Optional[str]:
        """Path to the newest published checkpoint, or ``None`` before round 1."""
        return self.trainer.get_current_model()

    @property
    def model_version(self) -> int:
        """Version of the newest published checkpoint (0 = never trained)."""
        return int(self.ns.get(MODEL_VERSION_KEY, 0))

    def ready_to_train(self) -> bool:
        """Whether enough fresh data has accumulated for a round."""
        return self.data.ready_to_train()

    # -- stream API ---------------------------------------------------------

    def submit(self, payload: Any, stream: Optional[str] = None) -> str:
        """Queue one inference or scoring request; returns its request id."""
        return self.stream.submit(payload, stream=stream)

    def get_outputs(self, stream: Optional[str] = None,
                    consume: bool = True) -> List[Dict[str, Any]]:
        """Collect finished stream results."""
        return self.stream.get_outputs(stream=stream, consume=consume)

    async def reload_model(self, model_path: Optional[str] = None,
                           wait_for_reload: bool = True) -> None:
        """Push a checkpoint into the running streams."""
        await self.stream.reload_model(model_path, wait_for_reload=wait_for_reload)

    def get_stream_status(self, stream: Optional[str] = None) -> List[StreamStatus]:
        """Status of the running stream tasks."""
        return self.stream.get_status(stream=stream)

    # -- introspection ------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    def report(self) -> Dict[str, Any]:
        """One-call snapshot of all three components."""
        return {
            "started": self._started,
            "data": {
                "total": self.data.total_count,
                "unconsumed": self.data.unconsumed_count,
                "ready_to_train": self.data.ready_to_train(),
            },
            "training": self.trainer.report(),
            "streams": self.stream.report(),
            "model_path": self.ns.get(MODEL_PATH_KEY),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<rome.Manager started={self._started} "
            f"records={self.data.total_count} "
            f"model_version={self.model_version}>"
        )
