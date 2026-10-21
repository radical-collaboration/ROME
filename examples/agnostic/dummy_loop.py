"""ROME end to end with no model: a dummy inference stream and a dummy trainer.

The smallest complete demonstration of the closed loop. An inference stream
serves placeholder outputs, the workflow scores them into the corpus, and once
enough accumulate the training manager runs a round that sleeps instead of
fine-tuning --- and the *same running stream* starts answering from the new
checkpoint, without this script orchestrating the handover.

What to watch in the output: ``model v`` climbs from 0, and it climbs while the
stream keeps serving. The stream is never restarted.

    $ dragon examples/agnostic/dummy_loop.py                 # LocalExecutionBackend
    $ ROME_BACKEND=dragon dragon -s examples/agnostic/dummy_loop.py   # real placement

Everything real about this run is the machinery: tasks are placed by the
workflow engine, state crosses the DDict, the checkpoint is a file that is
genuinely written and genuinely read back. Only the model and the gradient
steps are fake, which makes this the right thing to run first on a new backend
or a new allocation.

``ROME_BACKEND=dragon`` swaps ``LocalExecutionBackend`` (task bodies as threads
in this process) for ``rhapsody``'s ``DragonExecutionBackend`` (task bodies in
real processes/nodes). Two knobs matter there, both because a stream is a
never-returning service task that holds a slot for the whole run:

* ``ROME_STREAM_REPLICAS`` (default 1) — keep it below the allocation's
  concurrent-task capacity so a slot is left for the training round. On a small
  node that capacity is ~2; measure yours with
  ``tests/dragon/test_task_capacity_dragon.py``.
* ``ROME_GPUS`` (default 0) — GPUs to request per task. Leave at 0 on a
  GPU-less node, or the task is accepted and never placed.
"""

import asyncio
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine

import rome
from rome.dummy import DummyTrainer, dummy_infer, dummy_load

ROUNDS = 12
PROMPTS_PER_ROUND = 4
GPUS = int(os.environ.get("ROME_GPUS", 0))
STREAM_REPLICAS = int(os.environ.get("ROME_STREAM_REPLICAS", 2))


async def _build_backend():
    """LocalExecutionBackend by default; DragonExecutionBackend on request."""
    if os.environ.get("ROME_BACKEND", "local").lower() == "dragon":
        try:
            from rhapsody.backends import DragonExecutionBackend
        except ImportError:
            from rhapsody.backends import DragonExecutionBackendV3 as DragonExecutionBackend

        return await DragonExecutionBackend(
            {"results_ddict_mem": int(os.environ.get("ROME_RESULTS_MEM",
                                                     512 * 1024 ** 2))}
        )
    from radical.asyncflow import LocalExecutionBackend

    return await LocalExecutionBackend(ThreadPoolExecutor())


async def main():
    backend = await _build_backend()
    flow = await WorkflowEngine.create(backend=backend)
    checkpoint_dir = tempfile.mkdtemp(prefix="rome_dummy_")

    # On the Dragon backend one replica leaves a slot for the round on a small
    # node; on LocalExecutionBackend replicas are threads, so use two.
    replicas = STREAM_REPLICAS
    if os.environ.get("ROME_BACKEND", "local").lower() == "dragon":
        replicas = int(os.environ.get("ROME_STREAM_REPLICAS", 1))

    manager = rome.Manager(
        flow,
        # One iteration's worth of prompts is enough for a round, so the
        # first swap lands within a couple of iterations and every second
        # iteration after that triggers another.
        data_config=rome.DataConfig(min_samples=PROMPTS_PER_ROUND),
        trainer_config=rome.TrainerConfig(
            trainer=DummyTrainer(train_seconds=1.0, gpus=GPUS),
            checkpoint_dir=checkpoint_dir,
            poll_interval=0.5,
            # On the Dragon execution backend a finished round's result can go
            # undelivered while a stream service task is running (a rhapsody
            # monitor limit — see docs/dragon.md). The round still writes its
            # checkpoint, so after this grace ROME publishes from disk. The
            # default is 60s, right for a multi-minute real round but longer
            # than this whole demo; a 1s dummy round wants a few seconds.
            result_fallback_seconds=float(os.environ.get("ROME_FALLBACK", 4)),
        ),
        stream_configs=[
            rome.StreamConfig(
                name="infer",
                kind=rome.StreamKind.INFERENCE,
                load_func=dummy_load,
                process_func=dummy_infer,
                load_kwargs={"latency": 0.05},   # pretend generation cost
                num_streams=replicas,
                num_gpus=GPUS,
                batch_size=PROMPTS_PER_ROUND,
                poll_interval=0.05,
            )
        ],
    )
    await manager.start()

    print(f"checkpoints -> {checkpoint_dir}\n")

    try:
        for round_index in range(ROUNDS):
            # Ask the inference stream for a batch of outputs.
            request_ids = manager.stream.submit_batch(
                [f"prompt-{round_index}-{i}" for i in range(PROMPTS_PER_ROUND)]
            )

            # Collect them and score them into the corpus. In a real campaign
            # the score comes from a simulation or a reward model; here it is
            # noise, because nothing downstream depends on its value.
            for request_id in request_ids:
                record = await manager.stream.get_output(request_id, timeout=30.0)
                if record is None:
                    print(f"  request {request_id[:8]} timed out")
                    continue
                manager.add_training_data(
                    completion=record["result"],
                    score=random.random(),
                    produced_by_version=record["model_version"],
                )
                print(f"  v{record['model_version']} | {record['result']}")

            print(
                f"round {round_index}: corpus {manager.data.total_count:3d} "
                f"({manager.data.unconsumed_count} fresh) | "
                f"model v{manager.model_version} | "
                f"{manager.get_training_status().name}\n"
            )

            # A round takes 1s of "compute", so this gives the training
            # manager time to notice the threshold, run, and publish. Nothing
            # here waits on training: the stream keeps serving throughout, and
            # picks up the new checkpoint between batches on its own.
            await asyncio.sleep(1.0)

        print("final report:", manager.report())
    finally:
        # Stops the trainer first, waiting out any in-flight round rather than
        # cancelling it, then drains and stops the streams.
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
