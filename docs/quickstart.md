# Quickstart

This page runs ROME's closed loop end to end — inference, scoring, training, a
hot weight swap — in under a minute, with no model, no GPU and no cluster.

Everything real about the run is the machinery. Tasks are placed by the workflow
engine, state crosses the Dragon dictionary, and the checkpoint is a file that is
genuinely written and genuinely read back. Only the model and the gradient steps
are fake, which is what makes this the right first thing to run on a new backend
or a new allocation.

## Run it

```bash
dragon examples/agnostic/dummy_loop.py
```

```text
checkpoints -> /tmp/rome_dummy_qk1m4d2p

  v0 | model example output [7af1236d-dad6-4300-a8c3-a00d53a4ca65]
  v0 | model example output [b1c9e0a4-2f61-4c8f-9a55-0d3e1f77c210]
round 0: corpus   4 (4 fresh) | model v0 | WAITING

  v1 | model example output [3d0a51bc-9e77-4a02-8f13-6c2b5ad9e401]
round 2: corpus  12 (4 fresh) | model v1 | NOT_ENOUGH_DATA
  ...
  v3 | model example output [76ad7d76-0aef-4de1-a0fd-de6ba02390ad]
round 6: corpus  28 (4 fresh) | model v3 | WAITING
```

**What to watch: `model v` climbs from 0, and it climbs while the stream keeps
serving.** The stream task is never restarted, never resubmitted, and the script
never orchestrates the handover. It picks up each new checkpoint between batches
on its own.

## What the script does

The whole example is ~90 lines. Its shape is the shape of every ROME adoption.

### 1. Build a manager

```python
manager = rome.Manager(
    backend=backend,                            # execution backend ROME builds its engine on
    data_config=rome.DataConfig(min_samples=PROMPTS_PER_ROUND),
    trainer_config=rome.TrainerConfig(
        trainer=DummyTrainer(train_seconds=1.0, gpus=GPUS),
        checkpoint_dir=checkpoint_dir,
        poll_interval=0.5,
        result_fallback_seconds=4,
    ),
    stream_configs=[
        rome.StreamConfig(
            name="infer",
            kind=rome.StreamKind.INFERENCE,
            load_func=dummy_load,
            process_func=dummy_infer,
            load_kwargs={"latency": 0.05},
            num_streams=replicas,
            batch_size=PROMPTS_PER_ROUND,
            poll_interval=0.05,
        )
    ],
)
await manager.start()
```

`start()` brings the streams up first, then the trainer — so the streams are
already draining requests, and the checkpoint hook is registered, before a round
can possibly fire.

### 2. Ask the stream for work

```python
request_ids = manager.stream.submit_batch(
    [f"prompt-{round_index}-{i}" for i in range(PROMPTS_PER_ROUND)]
)
```

Requests round-robin across the group's replicas. Each replica *pops* the
requests it claims, so two replicas never process the same one.

### 3. Score the results into the corpus

```python
for request_id in request_ids:
    record = await manager.stream.get_output(request_id, timeout=30.0)
    manager.add_training_data(
        completion=record["result"],
        score=random.random(),
        produced_by_version=record["model_version"],
    )
```

In a real campaign the score comes from a simulation, a structure predictor or a
reward model. Here it is noise, because nothing downstream depends on its value.

`add_training_data` is callable from *anywhere* — any task, on any node. The
records land in the shared dictionary one key each.

### 4. Do nothing about training

There is no fourth step. The training manager polls the corpus, notices it has
crossed `min_samples`, submits a round to the workflow engine, publishes the
checkpoint, and fires the callback that tells the streams to reload. The script's
loop just keeps generating and scoring:

```python
await asyncio.sleep(1.0)   # the round happens over here, unattended
```

### 5. Stop

```python
await manager.stop()
```

Trainer first — an in-flight round is *waited out*, not cancelled, because
killing a half-finished fine-tune would leave a torn checkpoint — then the
streams drain and stop.

## Reading the status line

```text
round 0: corpus   4 (4 fresh) | model v0 | WAITING
```

| Field | Meaning |
| --- | --- |
| `corpus 4` | Total accepted records. |
| `(4 fresh)` | Records added since the last round consumed the corpus. Compared against `min_samples`. |
| `model v0` | Published checkpoint version. `0` means never trained. |
| `WAITING` | Trainer status: a round is possible and about to start. `NOT_ENOUGH_DATA` means it is not yet possible; `RUNNING` means one is in flight. |

`WAITING` and `NOT_ENOUGH_DATA` are both idle, but they answer different
questions — see [`TrainerStatus`](api/rome/trainer.md#rome.trainer.TrainerStatus).

## Running it on a real backend

```bash
ROME_BACKEND=dragon dragon -s examples/agnostic/dummy_loop.py
```

This swaps `LocalExecutionBackend` (task bodies as threads in this process) for
rhapsody's `DragonExecutionBackend` (task bodies in real processes, on real
nodes). Two environment knobs matter there, both because a stream is a
never-returning service task that holds an execution slot for the whole run:

* **`ROME_STREAM_REPLICAS`** (default 1 on Dragon) — keep it below the
  allocation's concurrent-task capacity, or no slot is left for the training
  round and it is accepted but never placed. On a small node that capacity is
  ~2; measure yours with `tests/dragon/test_task_capacity_dragon.py`.
* **`ROME_GPUS`** (default 0) — GPUs requested per task. Leave at 0 on a
  GPU-less node, or the task is accepted and never placed.

A third, `ROME_FALLBACK`, sets `result_fallback_seconds`. On Dragon a finished
round's result can go undelivered while a stream service is running; ROME then
publishes from disk after that grace period. The default of 60 s is right for a
multi-minute real round and far longer than this whole demo, so the example
lowers it to 4 s. [Why that fallback exists](design/execution.md#when-the-backend-never-delivers-a-result)
is worth reading before you tune it.

## Next

* [Adopting ROME](guide/adoption.md) — the same loop, attached to a workflow
  you already have.
* [Examples](examples/index.md) — the LLM/GRPO version with all three managers,
  and the real IMPRESS-R campaign.
* [Architecture](design/architecture.md) — what each manager owns, and why the
  boundary falls where it does.
