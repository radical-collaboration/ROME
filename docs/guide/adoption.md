# Adopting ROME

ROME's design goal is that adoption costs a few API calls and the host
workflow's own code does not move. This page is what those calls are.

## The four calls

```python
import rome

# 1. setup — a data policy and a trainer
manager = rome.Manager(
    asyncflow,
    data_config=rome.DataConfig(min_samples=24),
    trainer_config=rome.TrainerConfig(trainer=MyTrainer()),
)
await manager.start()

# ... the workflow runs unchanged ...

# 2. contribute — from anywhere, on any node
manager.add_training_data(sequence=seq, pdb_path=pdb, score=plddt)

# 3. collect — the improved model, once there is one
weights = manager.get_current_model()      # None before round 1
if weights:
    my_pipeline.load(weights)

# 4. teardown
await manager.stop()
```

Calls 2 and 3 are the only ones that live inside the workflow's own loop, and in
the IMPRESS-R integration they are literally two added lines inside the campaign's
adaptive step. Everything between them — deciding a round is possible, building
the dataset, placing the training task on HPC, publishing the checkpoint — happens
without the workflow's involvement.

!!! tip "Start from the smallest version"

    `examples/agnostic/impress_r.py` is the shortest complete demonstration
    of what adoption looks like — four ROME calls against a stand-in pipeline,
    no IMPRESS dependency needed. For the full IMPRESS integration see
    `tests/unit/test_impress_r_hooks.py` and
    `examples/impress_r/protein_binding/run_protein_binding_rome.py`.

## Sharing the workflow engine

ROME schedules nothing itself. Every training round and every stream task is
submitted to a `radical.asyncflow` `WorkflowEngine`, and the point of passing in
the host's is that ROME's tasks are scheduled against the *same allocation* as
the campaign's own:

```python
manager = rome.Manager(impress_manager.flow, ...)
```

Leave it out and ROME builds its own engine at `start()` and shuts it down at
`stop()`:

```python
manager = rome.Manager(backend=my_backend, ...)
```

That is the right choice when ROME should manage its tasks independently of
the host, or when the host creates its engine internally and does not hand one
out. ROME only shuts down an engine it built — a host's engine is still running
the host's tasks.

!!! warning "Don't leave the backend unset for a GPU fine-tune"

    With neither `asyncflow` nor `backend` given, asyncflow falls back to a
    **local, in-process** backend. A training round then runs inside the driver
    process, so the model's CUDA context stays resident for the whole campaign
    instead of being freed when the round ends.

    Pass a process-based backend — `DragonExecutionBackend` on Delta, or a
    `ConcurrentExecutionBackend(ProcessPoolExecutor())` — so each round runs in a
    task process that releases its VRAM on exit. See
    `examples/impress_r/protein_binding/run_protein_binding_rome.py`, and
    [Execution](../design/execution.md).

## Sharing the dictionary

ROME keeps its shared state in a Dragon `DDict`. By default it allocates one;
pass your own to share the host workflow's:

```python
manager = rome.Manager(flow, ddict=impress_ddict, ...)
```

Sharing is safe because ROME namespaces every key it writes under `rome|`, so
nothing collides with the workflow's own state. Passing the DDict object into a
task is all it takes to reach it from another node — Dragon attaches the
receiving process automatically.

When ROME allocates its own, it asks for more room than Dragon's 3 MiB default,
because a campaign corpus of a few thousand records goes past that and the
failure mode is an allocation error deep inside a manager rather than anything
ROME can explain. Override with `ddict_kwargs`:

```python
rome.Manager(flow, ddict_kwargs={"n_nodes": 2, "total_mem": 4 * 1024**3})
```

Each *stream group* gets a **separate** dictionary for its request and result
queues, so the cost of a replica's poll does not grow with the corpus. See
[Shared state](../design/state.md#why-each-stream-group-owns-a-dictionary).

## Which halves you need

ROME's three managers are independent enough that you can take two of them.

### Data + training only

The common case when the host workflow already owns inference. IMPRESS runs its
own ProteinMPNN and AlphaFold tasks; ROME has no business duplicating them, so
IMPRESS-R uses only the data and training halves and hands the improved weights
back through `get_current_model()`.

```python
manager = rome.Manager(
    flow,
    data_config=rome.DataConfig(min_samples=24, filter_func=my_filter),
    trainer_config=rome.TrainerConfig(trainer=my_trainer),
)
```

### All three

Workflows that also want ROME to own inference add stream configs:

```python
manager = rome.Manager(
    flow,
    stream_configs=[
        rome.StreamConfig(name="generate", load_func=my_load, process_func=my_infer,
                          num_streams=4, num_gpus=1),
        rome.StreamConfig(name="score", kind=rome.StreamKind.REWARD,
                          process_func=my_reward, num_streams=2),
    ],
    data_config=...,
    trainer_config=...,
)
```

Reward-stream outputs feed the data manager automatically: a reward stream's
whole point is to score things for training, so unless you set `on_output`
yourself, its results go straight into the corpus.

### Data only

Omit `trainer_config` and the manager never trains — you get a synchronized,
cross-node corpus you can pull a dataset out of whenever you like:

```python
dataset = manager.get_training_dataset()
```

## Driving training by hand

With `auto_train` on (the default), a poll loop fires a round as soon as
`min_samples` fresh records have accumulated. Turn it off to decide yourself:

```python
trainer_config=rome.TrainerConfig(trainer=my_trainer, auto_train=False)
...
if manager.ready_to_train():
    checkpoint = await manager.start_training()
```

`start_training()` defaults to `force=True`, so it runs a round whether or not
the threshold is met. Failures propagate from a manual call — a manual trigger is
expected to surface its own errors, unlike the auto-train loop, which records
them in `get_training_status()` and keeps polling.

## Watching what it is doing

```python
manager.report()
```

```python
{
  "started": True,
  "data": {"total": 28, "unconsumed": 4, "ready_to_train": False},
  "training": {"status": "NOT_ENOUGH_DATA", "model_version": 3,
               "model_path": "/scratch/ckpt/dummy/v3", "rounds_completed": 3,
               "corpus_size": 28, "unconsumed": 4, "last_error": None},
  "streams": {"infer": {"kind": "inference", "tasks": 2,
                        "status": ["RUNNING", "RUNNING"], "pending": 0,
                        "model_version": 3}},
  "model_path": "/scratch/ckpt/dummy/v3",
}
```

ROME also logs one line per lifecycle event, styled to sit alongside IMPRESS's
own log lines. See [Logging](logging.md).

## Async context manager

`Manager` is an async context manager, so a script that owns the whole lifetime
can skip the explicit `start`/`stop`:

```python
async with rome.Manager(flow, data_config=..., trainer_config=...) as manager:
    ...
```
