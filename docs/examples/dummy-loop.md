# The dummy loop

`examples/agnostic/dummy_loop.py`

The smallest complete demonstration of ROME's closed loop, with no model in it.
An inference stream serves placeholder outputs, the script scores them into the
corpus, and once enough accumulate the training manager runs a round that sleeps
instead of fine-tuning — and the *same running stream* starts answering from the
new checkpoint, without the script orchestrating the handover.

```bash
dragon examples/agnostic/dummy_loop.py
```

A full walkthrough of the output and each step is in the
[Quickstart](../quickstart.md). This page is about why the example exists.

## Everything real about it is the machinery

The reason to run this first on a new backend or allocation is that only the
model and the gradient steps are fake. Everything else fails in exactly the
places a real run would:

| Fake | Real |
| --- | --- |
| `DummyTrainer` sleeps for `train_seconds` instead of fine-tuning. | The round is submitted to the workflow engine and placed on the allocation like any other task. |
| `DummyModel.generate()` returns `model example output [<uuid>]`. | The stream is a persistent asyncflow service holding a real execution slot. |
| The reward is `random.random()`. | The corpus is a real Dragon DDict, written from the driver and read by the training task. |
| — | The checkpoint is **a file that is genuinely written and genuinely read back**. |

So an unwritable checkpoint directory, a DDict that is not actually shared, a
task that is accepted but never placed, or a stream that never sees a publication
all fail here, cheaply, before a GPU or a real model is involved.

## The dummy checkpoint is a real file

This is the detail that makes the demo meaningful. `write_dummy_checkpoint()`
writes a small JSON file holding the round's version, and `DummyModel` reads its
version *back out of that file* rather than being told what it is:

```json
{"version": 3, "samples": 28, "trained_at": 1724978401.2}
```

So when the output flips from `v2` to `v3`, that is a reload having genuinely
occurred and genuinely found new bytes on disk — not a counter incremented in the
driver process.

`read_dummy_checkpoint()` tolerates a checkpoint that was never written, reporting
version 0. A stream starts before the first round completes, so "no checkpoint
yet" is the normal initial state, not an error.

## Why the dummies block

Both `DummyTrainer.train` and `DummyModel.generate` use `time.sleep`. That is
correct rather than sloppy: ROME runs a synchronous trainer inside an asyncflow
task and a synchronous `process_func` in a worker thread, so neither sleep stalls
the event loop the other streams are sharing. Writing them `async` would test
something ROME does not actually do.

The sleep is also what makes the machinery *observable*. A round that takes a
measurable second is long enough to watch `RUNNING` appear, watch the streams keep
serving through it, and watch `stop()` wait it out rather than cancelling it.

## Knobs

| Variable | Default | Notes |
| --- | --- | --- |
| `ROME_BACKEND` | `local` | `dragon` swaps `LocalExecutionBackend` for `DragonExecutionBackend`. |
| `ROME_STREAM_REPLICAS` | 2 local / 1 Dragon | Keep below the allocation's concurrent-task capacity, or no slot is left for the round. |
| `ROME_GPUS` | 0 | Leave at 0 on a GPU-less node, or tasks are accepted and never placed. |
| `ROME_FALLBACK` | 4 | `result_fallback_seconds`. The 60 s default is right for a real round and longer than this whole demo. |
| `ROME_RESULTS_MEM` | 512 MiB | Dragon backend results dictionary size. |

## Using the dummies in your own run

`rome.dummy` is importable, and standing one half of a campaign in while you
build the other is a legitimate use:

```python
from rome.dummy import DummyTrainer, dummy_infer, dummy_load, dummy_reward

# real streams, fake training
rome.TrainerConfig(trainer=DummyTrainer(train_seconds=30.0))

# real training, fake inference
rome.StreamConfig(name="infer", load_func=dummy_load, process_func=dummy_infer)

# real inference, fake scoring — accumulates a corpus with no scoring code at all
rome.StreamConfig(name="score", kind=rome.StreamKind.REWARD, process_func=dummy_reward)
```

`DummyTrainer(fail_every=3)` raises on every third round, which is how the
training manager's failure handling is exercised without an actual failure.

API reference: [`rome.dummy`](../api/rome/dummy.md).
