# Writing a trainer

> "Adding new models and training algorithms requires just one task."

This page is that task's interface. API reference:
[`rome.train.base`][rome.train.base].

## The contract

A trainer receives a dataset built by the data manager and an output directory,
does whatever it does, and returns the path to a fresh checkpoint. The training
manager takes it from there: publishing the checkpoint, bumping the model
version, and letting the stream manager hot-swap weights.

```python
class MyTrainer(rome.TrainTask):
    def train(self, dataset, output_dir, **kwargs) -> str:
        ...                       # dataset is what the data manager built
        return output_dir         # path the streams will reload from

rome.TrainerConfig(trainer=MyTrainer(gpus=4, nodes=2))
```

That is the whole required surface. `gpus` and `nodes` are declared on the task
itself, and the training manager turns them into the workflow engine's resource
request — so the algorithm says what it needs and nothing else has to be
configured.

!!! info "Blocking is fine"

    `train` is a plain synchronous method on purpose. The training manager runs
    it off the event loop (in a worker thread, inside an asyncflow task), so a
    trainer that blocks for an hour is exactly right.

## From a bare function

The cheapest possible adoption — no subclassing at all:

```python
def my_finetune(dataset, output_dir, **kwargs) -> str:
    ...
    return output_dir

rome.TrainerConfig(trainer=my_finetune)
```

Any callable `(dataset, output_dir, **kwargs) -> checkpoint_path` is wrapped in a
[`FunctionTrainer`][rome.train.base.FunctionTrainer] automatically. A function
that returns nothing is assumed to have written its checkpoint where it was told
to, and `output_dir` is published instead.

Wrap it yourself to set resources or ask for a HuggingFace dataset:

```python
rome.TrainerConfig(trainer=rome.FunctionTrainer(my_finetune, gpus=4,
                                                wants_hf_dataset=True))
```

## What `dataset` is

Whatever [`DataManager.get_dataset()`][rome.data.DataManager.get_dataset]
produced — a list of record dicts by default:

```python
[{"uid": "...", "sequence": "MKT...", "pdb_path": "/…/d.pdb",
  "score": 91.2, "added_at": 1724978401.2, "model_version": 2}, ...]
```

Set `wants_hf_dataset = True` on the class to get a `datasets.Dataset` instead.
The GRPO trainer does, because TRL wants one; the ProteinMPNN trainer does not,
which is why it is opt-in rather than always paid for.

## Validating before you burn an allocation

```python
class MyTrainer(rome.TrainTask):
    def validate(self, dataset):
        super().validate(dataset)            # "not empty"
        if "sequence" not in dataset[0]:
            raise ValueError("MyTrainer needs a 'sequence' field")
```

`validate` runs **on the manager, before the task is scheduled**, so an unusable
corpus fails fast instead of after a queue wait. The GRPO trainer uses this to
check the prompt column exists and to say what the corpus does have:

```text
GRPO needs a 'prompt' column; the corpus has ['completion', 'score', 'uid'].
Add prompts via add_training_data(prompt=...) or set GRPOConfig.prompt_column.
```

## Running a round as a command

By default a round is a **function task**: `train` runs in a worker inside an
asyncflow task. Implement `as_command()` instead and the round becomes an
**executable task** — a shell command run as its own process on the allocation's
resources, the way IMPRESS submits its wrapper scripts.

```python
class MyTrainer(rome.TrainTask):
    def as_command(self, dataset, output_dir, **kwargs):
        job = write_job_spec(dataset, output_dir, self.config)
        checkpoint = os.path.join(output_dir, "weights.pt")
        return f"python {WRAPPER} --job {job}", checkpoint
```

Return `(shell_command, checkpoint_path)`; return `None` (the default) to stay a
function task. `checkpoint_path` has to be declared because an executable task's
return value is the backend's execution result, not your path.

**A GPU fine-tune should prefer this form.** The subprocess exits when the round
ends, so its VRAM is released with it — where an in-process round leaves a CUDA
context resident for the whole campaign.

!!! danger "The completion marker is part of the contract"

    A command **must** write a file named `train_complete`
    ([`rome.trainer.TRAIN_COMPLETE_MARKER`][rome.trainer.TRAIN_COMPLETE_MARKER])
    into its `output_dir` as its **final** action, after the checkpoint is safely
    on disk.

    The training manager polls for that marker to detect that the round finished,
    because on Dragon a task can run to completion and never resolve its future.
    The marker, not the checkpoint, is the signal: with publish-in-place the
    checkpoint path already exists from the previous round or from the initial
    weights, so its presence proves nothing.

    See [Execution](../design/execution.md#when-the-backend-never-delivers-a-result).

## Checkpoint layout

`TrainTask.prepare_output_dir(base, version, name)` gives every trainer the same
layout, and the manager calls it for you:

```text
<checkpoint_dir>/<trainer name>/v<version>/
```

`name` defaults to the class name; pass `name=` to the constructor to change it.

## Shipped trainers

### `rome.train.llm.GRPOTrainer`

TRL's GRPO over the corpus, writing a LoRA adapter when one is configured and the
full model otherwise. It is the demonstration that adding an algorithm is one
task: everything in `rome/train/llm.py` is TRL-specific, and nothing above it —
the data manager, the training manager, the stream manager — knows an LLM is
involved.

```python
from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

trainer = GRPOTrainer(GRPOConfig(
    model_config=ModelConfig(
        base_model_name="meta-llama/Llama-3.2-1B-Instruct",
        lora_name="./adapters/math",
        required_gpus=1,
    ),
    num_epochs=1,
    num_generations=4,
    reward_funcs=[my_inline_reward],
))
```

Anything else TRL accepts goes in `extra_args` (merged into the constructed TRL
config) or `trl_config` (a fully-specified `trl.GRPOConfig`, which overrides the
individual knobs).

!!! tip "Inline rewards versus reward streams"

    `reward_funcs` are called by TRL *inside* the training round, with the
    signature `fn(prompts, completions, **kwargs) -> list[float]`. Rewards that
    are expensive or need their own resources belong in a ROME
    [reward stream](streams.md#reward-streams-feed-the-corpus) instead, where
    they get their own tasks and their own nodes.

`load_model()` and `save_model()` are exported alongside it, and an inference
stream's `load_func` should use `load_model` — that is how the stream and the
trainer agree on what a checkpoint means.

### `examples.impress_r.protein_binding.mpnn_trainer.ProteinMPNNTrainer`

Fine-tunes the **original `dauparas/ProteinMPNN`** — the same implementation
IMPRESS runs — on the campaign's dimers, scoring the designed chain with the
target peptide as context, and writes an original-format checkpoint that
`protein_mpnn_run.py` loads.

```python
from examples.impress_r.protein_binding.mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer

trainer = ProteinMPNNTrainer(ProteinMPNNConfig(
    mpnn_repo="/path/to/dauparas/ProteinMPNN",
    publish_into_repo=True,      # drop new weights into the repo IMPRESS runs
))
```

It lives with the example rather than in `rome/` on purpose: it is an IMPRESS-R
*integration*, and ROME is workflow agnostic. See
[Fine-tuning ProteinMPNN](../proteinmpnn_training.md) and
[the API reference](../api/examples/impress_r/protein_binding/mpnn_trainer.md).

### `rome.dummy.DummyTrainer`

Sleeps instead of fine-tuning and writes a checkpoint recording what the round
would have learned from. The sleep is the point: it makes a round take long
enough that the surrounding machinery — the `RUNNING` status, streams continuing
to serve mid-round, a stop request waiting for the round rather than cancelling
it — is actually observable.

```python
rome.TrainerConfig(trainer=rome.dummy.DummyTrainer(train_seconds=2.0,
                                                   fail_every=3))
```

`fail_every` raises on every *n*-th round, which is how the training manager's
failure handling is exercised.

## Checklist for a new trainer

- [ ] Subclass `TrainTask` (or pass a bare function).
- [ ] Implement `train(dataset, output_dir, **kwargs) -> checkpoint_path`.
- [ ] Declare `gpus` / `nodes` so the round is placed correctly.
- [ ] Override `validate()` if the corpus needs specific fields.
- [ ] Set `wants_hf_dataset = True` if you want a `datasets.Dataset`.
- [ ] Implement `as_command()` if the round should run in its own process — and
      write the `train_complete` marker last.
- [ ] Tolerate unknown `**kwargs`; the manager always passes `model_version`.
