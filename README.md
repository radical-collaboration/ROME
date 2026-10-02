# ROME

The RADICAL Optimizer for Model Enhancement (ROME).

📖 **[Documentation](https://radical-collaboration.github.io/ROME)** — usage guide, design
notes and API reference. Build it locally with
`pip install -r docs/requirements.txt && mkdocs serve`.

## What it is

ROME is workflow agnostic. It improves the model your workflow uses through a
set of pluggable modules you add to that workflow, rather than a workflow you
adopt. Each component is a configurable unit, adding a new model or training
algorithm is a single change, and adoption costs a few API calls.

### The three managers

| Component | What it does | API |
| --- | --- | --- |
| **Data Manager** (`rome.data`) | Collects scored outputs from the host workflow and builds them into a training dataset. Records from any node are synchronized automatically. | `add_training_data`, `get_training_dataset` |
| **Stream Manager** (`rome.stream`) | Runs inference and reward as persistent asynchronous tasks using workflow-supplied code, and reloads the model when a new checkpoint is published. | `submit`, `get_outputs`, `reload_model` |
| **Training Manager** (`rome.trainer`) | Schedules training tasks on HPC, publishes updated checkpoints back to the workflow, and reports whether training is possible, running, or finished. | `start_training`, `get_training_status`, `get_current_model` |

`rome.Manager` wires the three together. The training manager's checkpoint
callback is the stream manager's reload hook: finishing a round *is* the event
that swaps inference onto the new model.

### Quickstart

See **[docs/quickstart.md](https://github.com/radical-collaboration/ROME/blob/main/docs/quickstart.md)** — runs the closed loop with no model in under five minutes.

### Adoption

```python
import rome
from examples.impress_r.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer, percentile_sampler

manager = rome.Manager(
    asyncflow,                                  # your existing WorkflowEngine
    data_config=rome.DataConfig(
        min_samples=24,
        sample_func=percentile_sampler(0.33),   # train on the campaign's best third
    ),
    trainer_config=rome.TrainerConfig(
        trainer=ProteinMPNNTrainer(ProteinMPNNConfig(
            mpnn_repo="/path/to/dauparas/ProteinMPNN",   # the repo IMPRESS runs
            publish_into_repo=True,                      # so the next pass runs it
        )),
    ),
)
await manager.start()

# ... your workflow runs unchanged ...
manager.add_training_data(sequence=seq, pdb_path=pdb, score=plddt)
weights = manager.get_current_model()           # None until the first round

await manager.stop()
```

Training starts automatically once `min_samples` fresh records accumulate, or
on demand via `await manager.start_training()`.

To have ROME also manage inference, add stream configs:

```python
manager = rome.Manager(
    asyncflow,
    stream_configs=[
        rome.StreamConfig(name="generate", load_func=my_load, process_func=my_infer,
                          num_streams=4, num_gpus=1),
        rome.StreamConfig(name="score", kind=rome.StreamKind.REWARD,
                          process_func=my_reward, num_streams=2),
    ],
    ...
)
```

Reward-stream outputs feed the data manager automatically.

### DataConfig knobs

| Field | Default | What it does |
| --- | --- | --- |
| `min_samples` | `32` | Unconsumed records needed before a round can start. |
| `max_records` | `None` | Soft corpus cap; oldest evicted on `add`. Leave unset on Dragon — see `docs/dragon.md`. |
| `score_key` | `"score"` | Record field holding the scalar quality score. |
| `min_score` | `None` | Reject records below this on the way in. |
| `filter_func` | `None` | Admission predicate `(record) -> bool`, applied after `min_score`. |
| `dedup_key` | `None` | `(record) -> hashable` identity; duplicates are dropped. |
| `sample_func` | `None` | Full shard builder; overrides `sampling`. |
| `sampling` | `'all'` | Built-in shard strategy: `'all'`, `'top_k'` (best by `score_key`), or `'recent'`. |
| `shard_size` | `None` | Records the built-in samplers draw; `None` means no limit. |
| `consume_on_train` | `True` | Mark records consumed after a round, so the next waits for `min_samples` fresh ones. |
| `metadata` | `{}` | Extra fields stamped onto every record (run id, campaign name, …). |

### Adding a training algorithm

Subclass `TrainTask`, implement one method, declare what it needs:

```python
class MyTrainer(rome.TrainTask):
    def train(self, dataset, output_dir, **kwargs) -> str:
        ...                       # dataset is what the data manager built
        return output_dir         # path the streams will reload from

rome.TrainerConfig(trainer=MyTrainer(gpus=4, nodes=2))
```

A bare `(dataset, output_dir, **kwargs) -> checkpoint_path` function works too —
it is wrapped in a `FunctionTrainer` for you. The LLM trainers ship with ROME —
`rome.train.llm.GRPOTrainer` (TRL/GRPO) and `rome.train.llm.SFTTrainer`
(supervised fine-tuning on chosen responses) — while
`examples.impress_r.mpnn.ProteinMPNNTrainer` (IMPRESS-R) lives with its example.

#### Which trainer should I use?

| Situation | Trainer |
| --- | --- |
| You have prompts, sampled completions, and a scalar reward signal | `GRPOTrainer` |
| You have good (prompt, completion) pairs and want the model to imitate them | `SFTTrainer` |
| You want rejection-sampling / STaR: generate, keep the correct ones, fine-tune | `SFTTrainer` (selection lives in the data manager's sampler) |
| You are fine-tuning ProteinMPNN in an IMPRESS campaign | `ProteinMPNNTrainer` |
| You want to smoke-test the loop on a new backend with no model | `DummyTrainer` |
| You have an existing training script to run as a subprocess | Subclass `TrainTask`, implement `as_command` |

#### GRPO vs SFT

Both use the same three managers. The difference is what the corpus holds and
what the training objective is.

**GRPO** — reinforcement learning on sampled completions. An inference stream
generates multiple completions per prompt, a reward stream scores them, and GRPO
updates the model to raise the expected reward. Records need a `prompt` field
(or whatever `GRPOConfig.prompt_column` names) and a `score`:

```python
from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

trainer = GRPOTrainer(
    GRPOConfig(
        model_config=ModelConfig(
            base_model_name="meta-llama/Llama-3.1-8B-Instruct",
            lora_name="./adapters/lora",
        ),
        reward_funcs=[my_inline_reward],   # cheap rewards go here
        num_generations=4,
    ),
    gpus=4,
)
manager.add_training_data(prompt=prompt, score=reward)
```

**SFT** — supervised fine-tuning on chosen completions. The model learns to
imitate `(prompt, completion)` pairs; the selection step is the data manager's
sampler, not the training objective. This is the STaR / rejection-sampling
pattern: generate, keep the ones that pass a bar, fine-tune on those:

```python
from rome.train.llm import SFTConfig, SFTTrainer, ModelConfig

trainer = SFTTrainer(
    SFTConfig(
        model_config=ModelConfig(
            base_model_name="meta-llama/Llama-3.1-8B-Instruct",
            lora_name="./adapters/lora",
        ),
    ),
    gpus=4,
)
# store only the completions you want the model to imitate
manager.add_training_data(prompt=prompt, completion=chosen, score=reward)
```

The corpus sampler (`sample_func` or the built-in `top_k`) decides which records
reach a training round; SFT trains on whatever the sampler picked.

#### `as_command` and the `train_complete` contract

`TrainTask.as_command(dataset, output_dir, **kwargs)` is an optional override.
Return `(shell_command, checkpoint_path)` to have the training manager run the
round as a subprocess instead of pickling `train` into a worker. This is the
right form for a GPU fine-tune: the subprocess exits when the round ends and
releases VRAM immediately.

The command must write a file named `train_complete`
(`rome.trainer.TRAIN_COMPLETE_MARKER`) into `output_dir` as its **final** action,
after the checkpoint is on disk. The training manager polls for that file because
on Dragon a finished task can fail to deliver its result — a running stream
service blocks result delivery (see `docs/dragon.md`). With `publish_into_repo=True`
the checkpoint path already exists from the prior round, so the marker is the
only unambiguous completion signal.

Return `None` (the default) to use the plain `train` function path.

### Runtime

ROME schedules nothing itself. Training rounds and stream tasks are submitted
to the `radical.asyncflow` `WorkflowEngine` the host workflow passes in, with
per-task resources given as an asyncflow `task_description`.

Shared state lives in Dragon `DDict`s. The manager's DDict holds the corpus and
the published checkpoint — pass your own via `Manager(..., ddict=...)` and ROME
namespaces its keys under `rome|` so nothing collides with the workflow's own.
Each stream group gets a **separate** dictionary for its request and result
queues, so the cost of a replica's poll does not grow with the corpus; supply
`StreamConfig.ddict` to use one you already own.

Completed outputs land in the group's `out` sub-namespace keyed by request id;
`get_outputs(name)` on the manager drains them. Reward-stream outputs are piped
into the data manager automatically — the host workflow only calls `get_outputs`
for its own inference streams.

`TrainerConfig.on_checkpoint` is called with `(checkpoint_path, version)` after
each successful round. `Manager` wires the stream manager's reload into this
hook automatically. Add your own to be notified when a new model is published:

```python
rome.TrainerConfig(
    trainer=my_trainer,
    on_checkpoint=lambda path, ver: print(f"new model v{ver} at {path}"),
)
```

`StreamConfig.auto_reload` (default `True`) means each stream replica watches
the published model version and calls `load_func` on its own when a newer
checkpoint appears — no manual handover coordination needed.

### Use case: IMPRESS-R

IMPRESS runs backbone → ProteinMPNN → structure prediction → pLDDT/pTM/pAE →
keep/fallback/migrate/drop. It is open loop: each campaign improves the designs,
never the model. IMPRESS-R closes that loop — the campaign's highest-confidence
sequences fine-tune ProteinMPNN mid-campaign and the improved model returns to
the pipeline. IMPRESS itself runs unchanged.

See `examples/agnostic/impress_r.py` (data + training),
`examples/agnostic/llm_grpo_streams.py` (all three managers, GRPO), and
`examples/agnostic/llm_sft_streams.py` (all three managers, SFT on the model's
own correct answers — rejection sampling / STaR). For the protein side,
`examples/impress_r/mpnn.py` is the ProteinMPNN trainer and
`examples/impress_r/mpnn_stream.py` runs ProteinMPNN design as an inference
stream — supporting both a ROME-native path (`protein_mpnn_run.py`) and an
IMPRESS wrapper (`mpnn_wrapper.py`) — and hot-swaps onto each published
checkpoint.

`examples/impress_r/dummy_adaptive_rome.py` is the smallest version of the
integration: IMPRESS's own dummy adaptive example with **two lines of ROME**
added inside `adaptive_fn` — `add_training_data` to contribute a generation's
designs, `get_current_model` to collect the improved model — and the
`DummyTrainer` running a round on its own once enough designs arrive. Start here.

`examples/impress_r/protein_binding_rome.py` is the real campaign:
IMPRESS's own `run_protein_binding.py` driving the real `ProteinBindingPipeline`
(MPNN → AlphaFold → pLDDT extraction, the migration logic, all of it), with the
two ROME calls added inside `adaptive_decision` and nothing else changed. It
fine-tunes ProteinMPNN and publishes the new weights back into the checkout so
the next pass runs them. Run it from the usecase directory on Delta; the hook
wiring is covered offline by `tests/unit/test_impress_r_hooks.py`.

`examples/impress_r/adaptive_rome.py` is the same seam with executables stubbed,
so it runs anywhere. `docs/impress.md` covers installing IMPRESS from the
`archive/ipdps_pdz_usecase` branch and both halves of the integration.

`docs/proteinmpnn_training.md` covers the trainer: it fine-tunes the **original
`dauparas/ProteinMPNN`** — the same implementation IMPRESS runs — on the
campaign's dimers (designed chain scored, target peptide as context), and
publishes an original-format checkpoint straight into the repo's weights
directory so the next pass runs it.

### Trying it without a model

`rome.dummy` provides a trainer and inference/reward functions that exercise the
full machinery without a model or GPU:

| Name | Kind | What it does |
| --- | --- | --- |
| `DummyTrainer` | `TrainTask` subclass | Sleeps instead of fine-tuning; writes a real checkpoint file. |
| `DummyModel` | class | Generates placeholder strings; reports its version from the checkpoint. |
| `dummy_load` | `load_func` | Constructs a `DummyModel` from a checkpoint path; pass as `StreamConfig.load_func`. |
| `dummy_infer` | `process_func` | Calls `DummyModel.generate_batch`; pass as `StreamConfig.process_func`. |
| `dummy_reward` | `process_func` | Emits constant-score corpus records; stand-in for a real reward stream. |
| `write_dummy_checkpoint` | helper | Writes the JSON checkpoint a `DummyTrainer` round produces. |
| `read_dummy_checkpoint` | helper | Reads a dummy checkpoint; returns version 0 when none exists yet. |

Everything else is real: tasks are placed by the workflow engine, state crosses
the DDict, and the checkpoint is a file genuinely written and read back. Run
this first on any new backend or allocation.

## Layout

```
rome/            ROME
  manager.py       Manager — wires the three components together
  data.py          Data Manager
  stream.py        Stream Manager
  trainer.py       Training Manager
  train/           trainer tasks (base; llm — GRPO + SFT)
  utils.py         DDict layout helpers + asyncflow submission
  dummy.py         model-free trainer and streams, for smoke tests
examples/        ROME adoption examples
  agnostic/        framework only — no IMPRESS needed
  impress_r/       the IMPRESS-R integration: trainer, seams, campaign tooling
tests/           unit, mocked-integration and Dragon checks
```

**`rome/` is the framework** — the only thing `pip install` ships. Everything
beside it is there to be read or run, not imported. `examples/impress_r/` carries
everything specific to that campaign: the ProteinMPNN trainer, the pipeline
seams, and three operational tools. See `docs/impress.md` for what each is for —
`populate_best_models.py` in particular is needed whenever IMPRESS runs off
RadicalExecutionBackend.

## Running it on a cluster

`docs/delta.md` is the end-to-end setup: environment, installing Dragon, ROME
and IMPRESS, a smoke-test ladder that proves one layer at a time, a Slurm
script, and what still needs swapping in for a real campaign.

## Tests

```bash
pip install -e '.[test]'
pytest -m fast      # unit + mocked integration, no GPUs
```

Dragon-specific checks are scripts, not pytest modules, because the Dragon
launcher runs a script rather than a test session:

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop, 4 replicas
dragon-cleanup-deprecated                         # after every Dragon run
```

Four of those Dragon scripts import no ROME at all — they probe Dragon,
rhapsody and the allocation itself, and are the reproducers behind the findings
in `docs/dragon.md`: why `max_records` carries a warning, why
`result_fallback_seconds` exists, and why a stream replica count has to stay
under the allocation's task capacity. `docs/installation.md` lists what every
test covers.

## Documentation

The full site is built with MkDocs from `docs/`:

```bash
pip install -r docs/requirements.txt
mkdocs serve                 # http://127.0.0.1:8000
mkdocs build --strict        # what CI runs
```

| Section | What is in it |
| --- | --- |
| [Quickstart](https://github.com/radical-collaboration/ROME/blob/main/docs/quickstart.md) | Runs the closed loop with no model in under five minutes. |
| [Installation](https://github.com/radical-collaboration/ROME/blob/main/docs/installation.md) | Environment setup and what every test covers. |
| User Guide | One page per manager, plus writing a trainer and reading the logs. |
| [Design](https://github.com/radical-collaboration/ROME/blob/main/docs/design/architecture.md) | Architecture, the DDict state layout, and how tasks reach the execution backend. |
| [IMPRESS integration](https://github.com/radical-collaboration/ROME/blob/main/docs/impress.md) | Installing IMPRESS, both halves of the IMPRESS-R integration, and operational tooling. |
| [ProteinMPNN training](https://github.com/radical-collaboration/ROME/blob/main/docs/proteinmpnn_training.md) | Fine-tuning the original `dauparas/ProteinMPNN` on campaign dimers. |
| [HPC / Delta](https://github.com/radical-collaboration/ROME/blob/main/docs/delta.md) | End-to-end cluster setup, Slurm script, and smoke-test ladder. |
| [Dragon notes](https://github.com/radical-collaboration/ROME/blob/main/docs/dragon.md) | DDict caveats, `result_fallback_seconds`, and task-capacity limits. |
| API Reference | Generated from the source at build time, so it cannot drift. |

The API reference needs none of ROME's runtime dependencies — mkdocstrings reads
the source statically — so the docs build on any machine.
