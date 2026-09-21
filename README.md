# ROME

The RADICAL Optimizer for Model Enhancement (ROME).

📖 **[Documentation](https://iznoanygod.github.io/ROME)** — usage guide, design
notes and API reference. Build it locally with
`pip install -r docs/requirements.txt && mkdocs serve`.

## What it is

ROME is workflow agnostic. Model improvement is a set of pluggable modules you
add to a workflow you already have, rather than a workflow you adopt. Each
component is a configurable unit, adding a new model or training algorithm is a
single task, and adoption costs a few API calls.

### The three managers

| Component | What it does | API |
| --- | --- | --- |
| **Data Manager** (`rome.data`) | Collects scored outputs from the host workflow and builds them into a training dataset. Handles organization and synchronization of the data across nodes and tasks, so a task anywhere in the workflow can contribute. | `add_training_data`, `get_training_dataset` |
| **Stream Manager** (`rome.stream`) | Runs inference and reward as persistent asynchronous tasks using workflow-supplied code, and reloads the model when it sees a new checkpoint. | `submit`, `get_outputs`, `reload_model` |
| **Training Manager** (`rome.trainer`) | Creates and schedules training tasks on HPC, publishes updated checkpoints back to the workflow, and reports whether training is possible, running or finished. | `start_training`, `get_training_status`, `get_current_model` |

`rome.Manager` wires the three together. The line that closes the loop is the
training manager's checkpoint callback being the stream manager's reload hook:
a completed training round *is* the event that swaps inference onto the new
model.

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

Workflows that also want ROME to own inference add stream configs:

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

### Runtime

ROME schedules nothing itself. Training rounds and stream tasks are submitted
to the `radical.asyncflow` `WorkflowEngine` the host workflow passes in, with
per-task resources given as an asyncflow `task_description`.

Shared state lives in Dragon `DDict`s. The manager's holds the corpus and the
published checkpoint — pass your own via `Manager(..., ddict=...)` and ROME
namespaces its keys under `rome|` so nothing collides with the workflow's own.
Each stream group gets a **separate** dictionary for its request and result
queues, so the cost of a replica's poll does not grow with the corpus; supply
`StreamConfig.ddict` to use one you already own.

### Use case: IMPRESS-R

IMPRESS runs backbone → ProteinMPNN → structure prediction → pLDDT/pTM/pAE →
keep/fallback/migrate/drop. It is open loop: each campaign improves the designs,
never the model. IMPRESS-R adds ROME so the campaign's own highest-confidence
sequences fine-tune ProteinMPNN mid-campaign, and the improved model returns to
the pipeline. IMPRESS itself runs unchanged.

See `examples/agnostic/impress_r.py` (data + training),
`examples/agnostic/llm_grpo_streams.py` (all three managers, GRPO), and
`examples/agnostic/llm_sft_streams.py` (all three managers, SFT on the model's
own correct answers — rejection sampling / STaR). For the protein side,
`examples/impress_r/mpnn.py` is the ProteinMPNN trainer and
`examples/impress_r/mpnn_stream.py` runs ProteinMPNN design as an inference
stream (a ROME-native `protein_mpnn_run.py` path and an IMPRESS `mpnn_wrapper.py`
path) that hot-swaps onto each published checkpoint — the generation half of the
same loop.

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

`examples/impress_r/adaptive_rome.py` is the same seam with the executables
stubbed, so it runs anywhere. `docs/impress.md` covers installing IMPRESS from
the `archive/ipdps_pdz_usecase` branch and both halves of the integration.

`docs/proteinmpnn_training.md` covers the trainer: it fine-tunes the **original
`dauparas/ProteinMPNN`** — the same implementation IMPRESS runs — on the
campaign's dimers (designed chain scored, target peptide as context), and
publishes an original-format checkpoint straight into the repo's weights
directory so the next pass runs it.

### Trying it without a model

`rome.dummy` ships a trainer that sleeps instead of fine-tuning and an
inference stream that emits `model example output [<uuid>]`. Everything else in
the run is real — tasks are placed by the workflow engine, state crosses the
DDict, and the checkpoint is a file that is genuinely written and read back —
so it is the right first thing to run on a new backend or allocation:

```bash
dragon examples/agnostic/dummy_loop.py
```

```
  v0 | model example output [7af1236d-dad6-4300-a8c3-a00d53a4ca65]
round 0: corpus   4 (4 fresh) | model v0 | WAITING
  ...
  v3 | model example output [76ad7d76-0aef-4de1-a0fd-de6ba02390ad]
round 6: corpus  28 (4 fresh) | model v3 | WAITING
```

The version climbs while the stream keeps serving; it is never restarted.

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

`docs/delta.md` is the end-to-end setup: environment, installing Dragon,
ROME and IMPRESS, a smoke-test ladder that proves one layer at a time, a
Slurm script, and what still needs swapping in for a real campaign.

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
| Home | Overview, installation, and a quickstart that runs the closed loop with no model. |
| User Guide | One page per manager, plus writing a trainer and reading the logs. |
| Design | Architecture, the DDict state layout, and how tasks reach the execution backend. |
| Examples | The dummy loop, the LLM/GRPO run, and the IMPRESS-R integration. |
| HPC | The existing Delta, IMPRESS and ProteinMPNN notes. |
| API Reference | Generated from the source at build time, so it cannot drift. |

The API reference needs none of ROME's runtime dependencies — mkdocstrings
reads the source statically — so the docs build on any machine.
