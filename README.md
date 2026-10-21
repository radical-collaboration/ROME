# ROME

**RADICAL Optimizer for Model Enhancement** — adds online fine-tuning to an
existing HPC workflow without touching the workflow's own code.

📖 **[Documentation](https://radical-collaboration.github.io/ROME)**

## What it does

Three pluggable managers sit alongside any `radical.asyncflow` workflow:

- **Data Manager** — collects scored outputs from any node into a shared corpus
- **Training Manager** — fires training rounds automatically once enough data accumulates, publishes checkpoints
- **Stream Manager** — runs inference / reward as persistent tasks, hot-swaps weights when a new checkpoint arrives

`rome.Manager` wires the three together. Adoption is four API calls:

```python
import rome

manager = rome.Manager(
    asyncflow,                                         # your existing WorkflowEngine
    data_config=rome.DataConfig(min_samples=24),
    trainer_config=rome.TrainerConfig(trainer=MyTrainer()),
)
await manager.start()

manager.add_training_data(sequence=seq, score=plddt)  # from anywhere in the workflow
weights = manager.get_current_model()                  # None until the first round

await manager.stop()
```

## Install

```bash
pip install -e .
pip install -e '.[test]'    # for tests
```

Requires Python ≥ 3.11, `radical-asyncflow`, `dragonhpc` (on HPC).
See [Installation](docs/installation.md) for the full environment setup.

## Tests

```bash
pytest -m fast              # unit + mocked integration, no GPU needed
```

Dragon-specific checks (run on-cluster):

```bash
dragon -s tests/dragon/test_manager_dragon.py
dragon-cleanup-deprecated
```

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Closed loop with no model, under five minutes |
| [Installation](docs/installation.md) | Environment setup, what each test covers |
| [IMPRESS-R example](docs/examples/impress-r.md) | Protein-design campaign adoption walkthrough |
| [Setting up on Delta](docs/delta.md) | End-to-end HPC setup and SLURM script |
| [Fine-tuning ProteinMPNN](docs/proteinmpnn_training.md) | ProteinMPNN trainer deep-dive |
| [User Guide](docs/guide/) | One page per manager, trainers, logging |
| [Design](docs/design/) | Architecture, DDict state layout, execution |
| [Dragon notes](docs/dragon.md) | DDict caveats, result fallback, task-capacity limits |
| API Reference | Generated at build time from source |

Build the docs locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve       # http://127.0.0.1:8000
```

## Layout

```
rome/          framework (the only thing pip install ships)
  manager.py     Manager — wires the three components
  data.py        Data Manager
  stream.py      Stream Manager
  trainer.py     Training Manager
  train/         trainer tasks: base, llm (GRPO + SFT)
  dummy.py       model-free trainer and streams for smoke tests
examples/      adoption examples
  agnostic/      framework only — no IMPRESS needed
  impress_r/     IMPRESS-R protein-design integration
tests/         unit, mocked-integration, Dragon checks
docs/          MkDocs site
```
