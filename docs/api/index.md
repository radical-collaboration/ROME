# API Reference

Generated from the source at build time, so it cannot drift.

## The one class most workflows touch

[`rome.Manager`](rome/manager.md) wires the three managers together and exposes
their whole API on one object. Adoption is `Manager(...)`, `start()`,
`add_training_data()`, `get_current_model()`, `stop()`.

## Framework

| Module | Contents |
| --- | --- |
| [`rome.manager`](rome/manager.md) | `Manager` — the single object a host workflow talks to. |
| [`rome.data`](rome/data.md) | `DataManager`, `DataConfig` — the corpus and how a dataset is built from it. |
| [`rome.trainer`](rome/trainer.md) | `Trainer`, `TrainerConfig`, `TrainerStatus` — scheduling rounds, publishing checkpoints. |
| [`rome.stream`](rome/stream.md) | `Stream`, `StreamConfig`, `StreamContext`, `StreamTask`, `StreamKind`, `StreamStatus` — persistent inference and reward. |
| [`rome.utils`](rome/utils.md) | `Namespace`, `thread_handle`, `submit_task` — DDict layout and asyncflow submission. |
| [`rome.dummy`](rome/dummy.md) | `DummyTrainer`, `DummyModel` and stream functions, for model-free runs. |

## Trainers

| Module | Contents |
| --- | --- |
| [`rome.train.base`](rome/train/base.md) | `TrainTask`, `FunctionTrainer` — the interface every training algorithm implements. |
| [`rome.train.llm`](rome/train/llm.md) | `GRPOTrainer`, `GRPOConfig`, `ModelConfig`, `load_model`, `save_model`. |
| [`examples.impress_r.protein_binding.mpnn_trainer`](examples/impress_r/protein_binding/mpnn_trainer.md) | `ProteinMPNNTrainer`, `ProteinMPNNConfig`, `percentile_sampler` — the IMPRESS-R integration. |

`examples.impress_r.protein_binding.mpnn_trainer` ships with the example rather than with the framework,
because ROME is workflow agnostic and the ProteinMPNN trainer is an IMPRESS-R
integration. It is documented here because it is public API for anyone adopting
IMPRESS-R.

## Reading the reference alongside the guide

The reference is exhaustive; the guide explains what to reach for. Pair them:

* [Data Manager](../guide/data.md) ↔ `rome.data`
* [Training Manager](../guide/training.md) ↔ `rome.trainer`
* [Stream Manager](../guide/streams.md) ↔ `rome.stream`
* [Writing a trainer](../guide/trainers.md) ↔ `rome.train.base`
* [Shared state](../design/state.md) ↔ `rome.utils`
