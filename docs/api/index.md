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
| [`examples.impress_r.interface_targeting.itgt_trainer`](examples/impress_r/interface_targeting/itgt_trainer.md) | `ItgtSurrogateTrainer`, `ItgtSurrogateConfig`, `itgt_reward_fn` — the online surrogate for the RFD3 diffusion step. |
| [`examples.impress_r.interface_targeting.itgt_spec`](examples/impress_r/interface_targeting/itgt_spec.md) | `patch_spec`, `template_partition`, `coverage_ok` — reading and patching the RFD3 base design spec. Pure standard library. |
| [`examples.impress_r.interface_targeting.itgt_propose`](examples/impress_r/interface_targeting/itgt_propose.md) | `propose_groups`, `apply_to_pipeline`, `record_action` — turning a published surrogate into the next pass's diffusion parameters. |

The `examples.impress_r.*` modules ship with their examples rather than with the
framework, because ROME is workflow agnostic and both of these are IMPRESS-R
integrations. They are documented here because they are public API for anyone
adopting IMPRESS-R.

The `interface_targeting` modules are the one integration that improves
something other than a model: they learn the `select_buried`/`select_exposed`
partition RFD3's backbone diffusion is conditioned on, which every campaign has
so far hand-authored and frozen. `itgt_spec` is deliberately dependency-free, so
the spec-patching surface — where most of that integration's correctness lives —
can be tested and read without a scientific stack.

## Reading the reference alongside the guide

The reference is exhaustive; the guide explains what to reach for. Pair them:

* [Data Manager](../guide/data.md) ↔ `rome.data`
* [Training Manager](../guide/training.md) ↔ `rome.trainer`
* [Stream Manager](../guide/streams.md) ↔ `rome.stream`
* [Writing a trainer](../guide/trainers.md) ↔ `rome.train.base`
* [Shared state](../design/state.md) ↔ `rome.utils`
