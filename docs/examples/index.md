# Examples

Every example in the repository is runnable, and they are ordered here by how
much of a real campaign they involve. Start at the top.

| Example | What it shows | Needs |
| --- | --- | --- |
| [`agnostic/dummy_loop.py`](dummy-loop.md) | The closed loop end to end, with no model. | Nothing. |
| [`impress_r/dummy_adaptive_rome.py`](impress-r.md#the-smallest-integration) | Two lines of ROME inside a real host workflow's adaptive step. | IMPRESS. |
| [`agnostic/impress_r.py`](impress-r.md#the-four-calls-against-a-stand-in-pipeline) | The data + training halves against a stand-in pipeline. | Nothing. |
| [`agnostic/llm_grpo_streams.py`](llm-grpo.md) | All three managers: generation and scoring as streams, GRPO training. | A GPU, TRL, a HF model. |
| [`impress_r/adaptive_rome.py`](impress-r.md#the-real-seam-with-executables-stubbed) | The real campaign seam with the executables stubbed, so it runs anywhere. | IMPRESS. |
| [`impress_r/protein_binding_rome.py`](impress-r.md#the-real-campaign) | The real campaign: MPNN → AlphaFold → pLDDT, fine-tuning ProteinMPNN mid-run. | Delta, IMPRESS, ProteinMPNN, AlphaFold. |
| [`impress_r/interface_targeting/`](interface-targeting.md) | Improving something other than a model: the surrogate learns the target RFD3's backbone diffusion is conditioned on. | Delta, IMPRESS, itgt. |

## Running them

Anything that touches the Dragon `DDict` — which is everything, since that is
where ROME keeps its state — runs under the Dragon launcher:

```bash
dragon examples/agnostic/dummy_loop.py          # single node
dragon -s examples/impress_r/adaptive_rome.py   # real placement
dragon-cleanup-deprecated                       # after every run
```

## Choosing a backend

Most examples pick their backend from the environment, and the choice matters
more than it looks:

| Backend | Task bodies run | Right for |
| --- | --- | --- |
| `LocalExecutionBackend(ThreadPoolExecutor())` | Threads in the driver process. | Tests, CPU work, seeing the machinery. |
| `ConcurrentExecutionBackend(ProcessPoolExecutor())` | Separate processes. | A GPU fine-tune on one node — VRAM is released when a round ends. |
| `DragonExecutionBackendV3` | Real processes on real nodes. | A campaign. |

A GPU fine-tune on the local backend keeps its CUDA context resident for the
whole campaign. See [Execution](../design/execution.md#why-a-gpu-round-should-be-a-command).

## Where the trainers live

`rome.train.llm.GRPOTrainer` ships with the framework. `ProteinMPNNTrainer` does
**not** — it lives in `examples/impress_r/protein_binding/mpnn_trainer.py`, because it is an IMPRESS-R
integration and ROME is workflow agnostic. Import it from there:

```python
from examples.impress_r.protein_binding.mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer
```
