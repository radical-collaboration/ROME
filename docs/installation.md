# Installation

ROME is a Python package. It needs Python **3.9+** for the framework itself,
and Python **3.10–3.12** if you want the Dragon runtime that backs its shared
state on a cluster.

## From a checkout

```bash
git clone https://github.com/radical-collaboration/ROME.git
cd ROME
pip install -e .
```

This pulls in the framework and its trainers:

| Dependency | Why |
| --- | --- |
| `radical.asyncflow` | The workflow engine ROME submits every task to. ROME schedules nothing itself. |
| `rhapsody-py[radical_pilot]` | Execution backends — RADICAL-Pilot for HPC. |
| `rhapsody-py[dragon]` | The Dragon execution backend (Python 3.10–3.12 only). |
| `torch`, `transformers`, `peft`, `trl`, `datasets` | The built-in LLM/GRPO trainer. |
| `numpy`, `pandas` | Dataset assembly for the ProteinMPNN trainer's manifest. |

!!! note "Heavy imports stay out of the driver process"

    `rome.train.llm` imports TRL, transformers and peft *inside* the functions
    that need them, and `rome.train.__init__` imports the GRPO trainer lazily.
    A workflow that never trains an LLM never pays for that stack, even though
    it is installed.

## Tests

```bash
pip install -e '.[test]'
pytest -m fast          # unit + mocked integration, no GPUs
```

`conftest.py` tags every test that is not explicitly marked `slow` as `fast`, so
`-m fast` is "everything that runs on a laptop". It needs no GPU, no allocation
and no Dragon runtime: the fixtures stub the heavyweight third-party modules with
*working* stand-ins — a plain `dict` for a `DDict`, a `threading.Event` for
Dragon's — so the real ROME code paths are exercised rather than mocked out.

What the suite covers:

| Tests | What they check |
| --- | --- |
| `unit/test_data_manager.py`, `test_training_manager.py`, `test_stream_manager.py` | Each manager in isolation — admission, sampling, consumption; round scheduling, publication, status; request claiming, reload, drain. |
| `unit/test_train_tasks.py`, `test_dummy.py`, `test_namespace.py`, `test_logging.py` | The `TrainTask` interface, the model-free stand-ins, the DDict namespace helpers, the log format. |
| `unit/test_mpnn_trainer_data.py` | The ProteinMPNN trainer's dataset assembly and chain designation. |
| `integration/test_rome_agnostic_loop.py`, `test_dummy_loop.py` | The whole closed loop against stubs — data in, round fires, checkpoint published, stream reloads. |
| `integration/test_impress_r.py`, `unit/test_impress_r_hooks.py` | The IMPRESS-R seam. Skipped via `importorskip` unless IMPRESS and rhapsody are installed. |
| `integration/test_mpnn_train_real.py` | A real ProteinMPNN fine-tune. Skipped unless `ROME_MPNN_TEST_REPO` points at a `dauparas/ProteinMPNN` checkout. |

The heavyweight tests are gated by those skip conditions rather than by the
`slow` marker, which nothing currently carries — so `-m fast` and a bare `pytest`
select the same set.

## Documentation

To build this site locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open <http://127.0.0.1:8000>. `mkdocs build --strict` is what CI runs; it
turns broken internal links and unresolved API cross-references into build
failures.

Note that the docs dependencies are a separate requirements file rather than a
`[docs]` extra: mkdocstrings reads the source **statically** (griffe parses it
rather than importing it), so the API reference builds with none of ROME's
runtime dependencies installed — no Dragon, no torch, no GPU. An extra would have
dragged all of them in.

## Dragon

ROME keeps its cross-node state in a Dragon `DDict` and its stop/reload
signals in Dragon `Event`s, so `rome.manager` and `rome.stream` import
`dragon.data.ddict` and `dragon.native.event` at module scope. Dragon ships as
part of the `rhapsody-py[dragon]` extra on supported Python versions, but on a
cluster you generally build it against the site's MPI and network stack instead.
[Setting up ROME + IMPRESS on Delta](delta.md) walks through that build.

Anything launched under Dragon runs through its launcher rather than plain
`python`:

```bash
dragon examples/agnostic/dummy_loop.py       # single-node
dragon -s examples/agnostic/dummy_loop.py    # multi-node placement
dragon-cleanup-deprecated                    # after every Dragon run
```

The `-s` form is also how ROME's Dragon checks are run. They live in
`tests/dragon/` but are **scripts, not pytest modules**, because the launcher
runs a script rather than a test session — `pytest -m fast` does not collect
them:

```bash
dragon -s tests/dragon/test_namespace_dragon.py   # DDict/Event primitives
dragon -s tests/dragon/test_manager_dragon.py     # the whole loop, 4 replicas
dragon-cleanup-deprecated                         # after every one
```

They split into two kinds:

| Script | Exercises |
| --- | --- |
| `test_namespace_dragon.py` | ROME's `Namespace` against a real DDict |
| `test_manager_dragon.py` | The whole loop, 4 stream replicas, on real placement |
| `test_stream_pickle_dragon.py` | That a `StreamTask` survives the process boundary |
| `test_result_delivery_dragon.py` | That a finished round's checkpoint reaches the driver |
| `test_keys_race_dragon.py`, `test_keys_union_dragon.py` | **Dragon itself** — that `keys()` truncates under concurrent pops |
| `test_service_blocks_results_dragon.py`, `test_executable_result_hang_dragon.py` | **rhapsody itself** — that a running service blocks result delivery behind it |
| `test_task_capacity_dragon.py` | **The allocation** — how many concurrent tasks it will actually place |

The second group imports no ROME at all. They are the reproducers behind the
findings in [ROME on Dragon](dragon.md): each one isolates a backend behaviour
that forced a design decision — why `max_records` carries a warning, why
`result_fallback_seconds` exists, why a stream replica count has to stay under
the allocation's capacity. Keep them; when a Dragon or rhapsody version changes,
they are how you find out whether the workaround is still needed.

[ROME on Dragon](dragon.md) records what running there turned up — notably
that a DDict client handle cannot be shared across threads — and the one known
scaling limit.

## IMPRESS

The IMPRESS-R integration needs IMPRESS installed, plus a `dauparas/ProteinMPNN`
checkout for the trainer to fine-tune. Neither is a ROME dependency — ROME is
workflow agnostic, and the ProteinMPNN trainer ships with the example rather
than with the framework. See [IMPRESS-R](examples/impress-r.md) and
[Setting up on Delta](delta.md).

## Checking the install

The fastest end-to-end check needs no model, no GPU and no cluster:

```bash
python -c "import rome; print(rome.__version__)"
dragon examples/agnostic/dummy_loop.py
```

If the second one prints a climbing `model v` while the stream keeps serving,
every moving part is working: task placement, the shared dictionary, checkpoint
publication and the hot swap. Go on to the [Quickstart](quickstart.md) for what
that output means.
