# Setting up ROME + IMPRESS on Delta

End-to-end: environment, install, a ladder of smoke tests that each prove one
more layer, then running an IMPRESS campaign with ROME attached.

---

## 1. Environment

Delta gives you `/work/nvme` (fast, for code and checkpoints) and `/work/hdd`
(bulk, for structures and caches). Keep code and checkpoints on
`/work/nvme/<project>/<user>/...` and the bulk artifacts on `/work/hdd`.

```bash
export PROJ=/work/nvme/bdyk/$USER          # adjust to your project/user
export BULK=/work/hdd/bdyk/$USER
mkdir -p $PROJ $BULK
```

Python 3.12 is what everything below was verified against. Dragon publishes
wheels per CPython version, so the interpreter version is not a free choice —
check what `dragonhpc` has for your Python before committing to one.

```bash
module avail python anaconda          # Delta-specific: see what is offered
python3 -m venv $PROJ/venv-rome
source $PROJ/venv-rome/bin/activate
pip install --upgrade pip
```

A venv is easier than conda here because Dragon, `radical.asyncflow` and
`rhapsody-py` are all plain wheels.

## 2. Install Dragon

```bash
pip install dragonhpc
dragon --version          # Dragon Version 0.14.1
```

Two things to know from the start:

* **Run programs as `dragon -s script.py`**, not `python script.py`. Dragon's
  API imports without the runtime but every object it creates asserts on launch
  parameters that only exist inside a Dragon launch, so plain `python` fails
  with `Launch parameter not initialized: GS_CD`.
* **Run `dragon-cleanup-deprecated` after every Dragon program**, including
  after a crash or a timeout. Leftovers stop the next run from starting.

## 3. Install ROME

```bash
cd $PROJ
git clone https://github.com/radical-collaboration/ROME.git && cd ROME
pip install -e '.[test]'
pip install 'rhapsody-py[dragon]'     # Dragon execution backend for asyncflow
```

Sanity check without any HPC involvement:

```bash
pytest -m fast
```

Expect a clean run — every test here is CPU-only and needs no allocation.

## 4. Install IMPRESS

Clone IMPRESS and run its own use-case setup script to create the base venv:

```bash
cd $PROJ
git clone https://github.com/radical-collaboration/IMPRESS.git
cd IMPRESS
pip install -e .

# use-case-specific setup (creates the venv and installs tool deps):
bash examples/protein_binding/delta_env_setup.sh      # protein binding
# or
bash examples/small_molecule_binding/delta_env_setup.sh  # small molecule binding
```

---

## 5. Smoke-test ladder

Run these in order on a compute node. Each one proves one more layer, so when
something breaks you know which layer to look at. Every command below was run
and produced the output shown.

```bash
cd $PROJ/ROME
```

**(a) Dragon primitives — is the DDict reachable?**

```bash
dragon -s tests/dragon/test_namespace_dragon.py && dragon-cleanup-deprecated
```
```
ok    single-key round trip
ok    missing key returns default
ok    dict records survive pickling
ok    prefix scan is namespace-scoped
ok    delete and pop
ok    drain claims exactly once
ok    increment counter
ok    model version defaults to 0
ok    host workflow keys untouched
ok    Event set/clear/is_set

all DDict/Event checks passed
```

**(b) The whole ROME loop — 4 stream replicas, a trainer, one real DDict.**

```bash
dragon -s tests/dragon/test_manager_dragon.py && dragon-cleanup-deprecated
```
```
ok    every request answered exactly once
ok    work spread over replicas
ok    outputs are distinct
ok    concurrent writers lose nothing
HH:MM:SS.sss [INFO] [ROME-TRAINER] submitting training round 1 (100 designs, trainer dummy) -> v1
HH:MM:SS.sss [INFO] [ROME-MODEL]   published v1 (100 designs) -> /tmp/.../dummy/v1
ok    training fired and published
ok    streams swapped onto the checkpoint
ok    host workflow keys untouched
HH:MM:SS.sss [INFO] [ROME-MANAGER] stopping — corpus 100, 1 round completed, model v1

ROME works on Dragon
```

**(c) The worked example — watch a model version climb while inference serves.**

```bash
dragon -s examples/agnostic/dummy_loop.py && dragon-cleanup-deprecated
```

That runs on `LocalExecutionBackend` (task bodies as threads). To exercise real
multi-process placement — streams and a training round in separate processes,
which is what a real allocation does — run the same example on the Dragon
execution backend:

```bash
ROME_BACKEND=dragon ROME_STREAM_REPLICAS=1 ROME_GPUS=0 \
  dragon -s examples/agnostic/dummy_loop.py && dragon-cleanup-deprecated
```

The version still climbs, but two things differ and both are backend facts, not
ROME ones (see §6 and `docs/dragon.md`): keep `ROME_STREAM_REPLICAS` below the
allocation's concurrent-task capacity so the round gets a slot, and the round's
result is published *from disk* after a short grace because a stream service task
blocks rhapsody's result delivery. On a real multi-node allocation raise the
replica count and leave the fallback at its minutes-scale default.

**(d) IMPRESS-R — real IMPRESS pipeline, real ROME.**

With IMPRESS installed (§4), run the integration tests directly:

```bash
pytest tests/unit/test_impress_r_hooks.py tests/integration/test_impress_r.py -v
```

These skip automatically if IMPRESS or rhapsody are absent, and pass on any
machine — no GPU, no allocation. At this point the seam between ROME and IMPRESS
is proven. The next step (§7) runs the full campaign on a real allocation.

---

## 6. Moving off one node: the Dragon execution backend

Steps (a)–(d) use `LocalExecutionBackend`, which runs task bodies as threads in
the driver process. For a real allocation you want tasks placed on nodes, which
is `rhapsody`'s Dragon backend:

```python
from rhapsody.backends import DragonExecutionBackend

backend = DragonExecutionBackend(batch_kwargs={
    "num_nodes": 2,                     # defaults to the whole allocation
    "results_ddict_mem": 4 * 1024**3,   # raise for large returns / many tasks
})
manager = rome.Manager(backend=backend, ...)   # ROME builds its own engine
```

`batch_kwargs` are forwarded verbatim to `dragon.workflows.batch.Batch()`.

**Verified on this backend:** the data + training path, which is what IMPRESS-R
uses. A run publishing three checkpoints across processes:

```
cycle 1: corpus 4  | v1 | model=v1
cycle 3: corpus 8  | v2 | model=v2
cycle 5: corpus 12 | v3 | model=v3
```

Two things this turned up that you will hit:

* **A `TrainTask`'s in-process state does not come back.** The task body runs in
  a different process, so anything it records on `self` is invisible to the
  driver — in the run above the trainer object reported `0` rounds while three
  checkpoints were on disk. Only the returned checkpoint path crosses back.
  Write results to the DDict or to disk, not to instance attributes.
* **`DDict.get(key, default)` hangs.** Not raises — hangs. Use `d[key]` in a
  `try/except KeyError`. ROME's `Namespace.get` already does; this matters if
  you touch a DDict directly in your own pipeline code.

**Streams work on this backend.** They did not until recently, and the cause was
a ROME bug rather than anything about Dragon:

`StreamManager.start()` submits a body that closes over the `StreamTask` **by
reference**, then assigns the returned future to `task.task_fut`. A
multi-process backend pickles that body from its dispatcher thread, which
happens *after* the assignment — so the task now carries an `_asyncio.Future`,
`cannot pickle '_asyncio.Future' object` is raised inside the dispatcher, and
the dispatcher thread dies. Every task queued behind it, ROME's or the host
workflow's, then silently never runs. `LocalExecutionBackend` never pickles
anything, which is why the bug was invisible there.

`StreamTask.__getstate__` now drops driver-only attributes, so the body pickles
whenever the backend gets round to it. With that fix all the stream checks pass
on `DragonExecutionBackend`: every request answered exactly once, work spread
across replicas, distinct outputs, and streams swapping onto a new checkpoint.

**Budget one task slot per stream, plus one for training.** This is the thing to
size for. A stream is a *service* task that never returns, so it holds its slot
for the whole run. Measured on a 4-CPU single node, this backend ran only **2
concurrent never-returning tasks** — six submitted, two started, four stuck in
`STARTING` forever. `scheduler_workers` did not raise it. Two consequences:

* Requests routed to a replica that never started are never claimed. With four
  replicas on a two-slot box, exactly half the batch was processed and the rest
  sat in the queue.
* **Training starves.** A round is a task like any other, so if the streams
  occupy every slot, `min_samples` is reached and no round ever runs. Verified:
  the trainer alone publishes `v1` in two seconds on this backend, and the same
  trainer never fires with a stream holding a slot.

So the allocation needs at least `num_streams + 1` concurrent task slots. On a
real multi-node allocation that is not a constraint; on one node it is, and
`tests/dragon/test_manager_dragon.py` takes `ROME_STREAM_REPLICAS` for exactly
that reason.

**Ask for GPUs only if you have them.** `StreamConfig.num_gpus` defaults to `1`
and `TrainTask.gpus` is passed through, both of which put `gpus_per_rank` into
the task description. On a node without GPUs the task is accepted and never
placed — no error, just `STARTING` forever.

---

## 7. Running a campaign under Slurm

`examples/impress_r/submit.sh` is the entry point. It sets `--job-name` so SLURM
routes logs to `<use_case>/logs/` automatically and creates the directory before
submitting `delta_gpu_run.sh`:

```bash
cd $PROJ/ROME/examples/impress_r

export SBATCH_ACCOUNT=<your-account>          # or set #SBATCH --account in delta_gpu_run.sh
export WORK_DIR=/work/nvme/bdyk/$USER
export IMPRESS_USECASE=protein_binding        # or small_molecule_binding

bash submit.sh
```

Logs land in `<use_case>/logs/impress_<jobid>.out` (relative to where you run
`submit.sh` from). See `README.md` in the same directory for the full list of
env vars and their defaults.

For a quick smoke test before committing to a long run, start with 1 pipeline and
a low `ROME_MIN_SAMPLES` to confirm the loop closes:

```bash
IMPRESS_N_PIPELINES=1 ROME_MIN_SAMPLES=2 bash submit.sh
```

Check `sinfo -s` and `accounts` first to confirm the partition name and account
for your allocation.

## 8. What the campaign script needs

`examples/impress_r/protein_binding/run_protein_binding_rome.py` is the real
production script — no stubs. It runs `ProteinMPNNTrainer` directly and wraps
IMPRESS's own `adaptive_decision` with two hooks: corpus staging and ROME model
delivery. The following must be in place before submitting:

| Requirement | Env var | Default |
|---|---|---|
| ProteinMPNN checkout (`dauparas/ProteinMPNN`) | `MPNN_PATH` | `$WORK_DIR/ProteinMPNN` |
| IMPRESS checkout | `IMPRESS_DIR` | `$WORK_DIR/IMPRESS` |
| Boltz venv | `BOLTZ_VENV` | `$WORK_DIR/ve/boltz` |
| Input PDB directory (parent of `prod_in/`) | `IMPRESS_BASE_DIR` | `$WORK_DIR/IMPRESS_inputs` |

`delta_gpu_run.sh` exports all of these with their defaults and prints them at
job start. Any that differ from the defaults can be overridden before `submit.sh`.

The trainer fine-tunes the **original ProteinMPNN weights** at `$MPNN_PATH` —
the same weights IMPRESS runs — via `ProteinMPNNConfig(mpnn_repo=...)`. With
`publish_into_repo=True` (the default) it writes the new weights into
`{mpnn_repo}/vanilla_model_weights/{model_name}.pt`, so the next MPNN pass picks
them up with no change to the IMPRESS pipeline scripts. See
`docs/proteinmpnn_training.md` for the data prep and checkpoint format.

One open item: fine-tuning only on self-generated designs will drift the model.
The standard mitigation — mixing in a slice of the original PDB training
distribution — needs a held-out set the campaign does not provide.

## 9. Selecting designs without knowing your thresholds yet

`impress_corpus_filter()`'s defaults admit **83%** of a real campaign. They are
IMPRESS's own keep/drop thresholds, and everything reaching the score CSVs has
already cleared those, so the filter is applied downstream of itself and selects
nothing. Fine-tuning on that corpus trains ProteinMPNN on its own median output.

Replacing them needs a distribution, and confidence scales are predictor
specific — an AlphaFold2-multimer campaign and a Boltz one are not comparable —
so the numbers have to come from the run you are doing. Two ways to get there,
and the first needs nothing up front.

**Rank instead of threshold (recommended for a first run).**

```python
from examples.impress_r.protein_binding.mpnn_trainer import percentile_sampler

rome.DataConfig(
    min_samples=24,
    sample_func=percentile_sampler(0.33, on_summary=print),
)
```

"The best third of what this campaign has produced" needs no scale, so it works
on the first round before any distribution exists, and keeps working if you
switch predictors. Ranking is by average rank across pAE (down) and pTM (up), so
the two contribute equally without normalisation and neither's outliers dominate.
`on_summary=print` reports, every round, the corpus size, how many were selected,
and **the cutoffs an equivalent fixed filter would have used** — which is how the
run hands you the calibration data as a byproduct.

Leave `filter_func` off, or keep it only to reject malformed records. Admission
and selection are different jobs; this does the selecting.

**Watch the distribution directly.**

The `af_stats_*.csv` files written by the pipeline accumulate in
`$IMPRESS_OUTPUT_DIR`. Once a few passes have landed, inspect them to choose
fixed thresholds:

```python
filter_func=impress_corpus_filter(min_pLDDT=..., min_pTM=..., max_pAE=...)
```

One trap: setting each of the three clauses at its 33rd percentile does **not**
admit a third. On measured data it admitted 6%, because the three scores
correlate. Verify the joint admission rate rather than reasoning clause by
clause.

Whichever route, `sampling="top_k"` with `score_key="pLDDT"` is worth avoiding:
pLDDT never fell below 88 across 176 measured records, so ranking on it is close
to ranking at random.
