# IMPRESS-R: ROME adaptive fine-tuning for IMPRESS pipelines

ROME-A closes IMPRESS's open loop mid-campaign. Two of the three use cases here
do it by fine-tuning the **sequence design model** (ProteinMPNN or LigandMPNN)
on high-confidence designs. The third closes the loop one stage earlier, around
**backbone diffusion**, by learning the parameters RFD3 is conditioned on
instead of retraining any model in the pipeline.

| Use case | What improves | What stays fixed |
|----------|---------------|------------------|
| `protein_binding` | ProteinMPNN weights | the diffusion target |
| `small_molecule_binding` | LigandMPNN weights | the diffusion target |
| `interface_targeting` | the **diffusion target** (which ligand atoms the binder is designed to contact) | every model in the pipeline |

## Prerequisites

```bash
export WORK_DIR=/work/nvme/bdyk/$USER

# 1. Clone IMPRESS and ROME under $WORK_DIR
git clone <impress-repo> $WORK_DIR/IMPRESS
git clone <rome-repo>    $WORK_DIR/ROME

# 2. Run IMPRESS's own setup for the use case (creates the base venv)
#    protein_binding:       creates $WORK_DIR/ve/impress
#    small_molecule_binding: creates $WORK_DIR/ve/small_mol
bash $WORK_DIR/IMPRESS/examples/<use_case>/delta_env_setup.sh

# 3. Add ROME on top
export IMPRESS_USECASE=protein_binding   # or small_molecule_binding
bash $WORK_DIR/ROME/examples/impress_r/delta_env_setup.sh
```

## Run

```bash
cd $WORK_DIR/ROME/examples/impress_r

export SBATCH_ACCOUNT=<project>-delta-gpu
export WORK_DIR=/work/nvme/bdyk/$USER
export IMPRESS_USECASE=protein_binding   # or small_molecule_binding,
                                         # or interface_targeting

bash submit.sh
```

Logs land in `<use_case>/logs/` and outputs in `<use_case>/campaign_outputs/`:
- `rome_checkpoints/` — versioned fine-tuned model weights
- `af_pipeline_outputs_multi/` — per-pipeline MPNN + structure prediction results

---

## Required env vars

| Var | Both use cases |
|-----|----------------|
| `WORK_DIR` | Base path for all tool installations |
| `SBATCH_ACCOUNT` | SLURM account (or set via `#SBATCH --account`) |

### protein_binding

| Var | Default | Description |
|-----|---------|-------------|
| `MPNN_PATH` | `$WORK_DIR/ProteinMPNN` | dauparas/ProteinMPNN checkout |
| `BOLTZ_VENV` | `$WORK_DIR/ve/boltz` | Boltz virtual environment |
| `BOLTZ_CACHE_DIR` | `~/boltz` | Boltz model weight cache |
| `IMPRESS_SCRIPTS_DIR` | `$WORK_DIR/IMPRESS/examples/protein_binding` | IMPRESS pipeline scripts dir |
| `IMPRESS_BASE_DIR` | `$WORK_DIR/IMPRESS_inputs` | Parent of `prod_in/` (input PDB files) |
| `IMPRESS_OUTPUT_DIR` | `protein_binding/campaign_outputs` | Campaign output root |
| `IMPRESS_N_PIPELINES` | `1` | Number of top-level pipelines |
| `ROME_MIN_SAMPLES` | `4` | Corpus size before first training round |

### small_molecule_binding

| Var | Default | Description |
|-----|---------|-------------|
| `MPNN_DIR` | `$WORK_DIR/LigandMPNN` | dauparas/LigandMPNN checkout |
| `BOLTZ_CACHE` | `$WORK_DIR/.cache/boltz` | Boltz model weight cache |
| `FOUNDRY_SIF_PATH` | auto-extracted from `$WORK_DIR/foundry_sandbox.tar.gz` | RFD3 singularity sandbox |
| `IMPRESS_DIR` | `$WORK_DIR/IMPRESS` | IMPRESS checkout root |
| `IMPRESS_OUTPUT_DIR` | `small_molecule_binding/campaign_outputs` | Campaign output root |
| `IMPRESS_N_PIPELINES` | `2` | Number of top-level pipelines |
| `ROME_MIN_SAMPLES` | `8` | Corpus size before first training round |

### interface_targeting

Drives the same `SmallMoleculeBindingPipeline` as `small_molecule_binding`, so
it needs **every variable in that section too** (`MPNN_DIR`, `BOLTZ_CACHE`,
`FOUNDRY_SIF_PATH`, `IMPRESS_DIR`) plus these:

| Var | Default | Description |
|-----|---------|-------------|
| `ITGT_DIR` | `$WORK_DIR/interface-targeting` | interface-targeting checkout supplying the models |
| `ROME_ITGT_COMPONENT` | `reward_ridge` | `reward_ridge` (action→outcome surrogate), `burial_prior` or `trunk` |
| `ROME_ITGT_CANDIDATES` | `64` | Candidate partitions ranked per proposal |
| `ROME_ITGT_SIZE_BAND` | `4` | Half-width of the selection-size band around the template's own selection |
| `IMPRESS_OUTPUT_DIR` | `interface_targeting/campaign_outputs` | Campaign output root |

An unset or missing `ITGT_DIR` is not fatal: the campaign runs with the
hand-authored partition unchanged, which is exactly what `small_molecule_binding`
does today.

> **`burial_prior` and `trunk` are self-confirming.** They fit itgt's per-atom
> scorer on the burial of the designed binder — but the binder buried those
> atoms *because the previous round's `select_buried` told RFD3 to bury them*,
> so the label measures compliance with our own instruction and `ligand_iptm`
> never enters the objective. They are offered as a cold-start prior and an
> ablation. `reward_ridge` is the component that optimises the reward.

### Shared ROME settings (all use cases)

| Var | Default | Description |
|-----|---------|-------------|
| `ROME_DIR` | `$WORK_DIR/ROME` | ROME checkout |
| `ROME_TRAINER` | `mpnn` (`itgt` for `interface_targeting`) | `mpnn` = ProteinMPNN/LigandMPNN fine-tune; `itgt` = interface-target surrogate; `dummy` = exercise the loop with no model |
| `ROME_FALLBACK` | `120` | Seconds to wait for Dragon result before reading checkpoint from disk |
| `ROME_REWARD_FN` | use-case default | `module:function` reward override (e.g. `mpnn_trainer:pb_reward_fn`) |

---

## Customising the reward function

Each use case ships a default reward function in its trainer module:

| Use case | Module | Function |
|----------|--------|----------|
| protein_binding | `protein_binding/mpnn_trainer.py` | `pb_reward_fn` |
| small_molecule_binding | `small_molecule_binding/ligandmpnn_trainer.py` | `smb_reward_fn` |
| interface_targeting | `interface_targeting/itgt_trainer.py` | `itgt_reward_fn` |

To use a different function, set `ROME_REWARD_FN=<module>:<function>` before submitting.
The module is resolved relative to the use-case directory, and the function must accept
a single `dict` of per-design metrics and return a `float` in `[0, 1]`.

```python
# small_molecule_binding/my_rewards.py
def tight_binder_reward(record: dict) -> float:
    return float(record.get("interaction_energy", 0.0))
```

```bash
ROME_REWARD_FN=my_rewards:tight_binder_reward bash submit.sh
```

See `skill.md` for the full list of fields available in each use case's record dict.

---

## Quick commands

```bash
# Protein binding quick validation (~3 h, 1 pipeline, 2 passes to trigger ROME)
IMPRESS_USECASE=protein_binding ROME_MIN_SAMPLES=2 bash submit.sh

# Small molecule binding quick validation
IMPRESS_USECASE=small_molecule_binding IMPRESS_N_PIPELINES=1 ROME_MIN_SAMPLES=2 bash submit.sh

# Interface targeting: exercise the loop with no surrogate at all
IMPRESS_USECASE=interface_targeting ROME_TRAINER=dummy IMPRESS_N_PIPELINES=1 \
    ROME_MIN_SAMPLES=2 bash submit.sh

# Interface targeting, real surrogate
IMPRESS_USECASE=interface_targeting IMPRESS_N_PIPELINES=1 ROME_MIN_SAMPLES=2 bash submit.sh
```

## Reading the output

Logs are written to `<use_case>/logs/impress_<jobid>.out`. Key prefixes:

| Prefix | Meaning |
|--------|---------|
| `[ROME-DATA]` | Corpus event: design received, accepted, or rejected (with reason) |
| `[ROME-TRAINER]` | Training round submitted / completed / failed |
| `[ROME-MODEL]` | In-place weight update — model just improved |
| `[ROME] v{n} published` | Checkpoint callback: version, corpus size, weights filename |
| `[PIPELINE-P{n}]` | Per-pipeline status: corpus count, acceptance, training state |

A healthy round looks like:

```
[ROME-DATA] received design 2b3ea24b (score=92.0) — corpus 1 (1 unconsumed)
[PIPELINE-P1] ROME: corpus 1 (+1 this pass) | WAITING
[ROME-TRAINER] submitting training round 1 (4 designs, trainer proteinmpnn) -> v1
[ROME] v1 published — corpus 4 designs → v_48_020_v1.pt
```

### Checkpoints

Each completed round writes two copies:

- **versioned**: `<use_case>/campaign_outputs/rome_checkpoints/<model>/v{n}/` — accumulates across rounds, survives the job
- **in-place**: back into `$MPNN_PATH` (or `$MPNN_DIR`) model weights — picked up automatically by the next inference pass

---

## Directory layout

```
impress_r/
  submit.sh                     — launch wrapper (sets --job-name for log routing)
  delta_gpu_run.sh              — unified SLURM batch script
  delta_env_setup.sh            — unified ROME addon installer
  skill.md                      — reward functions, filters, checkpoint guide
  protein_binding/
    run_protein_binding_rome.py — entry point (light wrapper around IMPRESS)
    mpnn_trainer.py             — ProteinMPNN trainer + pb_reward_fn
    mpnn_train_wrapper.py       — fine-tune CLI (invoked as a task subprocess)
    mpnn_stream.py              — inference stream with hot-swap weights
    logs/                       — SLURM logs (created by submit.sh)
    campaign_outputs/           — created at runtime
  small_molecule_binding/
    run_smb_rome.py             — entry point (light wrapper around IMPRESS)
    ligandmpnn_trainer.py       — LigandMPNN trainer + smb_reward_fn
    logs/                       — SLURM logs (created by submit.sh)
    campaign_outputs/           — created at runtime
  interface_targeting/
    run_interface_targeting_rome.py — entry point (same IMPRESS pipeline, 3 hooks)
    itgt_spec.py                — read/patch the RFD3 base spec (pure stdlib)
    itgt_trainer.py             — surrogate trainer + itgt_reward_fn
    itgt_train_wrapper.py       — fit CLI (invoked as a task subprocess)
    itgt_propose.py             — propose a partition and install it
    logs/                       — SLURM logs (created by submit.sh)
    campaign_outputs/           — created at runtime
      rome_inputs/p{N}_in/      — staged pipeline inputs (written every round)
        .itgt/template.json     — pristine spec; every round patches from this
        .itgt/history/v{n}.json — the partition each round installed
      itgt_replay.jsonl         — the (action, reward) pairs the surrogate fits
```

---

## How `interface_targeting` differs

The diffusion target lives in each pipeline's base RFD3 spec
(`p{N}_in/ALR_binder_design.json`) as the `select_buried`/`select_exposed`
partition of the ligand's heavy atoms. It has been hand-authored and identical
in every campaign to date. This use case makes it a learned quantity:

1. **Harvest** — each pass's `ligand_iptm` is paired with the partition that
   pass actually ran under, and appended to `itgt_replay.jsonl`.
2. **Fit** — a ridge regression over those `(action, reward)` pairs, reported
   with leave-one-group-out over `(pipeline, pass)` groups.
3. **Propose** — random connected surface patches are scored by that model and
   the best is installed as the next pass's partition.

Three consequences worth knowing:

- **Failed folds are kept.** The pipeline records a fold model whenever one was
  produced, not only on a pass, and a reward regression needs its negatives.
- **The spec is patched, never regenerated**, and always from
  `.itgt/template.json` rather than from the previous round's output, so
  proposals cannot compound.
- **`scripts/validate_run.py` check 3 will fail** for rounds older than the
  current base spec: it asserts every guided JSON matches the base verbatim,
  which stops being true once the base moves. `.itgt/history/v{n}.json` is what
  keeps those rounds auditable.
