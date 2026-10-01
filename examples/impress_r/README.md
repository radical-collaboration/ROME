# IMPRESS-R: ROME adaptive fine-tuning for IMPRESS pipelines

ROME-A closes IMPRESS's open loop by fine-tuning the sequence design model
(ProteinMPNN or LigandMPNN) mid-campaign on high-confidence designs.

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
export IMPRESS_USECASE=protein_binding   # or small_molecule_binding

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

### Shared ROME settings (both use cases)

| Var | Default | Description |
|-----|---------|-------------|
| `ROME_DIR` | `$WORK_DIR/ROME` | ROME checkout |
| `ROME_TRAINER` | `mpnn` | `mpnn` = ProteinMPNN/LigandMPNN fine-tune |
| `ROME_FALLBACK` | `120` | Seconds to wait for Dragon result before reading checkpoint from disk |
| `ROME_REWARD_FN` | use-case default | `module:function` reward override (e.g. `mpnn_trainer:pb_reward_fn`) |

---

## Customising the reward function

Each use case ships a default reward function in its trainer module:

| Use case | Module | Function |
|----------|--------|----------|
| protein_binding | `protein_binding/mpnn_trainer.py` | `pb_reward_fn` |
| small_molecule_binding | `small_molecule_binding/ligandmpnn_trainer.py` | `smb_reward_fn` |

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
```
