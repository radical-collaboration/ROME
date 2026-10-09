# Customizing ROME-A — Reward Functions, Filters, and Checkpoints

Covers all scientist-visible extension points for both use cases.
You do not need to edit core pipeline files — everything is exposed via env vars or importable hooks.

---

## The reward function

The reward function is called on every design in a training batch and returns a scalar weight.
Higher weight → more gradient for that sample. Weights are normalized within each epoch so only relative values matter.

### Signature

```python
def my_reward_fn(record: dict) -> float:
    ...
```

`record` fields vary by use case. All are optional; use `.get()` with a default.

#### protein_binding

| Field | Type | Description |
|---|---|---|
| `pLDDT` | float | Mean Boltz pLDDT [0–100] |
| `pTM` | float | Predicted TM-score [0–1] |
| `pAE` | float | Mean predicted aligned error [Å; lower is better] |
| `path` | str | Path to the staged PDB |
| `uid` | str | 32-char hex corpus ID |

#### small_molecule_binding

| Field | Type | Description |
|---|---|---|
| `plddt` | float | Mean AF2 pLDDT [0–100] |
| `interaction_energy` | float | Rosetta interaction energy [REU; negative is better] |
| `max_sc` | float | Shape complementarity [0–1; higher is better] |
| `path` | str | Path to the staged PDB |
| `uid` | str | 32-char hex corpus ID |

### Default implementations

**protein_binding** — `mpnn_trainer.pb_reward_fn`:
```python
# r = (pLDDT/100)*0.4 + pTM*0.3 + max(0, 1 - pAE/30)*0.3
```

**small_molecule_binding** — `ligandmpnn_trainer.smb_reward_fn`:
```python
# r_plddt    = max(0, plddt - 75)   / 25    # 75 → 0, 100 → 1
# r_interact = max(0, -interact - 5) / 15   # -5 → 0, -20 → 1
# r_max_sc   = max(0, max_sc - 0.5) / 0.3   # 0.5 → 0, 0.8 → 1
# return r_plddt + r_interact + r_max_sc
```

---

## Plugging in a custom reward

### Step 1 — Write the function in an importable module

Place the file in the use-case directory (always on `sys.path`) or anywhere on `PYTHONPATH`:

```python
# my_campaign.py

def my_reward(record: dict) -> float:
    plddt  = float(record.get("pLDDT") or record.get("plddt") or 0.0)
    max_sc = float(record.get("max_sc") or 0.0)

    if plddt < 70.0:
        return 0.0

    r_sc    = max(0.0, max_sc - 0.5) / 0.3
    r_plddt = max(0.0, plddt - 70.0) / 30.0
    return 2 * r_sc + r_plddt
```

### Step 2 — Point `ROME_REWARD_FN` at it

```bash
export ROME_REWARD_FN="my_campaign:my_reward"
bash submit.sh
```

Format: `module_name:function_name` — imported at job startup via `importlib.import_module`.

### Step 3 — Verify at import time

```bash
cd /work/nvme/bdyk/$USER/ROME/examples/impress_r
ROME_REWARD_FN=my_campaign:my_reward \
  python -c "
import os, importlib
spec = os.environ['ROME_REWARD_FN']
mod, fn = spec.rsplit(':', 1)
f = getattr(importlib.import_module(mod), fn)
print(f({'pLDDT': 88.0, 'pTM': 0.85, 'pAE': 4.2}))
"
```

---

## More reward function recipes

### Protein binding — weight pTM heavily (fold quality)

```python
def ptm_focused(record: dict) -> float:
    plddt = float(record.get("pLDDT") or 0.0)
    ptm   = float(record.get("pTM")   or 0.0)
    pae   = float(record.get("pAE")   or 30.0)

    if plddt < 75.0 or ptm < 0.7:
        return 0.0

    return ptm * 0.6 + (plddt / 100.0) * 0.3 + max(0.0, 1.0 - pae / 20.0) * 0.1
```

### Small molecule — interaction energy only (when pLDDT is uniformly high)

```python
def interact_only(record: dict) -> float:
    interact = float(record.get("interaction_energy") or 0.0)
    return max(0.0, -interact - 5.0) / 20.0   # -5 → 0, -25 → 1
```

### Composite with a custom field

If you pass a custom metric via `rome_manager.add_training_data(..., buried_sasa=value)`, it lands in `record`:

```python
def ppi_reward(record: dict) -> float:
    buried_sasa = float(record.get("buried_sasa") or 0.0)
    plddt       = float(record.get("pLDDT") or record.get("plddt") or 0.0)
    return max(0.0, buried_sasa - 500) / 1000 + max(0.0, plddt - 75) / 25
```

---

## Corpus filter

Gates admission — designs that fail are never trained on, regardless of reward.

To use a custom filter, modify `DataConfig` in the use-case entry point:

```python
# protein_binding: run_protein_binding_rome.py
from mpnn_trainer import percentile_sampler

rome_manager = rome.Manager(
    data_config=rome.DataConfig(
        min_samples=int(os.environ.get("ROME_MIN_SAMPLES", 4)),
        sample_func=percentile_sampler(0.33, on_summary=print),
        filter_func=lambda r: float(r.get("pLDDT") or 0) >= 80.0,
    ),
    ...
)
```

Rejected designs appear as:
```
[ROME-DATA] rejected design 3d780d7f (filtered: ...)
```

---

## Percentile sampler

When the corpus grows large, `percentile_sampler` selects the top fraction for each training round rather than using everything, keeping rounds fast and biasing learning toward the best examples.

```python
# Use the top 25% of corpus by combined rank (default is 33%)
from mpnn_trainer import percentile_sampler          # protein_binding
# from ligandmpnn_trainer import percentile_sampler  # small_molecule_binding

sample_func=percentile_sampler(0.25, on_summary=print)
```

`on_summary=print` prints per-round selection statistics:
```
{'corpus': 12, 'selected': 4, 'ranked_by': {...}, 'cutoffs': {...}}
```

---

## Versioned checkpoints

Every completed training round writes:

1. **Versioned** — `<use_case>/campaign_outputs/rome_checkpoints/<model>/v{n}/` — accumulates across rounds, survives the job
2. **In-place** — back into `$MPNN_PATH` / `$MPNN_DIR` model weights — picked up automatically by the next inference pass

The `on_checkpoint` hook logs every version as it goes live:
```
[ROME] v3 published — corpus 8 designs → v_48_020_v3.pt
```

To write checkpoints elsewhere:
```bash
export IMPRESS_OUTPUT_DIR=$WORK_DIR/my_campaign
bash submit.sh
```

---

## Observability cheat sheet

```
[ROME-DATA] received design <uid8> (score=<val>) — corpus <n> (<k> unconsumed)
[ROME-DATA] rejected design <uid8> (filtered: ...)       ← corpus filter blocked it
[ROME-DATA] rejected design <uid8> (duplicate)           ← already in corpus

[PIPELINE-Px] ROME: corpus <n> (+1 this pass) | WAITING  ← not enough data yet
[PIPELINE-Px] ROME: corpus <n> (+1 this pass) | TRAINING ← round in progress
[PIPELINE-Px] ROME: corpus <n> (+0 — filtered by ROME) | ...

[ROME-TRAINER] submitting training round <n> (<k> designs) -> v<n>
[ROME-TRAINER] round <path>/train_complete: backend has not delivered a result
               after <t>s, but checkpoint is on disk — publishing from disk.
[ROME-TRAINER] training round <n> failed: <error>

[ROME] v<n> published — corpus <k> designs → <basename>
```

The Dragon "backend has not delivered a result" warning is expected — training succeeds and the checkpoint is published from disk. See `docs/dragon.md` for background.
