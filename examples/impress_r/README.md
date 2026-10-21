# IMPRESS-R

Quick reference for running and customising IMPRESS-R campaigns.
Full documentation: `docs/examples/impress-r.md` and `docs/delta.md`.

## Run

```bash
export SBATCH_ACCOUNT=<project>-delta-gpu
export WORK_DIR=/work/nvme/bdyk/$USER
export IMPRESS_USECASE=protein_binding   # or small_molecule_binding

bash submit.sh
```

Quick validation (2 passes, triggers one ROME round):

```bash
IMPRESS_USECASE=protein_binding ROME_MIN_SAMPLES=2 bash submit.sh
IMPRESS_USECASE=small_molecule_binding IMPRESS_N_PIPELINES=1 ROME_MIN_SAMPLES=2 bash submit.sh
```

Logs: `<use_case>/logs/impress_<jobid>.out`

## Customising the reward function

```bash
ROME_REWARD_FN=my_rewards:my_fn bash submit.sh
```

The function receives a `dict` of per-design metrics and returns a `float` in
`[0, 1]`. Module is resolved relative to the use-case directory.
See `skill.md` for the full field list per use case.

---

## IMPRESS API surface

Everything ROME reads or writes on IMPRESS objects. **If IMPRESS developers
change any of these, the integration breaks.**

### `ImpressBasePipeline` attributes (read inside `adaptive_decision`)

| Attribute | Type | Used for |
|-----------|------|---------|
| `pipeline.name` | `str` | pipeline identifier; used to name staged files and corpus records |
| `pipeline.passes` | `int` | current pass count; used to make staged filenames unique across passes |
| `pipeline.output_path_af` | `str` | directory where AF/Boltz prediction PDBs land — **one file per design, overwritten every pass** |
| `pipeline.iter_seqs` | `dict[str, list[list[str]]]` | designed sequences keyed by design name; `[name][rank][0]` is the sequence that was folded |
| `pipeline.mpnn_weights` | `str` | path to the MPNN weights currently in use; **ROME writes here** to hot-swap the model for the next pass |

### Score CSV: `af_stats_{pipeline.name}_pass_{pipeline.passes}.csv`

Written by IMPRESS's `plddt_extract_pipeline.py` after every pass.

| Column | Notes |
|--------|-------|
| `ID` | Input PDB basename — identical every pass, **not a unique record key**. Use `(pipeline.name, pipeline.passes)` to key corpus records. |
| `avg_plddt` | Per-design pLDDT; narrow range (everything here already cleared IMPRESS's own filter) |
| `ptm` | pTM confidence score |
| `avg_pae` | Interface pAE — widest spread, preferred ranking metric |

### `ImpressManager` (used at campaign level)

| Symbol | Used for |
|--------|---------|
| `ImpressManager(...)` | Constructed normally; ROME wraps it, does not subclass it |
| `impress_manager.flow` | asyncflow `WorkflowEngine` — passed to `rome.Manager` when sharing the engine |

### asyncflow task-description keys ROME depends on

| Key | Backend support |
|-----|----------------|
| `auto_register_task(local_task=True)` | Keeps a function as a plain Python call rather than an executable task |
| `PipelineSetup(kwargs={...})` | Threads constructor kwargs (e.g. `base_path`) into the pipeline without modifying the class |
| `pre_exec` / `post_exec` | **RADICAL-Pilot only** — silently ignored on `LocalExecutionBackend` and Dragon. Do not rely on these for logic that must run on every backend. |
