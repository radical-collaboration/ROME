# Fine-tuning ProteinMPNN for IMPRESS-R

What the trainer does, what data it needs, and what still has to be validated on
a real checkout. This replaces an earlier version written against foundry's
ProteinMPNN re-implementation — that was the wrong target, for the reason in §1.

---

## 1. Which ProteinMPNN — and why it matters

IMPRESS runs the **original `dauparas/ProteinMPNN`**. Its `mpnn_wrapper.py`
shells out to `protein_mpnn_run.py` with the original CLI
(`--jsonl_path`, `--chain_id_jsonl`, `--fixed_positions_jsonl`, …) and its
`helper_scripts/`, and the setup clones that repo directly:

```
git clone https://github.com/dauparas/ProteinMPNN.git
```

So the trainer fine-tunes **that** model and writes a checkpoint in **that**
format. This is not a detail — a checkpoint has to load into the model IMPRESS
runs at inference:

* the original loads `torch.load(path)['model_state_dict']` and reads
  `checkpoint['num_edges']` to size the neighbour graph;
* foundry's re-implementation uses a different `{"model": ...}` layout, different
  parameter names, and a reordered amino-acid vocabulary.

A foundry checkpoint will not load into `protein_mpnn_run.py`, and there is no
foundry→original converter. Targeting the original removes the mismatch entirely.

(PyRosetta is in the IMPRESS stack, but for FastRelax and pLDDT extraction —
`plddt_extract_pipeline.py` — not for sequence design. The design model is
vanilla ProteinMPNN.)

## 2. The training example is a *dimer*, scored on the designed chain

An IMPRESS protein-binding design is a complex, not a monomer:
`protein_binding.py` writes the prediction input as two chains — a designed
`>pdz` and a constant `>pep` of `EGYQDYEPEA` — and the AF/Boltz prediction of
that pair (coordinates + the designed sequence in chain A) is the training
example.

Fine-tuning has to respect that. ProteinMPNN's chain mask expresses exactly the
right thing:

* **chain A — designed.** Predicted and scored: this is the sequence recovery
  the campaign is optimising.
* **chain B — context.** Its backbone *and* its known sequence condition the
  prediction of chain A, but it is excluded from the loss. Learning to design a
  binder means learning it *in the presence of* the target, which is what makes
  the peptide context rather than a second target.

`build_chain_designation` produces `{design_name: (["A"], ["B"])}`, which is what
the original repo's `tied_featurize` consumes to build that mask. The loss is
computed over `mask * chain_M` — resolved residues of the designed chain only.

Defaults are `design_chains=("A",)`, `context_chains=("B",)`. A monomer campaign
sets `context_chains=()`; a per-structure rule can be supplied as
`chains_func(record, present_chains) -> (designed, context)`. Chains named but
absent from a file are dropped, and a design with no designable chain present is
rejected in `validate` rather than producing an empty loss on the GPU.

## 3. What data a round needs

Per accepted design, a corpus record needs:

```python
manager.add_training_data(
    path=<the design's predicted structure, chain A designed, chain B context>,
    sequence=<designed sequence>,     # optional: manifest + sizing only
    backbone_id=<parent pipeline>,    # audit
    pLDDT=..., pTM=..., pAE=...,       # selection (see docs/impress.md)
)
```

Only `path` is required — ProteinMPNN reads the label (the designed chain's
residues) out of the structure itself; there is no separate sequence label. Two
things the integration must get right, both from `docs/impress.md`:

1. **The path must be a stable snapshot.** IMPRESS keys its prediction output by
   pipeline, not by pass, and `finalize()` deletes it, so a recorded path goes
   stale. The trainer defends against basename collisions by *staging* every
   structure into the round's own directory under a unique name
   (`stage_structures`), but the contribution step still has to record a path
   whose contents are correct at record time.
2. **Rank by pAE/pTM, not pLDDT** when selecting which designs enter the corpus —
   pLDDT barely moves. Prefer `percentile_sampler`; see `docs/impress.md` §9.

## 4. What a round produces, and where it goes

Every round writes a **versioned checkpoint that is never overwritten** —
`{model_name}_v{version}.pt`, in that round's own directory
(`<checkpoint_dir>/proteinmpnn/v<version>/`) — in the original format
`{"model_state_dict": ..., "num_edges": 48, "noise_level": 0.2}` (via
`original_checkpoint`), the exact dict `protein_mpnn_run.py` loads. So the whole
history is kept, one file per version, with the version in the name.
`versioned_checkpoint_path(config, output_dir, version)` names it, and it is what
`manager.get_current_model()` returns — so anything that reloads off the current
model reads the exact versioned file.

Getting the latest *back into the campaign* is the seam that closes the loop.
`mpnn_wrapper.py` never passes `--path_to_model_weights`, so `protein_mpnn_run.py`
loads the fixed path `{mpnn_repo}/vanilla_model_weights/{model_name}.pt` by
default. With `publish_into_repo=True` the round **copies** its versioned
checkpoint onto that fixed pointer (atomically — temp file then `os.replace`, so a
mid-pass reader never sees a half-written file), so the next pass runs the new
weights with no wrapper change. `repo_pointer_path(config)` names that pointer;
`model_name` must match the `--model_name` IMPRESS runs (`v_48_020` by default).
The pointer is just "the current version" — the versioned files remain the kept
archive.

Without `publish_into_repo`, only the versioned checkpoints are written, and
pointing MPNN at the one you want is the integration's job — e.g. pass its path
via `--path_to_model_weights`, reading it from `get_current_model()`.

## 5. Config that must match the weights

The public `v_48_*` weights fix the architecture, and `protein_mpnn_run.py`
hardcodes most of it (hidden_dim 128, 3+3 layers) while reading `k_neighbors`
back from the checkpoint's `num_edges`. So a fine-tune must keep
`num_neighbors=48`, `hidden_dim=128`, `num_layers=3`, or the published checkpoint
will not load. These are the `ProteinMPNNConfig` defaults; change them only if
you are also changing the weights IMPRESS runs.

A round is short by design: `max_epochs=3`, batching to `batch_tokens=10000`,
resuming from the previously published checkpoint (or `initial_weights` on round
one) with a Noam schedule, label-smoothed NLL — the original trainer's own recipe.

## 6. How a round is submitted — a command, not a function

Like IMPRESS, which submits `mpnn_wrapper.py` as a shell command rather than
calling ProteinMPNN inside the campaign process, ROME submits a training round
as an **executable task**. `ProteinMPNNTrainer.as_command(dataset, output_dir)`
stages the round's structures, writes a self-contained job spec
(`<output_dir>/train_job.json`), and returns a command line:

    python examples/impress_r/mpnn_train_wrapper.py --job <output_dir>/train_job.json

The training manager runs *that* on the execution backend (with
`{"gpus_per_rank": 1}`), so the fine-tune is a separate process on its own GPU —
nothing about it lives in the manager's address space, and the process exits
when the round ends, releasing its VRAM. `examples/impress_r/mpnn_train_wrapper.py` is
deliberately dragon-free (it imports only the standard library, torch, and the
checkout named in the job), so you can run and debug it on its own. Point
`config.train_script` at a copy staged elsewhere on the cluster if the bundled
path is not reachable from the compute node (as IMPRESS points `-mpnn` at its own
checkout). A `pre_exec` to activate the environment can be supplied through
`TrainerConfig.task_description`.

## 7. How the loop is built, and that it is verified

The loop lives in **one** place — `examples/impress_r/mpnn_train_wrapper.py`’s `run_round` — which
the command and the in-process `train()` path (used by the tests) both call, so
there is no second copy to drift. It does **not** reimplement anything: it
imports the checkout's own training modules and reproduces
`training/training.py`'s inner loop exactly:

* `featurize`, `loss_smoothed`, `NoamOpt`, and the training `ProteinMPNN` from
  `training/model_utils.py` — note this is the *training* `ProteinMPNN`, whose
  `forward(X, S, mask, chain_M, residue_idx, chain_encoding_all)` builds the
  decoding order itself (the inference one in `protein_mpnn_utils.py` takes an
  extra `randn` and is the wrong one to train with — the previous version's bug);
* `parse_PDB` from `protein_mpnn_utils.py`, `StructureDataset` / `StructureLoader`
  from `training/utils.py`.

The dimer split is carried the way upstream carries it: each parsed structure
gets a `masked_list` (designed chains → `chain_M == 1`) and `visible_list`
(context → `0`), and the loss is `loss_smoothed(S, log_probs, mask * chain_M)` —
resolved residues of the designed chain only.

**Verified end to end**, not just described:
`tests/integration/test_mpnn_train_real.py` fine-tunes from the public
`v_48_020` weights on a real multi-chain PDB, then loads the published checkpoint
back into `protein_mpnn_utils.ProteinMPNN` (the inference model
`protein_mpnn_run.py` uses) — proving the checkpoint is compatible — and checks a
second round resumes and advances the Noam step. A third test runs the wrapper
**as an actual subprocess** the way the manager submits it (`as_command` →
`python mpnn_wrapper.py --job …`) and reloads what it wrote, proving the
out-of-process path. All are gated on torch and a checkout, so they skip in CI;
run them on your allocation with::

    ROME_MPNN_TEST_REPO=$WORK/ProteinMPNN \
      pytest tests/integration/test_mpnn_train_real.py

The pure-Python half (chain designation, staging, `validate`, the manifest, the
checkpoint format and publication path) is covered unconditionally in
`tests/unit/test_mpnn_trainer_data.py` and `tests/unit/test_train_tasks.py`.

`config.train_func` still lets you drop in your own
`(manifest_path, output_dir, config) -> checkpoint_path` — for a fork, or to
bring the campaign up one layer at a time before switching the real loop on.

## 8. Generation as a stream (the other half of the loop)

The trainer *improves* ProteinMPNN; `examples/impress_r/mpnn_stream.py` *runs* it
as a ROME inference stream, so a protein workflow can use all three managers
(generate → score → train → hot-swap) the way the LLM examples do. Feed the
stream backbone PDBs, get designed sequences back; when the trainer publishes a
new checkpoint the stream reloads onto it and the next batch is designed with the
improved model.

Two implementations, both **subprocess** — neither reimplements ProteinMPNN's
sampling, so both run the checkout's own tested inference:

* `mpnn_run_stream` — the ROME-native path: `protein_mpnn_run.py` per backbone,
  passing `--path_to_model_weights` so it uses the *exact* published versioned
  checkpoint (clean per-version hot-swap).
* `impress_mpnn_stream` — the IMPRESS path: this example's `mpnn_wrapper.py`
  (chain parse/assign + `protein_mpnn_run.py`), i.e. exactly what the campaign
  runs. Its wrapper can't take an explicit weights path, so on each reload the
  stream refreshes the repo's fixed `{model_name}.pt` pointer to the new
  checkpoint — the same pointer `publish_into_repo` keeps current.

A request is `{"backbone_id", "pdb_path", "design_chains"?, "num_seqs"?}`; each
output record is `{"backbone_id", "sequence", "score", "sample", "model_version",
"pdb_path"}` — hand it to a reward stream (AlphaFold) and then
`add_training_data`, closing the loop back to the trainer. The FASTA parser,
command builders and reload hook are covered in `tests/unit/test_mpnn_stream.py`;
the subprocess paths need the checkout and a GPU, so they run on-cluster.

## 9. Open items

* **Drift.** Fine-tuning only on self-generated designs pulls the model toward
  the campaign. The standard mitigation mixes in a slice of the original PDB
  training distribution; the campaign does not provide one, so either hold out a
  validation set from an earlier campaign and watch sequence recovery per round,
  or keep rounds short and the learning rate low.
* **Pin the checkout.** The dauparas repo's `tied_featurize` and helper layout
  have changed over time. Pin a commit and validate the loop against it once.
