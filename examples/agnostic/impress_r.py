"""IMPRESS-R: adding model improvement to the IMPRESS pipeline with ROME.

IMPRESS today runs backbone -> ProteinMPNN -> structure prediction ->
pLDDT/pTM/pAE -> keep/fallback/migrate/drop. It is open loop: every campaign
improves the designs, never the model.

This example is what closing that loop costs. The IMPRESS pipeline below is a
stand-in (``run_impress_cycle``), and it runs *unchanged* — ROME is four
calls:

    1. build a Manager with a data policy and a trainer          (setup)
    2. rome.add_training_data(...) as designs are scored          (in the loop)
    3. rome.get_current_model() to pick up improved weights       (in the loop)
    4. await rome.stop()                                          (teardown)

ROME keeps its shared state in a Dragon DDict, so the example runs under the
Dragon runtime with a single-node execution backend::

    dragon examples/agnostic/impress_r.py

On a real allocation, swap ``LocalExecutionBackend`` for the Dragon or RADICAL
backend the campaign already uses — ROME submits its tasks to whatever engine
you hand it.
"""

import asyncio
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import rome

# The ProteinMPNN trainer ships with the IMPRESS-R example, not the framework.
# Put the repo root on sys.path so the example package resolves whether this is
# run as a script or imported.
try:
    from examples.impress_r.protein_binding.mpnn_trainer import (
        ProteinMPNNConfig,
        ProteinMPNNTrainer,
        impress_corpus_filter,
    )
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from examples.impress_r.protein_binding.mpnn_trainer import (
        ProteinMPNNConfig,
        ProteinMPNNTrainer,
        impress_corpus_filter,
    )

NUM_CYCLES = 10
DESIGNS_PER_CYCLE = 8


# ---------------------------------------------------------------------------
# The host workflow. ROME does not know or care what is in here.
# ---------------------------------------------------------------------------

def run_impress_cycle(cycle, mpnn_weights, model_version):
    """Stand-in for backbone -> MPNN -> predict -> extract metrics.

    Returns the per-design metrics a real campaign would parse out of
    ``plddt_extract_pipeline.py``. The fake "improvement" with each published
    checkpoint is only here so the example prints something interesting.
    """
    boost = 4.0 * model_version

    designs = []
    for i in range(DESIGNS_PER_CYCLE):
        designs.append({
            "backbone_id": f"bb{i % 3}",
            "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=40)),
            # The structure file IS the training example: its coordinates
            # and its sequence are the pair ProteinMPNN learns from. For
            # IMPRESS-R that is the structure prediction of the designed
            # sequence, which the campaign already wrote to disk.
            "path": f"/scratch/cycle{cycle}/design{i}_unrelaxed_rank_001.pdb",
            # Baseline sits right at the admission thresholds, so roughly a
            # fifth of designs are accepted before any training; each published
            # checkpoint shifts the distribution and more of them clear.
            "pLDDT": min(99.0, random.gauss(80.0 + boost, 6.0)),
            "pTM": min(0.99, random.gauss(0.82 + boost / 100, 0.05)),
            "pAE": max(0.5, random.gauss(4.5 - boost / 10, 1.2)),
        })
    return designs


def train_proteinmpnn(manifest_path, output_dir, config):
    """Stand-in for the original ProteinMPNN fine-tuning loop.

    ``manifest_path`` is the real thing: the parquet manifest of staged
    structures, chain designations and scores the round would train on. A real
    deployment drops ``train_func`` and lets :class:`ProteinMPNNTrainer` fine-tune
    ``dauparas/ProteinMPNN`` itself (``config.mpnn_repo``); this keeps the example
    runnable without torch or the ProteinMPNN checkout.

    It returns a checkpoint in the *original* ``{"model_state_dict": ...}`` format
    — the format ``protein_mpnn_run.py`` loads — so the stub mirrors what the real
    trainer publishes.
    """
    checkpoint = os.path.join(output_dir, f"{config.model_name}.pt")
    with open(checkpoint, "w") as fd:
        fd.write(f"model_state_dict trained on {manifest_path}\n")
    # The training manager publishes whatever comes back; MPNN loads a file.
    return checkpoint


# ---------------------------------------------------------------------------
# ROME adoption
# ---------------------------------------------------------------------------

async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)
    checkpoint_dir = tempfile.mkdtemp(prefix="impress_r_")

    # (1) Setup. The data policy is IMPRESS's own confidence thresholds: only
    # designs the campaign is already confident in are worth training on.
    manager = rome.Manager(
        flow,
        data_config=rome.DataConfig(
            # Only a fraction of designs clear the confidence thresholds, so
            # the threshold is in accepted designs, not designs attempted.
            min_samples=4,
            filter_func=impress_corpus_filter(min_pLDDT=80.0, min_pTM=0.8, max_pAE=5.0),
            dedup_key=lambda record: record["sequence"],
            sampling="top_k",
            shard_size=64,
            score_key="pLDDT",
        ),
        trainer_config=rome.TrainerConfig(
            trainer=ProteinMPNNTrainer(
                ProteinMPNNConfig(
                    train_func=train_proteinmpnn,
                    # A binder design is a complex: chain A is designed, chain B
                    # (the target peptide) is fixed context — scored for the
                    # former, not the latter.
                    design_chains=("A",),
                    context_chains=("B",),
                ),
                gpus=1,
            ),
            checkpoint_dir=checkpoint_dir,
            poll_interval=0.5,
        ),
    )
    await manager.start()

    try:
        for cycle in range(NUM_CYCLES):
            # (3) The pipeline picks up improved weights when they exist. Before
            # the first round this is None and IMPRESS runs exactly as it always
            # has.
            weights = manager.get_current_model()

            designs = run_impress_cycle(cycle, weights, manager.model_version)

            # (2) Scored outputs go into the corpus. This is the only line the
            # host workflow adds to its inner loop.
            for design in designs:
                manager.add_training_data(score=design["pLDDT"], **design)

            mean_plddt = sum(d["pLDDT"] for d in designs) / len(designs)
            print(
                f"cycle {cycle}: mean pLDDT {mean_plddt:5.1f} | "
                f"corpus {manager.data.total_count:3d} "
                f"({manager.data.unconsumed_count} fresh) | "
                f"model v{manager.model_version} | "
                f"{manager.get_training_status().name}"
            )

            # Give the training manager room to notice the threshold and run a
            # round. A real campaign spends this time folding structures.
            await asyncio.sleep(1.0)

        print("\nfinal report:", manager.report())
    finally:
        # (4) Teardown. An in-flight round is waited out, not killed.
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
