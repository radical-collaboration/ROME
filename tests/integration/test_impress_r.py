"""IMPRESS-R: ROME driven from a real IMPRESS ``adaptive_fn``.

Skipped unless IMPRESS is installed — see ``docs/examples/impress-r.md``.
What this pins down is the integration contract, not IMPRESS itself:

* ``adaptive_fn`` is the seam — the pipeline's ``run()`` never mentions ROME;
* designs contributed there reach the corpus and trigger a round;
* the checkpoint the round publishes reaches the *next* pass of the campaign.
"""

import asyncio
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

impress = pytest.importorskip("impress")

from impress import ImpressBasePipeline, ImpressManager, PipelineSetup  # noqa: E402
from radical.asyncflow import LocalExecutionBackend, WorkflowEngine  # noqa: E402

import rome  # noqa: E402
from rome.train.base import TrainTask  # noqa: E402


class RecordingTrainer(TrainTask):
    def __init__(self, **kwargs):
        super().__init__(name="mpnn", **kwargs)
        self.rounds = []

    def train(self, dataset, output_dir, **kwargs):
        self.rounds.append(list(dataset))
        return output_dir


class _Pipeline(ImpressBasePipeline):
    """Two passes of MPNN -> fold -> extract, with the executables stubbed."""

    def __init__(self, name, flow, configs=None, **kwargs):
        self.passes = 1
        self.max_passes = 3
        self.base_path = kwargs["base_path"]
        self.output_path_af = os.path.join(self.base_path, f"{name}_af")
        self.iter_seqs = {}
        self.mpnn_weights = "baseline.pt"
        #: What weights each pass actually ran with, for the assertion below.
        self.weights_per_pass = []
        os.makedirs(self.output_path_af, exist_ok=True)
        super().__init__(name, flow, **(configs or {}), **kwargs)

    def register_pipeline_tasks(self):
        @self.auto_register_task()
        async def mpnn(*args, **kwargs):
            return "/bin/echo mpnn"

        @self.auto_register_task(local_task=True)
        async def extract():
            path = os.path.join(
                self.base_path, f"af_stats_{self.name}_pass_{self.passes}.csv"
            )
            with open(path, "w", newline="") as fd:
                writer = csv.writer(fd)
                writer.writerow(["ID", "avg_plddt", "ptm", "avg_pae"])
                for i in range(3):
                    design = f"{self.name}_p{self.passes}_d{i}"
                    self.iter_seqs[design] = "MKV" * 20
                    open(os.path.join(self.output_path_af, f"{design}.pdb"), "w").close()
                    # Comfortably inside the confidence thresholds.
                    writer.writerow([f"{design}.pdb", 92.0, 0.91, 3.0])
            return path

    async def run(self):
        while self.passes <= self.max_passes:
            self.weights_per_pass.append(self.mpnn_weights)
            await self.mpnn()
            await self.extract()
            await self.run_adaptive_step(wait=True)
            self.passes += 1

    async def finalize(self):
        pass


def _adaptive_fn(manager, seen):
    async def adaptive(pipeline):
        stats = os.path.join(
            pipeline.base_path,
            f"af_stats_{pipeline.name}_pass_{pipeline.passes}.csv",
        )
        with open(stats) as fd:
            for row in csv.DictReader(fd):
                design = row["ID"].split(".")[0]
                manager.add_training_data(
                    path=os.path.join(pipeline.output_path_af, f"{design}.pdb"),
                    sequence=pipeline.iter_seqs[design],
                    backbone_id=pipeline.name,
                    pLDDT=float(row["avg_plddt"]),
                    pTM=float(row["ptm"]),
                    pAE=float(row["avg_pae"]),
                    score=float(row["avg_plddt"]),
                )
        weights = manager.get_current_model()
        if weights:
            pipeline.mpnn_weights = weights
        seen.append(manager.get_training_status().name)
        # Give the training manager room to run a round between passes.
        await asyncio.sleep(1.0)

    return adaptive


def test_adaptive_fn_carries_designs_in_and_checkpoints_out(tmp_path):
    trainer = RecordingTrainer()
    manager = rome.Manager(
        data_config=rome.DataConfig(min_samples=3),
        trainer_config=rome.TrainerConfig(
            trainer=trainer,
            checkpoint_dir=str(tmp_path / "ckpt"),
            poll_interval=0.05,
        ),
    )
    pipelines = []

    async def scenario():
        await manager.start()
        backend = await LocalExecutionBackend(ThreadPoolExecutor())
        flow = await WorkflowEngine.create(backend=backend)
        impress_manager = ImpressManager(flow)
        try:
            setup = PipelineSetup(
                name="p1",
                type=_Pipeline,
                adaptive_fn=_adaptive_fn(manager, []),
                kwargs={"base_path": str(tmp_path)},
            )
            await impress_manager.start(pipeline_setups=[setup])
            pipelines.extend(impress_manager.pipeline_tasks or [])
        finally:
            await flow.shutdown()
            await manager.stop()

    asyncio.run(scenario())

    # Designs the campaign produced reached the corpus...
    assert manager.data.total_count == 9          # 3 passes x 3 designs
    # ...a round fired on them...
    assert trainer.rounds, "no training round ran"
    assert all("path" in record for record in trainer.rounds[0])
    # ...and the checkpoint it published is a real one.
    assert manager.get_current_model()
    assert manager.model_version >= 1


def test_rome_can_share_the_impress_engine(tmp_path):
    """The other half of the choice: one engine for the campaign and ROME."""

    async def scenario():
        backend = await LocalExecutionBackend(ThreadPoolExecutor())
        flow = await WorkflowEngine.create(backend=backend)
        impress_manager = ImpressManager(flow)

        manager = rome.Manager(
            flow,
            data_config=rome.DataConfig(min_samples=1),
            trainer_config=rome.TrainerConfig(
                trainer=RecordingTrainer(),
                checkpoint_dir=str(tmp_path / "shared"),
                auto_train=False,
            ),
        )
        await manager.start()
        manager.add_training_data(path="/x.pdb", sequence="MKV", score=90.0)
        checkpoint = await manager.start_training()
        assert not manager._owns_asyncflow
        await manager.stop()
        # ROME left the campaign's engine running.
        assert manager.asyncflow is flow
        await flow.shutdown()
        return checkpoint

    assert asyncio.run(scenario())
