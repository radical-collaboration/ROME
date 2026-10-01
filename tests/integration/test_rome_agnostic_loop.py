"""End-to-end ROME: a host workflow that improves the model it is using.

These run the three managers together against a fake asyncflow engine and a
plain dict standing in for a Dragon DDict. No GPUs, no real model — the point
is the wiring: scored outputs accumulate, a round fires, a checkpoint is
published, and the running inference stream swaps onto it mid-campaign.
"""

import asyncio

from rome import (
    DataConfig,
    Manager,
    StreamConfig,
    StreamKind,
    TrainerConfig,
    TrainerStatus,
)
from rome.train.base import TrainTask


class FakeTrainer(TrainTask):
    """Writes a marker file instead of training, and remembers its input."""

    def __init__(self, **kwargs):
        super().__init__(name="fake", **kwargs)
        self.rounds = []

    def train(self, dataset, output_dir, **kwargs):
        self.rounds.append(list(dataset))
        return output_dir


async def _settle(predicate, timeout=3.0, interval=0.01):
    for _ in range(int(timeout / interval)):
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# -- data -> training -> checkpoint ---------------------------------------

def test_data_accumulates_then_training_fires_and_publishes(asyncflow, tmp_path):
    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=4),
        trainer_config=TrainerConfig(
            trainer=trainer,
            checkpoint_dir=str(tmp_path / "ckpt"),
            poll_interval=0.01,
        ),
    )

    async def scenario():
        await rome.start()
        assert rome.get_training_status() is TrainerStatus.NOT_ENOUGH_DATA

        # The host workflow scores things and hands them over. From anywhere.
        for i in range(4):
            rome.add_training_data(prompt=f"p{i}", completion=f"c{i}", score=i / 4)

        await _settle(lambda: rome.model_version == 1)
        await rome.stop()

    asyncio.run(scenario())
    assert len(trainer.rounds) == 1
    assert len(trainer.rounds[0]) == 4
    assert rome.get_current_model().endswith("fake/v1")


def test_manual_trigger_trains_before_the_threshold(asyncflow, tmp_path):
    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=1000),
        trainer_config=TrainerConfig(
            trainer=trainer, checkpoint_dir=str(tmp_path), auto_train=False
        ),
    )

    async def scenario():
        await rome.start()
        rome.add_training_data(prompt="p", score=1.0)
        checkpoint = await rome.start_training()
        await rome.stop()
        return checkpoint

    checkpoint = asyncio.run(scenario())
    assert checkpoint is not None
    assert len(trainer.rounds) == 1


# -- the closed loop -------------------------------------------------------

def test_streams_swap_onto_the_checkpoint_the_campaign_produced(asyncflow, tmp_path):
    """The whole ROME story in one test.

    An inference stream serves requests using model v0. The workflow scores the
    results into the corpus. Once enough accumulate the training manager runs a
    round and publishes v1 — and the *same running stream* starts answering
    with v1 without the workflow orchestrating anything.
    """
    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=3),
        trainer_config=TrainerConfig(
            trainer=trainer,
            checkpoint_dir=str(tmp_path / "ckpt"),
            poll_interval=0.01,
            max_rounds=1,
        ),
        stream_configs=[
            StreamConfig(
                name="infer",
                model_path="v0",
                load_func=lambda path: path,          # "loading" is identity here
                process_func=lambda prompts, ctx: [f"{p}@{ctx.model}" for p in prompts],
                poll_interval=0.01,
            )
        ],
    )

    async def scenario():
        await rome.start()

        before = await rome.stream.get_output(rome.submit("p0"), timeout=3.0)

        for i in range(3):
            rome.add_training_data(prompt=f"p{i}", score=1.0)

        await _settle(lambda: rome.model_version == 1)

        after = await rome.stream.get_output(rome.submit("p1"), timeout=3.0)
        await rome.stop()
        return before, after

    before, after = asyncio.run(scenario())
    assert before["result"] == "p0@v0"
    assert after["result"].endswith("fake/v1")
    assert after["model_version"] == 1


def test_reward_stream_results_land_in_the_corpus(asyncflow, tmp_path):
    """A reward stream is wired to the data manager without the workflow asking."""
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=1000),   # never auto-fires
        trainer_config=TrainerConfig(
            trainer=FakeTrainer(), checkpoint_dir=str(tmp_path), auto_train=False
        ),
        stream_configs=[
            StreamConfig(
                name="score",
                kind=StreamKind.REWARD,
                process_func=lambda seqs: [
                    {"sequence": s, "score": float(len(s))} for s in seqs
                ],
                poll_interval=0.01,
            )
        ],
    )

    async def scenario():
        await rome.start()
        rome.stream.submit_batch(["MK", "MKV", "MKVA"], stream="score")
        await _settle(lambda: rome.data.total_count == 3)
        await rome.stop()

    asyncio.run(scenario())
    assert sorted(r["score"] for r in rome.data.get_records()) == [2.0, 3.0, 4.0]


def test_a_workflow_can_use_only_the_data_and_training_halves(asyncflow, tmp_path):
    """No streams at all — IMPRESS-R's shape, where inference stays in the host."""
    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=2),
        trainer_config=TrainerConfig(
            trainer=trainer, checkpoint_dir=str(tmp_path), poll_interval=0.01
        ),
    )

    async def scenario():
        await rome.start()
        assert rome.get_stream_status() == []
        rome.add_training_data(sequence="MKV", pdb_path="/x.pdb", score=91.0)
        rome.add_training_data(sequence="MKW", pdb_path="/y.pdb", score=88.0)
        await _settle(lambda: rome.model_version == 1)
        await rome.stop()

    asyncio.run(scenario())
    assert [r["sequence"] for r in trainer.rounds[0]] == ["MKV", "MKW"]


# -- data-manager policy end to end ---------------------------------------

def test_corpus_filters_keep_low_confidence_designs_out_of_training(asyncflow, tmp_path):
    from examples.impress_r.protein_binding.mpnn_trainer import impress_corpus_filter

    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(
            min_samples=2,
            filter_func=impress_corpus_filter(min_pLDDT=80.0, min_pTM=0.8, max_pAE=5.0),
        ),
        trainer_config=TrainerConfig(
            trainer=trainer, checkpoint_dir=str(tmp_path), poll_interval=0.01
        ),
    )

    async def scenario():
        await rome.start()
        # Rejected: pLDDT too low, then pAE too high.
        assert rome.add_training_data(sequence="bad1", pLDDT=60.0, pTM=0.9, pAE=3.0) is None
        assert rome.add_training_data(sequence="bad2", pLDDT=90.0, pTM=0.9, pAE=9.0) is None
        rome.add_training_data(sequence="good1", pLDDT=90.0, pTM=0.9, pAE=3.0)
        rome.add_training_data(sequence="good2", pLDDT=85.0, pTM=0.85, pAE=4.0)
        await _settle(lambda: rome.model_version == 1)
        await rome.stop()

    asyncio.run(scenario())
    assert [r["sequence"] for r in trainer.rounds[0]] == ["good1", "good2"]


def test_second_round_only_sees_data_added_after_the_first(asyncflow, tmp_path):
    trainer = FakeTrainer()
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=2, sampling="recent", shard_size=2),
        trainer_config=TrainerConfig(
            trainer=trainer, checkpoint_dir=str(tmp_path), poll_interval=0.01
        ),
    )

    async def scenario():
        await rome.start()
        rome.add_training_data(prompt="a", score=1.0)
        rome.add_training_data(prompt="b", score=1.0)
        await _settle(lambda: rome.model_version == 1)
        rome.add_training_data(prompt="c", score=1.0)
        rome.add_training_data(prompt="d", score=1.0)
        await _settle(lambda: rome.model_version == 2)
        await rome.stop()

    asyncio.run(scenario())
    assert [r["prompt"] for r in trainer.rounds[0]] == ["a", "b"]
    assert [r["prompt"] for r in trainer.rounds[1]] == ["c", "d"]


# -- housekeeping ----------------------------------------------------------

def test_rome_namespaces_its_keys_in_a_shared_ddict(asyncflow, tmp_path, ddict):
    ddict["workflow_state"] = "mine"
    rome = Manager(
        asyncflow,
        ddict=ddict,
        trainer_config=TrainerConfig(
            trainer=FakeTrainer(), checkpoint_dir=str(tmp_path), auto_train=False
        ),
    )

    async def scenario():
        await rome.start()
        rome.add_training_data(prompt="p", score=1.0)
        await rome.start_training()
        await rome.stop()

    asyncio.run(scenario())
    assert ddict["workflow_state"] == "mine"
    assert all(k == "workflow_state" or k.startswith("rome|") for k in ddict)


def test_report_covers_all_three_components(asyncflow, tmp_path):
    rome = Manager(
        asyncflow,
        data_config=DataConfig(min_samples=2),
        trainer_config=TrainerConfig(
            trainer=FakeTrainer(), checkpoint_dir=str(tmp_path), auto_train=False
        ),
        stream_configs=[
            StreamConfig(name="infer", process_func=lambda xs: xs, poll_interval=0.01)
        ],
    )

    async def scenario():
        await rome.start()
        rome.add_training_data(prompt="p", score=1.0)
        report = rome.report()
        await rome.stop()
        return report

    report = asyncio.run(scenario())
    assert report["data"]["total"] == 1
    assert report["data"]["ready_to_train"] is False
    assert report["training"]["status"] == TrainerStatus.NOT_ENOUGH_DATA.name
    assert report["streams"]["infer"]["tasks"] == 1


def test_manager_works_as_an_async_context_manager(asyncflow, tmp_path):
    rome = Manager(
        asyncflow,
        trainer_config=TrainerConfig(
            trainer=FakeTrainer(), checkpoint_dir=str(tmp_path), auto_train=False
        ),
    )

    async def scenario():
        async with rome:
            assert rome.started
        assert not rome.started

    asyncio.run(scenario())


# -- who owns the workflow engine -----------------------------------------

def test_rome_builds_its_own_engine_when_not_given_one(tmp_path):
    """A host that keeps its engine private can still hand ROME nothing."""
    trainer = FakeTrainer()
    rome = Manager(
        data_config=DataConfig(min_samples=2),
        trainer_config=TrainerConfig(
            trainer=trainer, checkpoint_dir=str(tmp_path), poll_interval=0.01
        ),
    )
    assert rome.asyncflow is None

    async def scenario():
        await rome.start()
        assert rome.asyncflow is not None
        assert rome._owns_asyncflow
        # The engine it built is the one its sub-managers submit to.
        assert rome.trainer.asyncflow is rome.asyncflow
        assert rome.stream.asyncflow is rome.asyncflow

        rome.add_training_data(prompt="a", score=1.0)
        rome.add_training_data(prompt="b", score=1.0)
        await _settle(lambda: rome.model_version == 1, timeout=10.0)
        await rome.stop()

    asyncio.run(scenario())
    assert len(trainer.rounds) == 1
    # Having built it, it shut it down.
    assert rome.asyncflow is None
    assert not rome._owns_asyncflow


def test_a_supplied_engine_is_not_shut_down(asyncflow, tmp_path):
    """The host workflow's engine is still running the host's own tasks."""
    rome = Manager(
        asyncflow,
        trainer_config=TrainerConfig(
            trainer=FakeTrainer(), checkpoint_dir=str(tmp_path), auto_train=False
        ),
    )

    async def scenario():
        await rome.start()
        assert not rome._owns_asyncflow
        await rome.stop()

    asyncio.run(scenario())
    assert rome.asyncflow is asyncflow      # untouched, not torn down
