"""The two shipped trainer tasks, as far as they go without GPUs."""

import os

import pytest

from rome.train.base import FunctionTrainer, TrainTask


# -- TrainTask contract ----------------------------------------------------

def test_base_train_is_abstract():
    with pytest.raises(NotImplementedError):
        TrainTask().train([], "/tmp")


def test_validate_rejects_an_empty_dataset():
    with pytest.raises(ValueError, match="empty dataset"):
        TrainTask(name="t").validate([])


def test_checkpoint_layout_is_name_and_version(tmp_path):
    path = TrainTask.prepare_output_dir(str(tmp_path), 3, "grpo")
    assert path == os.path.join(str(tmp_path), "grpo", "v3")
    assert os.path.isdir(path)


def test_function_trainer_defaults_to_the_output_dir(tmp_path):
    trainer = FunctionTrainer(lambda dataset, output_dir, **kw: None)
    assert trainer.train([1], str(tmp_path)) == str(tmp_path)


def test_function_trainer_takes_the_functions_name():
    def my_finetune(dataset, output_dir, **kwargs):
        return output_dir

    assert FunctionTrainer(my_finetune).name == "my_finetune"


# -- ProteinMPNN (original dauparas implementation) ------------------------

_MONO_PDB = "ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 90.00           C\nEND\n"
_DIMER_PDB = (
    "ATOM      1  CA  MET A   1      0.000   0.000   0.000  1.00 90.00           C\n"
    "ATOM      2  CA  GLU B   1      3.800   0.000   0.000  1.00 88.00           C\nEND\n"
)


def _design(tmp_path, uid, *, backbone="p1", seq="MKTAYIAKQR", pdb=_DIMER_PDB):
    path = tmp_path / f"{uid}.pdb"
    path.write_text(pdb)
    return {"uid": uid, "backbone_id": backbone, "path": str(path),
            "sequence": seq, "pLDDT": 95.0, "pTM": 0.9, "pAE": 3.0}


def test_mpnn_needs_a_structure_path(tmp_path):
    from examples.impress_r.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    with pytest.raises(ValueError, match="path"):
        trainer.validate([{"sequence": "MKV"}])


def test_config_requires_a_repo_or_a_train_func():
    """The built-in loop needs the ProteinMPNN checkout; the escape hatch does not."""
    from examples.impress_r.mpnn import ProteinMPNNConfig

    with pytest.raises(ValueError, match="mpnn_repo"):
        ProteinMPNNConfig().validate()
    ProteinMPNNConfig(mpnn_repo="/opt/ProteinMPNN").validate()      # fine
    ProteinMPNNConfig(train_func=lambda *a: "x").validate()          # fine


def test_manifest_records_chain_designation(tmp_path):
    """The manifest is the audit trail, and it carries the dimer designation."""
    import pandas as pd
    from examples.impress_r.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=lambda *a: "x"))
    path = trainer.write_manifest([_design(tmp_path, "d1")], str(tmp_path / "r"))
    row = pd.read_parquet(path).iloc[0]
    assert row["designed_chains"] == "A"      # chain A designed
    assert row["context_chains"] == "B"       # peptide is context
    assert row["design_id"] == "d1"


def test_custom_train_func_receives_the_manifest(tmp_path):
    from examples.impress_r.mpnn import ProteinMPNNConfig, ProteinMPNNTrainer

    seen = {}

    def my_train(manifest_path, output_dir, config):
        seen["manifest"] = manifest_path
        return output_dir

    trainer = ProteinMPNNTrainer(ProteinMPNNConfig(train_func=my_train))
    result = trainer.train([_design(tmp_path, "d1")], str(tmp_path))
    assert result == str(tmp_path)
    assert seen["manifest"].endswith(".parquet")


def test_published_checkpoint_is_original_format(tmp_path):
    """protein_mpnn_run.py loads model_state_dict + num_edges; both must be there."""
    from examples.impress_r.mpnn import original_checkpoint

    ckpt = original_checkpoint({"w": 1}, num_edges=48, noise_level=0.2)
    assert ckpt["model_state_dict"] == {"w": 1}
    assert ckpt["num_edges"] == 48 and ckpt["noise_level"] == 0.2


def test_weights_publish_into_the_repo_when_asked(tmp_path):
    """publish_into_repo names the fixed pointer IMPRESS's next pass loads."""
    from examples.impress_r.mpnn import (
        ProteinMPNNConfig,
        repo_pointer_path,
        versioned_checkpoint_path,
    )

    repo = str(tmp_path / "ProteinMPNN")
    cfg = ProteinMPNNConfig(mpnn_repo=repo, model_name="v_48_020",
                            publish_into_repo=True)
    # The pointer the wrapper copies the latest version onto (no version in it).
    assert repo_pointer_path(cfg) == f"{repo}/vanilla_model_weights/v_48_020.pt"

    # Without publish_into_repo there is no repo pointer.
    cfg2 = ProteinMPNNConfig(train_func=lambda *a: "x", model_name="v_48_020")
    assert repo_pointer_path(cfg2) is None

    # The durable checkpoint is versioned and lives in the round's own dir.
    out = str(tmp_path / "round")
    assert versioned_checkpoint_path(cfg, out, 3) == f"{out}/v_48_020_v3.pt"



@pytest.mark.parametrize(
    "record,expected",
    [
        ({"pLDDT": 90.0, "pTM": 0.9, "pAE": 3.0}, True),
        ({"pLDDT": 79.9, "pTM": 0.9, "pAE": 3.0}, False),   # pLDDT too low
        ({"pLDDT": 90.0, "pTM": 0.7, "pAE": 3.0}, False),   # pTM too low
        ({"pLDDT": 90.0, "pTM": 0.9, "pAE": 5.1}, False),   # pAE too high
        ({"score": 90.0}, True),                            # score stands in for pLDDT
        ({}, False),                                        # nothing to judge on
    ],
)
def test_impress_corpus_filter(record, expected):
    from examples.impress_r.mpnn import impress_corpus_filter

    assert impress_corpus_filter()(record) is expected


# -- LLM / GRPO ------------------------------------------------------------

def test_model_config_resolves_weights_and_tokenizer():
    from rome.train.llm import ModelConfig

    cfg = ModelConfig(base_model_name="base", model_name="finetuned")
    assert cfg.resolved_model_name() == "finetuned"    # prefer the fine-tune
    assert cfg.tokenizer_name() == "base"              # tokenizer from the base

    only_base = ModelConfig(base_model_name="base")
    assert only_base.resolved_model_name() == "base"


def test_model_config_needs_a_model():
    from rome.train.llm import ModelConfig

    with pytest.raises(ValueError, match="base_model_name or model_name"):
        ModelConfig().resolved_model_name()


def test_grpo_trainer_needs_a_model_config():
    from rome.train.llm import GRPOConfig, GRPOTrainer

    with pytest.raises(ValueError, match="model_config must be set"):
        GRPOTrainer(GRPOConfig())


def test_grpo_takes_its_gpus_from_the_model_config():
    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(
        GRPOConfig(model_config=ModelConfig(base_model_name="m", required_gpus=8))
    )
    assert task.gpus == 8
    assert task.wants_hf_dataset is True


def test_grpo_rejects_a_corpus_without_prompts():
    from datasets import Dataset

    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(GRPOConfig(model_config=ModelConfig(base_model_name="m")))
    dataset = Dataset.from_list([{"sequence": "MKV", "score": 1.0}])
    with pytest.raises(ValueError, match="needs a 'prompt' column"):
        task.validate(dataset)


def test_grpo_accepts_a_corpus_with_prompts():
    from datasets import Dataset

    from rome.train.llm import GRPOConfig, GRPOTrainer, ModelConfig

    task = GRPOTrainer(GRPOConfig(model_config=ModelConfig(base_model_name="m")))
    task.validate(Dataset.from_list([{"prompt": "2+2?", "score": 1.0}]))


def test_grpo_builds_a_trl_config_pointing_at_the_round_output(tmp_path):
    from rome.train.llm import GRPOConfig, ModelConfig

    config = GRPOConfig(
        model_config=ModelConfig(base_model_name="m"),
        learning_rate=1e-5,
        extra_args={"seed": 7},
    )
    trl_config = config.build_trl_config(str(tmp_path))
    assert trl_config.output_dir == str(tmp_path)
    assert trl_config.learning_rate == 1e-5
    assert trl_config.seed == 7


# -- SFT (the "train on the good ones" trainer) ----------------------------

def test_sft_trainer_needs_a_model_config():
    from rome.train.llm import SFTConfig, SFTTrainer

    with pytest.raises(ValueError, match="model_config must be set"):
        SFTTrainer(SFTConfig())


def test_sft_takes_its_gpus_from_the_model_config():
    from rome.train.llm import ModelConfig, SFTConfig, SFTTrainer

    task = SFTTrainer(
        SFTConfig(model_config=ModelConfig(base_model_name="m", required_gpus=4))
    )
    assert task.gpus == 4 and task.name == "sft" and task.wants_hf_dataset is True


def test_sft_accepts_prompt_completion_or_preformatted_text():
    from datasets import Dataset

    from rome.train.llm import ModelConfig, SFTConfig, SFTTrainer

    task = SFTTrainer(SFTConfig(model_config=ModelConfig(base_model_name="m")))
    task.validate(Dataset.from_list([{"prompt": "2+2?", "completion": "4"}]))
    task.validate(Dataset.from_list([{"text": "Q: 2+2? A: 4"}]))


def test_sft_rejects_a_corpus_missing_both():
    from datasets import Dataset

    from rome.train.llm import ModelConfig, SFTConfig, SFTTrainer

    task = SFTTrainer(SFTConfig(model_config=ModelConfig(base_model_name="m")))
    with pytest.raises(ValueError, match="needs either a 'text'"):
        task.validate(Dataset.from_list([{"prompt": "2+2?", "score": 1.0}]))


def test_sft_builds_a_trl_config_pointing_at_the_round_output(tmp_path):
    from rome.train.llm import ModelConfig, SFTConfig

    config = SFTConfig(model_config=ModelConfig(base_model_name="m"),
                       learning_rate=3e-5, extra_args={"seed": 11})
    trl_config = config.build_trl_config(str(tmp_path))
    assert trl_config.output_dir == str(tmp_path)
    assert trl_config.learning_rate == 3e-5
    assert trl_config.seed == 11


def test_format_chat_falls_back_without_a_template():
    from rome.train.llm import _format_chat

    class _NoTemplate:            # a bare tokenizer, no apply_chat_template
        pass

    text = _format_chat(_NoTemplate(), "2+2?", "4")
    assert "2+2?" in text and "4" in text


# -- on-the-fly calibration ------------------------------------------------

def _scored(uid, ptm, pae):
    return {"uid": uid, "backbone_id": "bb", "path": f"/{uid}.pdb",
            "sequence": "A" * 90, "pTM": ptm, "pAE": pae}


def test_percentile_sampler_keeps_the_best_fraction():
    from examples.impress_r.mpnn import percentile_sampler

    corpus = [_scored(f"d{i}", 0.70 + i * 0.02, 8.0 - i * 0.4) for i in range(20)]
    kept = percentile_sampler(0.25, min_shard=1)(corpus)

    assert len(kept) == 5
    assert [r["uid"] for r in kept] == ["d19", "d18", "d17", "d16", "d15"]


def test_percentile_sampler_needs_no_threshold_to_calibrate():
    """The same fraction selects sensibly on two incompatible score scales."""
    from examples.impress_r.mpnn import percentile_sampler

    sampler = percentile_sampler(0.5, min_shard=1)
    # A generous predictor and a harsh one, same ordering.
    generous = [_scored(f"g{i}", 0.90 + i * 0.005, 3.0 - i * 0.1) for i in range(10)]
    harsh = [_scored(f"h{i}", 0.40 + i * 0.03, 18.0 - i * 0.9) for i in range(10)]

    assert [r["uid"][1:] for r in sampler(generous)] == \
           [r["uid"][1:] for r in sampler(harsh)]


def test_percentile_sampler_balances_the_two_metrics():
    """Rank-averaging, so pAE's open-ended scale cannot swamp pTM's 0-1."""
    from examples.impress_r.mpnn import percentile_sampler

    corpus = [
        _scored("both_good", 0.95, 2.0),
        _scored("ptm_only", 0.94, 30.0),      # a pAE outlier
        _scored("pae_only", 0.60, 2.1),
        _scored("both_bad", 0.55, 25.0),
    ]
    kept = [r["uid"] for r in percentile_sampler(0.5, min_shard=1)(corpus)]
    assert kept[0] == "both_good"
    assert "both_bad" not in kept


def test_percentile_sampler_floors_the_shard_on_a_small_corpus():
    from examples.impress_r.mpnn import percentile_sampler

    corpus = [_scored(f"d{i}", 0.8 + i * 0.01, 5.0 - i * 0.1) for i in range(6)]
    assert len(percentile_sampler(0.1, min_shard=4)(corpus)) == 4
    # ...but never more than the corpus holds.
    assert len(percentile_sampler(0.1, min_shard=99)(corpus)) == 6


def test_percentile_sampler_ranks_on_whatever_the_corpus_carries():
    """A campaign emitting only pTM still ranks; a scoreless one is kept whole."""
    from examples.impress_r.mpnn import percentile_sampler

    ptm_only = [{"uid": f"d{i}", "pTM": 0.5 + i * 0.1} for i in range(4)]
    assert [r["uid"] for r in percentile_sampler(0.5, min_shard=1)(ptm_only)] == ["d3", "d2"]

    unscored = [{"uid": "a"}, {"uid": "b"}]
    assert len(percentile_sampler(0.5, min_shard=1)(unscored)) == 2


def test_percentile_sampler_reports_the_equivalent_thresholds():
    """on_summary is how a run tells you what a fixed filter would have used."""
    from examples.impress_r.mpnn import percentile_sampler

    seen = {}
    corpus = [_scored(f"d{i}", 0.70 + i * 0.02, 8.0 - i * 0.4) for i in range(10)]
    percentile_sampler(0.5, min_shard=1, on_summary=seen.update)(corpus)

    assert seen["corpus"] == 10 and seen["selected"] == 5
    assert seen["cutoffs"]["pTM"] == pytest.approx(0.80)   # lowest kept pTM
    assert seen["cutoffs"]["pAE"] == pytest.approx(6.0)    # highest kept pAE
    assert seen["percentiles"]["pTM"]["n"] == 10


def test_percentile_sampler_rejects_a_nonsense_fraction():
    from examples.impress_r.mpnn import percentile_sampler

    with pytest.raises(ValueError, match="fraction"):
        percentile_sampler(0.0)
    with pytest.raises(ValueError, match="high.*low"):
        percentile_sampler(0.5, rank_by={"pTM": "sideways"})


def test_score_percentiles_summarises_what_the_campaign_produced():
    from examples.impress_r.mpnn import score_percentiles

    corpus = [_scored(f"d{i}", 0.5 + i * 0.05, float(i)) for i in range(10)]
    summary = score_percentiles(corpus, keys=("pTM", "pAE", "pLDDT"))

    assert "pLDDT" not in summary            # no record carries it
    assert summary["pTM"]["n"] == 10
    assert summary["pAE"]["min"] == 0.0 and summary["pAE"]["max"] == 9.0
    assert summary["pAE"]["median"] == pytest.approx(5.0)
