"""End-to-end check of the interface-targeting surrogate, with real itgt.

Marked ``slow`` and skipped unless both ``itgt`` and a populated IMPRESS
small-molecule input directory are present. No GPU, no Dragon and no IMPRESS
*runtime* -- only its input files, because the point is to prove that the
surrogate round and the spec patch agree with the real target rather than with a
fixture of our own making.

What this covers that the unit tests cannot:

* a round writes a loadable itgt artifact, and writes ``train_complete`` last
* ``artifacts.load``'s feature guard accepts the proposer's view of the artifact
  -- the mismatch that would otherwise surface mid-campaign
* the proposal is well-formed against the *real* 30-atom ligand and passes
  itgt's own RFD3 spec validator
* the patched spec stays loadable by the same code that produced it, so the
  loop can run a second round
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

pytestmark = pytest.mark.slow

ITGT_CHECKOUT = "/home/mason/exdrive/rad/interface-targeting"
IMPRESS_P1_IN = (
    "/home/mason/exdrive/rad/IMPRESS/examples/small_molecule_binding/p1_in"
)


def _require_itgt():
    """Import itgt, falling back to the sibling checkout when not installed."""
    try:
        import itgt  # noqa: F401
    except ImportError:
        if not os.path.isdir(ITGT_CHECKOUT):
            pytest.skip("itgt is neither installed nor checked out")
        import sys

        sys.path.insert(0, ITGT_CHECKOUT)
    return pytest.importorskip("itgt")


@pytest.fixture
def staged(tmp_path):
    """A private copy of a real pipeline input directory."""
    _require_itgt()
    if not os.path.exists(os.path.join(IMPRESS_P1_IN, "ALR_binder_design.json")):
        pytest.skip(f"no populated IMPRESS input dir at {IMPRESS_P1_IN}")
    target = tmp_path / "p1_in"
    shutil.copytree(IMPRESS_P1_IN, target)
    return str(target)


def _fabricate_pairs(input_dir, buffer_path, n=24):
    """Seed the replay buffer with (partition, reward) pairs.

    The reward is the patch's mean oxygen content -- a signal the pooled
    features genuinely carry, so a *working* pipeline recovers it and a broken
    one does not. This stands in for the campaign, which is what supplies these
    pairs in production.
    """
    from itgt.common.datasets import DatasetConfig, featurize_ligand
    from itgt.common.features import feature_names
    from itgt.method1_random.sampler import SamplerConfig, sample_targets

    from examples.impress_r.interface_targeting import itgt_propose, itgt_spec
    from examples.impress_r.interface_targeting.itgt_train_wrapper import (
        pooled_features,
    )

    path = itgt_spec.spec_path(input_dir)
    spec = itgt_spec.load_spec(path)
    ligand_key = itgt_spec.ligand_key(spec)
    reference = itgt_spec.reference_pdb(path, spec)

    config = DatasetConfig()
    ligand = featurize_ligand(reference, ligand_key, config)
    oxygen = feature_names(config.features).index("element_O")

    targets = sample_targets(
        reference,
        ligand_key,
        n=n,
        config=SamplerConfig(min_atoms=8, max_atoms=16, mark_fringe_partial=False,
                             seed=11),
    )
    assert targets, "the sampler produced no candidate partitions"
    for index, target in enumerate(targets):
        features = pooled_features(ligand, target.flat_ligand_atoms())
        itgt_propose.record_action(
            buffer_path,
            target,
            {"ligand_iptm": float(features[oxygen])},
            group=f"p1|pass{index}",
            source="synthetic",
            logger=lambda _m: None,
        )
    return reference, ligand_key, len(targets)


def test_a_round_fits_publishes_and_proposes(staged, tmp_path):
    from examples.impress_r.interface_targeting import itgt_propose, itgt_spec
    from examples.impress_r.interface_targeting.itgt_train_wrapper import (
        TRAIN_COMPLETE_MARKER,
        run_round,
    )

    itgt_spec.snapshot_template(staged)
    buffer_path = str(tmp_path / "replay.jsonl")
    # The v0 action, so the first reward has something to attach to.
    assert itgt_propose.seed_template_action(buffer_path, [staged]) == 1

    reference, ligand_key, n_pairs = _fabricate_pairs(staged, buffer_path)

    output_dir = str(tmp_path / "round_v1")
    job = {
        "config": {
            "itgt_dir": ITGT_CHECKOUT,
            "component": "reward_ridge",
            "replay_buffer": buffer_path,
            "reference_pdb": reference,
            "ligand_key": ligand_key,
            "ridge_alpha": 1.0,
            "seed": 0,
        },
        "artifact": os.path.join(output_dir, "artifact"),
        "output_dir": output_dir,
        "report": os.path.join(output_dir, "report.json"),
        "error_log": os.path.join(output_dir, "err.log"),
        "designs": [],
        "model_version": 1,
    }

    artifact = run_round(job)

    # The marker is the training manager's completion signal on Dragon, and it
    # has to land after the artifact.
    assert os.path.exists(os.path.join(output_dir, TRAIN_COMPLETE_MARKER))
    assert os.path.exists(os.path.join(artifact, "meta.json"))
    assert itgt_propose.artifact_kind(artifact) == "RewardRidge"

    report = json.loads(open(job["report"]).read())
    assert report["n_pairs"] == n_pairs
    assert report["n_skipped"] == 1       # the v0 action, which has no reward yet
    assert report["validated"] is True
    # The fabricated signal is real and linear, so a working pipeline beats
    # predicting the campaign mean. This is the assertion that fails if the
    # feature pooling, the artifact round-trip or the fit is broken.
    assert report["logo_r2"] > 0.0

    # The proposer's view of the artifact must satisfy artifacts.load's feature
    # guard -- this is the mismatch that would otherwise appear mid-campaign.
    groups, target, _ = itgt_propose.propose_groups(
        artifact, reference, ligand_key, n_candidates=32, min_atoms=10,
        max_atoms=18, seed=5,
    )
    present = itgt_spec.ligand_atom_names(reference, ligand_key)
    ok, missing = itgt_spec.coverage_ok(groups, present)
    assert ok, f"proposal names atoms absent from the reference: {missing}"
    assert sum(len(v) for v in groups.values()) == len(present)
    assert 10 <= len(groups["buried"]) <= 18


def test_apply_rewrites_the_spec_and_keeps_it_loadable(staged, tmp_path):
    """A second round has to be able to read what the first one wrote."""
    from examples.impress_r.interface_targeting import itgt_propose, itgt_spec
    from examples.impress_r.interface_targeting.itgt_train_wrapper import run_round

    itgt_spec.snapshot_template(staged)
    buffer_path = str(tmp_path / "replay.jsonl")
    reference, ligand_key, _ = _fabricate_pairs(staged, buffer_path, n=16)

    output_dir = str(tmp_path / "round")
    artifact = run_round(
        {
            "config": {
                "itgt_dir": ITGT_CHECKOUT, "component": "reward_ridge",
                "replay_buffer": buffer_path, "reference_pdb": reference,
                "ligand_key": ligand_key, "seed": 0,
            },
            "artifact": os.path.join(output_dir, "artifact"),
            "output_dir": output_dir, "designs": [], "model_version": 1,
            "report": os.path.join(output_dir, "report.json"),
            "error_log": os.path.join(output_dir, "err.log"),
        }
    )

    before = itgt_spec.template_partition(
        itgt_spec.load_spec(itgt_spec.spec_path(staged))
    )
    # Capture the log rather than discarding it: apply_to_pipeline never
    # raises, so a no-op logger would turn any failure into a bare `False`.
    messages: list = []
    assert itgt_propose.apply_to_pipeline(
        staged, artifact, 1, itgt_dir=ITGT_CHECKOUT, n_candidates=32, seed=5,
        logger=messages.append,
    ), messages
    after = itgt_spec.template_partition(
        itgt_spec.load_spec(itgt_spec.spec_path(staged))
    )
    assert before != after

    # Configuration the workflow does not own survived.
    live = itgt_spec.load_spec(itgt_spec.spec_path(staged))
    template = itgt_spec.load_template(staged)
    for field in ("input", "ligand", "length"):
        assert live["partial"][field] == template["partial"][field]
    assert "select_fixed_atoms" not in live["partial"]

    # The round's partition is auditable after the base spec moved.
    assert os.path.exists(
        os.path.join(staged, itgt_spec.ITGT_DIR_NAME, "history", "v1.json")
    )

    # The action a pass would now run under is the patched one, not the
    # template -- which is what makes the next (action, reward) pair informative.
    assert (
        itgt_propose.installed_action(staged).key()
        != itgt_propose.template_action(staged).key()
    )
