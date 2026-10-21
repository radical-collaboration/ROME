"""Real ProteinMPNN fine-tune, end to end — gated on torch + a checkout.

This runs :func:`examples.impress_r.protein_binding.mpnn_train_wrapper.run_round` for real: parse a dimer,
fine-tune the training ``ProteinMPNN`` with the repo's own ``featurize`` /
``loss_smoothed`` / ``NoamOpt``, publish an original-format checkpoint, and load
it back into the *inference* ``ProteinMPNN`` (``protein_mpnn_run.py``'s model).
It exercises both the in-process ``train()`` path and the wrapper run as a
subprocess the way the manager submits it (``as_command``).

It needs torch and a ``dauparas/ProteinMPNN`` checkout, so it skips unless both
are present. Point it at the checkout IMPRESS runs::

    ROME_MPNN_TEST_REPO=$WORK/ProteinMPNN pytest tests/integration/test_mpnn_train_real.py

The structure is a gap-free multi-chain PDB the checkout ships (repo ``inputs/``),
so no external data is needed — the point is that the loop runs and the published
checkpoint reloads into the inference model, not that the design is good.
"""

import os

import pytest

torch = pytest.importorskip("torch")

REPO = os.environ.get("ROME_MPNN_TEST_REPO")
pytestmark = pytest.mark.skipif(
    not (REPO and os.path.isdir(os.path.join(REPO or "", "training"))),
    reason="set ROME_MPNN_TEST_REPO to a dauparas/ProteinMPNN checkout to run",
)

def _repo_complex_pdb():
    """A gap-free multi-chain PDB the checkout ships — parseable by parse_PDB.

    Hand-formatting PDB columns is fragile (parse_PDB reads fixed columns), so
    the test uses a real complex from the repo's own inputs and designates its
    first chain designed, the rest context — the exact split IMPRESS-R relies on.
    It must be gap-free: ``StructureDataset`` drops any sequence with a char
    outside its alphabet (a crystal-structure gap is ``-``). AF predictions, the
    real training input, have no gaps.
    """
    import glob
    import sys

    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "training"))
    from protein_mpnn_utils import parse_PDB  # type: ignore

    alphabet = set("ACDEFGHIKLMNPQRSTVWYX")
    for path in sorted(glob.glob(os.path.join(REPO, "inputs", "**", "*.pdb"),
                                 recursive=True)):
        try:
            entries = parse_PDB(path)
        except Exception:
            continue
        for entry in entries:
            chains = [k[len("seq_chain_"):] for k in entry
                      if k.startswith("seq_chain_")]
            if len(chains) >= 2 and set(entry["seq"]) <= alphabet:
                return path, chains
    pytest.skip("no gap-free multi-chain PDB under the checkout's inputs/")


def test_real_finetune_publishes_an_inference_loadable_checkpoint(tmp_path):
    from examples.impress_r.protein_binding.mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer

    pdb, chains = _repo_complex_pdb()
    records = [{"uid": f"d{i}", "path": pdb, "sequence": "",
                "backbone_id": "t", "pLDDT": 90.0, "pTM": 0.9, "pAE": 3.0}
               for i in range(2)]

    cfg = ProteinMPNNConfig(
        mpnn_repo=REPO,
        initial_weights=os.path.join(REPO, "vanilla_model_weights", "v_48_020.pt"),
        model_name="v_48_020", publish_into_repo=False,
        max_epochs=2, device="cpu",
        # design the first chain, keep the rest as context
        chains_func=lambda rec, present: ([chains[0]], chains[1:]),
    )
    trainer = ProteinMPNNTrainer(cfg)
    trainer.validate(records)

    ckpt = trainer.train(records, str(tmp_path / "v1"), model_version=1)
    assert os.path.isfile(ckpt)

    state = torch.load(ckpt, map_location="cpu")
    assert state["num_edges"] == 48
    assert "model_state_dict" in state and "step" in state

    # The real inference model must accept it (protein_mpnn_run.py path).
    import sys
    sys.path.insert(0, REPO)
    import importlib

    inf = importlib.import_module("protein_mpnn_utils")
    model = inf.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                            hidden_dim=128, num_encoder_layers=3,
                            num_decoder_layers=3, k_neighbors=48, augment_eps=0.2)
    model.load_state_dict(state["model_state_dict"])   # raises if incompatible


def test_wrapper_runs_as_a_subprocess_the_way_the_manager_submits_it(tmp_path):
    """The production path: as_command -> `python mpnn_train_wrapper.py --job ...`.

    This is what the training manager actually submits (an executable task), so
    it runs the wrapper as a real subprocess and checks the checkpoint it wrote
    reloads into the inference model — proving the round works out-of-process,
    with nothing about the fine-tune living in the caller.
    """
    import shlex
    import subprocess

    from examples.impress_r.protein_binding.mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer

    pdb, chains = _repo_complex_pdb()
    records = [{"uid": f"d{i}", "path": pdb, "sequence": "", "backbone_id": "t"}
               for i in range(2)]
    cfg = ProteinMPNNConfig(
        mpnn_repo=REPO,
        initial_weights=os.path.join(REPO, "vanilla_model_weights", "v_48_020.pt"),
        model_name="v_48_020", publish_into_repo=False, max_epochs=1, device="cpu",
        chains_func=lambda rec, present: ([chains[0]], chains[1:]),
    )
    trainer = ProteinMPNNTrainer(cfg)
    outdir = str(tmp_path / "v1")
    os.makedirs(outdir)

    command, checkpoint = trainer.as_command(records, outdir, model_version=1)
    result = subprocess.run(shlex.split(command), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == checkpoint   # printed path
    assert os.path.isfile(checkpoint)

    state = torch.load(checkpoint, map_location="cpu")
    assert state["num_edges"] == 48 and "model_state_dict" in state

    import importlib
    import sys
    sys.path.insert(0, REPO)
    inf = importlib.import_module("protein_mpnn_utils")
    model = inf.ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                            hidden_dim=128, num_encoder_layers=3,
                            num_decoder_layers=3, k_neighbors=48, augment_eps=0.2)
    model.load_state_dict(state["model_state_dict"])   # raises if incompatible


def test_second_round_resumes_and_advances_the_schedule(tmp_path):
    from examples.impress_r.protein_binding.mpnn_trainer import ProteinMPNNConfig, ProteinMPNNTrainer

    pdb, chains = _repo_complex_pdb()
    records = [{"uid": "d0", "path": pdb, "sequence": "", "backbone_id": "t"}]
    cfg = ProteinMPNNConfig(
        mpnn_repo=REPO,
        initial_weights=os.path.join(REPO, "vanilla_model_weights", "v_48_020.pt"),
        max_epochs=1, device="cpu",
        chains_func=lambda rec, present: ([chains[0]], chains[1:]),
    )
    trainer = ProteinMPNNTrainer(cfg)

    first = trainer.train(records, str(tmp_path / "v1"), model_version=1)
    second = trainer.train(records, str(tmp_path / "v2"), model_version=2,
                           model_path=first)

    s1 = torch.load(first, map_location="cpu")["step"]
    s2 = torch.load(second, map_location="cpu")["step"]
    assert s2 > s1                       # the Noam step continued across rounds
