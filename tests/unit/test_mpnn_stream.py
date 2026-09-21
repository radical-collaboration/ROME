"""ProteinMPNN inference stream: everything reachable without the checkout.

The subprocess design paths need a real ``dauparas/ProteinMPNN`` checkout and a
GPU, so they are exercised on-cluster. What is testable here is the seam that a
wiring bug would hide: parsing ProteinMPNN's FASTA into design records, building
the two subprocess command lines, the reload hook's bookkeeping (and its repo
pointer refresh), and that the builders produce a well-formed INFERENCE stream.
"""

import os
from types import SimpleNamespace

from examples.impress_r.mpnn_stream import (
    MPNNStreamConfig,
    build_impress_command,
    build_run_command,
    mpnn_load,
    parse_mpnn_fasta,
)

# A protein_mpnn_run.py output FASTA: native record first, then two designs.
_FASTA = """\
>backbone, score=1.2000, global_score=1.2000, seq_recovery=1.0
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
>T=0.1, sample=1, score=0.8500, global_score=0.90, seq_recovery=0.31
MKTVYIAKQRQISFVKSHFSRQLEERLGLIEVA
>T=0.1, sample=2, score=0.7500, global_score=0.80, seq_recovery=0.29
MKTLYIAKQRQISFVKSHFSRQLEERLGLIEVC
"""


def test_parse_fasta_returns_designs_best_first_skipping_native(tmp_path):
    fa = tmp_path / "b1.fa"
    fa.write_text(_FASTA)
    designs = parse_mpnn_fasta(str(fa))
    assert len(designs) == 2                       # the native record is dropped
    assert [d["sample"] for d in designs] == [2, 1]   # sorted by score ascending
    assert designs[0]["score"] == 0.75
    assert designs[0]["sequence"].startswith("MKTL")


def test_parse_fasta_tolerates_a_native_only_file(tmp_path):
    fa = tmp_path / "n.fa"
    fa.write_text(">backbone, score=1.0\nMKTAYIAKQR\n")
    assert parse_mpnn_fasta(str(fa)) == []


def test_run_command_uses_the_exact_weights_and_designed_chains():
    settings = MPNNStreamConfig(mpnn_repo="/opt/ProteinMPNN", model_name="v_48_020")
    cmd = build_run_command(settings, "/in/b1.pdb", "/out",
                            weights_path="/ckpt/v_48_020_v3.pt",
                            design_chains="A", num_seqs=8)
    joined = " ".join(cmd)
    assert cmd[1].endswith("protein_mpnn_run.py")
    assert "--pdb_path=/in/b1.pdb" in cmd
    assert "--pdb_path_chains=A" in cmd
    assert "--path_to_model_weights=/ckpt/v_48_020_v3.pt" in cmd
    assert "--num_seq_per_target=8" in joined
    assert "--model_name" not in joined               # exact weights win over name


def test_run_command_falls_back_to_model_name_without_weights():
    settings = MPNNStreamConfig(mpnn_repo="/opt/ProteinMPNN", model_name="v_48_020")
    cmd = build_run_command(settings, "/in/b1.pdb", "/out", weights_path=None,
                            design_chains="A", num_seqs=4)
    assert "--model_name=v_48_020" in cmd
    assert not any(c.startswith("--path_to_model_weights") for c in cmd)


def test_impress_command_matches_the_wrapper_cli():
    settings = MPNNStreamConfig(mpnn_repo="/opt/ProteinMPNN",
                                wrapper_script="/x/mpnn_wrapper.py")
    cmd = build_impress_command(settings, "/in", "/out", design_chains="A",
                                num_seqs=8)
    assert cmd[1] == "/x/mpnn_wrapper.py"
    assert "-pdb=/in" in cmd and "-out=/out" in cmd
    assert "-mpnn=/opt/ProteinMPNN" in cmd
    assert "-seqs=8" in cmd and "-chains=A" in cmd and "-is_monomer=0" in cmd


def test_load_records_the_active_checkpoint_and_version():
    settings = MPNNStreamConfig(mpnn_repo="/opt/ProteinMPNN")
    ctx = SimpleNamespace(model_path="/ckpt/v2.pt", model_version=2)
    active = mpnn_load(None, ctx, settings=settings)   # falls back to ctx.model_path
    assert active["weights"] == "/ckpt/v2.pt"
    assert active["version"] == 2 and active["settings"] is settings


def test_load_refreshes_the_repo_pointer_for_the_impress_flavor(tmp_path):
    repo = tmp_path / "ProteinMPNN"
    (repo / "vanilla_model_weights").mkdir(parents=True)
    ckpt = tmp_path / "v_48_020_v5.pt"
    ckpt.write_text("WEIGHTS_V5")

    settings = MPNNStreamConfig(mpnn_repo=str(repo), model_name="v_48_020")
    ctx = SimpleNamespace(model_path=None, model_version=5)
    mpnn_load(str(ckpt), ctx, settings=settings, impress_pointer=True)

    pointer = repo / "vanilla_model_weights" / "v_48_020.pt"
    assert pointer.read_text() == "WEIGHTS_V5"      # exact version copied onto it


def test_builders_make_an_inference_stream_carrying_settings():
    from examples.impress_r.mpnn_stream import impress_mpnn_stream, mpnn_run_stream

    settings = MPNNStreamConfig(mpnn_repo="/opt/ProteinMPNN")

    native = mpnn_run_stream(settings, name="design", num_streams=2)
    assert native.name == "design" and native.num_streams == 2
    assert native.load_kwargs["settings"] is settings
    assert native.load_kwargs["impress_pointer"] is False

    impress = impress_mpnn_stream(settings)
    assert impress.load_kwargs["impress_pointer"] is True
    # Both are INFERENCE streams driven by the module-level load hook.
    assert native.load_func is impress.load_func
