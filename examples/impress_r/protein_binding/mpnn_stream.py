"""ProteinMPNN inference as a ROME stream — generate sequences, hot-swap weights.

The training half of IMPRESS-R (``mpnn.py``) fine-tunes ProteinMPNN; this is the
*generation* half as a persistent :class:`rome.StreamConfig`. An inference stream
runs sequence design continuously and reloads onto each checkpoint the trainer
publishes — the same loop the LLM examples use, for proteins. Feed it backbone
PDBs, get designed sequences back; when a new checkpoint lands the next batch is
designed with the improved model, no orchestration.

Two implementations, both **subprocess** — neither reimplements ProteinMPNN's
sampling (that is the detail-heavy part the trainer was careful to reuse rather
than rewrite), so both run the checkout's own tested inference:

* :func:`mpnn_run_stream` — the ROME-native path: calls the checkout's
  ``protein_mpnn_run.py`` directly on each PDB, passing ``--path_to_model_weights``
  so it uses the *exact* published checkpoint (clean per-version hot-swap).
* :func:`impress_mpnn_stream` — the IMPRESS path: calls this example's
  ``mpnn_wrapper.py`` (chain parsing/assignment + ``protein_mpnn_run.py``), i.e.
  exactly what the IMPRESS pipeline runs. It reads weights from the repo's fixed
  ``{model_name}.pt`` pointer, which ``ProteinMPNNConfig(publish_into_repo=True)``
  keeps current, so a fine-tune reaches it with no wrapper change.

A request is a dict ``{"backbone_id", "pdb_path", "design_chains"?, "num_seqs"?}``;
each output record is ``{"backbone_id", "sequence", "score", "sample",
"model_version", "pdb_path"}`` — ready to hand to a reward stream (AlphaFold) and
then :meth:`rome.Manager.add_training_data`, closing the loop back to the trainer.

    backbone ──▶ MPNN inference stream ──seqs──▶ (AF reward) ──▶ Data Manager
                        ▲                                            │
                        └──────── checkpoint ◀── ProteinMPNN Trainer ◀┘

Usage::

    import rome
    from examples.impress_r.mpnn_stream import MPNNStreamConfig, impress_mpnn_stream

    settings = MPNNStreamConfig(mpnn_repo="/work/ProteinMPNN", model_name="v_48_020")
    manager = rome.Manager(flow, stream_configs=[impress_mpnn_stream(settings)])
    ...
    manager.stream.submit({"backbone_id": "b1", "pdb_path": ".../b1.pdb"},
                          stream="mpnn")
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MPNNStreamConfig:
    """Settings for a ProteinMPNN inference stream.

    Kept separate from :class:`~examples.impress_r.mpnn.ProteinMPNNConfig` (the
    *training* config) because a stream only needs the checkout, the model name,
    and how to sample — no optimiser or chain-loss knobs. Picklable, so it rides
    to the stream's task process via ``StreamConfig.load_kwargs``.

    Parameters
    ----------
    mpnn_repo : str
        The ``dauparas/ProteinMPNN`` checkout — the same directory the trainer
        and IMPRESS use.
    model_name : str
        Weights basename (no extension); must match the trainer's, so the stream
        loads what the trainer publishes. Default ``v_48_020``.
    num_seqs : int
        Sequences designed per backbone per request.
    design_chains : str
        Space-separated chains to design (multimer). ``"A"`` for the IMPRESS PDZ
        binder; the peptide chain is left fixed as context.
    is_monomer : bool
        Passed through to the IMPRESS wrapper (``-is_monomer``).
    sampling_temp : float
        ProteinMPNN sampling temperature.
    seed, batch_size : int
        Reproducibility / batching for ``protein_mpnn_run.py``.
    wrapper_script : Optional[str]
        The IMPRESS ``mpnn_wrapper.py`` to run (impress flavor). Defaults to the
        one beside this file.
    python : str
        Interpreter used to launch the subprocess. Defaults to ``sys.executable``.
    """

    mpnn_repo: Optional[str] = None
    model_name: str = "v_48_020"
    num_seqs: int = 8
    design_chains: str = "A"
    is_monomer: bool = False
    sampling_temp: float = 0.1
    seed: int = 37
    batch_size: int = 1
    wrapper_script: Optional[str] = None
    python: Optional[str] = None

    def interpreter(self) -> str:
        return self.python or sys.executable

    def wrapper(self) -> str:
        return self.wrapper_script or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mpnn_wrapper.py")


# ---------------------------------------------------------------------------
# Pure helpers (no torch, no rome) — the testable core
# ---------------------------------------------------------------------------

def parse_mpnn_fasta(fa_path: str) -> List[Dict[str, Any]]:
    """Parse one ``protein_mpnn_run.py`` output FASTA into designed sequences.

    The file's first record is the *native* sequence (a header line then the
    sequence); every record after it is a sampled design whose header carries
    ``sample=..`` and ``score=..`` (global score, lower is better). This mirrors
    the parse IMPRESS's own post-MPNN step does, returning only the designs.

    Returns ``[{"sequence", "score", "sample"}]``, best (lowest) score first.
    """
    with open(fa_path) as fd:
        lines = [ln.strip() for ln in fd if ln.strip()]

    designs: List[Dict[str, Any]] = []
    # Records are (header, sequence) pairs; skip the first (native) record.
    header = None
    for line in lines:
        if line.startswith(">"):
            header = line
            continue
        if header is None:
            continue
        fields = _parse_header(header)
        if "sample" in fields:            # a sampled design, not the native seq
            designs.append({
                "sequence": line,
                "score": fields.get("score"),
                "sample": fields.get("sample"),
            })
        header = None

    designs.sort(key=lambda d: (d["score"] is None, d["score"] if d["score"] is not None else 0.0))
    return designs


def _parse_header(header: str) -> Dict[str, Any]:
    """``>T=0.1, sample=1, score=1.234, global_score=1.3`` -> a dict of its fields.

    ``score`` (the per-sequence score IMPRESS ranks on) and ``global_score`` are
    kept distinct — collapsing them lets the global value clobber the real one.
    """
    out: Dict[str, Any] = {}
    for part in header.lstrip(">").split(","):
        if "=" not in part:
            continue
        key, _, value = part.strip().partition("=")
        key = key.strip()
        value = value.strip()
        if key == "sample":
            try:
                out["sample"] = int(value)
            except ValueError:
                out["sample"] = value
        else:
            try:
                out[key] = float(value)     # score, global_score, T, seq_recovery
            except ValueError:
                pass
    return out


def build_run_command(settings: MPNNStreamConfig, pdb_path: str, out_dir: str,
                      weights_path: Optional[str], design_chains: str,
                      num_seqs: int) -> List[str]:
    """The ROME-native ``protein_mpnn_run.py`` invocation for one backbone.

    Uses ``--pdb_path`` (single structure) and ``--pdb_path_chains`` to pick the
    designed chains, and ``--path_to_model_weights`` so the exact published
    checkpoint is used rather than whatever the repo pointer happens to hold.
    """
    cmd = [
        settings.interpreter(),
        os.path.join(settings.mpnn_repo, "protein_mpnn_run.py"),
        f"--pdb_path={pdb_path}",
        f"--out_folder={out_dir}",
        f"--num_seq_per_target={num_seqs}",
        f"--sampling_temp={settings.sampling_temp}",
        f"--seed={settings.seed}",
        f"--batch_size={settings.batch_size}",
    ]
    if design_chains.strip():
        cmd.append(f"--pdb_path_chains={design_chains}")
    if weights_path:
        cmd.append(f"--path_to_model_weights={weights_path}")
    else:
        cmd.append(f"--model_name={settings.model_name}")
    return cmd


def build_impress_command(settings: MPNNStreamConfig, input_dir: str,
                          out_dir: str, design_chains: str,
                          num_seqs: int) -> List[str]:
    """The IMPRESS ``mpnn_wrapper.py`` invocation for a directory of backbones —
    the exact shape the IMPRESS protein-binding pipeline uses (see its ``s1``)."""
    return [
        settings.interpreter(), settings.wrapper(),
        f"-pdb={input_dir}",
        f"-out={out_dir}",
        f"-mpnn={settings.mpnn_repo}",
        f"-seqs={num_seqs}",
        f"-is_monomer={1 if settings.is_monomer else 0}",
        f"-chains={design_chains}",
    ]


def _collect_designs(seqs_dir: str, backbone_id: str, pdb_path: str,
                     model_version: int) -> List[Dict[str, Any]]:
    """Turn every ``<seqs_dir>/*.fa`` into corpus-shaped design records."""
    records: List[Dict[str, Any]] = []
    for fa in sorted(glob.glob(os.path.join(seqs_dir, "*.fa"))):
        for design in parse_mpnn_fasta(fa):
            records.append({
                "backbone_id": backbone_id,
                "pdb_path": pdb_path,
                "sequence": design["sequence"],
                "score": design["score"],
                "sample": design["sample"],
                "model_version": model_version,
            })
    return records


# ---------------------------------------------------------------------------
# Stream funcs — module-level so they pickle by reference to the task process
# ---------------------------------------------------------------------------

def mpnn_load(checkpoint_path: Optional[str], ctx: Any, *,
              settings: MPNNStreamConfig,
              impress_pointer: bool = False) -> Dict[str, Any]:
    """Reload hook for both flavors — module-level so it pickles by reference.

    ProteinMPNN inference reads its weights from disk on every invocation, so
    there is no in-memory model to swap — "loading" just records which checkpoint
    the next batch should use. ``mpnn_run_*`` passes that path straight to
    ``protein_mpnn_run.py``; ``impress_mpnn_*`` cannot (its wrapper does not
    forward the path), so with ``impress_pointer=True`` it also copies the
    checkpoint onto the repo's fixed pointer, which is what that wrapper loads.
    """
    weights = checkpoint_path or getattr(ctx, "model_path", None)
    active = {"settings": settings, "weights": weights,
              "version": int(getattr(ctx, "model_version", 0) or 0)}
    if impress_pointer and weights and settings.mpnn_repo:
        pointer = os.path.join(settings.mpnn_repo, "vanilla_model_weights",
                               f"{settings.model_name}.pt")
        try:
            os.makedirs(os.path.dirname(pointer), exist_ok=True)
            tmp = pointer + ".tmp"
            shutil.copyfile(weights, tmp)
            os.replace(tmp, pointer)
        except OSError:
            pass                          # fall back to whatever the repo holds
    return active


def _design(inputs: List[Dict[str, Any]], ctx: Any, *, use_wrapper: bool
            ) -> List[List[Dict[str, Any]]]:
    """Shared body: run one flavor over a batch of backbone requests.

    Returns one list of design records per input request (the stream emits each
    request's results together).
    """
    active = ctx.model or {}
    settings: MPNNStreamConfig = active["settings"]
    weights = active.get("weights")
    version = active.get("version", 0)
    out: List[List[Dict[str, Any]]] = []

    for req in inputs:
        backbone_id = req.get("backbone_id") or os.path.splitext(
            os.path.basename(req.get("pdb_path", "design")))[0]
        pdb_path = req["pdb_path"]
        chains = req.get("design_chains", settings.design_chains)
        num_seqs = int(req.get("num_seqs", settings.num_seqs))
        work = tempfile.mkdtemp(prefix=f"mpnn_stream_{backbone_id}_")
        try:
            if use_wrapper:
                # The wrapper takes a *directory* of PDBs; stage this one in.
                in_dir = os.path.join(work, "in")
                os.makedirs(in_dir, exist_ok=True)
                shutil.copyfile(pdb_path, os.path.join(in_dir, f"{backbone_id}.pdb"))
                cmd = build_impress_command(settings, in_dir, work, chains, num_seqs)
            else:
                cmd = build_run_command(settings, pdb_path, work, weights,
                                        chains, num_seqs)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                out.append([{"backbone_id": backbone_id, "pdb_path": pdb_path,
                             "error": (result.stderr or "").strip()[-500:]}])
                continue
            out.append(_collect_designs(os.path.join(work, "seqs"),
                                        backbone_id, pdb_path, version))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return out


def mpnn_run_design(inputs: List[Dict[str, Any]], ctx: Any) -> List[List[Dict[str, Any]]]:
    """process_func: ROME-native, ``protein_mpnn_run.py`` per backbone."""
    return _design(inputs, ctx, use_wrapper=False)


def impress_mpnn_design(inputs: List[Dict[str, Any]], ctx: Any) -> List[List[Dict[str, Any]]]:
    """process_func: IMPRESS path, this example's ``mpnn_wrapper.py``."""
    return _design(inputs, ctx, use_wrapper=True)


# ---------------------------------------------------------------------------
# StreamConfig builders (rome imported lazily so the helpers above stay dragon-free)
# ---------------------------------------------------------------------------

def _stream_config(settings: MPNNStreamConfig, process_func, *, name: str,
                   num_streams: int, num_gpus: int, batch_size: int,
                   impress_pointer: bool, extra: Dict[str, Any]) -> Any:
    import rome

    # load_func + process_func are module-level (pickle by reference to the task
    # process); settings ride along as load_kwargs, which the stream forwards to
    # load_func. Anything the caller passes in `extra` (e.g. num_nodes) wins.
    load_kwargs = {"settings": settings, "impress_pointer": impress_pointer}
    load_kwargs.update(extra.pop("load_kwargs", {}))
    return rome.StreamConfig(
        name=name,
        kind=rome.StreamKind.INFERENCE,
        load_func=mpnn_load,
        process_func=process_func,
        load_kwargs=load_kwargs,
        num_streams=num_streams,
        num_gpus=num_gpus,
        batch_size=batch_size,
        **extra,
    )


def mpnn_run_stream(settings: MPNNStreamConfig, *, name: str = "mpnn",
                    num_streams: int = 1, num_gpus: int = 1, batch_size: int = 1,
                    **extra: Any) -> Any:
    """A ROME inference stream that designs sequences with ``protein_mpnn_run.py``,
    using the exact published checkpoint via ``--path_to_model_weights``."""
    return _stream_config(settings, mpnn_run_design, name=name,
                          num_streams=num_streams, num_gpus=num_gpus,
                          batch_size=batch_size, impress_pointer=False, extra=extra)


def impress_mpnn_stream(settings: MPNNStreamConfig, *, name: str = "mpnn",
                        num_streams: int = 1, num_gpus: int = 1, batch_size: int = 1,
                        **extra: Any) -> Any:
    """A ROME inference stream that designs sequences with IMPRESS's
    ``mpnn_wrapper.py`` — the exact inference path the campaign runs. On each
    reload it refreshes the repo's fixed weights pointer to the new checkpoint."""
    return _stream_config(settings, impress_mpnn_design, name=name,
                          num_streams=num_streams, num_gpus=num_gpus,
                          batch_size=batch_size, impress_pointer=True, extra=extra)


__all__ = [
    "MPNNStreamConfig",
    "parse_mpnn_fasta",
    "build_run_command",
    "build_impress_command",
    "mpnn_load",
    "mpnn_run_design",
    "impress_mpnn_design",
    "mpnn_run_stream",
    "impress_mpnn_stream",
]
