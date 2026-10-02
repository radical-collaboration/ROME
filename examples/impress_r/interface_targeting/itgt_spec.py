"""Reading and patching the per-pipeline RFD3 base design spec.

This module is the contract with IMPRESS's small-molecule pipeline, and it is
deliberately **pure standard library** -- no ``itgt``, no numpy, no torch. That
is what lets the whole patching surface be unit-tested on a laptop with nothing
installed, and it is where most of this workflow's correctness lives.

What the pipeline actually reads
--------------------------------

``SmallMoleculeBindingPipeline`` reads one file per pipeline::

    {input_dir}/ALR_binder_design.json

whose live content is a *single* design entry named ``partial``::

    {"partial": {"input":  "./input_pdbs/scaffold-with-ALR.pdb",
                 "ligand": "A:R",
                 "length": "70-170",
                 "select_exposed": {"A:R": "C13,...,S2"},
                 "select_buried":  {"A:R": "C18,...,S1"}}}

Those two selections are the RFD3 ``rasa_bin`` conditioning signal -- the thing
this workflow learns. Everything else in the entry is pipeline configuration we
must not touch.

Five constraints, each read out of the pipeline source rather than assumed:

* ``_write_guided_rfd3_json`` indexes the spec **by the entry name**
  (``base['partial']``), so the spec is *patched in place*; it is never
  regenerated. ``itgt.workflow.small_molecule_binding.write_campaign_inputs``
  is therefore not usable here -- it would rewrite ``input_pdbs/`` and collapse
  the entry name.
* Selections are keyed by the entry's own ``ligand`` value. On this target that
  string (``"A:R"``) happens to equal the ligand's PDB resname, so it also
  equals ``InterfaceTarget``'s component key -- but they are different concepts
  and :func:`patch_spec` keys off the template's value, not the target's.
* ``select_fixed_atoms`` is absent from the live spec and must stay absent. Its
  *keys* leak downstream into LigandMPNN's fixed-residue list, so synthesising
  it here would corrupt stages 2 and 3, not stage 1.
* ``_write_guided_rfd3_json`` fails safe (returns False, falling back to
  unguided diffusion) when an atom name in ``select_exposed``/``select_buried``
  is absent from the guided PDB -- but its check **omits**
  ``select_partially_buried``, which it nonetheless copies through. A naming
  slip in that third field turns a clean fallback into a mid-run RFD3
  ``ComponentValidationError``, so :func:`coverage_ok` checks all three.
* ``partial.input`` is resolved relative to the spec's own directory.
  :func:`reference_pdb` is a verbatim port of the pipeline's
  ``_resolve_reference_pdb_path`` so we featurize the same file RFD3 does.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

#: The spec filename ``small_molecule_binding.py`` hardcodes.
DEFAULT_SPEC_NAME = "ALR_binder_design.json"

#: The design entry the guided (partial-diffusion) path reads by name.
GUIDED_ENTRY = "partial"

#: Burial level -> RFD3 selection field. Mirrors
#: ``itgt.common.emit.rfd3.RASA_FIELD``; duplicated as a literal so this module
#: imports with nothing installed. Keep the two in step.
RASA_FIELD: Dict[str, str] = {
    "buried": "select_buried",
    "partial": "select_partially_buried",
    "exposed": "select_exposed",
}

#: Burial levels in canonical order (``itgt.common.selection.BURIAL_LEVELS``).
BURIAL_LEVELS: Tuple[str, ...] = ("buried", "partial", "exposed")

#: Per-pipeline bookkeeping directory, written beside the staged spec.
ITGT_DIR_NAME = ".itgt"

TEMPLATE_NAME = "template.json"
HISTORY_DIR_NAME = "history"

__all__ = [
    "DEFAULT_SPEC_NAME",
    "GUIDED_ENTRY",
    "RASA_FIELD",
    "BURIAL_LEVELS",
    "spec_path",
    "load_spec",
    "entry_name",
    "ligand_key",
    "reference_pdb",
    "template_partition",
    "patch_spec",
    "coverage_ok",
    "ligand_atom_names",
    "snapshot_template",
    "load_template",
    "write_spec_atomic",
]


# ── Locating and reading ──────────────────────────────────────────────────────


def spec_path(input_dir: str, spec_name: str = DEFAULT_SPEC_NAME) -> str:
    """Path of the base design spec inside a pipeline's input directory."""
    return os.path.join(input_dir, spec_name)


def load_spec(path: str) -> dict:
    """Parse a design spec file."""
    with open(path) as fh:
        return json.load(fh)


def entry_name(spec: Mapping) -> str:
    """The design entry the guided path will read.

    Asserts ``partial`` is present rather than guessing: the pipeline indexes
    ``base['partial']`` unconditionally, so a spec without it would take the
    unguided fallback for the whole campaign and silently ignore everything
    this workflow does.
    """
    if GUIDED_ENTRY not in spec:
        raise KeyError(
            f"design spec has no {GUIDED_ENTRY!r} entry (found {sorted(spec)}); "
            "the small-molecule pipeline reads base['partial'] by name"
        )
    return GUIDED_ENTRY


def ligand_key(spec: Mapping, entry: Optional[str] = None) -> str:
    """The entry's own ``ligand`` value -- the key every selection uses."""
    entry = entry or entry_name(spec)
    key = spec[entry].get("ligand")
    if not key:
        raise KeyError(f"design entry {entry!r} has no 'ligand' field")
    return key


def reference_pdb(path: str, spec: Optional[Mapping] = None) -> str:
    """Resolve ``partial.input`` the way the pipeline does.

    A verbatim port of ``small_molecule_binding._resolve_reference_pdb_path``:
    absolute paths pass through, relative ones resolve against the *spec's own
    directory*. Diverging here would featurize a different structure than RFD3
    is handed, and the atom names would stop matching.
    """
    spec = spec if spec is not None else load_spec(path)
    ref = spec[entry_name(spec)]["input"]
    if os.path.isabs(ref):
        return ref
    return os.path.join(os.path.dirname(path), ref)


def template_partition(
    spec: Mapping, entry: Optional[str] = None, ligand: Optional[str] = None
) -> Dict[str, List[str]]:
    """Invert a spec's selections back into ``{level: [atom names]}``.

    Used two ways: to seed the replay buffer with the hand-authored v0 action so
    the first reward observation has an action attached, and to size the first
    proposal to the template's own selection so round 1 is a like-for-like
    comparison rather than a change of two variables at once.
    """
    entry = entry or entry_name(spec)
    ligand = ligand or ligand_key(spec, entry)
    body = spec[entry]
    groups: Dict[str, List[str]] = {}
    for level in BURIAL_LEVELS:
        raw = body.get(RASA_FIELD[level], {}).get(ligand, "")
        names = [n for n in (part.strip() for part in raw.split(",")) if n]
        if names:
            groups[level] = names
    return groups


def ligand_atom_names(pdb_path: str, resname: str) -> List[str]:
    """Atom names of every HETATM record whose resname matches exactly.

    Fixed-column parsing, not a whitespace split: this target's resname is
    literally ``A:R`` and sits in columns 18-20, so splitting on whitespace
    mangles it. Mirrors the pipeline's ``_iter_ligand_atom_names`` for the same
    reason it exists there.
    """
    names: List[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HETATM") and line[17:20].strip() == resname:
                names.append(line[12:16].strip())
    return names


# ── Patching ──────────────────────────────────────────────────────────────────


def patch_spec(
    template: Mapping,
    groups: Mapping[str, Sequence[str]],
    entry: Optional[str] = None,
    ligand: Optional[str] = None,
) -> dict:
    """A copy of ``template`` with only its rASA selections replaced.

    ``groups`` is ``{burial level: [atom names]}`` as
    ``InterfaceTarget.burial_groups`` returns it -- already sorted, with empty
    levels omitted, which is required because RFD3 raises on a selection that
    matches no atoms.

    Everything else is preserved exactly: the entry name, ``input``, ``length``,
    ``ligand``, and any field this workflow does not own. A level absent from
    ``groups`` has its field *removed* rather than written empty, so a two-bin
    proposal produces a two-bin spec of the same shape as the hand-authored one.
    """
    entry = entry or entry_name(template)
    ligand = ligand or ligand_key(template, entry)

    unknown = set(groups) - set(RASA_FIELD)
    if unknown:
        raise ValueError(f"unknown burial level(s): {sorted(unknown)}")
    if not groups:
        raise ValueError("refusing to write a spec with no selections")

    spec = copy.deepcopy(dict(template))
    body = spec[entry]
    for level, field in RASA_FIELD.items():
        names = list(groups.get(level, ()))
        if names:
            body[field] = {ligand: ",".join(names)}
        else:
            body.pop(field, None)
    return spec


def coverage_ok(
    groups: Mapping[str, Sequence[str]], present: Iterable[str]
) -> Tuple[bool, Set[str]]:
    """Is every selected atom name actually present in the structure?

    Returns ``(ok, missing)``. Checked over **all three** selection fields,
    unlike the pipeline's own guard, which validates only
    ``select_exposed``/``select_buried`` while copying
    ``select_partially_buried`` through unchecked -- so a slip there escalates
    from a clean fallback to an RFD3 validation error mid-run.
    """
    present = set(present)
    wanted = {name for names in groups.values() for name in names}
    missing = wanted - present
    return (not missing), missing


# ── Staging and durable history ───────────────────────────────────────────────


def _itgt_dir(input_dir: str) -> str:
    return os.path.join(input_dir, ITGT_DIR_NAME)


def snapshot_template(input_dir: str, spec_name: str = DEFAULT_SPEC_NAME) -> str:
    """Copy the pristine spec aside, once, and return the snapshot's path.

    Every round patches from this snapshot rather than from the previous round's
    output, so proposals cannot compound: round *n* is a function of the model
    and the original template, nothing else. Idempotent -- an existing snapshot
    is left alone, which is what makes it safe to call on every startup.
    """
    target = os.path.join(_itgt_dir(input_dir), TEMPLATE_NAME)
    if os.path.exists(target):
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    spec = load_spec(spec_path(input_dir, spec_name))
    _write_json_atomic(spec, target)
    return target


def load_template(input_dir: str, spec_name: str = DEFAULT_SPEC_NAME) -> dict:
    """The pristine template: the snapshot if one exists, else the live spec."""
    snapshot = os.path.join(_itgt_dir(input_dir), TEMPLATE_NAME)
    if os.path.exists(snapshot):
        return load_spec(snapshot)
    return load_spec(spec_path(input_dir, spec_name))


def _write_json_atomic(payload: Mapping, path: str) -> str:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=4)
    os.replace(tmp, path)
    return path


def write_spec_atomic(
    spec: Mapping,
    input_dir: str,
    version: Optional[int] = None,
    spec_name: str = DEFAULT_SPEC_NAME,
) -> str:
    """Install a patched spec, keeping a per-version copy.

    ``os.replace`` on the same filesystem, so a concurrent reader sees the old
    or the new file and never a splice. Atomicity alone is not sufficient for
    consistency -- ``_prepare_guided_rfd3_inputs`` reads the base spec twice,
    once for ``partial.input`` and once for the selections -- which is why the
    caller applies this at a pass boundary rather than on checkpoint publish.

    The ``history/v{n}.json`` copy exists because ``scripts/validate_run.py``
    check 3 asserts every guided JSON matches *the current* base spec verbatim.
    Once the base moves, guided JSONs from earlier rounds fail that check; the
    history is what makes them auditable after the fact.
    """
    if version is not None:
        history = os.path.join(_itgt_dir(input_dir), HISTORY_DIR_NAME)
        os.makedirs(history, exist_ok=True)
        _write_json_atomic(spec, os.path.join(history, f"v{version}.json"))
    return _write_json_atomic(spec, spec_path(input_dir, spec_name))
