"""Generate the API reference pages and their navigation.

One ``::: module.path`` stub per module, written into ``api/`` at build time and
never checked in, so the reference cannot drift from the source. mkdocstrings
reads the modules statically (griffe parses them rather than importing them),
which is what lets the reference build on a machine with no Dragon, no torch and
no GPU.

``EXTRA_PACKAGES`` is why this is not quite the stock script: ROME's
ProteinMPNN trainer ships with the IMPRESS-R *example* rather than with the
framework, because ROME is workflow agnostic — but it is public API for anyone
adopting IMPRESS-R, so it belongs in the reference too.
"""

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent

#: (import root, package directory) pairs to document. The import root is what
#: module paths are made relative to, so ``rome/data.py`` documents as
#: ``rome.data`` and ``examples/impress_r/mpnn.py`` as
#: ``examples.impress_r.mpnn``.
PACKAGES = [(ROOT, ROOT / "rome")]

#: Individual modules outside the framework package that are still public API.
EXTRA_MODULES = [(ROOT, ROOT / "examples" / "impress_r" / "protein_binding" / "mpnn_trainer.py")]

#: Private modules — documented nowhere, since they are not API.
SKIP_STEMS = {"_logging"}

nav = mkdocs_gen_files.Nav()


def document(src_root: Path, path: Path) -> None:
    """Write one ``api/<module path>.md`` stub for ``path``."""
    module_path = path.relative_to(src_root).with_suffix("")
    doc_path = path.relative_to(src_root).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        return

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(ROOT))


for src_root, package_dir in PACKAGES:
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts or path.stem in SKIP_STEMS:
            continue
        document(src_root, path)

for src_root, module in EXTRA_MODULES:
    document(src_root, module)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    # The hand-written landing page first, then the generated tree. Without this
    # line api/index.md is a page outside the nav, which `--strict` complains
    # about and which leaves the API tab with nothing to open onto.
    nav_file.write("- [Overview](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
