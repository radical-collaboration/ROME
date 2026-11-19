"""Importing ROME requires nothing; running it across processes requires Dragon.

The defect this pins: `manager.py`, `stream.py` and `trainer.py` imported
`dragon.data.ddict` and `dragon.native.event` at module scope, so `import rome` itself
required a Dragon build -- and Dragon is built from source, not installed from a wheel. That
put the dummy trainer, the dummy stream, `DataManager` and the corpus policies out of reach
on any machine without one, none of which touch Dragon.

**This repository was already paying for it.** `tests/conftest.py` force-stubs `dragon`,
`dragon.data` and `dragon.native.event` so the unit suite can import `rome` at all, and
`examples/agnostic/dummy_loop.py` -- the example whose whole point is "no model, the smallest
complete demonstration of the closed loop" -- has to be launched with `dragon` even for its
`LocalExecutionBackend` case. The seam was always there: `Namespace` is documented as taking
"any mapping", and `Manager` already accepts an injected `ddict=`. Only the import closed it.

**The cold-interpreter test is the one that matters.** The rest of this suite runs with
conftest's stubs in `sys.modules`, so a test that merely imports `rome` here would pass with
the defect fully present. Each test below therefore spawns a fresh interpreter with no
conftest.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cold(code: str, **env: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with no conftest and no inherited Dragon state."""
    clean = {k: v for k, v in os.environ.items() if k != "DRAGON_GS_CD"}
    clean["PYTHONPATH"] = str(REPO)
    clean.update(env)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=clean
    )


def test_rome_imports_in_a_cold_interpreter_without_dragon():
    """The headline. No stubs, no conftest, no Dragon build."""
    out = _cold("import rome; print('ok')")
    assert out.returncode == 0, f"import rome failed without Dragon:\n{out.stderr[-1500:]}"
    assert out.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "target",
    [
        "rome.manager",
        "rome.stream",
        "rome.trainer",
        "rome.data",
        "rome.dummy",
        "rome.utils",
    ],
)
def test_every_module_that_imported_dragon_now_imports_without_it(target):
    """Each of the three that imported Dragon at module scope, plus the three that never did
    and were taken down with them."""
    out = _cold(f"import {target}; print('ok')")
    assert out.returncode == 0, f"{target} failed:\n{out.stderr[-1200:]}"


def test_the_dummy_loop_pieces_are_reachable_without_dragon():
    """The concrete loss the defect caused, stated as what is now possible.

    `DummyTrainer`, `dummy_load` and `dummy_infer` are the substrate for any laptop test of
    the closed loop -- and for anything embedding ROME that wants to check its own wiring
    without an allocation.
    """
    out = _cold(
        "from rome.dummy import DummyTrainer, dummy_load, dummy_infer;"
        "from rome.manager import Manager;"
        "print(DummyTrainer.__name__, dummy_load.__name__, dummy_infer.__name__)"
    )
    assert out.returncode == 0, out.stderr[-1200:]
    assert out.stdout.split() == ["DummyTrainer", "dummy_load", "dummy_infer"]


def test_without_a_dragon_runtime_the_store_is_an_in_process_dict():
    """And it reports itself as such, rather than being inferred from the machine."""
    out = _cold(
        "from rome._substrate import using_dragon, allocate_store, store_summary;"
        "s = allocate_store(n_nodes=1, managers_per_node=1, total_mem=1);"
        "print(using_dragon(), store_summary(s)['substrate'], len(s))"
    )
    assert out.returncode == 0, out.stderr[-1200:]
    assert out.stdout.strip() == "False in-process-dict 0"


def test_dragons_allocation_kwargs_are_ignored_rather_than_rejected():
    """A caller must not have to know which substrate it is on.

    `Manager` passes `DEFAULT_DDICT_KWARGS` unconditionally; rejecting them off-Dragon would
    mean every caller branching on the substrate, which is the coupling this removes.
    """
    out = _cold(
        "from rome._substrate import allocate_store;"
        "print(type(allocate_store(n_nodes=4, nonsense='ignored')).__name__)"
    )
    assert out.returncode == 0, out.stderr[-1200:]
    assert out.stdout.strip() == "dict"


def test_a_namespace_over_the_in_process_store_round_trips():
    """`Namespace` is the only thing that touches the store, and it was always written for
    this: "any mapping works, which is what makes the components unit-testable"."""
    out = _cold(
        "from rome._substrate import allocate_store;"
        "from rome.utils import Namespace;"
        "ns = Namespace(allocate_store(), 'p|');"
        "ns['k'] = 7;"
        "print(ns['k'], ns.get('absent', 'default'), list(ns))"
    )
    assert out.returncode == 0, out.stderr[-1200:]
    assert out.stdout.strip() == "7 default ['k']"


def test_under_a_dragon_launch_without_dragon_installed_it_refuses():
    """The one case that must fail loudly rather than fall back.

    `DRAGON_GS_CD` set means this process is inside a `dragon` launch. Running the in-process
    substrate there would be a single-process loop inside an allocation, reporting success --
    the most expensive possible silent failure, and the reason selection is on the runtime
    rather than on whether the package imports.
    """
    out = _cold("import rome._substrate", DRAGON_GS_CD="/fake/gs/cd")
    assert out.returncode != 0, "a Dragon launch with no Dragon package was accepted"
    assert "under a Dragon launch" in out.stderr
    assert "refusing" in out.stderr
