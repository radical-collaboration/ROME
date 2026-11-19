"""Dragon's shared-state primitives, or in-process stand-ins when Dragon is absent.

ROME's state substrate is a Dragon ``DDict`` and its signals are Dragon ``Event``\\ s. Both
were imported at module scope in :mod:`rome.manager`, :mod:`rome.stream` and
:mod:`rome.trainer`, which made ``import rome`` itself require Dragon --- and Dragon is built
from source rather than installed from a wheel. The effect was that **nothing in ROME could
be imported on a machine without a Dragon build**, including the parts that do not use it:
the dummy trainer, the dummy stream, ``DataManager``, the corpus policies and every unit test
that drives them.

That was not the design, and this module is the evidence. :class:`rome.utils.Namespace` --
the only thing that touches the store -- is documented as *"A prefixed view over a Dragon
``DDict`` (**or any mapping**)"*, says of its argument *"Normally a
``dragon.data.ddict.DDict``; any mapping works, which is what makes the components
unit-testable"*, and deliberately routes reads through ``__getitem__`` and ``pop`` so that
*"[it] behaves the same on both"*. :class:`rome.manager.Manager` already accepts an injected
``ddict=`` and allocates its own only when none is given. The seam was there; the import
closed it.

So the rule this module establishes:

    **Importing ROME requires nothing. Running ROME across processes requires Dragon.**

What the in-process path gives you is a correct single-process loop: the manager, a trainer,
a stream and their handover over one plain ``dict``. What it does **not** give you, and must
not be mistaken for:

* **cross-process or cross-node sharing.** A ``dict`` lives in one interpreter. Task bodies
  that run as real processes under ``rhapsody``'s Dragon backend see nothing in it, so the
  in-process path is for ``LocalExecutionBackend`` only.
* **the DDict's atomicity.** Per-key writes in a DDict are atomic, which is why
  ``Namespace`` can be shared across the manager and its stream workers without a lock.
  Within one interpreter the GIL makes single ``dict`` operations atomic enough for the same
  usage, but a caller doing read-modify-write is on its own in both worlds.
* **the DDict's documented quirks**, which ``Namespace`` exists partly to paper over: a
  concurrent pop making ``keys()`` return a truncated list, and ``get(key, default)``
  hanging rather than raising. A ``dict`` has neither, so **a bug that only appears under
  Dragon will not appear here** -- which is exactly why the Dragon integration tests under
  ``tests/dragon/`` are not replaced by anything in this module.

Selection is by **whether a Dragon runtime is up** -- ``DRAGON_GS_CD`` in the environment --
not by whether the package imports and not by a flag. A flag would let a machine that has
Dragon silently run the in-process path, and the resulting test pass would be about the wrong
substrate. Availability alone would be worse: Dragon's API imports fine outside a ``dragon``
launch and then every object it creates asserts on launch parameters, so "importable" means a
crash deeper in. :func:`using_dragon` reports which substrate is live, so a caller -- or a
test -- can say so out loud rather than infer it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, MutableMapping

__all__ = [
    "DDict",
    "Event",
    "allocate_store",
    "dragon_importable",
    "require_dragon",
    "store_summary",
    "using_dragon",
]

def _runtime_active() -> bool:
    """Is a Dragon *runtime* up, as opposed to a Dragon package being importable?

    **"Is Dragon installed" is the wrong question**, and ROME's own test harness says so:
    ``tests/conftest.py:48`` forces its stubs whenever ``DRAGON_GS_CD`` is unset, with the
    reason stated at ``:122`` -- *"Force the stubs even when dragon is importable: outside a
    Dragon launch its objects cannot be constructed at all."* Every Dragon object asserts on
    launch parameters that exist only inside a ``dragon`` launch, so an importable-but-not-
    launched Dragon fails at construction with ``Launch parameter not initialized: GS_CD``,
    several frames from anything a reader would connect to the cause.

    So the test here is the same test the harness already uses, deliberately rather than
    coincidentally: one rule, in one place, instead of a second rule that disagrees with it
    on the machines where it matters.
    """
    return bool(os.environ.get("DRAGON_GS_CD"))


_DDICT_CLASS = None
_EVENT_CLASS = None

try:  # the CLASS being importable is a separate question from a runtime being up
    from dragon.data.ddict import DDict as _DDICT_CLASS  # type: ignore[no-redef]
    from dragon.native.event import Event as _EVENT_CLASS  # type: ignore[no-redef]
except ImportError:
    pass

if _runtime_active() and _DDICT_CLASS is None:
    # Under a Dragon launch with no Dragon package is a broken environment, not a laptop.
    # Say so rather than silently running single-process inside an allocation.
    raise RuntimeError(
        "DRAGON_GS_CD is set, so this process is under a Dragon launch, but dragon is "
        "not importable. The in-process substrate would run silently and "
        "single-process inside an allocation; refusing instead."
    )

#: Dragon's ``Event`` when its class is importable, else ``threading.Event``. The two are
#: interchangeable for ROME's usage -- ``set()``, ``clear()``, ``is_set()``, ``wait()``, at
#: ``manager.py``'s ``stop_event``, ``stream.py``'s ``reload_event``/``stop_event`` and
#: ``trainer.py``'s ``stop_event``. A Dragon Event additionally crosses process boundaries,
#: which is the whole difference and the reason the in-process path is single-process.
Event = _EVENT_CLASS if _EVENT_CLASS is not None else threading.Event

#: Kept bound so annotations like ``Optional[DDict]`` resolve either way. Without Dragon it
#: is ``MutableMapping``, which is the type ``Namespace`` actually requires.
DDict = _DDICT_CLASS if _DDICT_CLASS is not None else MutableMapping


def using_dragon() -> bool:
    """Is a Dragon runtime up in this interpreter?

    Worth reporting rather than inferring: a log line or a test assertion naming the
    substrate is the difference between "the loop works" and "the loop works here".
    """
    return _runtime_active()


def dragon_importable() -> bool:
    """Is Dragon's API importable, whether or not a runtime is up?

    A deliberately separate question from :func:`using_dragon`, and the distinction is the
    one this module got wrong first. ``tests/conftest.py`` installs stub ``DDict`` and
    ``Event`` classes so the unit suite can exercise the ownership and signalling paths
    *without* a runtime -- so a selection that keyed on the runtime alone would hand those
    tests a plain ``dict`` and silently delete their subject. Measured:
    ``test_stream_manager.py::test_close_releases_allocated_dictionaries`` began failing,
    because the stream manager's ``_owned`` entries no longer had anything to release.
    """
    return _DDICT_CLASS is not None


def allocate_store(**ddict_kwargs: Any) -> MutableMapping:
    """A new shared store: a Dragon ``DDict``, or a ``dict`` when Dragon is absent.

    ``ddict_kwargs`` -- ``n_nodes``, ``managers_per_node``, ``total_mem`` and friends -- are
    Dragon's allocation parameters and have no meaning for a ``dict``. They are **ignored**
    rather than rejected on the in-process path, because the caller that passes them is
    ROME's own default (:data:`rome.manager.DEFAULT_DDICT_KWARGS`) and failing on them would
    mean every caller needed to know which substrate it was on -- which is the coupling this
    module removes.
    """
    if _DDICT_CLASS is None:
        return {}
    try:
        return _DDICT_CLASS(**ddict_kwargs)
    except Exception:
        # Dragon's API imports fine outside a launch and then every object asserts on
        # launch parameters (`Launch parameter not initialized: GS_CD`). Outside a runtime
        # that is a laptop and the dict is the right answer; inside one it is fatal, and
        # the refusal above has already fired for the case where the package is missing.
        if _runtime_active():
            raise
        return {}


def require_dragon(what: str) -> None:
    """Raise a message that names the missing build, for a path that genuinely needs Dragon.

    For use where the in-process store cannot stand in -- a task body placed in another
    process, for instance. The failure a caller would otherwise get is an ``AttributeError``
    on a ``dict``, several frames from the cause.
    """
    if not _runtime_active():
        raise RuntimeError(
            f"{what} requires a Dragon runtime, and none is up here (DRAGON_GS_CD unset). "
            "ROME imports and runs single-process without one (see rome._substrate); "
            "cross-process placement and a shared DDict do not."
        )


def store_summary(store: MutableMapping) -> Dict[str, Any]:
    """What a log line should say about the store it is using. Used by the manager at start."""
    return {
        "substrate": "dragon-ddict" if not isinstance(store, dict) else "in-process-dict",
        "cross_process": _runtime_active(),
        "type": type(store).__name__,
    }
