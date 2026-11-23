"""Test fixtures and shared helpers.

Stubs out heavyweight third-party modules (transformers, trl, dragon, rose,
radical) when they aren't installed so unit and mocked-integration tests can
run on a minimal environment. Tests that need the real packages should call
``pytest.importorskip`` explicitly.

The Dragon stubs are *working* stand-ins rather than empty classes: ROME's
components are written against the DDict mapping protocol and Event's
set/clear/is_set, so a plain ``dict`` and a ``threading.Event`` exercise the
real code paths without a Dragon runtime.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
from functools import wraps

import pytest


def pytest_collection_modifyitems(config, items):
    """Tag every test that isn't explicitly `slow` as `fast`.

    Lets callers pick:
        pytest                 # run everything
        pytest -m fast         # unit + smoke + mocked-integration
        pytest -m slow         # the heavyweight end-to-end test
    """
    for item in items:
        if "slow" not in item.keywords:
            item.add_marker(pytest.mark.fast)


class _StubConfig:
    """Drop-in stand-in for trl.GRPOConfig / trl.SFTConfig.

    Stores whatever kwargs callers pass so tests can assert on them.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


#: The descriptors Dragon sets only inside a launch, in both spellings it has used:
#: ``*_CD`` through 0.14, ``*_QD`` from 0.15, which renamed every one of them. This is the
#: same rule as `rome._substrate._runtime_active`, spelled out a second time on purpose:
#: that one asks `dragon.infrastructure.facts` for the names, and this file must decide
#: whether to stub `dragon` out *before* anything imports it.
_LAUNCH_DESCRIPTORS = (
    "DRAGON_GS_CD", "DRAGON_GS_RET_CD", "DRAGON_LOCAL_SHEP_CD", "DRAGON_LOCAL_BE_CD",
    "DRAGON_GS_QD", "DRAGON_GS_RET_QD", "DRAGON_LOCAL_LS_QD", "DRAGON_LOCAL_BE_QD",
)


def _dragon_runtime_active() -> bool:
    """Whether we are running under ``dragon``.

    Dragon's Python API is importable without the runtime, but every object it
    creates asserts on launch parameters that only exist inside a Dragon
    launch. So "is dragon installed" is the wrong question — installing it and
    then running plain ``pytest`` would fail on a missing launch parameter.
    This asks whether the runtime is actually up.

    **Not by naming one descriptor.** This checked ``DRAGON_GS_CD`` until 2026-11-22, and
    Dragon 0.15 renamed it to ``DRAGON_GS_QD``; under a 0.15 launch the old check said no
    runtime was up, which in `rome._substrate` meant a `threading.Event` that cannot be
    pickled to a Dragon task (Delta job ``24400895``).
    """
    return any(os.environ.get(name) for name in _LAUNCH_DESCRIPTORS)


def _force_module(name: str) -> types.ModuleType:
    """Replace ``name`` in ``sys.modules`` with a fresh empty module."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _stub_if_missing() -> None:
    try:
        import torch
    except ImportError:
        m = _ensure_module("torch")

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        m.no_grad = _NoGrad

    try:
        import transformers  # noqa: F401
    except ImportError:
        m = _ensure_module("transformers")
        m.GenerationConfig = _StubConfig
        m.AutoTokenizer = type("AutoTokenizer", (), {})
        m.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})

    try:
        import trl
    except ImportError:
        m = _ensure_module("trl")
        m.GRPOConfig = _StubConfig
        m.SFTConfig = _StubConfig
        m.GRPOTrainer = type("GRPOTrainer", (), {})

    try:
        import peft
    except ImportError:
        m = _ensure_module("peft")
        m.get_peft_model = lambda model, cfg: model
        m.LoraConfig = _StubConfig
        m.PeftModel = type("PeftModel", (), {})

    try:
        import radical.asyncflow  # noqa: F401
    except ImportError:
        radical = _ensure_module("radical")
        af = _ensure_module("radical.asyncflow")
        radical.asyncflow = af
        af.WorkflowEngine = type("WorkflowEngine", (), {})

    if not _dragon_runtime_active():
        # Force the stubs even when dragon is importable: outside a Dragon
        # launch its objects cannot be constructed at all.
        _force_module("dragon")
        data = _force_module("dragon.data")
        ddict = _force_module("dragon.data.ddict")

        class _StubDDict(dict):
            """In-process stand-in for a Dragon DDict.

            A plain ``dict`` is faithful for the mapping operations ROME uses,
            but it is not constructor-compatible: ``dict(total_mem=...)`` would
            turn the sizing arguments into *entries*, which then show up in
            every prefix scan. So the kwargs are accepted and dropped, and
            ``destroy`` exists because the stream manager releases the
            dictionaries it allocates.
            """

            def __init__(self, *args, **kwargs):
                super().__init__()

            def destroy(self):
                self.clear()

        ddict.DDict = _StubDDict
        data.ddict = ddict
        native = _force_module("dragon.native")
        event = _force_module("dragon.native.event")
        event.Event = threading.Event
        native.event = event

    try:
        import rose  # noqa: F401
    except ImportError:
        _ensure_module("rose")
        metrics = _ensure_module("rose.metrics")
        metrics.GREATER_THAN_THRESHOLD = "greater_than_threshold"
        learner = _ensure_module("rose.learner")
        learner.SequentialReinforcementLearner = type(
            "SequentialReinforcementLearner",
            (),
            {"__init__": lambda self, asyncflow=None: None},
        )

    try:
        import datasets  # noqa: F401
    except ImportError:
        m = _ensure_module("datasets")

        class _Dataset(list):
            """Enough of ``datasets.Dataset`` for ROME's trainers."""

            @classmethod
            def from_list(cls, rows):
                return cls(rows)

            @property
            def column_names(self):
                return sorted({k for row in self for k in row})

        m.Dataset = _Dataset


_stub_if_missing()


class FakeWorkflowEngine:
    """Minimal stand-in for ``radical.asyncflow.WorkflowEngine``.

    Mirrors the real engine's contract for the one thing ROME uses:
    ``function_task`` is usable bare or called with ``service=...``, only
    accepts coroutine functions, pops a ``task_description`` kwarg at call
    time, and returns a future. Submissions are recorded so tests can assert
    on what ROME asked the backend for.
    """

    def __init__(self):
        self.submissions = []

    def function_task(self, possible_func=None, service=False, **_):
        def actual_decorator(func):
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Function {func.__name__} must be async.")

            @wraps(func)
            def wrapped(*args, **kwargs):
                description = kwargs.pop("task_description", None) or {}
                self.submissions.append(
                    {
                        "name": func.__name__,
                        "task_description": description,
                        "service": service,
                    }
                )
                return asyncio.ensure_future(func(*args, **kwargs))

            return wrapped

        if callable(possible_func):
            return actual_decorator(possible_func)
        return actual_decorator


@pytest.fixture
def asyncflow():
    """A fake workflow engine ROME can submit tasks to."""
    return FakeWorkflowEngine()


@pytest.fixture
def ddict():
    """A shared dictionary standing in for a Dragon DDict."""
    return {}


@pytest.fixture
def namespace(ddict):
    """A ROME namespace over the shared dictionary."""
    from rome.utils import Namespace

    return Namespace(ddict, "rome|")
