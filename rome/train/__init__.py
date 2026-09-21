"""Training algorithms available to ROME's training manager.

Adding one is a single task: subclass :class:`~rome.train.base.TrainTask`,
implement ``train(dataset, output_dir) -> checkpoint_path``, and hand the
instance to a :class:`~rome.trainer.TrainerConfig`. The LLM/GRPO trainer here is
imported lazily so a workflow that does not use it never pays for TRL.

The ProteinMPNN trainer is *not* here: it is an IMPRESS-R integration, not
framework core, so it ships with that example —
``examples/impress_r/mpnn.py`` (``from examples.impress_r.mpnn import
ProteinMPNNTrainer, ProteinMPNNConfig``).
"""

from rome.train.base import FunctionTrainer, TrainTask

__all__ = [
    "TrainTask",
    "FunctionTrainer",
    "GRPOTrainer",
    "GRPOConfig",
    "SFTTrainer",
    "SFTConfig",
    "ModelConfig",
]

_LAZY = {
    "GRPOTrainer": "rome.train.llm",
    "GRPOConfig": "rome.train.llm",
    "SFTTrainer": "rome.train.llm",
    "SFTConfig": "rome.train.llm",
    "ModelConfig": "rome.train.llm",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
