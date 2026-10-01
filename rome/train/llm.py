"""LLM trainer tasks: GRPO (reinforcement) and SFT (supervised).

The trainer shipped with the framework (the ProteinMPNN trainer lives with the
IMPRESS-R example, ``examples/impress_r/mpnn.py``). It is the demonstration that
"adding new models and training algorithms requires just one task": everything
below this line is TRL-specific, and nothing above it — the data manager, the
training manager, the stream manager — knows that an LLM is involved.

The task takes the dataset the data manager built, runs TRL's GRPO over it, and
writes a checkpoint (a LoRA adapter when one is configured, otherwise the full
model). The training manager publishes that path and the inference streams
hot-swap onto it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from rome.train.base import TrainTask

if TYPE_CHECKING:  # heavyweight imports stay out of the manager process
    from peft import LoraConfig
    from transformers import GenerationConfig


@dataclass
class ModelConfig:
    """Model configurations used for loading the model and tokenizer, as well as generation.

    Parameters
    ----------
    base_model_name: Optional[str]
        Foundation model name, used to load tokenizer and model weights (when
        model_name is not provided). Should be a model name recognized by
        HuggingFace's AutoModelForCausalLM and AutoTokenizer classes.
    model_name: Optional[str]
        Full model name, used to load model weights. If no model_name is
        provided, base_model_name is used.
    lora_name: Optional[str]
        LoRA adapter name, used to load LoRA adapter weights. If no lora_name is
        provided, the full model is trained and saved.
    lora_config: Optional[LoraConfig]
        Adapter shape, used when ``lora_name`` names a directory that does not
        exist yet. ROME's defaults are used when omitted.
    generation_config: Optional[GenerationConfig]
        Generation parameters, used by inference streams sharing this config.
    dtype, device_map, required_gpus, max_seq_length
        Loading and placement knobs forwarded to transformers.
    """

    base_model_name: Optional[str] = None
    model_name: Optional[str] = None
    lora_name: Optional[str] = None
    lora_config: Optional["LoraConfig"] = None
    generation_config: Optional["GenerationConfig"] = None
    dtype: Optional[str] = "auto"
    required_gpus: int = 1
    device_map: Any = "auto"
    max_seq_length: int = 2048

    def resolved_model_name(self) -> str:
        """The weights to load, preferring a fine-tuned checkpoint over the base."""
        name = self.model_name or self.base_model_name
        if name is None:
            raise ValueError(
                "ModelConfig needs either base_model_name or model_name"
            )
        return name

    def tokenizer_name(self) -> str:
        """Tokenizers come from the foundation model whenever one is named."""
        name = self.base_model_name or self.model_name
        if name is None:
            raise ValueError(
                "ModelConfig needs either base_model_name or model_name"
            )
        return name


@dataclass
class GRPOConfig:
    """Configuration class for managing GRPO training settings in the ROME framework.

    Holds only the knobs ROME itself cares about. Anything else TRL accepts
    goes in ``trl_config`` (a ``trl.GRPOConfig``) or in ``extra_args``, which is
    merged into the TRL config that gets built.

    Parameters
    ----------
    model_config : ModelConfig
        What to train. Required.
    learning_rate, batch_size, num_epochs, gradient_accumulation_steps
        The usual. Defaults follow the values ROME used for its LLM runs.
    num_generations : int
        Completions GRPO samples per prompt.
    reward_funcs : List[Callable]
        Reward functions TRL calls inline during training, with the signature
        ``fn(prompts, completions, **kwargs) -> list[float]``. Rewards that are
        expensive or need their own resources belong in a ROME *reward
        stream* instead — see :mod:`rome.stream`.
    prompt_column : str
        Field of a corpus record holding the prompt TRL should train on.
    trl_config : Optional[trl.GRPOConfig]
        Fully-specified TRL config. Overrides the individual knobs above.
    extra_args : dict
        Additional keyword arguments merged into the constructed TRL config.
    """

    model_config: Optional[ModelConfig] = None
    learning_rate: float = 5e-6
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 16
    num_generations: int = 4
    reward_funcs: List[Callable] = field(default_factory=list)
    prompt_column: str = "prompt"
    rollout_func: Optional[Callable] = None
    trl_config: Any = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def build_trl_config(self, output_dir: str) -> Any:
        """Materialize a ``trl.GRPOConfig``.

        Imported here rather than at module scope so the manager process — which
        only ever constructs configs — never has to pay for TRL.
        """
        if self.trl_config is not None:
            return self.trl_config
        from trl import GRPOConfig as TRLGRPOConfig

        args = dict(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.01,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            logging_steps=1,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_generations=self.num_generations,
            # Generation batches must not exceed the training batch, or TRL
            # ends up padding partial groups.
            generation_batch_size=self.batch_size,
        )
        args.update(self.extra_args)
        return TRLGRPOConfig(**args)


class GRPOTrainer(TrainTask):
    """GRPO trainer task in the ROME framework.

    Parameters
    ----------
    config : GRPOConfig
        Training settings, including the :class:`ModelConfig` to train.
    gpus, nodes : int
        Resources one training round needs. Forwarded to asyncflow by the
        training manager. Defaults to ``model_config.required_gpus``.
    trainer_callbacks : Optional[List[Any]]
        ``transformers`` callbacks passed through to TRL — e.g. a callback that
        checkpoints mid-round so streams can reload before the round ends.
    """

    def __init__(
        self,
        config: GRPOConfig,
        *,
        gpus: Optional[int] = None,
        nodes: int = 1,
        trainer_callbacks: Optional[List[Any]] = None,
        name: Optional[str] = None,
    ):
        if config.model_config is None:
            raise ValueError("GRPOConfig.model_config must be set")
        super().__init__(
            gpus=gpus if gpus is not None else config.model_config.required_gpus,
            nodes=nodes,
            name=name or "grpo",
        )
        self.config = config
        self.trainer_callbacks = trainer_callbacks

    wants_hf_dataset = True
    """TRL wants a ``datasets.Dataset``, so the data manager builds one."""

    def validate(self, dataset: Any) -> None:
        super().validate(dataset)
        column = self.config.prompt_column
        columns = getattr(dataset, "column_names", None)
        if columns is not None and column not in columns:
            raise ValueError(
                f"GRPO needs a {column!r} column; the corpus has {sorted(columns)}. "
                "Add prompts via add_training_data(prompt=...) or set "
                "GRPOConfig.prompt_column."
            )

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Run one GRPO round and return the checkpoint path.

        Runs inside the asyncflow task, so this is the first point where torch,
        transformers, TRL and peft are actually imported — on the node that has
        the GPUs.
        """
        from trl import GRPOTrainer as TRLGRPOTrainer

        model_config = self.config.model_config
        model, tokenizer = load_model(model_config)

        trainer = TRLGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=list(self.config.reward_funcs),
            args=self.config.build_trl_config(output_dir),
            train_dataset=dataset,
            callbacks=self.trainer_callbacks,
            **({"rollout_func": self.config.rollout_func}
               if self.config.rollout_func is not None else {}),
        )
        trainer.train()
        return save_model(model, model_config, output_dir)


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT) — "train on the good ones"
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning in the ROME framework.

    SFT trains the model to imitate chosen ``(prompt, completion)`` pairs. In the
    self-improvement loop it is the "train on the good ones" half: an inference
    stream generates candidates, a reward stream scores them, the data manager
    keeps the best/correct ones, and this fine-tunes on those — rejection-sampling
    / STaR-style, with no preference or RL objective. The reward that *selects*
    the data lives in the data manager (a filter/sampler) or a reward stream, not
    here; SFT only sees the winners.

    Parameters
    ----------
    model_config : ModelConfig
        What to train. Required.
    learning_rate, batch_size, num_epochs, gradient_accumulation_steps
        The usual SFT knobs.
    prompt_column, completion_column : str
        Corpus fields holding the instruction and the response to imitate.
    text_column : str
        Field holding a fully-formatted training string. When a record lacks it,
        the trainer builds it from prompt+completion via the tokenizer's chat
        template; when present, it is used as-is.
    trl_config : Optional[trl.SFTConfig]
        Fully-specified TRL config. Overrides the individual knobs above.
    extra_args : dict
        Additional keyword arguments merged into the constructed TRL config —
        e.g. ``max_seq_length`` / ``max_length``, whose name varies by TRL
        version, so it is left out of the defaults on purpose.
    """

    model_config: Optional[ModelConfig] = None
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 8
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    text_column: str = "text"
    trl_config: Any = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def build_trl_config(self, output_dir: str) -> Any:
        """Materialize a ``trl.SFTConfig`` (imported here so the manager never
        pays for TRL)."""
        if self.trl_config is not None:
            return self.trl_config
        from trl import SFTConfig as TRLSFTConfig

        args = dict(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            logging_steps=1,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        args.update(self.extra_args)
        return TRLSFTConfig(**args)


def _format_chat(tokenizer: Any, prompt: str, completion: str) -> str:
    """Format one prompt+completion as the model's own chat string.

    Uses the tokenizer's chat template so an instruct model trains on the exact
    format it is prompted in; falls back to a plain join for a base tokenizer
    with no template.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None:
        try:
            return apply(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completion}],
                tokenize=False,
            )
        except Exception:  # noqa: BLE001 - a template that rejects the turns
            pass
    return f"{prompt}\n{completion}"


class SFTTrainer(TrainTask):
    """SFT trainer task in the ROME framework — the counterpart to GRPO.

    Parameters mirror :class:`GRPOTrainer`: a :class:`SFTConfig` (with the
    :class:`ModelConfig` to train), resources forwarded to asyncflow, and
    optional ``transformers`` callbacks.
    """

    def __init__(
        self,
        config: SFTConfig,
        *,
        gpus: Optional[int] = None,
        nodes: int = 1,
        trainer_callbacks: Optional[List[Any]] = None,
        name: Optional[str] = None,
    ):
        if config.model_config is None:
            raise ValueError("SFTConfig.model_config must be set")
        super().__init__(
            gpus=gpus if gpus is not None else config.model_config.required_gpus,
            nodes=nodes,
            name=name or "sft",
        )
        self.config = config
        self.trainer_callbacks = trainer_callbacks

    #: TRL wants a ``datasets.Dataset``, so the data manager builds one.
    wants_hf_dataset = True

    def validate(self, dataset: Any) -> None:
        super().validate(dataset)
        cfg = self.config
        columns = getattr(dataset, "column_names", None)
        if columns is None:
            return
        has_text = cfg.text_column in columns
        has_pair = (cfg.prompt_column in columns
                    and cfg.completion_column in columns)
        if not (has_text or has_pair):
            raise ValueError(
                f"SFT needs either a {cfg.text_column!r} column or both "
                f"{cfg.prompt_column!r} and {cfg.completion_column!r}; the corpus "
                f"has {sorted(columns)}. Store the chosen response via "
                "add_training_data(prompt=..., completion=...)."
            )

    def train(self, dataset: Any, output_dir: str, **kwargs: Any) -> str:
        """Run one SFT round on the chosen pairs and return the checkpoint path.

        Like GRPO, torch/transformers/TRL/peft are imported here — on the node
        that has the GPUs, never in the manager process.
        """
        from trl import SFTTrainer as TRLSFTTrainer

        model_config = self.config.model_config
        model, tokenizer = load_model(model_config)
        dataset = self._ensure_text(dataset, tokenizer)

        trainer = TRLSFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=self.config.build_trl_config(output_dir),
            train_dataset=dataset,
            callbacks=self.trainer_callbacks,
        )
        trainer.train()
        return save_model(model, model_config, output_dir)

    def _ensure_text(self, dataset: Any, tokenizer: Any) -> Any:
        """Add ``text_column`` from prompt+completion when it is absent."""
        cfg = self.config
        columns = getattr(dataset, "column_names", None) or []
        if cfg.text_column in columns:
            return dataset

        def to_text(record: Dict[str, Any]) -> Dict[str, Any]:
            return {cfg.text_column: _format_chat(
                tokenizer, record.get(cfg.prompt_column, ""),
                record.get(cfg.completion_column, ""))}

        return dataset.map(to_text)


# ---------------------------------------------------------------------------
# Weight loading / saving
# ---------------------------------------------------------------------------

def load_model(model_config: ModelConfig):
    """Load model and tokenizer according to the provided model configuration.

    Also usable from an inference stream's ``load_func``, which is the point:
    the stream and the trainer agree on what a checkpoint means because they
    load it the same way.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name(), padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.resolved_model_name(),
        torch_dtype=model_config.dtype,
        device_map=model_config.device_map,
    )

    if model_config.lora_name is not None:
        from peft import LoraConfig, PeftModel, get_peft_model

        if os.path.exists(model_config.lora_name):
            model = PeftModel.from_pretrained(
                model, model_config.lora_name, is_trainable=True
            )
        else:
            lora_config = model_config.lora_config or LoraConfig(
                r=128,
                lora_alpha=128,
                lora_dropout=0.0,
                inference_mode=False,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    return model, tokenizer


def save_model(model, model_config: ModelConfig, output_dir: str) -> str:
    """Persist weights for cross-task reload; returns the directory written to.

    When a LoRA adapter is configured only the adapter is saved — that is the
    whole reason ROME can hot-swap weights mid-campaign without stalling
    inference on a multi-gigabyte read.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if model_config.lora_name is not None:
        # Keep the configured adapter location current too, so a stream that
        # was pointed at lora_name rather than at the published checkpoint
        # still sees the new weights.
        model.save_pretrained(model_config.lora_name)
    return output_dir


__all__ = ["ModelConfig", "GRPOConfig", "GRPOTrainer", "SFTConfig",
           "SFTTrainer", "load_model", "save_model"]
