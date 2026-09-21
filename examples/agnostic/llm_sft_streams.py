"""LLM self-improvement by SFT on its own best answers (rejection sampling / STaR).

The supervised counterpart to ``llm_grpo_streams.py``. Same three-manager loop —
an inference stream generates candidate answers, a reward stream scores them, the
data manager collects them — but the training half is **SFT, not GRPO**: instead
of an RL objective over rewards, the model is fine-tuned to *imitate its own
correct answers*. The reward is used only to **select** which generations become
training data ("train on the best-scoring and/or correct responses").

    inference stream ──generations──▶ reward stream ──scores──▶ Data Manager
             ▲                          (math reward)          (keep the correct)
             │                                                       │
             └──────────── adapter ◀── SFT Trainer ◀─────────────────┘
                          (imitate the winners)

This is the STaR loop: sample, keep what's correct, fine-tune on it, repeat — the
model bootstraps from problems it can already solve. The reward calculation is
the *same* math grader the GRPO example would use; here it gates the corpus
rather than feeding an RL loss.

Run under the Dragon runtime::

    dragon examples/agnostic/llm_sft_streams.py
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import rome
from rome.train.llm import ModelConfig, SFTConfig, SFTTrainer, load_model

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

#: Math problems with their gold answers — the answer is what the reward grades.
PROBLEMS = [
    {"prompt": "What is 17 * 23? Reply with just the number.", "gold": 391},
    {"prompt": "If x + 7 = 19, what is x? Reply with just the number.", "gold": 12},
    {"prompt": "A train goes 60 mph for 2.5 hours. How many miles? Just the number.",
     "gold": 150},
    {"prompt": "What is 144 / 12? Reply with just the number.", "gold": 12},
]


# ---------------------------------------------------------------------------
# The reward calculation — the same math grader GRPO would use.
# ---------------------------------------------------------------------------

def _final_integer(text: str):
    """The last integer appearing in the text, or None — the model's answer."""
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return int(matches[-1]) if matches else None


def math_reward(completion: str, gold: int) -> dict:
    """Grade one completion against its gold answer.

    Returns ``{"score", "correct"}``. Correctness (the answer matches) dominates;
    a small shaping term rewards being concise, so among correct answers the
    tidiest are preferred by the data manager's ``top_k`` selection.
    """
    answer = _final_integer(completion)
    correct = answer is not None and answer == gold
    score = 0.0
    if correct:
        score += 1.0
    if answer is not None:
        score += 0.1                      # produced a number at all
    if len(completion) <= 300:
        score += 0.1                      # concise
    return {"score": score, "correct": correct, "answer": answer}


# ---------------------------------------------------------------------------
# The workflow's own inference and reward code. ROME runs it in a loop.
# ---------------------------------------------------------------------------

model_config = ModelConfig(
    base_model_name=BASE_MODEL,
    lora_name="./adapters/math_sft",
    required_gpus=1,
)


def load_generator(checkpoint_path, ctx):
    """Startup + every published checkpoint. A LoRA-adapter read, so hot-swapping
    between batches never stalls generation."""
    config = ModelConfig(base_model_name=BASE_MODEL, lora_name=checkpoint_path,
                         required_gpus=1)
    model, tokenizer = load_model(config)
    return {"model": model, "tokenizer": tokenizer}


def generate(problems, ctx):
    """One batch of candidate answers.

    Each input is a ``{"prompt", "gold"}`` problem; the gold is threaded through
    so the reward stream can grade the answer. Only the *newly generated* tokens
    become the completion (the prompt is stripped), which is exactly what SFT
    should imitate.
    """
    import torch

    model, tokenizer = ctx.model["model"], ctx.model["tokenizer"]
    prompts = [p["prompt"] for p in problems]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                 temperature=0.8, top_p=0.95)
    # Slice off the prompt tokens so the completion is the answer alone.
    gen = outputs[:, inputs["input_ids"].shape[1]:]
    completions = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [
        {"prompt": p["prompt"], "gold": p["gold"], "completion": c}
        for p, c in zip(problems, completions)
    ]


def score(generations):
    """Grade each generation. Its returned dicts become corpus records — the
    reward stream's outputs are wired into the data manager automatically."""
    scored = []
    for item in generations:
        reward = math_reward(item["completion"], item["gold"])
        scored.append({**item, **reward})
    return scored


# ---------------------------------------------------------------------------
# ROME adoption
# ---------------------------------------------------------------------------

async def main():
    backend = await LocalExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)

    manager = rome.Manager(
        flow,
        data_config=rome.DataConfig(
            min_samples=16,
            # Selection = the reward. Keep only *correct* answers, then let the
            # shard be the best-scoring of those (concise ones win ties). This is
            # the "best-scoring and/or correct responses" the loop trains on.
            filter_func=lambda r: bool(r.get("correct")),
            sampling="top_k", score_key="score", shard_size=128,
            # One good answer per (problem, answer) is enough; drop duplicates.
            dedup_key=lambda r: (r.get("prompt"), r.get("completion")),
        ),
        trainer_config=rome.TrainerConfig(
            trainer=SFTTrainer(
                SFTConfig(
                    model_config=model_config,
                    prompt_column="prompt",
                    completion_column="completion",
                ),
            ),
            checkpoint_dir="./rome_checkpoints",
            poll_interval=5.0,
        ),
        stream_configs=[
            rome.StreamConfig(
                name="generate",
                kind=rome.StreamKind.INFERENCE,
                model_path=model_config.lora_name,
                load_func=load_generator,
                process_func=generate,
                num_streams=2,
                num_gpus=1,
                batch_size=4,
            ),
            rome.StreamConfig(
                name="score",
                kind=rome.StreamKind.REWARD,
                process_func=score,
                num_streams=1,
                num_gpus=0,
                batch_size=8,
            ),
        ],
    )
    await manager.start()

    try:
        for round_index in range(20):
            # Sample answers for every problem, then grade them. Correct ones
            # land in the corpus on their own; SFT fires once enough accumulate.
            manager.stream.submit_batch(PROBLEMS, stream="generate")
            for output in manager.stream.get_outputs(stream="generate"):
                manager.stream.submit(output["result"], stream="score")

            print(
                f"round {round_index}: corpus {manager.data.total_count} "
                f"(correct kept) | model v{manager.model_version} | "
                f"{manager.get_training_status().name}"
            )
            await asyncio.sleep(5.0)
    finally:
        await manager.stop()
        await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
