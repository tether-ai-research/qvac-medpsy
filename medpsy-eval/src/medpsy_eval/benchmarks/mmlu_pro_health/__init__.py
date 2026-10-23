"""MMLU-Pro health category, dataset id ``mmlu_pro_health``."""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate

from ...answer_scoring import JUDGE, Protocol, answer_scorer, samples
from ...generation_stats import generation_stats
from ...judging import JudgeArg
from . import prompts
from .dataset import OUTPUT_COLUMN, load_mmlu_pro

NAME = "mmlu_pro_health"
INFER_USER = {
    "v1": prompts.QUESTION_FORMAT,
}


@task
def mmlu_pro_health(
    *,
    protocol: Protocol = "v1",
    epochs: int = 3,
    judge: JudgeArg = JUDGE,
    tokenizer: str | None = None,
) -> Task:
    scorer = answer_scorer(protocol, user=prompts.JUDGE_USER, system=prompts.JUDGE_SYSTEM,
                           judge=judge)
    records = load_mmlu_pro("health")
    return Task(
        name=NAME,
        dataset=MemoryDataset(
            samples(NAME, records, INFER_USER[protocol], prompts.INFER_SYSTEM, OUTPUT_COLUMN),
            name=NAME,
        ),
        solver=generate(),
        scorer=[scorer, generation_stats(tokenizer)],
        epochs=Epochs(epochs, "mean"),
        metadata={"protocol": protocol},
    )


__all__ = ["mmlu_pro_health"]
