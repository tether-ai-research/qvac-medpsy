"""MMLU medical subjects as one benchmark: the six datasets ``lukaemon_mmlu_<subject>`` pooled
into one Task.

Metrics: ``accuracy`` over all questions (the leaderboard's pooled MMLU score), each subject's
accuracy, and ``subject_mean``, the unweighted mean of the six (the MedPsy v1 paper's
MMLU-Health). Sample ids are ``<subject>/<row>``.
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from ...answer_scoring import (
    JUDGE,
    Protocol,
    answer_scorer,
    multiple_answers_rate,
    no_answer_rate,
    samples,
)
from ...generation_stats import generation_stats
from ...judging import JudgeArg
from . import prompts
from .dataset import OUTPUT_COLUMN, SUBJECTS, load_mmlu

NAME = "mmlu"
INFER_USER = {
    "v1": prompts.QUESTION_FORMAT,
}


def subject_dataset(subject: str) -> str:
    return f"lukaemon_mmlu_{subject}"


SUBJECT_DATASETS = tuple(subject_dataset(subject) for subject in SUBJECTS)


@task
def mmlu(
    *,
    protocol: Protocol = "v1",
    epochs: int = 3,
    judge: JudgeArg = JUDGE,
    tokenizer: str | None = None,
) -> Task:
    scorer = answer_scorer(
        protocol, user=prompts.JUDGE_USER, system=prompts.JUDGE_SYSTEM, judge=judge,
        metrics=[accuracy(), stderr(),
                 grouped(accuracy(), "subject", all="groups", all_label="subject_mean"),
                 no_answer_rate(), multiple_answers_rate()],
    )
    pooled = []
    for subject in SUBJECTS:
        name = subject_dataset(subject)
        for sample in samples(name, load_mmlu(subject), INFER_USER[protocol],
                              prompts.INFER_SYSTEM, OUTPUT_COLUMN):
            sample.metadata = {**sample.metadata, "subject": subject}
            sample.id = f"{subject}/{sample.id}"
            pooled.append(sample)
    return Task(
        name=NAME,
        dataset=MemoryDataset(pooled, name=NAME),
        solver=generate(),
        scorer=[scorer, generation_stats(tokenizer)],
        epochs=Epochs(epochs, "mean"),
        metadata={"protocol": protocol},
    )


__all__ = ["SUBJECTS", "SUBJECT_DATASETS", "mmlu"]
