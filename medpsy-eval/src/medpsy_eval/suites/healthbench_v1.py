"""MedPsy v1 HealthBench suite."""

from inspect_ai import Task, task

from ..benchmarks.healthbench import healthbench
from ..judging import JudgeSettings
from ._compose import Suite, Variant, compose

VARIANTS = {"healthbench": Variant(healthbench)}


@task
def healthbench_v1(
    epochs: int = 1,
    judge_temperature: float = 0.5,
    judge_max_tokens: int = 2048,
    tokenizer: str | None = None,
) -> Task:
    return compose(
        VARIANTS,
        "healthbench",
        suite="healthbench_v1",
        epochs=epochs,
        judge=JudgeSettings(temperature=judge_temperature, max_tokens=judge_max_tokens),
        tokenizer=tokenizer,
    )


SUITES = (Suite.of(healthbench_v1, VARIANTS),)
