"""MedPsy v1 closed-ended suite (judge-only answer extraction)."""

from inspect_ai import Task, task

from ..benchmarks.afrimedqa import afrimedqa
from ..benchmarks.medmcqa import medmcqa
from ..benchmarks.medqa import medqa
from ..benchmarks.medxpertqa import medxpertqa
from ..benchmarks.mmlu import mmlu
from ..benchmarks.mmlu_pro_health import mmlu_pro_health
from ..benchmarks.pubmedqa import pubmedqa
from ..judging import JudgeSettings
from ._compose import Suite, Variant, compose

VARIANTS = {
    "medxpertqa": Variant(medxpertqa, {"protocol": "v1"}),
    "AfriMedQA": Variant(afrimedqa, {"protocol": "v1"}),
    "mmlu_pro_health": Variant(mmlu_pro_health, {"protocol": "v1"}),
    "MedQA_US": Variant(medqa, {"protocol": "v1"}),
    "medmcqa": Variant(medmcqa, {"protocol": "v1"}),
    "PubMedQA": Variant(pubmedqa, {"protocol": "v1"}),
    "mmlu": Variant(mmlu, {"protocol": "v1"}),
}


@task
def closed_ended_v1(
    dataset: str,
    epochs: int = 3,
    judge_temperature: float = 0.01,
    judge_max_tokens: int = 4096,
    tokenizer: str | None = None,
) -> Task:
    return compose(
        VARIANTS,
        dataset,
        suite="closed_ended_v1",
        epochs=epochs,
        judge=JudgeSettings(temperature=judge_temperature, max_tokens=judge_max_tokens),
        tokenizer=tokenizer,
    )


SUITES = (Suite.of(closed_ended_v1, VARIANTS),)
