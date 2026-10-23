"""LLM answer-extraction scorer.

The judge only extracts the candidate's final answer; correctness is decided locally by
``judge_verdict``. The judge is the ``grader`` model role.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessage, GenerateConfig, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    Value,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from ..judging import JUDGE_ROLE, answer_text
from ..prompting import render_messages
from ..reasoning import strip_thinking
from .extraction import MULTIPLE_ANSWERS, NO_ANSWER, judge_verdict


def judge_prompt(state: TaskState, target: Target, user: str,
                 system: str | None) -> tuple[str, list[ChatMessage]]:
    prediction = strip_thinking(answer_text(state))
    gold = target.text
    record = {**state.metadata["record"], "prediction": prediction, "obj_gold": gold,
              "reference": gold}
    return prediction, render_messages(user, record, system)


def prompt_metadata(messages: list[ChatMessage]) -> dict[str, str | None]:
    """The judge request as score metadata: system text (None without one) and user text."""
    system = messages[0].text if messages[0].role == "system" else None
    return {"judge_system": system, "judge_prompt": messages[-1].text}


async def run_judge(messages: list[ChatMessage],
                    config: GenerateConfig) -> tuple[str, str | None]:
    out = await get_model(role=JUDGE_ROLE).generate(messages, config=config)
    reasoning = None
    if out.message and not isinstance(out.message.content, str):
        parts = [c.reasoning for c in out.message.content if c.type == "reasoning"]
        reasoning = "\n".join(parts) or None
    return out.completion, reasoning


def answer_rate(scores: list[SampleScore], sentinel: str) -> float:
    """Percent of scores whose extracted answer is ``sentinel``."""
    if not scores:
        return float("nan")
    return 100.0 * sum(s.score.answer == sentinel for s in scores) / len(scores)


# Unreduced: the epoch reducer keeps only the first epoch's answer; these rates count every run.
@metric(scores="unreduced")
def no_answer_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return answer_rate(scores, NO_ANSWER)

    return compute


@metric(scores="unreduced")
def multiple_answers_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return answer_rate(scores, MULTIPLE_ANSWERS)

    return compute


METRICS = [accuracy(), stderr(), no_answer_rate(), multiple_answers_rate()]


@scorer(metrics=METRICS)
def judge_extraction(user: str, system: str | None, config: GenerateConfig):
    async def score(state: TaskState, target: Target) -> Score:
        prediction, messages = judge_prompt(state, target, user, system)
        metadata = {**prompt_metadata(messages), "prediction": prediction}
        response, reasoning = await run_judge(messages, config)
        metadata["judge_reasoning"] = reasoning
        extracted, ok = judge_verdict(response, target.text)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=extracted,
            explanation=response,
            metadata=metadata,
        )

    return score
