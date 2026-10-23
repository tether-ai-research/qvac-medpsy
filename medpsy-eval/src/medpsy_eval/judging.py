"""Judge request settings and the shared judge-call primitives (the judge is the ``grader``
model role)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import TaskState
from pydantic import BaseModel, ConfigDict

JUDGE_ROLE = "grader"


class JudgeSettings(BaseModel):
    """A pydantic model so Inspect writes it to the log's task args as JSON (it records other
    objects by type name only)."""

    model_config = ConfigDict(frozen=True)

    temperature: float
    max_tokens: int

    def generate_config(self) -> GenerateConfig:
        return GenerateConfig(temperature=self.temperature, max_tokens=self.max_tokens)


JudgeArg = JudgeSettings | Mapping[str, Any]


def judge_settings(value: JudgeArg) -> JudgeSettings:
    """``-T judge={temperature: ..., max_tokens: ...}`` and task args read back from a log
    arrive as a mapping."""
    return value if isinstance(value, JudgeSettings) else JudgeSettings(**value)


def answer_text(state: TaskState) -> str:
    """Model answer without the reasoning trace."""
    return state.output.completion if state.output else ""


async def judge_text(system: str, prompt: str, config: GenerateConfig) -> str:
    out = await get_model(role=JUDGE_ROLE).generate(
        [ChatMessageSystem(content=system), ChatMessageUser(content=prompt)], config=config)
    return out.completion


async def judge_with_retries(system: str, prompt: str, config: GenerateConfig, parse,
                             attempts: int):
    """Call the judge until ``parse`` returns a value (or attempts run out).

    Returns ``(parsed or None, last_response, attempts_used)``.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        last = await judge_text(system, prompt, config)
        parsed = parse(last)
        if parsed is not None:
            return parsed, last, attempt
    return None, last, attempts
