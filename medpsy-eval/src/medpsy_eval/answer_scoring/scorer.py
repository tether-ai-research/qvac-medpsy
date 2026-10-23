"""Closed-ended answer scoring by protocol: ``v1`` (MedPsy v1) scores with the answer-extraction
judge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

from inspect_ai._util.registry import RegistryInfo, registry_info, set_registry_info
from inspect_ai.scorer import Metric, Scorer

from ..judging import JudgeArg, JudgeSettings, judge_settings
from .judge_extraction import judge_extraction

Protocol = Literal["v1"]
PROTOCOLS: tuple[str, ...] = get_args(Protocol)

JUDGE = JudgeSettings(temperature=0.01, max_tokens=4096)


def with_metrics(scorer: Scorer, metrics: Sequence[Metric]) -> Scorer:
    """Replace one scorer's metrics. ``Task(metrics=...)`` replaces the metrics of every scorer
    on the task, including ``generation_stats``."""
    info = registry_info(scorer)
    set_registry_info(scorer, RegistryInfo(type=info.type, name=info.name,
                                           metadata={**info.metadata, "metrics": list(metrics)}))
    return scorer


def answer_scorer(
    protocol: Protocol,
    *,
    user: str,
    system: str | None,
    judge: JudgeArg,
    metrics: Sequence[Metric] | None = None,
) -> Scorer:
    """``user`` / ``system``: the answer-extraction judge templates. ``metrics`` replaces the
    scorer's default metrics."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    config = judge_settings(judge).generate_config()
    scorer = judge_extraction(user=user, system=system, config=config)
    return with_metrics(scorer, metrics) if metrics is not None else scorer
