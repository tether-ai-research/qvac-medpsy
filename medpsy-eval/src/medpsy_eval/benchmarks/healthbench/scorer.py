"""HealthBench rubric grading.

Each rubric item is graded independently by the judge role. A sample's score is
achieved positive-and-negative points over total positive points (unclipped); the reported
score is the mean over samples clipped to [0, 1], overall and per subset (hard, consensus).
The axis and theme tag scores are aggregated the same way.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import SampleScore, Score, Target, Value, metric, scorer
from inspect_ai.solver import TaskState

from ...judging import judge_with_retries
from ...reasoning import strip_thinking_tokens
from .prompts import GRADER_SYSTEM_MESSAGE, GRADER_TEMPLATE

MAX_PARSE_ATTEMPTS = 5


def rubric_item_text(item: dict[str, Any]) -> str:
    return f"[{item['points']}] {item['criterion']}"


def grader_prompt(conversation: Sequence[dict[str, Any]], response: str,
                  item: dict[str, Any]) -> str:
    convo = [*conversation, {"role": "assistant", "content": response}]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)
    return GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
        "<<rubric_item>>", rubric_item_text(item))


def parse_json_to_dict(text: str) -> dict:
    if not text or not text.strip():
        return {}
    text = strip_thinking_tokens(text)
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    for match in re.findall(r'\{[^{}]*"criteria_met"[^{}]*\}', cleaned, re.DOTALL):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    for match in re.findall(r"\{[^{}]+\}", cleaned):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "criteria_met" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    try:
        start, end = cleaned.rfind("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
    except json.JSONDecodeError:
        pass
    return {}


def calculate_score(items: Sequence[dict[str, Any]], met: Sequence[bool]) -> float | None:
    total = sum(it["points"] for it in items if it["points"] > 0)
    if total == 0:
        return None
    return sum(it["points"] for it, m in zip(items, met, strict=True) if m) / total


def tag_scores(example_tags: Sequence[str], items: Sequence[dict[str, Any]],
               met: Sequence[bool], overall: float) -> dict[str, float]:
    out = {tag: overall for tag in example_tags}
    by_tag: dict[str, list[tuple[dict, bool]]] = defaultdict(list)
    for it, m in zip(items, met):
        for tag in it.get("tags") or []:
            by_tag[tag].append((it, m))
    for tag, pairs in by_tag.items():
        s = calculate_score([p[0] for p in pairs], [p[1] for p in pairs])
        if s is not None:
            out[tag] = s
    return out


def _verdict(text: str) -> dict | None:
    parsed = parse_json_to_dict(text)
    return parsed if isinstance(parsed.get("criteria_met"), bool) else None


async def grade_item(prompt: str, config: GenerateConfig) -> dict[str, Any]:
    parsed, _, attempts = await judge_with_retries(
        GRADER_SYSTEM_MESSAGE, prompt, config, _verdict, MAX_PARSE_ATTEMPTS)
    if parsed is None:
        return {"criteria_met": False, "attempts": MAX_PARSE_ATTEMPTS,
                "explanation": "Judge failed to produce valid response after max retries"}
    return {**parsed, "attempts": attempts}


def clipped_mean(scores: list[SampleScore], subset: str | None = None) -> float:
    vals = [s.score.as_float() for s in scores
            if subset is None or subset in (s.score.metadata or {}).get("subsets", [])]
    return float(np.clip(np.mean(vals), 0, 1)) if vals else float("nan")


async def grade_sample(row: dict[str, Any], response: str,
                       config: GenerateConfig) -> list[dict[str, Any]]:
    prompts = [grader_prompt(row["prompt"], response, it) for it in row["rubrics"]]
    return list(await asyncio.gather(*(grade_item(p, config) for p in prompts)))


@metric
def healthbench_score():
    def compute(scores: list[SampleScore]) -> Value:
        return clipped_mean(scores)

    return compute


@metric
def healthbench_hard():
    def compute(scores: list[SampleScore]) -> Value:
        return clipped_mean(scores, "hard")

    return compute


@metric
def healthbench_consensus():
    def compute(scores: list[SampleScore]) -> Value:
        return clipped_mean(scores, "consensus")

    return compute


TAG_PREFIXES = ("axis:", "theme:")
SUBSET_PREFIXES = {None: "healthbench", "hard": "healthbench_hard",
                   "consensus": "healthbench_consensus"}


def tag_means(scores: list[SampleScore]) -> dict[str, float]:
    """Clipped mean of each axis/theme tag over the scores that carry it, per subset, keyed as
    ``healthbench-axis:accuracy``, ``healthbench_hard-theme:hedging``."""
    out = {}
    for subset, prefix in SUBSET_PREFIXES.items():
        by_tag: dict[str, list[float]] = defaultdict(list)
        for s in scores:
            meta = s.score.metadata or {}
            if subset is not None and subset not in meta.get("subsets", []):
                continue
            for tag, value in (meta.get("tag_scores") or {}).items():
                if tag.startswith(TAG_PREFIXES):
                    by_tag[tag].append(value)
        for tag in sorted(by_tag):
            out[f"{prefix}-{tag}"] = float(np.clip(np.mean(by_tag[tag]), 0, 1))
    return out


# Unreduced: the epoch reducer keeps only the first epoch's metadata, so per-epoch tag scores
# would be lost.
@metric(scores="unreduced")
def healthbench_tags():
    def compute(scores: list[SampleScore]) -> Value:
        return tag_means(scores)

    return compute


@scorer(metrics=[healthbench_score(), healthbench_hard(), healthbench_consensus(),
                 healthbench_tags()])
def healthbench_rubric(config: GenerateConfig, hard_ids: frozenset[str] = frozenset(),
                       consensus_ids: frozenset[str] = frozenset()):

    async def score(state: TaskState, target: Target) -> Score:
        row = state.metadata["record"]
        response = strip_thinking_tokens(state.output.completion if state.output else "")
        items = row["rubrics"]
        grades = await grade_sample(row, response, config)
        met = [bool(g["criteria_met"]) for g in grades]
        overall = calculate_score(items, met)
        assert overall is not None
        pid = row["prompt_id"]
        subsets = [name for name, ids in (("hard", hard_ids), ("consensus", consensus_ids))
                   if pid in ids]
        return Score(
            value=overall,
            answer=response,
            metadata={
                "prompt_id": pid,
                "subsets": subsets,
                "tag_scores": tag_scores(row["example_tags"], items, met, overall),
                "rubric_grades": [{**it, **g} for it, g in zip(items, grades)],
            },
        )

    return score

