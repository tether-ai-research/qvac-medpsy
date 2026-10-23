"""Generation diagnostics: token lengths of the reasoning trace and the final answer,
reasoning-block counts, empty answers, and max-token truncation. No model calls.

Each sample scores a dict; the metrics are the mean of each key, so rates are percentages and
token lengths are mean tokens per output, as on the leaderboard. Tokens are counted with the
evaluated model's Hugging Face tokenizer (``tokenizer``, default the model name); when it
cannot be loaded the token-length keys are unscored (NaN).
Truncation keys are unscored when the provider reports no stop reason.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from functools import cache

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)

TOKEN_LEN_REASONING = "token_len_reasoning"
TOKEN_LEN_ANSWER = "token_len_answer"
TOKEN_LEN_TOTAL = "token_len_total"
ZERO_TOKEN_ANSWER_RATE = "zero_token_answer_rate"
ZERO_REASONING_RATE = "zero_reasoning_rate"
SINGLE_REASONING_RATE = "single_reasoning_rate"
MULTIPLE_REASONING_RATE = "multiple_reasoning_rate"
TRUNCATION_RATE = "truncation_rate"
TRUNCATED_IN_REASONING_RATE = "truncated_in_reasoning_rate"
TRUNCATED_IN_ANSWER_RATE = "truncated_in_answer_rate"

KEYS = (
    TOKEN_LEN_REASONING, TOKEN_LEN_ANSWER, TOKEN_LEN_TOTAL, ZERO_TOKEN_ANSWER_RATE,
    ZERO_REASONING_RATE, SINGLE_REASONING_RATE, MULTIPLE_REASONING_RATE,
    TRUNCATION_RATE, TRUNCATED_IN_REASONING_RATE, TRUNCATED_IN_ANSWER_RATE,
)

_REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<unused94>", "<unused95>"),
    ("[THOUGHT]", "[/THOUGHT]"),
    ("<reasoning>", "</reasoning>"),
)
REASONING_END_MARKERS = tuple(end for _, end in _REASONING_TAG_PAIRS) + ("</thought>", "[/THINK]")
# vLLM reports both as finish_reason "length"
TRUNCATED_STOP_REASONS = ("max_tokens", "model_length")


def _split_embedded_reasoning(text: str) -> tuple[str, str]:
    for start, end in _REASONING_TAG_PAIRS:
        match = re.search(re.escape(start) + r"(.*?)" + re.escape(end) + r"(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        if end in text:
            reasoning, answer = text.rsplit(end, 1)
            return reasoning.strip(), answer.strip()
        if start in text:
            _, reasoning = text.rsplit(start, 1)
            return reasoning.strip(), ""
    return "", text.strip()


def split_reasoning_answer(reasoning: str, prediction: str) -> tuple[str, str]:
    """``(reasoning, answer)`` texts: the server-split reasoning (answer kept unstripped)
    or else reasoning tags embedded in the prediction."""
    if reasoning:
        return reasoning.strip(), prediction
    return _split_embedded_reasoning(prediction)


def count_reasoning_blocks(reasoning: str, prediction: str) -> int:
    """Closed reasoning blocks, counted by end markers. Server-split reasoning lost its first
    end marker, so non-empty split reasoning counts as one block."""
    markers = sum(text.count(m) for text in (reasoning, prediction) for m in REASONING_END_MARKERS)
    return (1 if reasoning.strip() else 0) + markers


@cache
def token_counter(name: str) -> Callable[[str], int] | None:
    """Token count with ``name``'s tokenizer, or None (logged once) when it cannot be loaded."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001
        logger.warning("generation_stats: no tokenizer %r (%s); token lengths unscored", name, e)
        return None
    return lambda text: len(tokenizer(text).input_ids) if text else 0


def _percent(flag: bool) -> float:
    return 100.0 if flag else 0.0


def sample_stats(reasoning: str, prediction: str, stop_reason: str | None,
                 count: Callable[[str], int] | None) -> dict[str, float]:
    reasoning_text, answer_text = split_reasoning_answer(reasoning, prediction)
    if count is None:
        r_len = a_len = math.nan
    else:
        r_len, a_len = float(count(reasoning_text)), float(count(answer_text))
    blocks = count_reasoning_blocks(reasoning, prediction)
    zero_reasoning = not reasoning.strip() and not _split_embedded_reasoning(prediction)[0]
    if stop_reason is None or stop_reason == "unknown":
        truncated = in_reasoning = math.nan
    else:
        is_truncated = stop_reason in TRUNCATED_STOP_REASONS
        no_answer = not prediction.strip() or not answer_text.strip()
        truncated = _percent(is_truncated)
        in_reasoning = _percent(is_truncated and no_answer)
    return {
        TOKEN_LEN_REASONING: r_len,
        TOKEN_LEN_ANSWER: a_len,
        TOKEN_LEN_TOTAL: r_len + a_len,
        ZERO_TOKEN_ANSWER_RATE: math.nan if count is None else _percent(a_len == 0),
        ZERO_REASONING_RATE: _percent(zero_reasoning),
        SINGLE_REASONING_RATE: _percent(blocks == 1),
        MULTIPLE_REASONING_RATE: _percent(blocks >= 2),
        TRUNCATION_RATE: truncated,
        TRUNCATED_IN_REASONING_RATE: in_reasoning,
        TRUNCATED_IN_ANSWER_RATE: truncated - in_reasoning,
    }


def output_parts(state: TaskState) -> tuple[str, str, str | None]:
    """``(reasoning, prediction, stop_reason)`` of the sample's model output."""
    output = state.output
    if output is None or not output.choices:
        return "", "", None
    choice = output.choices[0]
    content = choice.message.content
    reasoning = "" if isinstance(content, str) else "\n".join(
        c.reasoning for c in content if c.type == "reasoning")
    return reasoning, output.completion or "", choice.stop_reason


@scorer(metrics={key: [mean()] for key in KEYS})
def generation_stats(tokenizer: str | None = None):
    """``tokenizer``: Hugging Face tokenizer for token lengths (default: the model name)."""

    async def score(state: TaskState, target: Target) -> Score:
        reasoning, prediction, stop_reason = output_parts(state)
        count = token_counter(tokenizer or state.model.name)
        return Score(value=sample_stats(reasoning, prediction, stop_reason, count))

    return score
