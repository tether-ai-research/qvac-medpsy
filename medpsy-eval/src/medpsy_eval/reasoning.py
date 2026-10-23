"""Reasoning / final-answer split, matching the MedPsy paper evaluation.

Generation side: the server's reasoning parser (vLLM ``--reasoning-parser``) returns the
chain of thought in ``reasoning_content`` (or ``reasoning``) and the answer in ``content``.
If the server did not split, the last ``</think>``-style end marker is used as a fallback.

Scoring side: ``strip_thinking`` is applied to the answer before judge calls. A leading start marker without any end
marker means the model ran out of tokens mid-reasoning and is mapped to ``NO_ANSWER``.
"""

from __future__ import annotations

THINKING_END_MARKERS = [
    "</think>",
    "<unused95>",
    "</reasoning>",
    "</thought>",
    "[/THINK]",
]

THINKING_START_MARKERS = [
    "<think>",
    "<unused94>",
    "<reasoning>",
    "<thought>",
    "[THINK]",
]

# ~20k chars keeps predictions well inside a 32k-token judge context.
MAX_PREDICTION_CHARS = 20000

_SPLIT_PATTERNS = [
    ("</think>", "<think>"),
    ("<unused95>", "<unused94>"),
    ("[/THOUGHT]", "[THOUGHT]"),
    ("</reasoning>", "<reasoning>"),
]


def split_reasoning(content: str | None, reasoning: str | None = None) -> tuple[str, str]:
    """Return ``(reasoning, answer)`` for one completion."""
    content = content or ""
    if reasoning:
        return reasoning, content.strip()
    for end_token, start_token in _SPLIT_PATTERNS:
        if end_token in content:
            head, answer = content.rsplit(end_token, 1)
            if start_token in head:
                head = head.split(start_token, 1)[-1]
            return head.strip(), answer.strip()
    return "", content


def strip_thinking_tokens(text: str) -> str:
    """Answer part of ``split_reasoning``: the stripped text after the last end marker, else
    the text unchanged."""
    if not text:
        return text
    return split_reasoning(text)[1]


def strip_thinking(text: str, max_prediction_chars: int = MAX_PREDICTION_CHARS) -> str:
    """Closed-ended answer with any reasoning removed, truncated to ``max_prediction_chars``."""
    if not text:
        return text

    found_end_marker = False
    for marker in THINKING_END_MARKERS:
        idx = text.lower().rfind(marker.lower())
        if idx != -1:
            found_end_marker = True
            after_marker = text[idx + len(marker):].strip()
            if after_marker:
                text = after_marker
                break

    if not found_end_marker:
        text_lower = text.lower()
        for start_marker in THINKING_START_MARKERS:
            if text_lower.startswith(start_marker.lower()):
                text = "NO_ANSWER"
                break

    if max_prediction_chars > 0 and len(text) > max_prediction_chars:
        text = text[:max_prediction_chars] + "\n[... output truncated for judge ...]"
    return text
