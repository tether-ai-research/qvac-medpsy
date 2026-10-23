"""Answer comparison and judge-response parsing (pure functions, no model calls)."""

from __future__ import annotations

import re

ANSWER_MARKER = "Extracted Candidate's Answer:"
NO_ANSWER = "NO_ANSWER"
MULTIPLE_ANSWERS = "MULTIPLE_ANSWERS"


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    answer = answer.lower().strip()
    mc_match = re.match(r"^([a-h])[\.\)]\s*\w+", answer)
    if mc_match:
        return mc_match.group(1)
    answer = re.sub(r"^\(([a-h])\)$", r"\1", answer)
    answer = re.sub(r"[.,;:!?]$", "", answer)
    if len(answer.strip()) > 1:
        answer = re.sub(r"\b(a|an|the)\b", "", answer)
    return re.sub(r"\s+", " ", answer).strip()


def answers_match(extracted: str, gold: str) -> bool:
    extracted_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(gold)
    if extracted_norm == gold_norm:
        return True
    if (len(extracted_norm) == 1 and extracted_norm.isalpha()) or (
        len(gold_norm) == 1 and gold_norm.isalpha()
    ):
        return False
    if len(extracted_norm) > 1 and len(gold_norm) > 1:
        return gold_norm in extracted_norm or extracted_norm in gold_norm
    return False


def extract_answer_from_judge_response(judge_response: str) -> str:
    if ANSWER_MARKER in judge_response:
        return judge_response.split(ANSWER_MARKER, 1)[1].strip()
    return judge_response.strip()


def judge_verdict(judge_response: str, gold: str) -> tuple[str, bool]:
    """Return ``(extracted_answer, is_correct)`` for one judge response."""
    extracted = extract_answer_from_judge_response(str(judge_response)).strip()
    gold = str(gold).strip()
    if extracted in (NO_ANSWER, MULTIPLE_ANSWERS):
        return extracted, False
    return extracted, answers_match(extracted, gold)
