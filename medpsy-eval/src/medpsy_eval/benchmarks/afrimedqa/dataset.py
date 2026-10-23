"""AfriMed-QA v2 test MCQs; records carry the dataset fields plus ``gold``."""

from __future__ import annotations

import json
from typing import Any

from ...remote import hf_dataset

PATH = "afrimedqa/afrimedqa_v2"
REVISION = "3b4382fa0bb51bfc026f5813021ab0ec7be9de8f"
OUTPUT_COLUMN = "label"

_LETTERS = {f"option{i + 1}": chr(65 + i) for i in range(8)}


def load_afrimedqa() -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, split="train", revision=REVISION):
        if row.get("split") != "test" or row.get("question_type") != "mcq":
            continue
        question = row.get("question") or row.get("question_clean") or ""
        options_raw = row.get("answer_options")
        if not question or not options_raw:
            continue
        try:
            options = json.loads(options_raw) if isinstance(options_raw, str) else options_raw
        except json.JSONDecodeError:
            continue
        keys = sorted(k for k in options if k.startswith("option"))
        choices = "".join(f"{_LETTERS.get(k, k)}. {options[k]}\n" for k in keys)
        correct = row.get("correct_answer", "")
        if not choices.strip() or not correct:
            continue
        label = _LETTERS.get(correct, correct)
        out.append(
            {
                "question": question,
                "choices": choices.strip(),
                "label": label,
                "specialty": row.get("specialty", ""),
                "country": row.get("country", ""),
                "gold": label,
            }
        )
    return out
