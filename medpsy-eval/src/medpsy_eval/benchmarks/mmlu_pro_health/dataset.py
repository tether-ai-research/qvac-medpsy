"""MMLU-Pro test set, one category (``health``); records carry the dataset fields plus
``gold``."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

PATH = "TIGER-Lab/MMLU-Pro"
REVISION = "7db2dc66c66870201e1cf041a3a9297a7674c50a"
OUTPUT_COLUMN = "answer"

_LETTERS = "ABCDEFGHIJKLMNOP"


def load_mmlu_pro(category: str = "health") -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, split="test", revision=REVISION):
        if row["category"] != category:
            continue
        row = dict(row)
        s = ""
        for i, opt in enumerate(row["options"]):
            if opt == "N/A":
                continue
            s += f"{_LETTERS[i]}. {opt}\n"
        row["options_str"] = s.strip()
        row["gold"] = row["answer"]
        out.append(row)
    return out
