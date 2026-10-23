"""MedQA US test set; records carry the dataset fields plus ``gold``."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

PATH = "xuxuxuxuxu/MedQA_US_test"
REVISION = "f2b84f2b44f6fccd6872e8bdab99708290826456"
OUTPUT_COLUMN = "label"


def load_medqa_us() -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, split="train", revision=REVISION):
        row = dict(row)
        row["label"] = row["answer_idx"]
        row["choices"] = "".join(f"{k}. {v}\n" for k, v in row["options"].items())
        row["gold"] = row["label"]
        out.append(row)
    return out
