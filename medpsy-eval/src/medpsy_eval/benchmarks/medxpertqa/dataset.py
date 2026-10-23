"""MedXpertQA Text test set; records carry the dataset fields plus ``gold``."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

PATH = "TsinghuaC3I/MedXpertQA"
REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"
OUTPUT_COLUMN = "label"


def load_medxpertqa() -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, "Text", split="test", revision=REVISION):
        row = dict(row)
        row["start"] = "A"
        row["end"] = chr(65 + len(row.get("options", [])) - 1)
        row["prompt_mode"] = "zero-shot"
        row["gold"] = row["label"]
        out.append(row)
    return out
