"""MedMCQA validation set; records carry the dataset fields plus ``gold``."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

PATH = "openlifescienceai/medmcqa"
REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
OUTPUT_COLUMN = "label"


def load_medmcqa() -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, split="validation", revision=REVISION):
        row = dict(row)
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        row["options"] = options
        options_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        row["question"] = f"{row['question']}\n{options_str}"
        row["label"] = chr(65 + row["cop"])
        row["subject_name"] = row["subject_name"].replace("_", " ")
        row["prompt_mode"] = "zero-shot"
        row["start"] = "A"
        row["end"] = chr(65 + len(options) - 1)
        row["gold"] = row["label"]
        out.append(row)
    return out
