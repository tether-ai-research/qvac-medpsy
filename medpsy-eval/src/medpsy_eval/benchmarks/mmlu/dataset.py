"""MMLU test sets of the medical subjects; records carry the dataset fields plus ``gold``."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

PATH = "cais/mmlu"
REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
OUTPUT_COLUMN = "target"

SUBJECTS = (
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
)


def load_mmlu(subject: str) -> list[dict[str, Any]]:
    out = []
    for row in hf_dataset(PATH, subject, split="test", revision=REVISION):
        a, b, c, d = row["choices"]
        target = "ABCD"[row["answer"]]
        out.append({"input": row["question"], "A": a, "B": b, "C": c, "D": d, "target": target,
                    "gold": target})
    return out
