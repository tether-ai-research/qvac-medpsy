"""HealthBench loader."""

from __future__ import annotations

from typing import Any

from ...remote import hf_dataset

HEALTHBENCH_REPO = "huihuixu/healthbench"
HEALTHBENCH_REVISION = "e35c21171476f5e25803f936c6d48eb3df97b65e"
HEALTHBENCH_FILES = {
    "": "2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "hard_2025-05-08-21-00-10.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
}


def load_healthbench(subset: str = "") -> list[dict[str, Any]]:
    """Rows as loaded by ``datasets`` (column types matter: rubric ``points`` are
    rendered into the grader prompt)."""
    ds = hf_dataset(HEALTHBENCH_REPO, data_files={"test": HEALTHBENCH_FILES[subset]},
                    split="test", revision=HEALTHBENCH_REVISION)
    return [dict(row) for row in ds]


def subset_ids(subset: str) -> set[str]:
    return {row["prompt_id"] for row in load_healthbench(subset)}

