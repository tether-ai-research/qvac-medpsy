"""PubMedQA PQA-L test set (500), as served by the BigBio ``fold0_source`` config; records
carry the dataset fields plus ``gold``."""

from __future__ import annotations

import json
import zipfile
from typing import Any

from huggingface_hub import hf_hub_download

PATH = "bigbio/pubmed_qa"
REVISION = "15578e80ef02c2c9636d33fdd7eb6b05b987ba60"
OUTPUT_COLUMN = "label"


def load_pubmedqa() -> list[dict[str, Any]]:
    archive = hf_hub_download(PATH, "pqal.zip", repo_type="dataset", revision=REVISION)
    with zipfile.ZipFile(archive) as zf:
        data = json.loads(zf.read("pqal_test_set.json"))
    out = []
    for row in data.values():
        contexts = row.get("CONTEXTS", [])
        context_text = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        decision = row.get("final_decision", "")
        label = {"yes": "A. yes", "no": "B. no"}.get(decision, "C. maybe")
        out.append(
            {
                "question": f"Context:\n{context_text}\n\nQuestion: {row.get('QUESTION', '')}",
                "choices": "A. yes\nB. no\nC. maybe",
                "label": label,
                "gold": label,
            }
        )
    return out
