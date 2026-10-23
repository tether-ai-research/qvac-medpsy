"""Closed-ended inputs: one sample per dataset record."""

from __future__ import annotations

from typing import Any

from inspect_ai.dataset import Sample

from ..prompting import render_messages


def samples(name: str, records: list[dict[str, Any]], user: str, system: str | None,
            output_column: str) -> list[Sample]:
    """One sample per record; the row index is the sample id."""
    return [
        Sample(
            id=index,
            input=render_messages(user, record, system),
            target=str(record[output_column]),
            metadata={"record": record, "dataset": name},
        )
        for index, record in enumerate(records)
    ]
