"""Chat messages built from a benchmark's prompt templates and a dataset record.

Templates are filled the way the original MedPsy evaluation filled them, so
the prompt text matches the paper runs byte for byte: one plain ``str.replace`` of ``{field}`` per field
of the whole record, in record order, leaving every other brace untouched (``{{A}}`` stays
``{{A}}``). ``str.format`` would differ on templates with literal braces or unused fields.

The original evaluation folded the system text into the user message; this package sends it
as a real system message, without its trailing whitespace, followed by the user message.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser


def safe_format(template: str, fields: Iterable[tuple[str, Any]]) -> str:
    for key, value in fields:
        template = template.replace(f"{{{key}}}", str(value))
    return template


def render_messages(
    user_template: str,
    record: Mapping[str, Any],
    system: str | None = None,
) -> list[ChatMessage]:
    """``[system, user]`` (or ``[user]`` without a system template), both filled from
    ``record``."""
    fields = list(record.items())
    user = ChatMessageUser(content=safe_format(user_template, fields))
    if system is None:
        return [user]
    return [ChatMessageSystem(content=safe_format(system, fields).rstrip()), user]

