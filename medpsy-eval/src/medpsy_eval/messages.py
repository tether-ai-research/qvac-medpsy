"""Shared conversion helpers for conversational benchmarks."""

from __future__ import annotations

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)

_ROLES = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
}


def to_messages(conversation: list[dict]) -> list[ChatMessage]:
    return [_ROLES[message["role"]](content=message["content"]) for message in conversation]
