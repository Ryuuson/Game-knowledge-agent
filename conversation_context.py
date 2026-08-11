"""Prepare bounded model context without changing persisted chat history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


MESSAGE_OVERHEAD_TOKENS = 6


@dataclass(frozen=True)
class ContextSplit:
    """A complete-turn suffix and the older raw messages it replaces."""

    messages_to_summarize: list[BaseMessage]
    recent_messages: list[BaseMessage]


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def estimate_message_tokens(messages: Sequence[BaseMessage]) -> int:
    """Conservatively estimate tokens without depending on provider billing data."""
    total = 0
    for message in messages:
        text = _message_text(message)
        cjk_characters = sum("\u4e00" <= char <= "\u9fff" for char in text)
        other_characters = len(text) - cjk_characters
        total += math.ceil(cjk_characters * 1.3 + other_characters / 3.5)
        total += MESSAGE_OVERHEAD_TOKENS
    return total


def remove_internal_summary(text: str, summary: str) -> str:
    """Remove a verbatim internal-memory echo from model output."""
    normalized_summary = summary.strip()
    if not normalized_summary:
        return text
    return text.replace(normalized_summary, "").strip()


def split_complete_user_turns(
    messages: Sequence[BaseMessage],
    *,
    recent_user_turns: int,
) -> ContextSplit:
    """Keep whole recent user turns so tool messages never lose their parent call."""
    if recent_user_turns <= 0:
        raise ValueError("recent_user_turns must be greater than zero")

    user_message_indexes = [
        index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    if len(user_message_indexes) <= recent_user_turns:
        return ContextSplit([], list(messages))

    start_index = user_message_indexes[-recent_user_turns]
    return ContextSplit(list(messages[:start_index]), list(messages[start_index:]))
