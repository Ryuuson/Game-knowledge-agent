"""Bound model context while preserving complete recent user turns."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage


def recent_complete_user_turns(
    messages: Sequence[BaseMessage], *, recent_user_turns: int
) -> list[BaseMessage]:
    """Keep a suffix that starts at a user message, retaining tool-call integrity."""
    if recent_user_turns <= 0:
        raise ValueError("recent_user_turns must be greater than zero")
    user_indexes = [
        index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    if len(user_indexes) <= recent_user_turns:
        return list(messages)
    return list(messages[user_indexes[-recent_user_turns] :])
