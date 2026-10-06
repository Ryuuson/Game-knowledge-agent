"""Prepare bounded model context without changing persisted chat history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage


MESSAGE_OVERHEAD_TOKENS = 6


@dataclass(frozen=True)
class ContextSplit:
    """A complete-turn suffix and the older raw messages it replaces."""

    messages_to_summarize: list[BaseMessage]
    recent_messages: list[BaseMessage]


def _message_text(message: BaseMessage) -> str:
    content = message.content
    text = content if isinstance(content, str) else str(content)
    if getattr(message, "tool_calls", None):
        text += str(message.tool_calls)
    return text


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


def fit_model_context(messages: Sequence[BaseMessage], *, fixed: Sequence[BaseMessage],
                      token_budget: int) -> list[BaseMessage]:
    """Bound estimated input without mutating stored history or splitting tools.

    Discard complete older turns first. Only tool text is shortened; the latest
    user request and matching tool-call IDs are retained. Reject an input that
    still cannot fit rather than send an unbounded request.
    """
    recent = list(messages)
    users = sum(isinstance(m, HumanMessage) for m in recent)
    while estimate_message_tokens([*fixed, *recent]) > token_budget and users > 1:
        users -= 1
        recent = split_complete_user_turns(recent, recent_user_turns=users).recent_messages
    if estimate_message_tokens([*fixed, *recent]) > token_budget:
        for index, message in enumerate(recent):
            if isinstance(message, ToolMessage) and isinstance(message.content, str) and len(message.content) > 1000:
                recent[index] = message.model_copy(update={"content": message.content[:1000] + "\n[工具资料因上下文预算截断]"})
    if estimate_message_tokens([*fixed, *recent]) > token_budget:
        raise ValueError("当前问题与工具参数超出上下文预算，请缩短问题或减少资料。")
    return [*fixed, *recent]
