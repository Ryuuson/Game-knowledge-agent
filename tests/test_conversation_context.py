"""Pure-function tests for context budgeting, turn splitting, and summary cleanup.

None of these tests hit the network or a model API.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation_context import (
    estimate_message_tokens,
    remove_internal_summary,
    split_complete_user_turns,
)


def _full_turn(user_text: str, tool_call_id: str = "call_1") -> list:
    """One complete turn: question -> tool call -> tool result -> reply."""
    return [
        HumanMessage(content=user_text),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_game_knowledge",
                    "args": {"query": user_text},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="检索结果", tool_call_id=tool_call_id),
        AIMessage(content=f"根据资料回答：{user_text}"),
    ]


def test_estimate_message_tokens_counts_cjk_and_ascii():
    # 空消息只算 overhead。
    assert estimate_message_tokens([HumanMessage(content="")]) == 6

    # 10 个汉字：ceil(10 * 1.3) = 13，加 6 条 overhead。
    assert estimate_message_tokens([HumanMessage(content="游" * 10)]) == 19

    # 7 个 ASCII 字符：ceil(7 / 3.5) = 2，加 6 条 overhead。
    assert estimate_message_tokens([HumanMessage(content="a" * 7)]) == 8


def test_estimate_message_tokens_accumulates_across_messages():
    messages = [HumanMessage(content="游" * 10), AIMessage(content="ok")]
    # 汉字部分 19，ASCII "ok" = ceil(2 / 3.5) = 1 + 6 = 7，合计 26。
    assert estimate_message_tokens(messages) == 26


def test_split_keeps_all_messages_within_recent_turns():
    messages = _full_turn("问题一") + _full_turn("问题二")
    split = split_complete_user_turns(messages, recent_user_turns=4)
    assert split.messages_to_summarize == []
    assert split.recent_messages == messages


def test_split_cuts_at_the_last_recent_user_turn_boundary():
    messages = _full_turn("问题一") + _full_turn("问题二") + _full_turn("问题三")
    split = split_complete_user_turns(messages, recent_user_turns=2)
    # 保留最近两个完整回合（问题二、问题三），前面整段进入摘要区。
    assert len(split.messages_to_summarize) == 4
    assert split.recent_messages[0] == messages[4]
    assert isinstance(split.recent_messages[0], HumanMessage)
    assert split.recent_messages == messages[4:]


def test_split_never_splits_a_tool_message_from_its_parent_turn():
    messages = _full_turn("问题一") + _full_turn("问题二") + _full_turn("问题三")
    split = split_complete_user_turns(messages, recent_user_turns=1)
    # recent 的第一条必须是 HumanMessage，其后的 tool/AI 消息都跟着保留。
    assert isinstance(split.recent_messages[0], HumanMessage)
    assert split.recent_messages[0].content == "问题三"
    assert len(split.recent_messages) == 4


def test_split_rejects_non_positive_recent_turns():
    messages = _full_turn("问题一")
    with pytest.raises(ValueError):
        split_complete_user_turns(messages, recent_user_turns=0)
    with pytest.raises(ValueError):
        split_complete_user_turns(messages, recent_user_turns=-1)


def test_remove_internal_summary_removes_verbatim_echo():
    summary = "用户目标是设计经济系统。"
    text = f"根据资料回答。\n{summary}"
    assert remove_internal_summary(text, summary) == "根据资料回答。"


def test_remove_internal_summary_keeps_text_without_echo():
    summary = "用户目标是设计经济系统。"
    text = "根据资料回答。"
    assert remove_internal_summary(text, summary) == text


def test_remove_internal_summary_ignores_empty_summary():
    assert remove_internal_summary("原样返回。", "") == "原样返回。"
