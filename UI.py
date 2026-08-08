import streamlit as st
from Agent import graph, create_config, TOOL_START_MSGS, TOOL_END_MSGS, DEFAULT_START, DEFAULT_END
from conversation_store import (
    ConversationStore,
    LEGACY_TITLE,
    title_from_first_prompt,
)
from conversation_context import remove_internal_summary
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk, ToolMessage
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="游戏知识 Agent", page_icon="🎮")
st.title("🎮 游戏知识 Agent")
st.caption("基于游戏设计知识库的智能顾问")


def _new_thread_id() -> str:
    return f"web_{datetime.now().timestamp()}"


def _restore_messages(thread_id: str) -> list[dict[str, str]]:
    """Read the visible chat messages from LangGraph's persisted state."""
    state = graph.get_state(create_config(thread_id))
    restored = []
    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            restored.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content.strip():
                restored.append({"role": "assistant", "content": message.content})
    return restored


def _start_new_conversation() -> None:
    st.session_state.thread_id = _new_thread_id()
    st.session_state.messages = []


conversation_store = ConversationStore(Path(__file__).with_name("agent_memory.sqlite"))

# ---------- session state 初始化 ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = _new_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 当前会话的 config
config = create_config(st.session_state.thread_id)

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("### 游戏设计知识库")
    st.caption("检索游戏机制、系统设计与设计方法。")
    st.divider()
    st.markdown("### 会话管理")
    if st.button("新建会话", use_container_width=True):
        _start_new_conversation()
        st.rerun()

    conversations = conversation_store.list_conversations()
    for conversation in conversations:
        if conversation.title != LEGACY_TITLE:
            continue
        restored = _restore_messages(conversation.thread_id)
        first_prompt = next(
            (message["content"] for message in restored if message["role"] == "user"),
            "",
        )
        if first_prompt:
            conversation_store.rename(
                conversation.thread_id, title_from_first_prompt(first_prompt)
            )
    conversations = conversation_store.list_conversations()
    if conversations:
        st.caption("历史会话")
        for conversation in conversations:
            open_column, delete_column = st.columns([5, 1])
            is_active = conversation.thread_id == st.session_state.thread_id
            button_type = "primary" if is_active else "secondary"
            if open_column.button(
                conversation.title,
                key=f"open_{conversation.thread_id}",
                type=button_type,
                use_container_width=True,
            ):
                restored = _restore_messages(conversation.thread_id)
                st.session_state.thread_id = conversation.thread_id
                st.session_state.messages = restored
                st.rerun()
            if delete_column.button("删除", key=f"delete_{conversation.thread_id}"):
                conversation_store.delete(conversation.thread_id)
                if is_active:
                    _start_new_conversation()
                st.rerun()
    else:
        st.caption("还没有历史会话")

# ---------- 渲染历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 用户输入 ----------
if prompt := st.chat_input("例如：如何设计游戏的经济系统？"):
    if not st.session_state.messages:
        conversation_store.register(
            st.session_state.thread_id, title_from_first_prompt(prompt)
        )
    else:
        conversation_store.touch(st.session_state.thread_id)

    # 记录并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备调用图
    input_state = {"messages": [HumanMessage(content=prompt)]}
    summary_before = conversation_store.get_summary(st.session_state.thread_id)
    covered_before = summary_before.covered_message_count if summary_before else 0

    with st.chat_message("assistant"):
        reply_box = st.empty()
        status_box = st.empty()
        full_text = ""
        shown_tool_calls = set()
        active_tool_calls = {}

        # 流式执行
        for chunk, metadata in graph.stream(
            input_state,
            config=config,
            stream_mode="messages"
        ):
            # 工具调用开始
            if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name", "")
                    if tc_id and tc_id not in shown_tool_calls:
                        shown_tool_calls.add(tc_id)
                        active_tool_calls[tc_id] = tc_name or "未知"
                        start_msg = TOOL_START_MSGS.get(tc_name, DEFAULT_START)
                        status_box.info(f"🔧 {start_msg}")

            # 工具调用结束
            if isinstance(chunk, ToolMessage):
                tc_id = chunk.tool_call_id
                if tc_id in active_tool_calls:
                    tool_name = active_tool_calls.pop(tc_id)
                    end_msg = TOOL_END_MSGS.get(tool_name, DEFAULT_END)
                    status_box.success(f"✨ {end_msg}")

            # 收集模型文本；结束后统一脱敏再显示。
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                full_text += chunk.content

        # 模型若意外复述内部摘要，绝不把它显示或保存到聊天记录。
        stored_summary = conversation_store.get_summary(st.session_state.thread_id)
        summary_text = stored_summary.summary if stored_summary else ""
        full_text = remove_internal_summary(full_text, summary_text)
        reply_box.markdown(full_text)
        status_box.empty()  # 清除工具状态
        if stored_summary and stored_summary.covered_message_count > covered_before:
            st.caption("本轮已压缩较早的对话上下文，完整历史仍保留在此会话中。")

    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": full_text})
    conversation_store.touch(st.session_state.thread_id)
