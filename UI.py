"""Local Streamlit application: grounded chat and an API-free retrieval demo."""

from importlib import import_module
from time import perf_counter
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from conversation_store import ConversationStore, title_from_first_prompt
from game_agent.presentation import export_markdown, visible_history
from game_agent.settings import ROOT, Settings
from wiki_corpus.retrieval import RetrievalService
from wiki_corpus.provenance import describe_location

load_dotenv(ROOT / ".env")
settings = Settings.from_env()
st.set_page_config(page_title="游戏知识助手", page_icon="🎮", layout="wide")


@st.cache_resource
def runtime():
    return import_module("Agent")


@st.cache_resource
def retriever(mode):
    return RetrievalService(mode=mode)


@st.cache_resource
def store():
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return ConversationStore(settings.database_path)


def new_chat():
    st.session_state.thread_id = f"web_{uuid4().hex}"
    st.session_state.messages = []


def render_evidence(retrievals):
    if not retrievals:
        return
    with st.expander("查看检索依据"):
        for result in retrievals:
            label = {"bge": "语义检索", "keyword": "关键词检索", "hybrid": "混合检索"}.get(result["mode"], result["mode"])
            status = {"high_confidence": "高相关", "ambiguous": "需要核对语境", "keyword_only": "关键词候选",
                      "no_evidence": "无可用依据", "out_of_domain": "非游戏领域"}.get(result["status"], result["status"])
            st.caption(f"{label} · {status} · {result.get('elapsed_ms', 0):.0f} ms")
            if result.get("reused_from_prior_turn"):
                st.caption("本次回答引用了此前对话中检索到的资料。")
            if result.get("fallback"):
                st.info("语义服务暂不可用，以下为关键词候选。")
            for hit in result.get("hits", []):
                st.markdown(f"**{hit.get('title', '未命名资料')}**")
                st.caption(f"来源编号：{hit['chunk_id']} · {hit.get('collection_label', '')} · {hit.get('section_path') or '文章开头'}")
                source = str(hit.get("source_url", ""))
                if source.startswith(("https://", "http://")):
                    st.link_button("打开原始资料", source)
                else:
                    st.caption(f"资料标识：{source}")
                st.caption(describe_location(hit))
                st.text(str(hit.get("text", ""))[:2400])


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_evidence(message.get("retrievals", []))
        usage = message.get("usage", {})
        if usage.get("total_tokens"):
            st.caption(f"最终回答请求使用 {usage['total_tokens']} tokens（不含工具前轮及摘要请求）")


if "thread_id" not in st.session_state:
    new_chat()

st.title("🎮 游戏知识助手")
st.caption("从游戏设计资料中查找依据，讨论机制、数值、经济系统与制作流程。")

with st.sidebar:
    mode = st.radio("使用方式", ["对话助手", "离线检索演示"], index=1)
    st.caption("本地单用户应用。对话会调用已配置的聊天模型；离线演示只检索本机资料。")
    if mode == "对话助手":
        if st.button("新建会话", use_container_width=True):
            new_chat()
            st.rerun()
        for conversation in store().list_conversations():
            open_col, delete_col = st.columns([5, 1])
            if open_col.button(conversation.title, key=f"open_{conversation.thread_id}", use_container_width=True,
                               type="primary" if conversation.thread_id == st.session_state.thread_id else "secondary"):
                app = runtime()
                snapshot = app.graph.get_state(app.create_config(conversation.thread_id))
                st.session_state.thread_id = conversation.thread_id
                st.session_state.messages = visible_history(snapshot.values.get("messages", []))
                st.rerun()
            if delete_col.button("删除", key=f"delete_{conversation.thread_id}"):
                store().delete(conversation.thread_id)
                if conversation.thread_id == st.session_state.thread_id:
                    new_chat()
                st.rerun()
        if st.session_state.messages:
            st.download_button("导出对话与来源", export_markdown(st.session_state.messages),
                               file_name="game-knowledge-conversation.md", mime="text/markdown")

if mode == "离线检索演示":
    st.info("此模式不调用聊天模型。下面展示检索到的原始片段，不生成设计结论。")
    search_mode = st.selectbox("检索方式", ["bge", "keyword", "hybrid"],
                              format_func=lambda x: {"bge": "BGE 语义检索", "keyword": "BM25 关键词检索", "hybrid": "BGE + BM25 混合检索"}[x])
    query = st.text_input("游戏设计问题", value="玩家在游戏后期金币越来越多，经济系统应该怎么设计回收机制？")
    top_k = st.slider("展示片段数", 1, 10, 3)
    if st.button("检索本地知识库", type="primary"):
        try:
            with st.spinner("正在检索本地资料…"):
                result = retriever(search_mode).search(query, top_k=top_k)
            st.session_state.demo_result = result.to_dict()
        except Exception:
            st.session_state.pop("demo_result", None)
            st.error("无法检索，请运行 scripts/doctor.py 检查语料、索引和本地模型。")
    if "demo_result" in st.session_state:
        result = st.session_state.demo_result
        st.caption(f"问题：{result['query']}")
        if result["status"] == "out_of_domain":
            st.warning("该问题明确属于其他行业，本地游戏资料不适用。")
        elif not result["hits"]:
            st.warning("没有找到足够相关的资料，请补充游戏语境或调整问题。")
        render_evidence([result])
else:
    if not settings.chat_ready:
        st.warning("聊天模型尚未配置。请按 README 配置 LLM_*，或切换到离线检索演示。")
    for message in st.session_state.messages:
        render_message(message)

    if prompt := st.chat_input("例如：技能伤害和冷却如何搭配，才不容易出现唯一解？", disabled=not settings.chat_ready):
        if len(prompt) > settings.max_input_chars:
            st.error("问题超过 6000 字符，请缩短后发送。")
            st.stop()
        app = runtime()
        from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

        thread_id = st.session_state.thread_id
        if not st.session_state.messages:
            store().register(thread_id, title_from_first_prompt(prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            reply_box, status_box = st.empty(), st.empty()
            pending_text = ""
            tool_turn = False
            started = perf_counter()
            try:
                for chunk, metadata in app.graph.stream({"messages": [HumanMessage(content=prompt)]},
                                                        config=app.create_config(thread_id), stream_mode="messages"):
                    if "internal_summary" in metadata.get("tags", []):
                        continue
                    if isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                        tool_turn = True
                        pending_text = ""
                        reply_box.empty()
                        for call in chunk.tool_call_chunks:
                            if call.get("name"):
                                status_box.info(app.TOOL_START_MSGS.get(call["name"], app.DEFAULT_START))
                    elif isinstance(chunk, ToolMessage):
                        tool_turn = False
                        status_box.info(app.TOOL_END_MSGS.get(chunk.name, app.DEFAULT_END))
                    elif isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and not tool_turn:
                        pending_text += chunk.content
                        reply_box.markdown(pending_text)
                snapshot = app.graph.get_state(app.create_config(thread_id))
                st.session_state.messages = visible_history(snapshot.values.get("messages", []))
                final = st.session_state.messages[-1] if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" else None
                if final is None:
                    raise RuntimeError("empty_reply")
                reply_box.markdown(final["content"])
                status_box.empty()
                render_evidence(final.get("retrievals", []))
                st.caption(f"本轮耗时 {perf_counter() - started:.1f} 秒")
                store().touch(thread_id)
            except Exception:
                reply_box.empty()
                status_box.empty()
                # Recover the persisted user turn, so the next submit cannot
                # overwrite the title or hide the failed request in the UI.
                snapshot = app.graph.get_state(app.create_config(thread_id))
                st.session_state.messages = visible_history(snapshot.values.get("messages", []))
                st.error("本轮未完成。请检查模型配置、网络或运行 scripts/doctor.py 后重试；已有历史仍保留。")
