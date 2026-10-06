"""An offline graph still must persist artifacts, isolate threads and bound input."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import Agent
from conversation_context import fit_model_context, estimate_message_tokens
from conversation_store import ConversationStore
from game_agent.presentation import validate_citations, visible_history, export_markdown
from game_agent.settings import Settings, ROOT
from wiki_corpus.retrieval import RetrievalResult


def test_settings_blank_database_uses_default_and_repr_hides_key(monkeypatch):
    monkeypatch.setenv("GAME_AGENT_DB", "")
    monkeypatch.setenv("LLM_API_KEY", "sensitive-test-key")
    settings = Settings.from_env()
    assert settings.database_path == ROOT / "agent_memory.sqlite"
    assert "sensitive-test-key" not in repr(settings)


def test_graph_persists_real_artifacts_and_restores_after_reopening(tmp_path, monkeypatch):
    class FakeRetriever:
        def search(self, query, **kwargs):
            return RetrievalResult(query, "high_confidence", "bge", hits=[
                {"chunk_id": "real-source", "citation_id": "real-source", "title": "游戏经济", "text": "设置金币消耗", "score": .8}])

    class FakeModel:
        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="设计金币消耗 [来源:real-source]，不能使用 [来源:invented]。")
            return AIMessage(content="内部检索前言", tool_calls=[
                {"name": "search_game_knowledge", "args": {"query": "游戏金币回收"}, "id": "call-1", "type": "tool_call"}])

    monkeypatch.setattr(Agent, "_retrieval_service", FakeRetriever())
    monkeypatch.setattr(Agent, "model_with_tools", FakeModel())
    monkeypatch.setattr(Agent, "settings", Settings(api_key="test", base_url="http://invalid", model="test"))
    monkeypatch.setattr(Agent, "context_store", ConversationStore(tmp_path / "catalog.sqlite"))
    path = tmp_path / "checkpoints.sqlite"
    connection = sqlite3.connect(path, check_same_thread=False)
    graph = Agent.builder.compile(checkpointer=SqliteSaver(connection))
    config = Agent.create_config("web_graph_test")
    graph.invoke({"messages": [HumanMessage(content="游戏金币怎么回收？")]}, config)
    result = graph.get_state(config).values["messages"]
    visible = visible_history(result)
    assert len(visible) == 2  # intermediate tool-calling AI text is not a reply
    assert visible[-1]["retrievals"][0]["hits"][0]["chunk_id"] == "real-source"
    assert "[来源:invented]" not in visible[-1]["content"]
    assert "[来源:real-source]" in visible[-1]["content"]
    assert "real-source" in export_markdown(visible)
    assert graph.get_state(Agent.create_config("other_thread")).values == {}
    connection.close()
    connection = sqlite3.connect(path, check_same_thread=False)
    restarted = Agent.builder.compile(checkpointer=SqliteSaver(connection))
    assert visible_history(restarted.get_state(config).values["messages"]) == visible
    connection.close()


def test_summary_failure_keeps_history_but_model_context_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(Agent, "context_store", ConversationStore(tmp_path / "catalog.sqlite"))
    monkeypatch.setattr(Agent, "CONTEXT_TOKEN_BUDGET", 4000)
    monkeypatch.setattr(Agent, "RECENT_USER_TURNS", 1)
    def failed(*args):
        raise RuntimeError("simulated provider outage")
    monkeypatch.setattr(Agent, "_summarize_messages", failed)
    messages = [HumanMessage(content="旧问题" * 3000), AIMessage(content="旧答案"),
                HumanMessage(content="新问题"), AIMessage(content="新答案")]
    summary, context = Agent._model_context(messages, "thread")
    assert summary == "" and context == messages[-2:]
    assert len(messages[0].content) == 9000


def test_context_truncates_tool_text_without_losing_call_ids_or_mutating_history():
    messages = [HumanMessage(content="游戏经济"), AIMessage(content="", tool_calls=[
        {"name": "search", "args": {}, "id": "x", "type": "tool_call"}]),
        ToolMessage(content="资料" * 6000, tool_call_id="x")]
    fixed = [SystemMessage(content="规则")]
    bounded = fit_model_context(messages, fixed=fixed, token_budget=2000)
    assert estimate_message_tokens(bounded) <= 2000
    assert bounded[-1].tool_call_id == "x" and len(messages[-1].content) == 12000
    with pytest.raises(ValueError):
        fit_model_context([HumanMessage(content="长问题" * 10000)], fixed=fixed, token_budget=2000)


def test_tool_loop_limit_returns_without_calling_model(monkeypatch):
    monkeypatch.setattr(Agent, "settings", Settings(api_key="test", base_url="http://invalid", model="test", max_tool_rounds=1))
    messages = [HumanMessage(content="问题"), AIMessage(content="", tool_calls=[
        {"name": "search", "args": {}, "id": "x", "type": "tool_call"}]),
        ToolMessage(content="片段", tool_call_id="x")]
    result = Agent.call_model({"messages": messages}, Agent.create_config("limit"))
    assert "工具调用上限" in result["messages"][0].content


def test_followup_can_reuse_real_previous_evidence_without_new_tool_calls():
    artifact = {"mode": "bge", "status": "high_confidence", "hits": [
        {"chunk_id": "previous", "title": "技能", "text": "原文"}]}
    messages = [HumanMessage(content="技能伤害"),
                ToolMessage(content="[来源:previous] 原文", tool_call_id="x", artifact=artifact),
                AIMessage(content="解释 [来源:previous]"), HumanMessage(content="新手呢？"),
                AIMessage(content="延续分析 [来源:previous]")]
    text, unknown = validate_citations(messages[-1].content, messages[:-1])
    assert not unknown and text == messages[-1].content
    followup = visible_history(messages)[-1]
    assert followup["retrievals"][0]["reused_from_prior_turn"]
    assert followup["retrievals"][0]["hits"][0]["chunk_id"] == "previous"


def test_local_documents_cannot_escape_root_and_saved_notes_do_not_overwrite(tmp_path):
    from game_agent import local_files
    root = tmp_path / "knowledge"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    assert "不允许访问" in local_files.read_document(root, "../secret.txt")
    (root / "doc.md").write_text("第一行\n第二行\n第三行", encoding="utf-8")
    assert "2: 第二行" in local_files.read_section(root, "doc.md", 2, 2)
    assert "1: 第一行" not in local_files.read_section(root, "doc.md", 2, 2)
