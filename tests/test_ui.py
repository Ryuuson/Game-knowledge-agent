"""Exercise Streamlit controls with actual corpus retrieval and zero API calls."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_offline_demo_queries_and_rejects_other_domains():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "UI.py"), default_timeout=30).run()
    assert not app.exception
    assert app.radio[0].value == "离线检索演示"
    app.selectbox[0].select("keyword").run()
    app.text_input[0].set_value("游戏金币回收").run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["demo_result"]
    assert result["hits"] and result["status"] == "keyword_only"
    assert result["hits"][0]["chunk_id"]
    app.text_input[0].set_value("医院预约挂号流程").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["demo_result"]["status"] == "out_of_domain"
    assert "其他行业" in app.warning[0].value


def test_chat_controls_restore_and_delete_conversation(monkeypatch):
    import Agent
    from langchain_core.messages import AIMessage

    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content="这是离线测试生成的回复，用于验证会话控件。")

    monkeypatch.setattr(Agent, "model_with_tools", FakeModel())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "UI.py"), default_timeout=30).run()
    app.radio[0].set_value("对话助手").run()
    app.chat_input[0].set_value("测试会话一").run()
    assert not app.exception
    assert app.session_state["messages"][-1]["content"] == "这是离线测试生成的回复，用于验证会话控件。"
    old_id = app.session_state["thread_id"]
    next(button for button in app.button if button.label == "新建会话").click().run()
    assert app.session_state["thread_id"] != old_id and app.session_state["messages"] == []
    next(button for button in app.button if button.label == "测试会话一").click().run()
    assert app.session_state["thread_id"] == old_id
    assert len(app.session_state["messages"]) == 2
    next(button for button in app.button if button.key == f"delete_{old_id}").click().run()
    assert app.session_state["thread_id"] != old_id and app.session_state["messages"] == []
