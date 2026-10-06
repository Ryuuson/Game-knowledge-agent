"""All tests use dummy credentials, no telemetry and a disposable database."""

import os
import socket
import tempfile
from pathlib import Path

import pytest

_directory = tempfile.TemporaryDirectory(prefix="game-agent-tests-")
os.environ["GAME_AGENT_DB"] = str(Path(_directory.name) / "checkpoints.sqlite")
for name, value in {
    "LLM_API_KEY": "offline-placeholder", "LLM_BASE_URL": "http://127.0.0.1:9/v1", "LLM_MODEL": "offline-test",
    "VISION_API_KEY": "offline-placeholder", "VISION_BASE_URL": "http://127.0.0.1:9/v1", "VISION_MODEL": "offline-test",
    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
}.items():
    os.environ[name] = value


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Unit tests must not connect to the network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def pytest_sessionfinish(session, exitstatus):
    import sys
    agent = sys.modules.get("Agent")
    if agent is not None:
        agent.connection.close()
    _directory.cleanup()
