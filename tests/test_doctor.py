"""Diagnostics must work without application imports, network or private reads."""

import builtins
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts import doctor


def make_index(root, rows, backend="bge"):
    path = root / doctor.INDEX_FILES[backend]
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE chunks (embedding_model TEXT, embedding_dimensions INTEGER, content TEXT)"
        )
        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?)", rows)
    return path


def test_missing_resources_and_dependencies_are_reported_without_creating_files(tmp_path, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.importlib.metadata, "version", missing)
    report = doctor.collect_report(tmp_path, environ={})
    assert report["configuration"]["env_file_status"] == "missing"
    assert report["configuration"]["rag_backend"] == "bge"
    assert report["configuration"]["llm_key_configured"] is False
    assert all(item == {"installed": False, "version": None}
               for item in report["dependencies"].values())
    assert all(index["status"] == "missing" for index in report["indexes"].values())
    assert list(tmp_path.iterdir()) == []
    json.dumps(report)


def test_configuration_is_presence_only_and_environment_takes_precedence(tmp_path):
    secret = "sk-never-print-this-secret"
    (tmp_path / ".env").write_text(
        f'export ARK_API_KEY="{secret}" # legacy key\n'
        "LLM_API_KEY=dotenv-secret\n"
        "LLM_MODEL='private-model-id'\n"
        "LLM_BASE_URL=https://user:password@example.invalid/v1\n"
        "IGNORED_SECRET=unrelated-private-data\n"
        "RAG_BACKEND=ark\n",
        encoding="utf-8",
    )
    before = dict(__import__("os").environ)
    report = doctor.collect_report(tmp_path, environ={"LLM_API_KEY": "", "RAG_BACKEND": "bge"})
    assert report["configuration"]["llm_key_configured"] is True
    assert report["configuration"]["variables"]["LLM_API_KEY"] == {
        "configured": False, "source": "environment",
    }
    assert report["configuration"]["llm_model_configured"] is True
    assert report["configuration"]["rag_backend"] == "bge"
    serialized = json.dumps(report)
    for value in (secret, "dotenv-secret", "private-model-id", "password", "unrelated-private-data"):
        assert value not in serialized
    assert dict(__import__("os").environ) == before


@pytest.mark.parametrize("value", ["", "your_api_key", "offline-placeholder", "${PRIVATE_KEY}"])
def test_empty_and_placeholder_keys_are_not_configured(tmp_path, value):
    report = doctor.collect_report(tmp_path, environ={"LLM_API_KEY": value})
    assert report["configuration"]["llm_key_configured"] is False


def test_invalid_dotenv_and_backend_do_not_echo_values(tmp_path):
    (tmp_path / ".env").write_bytes(b"LLM_API_KEY=\xff")
    report = doctor.collect_report(tmp_path, environ={"RAG_BACKEND": "secret-invalid-backend"})
    assert report["configuration"]["env_file_status"] == "unreadable"
    assert report["configuration"]["rag_backend"] == "invalid"
    assert "secret-invalid-backend" not in json.dumps(report)


def test_index_metadata_is_aggregated_read_only_without_reading_private_columns(tmp_path, monkeypatch):
    model = "BAAI/bge-small-zh-v1.5"
    path = make_index(tmp_path, [(model, 512, "private document") for _ in range(3)])
    before = path.read_bytes()
    original_connect = sqlite3.connect
    queried = []

    def guarded_connect(database, *args, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        queried.append(database)
        connection = original_connect(database, *args, **kwargs)

        def authorize(action, arg1, arg2, database_name, trigger):
            if action == sqlite3.SQLITE_READ and arg1 == "chunks" and arg2 == "content":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(doctor.sqlite3, "connect", guarded_connect)
    report = doctor.collect_report(tmp_path, environ={})
    assert len(queried) == 1
    assert report["indexes"]["bge"] == {
        "file": doctor.INDEX_FILES["bge"], "status": "ready", "chunk_count": 3,
        "models": [model], "dimensions": [512], "matches_configured_model": True,
    }
    assert "private document" not in json.dumps(report)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*-*"))


@pytest.mark.parametrize("rows,status,dimensions", [
    ([], "empty", []),
    ([("model-a", 2, ""), ("model-b", 3, "")], "inconsistent", [2, 3]),
    ([("model-a", -1, "")], "inconsistent", []),
    ([(None, 2, "")], "inconsistent", [2]),
    ([("different-model", 2, "")], "ready", [2]),
])
def test_index_empty_mixed_invalid_and_model_mismatch(tmp_path, rows, status, dimensions):
    make_index(tmp_path, rows)
    index = doctor.collect_report(tmp_path, environ={})["indexes"]["bge"]
    assert index["status"] == status
    assert index["chunk_count"] == len(rows)
    assert index["dimensions"] == dimensions
    assert index["matches_configured_model"] is (False if rows else None)


def test_ark_model_matching_and_secret_model_redaction(tmp_path):
    secret = "private-api-token"
    make_index(tmp_path, [(f"ark:{secret}", 1024, "")], backend="ark")
    report = doctor.collect_report(tmp_path, environ={
        "ARK_API_KEY": secret, "ARK_EMBEDDING_MODEL": secret, "RAG_BACKEND": "ark",
    })
    assert report["indexes"]["ark"]["models"] == ["[redacted]"]
    assert report["indexes"]["ark"]["matches_configured_model"] is True
    assert secret not in json.dumps(report)


@pytest.mark.parametrize("kind", ["corrupt", "wrong_schema", "view"])
def test_invalid_indexes_are_reported_without_exceptions(tmp_path, kind):
    path = tmp_path / doctor.INDEX_FILES["bge"]
    path.parent.mkdir()
    if kind == "corrupt":
        path.write_bytes(b"private error payload")
    else:
        with sqlite3.connect(path) as connection:
            if kind == "view":
                connection.execute("CREATE TABLE conversations(secret TEXT)")
                connection.execute("CREATE VIEW chunks AS SELECT secret FROM conversations")
            else:
                connection.execute("CREATE TABLE chunks(content TEXT)")
    index = doctor.collect_report(tmp_path, environ={})["indexes"]["bge"]
    assert index["status"] == ("unreadable" if kind == "corrupt" else "invalid_schema")
    assert "private error payload" not in json.dumps(index)


def test_does_not_import_agent_or_open_corpus_conversation_and_other_private_files(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / doctor.CORPUS_FILES[0]).write_text("private corpus text", encoding="utf-8")
    (tmp_path / doctor.CORPUS_FILES[1]).touch()
    (tmp_path / doctor.CORPUS_DIRECTORIES[0]).mkdir(parents=True)
    (tmp_path / "agent_memory.sqlite").write_bytes(b"private conversation")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "private.md").write_text("private notes", encoding="utf-8")
    original_open = Path.open
    original_import = builtins.__import__

    def guarded_open(path, *args, **kwargs):
        # Installed package METADATA is legitimate diagnostic input. Guard only
        # files inside this fixture's project, where private content is placed.
        if path.is_relative_to(tmp_path):
            assert path == tmp_path / ".env", "doctor must not open corpus/private files"
        return original_open(path, *args, **kwargs)

    def guarded_import(name, *args, **kwargs):
        assert name not in {"Agent", "sentence_transformers", "torch"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report = doctor.collect_report(tmp_path, environ={})
    assert [item["status"] for item in report["corpus"]["files"]] == ["present", "empty"]
    assert report["corpus"]["directories"][0]["status"] == "present"
    assert "private" not in json.dumps(report)


def test_redirected_index_is_not_opened(tmp_path, monkeypatch):
    make_index(tmp_path, [("model", 2, "")])
    original_resolve = Path.resolve
    redirected = tmp_path / doctor.INDEX_FILES["bge"]

    def resolve(path, *args, **kwargs):
        if path == redirected:
            return tmp_path / "agent_memory.sqlite"
        return original_resolve(path, *args, **kwargs)

    def no_connect(*args, **kwargs):
        pytest.fail("redirected index must not be opened")

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(doctor.sqlite3, "connect", no_connect)
    assert doctor.collect_report(tmp_path, environ={})["indexes"]["bge"]["status"] == "redirected"


@pytest.mark.parametrize("flags", [[], ["--json"]])
def test_cli_emits_json_using_stdlib_only_outside_project(tmp_path, flags):
    script = Path(doctor.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--project-root", str(tmp_path), *flags],
        cwd=tmp_path, env={**{name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH") if name in os.environ},
                          "LLM_API_KEY": "sk-subprocess-private-key"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "sk-subprocess-private-key" not in result.stdout
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["configuration"]["llm_key_configured"] is True
