"""Read-only, offline diagnostics. Uses only stdlib and never imports Agent.

Both the default output and --json are JSON. Missing optional resources are
reported as statuses, not command failures. Configuration values, document
contents, vector payloads and conversation databases are never printed/read.
The small .env parser handles literal assignments only, without interpolation.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys


DEPENDENCIES = (
    "langgraph", "langgraph-checkpoint-sqlite", "langchain-openai", "requests",
    "python-dotenv", "numpy", "streamlit", "pytest", "sentence-transformers", "torch",
)
CONFIG_NAMES = (
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "ARK_API_KEY", "ARK_BASE_URL",
    "ARK_MODEL", "ARK_EMBEDDING_MODEL", "VISION_API_KEY", "VISION_BASE_URL",
    "VISION_MODEL", "ARK_VISION_MODEL", "METASO_API_KEY", "RAG_BACKEND", "GAME_AGENT_DB",
)
INDEX_FILES = {
    "bge": "data/game_knowledge_bge_combined_index.sqlite",
    "ark": "data/game_knowledge_combined_index.sqlite",
}
CORPUS_FILES = (
    "data/game_knowledge_chunks.jsonl", "data/new_knowledge_chunks.jsonl",
)
# Fixed public corpus roots; do not walk private directories or list filenames.
CORPUS_DIRECTORIES = (
    "knowledge/game-design-wiki", "knowledge/Game-Knowledge-Base",
    "knowledge/open-game-mechanics-dataset", "knowledge/Game_Num_Basics_And_Calc",
    "knowledge/gamedev_at_home", "knowledge/senior-game-designer",
)
PLACEHOLDERS = {"your_api_key", "your_chat_model", "your_key", "offline-placeholder"}


def _is_direct_path(root, relative):
    """Refuse redirected files/parents, including links to a conversation DB."""
    path = root / relative
    return path.resolve() == root.resolve() / relative


def _read_settings(root, environ):
    values = {}
    sources = {}
    env_file = root / ".env"
    file_status = "missing"
    try:
        if not _is_direct_path(root, ".env"):
            file_status = "redirected"
        elif env_file.is_file():
            file_status = "present"
            with env_file.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    match = re.match(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
                    if not match or match[1] not in CONFIG_NAMES:
                        continue
                    value = match[2]
                    if value.startswith(("'", '"')):
                        quoted = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", value)
                        if not quoted:
                            continue
                        value = quoted[2]
                    else:
                        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
                    values[match[1]] = value
                    sources[match[1]] = "dotenv"
    except (OSError, UnicodeError):
        # Never include exception text: it may contain paths or file contents.
        file_status = "unreadable"
    for name in CONFIG_NAMES:
        if name in environ:
            values[name] = environ[name]
            sources[name] = "environment"
    return values, sources, file_status


def _configured(value):
    return bool(value and value.strip() and value.strip().lower() not in PLACEHOLDERS
                and "${" not in value and not value.startswith("https://your-"))


def _safe_model(value, secrets):
    if not isinstance(value, str):
        return "[invalid]"
    if (any(secret in value for secret in secrets)
            or value.lower().startswith(("sk-", "bearer "))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}", value)):
        return "[redacted]"
    return value


def inspect_index(root, relative, expected_model, secrets=()):
    """Query only table schema and aggregate embedding metadata, in mode=ro."""
    result = {"file": relative, "status": "missing", "chunk_count": None,
              "models": [], "dimensions": [], "matches_configured_model": None}
    path = root / relative
    try:
        if not _is_direct_path(root, relative):
            result["status"] = "redirected"
            return result
        if not path.is_file():
            return result
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            connection.execute("PRAGMA query_only = ON")
            schema = connection.execute(
                "SELECT type FROM sqlite_master WHERE name = 'chunks'"
            ).fetchone()
            if not schema or schema[0] != "table":
                result["status"] = "invalid_schema"
                return result
            columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
            if not {"embedding_model", "embedding_dimensions"} <= columns:
                result["status"] = "invalid_schema"
                return result
            rows = connection.execute(
                "SELECT embedding_model, embedding_dimensions, COUNT(*) "
                "FROM chunks GROUP BY embedding_model, embedding_dimensions"
            ).fetchall()
        finally:
            connection.close()
        count = sum(row[2] for row in rows)
        models = {row[0] for row in rows}
        dimensions = {row[1] for row in rows}
        valid_dimensions = all(type(value) is int and value > 0 for value in dimensions)
        result.update(
            chunk_count=count,
            models=sorted({_safe_model(value, secrets) for value in models}),
            dimensions=sorted(value for value in dimensions if type(value) is int and value > 0),
            matches_configured_model=(models == {expected_model}) if count and expected_model else None,
            status=("empty" if not count else "ready" if len(models) == len(dimensions) == 1
                    and valid_dimensions and None not in models else "inconsistent"),
        )
    except (OSError, sqlite3.Error, ValueError):
        result["status"] = "unreadable"
    return result


def _corpus_status(root, relative, directory=False):
    result = {"path": relative, "status": "missing"}
    try:
        if not _is_direct_path(root, relative):
            result["status"] = "redirected"
        elif directory and (root / relative).is_dir():
            result["status"] = "present"
        elif not directory and (root / relative).is_file():
            size = (root / relative).stat().st_size
            result.update(status="present" if size else "empty", bytes=size)
    except OSError:
        result["status"] = "unreadable"
    return result


def collect_report(project_root=None, environ=None):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    environ = os.environ if environ is None else environ
    values, sources, env_status = _read_settings(root, environ)
    backend = values.get("RAG_BACKEND", "bge").lower()
    backend = backend if backend in INDEX_FILES else "invalid"
    secrets = tuple(value for name, value in values.items() if name.endswith("API_KEY") and value)
    dependencies = {}
    for name in DEPENDENCIES:
        try:
            dependencies[name] = {"installed": True, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = {"installed": False, "version": None}
    ark_model = values.get("ARK_EMBEDDING_MODEL")
    expected = {"bge": "BAAI/bge-small-zh-v1.5",
                "ark": f"ark:{ark_model}" if _configured(ark_model) else None}
    return {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "platform": sys.platform,
                        "python_310_or_newer": sys.version_info >= (3, 10),
                        "virtualenv": sys.prefix != sys.base_prefix},
        "dependencies": dependencies,
        "configuration": {
            "env_file_status": env_status,
            "variables": {name: {"configured": _configured(values.get(name)),
                                 "source": sources.get(name, "missing")} for name in CONFIG_NAMES},
            "llm_key_configured": any(_configured(values.get(name)) for name in ("LLM_API_KEY", "ARK_API_KEY")),
            "llm_model_configured": any(_configured(values.get(name)) for name in ("LLM_MODEL", "ARK_MODEL")),
            "llm_base_url_configured": any(_configured(values.get(name)) for name in ("LLM_BASE_URL", "ARK_BASE_URL")),
            "rag_backend": backend,
        },
        "indexes": {name: inspect_index(root, file, expected[name], secrets)
                    for name, file in INDEX_FILES.items()},
        "corpus": {
            "assessment": "existence_and_size_only",
            "files": [_corpus_status(root, file) for file in CORPUS_FILES],
            "directories": [_corpus_status(root, directory, directory=True)
                            for directory in CORPUS_DIRECTORIES],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON (also the default)")
    parser.add_argument("--project-root", type=Path, help="inspect another project root")
    arguments = parser.parse_args(argv)
    print(json.dumps(collect_report(arguments.project_root), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
