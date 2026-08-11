"""Persistent conversation directory stored beside LangGraph checkpoints."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TITLE = "新对话"
LEGACY_TITLE = "历史会话"


@dataclass(frozen=True)
class Conversation:
    thread_id: str
    title: str
    updated_at: str


@dataclass(frozen=True)
class ConversationSummary:
    thread_id: str
    summary: str
    covered_message_count: int
    updated_at: str


def title_from_first_prompt(prompt: str, *, maximum_length: int = 32) -> str:
    """Create a local, readable title without an extra LLM request."""
    normalized = " ".join(prompt.split())
    if not normalized:
        return DEFAULT_TITLE
    if len(normalized) <= maximum_length:
        return normalized
    return f"{normalized[: maximum_length - 3]}..."


class ConversationStore:
    """Manage the conversation list and its matching LangGraph checkpoint rows."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self._create_schema()

    def register(self, thread_id: str, title: str) -> None:
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
                """,
                (thread_id, title, timestamp, timestamp),
            )

    def touch(self, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
                (self._timestamp(), thread_id),
            )

    def rename(self, thread_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET title = ? WHERE thread_id = ?",
                (title, thread_id),
            )

    def list_conversations(self) -> list[Conversation]:
        with self._connect() as connection:
            self._register_legacy_threads(connection)
            rows = connection.execute(
                """
                SELECT thread_id, title, updated_at
                FROM conversations
                ORDER BY updated_at DESC, thread_id DESC
                """
            ).fetchall()
        return [Conversation(*row) for row in rows]

    def delete(self, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
            connection.execute(
                "DELETE FROM conversation_summaries WHERE thread_id = ?", (thread_id,)
            )
            for table_name in ("checkpoints", "writes"):
                if self._table_exists(connection, table_name):
                    connection.execute(
                        f"DELETE FROM {table_name} WHERE thread_id = ?", (thread_id,)
                    )

    def get_summary(self, thread_id: str) -> ConversationSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT thread_id, summary, covered_message_count, updated_at
                FROM conversation_summaries
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        return ConversationSummary(*row) if row else None

    def save_summary(
        self,
        thread_id: str,
        summary: str,
        covered_message_count: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_summaries (
                    thread_id, summary, covered_message_count, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary = excluded.summary,
                    covered_message_count = excluded.covered_message_count,
                    updated_at = excluded.updated_at
                """,
                (thread_id, summary, covered_message_count, self._timestamp()),
            )

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_updated_at_idx
                ON conversations (updated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    covered_message_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _register_legacy_threads(self, connection: sqlite3.Connection) -> None:
        if not self._table_exists(connection, "checkpoints"):
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations (thread_id, title, created_at, updated_at)
            SELECT thread_id, ?, '1970-01-01T00:00:00+00:00', '1970-01-01T00:00:00+00:00'
            FROM checkpoints
            WHERE thread_id GLOB 'web_*'
            GROUP BY thread_id
            """,
            (LEGACY_TITLE,),
        )

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
        ).fetchone()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")
