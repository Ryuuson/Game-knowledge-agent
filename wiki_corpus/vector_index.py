"""SQLite storage for retrieval chunks and their embedding vectors."""

import json
import sqlite3
from collections.abc import Sequence
from typing import Any


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the local vector-index table when it does not exist."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            alternate_titles TEXT NOT NULL,
            source_url TEXT NOT NULL,
            section_path TEXT,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            dump_date TEXT,
            license TEXT,
            matched_terms TEXT NOT NULL,
            collection_label TEXT,
            classification_reason TEXT,
            embedding_model TEXT NOT NULL,
            embedding_dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title)")
    connection.commit()


def _json_value(record: dict[str, Any], field: str) -> str:
    return json.dumps(record.get(field, []), ensure_ascii=False)


def replace_chunks(
    connection: sqlite3.Connection,
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[bytes],
    *,
    model_name: str,
    embedding_dimensions: int,
) -> None:
    """Atomically replace all rows with chunks and their matching embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    rows = [
        (
            chunk["chunk_id"],
            chunk["title"],
            _json_value(chunk, "alternate_titles"),
            chunk["source_url"],
            chunk.get("section_path"),
            chunk["start_line"],
            chunk["end_line"],
            chunk["text"],
            chunk.get("dump_date"),
            chunk.get("license"),
            _json_value(chunk, "matched_terms"),
            chunk.get("collection_label"),
            chunk.get("classification_reason"),
            model_name,
            embedding_dimensions,
            bytes(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    with connection:
        connection.execute("DELETE FROM chunks")
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, title, alternate_titles, source_url, section_path,
                start_line, end_line, content, dump_date, license, matched_terms,
                collection_label, classification_reason, embedding_model,
                embedding_dimensions, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
