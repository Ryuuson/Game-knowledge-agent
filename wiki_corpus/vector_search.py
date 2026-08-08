"""In-memory cosine ranking for vectors read from the local SQLite index."""

import json
import sqlite3
from collections.abc import Sequence
from typing import Any

import numpy as np


def load_index(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    """Restore SQLite BLOB embeddings into an in-memory float32 matrix."""
    rows = connection.execute(
        """
        SELECT chunk_id, title, alternate_titles, source_url, section_path,
               start_line, end_line, content, dump_date, license, matched_terms,
               collection_label, classification_reason, embedding_model,
               embedding_dimensions, embedding
        FROM chunks
        ORDER BY rowid
        """
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32), ""

    model_names = {row[13] for row in rows}
    dimensions = {row[14] for row in rows}
    if len(model_names) != 1 or len(dimensions) != 1:
        raise ValueError("the index must contain one embedding model and dimension")

    dimension = dimensions.pop()
    vectors = []
    chunks = []
    for row in rows:
        vector = np.frombuffer(row[15], dtype=np.float32)
        if len(vector) != dimension:
            raise ValueError(f"chunk {row[0]} has an invalid embedding length")
        vectors.append(vector)
        chunks.append(
            {
                "chunk_id": row[0],
                "title": row[1],
                "alternate_titles": json.loads(row[2]),
                "source_url": row[3],
                "section_path": row[4],
                "start_line": row[5],
                "end_line": row[6],
                "text": row[7],
                "dump_date": row[8],
                "license": row[9],
                "matched_terms": json.loads(row[10]),
                "collection_label": row[11],
                "classification_reason": row[12],
            }
        )
    return chunks, np.vstack(vectors), model_names.pop()


def rank_chunks(
    query_vector: np.ndarray,
    chunks: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the highest cosine-similarity chunks for one query vector."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not len(chunks):
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("query and embeddings must have compatible dimensions")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0 or np.any(vector_norms == 0):
        raise ValueError("zero vectors cannot be ranked by cosine similarity")

    scores = (matrix / vector_norms[:, None]) @ (query / query_norm)
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [{**chunks[index], "score": float(scores[index])} for index in indices]
