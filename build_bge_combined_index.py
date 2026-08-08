"""Build a local BGE index from all current game-knowledge chunks."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wiki_corpus.vector_index import create_schema, replace_chunks


ROOT = Path(__file__).resolve().parent
OLD_CHUNKS_PATH = ROOT / "data" / "game_knowledge_chunks.jsonl"
NEW_CHUNKS_PATH = ROOT / "data" / "new_knowledge_chunks.jsonl"
INDEX_PATH = ROOT / "data" / "game_knowledge_bge_combined_index.sqlite"
BGE_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 32


def load_combined_chunks(*paths: Path) -> list[dict[str, Any]]:
    """Load chunk sources in order and reject overlapping identifiers."""
    chunks = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("combined chunks contain duplicate chunk IDs")
    return chunks


def embedding_text(chunk: Mapping[str, Any]) -> str:
    """Embed each chunk with its title and section context."""
    return "\n".join(
        str(part)
        for part in (
            chunk["title"],
            chunk.get("section_path") or "",
            chunk["text"],
        )
        if part
    )


def build_index(
    *,
    old_chunks_path: Path = OLD_CHUNKS_PATH,
    new_chunks_path: Path = NEW_CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Embed every known chunk locally and write a replacement-safe SQLite index."""
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"index already exists: {index_path}; pass --overwrite to replace it")

    from sentence_transformers import SentenceTransformer

    chunks = load_combined_chunks(old_chunks_path, new_chunks_path)
    if not chunks:
        raise ValueError("combined chunk sources are empty")

    # Building is the one-time setup step that may download the model.
    # Runtime retrieval remains local-only after this cache has been populated.
    model = SentenceTransformer(BGE_EMBEDDING_MODEL)
    vectors = model.encode(
        [embedding_text(chunk) for chunk in chunks],
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(f"{index_path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        create_schema(connection)
        replace_chunks(
            connection,
            chunks,
            [vector.tobytes() for vector in vectors],
            model_name=BGE_EMBEDDING_MODEL,
            embedding_dimensions=int(vectors.shape[1]),
        )
    finally:
        connection.close()
    temporary_path.replace(index_path)

    return {
        "chunks": len(chunks),
        "embedding_model": BGE_EMBEDDING_MODEL,
        "embedding_dimensions": int(vectors.shape[1]),
        "index_path": str(index_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the combined local BGE game index")
    parser.add_argument("--old-chunks", type=Path, default=OLD_CHUNKS_PATH)
    parser.add_argument("--new-chunks", type=Path, default=NEW_CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_index(
        old_chunks_path=args.old_chunks,
        new_chunks_path=args.new_chunks,
        index_path=args.index_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
