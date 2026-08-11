"""Dependency-free lexical retrieval and reciprocal-rank fusion for local RAG."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import math
import re
from typing import Any

import numpy as np

from wiki_corpus.vector_search import rank_chunks


_TERM_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9][a-z0-9_+.#/-]*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin text without requiring a tokenizer package.

    Chinese is represented by overlapping character bigrams, while English,
    numbers, and common technical identifiers remain whole terms. This keeps
    exact terminology useful without adding a runtime dependency.
    """
    tokens: list[str] = []
    for match in _TERM_RE.finditer(text.casefold()):
        term = match.group()
        if "\u3400" <= term[0] <= "\u9fff":
            if len(term) == 1:
                tokens.append(term)
            else:
                tokens.extend(term[index : index + 2] for index in range(len(term) - 1))
        else:
            tokens.append(term)
    return tokens


def _chunk_terms(chunk: dict[str, Any]) -> str:
    """Select human-readable fields that should contribute to lexical recall."""
    alternate_titles = chunk.get("alternate_titles") or []
    matched_terms = chunk.get("matched_terms") or []
    return "\n".join(
        str(value)
        for value in (
            chunk.get("title", ""),
            " ".join(map(str, alternate_titles)),
            chunk.get("section_path", ""),
            " ".join(map(str, matched_terms)),
            chunk.get("text", ""),
        )
        if value
    )


class BM25Index:
    """Small in-memory BM25 index, appropriate for the current 6.5k chunks."""

    def __init__(self, chunks: Sequence[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.document_count = len(self.chunks)
        self.document_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for document_id, chunk in enumerate(self.chunks):
            term_counts = Counter(tokenize(_chunk_terms(chunk)))
            self.document_lengths.append(sum(term_counts.values()))
            for term, count in term_counts.items():
                self.postings[term].append((document_id, count))

        self.average_document_length = (
            sum(self.document_lengths) / self.document_count if self.document_count else 0.0
        )

    def rank(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks, retaining their raw lexical scores."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.document_count:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_frequency = math.log(
                1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_id, term_frequency in postings:
                document_length = self.document_lengths[document_id]
                length_normalizer = 1 - self.b + self.b * document_length / self.average_document_length
                scores[document_id] += inverse_frequency * (
                    term_frequency * (self.k1 + 1) / (term_frequency + self.k1 * length_normalizer)
                )

        ranked = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
        return [
            {**self.chunks[document_id], "lexical_score": scores[document_id]}
            for document_id in ranked[:top_k]
        ]


def fuse_rrf(
    dense_hits: Sequence[dict[str, Any]],
    lexical_hits: Sequence[dict[str, Any]],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 0.25,
) -> list[dict[str, Any]]:
    """Fuse dense and lexical rankings with reciprocal-rank fusion.

    A chunk may appear in both inputs. It is emitted once, with the original
    dense score retained for relevance gating by the caller.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")
    if dense_weight <= 0 or lexical_weight <= 0:
        raise ValueError("RRF channel weights must be greater than zero")

    fused: dict[str, dict[str, Any]] = {}
    for channel, hits, weight in (
        ("dense", dense_hits, dense_weight),
        ("lexical", lexical_hits, lexical_weight),
    ):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError("each retrieval hit must include chunk_id")
            entry = fused.setdefault(
                chunk_id,
                {"hit": dict(hit), "rrf_score": 0.0, "first_seen": len(fused)},
            )
            entry["hit"].update(hit)
            entry["rrf_score"] += weight / (rrf_k + rank)
            entry[f"{channel}_rank"] = rank

    ranked = sorted(fused.values(), key=lambda entry: (-entry["rrf_score"], entry["first_seen"]))
    results = []
    for entry in ranked[:top_k]:
        hit = entry["hit"]
        results.append(
            {
                **hit,
                "rrf_score": entry["rrf_score"],
                "dense_rank": entry.get("dense_rank"),
                "lexical_rank": entry.get("lexical_rank"),
            }
        )
    return results


def rank_hybrid(
    query: str,
    query_vector: np.ndarray,
    chunks: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    lexical_index: BM25Index,
    *,
    dense_candidates: int = 20,
    lexical_candidates: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 0.25,
) -> list[dict[str, Any]]:
    """Return the fused candidate pool before the caller applies relevance gates."""
    if dense_candidates <= 0 or lexical_candidates <= 0:
        raise ValueError("candidate counts must be greater than zero")

    dense_hits = rank_chunks(query_vector, chunks, embeddings, top_k=dense_candidates)
    lexical_hits = lexical_index.rank(query, top_k=lexical_candidates)
    return fuse_rrf(
        dense_hits,
        lexical_hits,
        top_k=dense_candidates + lexical_candidates,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )
