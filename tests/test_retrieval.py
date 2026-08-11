"""Pure-function tests for vector ranking and retrieval confidence tiers.

rank_chunks is tested directly; classify_game_retrieval thresholds are pinned
with monkeypatch so the tests do not depend on the RAG_BACKEND setting.
"""

import numpy as np
import pytest

import Agent
from wiki_corpus.vector_search import rank_chunks


def test_rank_chunks_returns_top_k_in_score_order():
    chunks = [{"chunk_id": str(i), "text": f"chunk {i}"} for i in range(3)]
    vectors = np.array(
        [
            [1.0, 0.0],  # 与 query 完全相同 → 1.0
            [0.9, 0.1],  # 接近 → 约 0.994
            [0.0, 1.0],  # 正交 → 0.0
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0], dtype=np.float32)

    hits = rank_chunks(query, chunks, vectors, top_k=2)

    assert [hit["chunk_id"] for hit in hits] == ["0", "1"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"] >= 0.0


def test_rank_chunks_honors_top_k_limit():
    chunks = [{"chunk_id": str(i)} for i in range(3)]
    vectors = np.eye(3, dtype=np.float32)
    query = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    hits = rank_chunks(query, chunks, vectors, top_k=1)
    assert len(hits) == 1
    assert "score" in hits[0]


def test_rank_chunks_empty_corpus_returns_empty_list():
    hits = rank_chunks(
        np.array([1.0], dtype=np.float32),
        [],
        np.empty((0, 1), dtype=np.float32),
    )
    assert hits == []


def test_rank_chunks_rejects_non_positive_top_k():
    chunks = [{"chunk_id": "0"}]
    vectors = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        rank_chunks(np.array([1.0]), chunks, vectors, top_k=0)


def test_rank_chunks_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rank_chunks(
            np.array([1.0]),
            [{"chunk_id": "0"}],
            np.empty((2, 1), dtype=np.float32),
        )


def test_rank_chunks_rejects_zero_vectors():
    with pytest.raises(ValueError):
        rank_chunks(
            np.array([0.0]),
            [{"chunk_id": "0"}],
            np.array([[1.0]], dtype=np.float32),
        )


def test_retrieval_confidence_tiers(monkeypatch):
    monkeypatch.setattr(Agent, "SEMANTIC_THRESHOLD", 0.5)
    monkeypatch.setattr(Agent, "HIGH_CONFIDENCE_THRESHOLD", 0.7)

    assert Agent.classify_game_retrieval(0.49) == "no_evidence"
    # 下限边界包含：0.5 属于"待确认"档。
    assert Agent.classify_game_retrieval(0.5) == "ambiguous"
    assert Agent.classify_game_retrieval(0.69) == "ambiguous"
    # 上限边界包含：0.7 属于"高相关"档。
    assert Agent.classify_game_retrieval(0.7) == "high_confidence"
    assert Agent.classify_game_retrieval(0.95) == "high_confidence"


def test_retrieval_tiers_follow_configured_backend(monkeypatch):
    # 模拟 ark 后端的阈值配置（0.51 / 0.63）。
    monkeypatch.setattr(Agent, "SEMANTIC_THRESHOLD", 0.51)
    monkeypatch.setattr(Agent, "HIGH_CONFIDENCE_THRESHOLD", 0.63)

    assert Agent.classify_game_retrieval(0.50) == "no_evidence"
    assert Agent.classify_game_retrieval(0.51) == "ambiguous"
    assert Agent.classify_game_retrieval(0.63) == "high_confidence"
