"""Pure-function tests for vector ranking and retrieval confidence tiers.

rank_chunks is tested directly; classify_game_retrieval thresholds are pinned
with monkeypatch so the tests do not depend on the RAG_BACKEND setting.
"""

import numpy as np
import pytest

import Agent
from scripts.evaluate_rag import evaluate_expectations
from wiki_corpus.hybrid_search import BM25Index, fuse_rrf, rank_hybrid, tokenize
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


def test_tokenize_handles_chinese_bigrams_and_technical_terms():
    assert tokenize("行为树 AI 伤害公式") == ["行为", "为树", "ai", "伤害", "害公", "公式"]


def test_bm25_prioritizes_an_exact_rare_term():
    index = BM25Index(
        [
            {"chunk_id": "generic", "title": "数值设计", "text": "伤害和冷却时间需要反复测试。"},
            {"chunk_id": "rare", "title": "Proc 系统", "text": "pity timer 可控制保底掉落。"},
        ]
    )

    hits = index.rank("pity timer 怎么设计", top_k=2)

    assert hits[0]["chunk_id"] == "rare"
    assert hits[0]["lexical_score"] > 0


def test_rrf_promotes_a_chunk_found_by_both_channels_once():
    dense_hits = [
        {"chunk_id": "dense-only", "score": 0.80},
        {"chunk_id": "both", "score": 0.75},
    ]
    lexical_hits = [
        {"chunk_id": "both", "lexical_score": 4.0},
        {"chunk_id": "lexical-only", "lexical_score": 3.0},
    ]

    hits = fuse_rrf(dense_hits, lexical_hits, top_k=3, rrf_k=60)

    assert [hit["chunk_id"] for hit in hits] == ["both", "dense-only", "lexical-only"]
    assert hits[0]["score"] == 0.75
    assert hits[0]["lexical_score"] == 4.0
    assert hits[0]["dense_rank"] == 2
    assert hits[0]["lexical_rank"] == 1


def test_weighted_rrf_keeps_a_stronger_dense_hit_ahead_of_a_generic_lexical_hit():
    hits = fuse_rrf(
        [
            {"chunk_id": "specific", "score": 0.80},
            {"chunk_id": "generic", "score": 0.64},
        ],
        [
            {"chunk_id": "generic", "lexical_score": 5.0},
            {"chunk_id": "specific", "lexical_score": 2.0},
        ],
        top_k=2,
        rrf_k=60,
    )

    assert [hit["chunk_id"] for hit in hits] == ["specific", "generic"]


def test_a_lexical_only_hit_cannot_pass_the_dense_relevance_gate():
    hits = fuse_rrf(
        [{"chunk_id": "weak-dense", "score": 0.40}],
        [{"chunk_id": "lexical-only", "lexical_score": 9.0}],
        top_k=2,
    )

    accepted = [hit for hit in hits if hit.get("score", 0.0) >= 0.62]

    assert accepted == []


def test_hybrid_ranking_uses_one_shared_candidate_for_both_channels():
    chunks = [
        {"chunk_id": "semantic", "title": "泛化结果", "text": "近义表达"},
        {"chunk_id": "both", "title": "Pity Timer", "text": "pity timer 保底掉落"},
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    lexical_index = BM25Index(chunks)

    hits = rank_hybrid(
        "pity timer",
        np.array([1.0, 0.0], dtype=np.float32),
        chunks,
        vectors,
        lexical_index,
        dense_candidates=2,
        lexical_candidates=2,
    )

    assert [hit["chunk_id"] for hit in hits] == ["both", "semantic"]
    assert hits[0]["score"] > 0.9


def test_expectation_evaluation_requires_all_reviewed_evidence_groups():
    records = [
        {
            "id": "positive",
            "domain_blocked": False,
            "evidence": [{"title": "目标资料", "section": "核心部分"}],
            "expectations": {
                "domain_blocked": False,
                "evidence_groups": [
                    {"name": "主题", "acceptable": [{"title": "目标资料"}]},
                    {"name": "章节", "acceptable": [{"section_contains": "核心"}]},
                ],
            },
        },
        {
            "id": "negative",
            "domain_blocked": False,
            "evidence": [],
            "expectations": {"domain_blocked": True, "evidence_groups": []},
        },
    ]

    evaluation = evaluate_expectations(records)

    assert evaluation["reviewed_cases"] == 2
    assert evaluation["passed_cases"] == 1
    assert [result["id"] for result in evaluation["failed_cases"]] == ["negative"]
