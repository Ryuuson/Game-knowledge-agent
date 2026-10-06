"""Service boundaries, real ranking behavior and honest recovery to keywords."""

import json

import numpy as np
import pytest

from wiki_corpus.hybrid_search import KeywordIndex, reciprocal_rank_fusion
from wiki_corpus.retrieval import BGE_MODEL, RetrievalService, format_evidence


CHUNKS = [
    {"chunk_id": "economy", "title": "游戏金币经济", "text": "金币回收与货币消耗机制", "source_url": "local/economy"},
    {"chunk_id": "combat", "title": "技能冷却", "text": "战斗伤害与冷却时间", "source_url": "local/combat"},
]


class Embedder:
    def encode(self, query, **kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_dense_and_hybrid_keep_cosine_separate_from_rank_fusion():
    vectors = np.array([[4, 0], [0, 3]], dtype=np.float32)
    dense = RetrievalService(chunks=CHUNKS, vectors=vectors, embedder=Embedder())
    hybrid = RetrievalService(chunks=CHUNKS, vectors=vectors, embedder=Embedder(), mode="hybrid")
    assert dense.search("游戏金币回收").hits[0]["chunk_id"] == "economy"
    result = hybrid.search("游戏金币回收")
    assert result.status == "high_confidence"
    assert len(result.hits) == 1  # the orthogonal hit is below the semantic gate
    assert result.hits[0]["score"] == pytest.approx(1.0)
    assert 0 < result.hits[0]["fusion_score"] < 0.1


def test_explicit_non_game_rejection_does_not_load_files(tmp_path):
    service = RetrievalService(index_path=tmp_path / "missing", corpus_paths=[tmp_path / "missing-corpus"])
    result = service.search("医院预约挂号流程怎么设计？")
    assert result.status == "out_of_domain" and result.hits == []
    assert service.chunks is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_and_corrupt_indexes_use_labelled_corpus_fallback(tmp_path, corrupt):
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CHUNKS), encoding="utf-8")
    index = tmp_path / "index.sqlite"
    if corrupt:
        index.write_bytes(b"not-a-sqlite-index")
    service = RetrievalService(index_path=index, corpus_paths=[corpus])
    result = service.search("游戏金币回收")
    assert result.fallback and result.status == "keyword_only"
    assert result.reason == ("invalid_index" if corrupt else "missing_index")
    assert result.hits[0]["chunk_id"] == "economy"
    assert "降级" in format_evidence(result)


def test_model_mismatch_never_encodes_or_reports_semantic_confidence():
    class NeverEncode:
        def encode(self, *args, **kwargs):
            raise AssertionError("incompatible model must not encode")
    service = RetrievalService(chunks=CHUNKS, vectors=np.eye(2), model_name="other-model", embedder=NeverEncode())
    result = service.search("游戏金币回收")
    assert result.reason == "model_mismatch" and result.status == "keyword_only"


@pytest.mark.parametrize("query, top_k", [("", 3), ("x" * 2001, 3), ("游戏", 0), ("游戏", 11), ("游戏", True)])
def test_invalid_inputs_are_rejected_before_loading(query, top_k):
    with pytest.raises(ValueError):
        RetrievalService().search(query, top_k=top_k)


def test_lexical_index_does_not_require_vectors_and_handles_english_terms():
    index = KeywordIndex([{"chunk_id": "a", "title": "Sequence", "text": "behavior tree"},
                          {"chunk_id": "b", "title": "金币", "text": "economy"}])
    assert index.search("sequence behavior")[0]["chunk_id"] == "a"
    assert index.search("zzzz_unknown") == []


def test_rrf_deduplicates_and_retains_both_rank_provenance():
    dense = [{"chunk_id": "a", "score": .8}, {"chunk_id": "b", "score": .7}]
    lexical = [{"chunk_id": "b", "keyword_score": 5}, {"chunk_id": "b", "keyword_score": 5}]
    result = reciprocal_rank_fusion(dense, lexical)
    assert [h["chunk_id"] for h in result] == ["b", "a"]
    assert result[0]["score"] == .7 and result[0]["keyword_score"] == 5
    assert result[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
