"""Evaluation contract tests, without Agent, online models or conversation DBs."""

import copy
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_retrieval as evaluation
from wiki_corpus.retrieval import BGE_MODEL, RetrievalResult, RetrievalService
from wiki_corpus.vector_index import create_schema, replace_chunks


def case(cid="positive", *, negative=False):
    return {"case_id": cid, "query": "医院预约挂号页面" if negative else "游戏跳跃时怎样减少输入失误？",
            "expected_chunk_ids": [] if negative else ["p1", "p2"],
            "expected_domain": "non_game" if negative else "game",
            "source_collections": [] if negative else ["fixture"],
            "dataset_kind": evaluation.DATASET_KIND, "rationale": "fixture annotation",
            "provenance": {"generator": "AI辅助人工构造", "independent_human_review": False, "blind_test": False},
            "evidence": []}


def result(query="q", *, mode="bge", hits=(), status=None, fallback=False, reason="", elapsed_ms=2):
    return RetrievalResult(query, status or ("ambiguous" if hits else "no_evidence"), mode,
                           [{"chunk_id": cid, "score": .65} for cid in hits], elapsed_ms, fallback, reason)


class FixtureService:
    threshold = .62
    high_confidence = .67

    def __init__(self, mode="bge", **kwargs):
        self.mode = mode
        self.calls = []

    def search(self, query, *, top_k, apply_threshold):
        self.calls.append((query, top_k, apply_threshold))
        if "医院" in query:
            return result(query, mode=self.mode, status="out_of_domain")
        # Gated search replenishes from deeper candidates, not raw[:k] filtering.
        return result(query, mode=self.mode, hits=["p2"] if apply_threshold else ["n", "p1"])


@pytest.fixture
def indexed_fixture(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    index_path = tmp_path / "index.sqlite"
    rows = [{"chunk_id": cid, "title": "游戏跳跃", "text": text, "source_url": "fixture.md",
             "section_path": "输入", "start_line": 1, "end_line": 2, "collection_label": "fixture"}
            for cid, text in [("p1", "游戏输入缓冲支持跳跃"), ("p2", "游戏土狼时间减少失误")]]
    corpus_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with sqlite3.connect(index_path) as conn:
        create_schema(conn)
        replace_chunks(conn, rows, [np.array(v, dtype=np.float32).tobytes() for v in [[1, 0], [0, 1]]],
                       model_name=BGE_MODEL, embedding_dimensions=2)
    positive = case()
    positive["evidence"] = [{"chunk_id": r["chunk_id"], "corpus_file": str(corpus_path),
                             **{k: r[k] for k in ("title", "source_url", "section_path", "start_line", "end_line", "collection_label")},
                             "excerpt": r["text"]} for r in rows]
    cases = [positive, case("negative", negative=True)]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    return SimpleNamespace(corpus=corpus_path, index=index_path, cases=cases, cases_path=cases_path)


def test_binary_metrics_have_correct_denominators():
    metrics = evaluation.ranking_metrics(["n", "p1", "p2"], ["p1", "p2", "p3"], 2)
    assert metrics["recall@k"] == pytest.approx(1 / 3)
    assert metrics["mrr"] == .5
    assert metrics["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert metrics["hit_rate"] == 1


def test_metrics_zero_hits_and_top_k_truncation():
    assert evaluation.ranking_metrics([], ["p1"], 5) == {"recall@k": 0, "mrr": 0, "ndcg": 0, "hit_rate": 0}
    assert evaluation.ranking_metrics(["n", "p1"], ["p1"], 1)["hit_rate"] == 0
    assert evaluation.ranking_metrics(["p1"], ["p1"], 5)["ndcg"] == 1


def test_duplicate_hits_cannot_inflate_metrics():
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.ranking_metrics(["p1", "p1"], ["p1", "p2"], 5)


def test_refusals_are_not_ranking_positives():
    with pytest.raises(ValueError, match="requires_positives"):
        evaluation.ranking_metrics([], [], 5)
    service = FixtureService()
    report = evaluation.evaluate_mode([case(), case("negative", negative=True)], "bge", 5, service)
    assert report["valid"]
    assert report["raw"]["ranking"]["count"] == 1
    assert report["raw"]["ranking"]["recall@k"] == .5
    assert report["raw"]["refusal"]["negative_count"] == 1
    assert report["raw"]["refusal"]["correct_refusal_rate"] == 1
    assert report["raw"]["latency"]["ranking_cases"]["count"] == 1
    assert report["raw"]["latency"]["refusal_cases"]["count"] == 1


def test_gated_is_a_separate_production_search_with_replenishment():
    service = FixtureService(mode="hybrid")
    report = evaluation.evaluate_mode([case()], "hybrid", 5, service)
    assert report["valid"]
    assert [c[2] for c in service.calls] == [False, False, True]  # warmup, raw, gated
    assert [h["chunk_id"] for h in report["cases"][0]["gated"]["hits"]] == ["p2"]
    assert report["raw"]["ranking"]["mrr"] == .5
    assert report["gated"]["ranking"]["mrr"] == 1


@pytest.mark.parametrize("mode", ["bge", "hybrid"])
def test_actual_embedding_failure_invalidates_dense_backend(indexed_fixture, mode):
    class BrokenEmbedder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("DO_NOT_PRINT_secret_token")

    service = RetrievalService(mode=mode, index_path=indexed_fixture.index, embedder=BrokenEmbedder())
    report = evaluation.evaluate_mode(indexed_fixture.cases, mode, 5, service)
    assert not report["valid"]
    assert report["status"] == "failed"
    assert report["warmup"]["fallback"] is True
    assert report["raw"] is None and report["gated"] is None
    assert "DO_NOT_PRINT" not in json.dumps(report)


@pytest.mark.parametrize("change", ["fallback", "mode", "keyword_only"])
def test_late_degradation_invalidates_all_averages(change):
    class LateFailure(FixtureService):
        def search(self, query, **kwargs):
            value = super().search(query, **kwargs)
            if kwargs["apply_threshold"]:
                if change == "fallback":
                    value.fallback = True
                elif change == "mode":
                    value.mode = "keyword"
                else:
                    value.status = "keyword_only"
            return value

    report = evaluation.evaluate_mode([case()], "bge", 5, LateFailure())
    assert report["completed_cases"] == 1
    assert report["raw"] is None and report["gated"] is None
    assert report["failures"][0]["stage"] == "gated"


def test_exception_messages_are_not_reported():
    class FailingService(FixtureService):
        def search(self, *args, **kwargs):
            raise OSError("DO_NOT_PRINT_secret_token")

    report = evaluation.evaluate_mode([case()], "bge", 5, FailingService())
    assert not report["valid"]
    assert report["failures"][0]["exception_type"] == "OSError"
    assert "DO_NOT_PRINT" not in evaluation.render_markdown({"status": "failed", "modes": {"bge": report},
                                                            "methodology": {}, "failures": []})


def test_refusal_requires_no_hits_and_a_refusal_status():
    assert evaluation.is_refusal(result(status="out_of_domain").to_dict())
    assert evaluation.is_refusal(result(status="no_evidence").to_dict())
    assert not evaluation.is_refusal(result(status="keyword_only").to_dict())
    assert not evaluation.is_refusal(result(hits=["n"], status="out_of_domain").to_dict())


def test_false_refusal_and_negative_leakage_are_separate():
    class SwappedService(FixtureService):
        def search(self, query, **kwargs):
            return result(query, mode="keyword", hits=["n"] if "医院" in query else [], status="keyword_only" if "医院" in query else "no_evidence")

    report = evaluation.evaluate_mode([case(), case("negative", negative=True)], "keyword", 5, SwappedService("keyword"))
    assert report["valid"]
    assert report["gated"]["ranking"]["hit_rate"] == 0
    refusal = report["gated"]["refusal"]
    assert refusal["false_refusal_rate"] == 1 and refusal["correct_refusal_rate"] == 0
    assert refusal["negative_cases_with_hits"] == 1


def test_latency_quantiles_and_empty_groups():
    summary = evaluation.latency_summary([0, 10, 20, 30])
    assert summary == {"count": 4, "p50_ms": 15, "p95_ms": pytest.approx(28.5)}
    assert evaluation.latency_summary([]) == {"count": 0, "p50_ms": None, "p95_ms": None}


def test_validation_matches_actual_index_and_literal_excerpts(indexed_fixture):
    value = evaluation.validate_annotations(indexed_fixture.cases, [indexed_fixture.corpus], indexed_fixture.index)
    assert value["validated_positive_annotations"] == 2
    assert value["index_chunks"] == 2
    assert value["embedding_models"] == [BGE_MODEL]


@pytest.mark.parametrize("mutation,code", [("excerpt", "excerpt_not_in_corpus"), ("source", "source_metadata"),
                                          ("collection", "collection_mismatch"), ("unknown", "unknown_positive")])
def test_stale_or_invented_annotations_fail(indexed_fixture, mutation, code):
    cases = copy.deepcopy(indexed_fixture.cases)
    positive = cases[0]
    if mutation == "excerpt":
        positive["evidence"][0]["excerpt"] = "invented quotation"
    elif mutation == "source":
        positive["evidence"][0]["source_url"] = "wrong.md"
    elif mutation == "collection":
        positive["source_collections"] = ["wrong"]
    else:
        positive["expected_chunk_ids"][0] = "missing"
        positive["evidence"][0]["chunk_id"] = "missing"
    with pytest.raises(ValueError, match=code):
        evaluation.validate_annotations(cases, [indexed_fixture.corpus], indexed_fixture.index)


def test_index_content_drift_fails(indexed_fixture):
    with sqlite3.connect(indexed_fixture.index) as conn:
        conn.execute("UPDATE chunks SET content='changed' WHERE chunk_id='p1'")
    with pytest.raises(ValueError, match="content_mismatch"):
        evaluation.validate_annotations(indexed_fixture.cases, [indexed_fixture.corpus], indexed_fixture.index)


@pytest.mark.parametrize("mutation", ["domain", "human_review", "blind", "duplicate"])
def test_invalid_dataset_contract_fails(tmp_path, mutation):
    cases = [case()]
    if mutation == "domain":
        cases[0]["expected_domain"] = "non_game"
    elif mutation == "human_review":
        cases[0]["provenance"]["independent_human_review"] = True
    elif mutation == "blind":
        cases[0]["provenance"]["blind_test"] = True
    else:
        cases.append(copy.deepcopy(cases[0]))
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.read_cases(path)


def test_project_dataset_coverage_and_annotations(tmp_path):
    cases = evaluation.read_cases(evaluation.DEFAULT_CASES)
    assert 30 <= len(cases) <= 50
    assert len([c for c in cases if not c["expected_chunk_ids"]]) >= 8
    counts = Counter(s for c in cases for s in c["source_collections"])
    assert len(counts) == 6 and min(counts.values()) >= 5
    index_path = evaluation.DEFAULT_INDEX
    if not index_path.is_file():
        # Clean clones do not contain private generated indexes or model weights.
        # This fixture only checks metadata/annotations; its zero vectors are never searched.
        rows = [row for path in evaluation.DEFAULT_CORPUS for row in evaluation.read_jsonl(path)]
        index_path = tmp_path / "metadata-only-index.sqlite"
        with sqlite3.connect(index_path) as conn:
            create_schema(conn)
            replace_chunks(conn, rows, [bytes(512 * 4)] * len(rows),
                           model_name=BGE_MODEL, embedding_dimensions=512)
    verified = evaluation.validate_annotations(cases, evaluation.DEFAULT_CORPUS, index_path)
    assert verified["corpus_chunks"] == verified["index_chunks"]
    assert verified["validated_positive_annotations"] == sum(len(c["expected_chunk_ids"]) for c in cases)


def args_for(fixture, tmp_path, backend="all"):
    return evaluation.parse_args(["--backend", backend, "--cases", str(fixture.cases_path), "--corpus", str(fixture.corpus),
                                  "--index-path", str(fixture.index), "--output", str(tmp_path / "result.json")])


def test_all_mode_report_hashes_and_persistence(indexed_fixture, tmp_path):
    args = args_for(indexed_fixture, tmp_path)
    report = evaluation.build_report(args, service_factory=FixtureService)
    assert report["status"] == "passed"
    assert list(report["modes"]) == ["bge", "keyword", "hybrid"]
    assert report["hashes"]["dataset_sha256"] == evaluation.sha256_file(indexed_fixture.cases_path)
    assert report["hashes"]["index"]["sha256"] == evaluation.sha256_file(indexed_fixture.index)
    assert report["hashes"]["configuration_sha256"] == evaluation.canonical_hash(report["configuration"])
    json_path, md_path = evaluation.write_reports(report, args.output)
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "passed"
    markdown = md_path.read_text(encoding="utf-8")
    assert "未经独立人工审核" in markdown and "FAILED" not in markdown
    assert "process lifetime" in markdown and "environment" in markdown


def test_all_mode_can_report_keyword_without_disguising_failed_dense(indexed_fixture, tmp_path):
    class BrokenDense(FixtureService):
        def search(self, query, **kwargs):
            if self.mode != "keyword":
                return result(query, mode=self.mode, hits=["p1"], fallback=True, status="keyword_only", reason="embedding_unavailable")
            return super().search(query, **kwargs)

    report = evaluation.build_report(args_for(indexed_fixture, tmp_path), service_factory=BrokenDense)
    assert report["status"] == "failed"
    assert report["modes"]["keyword"]["valid"]
    assert report["modes"]["bge"]["raw"] is None and report["modes"]["hybrid"]["gated"] is None
    assert "FAILED; metrics suppressed" in evaluation.render_markdown(report)


def test_main_writes_failure_reports_and_returns_nonzero(indexed_fixture, tmp_path, monkeypatch):
    args = args_for(indexed_fixture, tmp_path, "bge")
    report = evaluation.build_report(args, service_factory=FixtureService)
    report["status"] = "failed"
    monkeypatch.setattr(evaluation, "build_report", lambda args: report)
    assert evaluation.main(["--output", str(args.output), "--backend", "bge"]) == 1
    assert args.output.is_file() and args.output.with_suffix(".md").is_file()


def test_input_mutation_during_run_invalidates_metrics(indexed_fixture, tmp_path):
    class MutatingService(FixtureService):
        def search(self, query, **kwargs):
            indexed_fixture.cases_path.write_text(indexed_fixture.cases_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return super().search(query, **kwargs)

    report = evaluation.build_report(args_for(indexed_fixture, tmp_path, "keyword"), service_factory=MutatingService)
    assert report["status"] == "failed"
    assert report["failures"][0]["code"] == "inputs_or_code_changed_during_evaluation"
    assert report["modes"]["keyword"]["raw"] is None


def test_local_embedder_forces_offline_cpu(monkeypatch):
    calls = []
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "0")

    class LocalModel:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
        def encode(self, query, **kwargs):
            return [1, 0]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=LocalModel))
    embedder = evaluation.LocalBGEEmbedder()
    assert embedder.encode("游戏") == [1, 0]
    embedder.encode("游戏2")
    assert calls == [(BGE_MODEL, {"local_files_only": True, "device": "cpu"})]
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_import_has_no_agent_model_or_conversation_side_effects():
    code = "import sys; import scripts.evaluate_retrieval; assert 'Agent' not in sys.modules; assert 'sentence_transformers' not in sys.modules; assert 'conversation_store' not in sys.modules; assert 'langchain_openai' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], cwd=evaluation.ROOT, capture_output=True, timeout=20)
    assert completed.returncode == 0


def test_output_cannot_overwrite_input(indexed_fixture):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--cases", str(indexed_fixture.cases_path.with_suffix(".json")),
                               "--output", str(indexed_fixture.cases_path.with_suffix(".json"))])
