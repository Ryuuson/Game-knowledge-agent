"""Offline retrieval regression evaluation. Never import Agent or load .env."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wiki_corpus.retrieval import BGE_MODEL, DEFAULT_INDEX, RetrievalService

DATASET_KIND = "ai_assisted_human_constructed_regression"
DEFAULT_CASES = ROOT / "evals" / "retrieval_cases.jsonl"
DEFAULT_CORPUS = [ROOT / "data" / "game_knowledge_chunks.jsonl",
                  ROOT / "data" / "new_knowledge_chunks.jsonl"]
SAFE_REASONS = {"", "missing_index", "model_mismatch", "embedding_unavailable",
                "explicit_non_game_context"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_cases(path):
    cases = read_jsonl(path)
    if not cases:
        raise ValueError("empty_cases")
    seen_ids, seen_queries = set(), set()
    for case in cases:
        cid, query = case.get("case_id"), case.get("query")
        if not isinstance(cid, str) or not cid or cid in seen_ids:
            raise ValueError("invalid_or_duplicate_case_id")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or query.strip() in seen_queries:
            raise ValueError("invalid_or_duplicate_query")
        seen_ids.add(cid)
        seen_queries.add(query.strip())
        ids = case.get("expected_chunk_ids")
        domain = case.get("expected_domain")
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("invalid_positive_ids")
        if domain not in {"game", "non_game"} or bool(ids) != (domain == "game"):
            raise ValueError("refusal_and_ranking_labels_must_be_separate")
        provenance = case.get("provenance", {})
        if (case.get("dataset_kind") != DATASET_KIND
                or provenance.get("independent_human_review") is not False
                or provenance.get("blind_test") is not False
                or not provenance.get("generator")):
            raise ValueError("unsupported_dataset_provenance")
        if not case.get("rationale"):
            raise ValueError("missing_annotation_rationale")
    if not any(c["expected_chunk_ids"] for c in cases):
        raise ValueError("no_ranking_cases")
    return cases


def validate_annotations(cases, corpus_paths, index_path):
    """Check labels and literal excerpts against BOTH corpus and served index."""
    corpus, corpus_files = {}, {}
    for path in corpus_paths:
        for row in read_jsonl(path):
            cid = str(row["chunk_id"])
            if cid in corpus:
                raise ValueError("duplicate_corpus_chunk_id")
            corpus[cid], corpus_files[cid] = row, Path(path).resolve()
    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        has_provenance = "provenance" in {r[1] for r in connection.execute("PRAGMA table_info(chunks)")}
        provenance_column = "provenance" if has_provenance else "'{}'"
        index_rows = connection.execute(
            "SELECT chunk_id,title,source_url,section_path,start_line,end_line,content,"
            f"collection_label,embedding_model,embedding_dimensions,license,{provenance_column} FROM chunks ORDER BY chunk_id"
        ).fetchall()
    index = {str(r[0]): dict(zip(("chunk_id", "title", "source_url", "section_path", "start_line",
                                 "end_line", "text", "collection_label", "embedding_model",
                                 "embedding_dimensions", "license", "provenance"), r)) for r in index_rows}
    if not index or set(corpus) != set(index):
        raise ValueError("index_corpus_id_set_mismatch")
    fields = ("title", "source_url", "section_path", "start_line", "end_line", "text", "collection_label")
    for cid, row in corpus.items():
        if any(row.get(f) != index[cid].get(f) for f in fields):
            raise ValueError("index_corpus_content_mismatch")
        if (row.get("license") != index[cid]["license"]
                or row.get("provenance", {}) != json.loads(index[cid]["provenance"])):
            raise ValueError("index_corpus_provenance_mismatch")
    for case in cases:
        ids = case["expected_chunk_ids"]
        evidence = case.get("evidence", [])
        if len(evidence) != len(ids) or {e.get("chunk_id") for e in evidence} != set(ids):
            raise ValueError("evidence_id_mismatch")
        labels = set()
        for item in evidence:
            cid = item["chunk_id"]
            if cid not in corpus:
                raise ValueError("unknown_positive_chunk_id")
            row = corpus[cid]
            labels.add(row["collection_label"])
            if any(item.get(f) != row.get(f) for f in fields if f != "text"):
                raise ValueError("evidence_source_metadata_mismatch")
            path = Path(item.get("corpus_file", ""))
            if not path.is_absolute():
                path = ROOT / path
            if path.resolve() != corpus_files[cid]:
                raise ValueError("evidence_corpus_file_mismatch")
            excerpt = item.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip() or excerpt not in row["text"]:
                raise ValueError("evidence_excerpt_not_in_corpus")
        if set(case.get("source_collections", [])) != labels:
            raise ValueError("case_collection_mismatch")
    return {"corpus_chunks": len(corpus), "index_chunks": len(index),
            "collections": dict(sorted(Counter(r["collection_label"] for r in corpus.values()).items())),
            "embedding_models": sorted({r["embedding_model"] for r in index.values()}),
            "embedding_dimensions": sorted({r["embedding_dimensions"] for r in index.values()}),
            "index_manifest_sha256": canonical_hash(index_rows),
            "validated_positive_annotations": sum(len(c["expected_chunk_ids"]) for c in cases)}


def ranking_metrics(hit_ids, positives, top_k):
    """Binary relevance; no refusal cases and no duplicate result inflation."""
    relevant = set(positives)
    if not relevant:
        raise ValueError("ranking_requires_positives")
    retrieved = list(hit_ids)[:top_k]
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("duplicate_retrieved_chunk_id")
    ranks = [rank for rank, cid in enumerate(retrieved, 1) if cid in relevant]
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(top_k, len(relevant)) + 1))
    return {"recall@k": len(ranks) / len(relevant), "mrr": 1 / ranks[0] if ranks else 0.0,
            "ndcg": dcg / ideal, "hit_rate": float(bool(ranks))}


def percentile(values, fraction):
    """Linear interpolation, equivalent to NumPy's default percentile."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def latency_summary(values):
    return {"count": len(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95)}


def memory_snapshot():
    """Native process resident memory; process peak is NOT a backend allocation."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"rss_bytes": int(counters.WorkingSetSize),
                    "process_lifetime_peak_rss_bytes": int(counters.PeakWorkingSetSize),
                    "method": "Windows GetProcessMemoryInfo working set"}
    elif sys.platform.startswith("linux"):
        import resource
        try:
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            return {"rss_bytes": rss, "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "method": "Linux /proc/self/statm and getrusage"}
        except (OSError, ValueError):
            pass
    return {"rss_bytes": None, "process_lifetime_peak_rss_bytes": None, "method": "unavailable"}


class LocalBGEEmbedder:
    """Lazy CPU-only cached model. No network model provider is used."""

    def __init__(self):
        self.model = None

    def encode(self, query, **kwargs):
        if self.model is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(BGE_MODEL, local_files_only=True, device="cpu")
        return self.model.encode(query, show_progress_bar=False, **kwargs)


def compact_result(result, requested_mode):
    payload = result.to_dict()
    # Do not serialize arbitrary exception messages or model/corpus text.
    payload["reason"] = payload.get("reason", "") if payload.get("reason", "") in SAFE_REASONS else "redacted_reason"
    elapsed = float(payload["elapsed_ms"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid_service_latency")
    hits = payload["hits"]
    if len({str(h["chunk_id"]) for h in hits}) != len(hits):
        raise ValueError("duplicate_retrieved_chunk_id")
    payload["hits"] = [{key: hit[key] for key in ("chunk_id", "citation_id", "title", "collection_label",
                         "score", "keyword_score", "fusion_score", "dense_rank", "keyword_rank") if key in hit}
                       for hit in hits]
    invalid = payload.get("mode") != requested_mode or payload.get("fallback", False)
    if requested_mode in {"bge", "hybrid"} and payload.get("status") == "keyword_only":
        invalid = True
    return payload, invalid


def is_refusal(result):
    return not result["hits"] and result["status"] in {"out_of_domain", "no_evidence"}


def aggregate_stage(records, stage, top_k):
    positives = [r for r in records if r["expected_chunk_ids"]]
    negatives = [r for r in records if not r["expected_chunk_ids"]]
    scores = [ranking_metrics([h["chunk_id"] for h in r[stage]["hits"]], r["expected_chunk_ids"], top_k)
              for r in positives]
    ranking = {key: sum(s[key] for s in scores) / len(scores) if scores else None
               for key in ("recall@k", "mrr", "ndcg", "hit_rate")}
    ranking["count"] = len(positives)
    correct = sum(is_refusal(r[stage]) for r in negatives)
    false_refusals = sum(is_refusal(r[stage]) for r in positives)
    return {"ranking": ranking,
            "refusal": {"negative_count": len(negatives), "correct_refusals": correct,
                        "correct_refusal_rate": correct / len(negatives) if negatives else None,
                        "negative_cases_with_hits": sum(bool(r[stage]["hits"]) for r in negatives),
                        "positive_count": len(positives), "false_refusals": false_refusals,
                        "false_refusal_rate": false_refusals / len(positives) if positives else None,
                        "negative_status_counts": dict(Counter(r[stage]["status"] for r in negatives))},
            "latency": {"all": latency_summary([r[stage]["elapsed_ms"] for r in records]),
                        "ranking_cases": latency_summary([r[stage]["elapsed_ms"] for r in positives]),
                        "refusal_cases": latency_summary([r[stage]["elapsed_ms"] for r in negatives])},
            "by_collection": {label: aggregate_ranking_for_collection(positives, stage, label, top_k)
                              for label in sorted({s for r in positives for s in r["source_collections"]})}}


def aggregate_ranking_for_collection(records, stage, label, top_k):
    selected = [r for r in records if label in r["source_collections"]]
    metrics = [ranking_metrics([h["chunk_id"] for h in r[stage]["hits"]], r["expected_chunk_ids"], top_k)
               for r in selected]
    return {"count": len(metrics), **{key: sum(s[key] for s in metrics) / len(metrics)
            for key in ("recall@k", "mrr", "ndcg", "hit_rate")}}


def evaluate_mode(cases, mode, top_k, service):
    before = memory_snapshot()
    records, failures = [], []
    started = perf_counter()
    warmup = None
    try:
        first_query = next(c["query"] for c in cases if c["expected_chunk_ids"])
        warmup, invalid = compact_result(service.search(first_query, top_k=top_k, apply_threshold=False), mode)
        if invalid:
            failures.append({"phase": "warmup", "code": "backend_degraded_or_changed", "reason": warmup["reason"]})
        else:
            for case in cases:
                record = {key: case[key] for key in ("case_id", "query", "expected_domain", "expected_chunk_ids", "source_collections")}
                for stage in ("raw", "gated"):
                    result, invalid = compact_result(service.search(case["query"], top_k=top_k,
                                                                    apply_threshold=stage == "gated"), mode)
                    record[stage] = result
                    if invalid:
                        failures.append({"case_id": case["case_id"], "stage": stage,
                                         "code": "backend_degraded_or_changed", "reason": result["reason"]})
                records.append(record)
    except Exception as exc:
        # Exception text can contain paths, credentials or remote URLs.
        failures.append({"code": "evaluation_exception", "exception_type": type(exc).__name__})
    after = memory_snapshot()
    valid = not failures and len(records) == len(cases)
    return {"requested_mode": mode, "status": "passed" if valid else "failed", "valid": valid,
            "failures": failures, "warmup": warmup, "completed_cases": len(records),
            "wall_time_ms": (perf_counter() - started) * 1000,
            "raw": aggregate_stage(records, "raw", top_k) if valid else None,
            "gated": aggregate_stage(records, "gated", top_k) if valid else None,
            "memory": {"before_warmup": before, "after_cases": after,
                       "rss_delta_bytes": after["rss_bytes"] - before["rss_bytes"]
                       if after["rss_bytes"] is not None and before["rss_bytes"] is not None else None},
            "cases": records}


def environment_metadata():
    packages = {}
    for name in ("numpy", "sentence-transformers", "torch", "transformers", "huggingface-hub", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
            "executable": sys.executable, "os": platform.system(), "os_release": platform.release(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages,
            "embedding_device": "cpu", "online_model_api_used": False,
            "model_loading": "local_files_only=True; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1"}


def input_hashes(cases_path, corpus_paths, index_path):
    return {"dataset_sha256": sha256_file(cases_path),
            "corpus_files": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in corpus_paths],
            "index": {"path": str(Path(index_path).resolve()), "sha256": sha256_file(index_path)}}


def build_report(args, *, service_factory=RetrievalService, embedder=None):
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "failed", "environment": environment_metadata(), "failures": [], "modes": {},
              "methodology": {
                  "dataset_kind": DATASET_KIND, "independent_human_review": False, "blind_test": False,
                  "ranking_denominator": "macro average over positive cases only; selected chunk labels are not exhaustive",
                  "relevance": "binary exact chunk_id; unlabelled chunks count as non-relevant; recall denominator is all labelled positives",
                  "raw": "service search(apply_threshold=False); domain routing still applies",
                  "gated": "separate production search(apply_threshold=True), allowing candidate replenishment; keyword has no semantic gate",
                  "refusal": "negative cases separate from ranking; correct means no hits AND status out_of_domain/no_evidence",
                  "latency": "service elapsed_ms; one excluded positive warmup per mode; fixed case order raw then gated; no repeats; CPU",
                  "memory": "bytes; process resident working set including native CPU model, vectors and allocator; before warmup/after cases; delta is not attributable allocation; peak is process lifetime, not backend peak; modes share model in one process; no GPU memory measurement",
                  "failure": "any fallback, changed mode, keyword_only dense status or exception invalidates the entire mode and suppresses its averages; exit 1"}}
    try:
        before = input_hashes(args.cases, args.corpus, args.index_path)
        cases = read_cases(args.cases)
        validation = validate_annotations(cases, args.corpus, args.index_path)
        modes = ["bge", "keyword", "hybrid"] if args.backend == "all" else [args.backend]
        config = {"backends": modes, "top_k": args.top_k, "embedding_model": BGE_MODEL,
                  "threshold_override": args.threshold, "high_confidence_override": args.high_confidence,
                  "device": "cpu", "local_files_only": True, "resolved_thresholds": {},
                  "source_sha256": {name: sha256_file(ROOT / name) for name in (
                      "wiki_corpus/retrieval.py", "wiki_corpus/hybrid_search.py", "wiki_corpus/domain_signals.py",
                      "wiki_corpus/vector_search.py", "wiki_corpus/provenance.py", "scripts/evaluate_retrieval.py")}}
        report["dataset"] = {"path": str(args.cases.resolve()), "case_count": len(cases),
                             "ranking_case_count": sum(bool(c["expected_chunk_ids"]) for c in cases),
                             "refusal_case_count": sum(not c["expected_chunk_ids"] for c in cases),
                             "ranking_cases_by_collection": dict(Counter(s for c in cases for s in c["source_collections"]))}
        report["validation"], report["hashes"] = validation, dict(before)
        shared_embedder = embedder if embedder is not None else LocalBGEEmbedder()
        for mode in modes:
            kwargs = {"mode": mode, "index_path": args.index_path,
                      "embedder": None if mode == "keyword" else shared_embedder}
            if args.threshold is not None:
                kwargs["threshold"] = args.threshold
            if args.high_confidence is not None:
                kwargs["high_confidence"] = args.high_confidence
            try:
                service = service_factory(**kwargs)
                config["resolved_thresholds"][mode] = {"threshold": service.threshold, "high_confidence": service.high_confidence}
                report["modes"][mode] = evaluate_mode(cases, mode, args.top_k, service)
                del service
            except Exception as exc:
                report["modes"][mode] = {"status": "failed", "valid": False, "raw": None, "gated": None,
                                         "failures": [{"code": "service_initialization_failed", "exception_type": type(exc).__name__}]}
        report["configuration"] = config
        report["hashes"]["configuration_sha256"] = canonical_hash(config)
        report["hashes"]["corpus_sha256"] = canonical_hash(before["corpus_files"])
        after = input_hashes(args.cases, args.corpus, args.index_path)
        if before != after or any(config["source_sha256"][name] != sha256_file(ROOT / name) for name in config["source_sha256"]):
            report["failures"].append({"code": "inputs_or_code_changed_during_evaluation"})
            for result in report["modes"].values():
                result.update(status="failed", valid=False, raw=None, gated=None)
        if not report["failures"] and all(r["valid"] for r in report["modes"].values()):
            report["status"] = "passed"
    except Exception as exc:
        report["failures"].append({"code": "input_validation_or_setup_failed", "exception_type": type(exc).__name__})
    return report


def render_markdown(report):
    lines = ["# Offline retrieval regression report", "", f"Run status: **{report['status']}**", "",
             "AI辅助人工构造回归集；本次案例补齐与语料核对由 AI 执行。未经独立人工审核，不是盲测。", "",
             "Passed means the requested backends ran without degradation and inputs validated; it does not mean quality targets were met.", ""]
    if "dataset" in report:
        dataset = report["dataset"]
        lines += [f"Cases: {dataset['case_count']}; ranking positives: {dataset['ranking_case_count']}; refusals: {dataset['refusal_case_count']}.", ""]
    lines += ["| Mode | Stage | Recall@k | MRR@k | nDCG@k | Hit rate | Correct refusal | False refusal | p50 ms | p95 ms |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    for mode, result in report["modes"].items():
        if not result["valid"]:
            lines.append(f"| {mode} | FAILED; metrics suppressed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        for stage in ("raw", "gated"):
            data = result[stage]
            ranking, refusal, latency = data["ranking"], data["refusal"], data["latency"]["all"]
            values = [ranking[key] for key in ("recall@k", "mrr", "ndcg", "hit_rate")]
            values += [refusal["correct_refusal_rate"], refusal["false_refusal_rate"], latency["p50_ms"], latency["p95_ms"]]
            lines.append(f"| {mode} | {stage} | " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "## Methodology", ""]
    lines += [f"- {key}: {value}" for key, value in report["methodology"].items()]
    lines += ["", "## Reproducibility and environment", "", "```json",
              json.dumps({key: report[key] for key in ("dataset", "validation", "hashes", "configuration", "environment") if key in report},
                         ensure_ascii=False, indent=2), "```", "", "## Memory and warmup", "", "```json",
              json.dumps({mode: {key: result.get(key) for key in ("memory", "warmup", "wall_time_ms", "completed_cases")}
                          for mode, result in report["modes"].items()}, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## Per-source ranking", "", "```json", json.dumps({mode: {
        stage: result[stage]["by_collection"] for stage in ("raw", "gated")}
        for mode, result in report["modes"].items() if result["valid"]}, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## Failures and refusal diagnostics", "", "```json", json.dumps({
        "run_failures": report["failures"], "mode_failures": {mode: r["failures"] for mode, r in report["modes"].items()},
        "negative_cases": {mode: [{"case_id": c["case_id"], **{stage: c[stage] for stage in ("raw", "gated")}}
                                  for c in r.get("cases", []) if not c["expected_chunk_ids"]]
                           for mode, r in report["modes"].items()}}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def write_reports(report, output):
    output = Path(output)
    json_path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bge", "keyword", "hybrid", "all"), default="all")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "retrieval_evaluation.json")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--corpus", type=Path, nargs="+", default=DEFAULT_CORPUS)
    parser.add_argument("--threshold", type=float, default=None, help="Optional production threshold override; default: service value")
    parser.add_argument("--high-confidence", type=float, default=None, help="Optional high-confidence override; default: service value")
    args = parser.parse_args(argv)
    for value in (args.threshold, args.high_confidence):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            parser.error("thresholds must be finite numbers between 0 and 1")
    outputs = {args.output.with_suffix(".json").resolve(), args.output.with_suffix(".md").resolve()}
    if outputs & {p.resolve() for p in [args.cases, args.index_path, *args.corpus]}:
        parser.error("output must not overwrite evaluation inputs")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    json_path, md_path = write_reports(report, args.output)
    print(json.dumps({"status": report["status"], "json_report": str(json_path), "markdown_report": str(md_path)}, ensure_ascii=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
