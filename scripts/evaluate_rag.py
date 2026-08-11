"""Compare dense and hybrid local retrieval against a labeled routing case set.

The report measures retrieval routing only. A returned hit is not evidence that
the final answer is correct; inspect representative hits before changing gates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from wiki_corpus.domain_signals import classify_query_domain
from wiki_corpus.hybrid_search import BM25Index, rank_hybrid
from wiki_corpus.vector_search import load_index, rank_chunks


DEFAULT_INDEX = PROJECT_ROOT / "data" / "game_knowledge_bge_combined_index.sqlite"
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSON/JSONL cases containing id, question, and optional expectations."""
    if path.suffix == ".jsonl":
        raw_cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = (
            payload.get("cases", payload.get("results", payload))
            if isinstance(payload, dict)
            else payload
        )

    if not isinstance(raw_cases, list):
        raise ValueError("case file must contain a list or a results list")

    cases = []
    for position, raw_case in enumerate(raw_cases, start=1):
        question = str(raw_case.get("question", "")).strip()
        if not question:
            continue
        cases.append(
            {
                "id": str(raw_case.get("id", position)),
                "question": question,
                "type": str(raw_case.get("type", raw_case.get("kind", "unlabeled"))),
                "expectations": raw_case.get("expectations", {}),
            }
        )
    if not cases:
        raise ValueError("case file contains no usable questions")
    return cases


def routing_band(hits: list[dict[str, Any]], lower_threshold: float, high_threshold: float) -> str:
    """Apply the same semantic-score routing semantics as Agent.py."""
    accepted = [hit for hit in hits if hit.get("score", 0.0) >= lower_threshold]
    if not accepted:
        return "rejected"
    return "high_confidence" if max(hit["score"] for hit in accepted) >= high_threshold else "ambiguous"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize routing outcomes without making claims about answer correctness."""
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_type[record["type"]]["total"] += 1
        by_type[record["type"]][record["band"]] += 1
    return {label: dict(counts) for label, counts in sorted(by_type.items())}


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not thresholds or any(value < -1 or value > 1 for value in thresholds):
        raise ValueError("thresholds must be comma-separated values between -1 and 1")
    return thresholds


def serialize_evidence(hit: dict[str, Any], snippet_chars: int) -> dict[str, Any]:
    """Keep the context needed for a human to assess a retrieved chunk."""
    text = str(hit.get("text", "")).strip().replace("\n", " ")
    if len(text) > snippet_chars:
        text = f"{text[:snippet_chars].rstrip()}..."
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "title": hit.get("title", ""),
        "collection": hit.get("collection_label", ""),
        "source": hit.get("source_url", ""),
        "section": hit.get("section_path", ""),
        "semantic_score": round(float(hit.get("score", 0.0)), 6),
        "dense_rank": hit.get("dense_rank"),
        "lexical_rank": hit.get("lexical_rank"),
        "snippet": text,
    }


def evidence_matches(evidence: dict[str, Any], expected: dict[str, str]) -> bool:
    """Match one evidence item against a reviewed title/section expectation."""
    title = expected.get("title")
    section_contains = expected.get("section_contains")
    return (
        (not title or evidence.get("title") == title)
        and (not section_contains or section_contains in evidence.get("section", ""))
    )


def evaluate_expectations(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Score reviewed evidence groups without treating unreviewed cases as failures."""
    reviewed = [record for record in records if record.get("expectations")]
    if not reviewed:
        return None

    results = []
    for record in reviewed:
        expectations = record["expectations"]
        domain_expected = expectations.get("domain_blocked")
        domain_passed = (
            domain_expected is None or record["domain_blocked"] == domain_expected
        )
        group_results = []
        for group in expectations.get("evidence_groups", []):
            matched = any(
                evidence_matches(evidence, expected)
                for evidence in record["evidence"]
                for expected in group["acceptable"]
            )
            group_results.append({"name": group["name"], "passed": matched})
        passed = domain_passed and all(group["passed"] for group in group_results)
        results.append(
            {
                "id": record["id"],
                "passed": passed,
                "domain_passed": domain_passed,
                "evidence_groups": group_results,
            }
        )
    return {
        "reviewed_cases": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "failed_cases": [result for result in results if not result["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON or JSONL file with question cases")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--thresholds", default="0.60,0.62,0.64,0.66")
    parser.add_argument("--high-threshold", type=float, default=0.67)
    parser.add_argument("--dense-candidates", type=int, default=20)
    parser.add_argument("--lexical-candidates", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-rrf-weight", type=float, default=1.0)
    parser.add_argument("--lexical-rrf-weight", type=float, default=0.25)
    parser.add_argument("--snippet-chars", type=int, default=600)
    parser.add_argument("--output", type=Path, help="Write JSON report to this path instead of stdout")
    args = parser.parse_args()

    if args.top_k <= 0 or min(args.dense_candidates, args.lexical_candidates, args.rrf_k, args.snippet_chars, args.dense_rrf_weight, args.lexical_rrf_weight) <= 0:
        parser.error("top-k, candidate counts, RRF settings, and snippet-chars must be greater than zero")
    thresholds = parse_thresholds(args.thresholds)
    cases = load_cases(args.cases)
    if not args.index.is_file():
        parser.error(f"index does not exist: {args.index}")

    connection = sqlite3.connect(f"file:{args.index.as_posix()}?mode=ro", uri=True)
    try:
        chunks, vectors, model_name = load_index(connection)
    finally:
        connection.close()
    if model_name != args.model:
        parser.error(f"index model is {model_name!r}, not {args.model!r}")

    embedder = SentenceTransformer(args.model, local_files_only=True)
    lexical_index = BM25Index(chunks)
    reports: dict[str, Any] = {}
    for method in ("dense", "hybrid"):
        records_by_threshold: dict[str, list[dict[str, Any]]] = {str(value): [] for value in thresholds}
        for case in cases:
            domain_blocked = classify_query_domain(case["question"]).classification == "clear_non_game"
            if domain_blocked:
                candidates: list[dict[str, Any]] = []
            else:
                query_vector = embedder.encode(
                    case["question"], convert_to_numpy=True, normalize_embeddings=True
                )
                if method == "dense":
                    candidates = rank_chunks(
                        query_vector, chunks, vectors, top_k=max(args.top_k, args.dense_candidates)
                    )
                else:
                    candidates = rank_hybrid(
                        case["question"],
                        query_vector,
                        chunks,
                        vectors,
                        lexical_index,
                        dense_candidates=max(args.top_k, args.dense_candidates),
                        lexical_candidates=max(args.top_k, args.lexical_candidates),
                        rrf_k=args.rrf_k,
                        dense_weight=args.dense_rrf_weight,
                        lexical_weight=args.lexical_rrf_weight,
                    )
            for threshold in thresholds:
                accepted = [hit for hit in candidates if hit.get("score", 0.0) >= threshold][: args.top_k]
                records_by_threshold[str(threshold)].append(
                    {
                        **case,
                        "domain_blocked": domain_blocked,
                        "band": routing_band(candidates, threshold, args.high_threshold),
                        "top_semantic_score": accepted[0]["score"] if accepted else None,
                        "titles": [hit.get("title", "") for hit in accepted],
                        "evidence": [serialize_evidence(hit, args.snippet_chars) for hit in accepted],
                    }
                )
        reports[method] = {
            threshold: {
                "summary_by_type": summarize(records),
                "expectation_evaluation": evaluate_expectations(records),
                "records": records,
            }
            for threshold, records in records_by_threshold.items()
        }

    report = {
        "note": "Routing report only; it does not establish retrieval relevance or answer quality.",
        "case_count": len(cases),
        "index": str(args.index),
        "embedding_model": model_name,
        "top_k": args.top_k,
        "high_confidence_threshold": args.high_threshold,
        "reports": reports,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote routing report for {len(cases)} cases to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
