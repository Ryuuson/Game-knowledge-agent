"""Print questions whose dense and hybrid retrieval titles differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="JSON reports from evaluate_rag.py")
    parser.add_argument("--threshold", default="0.62")
    parser.add_argument("--case", help="Only show one case ID, such as user-11")
    args = parser.parse_args()

    changed = 0
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        dense = report["reports"]["dense"][args.threshold]["records"]
        hybrid = report["reports"]["hybrid"][args.threshold]["records"]
        for before, after in zip(dense, hybrid):
            if args.case and before["id"] != args.case:
                continue
            if not args.case and before["titles"] == after["titles"]:
                continue
            changed += 1
            print(f"\n[{path.stem} / {before['id']}] {before['question']}")
            for label, record in (("BGE", before), ("混合", after)):
                print(f"\n{label}（{record['band']}）：")
                for position, evidence in enumerate(record.get("evidence", []), start=1):
                    ranks = []
                    if evidence.get("dense_rank"):
                        ranks.append(f"dense #{evidence['dense_rank']}")
                    if evidence.get("lexical_rank"):
                        ranks.append(f"BM25 #{evidence['lexical_rank']}")
                    rank_text = f"，{'；'.join(ranks)}" if ranks else ""
                    print(
                        f"  [{position}] {evidence['title']} | {evidence['collection']} | "
                        f"语义 {evidence['semantic_score']:.3f}{rank_text}\n"
                        f"      章节：{evidence['section']}\n"
                        f"      来源：{evidence['source']}\n"
                        f"      摘要：{evidence['snippet']}"
                    )
    print(f"\n共 {changed} 条题目的 Top {report['top_k']} 标题发生变化。")


if __name__ == "__main__":
    main()
