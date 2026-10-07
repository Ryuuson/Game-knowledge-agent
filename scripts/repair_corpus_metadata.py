"""Verify public source snapshots and repair corpus metadata without re-embedding.

Requires existing upstream checkouts under knowledge/. Does not fetch or run them.
Default: dry run. --apply backs up inputs and updates JSONL, annotations and indices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wiki_corpus.provenance import HTMLSourceText, locate_text, source_link, source_path, update_index_metadata

SOURCES = {
    "game_design_wiki": ("Being09/game-design-wiki", "game-design-wiki", "MIT", "LICENSE"),
    "game_knowledge_base": ("diedie23/Game-Knowledge-Base", "Game-Knowledge-Base", None, None),
    "open_game_mechanics_dataset": ("Thaelith/open-game-mechanics-dataset", "open-game-mechanics-dataset", "CC0-1.0", "LICENSE"),
    "game_num_basics": ("lsc1414/Game_Num_Basics_And_Calc", "Game_Num_Basics_And_Calc", "MIT", "LICENSE"),
    "gamedev_at_home": ("zsc/gamedev_at_home", "gamedev_at_home", None, None),
    "senior_game_designer": ("tigermkiiiddd/senior-game-designer", "senior-game-designer", "MIT", "README.md"),
}
CORPUS = [ROOT / "data/game_knowledge_chunks.jsonl", ROOT / "data/new_knowledge_chunks.jsonl"]
CASES = ROOT / "evals/retrieval_cases.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")


def mechanic_excerpt(record, label):
    """Reproduce the existing structured projection to verify derived content."""
    name = record["name"]
    if label == "机制":
        return (f"机制：{name}\n分类：{record['category']}\n子分类：{record['subcategory']}\n"
                f"别名：{'、'.join(record['aliases'])}\n适用品类：{'、'.join(record['genres'])}")
    fields = {"定义": "description", "设计要点": "design_notes", "设计目的": "design_purpose",
              "玩家体验": "player_fantasy", "参数": "parameters", "平衡要点": "balancing_notes",
              "常见变体": "common_variants", "边界情况": "edge_cases", "常见错误": "common_bugs",
              "案例": "example_games", "常搭配": "combines_well_with"}
    value = record[fields[label]]
    if label == "参数":
        value = "；".join(f"{p['name']}（{p['typical_range']}）：{p['description']}" for p in value)
    elif label == "案例":
        value = "；".join(f"{p['title']}：{p['note']}" for p in value)
    elif isinstance(value, list):
        value = ("、" if label in {"常见变体", "常搭配"} else "；").join(value)
    return f"{label}（{name}）：{value}"


class Snapshot:
    def __init__(self, root, repository):
        self.root, self.repository = root.resolve(), repository
        self.revision = self.git("rev-parse", "HEAD").decode().strip()
        self.blobs = {}
        for entry in self.git("ls-tree", "-rz", "HEAD").split(b"\0"):
            if not entry:
                continue
            info, name = entry.split(b"\t", 1)
            _, kind, sha = info.decode().split()
            if kind == "blob":
                self.blobs[name.decode("utf-8")] = sha
        self.cache = {}

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "safe.directory=" + str(self.root),
                                        "-C", str(self.root), *args], stderr=subprocess.PIPE)

    def read(self, path):
        path = source_path(path)
        if path not in self.cache:
            if path not in self.blobs:
                raise ValueError(f"untracked_source_file: {self.repository}/{path}")
            file = (self.root / path).resolve()
            if not file.is_relative_to(self.root):
                raise ValueError("source_escapes_snapshot")
            data = file.read_bytes()
            # Git checkouts may contain CRLF; source line counts are unchanged.
            lf = data.replace(b"\r\n", b"\n")
            sha = hashlib.sha1(b"blob " + str(len(lf)).encode() + b"\0" + lf).hexdigest()
            if sha != self.blobs[path]:
                raise ValueError(f"source_snapshot_content_mismatch: {self.repository}/{path}")
            self.cache[path] = (lf.decode("utf-8"), hashlib.sha256(lf).hexdigest())
        return self.cache[path]


def verify_license(snapshot, expected, evidence_path):
    if evidence_path is None:
        # Absence is a source declaration status, never a guessed license.
        if any(Path(p).name.lower().startswith(("license", "licence", "copying")) for p in snapshot.blobs):
            raise ValueError("unexpected_license_file_requires_review")
        return {"status": "not_declared", "spdx": None}
    text, sha = snapshot.read(evidence_path)
    if expected == "CC0-1.0":
        if "SPDX-License-Identifier: CC0-1.0" not in text or "dataset content" not in text:
            raise ValueError("unsupported_dataset_license")
    elif evidence_path == "README.md":
        if "**License:** MIT License" not in text:
            raise ValueError("unsupported_readme_license")
    elif "MIT License" not in text or "Permission is hereby granted" not in text:
        raise ValueError("unsupported_mit_license")
    return {"status": "readme_declaration" if evidence_path == "README.md" else "license_file",
            "spdx": expected, "evidence_url": source_link(snapshot.repository, snapshot.revision, evidence_path),
            "evidence_sha256": sha}


def plan_repair(corpus_groups, *, knowledge_root, verified_at):
    snapshots, licenses = {}, {}
    for collection, (repository, directory, license_name, evidence) in SOURCES.items():
        snapshot = Snapshot(knowledge_root / directory, repository)
        snapshots[collection] = snapshot
        licenses[collection] = verify_license(snapshot, license_name, evidence)
    base = snapshots["game_knowledge_base"]
    search, _ = base.read("docs/search-index.json")
    articles, parsed_html = {}, {}
    def parse_html(path):
        if path not in parsed_html:
            text, _ = base.read(path)
            parsed_html[path] = HTMLSourceText(text)
        return parsed_html[path]
    for entry in json.loads(search)["entries"]:
        path = f"docs/knowledge-base/{entry['id']}.html"
        if path not in base.blobs:
            candidates = []
            for candidate in base.blobs:
                if candidate.startswith("docs/knowledge-base/") and candidate.endswith(".html"):
                    html = parse_html(candidate)
                    if locate_text(entry["content"][:220], html.text, html.lines):
                        candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(f"ambiguous_article_path: {entry['id']}: {candidates}")
            path = candidates[0]
        articles[entry["title"]] = (path, parse_html(path))
    groups, unresolved = [], []
    for corpus in corpus_groups:
        repaired = []
        for original in corpus:
            chunk = copy.deepcopy(original)
            collection = chunk["source_collection"]
            snapshot = snapshots[collection]
            old = (chunk.get("provenance") or {}).get("original_source_url", chunk["source_url"])
            if collection == "game_knowledge_base":
                path, html = articles[chunk["title"]]
                location = locate_text(chunk["text"], html.text, html.lines)
            else:
                prefix = SOURCES[collection][1] + "/" if collection == "open_game_mechanics_dataset" else collection + "/"
                if collection == "game_design_wiki":
                    path = "wiki/" + old
                else:
                    if not old.startswith(prefix):
                        raise ValueError("unrecognized_original_source_path")
                    path = old[len(prefix):]
                location = None
            text, sha = snapshot.read(path)
            provenance = {"schema_version": 1, "repository": snapshot.repository, "revision": snapshot.revision,
                          "source_path": path, "source_sha256": sha, "verified_at": verified_at,
                          "original_source_url": old, "license": licenses[collection]}
            if collection == "open_game_mechanics_dataset":
                record = json.loads(text)
                if record["name"] != chunk["title"] or not chunk["section_path"].startswith(record["category"] + " > "):
                    raise ValueError("json_source_record_identity_mismatch")
                label = chunk["text"].split("（", 1)[0].split("：", 1)[0]
                if mechanic_excerpt(record, label) != chunk["text"]:
                    raise ValueError(f"json_source_projection_mismatch: {chunk['chunk_id']}")
                location = {"start_line": 1, "end_line": len(text.splitlines()), "verification": "structured_projection"}
                provenance.update(line_scope="json_record", record_id=record["id"], json_pointer="",
                                  content_status="derived_text_not_literal_excerpt")
            else:
                if collection != "game_knowledge_base":
                    location = locate_text(chunk["text"], text)
                provenance["line_scope"] = "matched_text" if location else "document"
            if location:
                chunk["start_line"], chunk["end_line"] = location["start_line"], location["end_line"]
                provenance.update({k: v for k, v in location.items() if k not in {"start_line", "end_line"}})
            else:
                # A real document range is useful, but is not a fragment match.
                chunk["start_line"], chunk["end_line"] = 1, len(text.splitlines())
                provenance["verification"] = "document_identified_excerpt_unmatched"
                provenance["content_status"] = "excerpt_not_verified_against_snapshot"
                unresolved.append(chunk["chunk_id"])
            chunk["license"] = licenses[collection]["spdx"]
            chunk["source_url"] = source_link(snapshot.repository, snapshot.revision, path, chunk["start_line"], chunk["end_line"])
            chunk["provenance"] = provenance
            repaired.append(chunk)
        groups.append(repaired)
    chunks = [r for group in groups for r in group]
    report = {"verified_at": verified_at, "chunks": len(chunks),
              "scope_counts": dict(Counter(c["provenance"]["line_scope"] for c in chunks)),
              "verification_counts": dict(Counter(c["provenance"]["verification"] for c in chunks)),
              "license_counts": dict(Counter(c["license"] or "not_declared" for c in chunks)),
              "sources": {collection: {"repository": s.repository, "revision": s.revision,
                                       "license": licenses[collection],
                                       "chunks": sum(c["source_collection"] == collection for c in chunks)}
                          for collection, s in snapshots.items()},
              "unmatched_excerpt_ids": unresolved,
              "ingestion_date": "unknown; verified_at is a verification date, not the original ingestion date"}
    return groups, report


def apply_repair(groups, paths, cases_path, indices, backup_root):
    chunks = [r for group in groups for r in group]
    by_id = {r["chunk_id"]: r for r in chunks}
    cases = read_jsonl(cases_path)
    for case in cases:
        for evidence in case.get("evidence", []):
            chunk = by_id[evidence["chunk_id"]]
            if evidence["excerpt"] not in chunk["text"]:
                raise ValueError("annotation_excerpt_mismatch")
            for field in ("source_url", "start_line", "end_line"):
                evidence[field] = chunk[field]
    outputs = dict(zip(paths, map(jsonl_bytes, groups)))
    outputs[cases_path] = jsonl_bytes(cases)
    backup_root.mkdir(parents=True, exist_ok=False)
    backups, temporary_indices, results = {}, {}, {}
    try:
        for number, path in enumerate(outputs):
            destination = backup_root / f"{number}-{path.name}"
            shutil.copy2(path, destination)
            backups[path] = destination
        for number, path in enumerate(indices):
            destination = backup_root / f"index-{number}-{path.name}"
            temporary = backup_root / f"repaired-{number}-{path.name}"
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                with closing(sqlite3.connect(destination)) as backup:
                    connection.backup(backup)
            shutil.copy2(destination, temporary)
            with closing(sqlite3.connect(temporary)) as connection:
                with connection:
                    results[path.name] = update_index_metadata(connection, chunks)
            backups[path], temporary_indices[path] = destination, temporary
        # All validation/updates above happen on copies. Replace files only now.
        for path, data in outputs.items():
            temporary = path.with_suffix(path.suffix + ".metadata-tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        for path, temporary in temporary_indices.items():
            temporary.replace(path)
    except Exception:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
        raise
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verified-at", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--index", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/corpus_metadata_repair.json")
    args = parser.parse_args(argv)
    groups, report = plan_repair([read_jsonl(p) for p in CORPUS], knowledge_root=ROOT / "knowledge", verified_at=args.verified_at)
    if args.apply:
        report["indices"] = apply_repair(groups, CORPUS, CASES, args.index,
                                         ROOT / ".runtime" / ("metadata-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    report["applied"] = args.apply
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("applied", "chunks", "scope_counts", "license_counts", "unmatched_excerpt_ids")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
