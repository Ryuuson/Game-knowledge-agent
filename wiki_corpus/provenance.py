"""Source locations and metadata-only index updates; no model or network calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from html.parser import HTMLParser
from html import unescape
from pathlib import PurePosixPath
from urllib.parse import quote


def source_path(value: str) -> str:
    """Accept a repository-relative POSIX path, never an absolute/traversal path."""
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or any(part in {".", ".."} for part in value.split("/"))):
        raise ValueError("unsafe_source_path")
    return path.as_posix()


def source_link(repository: str, revision: str, path: str, start=0, end=0) -> str:
    url = f"https://github.com/{repository}/blob/{revision}/{quote(source_path(path), safe='/')}"
    return url + (f"#L{start}-L{end}" if 0 < start <= end else "")


def _normalized(text, lines, *, formatting=False):
    chars, locations = [], []
    for char, line in zip(text, lines):
        if char.isspace() or char == "\ufeff":
            continue
        if formatting and unicodedata.category(char) in {"So", "Mn", "Me", "Cf"}:
            continue
        chars.append(char)
        locations.append(line)
    return "".join(chars), locations


def text_lines(text):
    locations, line = [], 1
    for char in text:
        locations.append(line)
        line += char == "\n"
    return locations


def locate_text(excerpt, text, lines=None):
    """Locate exact or explicitly normalized text; return None if it is absent.

    Normalization removes whitespace/BOM, then optionally symbols/combining/format marks.
    The returned lines belong to the source file, never to the extracted chunk.
    Multiple matches are reported rather than asserted to be unique.
    """
    lines = text_lines(text) if lines is None else lines
    if len(lines) != len(text) or not excerpt.strip():
        raise ValueError("invalid_text_location_input")
    for mode in ("exact", "whitespace_normalized", "formatting_normalized"):
        if mode == "exact":
            haystack, locations, needle = text, lines, excerpt
        else:
            formatting = mode == "formatting_normalized"
            haystack, locations = _normalized(text, lines, formatting=formatting)
            needle, _ = _normalized(excerpt, [0] * len(excerpt), formatting=formatting)
        if not needle:
            continue
        offset = haystack.find(needle)
        if offset >= 0:
            occurrences = haystack.count(needle)
            return {"start_line": locations[offset], "end_line": locations[offset + len(needle) - 1],
                    "verification": mode, "match_count": occurrences}
    return None


class HTMLSourceText(HTMLParser):
    """Extract visible data with original HTML line numbers; ignore scripts/CSS."""

    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.parts, self.lines, self.skipped = [], [], []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skipped.append(tag)

    def handle_endtag(self, tag):
        if self.skipped and tag == self.skipped[-1]:
            self.skipped.pop()

    def handle_data(self, data):
        if self.skipped:
            return
        line = self.getpos()[0]
        self.parts.append(data)
        for char in data:
            self.lines.append(line)
            line += char == "\n"

    def handle_entityref(self, name):
        self._entity("&" + name + ";")

    def handle_charref(self, name):
        self._entity("&#" + name + ";")

    def _entity(self, encoded):
        if not self.skipped:
            decoded = unescape(encoded)
            self.parts.append(decoded)
            # Encoded newlines do not advance the original HTML source line.
            self.lines.extend([self.getpos()[0]] * len(decoded))

    @property
    def text(self):
        return "".join(self.parts)


def describe_location(hit):
    provenance = hit.get("provenance") or {}
    revision = provenance.get("revision", "")
    scope = provenance.get("line_scope")
    if scope == "matched_text":
        label = f"原文行 {hit['start_line']}–{hit['end_line']}"
        if provenance.get("verification") != "exact":
            label += "（忽略格式差异匹配）"
        if provenance.get("match_count", 1) > 1:
            label += "（原文多处出现，链接定位首处）"
    elif scope == "json_record":
        label = "JSON 记录范围；片段为整理文本，非逐字摘录"
    elif scope == "document":
        label = "来源文章；尚未定位片段行范围"
    else:
        label = "来源位置尚未核实"
    return label + (f" · 来源版本 {revision[:12]}" if revision else "")


def update_index_metadata(connection: sqlite3.Connection, chunks):
    """Reject changed embedding inputs, update provenance only, preserve BLOBs.

    Works with a subset index, but every served row must exist in the corpus.
    Validation happens before schema/data changes; transaction ownership is caller's.
    """
    by_id = {c["chunk_id"]: c for c in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("duplicate_chunk_id")
    rows = connection.execute("SELECT chunk_id,title,section_path,content,embedding FROM chunks ORDER BY rowid").fetchall()
    updates, digest = [], hashlib.sha256()
    for cid, title, section, content, vector in rows:
        chunk = by_id.get(cid)
        if chunk is None or (title, section, content) != (chunk["title"], chunk.get("section_path"), chunk["text"]):
            raise ValueError("index_embedding_input_mismatch")
        digest.update(cid.encode("utf-8"))
        digest.update(vector)
        updates.append((chunk["source_url"], chunk["start_line"], chunk["end_line"], chunk.get("license"),
                        json.dumps(chunk.get("provenance", {}), ensure_ascii=False, sort_keys=True), cid))
    columns = {r[1] for r in connection.execute("PRAGMA table_info(chunks)")}
    if "provenance" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN provenance TEXT NOT NULL DEFAULT '{}'")
    connection.executemany("UPDATE chunks SET source_url=?,start_line=?,end_line=?,license=?,provenance=? WHERE chunk_id=?", updates)
    after = hashlib.sha256()
    for cid, vector in connection.execute("SELECT chunk_id,embedding FROM chunks ORDER BY rowid"):
        after.update(cid.encode("utf-8"))
        after.update(vector)
    if digest.digest() != after.digest():
        raise ValueError("index_vectors_changed")
    return {"rows": len(rows), "vectors_sha256": digest.hexdigest(), "vectors_unchanged": True}
