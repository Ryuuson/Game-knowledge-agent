import copy
import json
import sqlite3

import numpy as np
import pytest

from scripts.repair_corpus_metadata import apply_repair, mechanic_excerpt
from wiki_corpus.provenance import HTMLSourceText, describe_location, locate_text, source_link, source_path, update_index_metadata
from wiki_corpus.vector_index import create_schema, replace_chunks
from wiki_corpus.vector_search import load_index


def test_locations_use_original_lines_and_mark_normalization():
    source = "# title\n\nhello   world\nsecond line\n"
    assert locate_text("hello   world\nsecond line", source) == {
        "start_line": 3, "end_line": 4, "verification": "exact", "match_count": 1}
    assert locate_text("hello world second line", source)["verification"] == "whitespace_normalized"
    assert locate_text("hello world second line", source)["end_line"] == 4
    assert locate_text("missing text", source) is None


def test_formatting_marks_are_not_presented_as_exact_matches():
    located = locate_text("1 choose settings", "header\n1️⃣ choose\u200d settings")
    assert located["verification"] == "formatting_normalized"
    assert located["start_line"] == located["end_line"] == 2


def test_multiple_occurrences_and_empty_excerpts():
    assert locate_text("repeat", "repeat\nrepeat")["match_count"] == 2
    with pytest.raises(ValueError):
        locate_text(" ", "hello")


def test_html_locations_ignore_executable_and_hidden_text():
    html = HTMLSourceText("<html>\n<style>secret CSS</style>\n<script>secret JS</script>\n<p>damage &lt; 20</p>\n</html>")
    assert "secret" not in html.text
    hit = locate_text("damage < 20", html.text, html.lines)
    assert hit["start_line"] == hit["end_line"] == 4
    # A removed numeric constraint is not equivalent to the original source.
    assert locate_text("damage 20", html.text, html.lines) is None


def test_encoded_newline_does_not_invent_html_source_lines():
    html = HTMLSourceText("<p>first&#10;second</p>\n<p>third</p>")
    assert locate_text("second", html.text, html.lines)["start_line"] == 1
    assert locate_text("third", html.text, html.lines)["start_line"] == 2


@pytest.mark.parametrize("path", ["../secret.md", "/secret.md", "C:/secret.md", "a/../b.md", "a\\b.md"])
def test_source_paths_are_bounded(path):
    with pytest.raises(ValueError):
        source_path(path)


def test_links_encode_unicode_paths_and_pin_versions():
    url = source_link("owner/repo", "a" * 40, "wiki/中文.md", 2, 4)
    assert "blob/" + "a" * 40 in url and "%E4%B8%AD" in url and url.endswith("#L2-L4")


def test_json_record_location_never_claims_literal_excerpt():
    description = describe_location({"start_line": 1, "end_line": 100,
                                     "provenance": {"line_scope": "json_record", "revision": "a" * 40}})
    assert "非逐字摘录" in description and "原文行" not in description
    assert "尚未定位片段" in describe_location({"provenance": {"line_scope": "document"}})


def index_fixture(connection):
    chunks = [{"chunk_id": "id", "title": "title", "text": "evidence", "section_path": "section",
               "source_url": "old.md", "start_line": 0, "end_line": 0, "license": "TBD"}]
    create_schema(connection)
    vector = np.array([.3, .7], dtype=np.float32).tobytes()
    replace_chunks(connection, chunks, [vector], model_name="fixture", embedding_dimensions=2)
    repaired = copy.deepcopy(chunks)
    repaired[0].update(source_url="https://github.com/o/r/blob/revision/source.md#L2-L3",
                       start_line=2, end_line=3, license="MIT",
                       provenance={"line_scope": "matched_text", "verification": "exact"})
    return chunks, repaired, vector


def test_metadata_update_preserves_vectors_and_is_idempotent():
    with sqlite3.connect(":memory:") as connection:
        _, repaired, vector = index_fixture(connection)
        first = update_index_metadata(connection, repaired)
        assert update_index_metadata(connection, repaired) == first
        chunks, vectors, _ = load_index(connection)
        assert chunks[0]["provenance"] == repaired[0]["provenance"]
        assert chunks[0]["license"] == "MIT" and chunks[0]["start_line"] == 2
        assert vectors.tobytes() == vector


def test_changed_embedding_input_rejects_metadata_repair_before_mutation():
    with sqlite3.connect(":memory:") as connection:
        _, repaired, _ = index_fixture(connection)
        repaired[0]["text"] = "changed content"
        with pytest.raises(ValueError, match="index_embedding_input_mismatch"):
            update_index_metadata(connection, repaired)
        assert connection.execute("SELECT source_url,license FROM chunks").fetchone() == ("old.md", "TBD")


def test_legacy_index_loading_and_schema_upgrade():
    with sqlite3.connect(":memory:") as connection:
        _, repaired, vector = index_fixture(connection)
        connection.execute("ALTER TABLE chunks DROP COLUMN provenance")
        assert load_index(connection)[0][0]["provenance"] == {}
        update_index_metadata(connection, repaired)
        assert load_index(connection)[1].tobytes() == vector


def test_failed_index_validation_does_not_change_corpus_or_annotations(tmp_path):
    corpus, cases, index = tmp_path / "corpus.jsonl", tmp_path / "cases.jsonl", tmp_path / "index.sqlite"
    with sqlite3.connect(index) as connection:
        chunks, repaired, _ = index_fixture(connection)
    corpus.write_text(json.dumps(chunks[0]) + "\n", encoding="utf-8")
    cases.write_text(json.dumps({"evidence": []}) + "\n", encoding="utf-8")
    original = [p.read_bytes() for p in (corpus, cases, index)]
    repaired[0]["title"] = "changed"
    with pytest.raises(ValueError, match="index_embedding_input_mismatch"):
        apply_repair([repaired], [corpus], cases, [index], tmp_path / "backup")
    assert [p.read_bytes() for p in (corpus, cases, index)] == original


def test_successful_repair_replaces_closed_index_and_updates_annotations(tmp_path):
    corpus, cases, index = tmp_path / "corpus.jsonl", tmp_path / "cases.jsonl", tmp_path / "index.sqlite"
    connection = sqlite3.connect(index)
    chunks, repaired, vector = index_fixture(connection)
    connection.close()
    corpus.write_text(json.dumps(chunks[0]) + "\n", encoding="utf-8")
    cases.write_text(json.dumps({"evidence": [{"chunk_id": "id", "excerpt": "evidence"}]}) + "\n", encoding="utf-8")
    result = apply_repair([repaired], [corpus], cases, [index], tmp_path / "backup")
    assert result[index.name]["vectors_unchanged"]
    assert json.loads(cases.read_text())["evidence"][0]["start_line"] == 2
    connection = sqlite3.connect(index)
    try:
        assert load_index(connection)[0][0]["provenance"] == repaired[0]["provenance"]
        assert load_index(connection)[1].tobytes() == vector
    finally:
        connection.close()


def test_public_corpus_contains_repaired_metadata_without_placeholders():
    from scripts.repair_corpus_metadata import CORPUS, read_jsonl
    chunks = [c for path in CORPUS for c in read_jsonl(path)]
    assert len({c["chunk_id"] for c in chunks}) == len(chunks) == 6515
    for chunk in chunks:
        provenance = chunk["provenance"]
        assert len(provenance["revision"]) == 40 and len(provenance["source_sha256"]) == 64
        assert chunk["source_url"].startswith("https://github.com/")
        assert provenance["revision"] in chunk["source_url"]
        assert 0 < chunk["start_line"] <= chunk["end_line"]
        assert chunk["license"] in {"MIT", "CC0-1.0", None}
        assert provenance["line_scope"] in {"matched_text", "json_record", "document"}
        if chunk["license"] is None:
            assert provenance["license"]["status"] == "not_declared"
        else:
            assert provenance["license"]["evidence_url"].startswith("https://github.com/")


def test_structured_projection_verifies_field_values():
    record = {"name": "Cooldown", "description": "Time before using again", "parameters": [
        {"name": "seconds", "typical_range": "1-10", "description": "duration"}]}
    assert mechanic_excerpt(record, "定义") == "定义（Cooldown）：Time before using again"
    assert mechanic_excerpt(record, "参数") == "参数（Cooldown）：seconds（1-10）：duration"
