"""Tests for save_note file naming: same-second saves must not overwrite."""

import Agent


def test_save_note_avoids_same_second_collision(monkeypatch, tmp_path):
    monkeypatch.setattr(Agent, "NOTES_DIR", tmp_path)
    monkeypatch.setenv("ENABLE_NOTE_WRITES", "true")

    first = Agent.save_note.func("第一条笔记")
    second = Agent.save_note.func("第二条笔记")

    assert first != second
    assert "已保存笔记：" in first
    assert "已保存笔记：" in second
    assert len(list(tmp_path.glob("*.md"))) == 2
    assert (tmp_path / first.removeprefix("已保存笔记：")).read_text(encoding="utf-8") == "第一条笔记"
    assert (tmp_path / second.removeprefix("已保存笔记：")).read_text(encoding="utf-8") == "第二条笔记"


def test_save_note_rejects_blank_content(monkeypatch, tmp_path):
    monkeypatch.setattr(Agent, "NOTES_DIR", tmp_path)
    monkeypatch.setenv("ENABLE_NOTE_WRITES", "true")

    result = Agent.save_note.func("   ")

    assert result == "笔记内容为空，未保存。"
    assert list(tmp_path.glob("*.md")) == []


def test_save_note_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(Agent, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("ENABLE_NOTE_WRITES", raising=False)

    result = Agent.save_note.func("不应写入")

    assert result.startswith("笔记写入已禁用")
    assert list(tmp_path.glob("*.md")) == []
