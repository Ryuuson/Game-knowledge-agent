"""Bounded local document access; callers must supply an allowed root."""

from pathlib import Path
from uuid import uuid4
from datetime import datetime

TEXT_SUFFIXES = {".md", ".txt"}


def resolve_document(root: Path, filename: str) -> Path:
    if filename != Path(filename).as_posix():
        raise ValueError("文件名包含非法字符或路径格式不正确。")
    resolved = (root / filename).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("不允许访问 knowledge 目录之外的文件。")
    return resolved


def read_document(root: Path, filename: str, *, max_bytes=100 * 1024) -> str:
    try:
        path = resolve_document(root, filename)
    except ValueError as exc:
        return str(exc)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return "只允许读取 .md 和 .txt 文件。"
    if not path.is_file():
        return f"没有找到文件：{filename}"
    if path.stat().st_size > max_bytes:
        return f"文件超过 {max_bytes // 1024} KB，请用 read_document_section 分段读取。"
    return path.read_text(encoding="utf-8")


def read_section(root: Path, filename: str, start_line: int, end_line: int) -> str:
    try:
        path = resolve_document(root, filename)
    except ValueError as exc:
        return str(exc)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return "只允许读取 .md 和 .txt 文件。"
    if not path.is_file():
        return f"没有找到文件：{filename}"
    if start_line < 1 or end_line < start_line or end_line - start_line >= 120:
        return "行号范围不合法，单次最多读取 120 行。"
    if path.stat().st_size > 5 * 1024 * 1024:
        return "文件超过分段读取上限（5 MB）。"
    lines = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            total = number
            if start_line <= number <= end_line:
                lines.append(f"{number}: {line.rstrip()}")
            if number >= end_line:
                break
    if not lines:
        return f"起始行超出文件范围；该文件共 {total} 行。"
    text = "\n".join(lines)
    return f"{filename} 第 {start_line}-{min(total, end_line)} 行：\n{text[:14000]}"


def search_documents(root: Path, query: str) -> str:
    query = query.strip()
    if not query or len(query) > 2000:
        return "查询关键词应为 1–2000 个字符。"
    matches = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 1024 * 1024:
            continue
        scanned += 1
        if scanned > 5000:
            break
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if query.casefold() in line.casefold():
                    matches.append(f"{path.relative_to(root).as_posix()} 第 {number} 行：{line.strip()[:1000]}")
                    if len(matches) >= 10:
                        return "\n".join(matches)
    return "\n".join(matches) if matches else f"没有找到包含“{query}”的内容。"


def save_note(root: Path, content: str) -> str:
    content = content.strip()
    if not content:
        return "笔记内容为空，未保存。"
    if len(content) > 20000:
        return "笔记超过 20000 字符，未保存。"
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y-%m-%d_%H-%M-%S}-{uuid4().hex[:8]}.md"
    with (root / filename).open("x", encoding="utf-8") as handle:
        handle.write(content)
    return f"已保存笔记：{filename}"
