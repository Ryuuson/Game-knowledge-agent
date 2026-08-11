"""Selective durable memory using OpenViking's Session extraction pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from openviking_client import (
    OpenVikingClient,
    OpenVikingConflict,
    OpenVikingNotFound,
)


MEMORY_ROOT = "viking://user/memories/"
MEMORY_TYPES = ("preferences", "entities", "events")
MAX_MEMORY_ITEMS = 6
MAX_MEMORY_ITEM_CHARS = 600
_SECRET_PATTERN = re.compile(
    r"\b(api[_ -]?key|password|secret|token|credential|bearer)\b"
    r"|\bsk-[A-Za-z0-9_-]{12,}\b|\bAKIA[A-Z0-9]{16}\b"
    r"|密码|密钥|令牌|凭据",
    re.IGNORECASE,
)
_LIST_PREFIX_PATTERN = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class MemoryItem:
    uri: str
    content: str


class DurableMemory:
    def __init__(self, client: OpenVikingClient) -> None:
        self.client = client

    def recall(self, query: str) -> str:
        matches = self.client.find(
            query,
            MEMORY_ROOT,
            limit=MAX_MEMORY_ITEMS,
            context_type="memory",
        )
        items: list[str] = []
        seen_uris: set[str] = set()
        for match in matches:
            uri = str(match.get("uri", ""))
            if not self._is_managed_memory_uri(uri) or uri in seen_uris:
                continue
            try:
                content = self.client.read(uri).strip()
            except OpenVikingNotFound:
                continue
            if content:
                seen_uris.add(uri)
                items.append(content[:MAX_MEMORY_ITEM_CHARS])
        return "\n\n".join(items)

    def capture(
        self,
        thread_id: str,
        user_prompt: str,
        assistant_reply: str,
        extract: Callable[[str], str],
    ) -> dict | None:
        candidate_text = extract(
            "用户消息：\n"
            f"{user_prompt}\n\n"
            "助手最终回答：\n"
            f"{assistant_reply}"
        )
        items = self._normalize_candidates(candidate_text)
        if not items:
            return None

        session_id = self._session_id(thread_id)
        try:
            self.client.create_session(session_id, memory_types=list(MEMORY_TYPES))
        except OpenVikingConflict:
            pass
        facts = "\n".join(f"- {item}" for item in items)
        self.client.batch_add_messages(
            session_id,
            [
                {
                    "role": "user",
                    "content": "请长期记住以下由用户明确确认的信息：\n" + facts,
                },
                {
                    "role": "assistant",
                    "content": "已确认，仅将这些稳定事实用于后续对话。",
                },
            ],
        )
        return self.client.commit_session(session_id)

    def list(self) -> list[MemoryItem]:
        try:
            entries = self.client.list(MEMORY_ROOT, limit=200)
        except OpenVikingNotFound:
            return []
        memories: list[MemoryItem] = []
        for entry in entries:
            uri = str(entry.get("uri") or entry.get("path") or "")
            if not self._is_managed_memory_uri(uri) or not uri.endswith(".md"):
                continue
            try:
                content = self.client.read(uri).strip()
            except OpenVikingNotFound:
                continue
            if content:
                memories.append(MemoryItem(uri=uri, content=content))
        return memories

    def delete(self, uri: str) -> None:
        if not self._is_managed_memory_uri(uri) or not uri.endswith(".md"):
            raise ValueError("只能删除当前用户的偏好、项目实体或事件记忆。")
        self.client.delete(uri)

    @staticmethod
    def _session_id(thread_id: str) -> str:
        digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:24]
        return f"gka-{digest}"

    @staticmethod
    def _is_managed_memory_uri(uri: str) -> bool:
        if not uri.startswith("viking://user/"):
            return False
        normalized = uri.rstrip("/") + "/"
        return any(f"/memories/{memory_type}/" in normalized for memory_type in MEMORY_TYPES)

    @staticmethod
    def _normalize_candidates(value: str) -> list[str]:
        if value.strip().upper() == "NO_MEMORY":
            return []
        items: list[str] = []
        for line in value.splitlines():
            normalized = _LIST_PREFIX_PATTERN.sub("", line.strip()).strip()
            if not normalized or _SECRET_PATTERN.search(normalized):
                continue
            if len(normalized) > MAX_MEMORY_ITEM_CHARS:
                normalized = normalized[:MAX_MEMORY_ITEM_CHARS].rstrip()
            if normalized not in items:
                items.append(normalized)
            if len(items) == MAX_MEMORY_ITEMS:
                break
        return items
