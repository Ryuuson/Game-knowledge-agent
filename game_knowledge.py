"""OpenViking-backed game knowledge retrieval."""

from __future__ import annotations

import os

from openviking_client import (
    OpenVikingClient,
    OpenVikingError,
    OpenVikingNotFound,
)
from wiki_corpus.domain_signals import classify_query_domain


GAME_KNOWLEDGE_URI = "viking://resources/game-knowledge/"
DEFAULT_EVIDENCE_BUDGET = 12_000
MAX_RESULTS = 5


class GameKnowledgeRetriever:
    def __init__(
        self,
        client: OpenVikingClient,
        *,
        evidence_char_budget: int | None = None,
    ) -> None:
        self.client = client
        self.evidence_char_budget = evidence_char_budget or int(
            os.getenv("OPENVIKING_EVIDENCE_CHAR_BUDGET", str(DEFAULT_EVIDENCE_BUDGET))
        )
        if self.evidence_char_budget <= 0:
            raise ValueError("OPENVIKING_EVIDENCE_CHAR_BUDGET must be greater than zero")

    def search(self, query: str, top_k: int = 3) -> str:
        query = query.strip()
        if not query:
            return "查询内容不能为空。"
        signals = classify_query_domain(query)
        if signals.classification == "clear_non_game":
            matched = "、".join(signals.non_game_signals)
            return (
                f"检索状态：领域待确认（非游戏信号：{matched}）。"
                "不要使用本地游戏知识直接回答；若用户实际在问游戏系统，请先澄清游戏语境。"
            )
        try:
            matches = self.client.find(
                query,
                GAME_KNOWLEDGE_URI,
                limit=max(1, min(top_k, MAX_RESULTS)),
                context_type="resource",
                levels=[2],
            )
            evidence = self._read_evidence(matches)
        except OpenVikingError as error:
            return f"检索状态：OpenViking 后端错误。{error}"
        if not evidence:
            return "检索状态：没有可读取的本地游戏知识证据。"
        return (
            "检索状态：已获得可读取的本地游戏知识证据。\n\n"
            + "\n\n---\n\n".join(evidence)
        )

    def catalog(self) -> str:
        """Return only facts that can be established from the resource tree."""
        try:
            entries = self.client.list(
                GAME_KNOWLEDGE_URI, limit=100, recursive=False
            )
        except OpenVikingNotFound:
            return "游戏知识库当前没有可读取的资源。"
        except OpenVikingError as error:
            return f"游戏知识库状态：OpenViking 后端错误。{error}"

        source_uris = {
            self._source_uri(str(entry.get("uri") or entry.get("path") or ""))
            for entry in entries
        }
        sources: list[tuple[str, int]] = []
        for source_uri in sorted(uri for uri in source_uris if uri):
            try:
                stat = self.client.stat(source_uri)
            except OpenVikingNotFound:
                continue
            count = stat.get("count")
            if not isinstance(count, int) or count <= 0:
                continue
            sources.append((source_uri.rsplit("/", 1)[-1], count))

        if not sources:
            return "游戏知识库当前没有可读取的资源。"
        lines = [f"游戏知识库当前有 {len(sources)} 个可读取来源："]
        lines.extend(f"- {name}：{count} 个资源节点" for name, count in sources)
        lines.append("以上仅为后端当前资源清单，不推断教材数量、主题覆盖或内容质量。")
        return "\n".join(lines)

    @staticmethod
    def _source_uri(uri: str) -> str:
        if not uri.startswith(GAME_KNOWLEDGE_URI):
            return ""
        relative = uri.removeprefix(GAME_KNOWLEDGE_URI).strip("/")
        if not relative:
            return ""
        return GAME_KNOWLEDGE_URI + relative.split("/", 1)[0]

    def _read_evidence(self, matches: list[dict]) -> list[str]:
        remaining = self.evidence_char_budget
        evidence: list[str] = []
        seen_uris: set[str] = set()
        for match in matches:
            uri = str(match.get("uri", ""))
            if (
                not uri.startswith(GAME_KNOWLEDGE_URI)
                or uri in seen_uris
                or remaining <= 0
            ):
                continue
            try:
                content = self.client.read(uri).strip()
            except OpenVikingNotFound:
                continue
            if not content:
                continue
            seen_uris.add(uri)
            chunk = content[:remaining]
            evidence.append(chunk)
            remaining -= len(chunk)
        return evidence
