"""Restore final replies and real retrieval artifacts, without tool preambles."""

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from wiki_corpus.provenance import describe_location


def validate_citations(text: str, messages) -> tuple[str, list[str]]:
    """Check source identifiers, not whether an assertion is entailed by a source."""
    allowed = set()
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, dict):
            allowed.update(str(h["chunk_id"]) for h in message.artifact.get("hits", []))
    unknown = sorted(set(re.findall(r"\[来源:([^\]\n]+)\]", text)) - allowed)
    if unknown:
        for citation in unknown:
            text = text.replace(f"[来源:{citation}]", "[来源编号未核实]")
        text += "\n\n部分来源编号未能与可核验的检索记录对应，请核对下方资料后再使用相关结论。"
    return text, unknown


def visible_history(messages) -> list[dict]:
    visible = []
    retrievals = []
    tool_names = []
    known_sources = {}
    for message in messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            retrievals, tool_names = [], []
            visible.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            tool_names.append(message.name or "tool")
            if isinstance(message.artifact, dict) and "hits" in message.artifact:
                retrievals.append(message.artifact)
                for hit in message.artifact["hits"]:
                    known_sources[str(hit["chunk_id"])] = (message.artifact, hit)
        elif isinstance(message, AIMessage) and not message.tool_calls and isinstance(message.content, str) and message.content.strip():
            displayed = list(retrievals)
            current_ids = {str(h["chunk_id"]) for r in retrievals for h in r["hits"]}
            for source_id in dict.fromkeys(re.findall(r"\[来源:([^\]\n]+)\]", message.content)):
                if source_id in known_sources and source_id not in current_ids:
                    prior, hit = known_sources[source_id]
                    displayed.append({**prior, "hits": [hit], "reused_from_prior_turn": True})
            visible.append({"role": "assistant", "content": message.content,
                            "retrievals": displayed, "tools": list(tool_names),
                            "usage": message.usage_metadata or {}})
    return visible


def export_markdown(messages: list[dict], title: str = "游戏知识助手对话") -> str:
    parts = [f"# {title}"]
    for message in messages:
        parts.append(f"\n## {'用户' if message['role'] == 'user' else '助手'}\n\n{message['content']}")
        for retrieval in message.get("retrievals", []):
            parts.append(f"\n检索方式：{retrieval['mode']}；状态：{retrieval['status']}。")
            for hit in retrieval.get("hits", []):
                parts.append(f"- 来源 `{hit['chunk_id']}`：{hit.get('title', '')}；{hit.get('source_url', '')}")
                parts.append(f"  定位：{describe_location(hit)}")
    return "\n".join(parts)
