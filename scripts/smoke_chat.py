"""Explicit opt-in live check with isolated history and local retrieval only.

This sends two fixed game-design prompts to the configured chat provider.
It never enables web search, OCR or note writing and does not touch user chats.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="allow calls to your configured chat provider")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime" / "chat_smoke.json")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Use --live to opt in to model API calls. Offline checks: pytest tests.")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = {"status": "failed", "scope": "two live turns; local retrieval only; no web/OCR/writes",
              "turns": [], "exception_type": None}
    with tempfile.TemporaryDirectory(prefix="game-agent-live-check-") as directory:
        os.environ["GAME_AGENT_DB"] = str(Path(directory) / "checkpoints.sqlite")
        import Agent
        from game_agent.presentation import visible_history
        from langchain_core.messages import HumanMessage
        report["provider_host"] = urlparse(Agent.settings.base_url or "").hostname
        Agent.settings = replace(Agent.settings, max_tool_rounds=2)
        Agent.model_with_tools = Agent.model.bind_tools([Agent.search_game_knowledge]).bind(max_tokens=900)
        prompts = ["请根据本地游戏资料，简要解释技能伤害和冷却时间怎样配合。给出来源编号。",
                   "延续刚才的技能设计讨论，如果面向新手玩家，需要注意哪些取舍？依据不足就说明。"]
        config = Agent.create_config("live_smoke")
        started = perf_counter()
        try:
            for prompt in prompts:
                result = Agent.graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
                history = visible_history(result["messages"])
                reply = history[-1]
                report["turns"].append({"prompt": prompt, "reply": reply["content"],
                                        "retrieved_chunk_ids": [h["chunk_id"] for r in reply.get("retrievals", []) for h in r["hits"]],
                                        "final_request_usage": reply.get("usage", {}),
                                        "invalid_citation_warning": "来源编号未核实" in reply["content"]})
            report["status"] = "passed" if (all(t["reply"].strip() and "工具调用上限" not in t["reply"]
                                                 and not t["invalid_citation_warning"] for t in report["turns"])
                                                 and any(t["retrieved_chunk_ids"] for t in report["turns"])) else "failed"
        except Exception as exc:
            report["exception_type"] = type(exc).__name__
        finally:
            report["wall_time_seconds"] = round(perf_counter() - started, 2)
            Agent.connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed_turns": len(report["turns"]),
                      "exception_type": report["exception_type"], "report": str(args.output)}, ensure_ascii=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
