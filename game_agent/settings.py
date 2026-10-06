"""Validated, provider-neutral settings; never includes secrets in repr output."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    model: str | None = None
    retrieval_mode: str = "bge"
    database_path: Path = ROOT / "agent_memory.sqlite"
    request_timeout: int = 60
    max_tool_rounds: int = 6
    max_input_chars: int = 6000

    @property
    def chat_ready(self):
        return bool(self.api_key and self.base_url and self.model
                    and self.api_key not in {"your_api_key", "offline-not-configured"}
                    and self.model != "your_chat_model"
                    and not self.base_url.startswith("https://your-"))

    @classmethod
    def from_env(cls):
        mode = os.getenv("RETRIEVAL_MODE", "bge").lower()
        if mode not in {"bge", "keyword", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE 只支持 bge、keyword、hybrid")
        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        rounds = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
        if not 1 <= timeout <= 300 or not 1 <= rounds <= 12:
            raise ValueError("LLM_TIMEOUT 应为 1–300 秒，MAX_TOOL_ROUNDS 应为 1–12")
        path = Path(os.getenv("GAME_AGENT_DB") or str(ROOT / "agent_memory.sqlite")).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return cls(api_key=os.getenv("LLM_API_KEY") or os.getenv("ARK_API_KEY"),
                   base_url=os.getenv("LLM_BASE_URL") or os.getenv("ARK_BASE_URL"),
                   model=os.getenv("LLM_MODEL") or os.getenv("ARK_MODEL"),
                   retrieval_mode=mode, database_path=path, request_timeout=timeout, max_tool_rounds=rounds)
