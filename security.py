"""Defense-in-depth boundaries for prompts, tool calls, and model outputs."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse


MAX_PROMPT_CHARS = 8_000
MAX_TOOL_TEXT_CHARS = 8_000
MAX_WEB_CONTENT_CHARS = 30_000
MAX_TOOL_RESULT_CHARS = 30_000
MAX_TOOL_CALLS_PER_TURN = 6

# No shell, code-execution, or arbitrary filesystem tool is authorized here.
TOOL_PERMISSION_MATRIX = {
    "list_files": {"access": "read", "roots": ("knowledge", "images")},
    "read_document": {"access": "read", "roots": ("knowledge",)},
    "read_document_section": {"access": "read", "roots": ("knowledge",)},
    "search_documents": {"access": "read", "roots": ("knowledge",)},
    "search_game_knowledge": {"access": "read", "roots": ("data",)},
    "ocr_image": {"access": "model", "roots": ("images",)},
    "metaso_search": {"access": "network", "roots": ()},
    "metaso_reader": {"access": "network", "roots": ()},
    "save_note": {"access": "write", "roots": ("notes",)},
}

_INJECTION_PATTERNS = (
    re.compile(
        r"(?:ignore|disregard|忽略|无视).{0,80}?(?:previous|prior|all|之前|上面|所有).{0,80}?"
        r"(?:instruction|prompt|rule|指令|提示|规则)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:system|assistant)\s*:\s*(?:call|read|write|ignore|执行|读取|写入|忽略)",
        re.IGNORECASE,
    ),
)
_SENSITIVE_OUTPUT_PATTERNS = (
    re.compile(r"\b(?:sk|ark|mk|lsv2|AIza)-?[-_A-Za-z0-9]{16,}\b", re.IGNORECASE),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[^\s,;]+"),
    re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\s\S]*?"
        r"-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
)
_audit_lock = Lock()


def bounded_text(value: str, *, maximum: int, field_name: str) -> str:
    """Return normalized text or reject input that could exhaust downstream services."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} must be at most {maximum} characters.")
    return normalized


def input_guardrail(user_input: str) -> tuple[bool, str]:
    """Reject oversized prompts and obvious attempts to override system policy."""
    try:
        normalized = bounded_text(user_input, maximum=MAX_PROMPT_CHARS, field_name="输入")
    except ValueError as error:
        return False, str(error)
    if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
        return False, "检测到可能覆盖系统指令的输入，未执行请求。"
    return True, normalized


def redact_sensitive_output(value: str) -> str:
    """Redact credential-shaped content before it reaches the UI or persisted chat."""
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern in _SENSITIVE_OUTPUT_PATTERNS:
        redacted = pattern.sub("[已隐藏敏感信息]", redacted)
    return redacted


def truncate_tool_result(value: str, *, maximum: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Bound untrusted tool output before it becomes model context."""
    value = redact_sensitive_output(value)
    if len(value) <= maximum:
        return value
    return value[:maximum].rstrip() + "\n\n[工具结果因长度限制被截断]"


def require_tool_enabled(tool_name: str) -> str | None:
    """Enforce the tool allowlist and default-deny local write capability."""
    if tool_name not in TOOL_PERMISSION_MATRIX:
        return "工具未获授权。"
    if tool_name == "save_note" and os.getenv("ENABLE_NOTE_WRITES", "false").lower() != "true":
        return "笔记写入已禁用。需要时请在 .env 中设置 ENABLE_NOTE_WRITES=true 后重启应用。"
    return None


def audit_event(event: str, *, thread_id: str, outcome: str, detail: str = "") -> None:
    """Record metadata-only audit events without prompts, arguments, or results."""
    audit_path = Path(__file__).with_name("agent_audit.jsonl")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "thread_id_suffix": thread_id[-12:],
        "outcome": outcome,
        "detail": detail[:80],
    }
    try:
        with _audit_lock:
            with audit_path.open("a", encoding="utf-8") as audit_file:
                os.chmod(audit_path, 0o600)
                audit_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        return


def resolve_path_within_root(root: Path, relative_path: str) -> Path:
    """Resolve a relative path and reject traversal, absolute paths, and symlink escapes."""
    candidate = Path(relative_path)
    if candidate.is_absolute() or relative_path != candidate.as_posix():
        raise ValueError("Invalid file path.")

    resolved_root = root.resolve()
    resolved_candidate = (resolved_root / candidate).resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("File path is outside the allowed directory.")
    return resolved_candidate


def validate_public_http_url(url: str) -> str:
    """Allow only ordinary public HTTP(S) URLs for remote-reader requests."""
    try:
        parsed = urlparse(url.strip())
        hostname = parsed.hostname
    except (AttributeError, ValueError) as error:
        raise ValueError("Invalid URL.") from error

    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("URL must use http or https and include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed.")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return parsed.geturl()

    if not address.is_global:
        raise ValueError("Private, loopback, and reserved network addresses are not allowed.")
    return parsed.geturl()
