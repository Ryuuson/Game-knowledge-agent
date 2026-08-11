"""Small, typed HTTP client for the OpenViking operations used by this app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import requests


class OpenVikingError(RuntimeError):
    """Base error for failed or malformed OpenViking requests."""


class OpenVikingUnavailable(OpenVikingError):
    """The configured OpenViking server cannot be reached."""


class OpenVikingHTTPError(OpenVikingError):
    """OpenViking returned a structured HTTP error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: str = "UNKNOWN",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class OpenVikingNotFound(OpenVikingHTTPError):
    """The requested OpenViking object does not exist."""


class OpenVikingConflict(OpenVikingHTTPError):
    """The requested create or update conflicts with existing state."""


@dataclass(frozen=True)
class OpenVikingSettings:
    base_url: str
    api_key: str | None
    timeout_seconds: float
    import_timeout_seconds: float
    actor_peer_id: str | None

    @classmethod
    def from_environment(cls) -> "OpenVikingSettings":
        timeout = _positive_float("OPENVIKING_TIMEOUT_SECONDS", 30.0)
        return cls(
            base_url=os.getenv("OPENVIKING_BASE_URL", "http://127.0.0.1:1933").rstrip("/"),
            api_key=os.getenv("OPENVIKING_API_KEY") or None,
            timeout_seconds=timeout,
            import_timeout_seconds=_positive_float(
                "OPENVIKING_IMPORT_TIMEOUT_SECONDS", 1800.0
            ),
            actor_peer_id=os.getenv("OPENVIKING_ACTOR_PEER_ID") or None,
        )


def _positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


class OpenVikingClient:
    """Expose only the scoped filesystem, retrieval, ingestion, and Session APIs."""

    def __init__(self, settings: OpenVikingSettings | None = None) -> None:
        self.settings = settings or OpenVikingSettings.from_environment()
        self.session = requests.Session()

    def find(
        self,
        query: str,
        target_uri: str,
        *,
        limit: int,
        context_type: str,
        levels: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "query": query,
            "target_uri": target_uri,
            "limit": limit,
            "context_type": context_type,
        }
        if levels is not None:
            body["level"] = list(levels)
        result = self._request("POST", "/api/v1/search/find", json=body)
        key = {"resource": "resources", "memory": "memories"}.get(context_type)
        if key is None:
            raise ValueError(f"Unsupported context type: {context_type}")
        matches = result.get(key, []) if isinstance(result, dict) else []
        return [match for match in matches if isinstance(match, dict)]

    def read(self, uri: str) -> str:
        result = self._request("GET", "/api/v1/content/read", params={"uri": uri})
        if not isinstance(result, str):
            raise OpenVikingError("OpenViking content/read returned a non-text result.")
        return result

    def list(
        self, uri: str, *, limit: int = 100, recursive: bool = True
    ) -> list[dict[str, Any]]:
        result = self._request(
            "GET",
            "/api/v1/fs/ls",
            params={
                "uri": uri,
                "recursive": "true" if recursive else "false",
                "node_limit": max(1, min(limit, 500)),
            },
        )
        if isinstance(result, list):
            return [entry for entry in result if isinstance(entry, dict)]
        if isinstance(result, dict):
            entries = result.get("entries", result.get("items", []))
            return [entry for entry in entries if isinstance(entry, dict)]
        raise OpenVikingError("OpenViking fs/ls returned an unexpected result.")

    def delete(self, uri: str) -> None:
        self._request("DELETE", "/api/v1/fs", params={"uri": uri, "recursive": "false"})

    def mkdir(self, uri: str) -> None:
        self._request("POST", "/api/v1/fs/mkdir", json={"uri": uri})

    def add_resource(
        self,
        source_url: str,
        target_uri: str,
        *,
        wait: bool,
        parser_args: dict[str, Any] | None = None,
        preserve_structure: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "path": source_url,
            "to": target_uri,
            "wait": wait,
            "processing_mode": "semantic_and_vectors",
            "args": parser_args or {},
        }
        if preserve_structure is not None:
            body["preserve_structure"] = preserve_structure
        result = self._request(
            "POST",
            "/api/v1/resources",
            json=body,
            timeout=self.settings.import_timeout_seconds if wait else None,
        )
        return self._dict_result(result, "resources")

    def stat(self, uri: str) -> dict[str, Any]:
        return self._dict_result(
            self._request("GET", "/api/v1/fs/stat", params={"uri": uri}),
            "fs/stat",
        )

    def batch_add_messages(
        self, session_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/messages/batch",
            json={"messages": messages},
        )
        return self._dict_result(result, "sessions/messages/batch")

    def create_session(
        self, session_id: str, *, memory_types: list[str]
    ) -> dict[str, Any]:
        result = self._request(
            "POST",
            "/api/v1/sessions",
            json={
                "session_id": session_id,
                "memory_policy": {
                    "self": {"enabled": True},
                    "peer": {"enabled": False},
                    "working_memory": {"enabled": False},
                    "memory_types": memory_types,
                },
            },
        )
        return self._dict_result(result, "sessions")

    def commit_session(self, session_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/commit",
            json={"keep_recent_count": 0},
        )
        return self._dict_result(result, "sessions/commit")

    @staticmethod
    def _dict_result(result: Any, operation: str) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise OpenVikingError(f"OpenViking {operation} returned an unexpected result.")
        return result

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        headers = {"Accept": "application/json"}
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key
        if self.settings.actor_peer_id:
            headers["X-OpenViking-Actor-Peer"] = self.settings.actor_peer_id
        try:
            response = self.session.request(
                method,
                f"{self.settings.base_url}{path}",
                headers=headers,
                timeout=timeout if timeout is not None else self.settings.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as error:
            raise OpenVikingUnavailable(
                f"OpenViking 后端不可用（{self.settings.base_url}）：{error}"
            ) from error

        payload = self._json_payload(response)
        if not response.ok or payload.get("status") != "ok":
            raise self._http_error(response.status_code, payload)
        return payload.get("result")

    @staticmethod
    def _json_payload(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            detail = response.text.strip()[:500]
            raise OpenVikingError(
                f"OpenViking 返回了无法解析的响应（HTTP {response.status_code}）：{detail}"
            ) from error
        if not isinstance(payload, dict):
            raise OpenVikingError("OpenViking returned a non-object JSON response.")
        return payload

    @staticmethod
    def _http_error(status_code: int, payload: dict[str, Any]) -> OpenVikingHTTPError:
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        code = str(error.get("code") or "UNKNOWN")
        message = str(error.get("message") or "OpenViking request failed")
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        error_type: type[OpenVikingHTTPError]
        if status_code == 404 or code == "NOT_FOUND":
            error_type = OpenVikingNotFound
        elif status_code == 409 or code in {"ALREADY_EXISTS", "CONFLICT", "ABORTED"}:
            error_type = OpenVikingConflict
        else:
            error_type = OpenVikingHTTPError
        return error_type(
            f"OpenViking 请求失败（HTTP {status_code}, {code}）：{message}",
            status_code=status_code,
            error_code=code,
            details=details,
        )
