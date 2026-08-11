from __future__ import annotations

import unittest

from game_knowledge import GAME_KNOWLEDGE_URI, GameKnowledgeRetriever
from ingestion import SOURCES, ingest_all, verify_all
from memory import DurableMemory
from openviking_client import (
    OpenVikingClient,
    OpenVikingHTTPError,
    OpenVikingNotFound,
    OpenVikingSettings,
)


def http_error(
    error_type: type[OpenVikingHTTPError], status: int, code: str
) -> OpenVikingHTTPError:
    return error_type(
        "failed",
        status_code=status,
        error_code=code,
    )


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


class ClientTests(unittest.TestCase):
    def client(self, response: FakeResponse) -> OpenVikingClient:
        client = OpenVikingClient(
            OpenVikingSettings("http://ov", "key", 3, 30, None)
        )
        client.session = FakeSession(response)
        return client

    def test_structured_not_found_error(self) -> None:
        client = self.client(
            FakeResponse(
                404,
                {
                    "status": "error",
                    "error": {"code": "NOT_FOUND", "message": "missing"},
                },
            )
        )
        with self.assertRaises(OpenVikingNotFound) as raised:
            client.stat("viking://resources/missing")
        self.assertEqual(raised.exception.error_code, "NOT_FOUND")
        self.assertEqual(raised.exception.status_code, 404)

    def test_find_sends_l2_filter_and_auth(self) -> None:
        client = self.client(
            FakeResponse(
                200,
                {"status": "ok", "result": {"resources": []}},
            )
        )
        client.find(
            "economy",
            GAME_KNOWLEDGE_URI,
            limit=3,
            context_type="resource",
            levels=[2],
        )
        _, _, kwargs = client.session.calls[0]
        self.assertEqual(kwargs["json"]["level"], [2])
        self.assertEqual(kwargs["headers"]["X-API-Key"], "key")


class RetrievalTests(unittest.TestCase):
    def test_total_evidence_budget_and_deduplication(self) -> None:
        class Client:
            def find(self, *args, **kwargs):
                self.levels = kwargs["levels"]
                return [
                    {"uri": GAME_KNOWLEDGE_URI + "a.md"},
                    {"uri": GAME_KNOWLEDGE_URI + "a.md"},
                    {"uri": GAME_KNOWLEDGE_URI + "b.md"},
                ]

            def read(self, uri: str) -> str:
                return "1234567890"

        client = Client()
        result = GameKnowledgeRetriever(client, evidence_char_budget=12).search(
            "游戏经济"
        )
        evidence = result.split("。\n\n", 1)[1].replace("\n\n---\n\n", "")
        self.assertEqual(evidence, "123456789012")
        self.assertEqual(client.levels, [2])

    def test_catalog_uses_only_nonempty_resource_roots(self) -> None:
        class Client:
            def list(self, uri: str, *, limit: int, recursive: bool):
                self.listed_uri = uri
                self.recursive = recursive
                return [
                    {"uri": GAME_KNOWLEDGE_URI + "valid/guide.md"},
                    {"uri": GAME_KNOWLEDGE_URI + "empty"},
                ]

            def stat(self, uri: str):
                return {"count": 7 if uri.endswith("valid") else 0}

        client = Client()
        result = GameKnowledgeRetriever(client).catalog()
        self.assertEqual(client.listed_uri, GAME_KNOWLEDGE_URI)
        self.assertFalse(client.recursive)
        self.assertIn("valid：7 个资源节点", result)
        self.assertNotIn("empty", result)
        self.assertIn("不推断教材数量", result)


class MemoryTests(unittest.TestCase):
    def test_normalization_preserves_leading_year_and_drops_secrets(self) -> None:
        items = DurableMemory._normalize_candidates(
            "- 2026 年上线\n2. 用户偏好简洁\n- API key 是 abc\n- sk-abcdefghijklmnop"
        )
        self.assertEqual(items, ["2026 年上线", "用户偏好简洁"])

    def test_capture_uses_scoped_session_and_commit(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.messages = None
                self.created = None
                self.committed = None

            def create_session(self, session_id: str, *, memory_types: list[str]):
                self.created = (session_id, memory_types)
                return {}

            def batch_add_messages(self, session_id: str, messages: list[dict]):
                self.messages = (session_id, messages)
                return {}

            def commit_session(self, session_id: str):
                self.committed = session_id
                return {"status": "accepted"}

        client = Client()
        result = DurableMemory(client).capture(
            "web_1",
            "记住我偏好简洁",
            "好的",
            lambda _: "用户偏好简洁",
        )
        self.assertEqual(result, {"status": "accepted"})
        self.assertEqual(client.created[1], ["preferences", "entities", "events"])
        self.assertEqual(client.messages[0], client.committed)
        self.assertNotIn("记住我偏好简洁", client.messages[1][0]["content"])

    def test_canonical_managed_memory_uri(self) -> None:
        self.assertTrue(
            DurableMemory._is_managed_memory_uri(
                "viking://user/alice/memories/preferences/style.md"
            )
        )
        self.assertFalse(
            DurableMemory._is_managed_memory_uri(
                "viking://user/alice/memories/identity.md"
            )
        )


class IngestionTests(unittest.TestCase):
    def test_ingest_continues_after_one_source_fails(self) -> None:
        class Client:
            def stat(self, uri: str):
                return {}

            def add_resource(self, source_url: str, target_uri: str, **kwargs):
                if target_uri.endswith(SOURCES[0].slug):
                    raise http_error(OpenVikingHTTPError, 400, "INVALID_ARGUMENT")
                return {"status": "accepted"}

        reports = ingest_all(Client(), wait=False)
        self.assertEqual(len(reports), 6)
        self.assertFalse(reports[0].succeeded)
        self.assertTrue(all(report.succeeded for report in reports[1:]))

    def test_verify_only_treats_not_found_as_missing(self) -> None:
        class Client:
            def __init__(self, error: Exception) -> None:
                self.error = error

            def stat(self, uri: str):
                raise self.error

            def list(self, uri: str, *, limit: int):
                return []

            def read(self, uri: str):
                return ""

        missing = verify_all(
            Client(http_error(OpenVikingNotFound, 404, "NOT_FOUND"))
        )
        self.assertEqual(missing, [source.slug for source in SOURCES])
        with self.assertRaises(OpenVikingHTTPError):
            verify_all(Client(http_error(OpenVikingHTTPError, 503, "UNAVAILABLE")))

    def test_verify_rejects_empty_resource_skeleton(self) -> None:
        class Client:
            def stat(self, uri: str):
                return {"isDir": True}

            def list(self, uri: str, *, limit: int):
                return []

            def read(self, uri: str):
                raise AssertionError("empty trees must not be read")

        self.assertEqual(verify_all(Client()), [source.slug for source in SOURCES])


if __name__ == "__main__":
    unittest.main()
