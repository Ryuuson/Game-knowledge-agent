"""Idempotent source manifest and CLI for public game-knowledge resources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from openviking_client import (
    OpenVikingClient,
    OpenVikingConflict,
    OpenVikingError,
    OpenVikingNotFound,
)


RESOURCE_ROOT = "viking://resources/game-knowledge"
READABLE_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".markdown",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Source:
    slug: str
    url: str
    parser_args: dict[str, Any] = field(default_factory=dict)
    preserve_structure: bool | None = None

    @property
    def target_uri(self) -> str:
        return f"{RESOURCE_ROOT}/{self.slug}"


@dataclass(frozen=True)
class IngestResult:
    slug: str
    succeeded: bool
    detail: str


SOURCES = (
    Source(
        "game-design-wiki",
        "https://github.com/Being09/game-design-wiki",
        preserve_structure=True,
    ),
    Source(
        "Game-Knowledge-Base",
        "https://github.com/diedie23/Game-Knowledge-Base",
        preserve_structure=True,
    ),
    Source(
        "open-game-mechanics-dataset",
        "https://github.com/Thaelith/open-game-mechanics-dataset",
        preserve_structure=True,
    ),
    Source(
        "Game_Num_Basics_And_Calc",
        "https://github.com/lsc1414/Game_Num_Basics_And_Calc",
        preserve_structure=True,
    ),
    Source(
        "gamedev_at_home",
        "https://github.com/zsc/gamedev_at_home",
        preserve_structure=True,
    ),
    Source(
        "senior-game-designer",
        "https://github.com/tigermkiiiddd/senior-game-designer",
        preserve_structure=True,
    ),
)


def ensure_resource_root(client: OpenVikingClient) -> None:
    try:
        client.stat(RESOURCE_ROOT)
        return
    except OpenVikingNotFound:
        pass
    try:
        client.mkdir(RESOURCE_ROOT)
    except OpenVikingConflict:
        pass


def ingest_all(
    client: OpenVikingClient,
    *,
    wait: bool,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    ensure_resource_root(client)
    reports: list[IngestResult] = []
    for source in SOURCES:
        try:
            result = client.add_resource(
                source.url,
                source.target_uri,
                wait=wait,
                parser_args=source.parser_args,
                preserve_structure=source.preserve_structure,
            )
        except OpenVikingError as error:
            report = IngestResult(source.slug, False, str(error))
            reports.append(report)
            if on_result:
                on_result(report)
            continue
        status = str(result.get("status") or "submitted")
        task_id = result.get("task_id")
        detail = status + (f" (task {task_id})" if task_id else "")
        report = IngestResult(source.slug, True, detail)
        reports.append(report)
        if on_result:
            on_result(report)
    return reports


def verify_all(client: OpenVikingClient) -> list[str]:
    missing: list[str] = []
    for source in SOURCES:
        try:
            client.stat(source.target_uri)
            entries = client.list(source.target_uri, limit=500)
        except OpenVikingNotFound:
            missing.append(source.slug)
            continue
        readable = False
        for entry in entries:
            uri = str(entry.get("uri") or entry.get("path") or "")
            suffix = "." + uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
            if (
                not uri.startswith(source.target_uri)
                or entry.get("isDir")
                or entry.get("is_dir")
                or suffix not in READABLE_SUFFIXES
            ):
                continue
            try:
                content = client.read(uri)
            except OpenVikingNotFound:
                continue
            if content.strip():
                readable = True
                break
        if not readable:
            missing.append(source.slug)
    return missing


def format_ingest_result(report: IngestResult) -> str:
    state = "ok" if report.succeeded else "failed"
    return f"{report.slug}: {state} - {report.detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import or verify Game Knowledge Agent resources."
    )
    parser.add_argument("command", choices=("ingest", "verify"))
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for semantic processing of each source.",
    )
    args = parser.parse_args()
    client = OpenVikingClient()
    try:
        if args.command == "ingest":
            reports = ingest_all(
                client,
                wait=args.wait,
                on_result=lambda report: print(
                    format_ingest_result(report), flush=True
                ),
            )
            return 0 if all(report.succeeded for report in reports) else 1
        missing = verify_all(client)
    except OpenVikingError as error:
        print(f"OpenViking error: {error}")
        return 1
    if missing:
        print("Missing resource roots: " + ", ".join(missing))
        return 1
    print("All six game-knowledge resource roots are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
