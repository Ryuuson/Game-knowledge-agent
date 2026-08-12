# Security Baseline

This application is designed for one local user. The bundled Streamlit config
listens only on `127.0.0.1` and enables Streamlit's CORS and XSRF protections.
Do not expose it directly to a LAN or the public internet.

## Defense in Depth

| Layer | Implemented control |
| --- | --- |
| Architecture boundary | No shell, Python execution, arbitrary file, or database-query tool exists. Files are constrained to fixed roots; path traversal and symbolic-link escapes are rejected. Streamlit is loopback-only. |
| Permission control | `TOOL_PERMISSION_MATRIX` is a static allowlist. Reads are limited to `knowledge`/`images`/`data`; `save_note` is default-deny and needs `ENABLE_NOTE_WRITES=true`. Tool arguments, result sizes, retrieval count, image size, and web-request timeouts are bounded. |
| Identity and audit | The local single-user mode uses unguessable UUID conversation IDs. `agent_audit.jsonl` records only timestamps, event type, tool names, and a thread-ID suffix, with mode `0600`; it does not log prompts, arguments, outputs, or credentials. |
| Data protection | `.env` and SQLite state are ignored by Git, database/audit files are set to mode `0600` where supported, credential-shaped output is redacted, and remote/tool output is length-limited before entering model context. |

The input guardrail rejects oversized requests and clear attempts to override
system instructions. It intentionally does not block ordinary questions that
contain words such as `token` or `password`; those are often valid security
questions. It instead prevents credential-shaped values from appearing in the
model's final response.

## Deployment Boundary

For shared or public deployment, put an authenticated TLS reverse proxy or an
identity-aware access gateway in front of the application. Do not rely on a
browser session or a Streamlit thread ID as user authentication. A public
deployment also requires application-level user-to-conversation authorization
before it can safely expose history or destructive operations.

Run the process as a dedicated non-administrator account. Use filesystem or
container isolation to mount only the project data it needs, as read-only
where possible. If future requirements need arbitrary code execution, use a
separate sandbox service without API keys, with CPU/memory/time/network and
filesystem limits. Do not add `subprocess`, shell, `exec`, `eval`, or a general
write-file tool to this Agent process.

## Credentials

Keep `.env` untracked and readable only by the service account. API keys must
be supplied through environment variables or a secrets manager in production.
Rotate any credential that has been placed in a terminal log, chat transcript,
repository, issue, or other shared channel.

## Network and Writes

`save_note` is disabled by default. Set `ENABLE_NOTE_WRITES=true` only for a
trusted local session that needs to create notes. Web search and web reading
are disabled until `METASO_API_KEY` is configured. Reader URLs are restricted
to public HTTP(S) targets, with private literal IP ranges rejected, and remote content is length-limited
before it is sent back to the model. DNS names are allowed for ordinary web
use; deploy an egress proxy with DNS-rebinding protection if hostile URL input
is in scope.

## Operational Checks

Before each release, run `pytest`, update pinned dependencies, and scan them
with your approved vulnerability scanner. Review audit metadata for denied
input/tool events. Do not log prompts, tool results, or provider request
headers in production because they can contain user data or credentials.
