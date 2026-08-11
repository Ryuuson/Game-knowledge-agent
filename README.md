# Game Knowledge Agent

Game Knowledge Agent uses OpenViking as its context backend. LangGraph retains
tool orchestration and Streamlit retains the visible conversation UI; OpenViking
owns game-knowledge retrieval and durable user/project memory.

## V2 Architecture

```text
Streamlit UI -> LangGraph -> OpenViking game Resource retrieval -> response
                       -> OpenViking durable-memory recall/capture

SQLite -> conversation IDs, titles, and raw LangGraph transcripts only
```

`search_game_knowledge` is the only game-knowledge tool exposed to the model.
It searches only `viking://resources/game-knowledge/`, reads bounded L2 content,
and returns usable evidence to the graph. There is no BGE/SQLite runtime fallback.

## Setup

Use WSL with Python 3.10+ and an OpenAI-compatible chat model. Start a configured
OpenViking server before running the Agent:

```bash
openviking-server doctor
openviking-server
curl http://127.0.0.1:1933/health
```

Create `.env` from `.env.example` and configure the chat model plus the
OpenViking endpoint. When OpenViking authentication is enabled,
`OPENVIKING_API_KEY` must be a tenant-scoped user or admin key.

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your_chat_model
OPENVIKING_BASE_URL=http://127.0.0.1:1933
# Leave blank for local dev-auth mode.
OPENVIKING_API_KEY=
```

Install this application's dependencies, import the six resource roots, then
start Streamlit:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python ingestion.py ingest --wait
.venv/bin/python ingestion.py verify
.venv/bin/streamlit run UI.py
```

The import targets are stable, so rerunning `ingestion.py ingest` updates the
same resource roots rather than creating a parallel corpus. It imports these
public sources under `viking://resources/game-knowledge/`:

- `game-design-wiki`
- `Game-Knowledge-Base`
- `open-game-mechanics-dataset`
- `Game_Num_Basics_And_Calc`
- `gamedev_at_home`
- `senior-game-designer`

## Durable Memory

Before every model call, the Agent recalls only relevant entries from the
authenticated user's OpenViking memory namespace. After a completed answer is
displayed and persisted, a separate filter sends only stable preferences,
confirmed project facts and constraints, explicit decisions, and continuing
tasks through OpenViking's Session commit/extraction pipeline. Extraction is
limited to `preferences`, `entities`, and `events`; secrets, raw transcripts,
small talk, speculation, and one-off questions are rejected.

The sidebar lists and deletes durable memory deliberately. “清空当前聊天” starts
a fresh visible conversation but does not delete durable memory. SQLite still
keeps conversation IDs, titles, checkpoints, and the complete raw transcript.

## Failure Behavior

If the OpenViking server is unavailable, game-knowledge retrieval returns a
clear backend error and the Agent must not silently replace the local corpus with
web search. Memory capture failures are logged after answer delivery and do not
block the response.

## Project Structure

```text
openviking_client.py  Scoped authenticated HTTP client
game_knowledge.py     Domain-gated retrieval and bounded L2 evidence
memory.py             Selective recall, capture, listing, and deletion
ingestion.py          Six-source manifest, idempotent import, verification
Agent.py              LangGraph and non-OpenViking tools
UI.py                 Streamlit chat and memory controls
conversation_store.py SQLite conversation directory
```
