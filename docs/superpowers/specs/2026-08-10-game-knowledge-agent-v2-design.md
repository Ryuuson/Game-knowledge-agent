# Game Knowledge Agent V2 Design

## Goal

Replace the Game Knowledge Agent's custom flat BGE/SQLite RAG and summary-based
long-term context mechanism with OpenViking. Keep LangGraph as the orchestration
layer and Streamlit as the user interface.

V2 is developed and run in WSL. The existing V1 commit history remains the rollback
point; V2 does not retain a runtime legacy retrieval mode.

## Scope

V2 will:

- import the original documents from all six public game-knowledge sources into
  OpenViking Resources;
- use OpenViking for game-knowledge retrieval and evidence reading;
- use OpenViking Memory to recall and store durable user and project context;
- retain the existing Streamlit session list and raw, visible chat history;
- retain LangGraph's domain routing, web fallback, file tools, vision tool, and
  response generation flow;
- expose memory deletion as a deliberate user-facing operation.

V2 will not:

- ingest the V1 JSONL chunks as its primary corpus;
- preserve the V1 BGE index, score thresholds, or runtime fallback switch;
- give the chat model unrestricted direct access to OpenViking's generic write,
  delete, or filesystem tools;
- replace the Agent with VikingBot or replace the Streamlit UI.

## Architecture

```text
Streamlit UI
  -> LangGraph routing and tool orchestration
     -> OpenViking retrieval adapter (game Resource search and L2 reads)
     -> web/file/vision tools where routing permits
     -> chat model response
     -> OpenViking memory adapter (recall before, selective capture after)

OpenViking Server
  -> Resource tree for public game knowledge
  -> per-user Resource tree for imported private materials
  -> per-user Memory tree for durable facts and decisions
  -> local BGE embedding and configured VLM for semantic processing

SQLite
  -> UI conversation IDs, titles, and complete raw transcript only
```

### Resource layout

The importer retains source identity and meaningful source structure below:

```text
viking://resources/game-knowledge/
  game-design-wiki/
  Game-Knowledge-Base/
  open-game-mechanics-dataset/
  Game_Num_Basics_And_Calc/
  gamedev_at_home/
  senior-game-designer/
```

Original Markdown, HTML-derived source content, and structured source files are
the import input. The V1 `game_knowledge_chunks.jsonl`,
`new_knowledge_chunks.jsonl`, and generated SQLite vector indexes are not V2
inputs. This lets OpenViking generate L0 abstracts, L1 overviews, and L2 original
content from coherent documents rather than from pre-cut fragments.

### Retrieval adapter

`search_game_knowledge` remains the only game-knowledge tool exposed to the
LangGraph model. Its implementation is replaced by an OpenViking adapter:

1. Apply the existing soft domain router before querying game resources.
2. Search only `viking://resources/game-knowledge/`.
3. Select relevant result URIs and read bounded L2 content for answer evidence.
4. Return evidence and retrieval state to the outer Agent, without exposing
   internal source paths to the end user by default.
5. If no usable evidence is found, preserve the existing clarification or web
   fallback policy.

V1 cosine thresholds (`0.62` and `0.67`) are removed. OpenViking score values have
a different meaning. V2 gates answers on the presence of relevant, readable L2
evidence plus the existing domain decision, rather than on transplanted numeric
thresholds.

### Memory adapter

Before each model call, the adapter recalls only memories relevant to the current
turn and injects them as internal context. It does not replay all old chat messages.
After a completed answer, it extracts and stores only durable items:

- stable user preferences;
- confirmed project facts and constraints;
- explicit design decisions;
- active, continuing tasks.

It must not persist ordinary small talk, provisional speculation, one-off factual
questions, secrets, or raw full transcripts as long-term memory. The UI offers a
way to list and delete stored memories. A `/clear`-style UI action clears the active
chat view but does not silently delete durable memory.

SQLite remains responsible for raw transcript retention because the UI needs to
display and reopen full sessions. The current summary table and internal summary
injection are removed. Model context becomes: system policy, relevant recalled
memories, bounded recent complete turns, current retrieved evidence, and the user
prompt.

## Module boundaries

The V2 codebase should replace the monolithic responsibilities in `Agent.py` with
separable modules:

- `openviking_client`: authenticated HTTP/client wrapper, URI scoping, timeout,
  and error translation;
- `game_knowledge`: deterministic retrieval and bounded L2 evidence formatting;
- `memory`: selective recall, capture, listing, and deletion;
- `ingestion`: source-specific import manifests and import verification;
- `graph`: LangGraph state, routing, and tool binding;
- `ui`: Streamlit presentation and session state only.

The exact directory names may follow the existing repository conventions, but these
ownership boundaries must remain explicit.

## Failure behavior

- If OpenViking is unavailable, show a clear backend error. Do not conceal the
  failure by silently using web search.
- If retrieval has no usable local evidence, follow the existing domain-aware
  clarification or optional web fallback policy.
- If memory capture fails, answer delivery still succeeds and the failure is logged;
  no raw transcript is substituted as a memory record.
- Imports are idempotent by source URI where OpenViking supports it and report
  source-level completion or failure.
- L2 content passed to the chat model has a size bound to avoid context overflow.

## Evaluation and acceptance

V2 is accepted only after all of the following pass:

1. Each of the six source roots is present in the OpenViking tree and sample L2
