# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_main.py::test_health -v

# Create / update Opik dataset
uv run python -m app.eval.create_dataset --dataset yja-dataset
uv run python -m app.eval.create_dataset --dataset yja-dataset --overwrite  # 기존 아이템 교체

# Run Opik evaluation (dataset must exist first)
uv run python -m app.eval.run_opik_eval --dataset yja-dataset --experiment exp-v1

# Lint / format
uv run ruff check .
uv run black .
```

## Environment Setup

Copy `env.sample` to `.env` and fill in required values:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | Yes | e.g. `gpt-4o` |
| `API_V1_PREFIX` | Yes | e.g. `/api/v1` |
| `ES_URL` | No | Elasticsearch URL (default: `https://elasticsearch-edu.didim365.app`) |
| `ES_USER` | No | Elasticsearch username (default: `elastic`) |
| `ES_PASSWORD` | No | Elasticsearch password |
| `ES_INDEX_NAME` | No | Elasticsearch index (default: `edu-collection`) |
| `MFDS_API_KEY` | No | 식품의약품안전처 공공데이터 API key (e약은요) |
| `HIRA_API_KEY` | No | 건강보험심사평가원 API key (병원정보) |
| `DEEPAGENT_RECURSION_LIMIT` | No | LangGraph max recursion (default 20) |
| `OPIK__URL_OVERRIDE` | No | Opik server URL (self-hosted) |
| `OPIK__API_KEY` | No | Opik Cloud API key |
| `OPIK__WORKSPACE` | No | Opik workspace name |
| `OPIK__PROJECT` | No | Opik project name |

Opik settings use Pydantic's nested delimiter (`OPIK__KEY`). `AgentService` maps these to the flat `OPIK_*` env vars that the Opik SDK reads.

## Architecture

This is a FastAPI + LangGraph medical AI assistant. All routes are mounted under `API_V1_PREFIX` (default `/api/v1`).

### Request Flow

```
POST /api/v1/chat
  -> app/api/routes/chat.py
  -> AgentService.process_query()   (async generator)
  -> LangGraph agent (astream, stream_mode="updates")
  -> StreamingResponse (text/event-stream / SSE)
```

Each SSE event is a JSON object with a `step` field:
- `{"step": "model", "tool_calls": [...]}` — agent is about to call tools
- `{"step": "tools", "name": "...", "content": {...}}` — tool result
- `{"step": "done", "message_id": "...", "role": "assistant", "content": "...", "metadata": {...}}` — final answer

### Key Components

**`app/agents/medical_agent.py`** — `create_medical_agent()` factory. Wires the LLM, three tools, a system prompt, and a checkpointer into a LangChain agent via `create_agent()`. The agent's structured output schema is `ChatResponse` (message_id, content, metadata).

**`app/agents/tools.py`** — Six `@tool`-decorated functions:
- `search_symptoms` — BM25 search against Elasticsearch (`ES_INDEX_NAME`), then summarises with a secondary LLM call
- `get_medication_info` — calls the Korean MFDS drug info API (e약은요); includes an English→Korean alias map (`_DRUG_NAME_ALIASES`)
- `find_nearby_hospitals` — calls the HIRA hospital info API; maps Korean region/specialty names to API codes via `_SIDO_CODE`, `_CL_CODE`, `_DEPT_CODE` dicts
- `check_drug_interaction` — queries e약은요 for both drugs and cross-checks interaction fields
- `classify_emergency` — LLM-based triage (즉시119 / 응급실 / 일반진료) from natural-language symptom description
- `get_first_aid_guide` — keyword-matched first-aid instructions from `_FIRST_AID_GUIDES`; falls back to LLM generation for unknown situations

**`app/services/agent_service.py`** — `AgentService` instantiated per request. Lazily initialises an `AsyncSqliteSaver` checkpointer (`checkpoints.db`) on first call. Drives the agent stream loop concurrently with a `progress_queue` for mid-stream progress events. Configures Opik tracing via `OpikTracer` + `track_langgraph` when `OPIK` settings are present.

**`app/core/config.py`** — Pydantic `Settings` with nested `OpikSettings`. Uses `env_nested_delimiter="__"` so `OPIK__PROJECT` maps to `settings.OPIK.PROJECT`.

**`app/eval/create_dataset.py`** — Uploads 20 hardcoded items (10 medical / 10 non-medical) to an Opik dataset. Non-medical items assert the agent should refuse. Run this before the eval script.

**`app/eval/run_opik_eval.py`** — CLI script that pulls a named dataset from Opik, runs each item through `AgentService`, and scores with the `Equals` metric. Dataset items use `input`/`expected_output` keys; `category` field can be used to split medical vs. non-medical experiments.

### Conversation Persistence

LangGraph thread-based memory: each chat request carries a `thread_id` (UUID). The `AsyncSqliteSaver` checkpointer persists conversation state in `checkpoints.db` keyed by `thread_id`, enabling multi-turn context within the agent.

`ConversationService` (`app/services/conversation_service.py`) is a separate in-memory store for thread metadata (title, timestamps) used by the threads API; it does not affect agent memory.

### API Endpoints

- `GET /` — root info
- `GET /health` — health check
- `GET /api/v1/threads` — list conversation threads
- `GET /api/v1/threads/{thread_id}` — get thread detail with messages
- `POST /api/v1/chat` — send a message (SSE streaming response)
