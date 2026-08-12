# Journaling Chatbot (Phase 1)

Phase 1 deliverable: A CLI-driven expense and mood logger powered by a deterministic, rule-based extraction engine and SQLite storage.

## Features
- **Deterministic NLP Extraction**: Rule-based regex and `dateparser` extraction for amounts, dates, and finance/mood intents.
- **Crisis Guardrail**: In-line safety keyword scanner prioritizing user safety with immediate helpline disclosures (KIRAN / Tele-MANAS).
- **FastAPI Orchestrator**: REST API supporting open health endpoints and guarded chat/finance/mood operations.
- **Rich CLI REPL & Subcommands**: Interactive terminal chat REPL and fast one-shot logging subcommands.

This is Phase 1 of a larger planned system (see `ARCHITECTURE.md`). It deliberately
contains **no LLM calls, no Docker, no Redis/Postgres/ChromaDB** — those arrive in later
phases. Everything here is regex/`dateparser`-based and is the baseline a future
fine-tuned model has to beat.

## Quickstart

### 1. Installation
Ensure `uv` is installed (it will fetch Python 3.12 automatically if needed).

```bash
uv venv --python 3.12
uv sync
```

### 2. Database Migrations
Run Alembic migrations to set up the SQLite schema:

```bash
uv run alembic upgrade head
```

### 3. Running the Server
Start the orchestrator backend server:

```bash
uv run journal serve
```
Or directly via Uvicorn:
```bash
uv run uvicorn orchestrator.main:app --reload
```

### 4. CLI Usage

#### Interactive Chat REPL
```bash
uv run journal
```

#### One-shot Logging
```bash
uv run journal log "Spent 450 on groceries today"
```

#### Interactive Mood Check-in
```bash
uv run journal mood
```

#### Financial Summary
```bash
uv run journal summary --range month
```

## Configuration
Server (`shared/core/config.py`, override via `.env` or env vars):
- `DATABASE_URL` (default `sqlite:///./data/journal.db`)
- `API_KEY` (default `dev-local-key` — guarded routes require header `X-API-Key`)
- `DEFAULT_USER_HANDLE` (default `me` — Phase 1 is single-user)

CLI (env vars, matching the server defaults so it works out of the box locally):
- `JOURNAL_API_URL` (default `http://127.0.0.1:8000`)
- `JOURNAL_API_KEY` (default `dev-local-key`)

## Running Tests
```bash
uv run pytest
```
23 tests covering extraction (amount/category/date parsing, intent classification),
the finance API (creation, SQL-aggregated summaries, chat-driven logging), and the
crisis guardrail (trigger phrases, idiom false-positive avoidance, both the `/chat` and
`/mood/checkin` paths).
