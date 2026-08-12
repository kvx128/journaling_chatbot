# Product Discovery — status

**16 questions answered** in discovery → recorded as D1–D19 in
[ARCHITECTURE.md §0](ARCHITECTURE.md). This file now tracks only what's **still running on
my default**.

Nothing below blocks Phase 1. Override any of them whenever — say the number.

---

## Running on default — override any time

### Interaction

| # | Question | Current default |
|---|---|---|
| 1 | Multi-user support? | **Single user**, static API key. `user_id` threaded everywhere so multi-user is a config change |
| 2 | Should the bot speak first? | **Reactive in-session**; a pending check-in surfaces when you open the CLI |
| 3 | Session cadence | Scattered messages, 30-min Redis TTL, day boundaries in IST |
| 4 | Proactive recall ("you mentioned this last week") | **On**, max once per session, high similarity threshold only |
| 5 | CLI shape | **Both** — bare `journal` opens the REPL, `journal <cmd>` runs one-shot |
| 6 | Small talk | Narrow canned set for greetings/thanks/help; everything else redirected to a journaling prompt |

### Finance

| # | Question | Current default |
|---|---|---|
| 7 | Category taxonomy | **Fixed enum, ~18 categories** + `OTHER` with free-text note. Fixed is what makes constrained decoding possible — an invalid category becomes unemittable |
| 8 | Savings goals ("save 50k by December") | **Out of v1.** Revisit after the insight engine lands |
| 9 | Confirm-before-save | Auto-save above 0.85 confidence, confirm below, `/undo` always available |

### Mental health

| # | Question | Current default |
|---|---|---|
| 10 | Guided CBT exercises | **No multi-turn protocols.** Retrieve-and-cite psychoeducation + one grounding exercise. Higher-risk content, needs review I'd rather do deliberately |
| 11 | Streaks / gamification | **None.** Breaking a streak when you're already low adds guilt — actively harmful here |
| 12 | Covariates tracked | Sleep hours, energy 1–5, social-contact flag. Three fields; they're what make correlation findings non-trivial |

### Memory & RAG

| # | Question | Current default |
|---|---|---|
| 13 | KB source material | **I'll source it** — WHO/NHS/public-domain CBT worksheets, provenance recorded per doc (repo is public, so licensing matters) |
| 14 | Retention | Indefinite; `/forget <range>` purges SQL + vectors + Redis together |

### Infra

| # | Question | Current default |
|---|---|---|
| 15 | Bot model serving | **GGUF Q4_K_M + llama-cpp-python.** Removes CUDA from bot images — the biggest lever on the 20s cold start. Ollama stays as the packaged demo path |
| 16 | DB progression | SQLAlchemy + Alembic from day one; SQLite → Postgres at phase 13 |
| 17 | Kubernetes | Manifests written, `kubeval`-validated, **not deployed** |
| 18 | MLflow | File-backed in dev, server in the Compose stack |
| 19 | Hosting | Local-only, but secrets via env from day one |

### Mobile

| # | Question | Current default |
|---|---|---|
| 20 | Mobile scope | **Dev only** via `claude --remote-control`. Using the *chatbot* from your phone (Telegram client + exposed API + real auth) is a strong stretch goal, flagged not scheduled |

---

## Worth revisiting later

**Telegram client (#20).** It's the change that would make the open/guarded API tier split
*matter* rather than just exist — right now the CLI is the only consumer, so the split is
architectural theatre. A second client with a real network boundary justifies it. Worth
doing once phases 1–9 are solid.

**Savings goals (#8).** The natural bridge between the two bots — "you're ahead on your
goal, how does that feel?" — is exactly the kind of cross-domain moment the orchestrator
exists to enable. Out of v1 only because it's scope, not because it's wrong.
