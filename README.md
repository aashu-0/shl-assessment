# Conversational SHL Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a **grounded shortlist of SHL assessments**
through dialogue. It clarifies when the need is underspecified, recommends 1–10
assessments once it has context, refines the shortlist as constraints change,
compares assessments on request, and refuses anything out of scope — never
recommending anything outside the SHL catalog.

Built for the SHL Labs AI Intern take-home assignment.

---

## Quickstart

```bash
# 1. Python 3.11–3.13 recommended — uv will install and manage it for you
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync

# 2. Configure the LLM key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...   (https://aistudio.google.com/apikey)

# 3. Run
uv run uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with SQL and Spring"}]}'
```

> The service also runs **without** an API key — it falls back to a
> deterministic BM25-only agent so `/health` and `/chat` never break — but the
> full conversational quality requires `GEMINI_API_KEY`.

---

## API

Stateless: every `POST /chat` carries the full conversation history; the server
stores nothing per conversation.

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /chat`
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```
Response:
```json
{
  "reply": "Here are assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is **empty** while the agent is clarifying or refusing, and an
array of **1–10** items once it commits to a shortlist. `end_of_conversation` is
`true` only when the task is complete.

**Test-type codes:** `A` Ability & Aptitude · `B` Biodata & Situational Judgment
· `C` Competencies · `D` Development & 360 · `E` Assessment Exercises ·
`K` Knowledge & Skills · `P` Personality & Behavior · `S` Simulations.

---

## Architecture

```
POST /chat
   │
   ▼
Agent.respond()
   ├─ build combined retrieval query from user turns
   ├─ Retriever.retrieve()  → BM25 (+ Gemini embeddings) via RRF, plus anchors
   ├─ prompt = system policy + conversation + grounded candidate list
   ├─ Gemini → strict JSON {action, reply, recommendation_ids, end_of_conversation}
   └─ post-validate: keep only ids present in candidates & catalog, clamp to 10
```

| Module | Responsibility |
|--------|----------------|
| `app/catalog.py` | Load/normalize the 377 individual test solutions; test-type mapping; **id→assessment grounding** (the anti-hallucination gate). |
| `app/retriever.py` | Hybrid **BM25 + Gemini-embedding** retrieval fused with Reciprocal Rank Fusion, plus curated flagship "anchor" instruments. |
| `app/llm.py` | Gemini wrappers: structured JSON chat + cached batch embeddings. |
| `app/prompts.py` | System policy (clarify/recommend/refine/compare/refuse) + per-turn prompt. |
| `app/agent.py` | Orchestration + guardrails + deterministic no-key fallback. |
| `app/main.py` | FastAPI app: `/health`, `/chat`. |

See [`APPROACH.md`](APPROACH.md) for design rationale, evaluation, and what
didn't work.

---

## Grounding (why it can't hallucinate a URL)

The LLM only ever returns **entity ids** chosen from a candidate list it is
shown. The server maps those ids back to real catalog records
(`Catalog.resolve_ids`), dropping anything not physically in the catalog. URLs,
names and test types come from catalog data, never from the model. If the model
returns an id it wasn't shown or an invented one, it is discarded before the
response is built.

---

## Evaluation

```bash
# Coverage: are the traces' expected assessments even in our catalog? (ceiling)
uv run python -m evalsuite.replay_eval --coverage      # 43/43 = 1.000

# Full replay: Recall@10 against the 10 sample traces (needs GEMINI_API_KEY)
uv run python -m evalsuite.replay_eval

# Unit + API + guardrail tests (no key needed)
uv run pytest -q
```

The eval harness (`evalsuite/`) parses the 10 provided conversation traces,
replays their user turns against the stateless agent, and scores the final
shortlist with Recall@K — mirroring the grader's replay approach.

---

## Deployment

Deploy-ready via Docker. On [Render](https://render.com): create a **Blueprint**
from this repo (`render.yaml`), then set `GEMINI_API_KEY` in the dashboard. Any
Docker host works:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

> The Dockerfile installs dependencies with `uv` as well, so image builds stay
> fast and reproducible via the project's lockfile.

The health check tolerates cold starts (first `/health` may take up to ~2 min on
free tiers while the catalog loads and embeddings warm).

---

## Tech stack

FastAPI · Pydantic v2 · Google Gemini (`gemini-2.0-flash` + `text-embedding-004`)
· in-process BM25 + embedding RRF retrieval · pytest · uv (dependency & venv
management). Rationale in [`APPROACH.md`](APPROACH.md).