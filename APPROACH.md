# Approach — Conversational SHL Assessment Recommender

## Problem framing
Turn a vague hiring intent into a grounded SHL assessment shortlist through
dialogue, while (a) never recommending anything outside the catalog, (b) obeying
a strict response schema, and (c) surviving a non-deterministic user within an
8-turn / 30-second-per-call budget. The hard constraints (schema, catalog-only
URLs, turn cap) are treated as invariants enforced in code, not left to the LLM.

## Stack & why
- **FastAPI + Pydantic v2** — schema is enforced by the response model, so a
  malformed reply is impossible by construction; the assignment says schema
  deviation fails the evaluator.
- **Google Gemini** (`gemini-2.0-flash` for chat, `text-embedding-004` for
  embeddings) — free tier, fast enough for the 30 s cap, and one JSON-native
  generation call per turn. The LLM layer is a thin wrapper so it is trivially
  mockable in tests.
- **In-process retrieval (BM25 + embeddings)** — the catalog is only 377 items,
  so a vector *server* (FAISS/Chroma/pgvector) is unnecessary weight; an
  in-memory hybrid index loads in milliseconds and deploys as a single service.

## Catalog handling
The scraped catalog is normalized into typed `Assessment` objects with a
precomputed **test-type code** (`Knowledge & Skills → K`, etc.) and a rich
`search_text` (name weighted, keys, job levels, description). The catalog is the
single source of truth for names/URLs/types.

## Retrieval
Per turn the agent builds **one combined query** from all user turns and
retrieves candidates via a hybrid of:
- **BM25** (custom, dependency-free) — precise on concrete skills ("Docker",
  "Spring", "HIPAA"); always available and deterministic.
- **Gemini embeddings** — semantic bridge for vocabulary gaps ("senior
  leadership" → OPQ/Verify). Catalog embeddings are computed once at startup and
  cached to disk keyed by a model+content hash.

The two rankings are merged with **Reciprocal Rank Fusion** (robust to
incomparable score scales). We additionally always include six curated
**anchor** instruments — SHL's genuinely cross-role defaults (OPQ32r, Verify
G+, Graduate Scenarios, Global Skills Assessment, DSI, MQ). These rarely overlap
lexically with a role description yet are legitimately recommended when
behavioural/cognitive fit matters; keeping them in the candidate pool lets the
LLM *choose* them per policy — it still decides whether they belong.

## Agent design & prompt
A single structured LLM call per turn returns strict JSON:
`{action, reply, recommendation_ids, end_of_conversation}` where `action ∈
{clarify, recommend, compare, refuse}`. The system prompt encodes the policy:
don't recommend on a vague turn-1 query; clarify with **one** focused question;
recommend the smallest well-justified 1–10 set; **keep prior picks stable on
refinement** (add/remove specific items, don't restart); compare using only
catalog descriptions; refuse off-topic / legal / prompt-injection with empty
recommendations. The prompt is told how many assistant turns remain so it
commits before the cap.

## Guardrails (defense in depth)
Grounding is enforced **after** the model: `recommendation_ids` are filtered to
ids actually shown as candidates and present in the catalog; anything else is
dropped, the shortlist is clamped to ≤10, and clarify/refuse force an empty
list. URLs are attached server-side from catalog data, so a hallucinated or
invented link cannot reach the response. If the LLM call fails or no key is set,
a deterministic BM25 fallback keeps `/chat` schema-compliant and non-crashing.

## Evaluation
`evalsuite/` parses the 10 provided traces, extracts each trace's final expected
shortlist (the labeled answer table), replays the user turns against the
stateless agent, and computes **Recall@10** — mirroring the grader's replay
harness. A `--coverage` mode checks the recall ceiling.

Measured results:
- **Coverage ceiling: 43/43 = 1.000** — every expected assessment exists in the
  catalog, so retrieval + selection can in principle reach perfect recall.
- **Retrieval recall into the candidate pool (BM25-only, no key): 0.79 @ 40
  candidates.** This is the ceiling the LLM can select from offline; the
  semantic embedding layer raises it further when a key is present.
- **Unit / API / guardrail tests: 22 passing** (`pytest`), covering schema
  shape, health, the vague-turn-1 no-recommend probe, off-topic refusal,
  hallucinated-id rejection, shortlist clamping, and anchor availability.

> Full end-to-end Recall@10 and behavior-probe pass-rate require a live
> `GEMINI_API_KEY`; run `python -m evalsuite.replay_eval` to reproduce with the
> real LLM agent.

## What didn't work / trade-offs
- **Per-turn multi-query RRF fusion** hurt recall (0.65 vs 0.79 @ same K):
  fusing many narrow per-turn rankings displaced specific technical tests in
  multi-constraint conversations. A single rich combined query won, so that is
  what ships (measured, in `evalsuite`).
- **Adding the latest turn as a second RRF query** was worse still (0.58),
  because rank-based fusion over-boosted one narrow ranking's top items.
- **Vector database** was considered and rejected — 377 items don't justify the
  operational surface area; an in-memory index is simpler and faster to deploy.

## AI-tool usage
Development was AI-assisted: an agentic coding assistant was used to scaffold
boilerplate (FastAPI wiring, Pydantic models, test skeletons) and to iterate on
the retrieval experiments. All design decisions — the grounding-by-id guardrail,
the anchor-instrument strategy, the combined-query finding, and the prompt
policy — were reviewed and are defensible independently of the tool.
