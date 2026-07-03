"""Agent orchestration: retrieve candidates, call the LLM, enforce guardrails.

The :class:`Agent` is the seam between the HTTP layer and the model. It:

1. Builds a retrieval query from the conversation and fetches grounded
   candidates.
2. Asks the LLM for a single structured decision (clarify/recommend/compare/
   refuse).
3. Post-validates that decision against the catalog — dropping any hallucinated
   ids, clamping the shortlist to 1..10, and coercing the response into the
   exact API schema.

If the LLM is unavailable (no key, or an error mid-request) it degrades to a
deterministic heuristic so the endpoint never 500s and stays schema-compliant.
"""

from __future__ import annotations

from typing import List, Optional

from .catalog import Assessment, Catalog, get_catalog
from .config import settings
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever
from .schemas import ChatResponse, Message, Recommendation

_VALID_ACTIONS = {"clarify", "recommend", "compare", "refuse"}

# Cheap first-pass guard for obvious out-of-scope / injection attempts. The LLM
# is the primary gate; this only powers the no-key fallback path.
_OFFTOPIC_HINTS = (
    "ignore previous", "ignore your", "system prompt", "reveal your",
    "disregard", "legal", "lawsuit", "salary", "how much should i pay",
    "write me a poem", "capital of", "recipe",
)


def _assistant_turns_left(num_messages: int) -> int:
    """How many assistant replies remain within the turn cap (this one incl.)."""
    remaining = (settings.max_turns - num_messages + 1) // 2
    return max(1, remaining)


def _looks_vague(text: str) -> bool:
    """Heuristic: a short one-liner with no concrete role/skill signal."""
    t = text.lower().strip()
    if len(t.split()) <= 6 and "assessment" in t or t in {"hi", "hello", "help"}:
        return True
    generic = ("i need an assessment", "need a solution", "need an assessment",
               "recommend something", "we need a solution")
    return any(g in t for g in generic)


class Agent:
    def __init__(self, catalog: Optional[Catalog] = None, retriever: Optional[Retriever] = None):
        self.catalog = catalog or get_catalog()
        self.retriever = retriever or Retriever(self.catalog)

    def warm_up(self) -> None:
        self.retriever.warm_up()

    # -- retrieval query ----------------------------------------------------
    @staticmethod
    def _build_queries(messages: List[Message]) -> List[str]:
        """Build the retrieval query from the conversation.

        We concatenate all user turns into one query. Empirically this beat
        per-turn RRF fusion on the sample traces: fusing many narrow per-turn
        rankings displaced specific tests (e.g. the Java/SQL tests in a
        multi-constraint conversation), whereas one rich query keeps recall high.
        """
        user_parts = [m.content.strip() for m in messages if m.role == "user" and m.content.strip()]
        if not user_parts:
            return []
        return [" ".join(user_parts)]

    # -- main entry ---------------------------------------------------------
    def respond(self, messages: List[Message]) -> ChatResponse:
        # Only real conversational turns count toward the cap.
        convo = [m for m in messages if m.role in ("user", "assistant")]
        turns_left = _assistant_turns_left(len(convo))

        queries = self._build_queries(convo)
        candidates = self.retriever.retrieve(queries) if queries else []

        if settings.llm_enabled:
            try:
                return self._llm_respond(convo, candidates, turns_left)
            except Exception:
                # Never leak an exception to the caller; fall back deterministically.
                return self._fallback(convo, candidates, turns_left)
        return self._fallback(convo, candidates, turns_left)

    # -- LLM path -----------------------------------------------------------
    def _llm_respond(
        self, messages: List[Message], candidates: List[Assessment], turns_left: int
    ) -> ChatResponse:
        from .llm import ChatClient

        client = ChatClient()
        user_prompt = build_user_prompt(messages, candidates, turns_left)
        data = client.complete_json(SYSTEM_PROMPT, user_prompt)

        action = str(data.get("action", "")).strip().lower()
        reply = str(data.get("reply", "")).strip()
        raw_ids = data.get("recommendation_ids") or []
        end = bool(data.get("end_of_conversation", False))

        if action not in _VALID_ACTIONS:
            action = "recommend" if raw_ids else "clarify"

        # Ground the ids: only ids the model was shown are allowed, and they must
        # exist in the catalog. This is the anti-hallucination backstop.
        allowed = {a.entity_id for a in candidates}
        wanted = [str(i).strip() for i in raw_ids if str(i).strip() in allowed]
        chosen = self.catalog.resolve_ids(wanted)

        if action in ("clarify", "refuse"):
            chosen = []  # never attach a shortlist while clarifying/refusing
            end = False
        else:
            chosen = chosen[: settings.max_recommendations]

        if not reply:
            reply = self._default_reply(action, chosen)

        return ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**a.to_recommendation()) for a in chosen],
            end_of_conversation=end and action in ("recommend", "compare"),
        )

    # -- deterministic fallback --------------------------------------------
    def _fallback(
        self, messages: List[Message], candidates: List[Assessment], turns_left: int
    ) -> ChatResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        lowered = last_user.lower()

        if any(h in lowered for h in _OFFTOPIC_HINTS):
            return ChatResponse(
                reply=(
                    "I can only help with selecting SHL assessments, so I can't help "
                    "with that. Tell me about the role you're hiring for and I'll "
                    "suggest assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        first_turn = sum(1 for m in messages if m.role == "user") <= 1
        if first_turn and _looks_vague(last_user) and turns_left > 1:
            return ChatResponse(
                reply=(
                    "Happy to help. Who is the assessment for — the role, seniority "
                    "level, and the key skills or behaviours you want to measure?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        top = candidates[: min(5, settings.max_recommendations)]
        if not top:
            return ChatResponse(
                reply=(
                    "Could you share a bit more about the role and the skills you "
                    "want to assess? That will let me suggest the right assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )
        return ChatResponse(
            reply=(
                "Based on what you've described, here are assessments from the SHL "
                "catalog that fit. Let me know if you'd like to refine the list."
            ),
            recommendations=[Recommendation(**a.to_recommendation()) for a in top],
            end_of_conversation=False,
        )

    @staticmethod
    def _default_reply(action: str, chosen: List[Assessment]) -> str:
        if action == "clarify":
            return "Could you tell me a bit more about the role and level so I can recommend well?"
        if action == "refuse":
            return "I can only help with selecting SHL assessments. What role are you hiring for?"
        if chosen:
            return "Here is a shortlist of SHL assessments that fit your needs."
        return "I couldn't find a matching SHL assessment for that. Could you refine the requirement?"


_AGENT: Optional[Agent] = None


def get_agent() -> Agent:
    """Process-wide singleton agent (keeps the retriever/embeddings warm)."""
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent()
        _AGENT.warm_up()
    return _AGENT
