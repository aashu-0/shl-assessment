"""Replay the sample traces against the agent and report Recall@K.

This mirrors (in spirit) the grader's replay harness: it feeds each trace's
user turns to our stateless agent, accumulates the conversation, and scores the
final committed shortlist against the trace's expected URLs.

Usage:
    python -m evalsuite.replay_eval            # replay + score
    python -m evalsuite.replay_eval --coverage # just check catalog coverage

Without GEMINI_API_KEY the agent runs its deterministic fallback, so numbers are
a floor; set the key to measure the real LLM agent.
"""

from __future__ import annotations

import argparse
from typing import List, Set

from app.agent import Agent
from app.catalog import get_catalog
from app.config import settings
from app.schemas import Message

from .traces import Trace, load_traces


def _norm(url: str) -> str:
    return url.rstrip("/").lower()


def catalog_urls() -> Set[str]:
    return {_norm(a.url) for a in get_catalog()}


def coverage_report() -> None:
    """How many expected URLs even exist in our catalog (recall ceiling)."""
    cat = catalog_urls()
    print("=== Expected-shortlist coverage vs catalog ===")
    total_hit = total = 0
    for t in load_traces():
        hit = sum(1 for u in t.expected_urls if u in cat)
        total_hit += hit
        total += len(t.expected_urls)
        missing = [u for u in t.expected_urls if u not in cat]
        flag = "" if not missing else f"  MISSING: {len(missing)}"
        print(f"{t.name:>4}: {hit}/{len(t.expected_urls)} in catalog{flag}")
    print(f"---- overall ceiling: {total_hit}/{total} = {total_hit / total:.3f}")


def replay_trace(agent: Agent, trace: Trace, k: int = 10) -> float:
    """Run the conversation and return Recall@k on the final shortlist."""
    messages: List[Message] = []
    final_urls: Set[str] = set()

    for user_turn in trace.user_turns:
        messages.append(Message(role="user", content=user_turn))
        resp = agent.respond(messages)
        messages.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            final_urls = {_norm(r.url) for r in resp.recommendations[:k]}

    if not trace.expected_urls:
        return 0.0
    hits = len(final_urls & trace.expected_urls)
    return hits / len(trace.expected_urls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", action="store_true", help="only print coverage")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    if args.coverage:
        coverage_report()
        return

    print(f"LLM enabled: {settings.llm_enabled} (model={settings.chat_model})")
    agent = Agent()
    agent.warm_up()

    scores = []
    print(f"=== Recall@{args.k} per trace ===")
    for t in load_traces():
        r = replay_trace(agent, t, k=args.k)
        scores.append(r)
        print(f"{t.name:>4}: {r:.3f}")
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"---- Mean Recall@{args.k}: {mean:.3f}")


if __name__ == "__main__":
    main()
