"""Grounded retrieval over the catalog.

The agent never invents assessments; it selects from a candidate set produced
here. Retrieval is a hybrid of:

* **BM25** lexical scoring — always available, offline, deterministic. Great at
  matching concrete skills ("Docker", "Spring", "HIPAA").
* **Gemini embeddings** semantic scoring — optional, enabled when an API key is
  present. Bridges vocabulary gaps ("senior leadership" -> OPQ / Verify).

Results are merged with Reciprocal Rank Fusion (RRF), which combines rankings
robustly without needing the two score scales to be comparable.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import Assessment, Catalog
from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

# SHL's flagship, cross-role instruments. These are broadly applicable defaults
# that hiring teams reach for regardless of the specific role (a personality
# baseline, a general-reasoning test, a graduate SJT, a skills self-assessment,
# a safety/dependability measure, and a motivation questionnaire). They rarely
# surface via keyword overlap with a role description ("senior leadership" has no
# lexical match to "Occupational Personality Questionnaire"), yet the agent
# legitimately recommends them when behavioural/cognitive fit matters. We keep
# them in the candidate pool so the LLM *can* choose them per policy — it still
# decides whether they actually belong in a given shortlist.
ANCHOR_ENTITY_IDS: Tuple[str, ...] = (
    "720",   # Occupational Personality Questionnaire OPQ32r        (P)
    "3971",  # SHL Verify Interactive G+                            (A)
    "741",   # Graduate Scenarios                                   (B)
    "4301",  # Global Skills Assessment                             (C,K)
    "731",   # Dependability and Safety Instrument (DSI)            (P)
    "724",   # Motivation Questionnaire MQM5                        (P)
)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Compact Okapi BM25 implementation (no third-party dependency)."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if corpus_tokens else 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            freqs: Dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_freqs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1
        n = len(corpus_tokens)
        # BM25+ style idf, floored at a small positive value.
        self.idf: Dict[str, float] = {
            tok: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()
        }

    def scores(self, query: str) -> List[float]:
        q_tokens = _tokenize(query)
        scores = [0.0] * len(self.corpus_tokens)
        for tok in q_tokens:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(tok, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores


def _rank_order(scores: Sequence[float]) -> List[int]:
    """Indices sorted by descending score."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class Retriever:
    """Hybrid retriever bound to a specific catalog."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.assessments: List[Assessment] = list(catalog)
        self._texts = [a.search_text() for a in self.assessments]
        self._bm25 = BM25([_tokenize(t) for t in self._texts])
        self._embeddings: Optional[List[List[float]]] = None
        self._embedder = None  # lazily created EmbeddingClient
        # Positions of anchor instruments within self.assessments.
        self._anchor_positions = [
            i for i, a in enumerate(self.assessments) if a.entity_id in ANCHOR_ENTITY_IDS
        ]

    # -- embeddings ---------------------------------------------------------
    def warm_up(self) -> None:
        """Precompute catalog embeddings if enabled. Safe to call repeatedly."""
        if not (settings.use_embeddings and settings.llm_enabled):
            return
        if self._embeddings is not None:
            return
        try:
            from .llm import EmbeddingClient

            self._embedder = EmbeddingClient()
            self._embeddings = self._embedder.embed_catalog(self._texts)
        except Exception:
            # Any failure (no key, quota, network) degrades gracefully to BM25.
            self._embeddings = None
            self._embedder = None

    def _semantic_scores(self, query: str) -> Optional[List[float]]:
        if self._embeddings is None or self._embedder is None:
            return None
        try:
            q = self._embedder.embed_query(query)
        except Exception:
            return None
        if q is None:
            return None
        return [_cosine(q, e) for e in self._embeddings]

    # -- public API ---------------------------------------------------------
    def _rankings_for_query(self, query: str) -> List[List[int]]:
        """Lexical (and semantic, if available) rank orders for one query."""
        lexical = self._bm25.scores(query)
        rankings = [_rank_order(lexical)]
        semantic = self._semantic_scores(query)
        if semantic is not None:
            rankings.append(_rank_order(semantic))
        return rankings

    def search(self, query: str, top_k: Optional[int] = None) -> List[Assessment]:
        """Single-query retrieval (no anchors). Used directly in tests."""
        top_k = top_k or settings.retrieval_candidates
        query = (query or "").strip()
        if not query:
            return []
        rankings = self._rankings_for_query(query)
        if len(rankings) == 1:  # lexical only — drop zero-score tail
            lexical = self._bm25.scores(query)
            order = rankings[0]
            ranked = [i for i in order if lexical[i] > 0] or order
            return [self.assessments[i] for i in ranked[:top_k]]
        fused = _rrf(rankings)
        return [self.assessments[i] for i, _ in fused[:top_k]]

    def retrieve(self, queries: Sequence[str], top_k: Optional[int] = None) -> List[Assessment]:
        """Multi-query retrieval used by the agent.

        Each query (typically each user turn plus the combined conversation) is
        ranked independently and fused with RRF, so a constraint mentioned early
        ("safety") is not drowned out by later text. The curated anchor
        instruments are always appended so the LLM may select them per policy.
        """
        top_k = top_k or settings.retrieval_candidates
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return [self.assessments[i] for i in self._anchor_positions]

        all_rankings: List[List[int]] = []
        for q in queries:
            all_rankings.extend(self._rankings_for_query(q))
        fused = _rrf(all_rankings)

        chosen: List[int] = [i for i, _ in fused[:top_k]]
        # Append anchors that were not already retrieved.
        seen = set(chosen)
        for i in self._anchor_positions:
            if i not in seen:
                chosen.append(i)
                seen.add(i)
        return [self.assessments[i] for i in chosen]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion. ``k`` dampens the contribution of low ranks."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
