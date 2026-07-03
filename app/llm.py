"""Thin wrappers around the Google Gemini SDK (``google-genai``).

Two clients:

* :class:`ChatClient` — one structured generation call per turn. It asks the
  model to return strict JSON and parses it defensively.
* :class:`EmbeddingClient` — batch-embeds the catalog once (cached to disk keyed
  by model + content hash) and embeds a query per turn.

Both are intentionally provider-thin so the rest of the app stays testable via
mocks without importing the SDK.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import List, Optional

from .config import settings

# JSON emitted by LLMs is often wrapped in ```json fences or trailed by prose.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _lazy_client():
    """Create a Gemini client, importing the SDK lazily.

    Raises a clear error if the key or SDK is missing so callers can fall back.
    """
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai  # imported lazily; heavy + optional in tests

    return genai.Client(api_key=settings.gemini_api_key)


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        raise ValueError("empty model output")
    candidate = text.strip()

    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    # Fast path: the whole thing is JSON.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the outermost {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(candidate[start : end + 1])
    raise ValueError("no JSON object found in model output")


class ChatClient:
    """Structured single-call chat client."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.chat_model
        self._client = _lazy_client()

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Run one generation and return the parsed JSON dict."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=1400,
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        text = getattr(resp, "text", None) or ""
        return extract_json(text)


class EmbeddingClient:
    """Batch/query embeddings with an on-disk cache for the catalog."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.embed_model
        self._client = _lazy_client()

    # -- catalog (batched, cached) -----------------------------------------
    def embed_catalog(self, texts: List[str]) -> List[List[float]]:
        cache_path = settings.embed_cache_path
        digest = self._digest(texts)
        cached = self._load_cache(cache_path, digest)
        if cached is not None:
            return cached

        vectors: List[List[float]] = []
        # The API accepts batches; keep them modest to stay within limits.
        batch = 50
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            vectors.extend(self._embed(chunk, task="retrieval_document"))
        self._save_cache(cache_path, digest, vectors)
        return vectors

    # -- single query -------------------------------------------------------
    def embed_query(self, text: str) -> Optional[List[float]]:
        out = self._embed([text], task="retrieval_query")
        return out[0] if out else None

    # -- internals ----------------------------------------------------------
    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        from google.genai import types

        resp = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        return [list(e.values) for e in resp.embeddings]

    @staticmethod
    def _digest(texts: List[str]) -> str:
        h = hashlib.sha256()
        h.update(settings.embed_model.encode("utf-8"))
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _load_cache(path: Path, digest: str) -> Optional[List[List[float]]]:
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("digest") == digest:
                return blob.get("vectors")
        except Exception:
            return None
        return None

    @staticmethod
    def _save_cache(path: Path, digest: str, vectors: List[List[float]]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({"digest": digest, "vectors": vectors}, fh)
        except Exception:
            pass  # cache is a nicety, never fatal
