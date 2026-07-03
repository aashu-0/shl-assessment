"""Runtime configuration, loaded from environment variables.

Everything the service needs to run is captured here so the rest of the code
never reaches into ``os.environ`` directly. Values are read once at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load a local .env if present. This is a convenience for local development;
# in production the platform injects real environment variables.
try:  # pragma: no cover - trivial glue
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


_ROOT = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the service configuration."""

    # --- LLM (Google Gemini) ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.0-flash").strip()
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-004").strip()

    # --- Catalog / data ---
    catalog_path: Path = Path(
        os.getenv("CATALOG_PATH", str(_ROOT / "data" / "shl_product_catalog.json"))
    )
    embed_cache_path: Path = Path(
        os.getenv("EMBED_CACHE_PATH", str(_ROOT / "data" / "embeddings.pkl"))
    )

    # --- Retrieval knobs ---
    # Number of candidate assessments handed to the LLM to choose from.
    retrieval_candidates: int = int(os.getenv("RETRIEVAL_CANDIDATES", "40"))
    # Turn on semantic embeddings (needs the API key). BM25 is always available.
    use_embeddings: bool = _get_bool("USE_EMBEDDINGS", True)

    # --- Agent / conversation ---
    # The evaluator caps a conversation at 8 messages (user + assistant).
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))
    # Max assessments in a committed shortlist (spec: 1..10).
    max_recommendations: int = int(os.getenv("MAX_RECOMMENDATIONS", "10"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
