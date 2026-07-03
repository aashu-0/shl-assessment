"""Loads the SHL product catalog and exposes it as typed :class:`Assessment`s.

The raw catalog is a JSON list of individual test solutions scraped from
shl.com. This module normalizes each record into a small, predictable object
and precomputes the searchable text and SHL test-type code used elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

# SHL groups every product under one or more "keys". The public catalog and the
# evaluator use single-letter test-type codes; this is the canonical mapping.
KEY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass(frozen=True)
class Assessment:
    """A single individual test solution from the SHL catalog."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: List[str] = field(default_factory=list)
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""

    @property
    def test_type(self) -> str:
        """Comma-joined SHL codes, e.g. ``"K,S"``. Empty string if unknown."""
        codes = [KEY_TO_CODE[k] for k in self.keys if k in KEY_TO_CODE]
        # Preserve order but drop duplicates.
        seen: List[str] = []
        for c in codes:
            if c not in seen:
                seen.append(c)
        return ",".join(seen)

    def search_text(self) -> str:
        """Rich text blob used for lexical and semantic retrieval."""
        parts = [
            self.name,
            self.name,  # weight the name a little more heavily
            ", ".join(self.keys),
            ", ".join(self.job_levels),
            self.description,
        ]
        return "\n".join(p for p in parts if p)

    def to_recommendation(self) -> dict:
        """Shape used in the API response ``recommendations`` array."""
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


class Catalog:
    """In-memory index of the catalog with a few convenient lookups."""

    def __init__(self, assessments: List[Assessment]):
        self.assessments = assessments
        self._by_id: Dict[str, Assessment] = {a.entity_id: a for a in assessments}
        # Lower-cased name -> assessment, for exact "compare X vs Y" resolution.
        self._by_name: Dict[str, Assessment] = {
            a.name.strip().lower(): a for a in assessments
        }

    def __len__(self) -> int:
        return len(self.assessments)

    def __iter__(self):
        return iter(self.assessments)

    def get(self, entity_id: str) -> Optional[Assessment]:
        return self._by_id.get(str(entity_id))

    def by_name(self, name: str) -> Optional[Assessment]:
        return self._by_name.get(name.strip().lower())

    def resolve_ids(self, ids: List[str]) -> List[Assessment]:
        """Map entity ids to real assessments, dropping anything unknown.

        This is the guardrail that makes URL hallucination impossible: the LLM
        may only return ids, and any id not physically in the catalog is
        silently discarded before a recommendation is built.
        """
        out: List[Assessment] = []
        seen = set()
        for i in ids:
            a = self.get(str(i).strip())
            if a is not None and a.entity_id not in seen:
                out.append(a)
                seen.add(a.entity_id)
        return out


def _parse_record(rec: dict) -> Optional[Assessment]:
    entity_id = str(rec.get("entity_id", "")).strip()
    name = (rec.get("name") or "").strip()
    url = (rec.get("link") or "").strip()
    if not entity_id or not name or not url:
        # Without a stable id, name and a real URL the record is unusable.
        return None
    return Assessment(
        entity_id=entity_id,
        name=name,
        url=url,
        description=(rec.get("description") or "").strip(),
        keys=list(rec.get("keys") or []),
        job_levels=list(rec.get("job_levels") or []),
        languages=list(rec.get("languages") or []),
        duration=(rec.get("duration") or "").strip(),
        remote=(rec.get("remote") or "").strip(),
        adaptive=(rec.get("adaptive") or "").strip(),
    )


def load_catalog(path: Optional[Path] = None) -> Catalog:
    """Read and normalize the catalog JSON into a :class:`Catalog`."""
    path = Path(path) if path is not None else settings.catalog_path
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    assessments: List[Assessment] = []
    for rec in raw:
        parsed = _parse_record(rec)
        if parsed is not None:
            assessments.append(parsed)

    if not assessments:
        raise RuntimeError(f"No usable assessments loaded from {path}")
    return Catalog(assessments)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Process-wide singleton catalog."""
    return load_catalog()
