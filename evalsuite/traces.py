"""Parse the provided sample conversation traces (``C1.md`` .. ``C10.md``).

Each trace is a multi-turn markdown conversation. The *expected shortlist* for a
trace is the final assistant recommendation table in that file (the labeled
"answer"). We extract:

* the ordered list of user turns (to replay against our agent), and
* the set of expected assessment names / URLs (to score Recall@K).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

TRACES_DIR = Path(__file__).resolve().parent / "traces"

_USER_BLOCK_RE = re.compile(
    r"\*\*User\*\*\s*\n+(.*?)(?=\n\*\*Agent\*\*|\Z)", re.DOTALL
)
# Markdown table rows that carry a catalog URL in the last cell.
_URL_RE = re.compile(r"https?://www\.shl\.com/products/product-catalog/view/[^\s<>|)]+")


def _clean_user_text(block: str) -> str:
    lines = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # User quotes are markdown blockquotes ("> ...").
        line = re.sub(r"^>\s?", "", line)
        lines.append(line)
    return " ".join(lines).strip()


@dataclass
class Trace:
    name: str
    user_turns: List[str] = field(default_factory=list)
    expected_urls: Set[str] = field(default_factory=set)
    expected_names: List[str] = field(default_factory=list)


def _normalize_url(url: str) -> str:
    return url.rstrip("/").lower()


def parse_trace(path: Path) -> Trace:
    text = path.read_text(encoding="utf-8")

    user_turns = [_clean_user_text(b) for b in _USER_BLOCK_RE.findall(text)]
    user_turns = [u for u in user_turns if u]

    # The expected shortlist = URLs in the FINAL agent table of the trace.
    # Split on turns and take the last block that contains catalog URLs.
    turn_chunks = re.split(r"### Turn \d+", text)
    expected_urls: Set[str] = set()
    for chunk in reversed(turn_chunks):
        urls = _URL_RE.findall(chunk)
        if urls:
            expected_urls = {_normalize_url(u) for u in urls}
            break

    return Trace(
        name=path.stem,
        user_turns=user_turns,
        expected_urls=expected_urls,
    )


def load_traces() -> List[Trace]:
    traces = []
    for p in sorted(TRACES_DIR.glob("C*.md"), key=lambda x: int(re.sub(r"\D", "", x.stem) or 0)):
        traces.append(parse_trace(p))
    return traces


if __name__ == "__main__":  # quick sanity dump
    for t in load_traces():
        print(f"{t.name}: {len(t.user_turns)} user turns, {len(t.expected_urls)} expected urls")
