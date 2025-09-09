"""
Loader for raw RSS-entry JSON files produced by rss_orchestrator.

Expected layout
---------------
data/raw/rss/<ticker>/<entry_*.json>

The file already contains the fields we care about:
    - title
    - summary
    - content
    - link          (original article URL)
    - source_url    (feed url)

We turn that into the “raw text” that the existing cleaners expect.
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Iterator


class RSSLoader:
    EXTENSIONS = {".json"}  # let the factory discover us

    def _iter_records(self, path: Path) -> Iterator[dict]:
        if "/rss/" not in str(path):  # cheap guard so we do *not* grab CL json
            return
        data = json.loads(path.read_text(encoding="utf-8"))

        ticker = path.parts[path.parts.index("rss") + 1].upper()

        title = data.get("title", "").strip()
        summary = data.get("summary", "").strip()
        content = data.get("content", "").strip()
        raw_text = " ".join(filter(None, (title, summary, content)))
        yield {
            "doc_id": path.stem,
            "speaker": ticker,
            "text": raw_text,
            "urls": [data.get("link")],
            "_src": str(path),
            # stash the parts for fallback
            #  "_rss_parts": (title, summary, content),
        }
