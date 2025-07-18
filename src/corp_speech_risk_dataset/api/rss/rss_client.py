# src/corp_speech_risk_dataset/api/rss/rss_client.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from corp_speech_risk_dataset.custom_types.base_types import APIConfig
from .rss_core import RSSCore


class RSSClient:
    """
    High-level client for fetching, normalizing, and saving RSS data
    for S&P 500 tickers.
    """

    def __init__(self, config: APIConfig, timeout: float = 10.0):
        self.config = config
        self.logger = logger.bind(client="rss")
        self.http = httpx.Client(timeout=timeout)
        self.core = RSSCore()

    def fetch(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch RSS entries for `ticker`. Returns a list of dicts.
        If `limit` is provided, truncates the combined list.
        """
        entries = self.core.fetch_for_ticker(ticker)
        if limit is not None:
            return entries[:limit]
        return entries

    def save(
        self,
        ticker: str,
        entries: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Save each entry as a JSON file under:
          data/raw/rss/{ticker}/entry_{slug}.json
        """
        if output_dir is None:
            output_dir = Path("data") / "raw" / "rss" / ticker.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        for e in entries:
            eid = e.get("id") or e.get("link")
            slug = self._slug(eid)
            path = output_dir / f"entry_{slug}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(e, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"Saved {len(entries)} RSS entries for {ticker} to {output_dir}"
        )

    def fetch_and_save(
        self,
        ticker: str,
        limit: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Convenience method to fetch then save in one call.
        """
        entries = self.fetch(ticker, limit=limit)
        self.save(ticker, entries, output_dir)

    @staticmethod
    def _slug(text: str) -> str:
        """Make filesystem-safe slug from text."""
        return (
            text.replace("://", "_")
            .replace("/", "_")
            .replace("?", "_")
            .replace("&", "_")
            .replace("=", "_")
        )
