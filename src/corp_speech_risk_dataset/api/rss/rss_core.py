# src/corp_speech_risk_dataset/api/rss/rss_core.py

from __future__ import annotations
import feedparser
from typing import Any, Dict, List
from corp_speech_risk_dataset.rss_config import RSS_FEEDS


class RSSCore:
    """Low-level RSS/Atom parsing logic."""

    @staticmethod
    def parse_feed(url: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS/Atom feed URL.
        Returns a list of normalized entry dicts.
        """
        parsed = feedparser.parse(url)
        entries: List[Dict[str, Any]] = []

        for e in parsed.entries:
            entries.append(
                {
                    "source_url": url,
                    "id": e.get("id", e.get("link")),
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "summary": e.get("summary", ""),
                    "content": getattr(e, "content", [{"value": ""}])[0]["value"],
                }
            )

        return entries

    def fetch_for_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Look up feed URLs for a given ticker in config,
        parse them all, and return a combined list.
        """
        urls = RSS_FEEDS.get(ticker.upper(), [])
        all_entries: List[Dict[str, Any]] = []
        for url in urls:
            all_entries.extend(self.parse_feed(url))
        return all_entries
