# src/corp_speech_risk_dataset/orchestrators/rss_orchestrator.py

from pathlib import Path
import json
from loguru import logger
from typing import List, Optional

from corp_speech_risk_dataset.api.adapters.rss.rss_client import RSSClient
from corp_speech_risk_dataset.types.schemas.models import APIConfig
from corp_speech_risk_dataset.api.config.rss_config import RSS_FEEDS


class RSSOrchestrator:
    """
    Orchestrates fetching, optional deduplication, and saving of RSS entries
    for a list of tickers.
    """

    def __init__(
        self,
        config: APIConfig,
        tickers: Optional[List[str]] = None,
        outdir: str = "data/raw/rss",
        limit_per_ticker: Optional[int] = None,
        dedupe: bool = True,
    ):
        self.client = RSSClient(config)
        # default to all tickers in config
        self.tickers = [t.upper() for t in (tickers or list(RSS_FEEDS.keys()))]
        self.outdir = Path(outdir)
        self.limit = limit_per_ticker
        self.dedupe = dedupe

    def run(self):
        logger.info(f"Starting RSS orchestration for {len(self.tickers)} tickers")
        for ticker in self.tickers:
            self._process_ticker(ticker)
        logger.success("RSS orchestration complete")

    def _process_ticker(self, ticker: str):
        logger.info(f"[{ticker}] Fetching entries…")
        entries = self.client.fetch(ticker, limit=self.limit)
        logger.info(f"[{ticker}] Retrieved {len(entries)} raw entries")

        if self.dedupe:
            entries = self._dedupe(entries)
            logger.info(f"[{ticker}] {len(entries)} entries after deduplication")

        ticker_dir = self.outdir / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        for e in entries:
            eid = e.get("id") or e.get("link")
            slug = self._slug(eid)
            path = ticker_dir / f"entry_{slug}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(e, f, ensure_ascii=False, indent=2)

        logger.info(f"[{ticker}] Saved {len(entries)} entries to {ticker_dir}")

    @staticmethod
    def _dedupe(entries: List[dict]) -> List[dict]:
        seen = set()
        unique = []
        for e in entries:
            key = e.get("id") or e.get("link")
            if key and key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    @staticmethod
    def _slug(text: str) -> str:
        return (
            text.replace("://", "_")
            .replace("/", "_")
            .replace("?", "_")
            .replace("&", "_")
            .replace("=", "_")
        )
