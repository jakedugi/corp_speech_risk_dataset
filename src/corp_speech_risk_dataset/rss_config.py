# src/corp_speech_risk_dataset/config.py

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

from loguru import logger


@dataclass
class Config:
    """Generic configuration (RSS does not require a token)."""

    api_token: Optional[str] = None
    rate_limit: float = 0.25


def load_config() -> Config:
    """Load config (RSS workflows don’t need an API token)."""
    try:
        token = os.getenv("API_TOKEN")
        cfg = Config(api_token=token, rate_limit=0.25)
        if cfg.api_token is None:
            logger.debug("No API_TOKEN set; proceeding in unauthenticated mode")
        return cfg
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


# -----------------------------------------------------------------------------
# RSS feed endpoints for key S&P 500 tickers
# -----------------------------------------------------------------------------
RSS_FEEDS: Dict[str, List[str]] = {
    "NVDA": [
        "https://nvidianews.nvidia.com/rss.xml",
        "https://feeds.feedburner.com/nvidiablog",
        "https://developer.nvidia.com/blog/feed",
        "https://rsshub.app/twitter/user/nvidia",
    ],
    "MSFT": [
        "https://news.microsoft.com/feed/",
        "https://blogs.microsoft.com/feed/",
        "https://rsshub.app/twitter/user/Microsoft",
    ],
    "AAPL": [
        "https://www.apple.com/newsroom/rss-feed.rss",
        "https://developer.apple.com/news/rss/news.rss",
        "https://rsshub.app/twitter/user/Apple",
    ],
    "AMZN": [
        "https://www.aboutamazon.com/about-amazon-rss.rss",
        "https://rsshub.app/twitter/user/amazon",
    ],
    "META": [
        "https://about.fb.com/feed/",
        "https://rsshub.app/twitter/user/Meta",
    ],
    "GOOG": ["https://blog.google/rss/", "https://rsshub.app/twitter/user/google"],
}
