#!/usr/bin/env python3
"""
CLI interface for corpus-api module.

Usage:
    python -m corpus_api.cli.fetch courtlistener --query query.yaml --output docs.jsonl
    python -m corpus_api.cli.fetch rss --feeds feeds.yaml --output docs.jsonl
    python -m corpus_api.cli.fetch wikipedia --pages pages.yaml --output docs.jsonl
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..adapters.courtlistener.courtlistener_client import CourtListenerClient
from ..adapters.rss.rss_client import RSSClient
from ..adapters.wikipedia.sandp_scraper import WikipediaScraper

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def courtlistener(
    query_file: Path = typer.Option(
        ..., "--query", help="Query configuration YAML file"
    ),
    output_file: Path = typer.Option(
        ..., "--output", help="Output JSONL file for documents"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="API configuration file"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="CourtListener API key (or set COURT_LISTENER_API_KEY env var)",
    ),
):
    """
    Fetch documents from CourtListener API based on query configuration.
    """
    logger.info(f"Fetching documents from CourtListener using query: {query_file}")
    logger.info(f"Output: {output_file}")

    # Load query configuration
    import yaml

    with open(query_file) as f:
        query_config = yaml.safe_load(f)

    # Initialize client
    client = CourtListenerClient(api_key=api_key)

    # Execute query
    documents = client.search_opinions(query_config.get("courtlistener", {}))

    # Write output
    with open(output_file, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Successfully fetched {len(documents)} documents")


@app.command()
def rss(
    feeds_file: Path = typer.Option(
        ..., "--feeds", help="RSS feeds configuration YAML file"
    ),
    output_file: Path = typer.Option(
        ..., "--output", help="Output JSONL file for documents"
    ),
    max_entries: int = typer.Option(
        100, "--max-entries", help="Maximum entries per feed"
    ),
):
    """
    Fetch documents from RSS feeds.
    """
    logger.info(f"Fetching documents from RSS feeds: {feeds_file}")
    logger.info(f"Output: {output_file}")

    # Load feeds configuration
    import yaml

    with open(feeds_file) as f:
        feeds_config = yaml.safe_load(f)

    # Initialize client
    client = RSSClient()

    # Fetch from all feeds
    all_documents = []
    for feed_name, feed_config in feeds_config.get("feeds", {}).items():
        documents = client.fetch_feed(feed_config["url"], max_entries=max_entries)
        all_documents.extend(documents)

    # Write output
    with open(output_file, "w") as f:
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Successfully fetched {len(all_documents)} documents from RSS feeds")


@app.command()
def wikipedia(
    pages_file: Path = typer.Option(
        ..., "--pages", help="Wikipedia pages configuration YAML file"
    ),
    output_file: Path = typer.Option(
        ..., "--output", help="Output JSONL file for documents"
    ),
):
    """
    Scrape documents from Wikipedia pages.
    """
    logger.info(f"Scraping documents from Wikipedia pages: {pages_file}")
    logger.info(f"Output: {output_file}")

    # Load pages configuration
    import yaml

    with open(pages_file) as f:
        pages_config = yaml.safe_load(f)

    # Initialize scraper
    scraper = WikipediaScraper()

    # Scrape all pages
    all_documents = []
    for page_name, page_config in pages_config.get("pages", {}).items():
        content = scraper.scrape_page(page_config["title"])
        if content:
            doc = {
                "doc_id": f"wiki_{page_name}",
                "source_uri": f"https://en.wikipedia.org/wiki/{page_config['title']}",
                "retrieved_at": "2024-01-01T00:00:00Z",  # Would use current timestamp
                "raw_text": content,
                "meta": {
                    "source": "wikipedia",
                    "page_title": page_config["title"],
                    "sections": page_config.get("sections", []),
                },
            }
            all_documents.append(doc)

    # Write output
    with open(output_file, "w") as f:
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Successfully scraped {len(all_documents)} documents from Wikipedia")


if __name__ == "__main__":
    app()
