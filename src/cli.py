"""Command-line interface for CourtListener batch processing."""

import os
from pathlib import Path
from typing import List
import argparse
import sys

import typer
from loguru import logger

from src.api.courtlistener import STATUTE_QUERIES, process_statutes
from src.config import load_config

app = typer.Typer(help="CourtListener batch search CLI")

@app.command()
def search(
    statutes: List[str] = typer.Option(
        list(STATUTE_QUERIES.keys()),
        "--statutes",
        "-s",
        help="Statutes to search (default: all)"
    ),
    pages: int = typer.Option(
        None,
        "--pages",
        "-p",
        help="Number of pages to fetch per statute"
    ),
    page_size: int = typer.Option(
        None,
        "--page-size",
        help="Results per page (max 100)"
    ),
    date_min: str = typer.Option(
        None,
        "--date-min",
        help="Earliest filing date (YYYY-MM-DD)"
    ),
    opinions: bool = typer.Option(
        False,
        "--opinions",
        help="Also fetch opinion texts"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: data/raw/courtlistener/YYYY-MM-DD)"
    )
):
    """Run batch searches for specified statutes."""
    # Load configuration
    config = load_config()
    
    # Validate statutes
    invalid_statutes = [s for s in statutes if s not in STATUTE_QUERIES]
    if invalid_statutes:
        logger.error(f"Unknown statutes: {', '.join(invalid_statutes)}")
        raise typer.Exit(1)
        
    # Run searches
    try:
        process_statutes(
            statutes=statutes,
            config=config,
            pages=pages or config.default_pages,
            page_size=page_size or config.default_page_size,
            date_min=date_min or config.default_date_min,
            output_dir=output_dir or config.output_dir
        )
    except Exception as e:
        logger.exception("Error during batch processing")
        raise typer.Exit(1)
        
def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Process statutes from CourtListener")
    parser.add_argument("--statutes", nargs="+", help="Statutes to process")
    parser.add_argument("--pages", type=int, help="Number of pages to fetch")
    parser.add_argument("--page-size", type=int, help="Number of results per page")
    parser.add_argument("--date-min", help="Minimum date to fetch from (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Directory to save results to")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Process statutes
    try:
        process_statutes(
            statutes=args.statutes,
            config=config,
            pages=args.pages or config.default_pages,
            page_size=args.page_size or config.default_page_size,
            date_min=args.date_min or config.default_date_min,
            output_dir=args.output_dir or config.output_dir
        )
    except Exception as e:
        logger.error(f"Error processing statutes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app() 