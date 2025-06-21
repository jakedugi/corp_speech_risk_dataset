"""Command-line interface for CourtListener batch processing."""

import os
from pathlib import Path
from typing import List, Optional
import argparse
import sys

import typer
from loguru import logger
import httpx

from corp_speech_risk_dataset.api.courtlistener import STATUTE_QUERIES, process_statutes, process_recap_data, process_docket_entries, process_recap_documents, process_full_docket, process_search_api, process_recap_fetch
from corp_speech_risk_dataset.config import load_config

app = typer.Typer(help="CourtListener batch search CLI")

RESOURCE_FIELDS = {
    "opinions": [
        "id", "caseName", "dateFiled", "snippet", "plain_text", "html", "html_lawbox", "cluster", "citations", "court", "judges", "type", "absolute_url"
    ],
    "dockets": [
        "id", "caseName", "court", "dateFiled", "docketNumber", "absolute_url"
    ],
    "docket_entries": [
        "id", "docket", "date_filed", "entry_number", "description", "recap_documents"
    ],
    "recap_docs": [
        "id", "docket_entry", "plain_text", "filepath_local", "ocr_status", "is_available", "date_created", "date_modified"
    ],
    "recap": [
        "id", "status", "created", "modified", "request_type", "docket", "recap_document"
    ]
}

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
    ),
    api_mode: str = typer.Option(
        "standard",
        "--api-mode",
        "-m",
        help="API mode to use (standard or recap)"
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
        
    # Validate API mode
    if api_mode not in ["standard", "recap"]:
        logger.error(f"Invalid API mode: {api_mode}. Must be 'standard' or 'recap'")
        raise typer.Exit(1)
        
    # Run searches
    try:
        process_statutes(
            statutes=statutes,
            config=config,
            pages=pages or config.default_pages,
            page_size=page_size or config.default_page_size,
            date_min=date_min or config.default_date_min,
            output_dir=output_dir or config.output_dir,
            api_mode=api_mode
        )
    except Exception as e:
        logger.exception("Error during batch processing")
        raise typer.Exit(1)

@app.command()
def recap(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search query for RECAP data"
    ),
    docket_id: Optional[int] = typer.Option(
        None,
        "--docket-id",
        "-d",
        help="Specific docket ID to fetch"
    ),
    pages: int = typer.Option(
        1,
        "--pages",
        "-p",
        help="Number of pages to fetch"
    ),
    page_size: int = typer.Option(
        50,
        "--page-size",
        help="Results per page (max 100)"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: data/raw/courtlistener/recap)"
    )
):
    """Fetch RECAP data from CourtListener."""
    # Load configuration
    config = load_config()
    
    # Run RECAP data processing
    try:
        process_recap_data(
            config=config,
            query=query,
            docket_id=docket_id,
            pages=pages,
            page_size=page_size,
            output_dir=output_dir
        )
    except Exception as e:
        logger.exception("Error during RECAP data processing")
        raise typer.Exit(1)

@app.command()
def docket_entries(
    docket_id: Optional[int] = typer.Option(
        None,
        "--docket-id",
        "-d",
        help="Specific docket ID to fetch entries for"
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search query for docket entries"
    ),
    order_by: str = typer.Option(
        "-date_filed",
        "--order-by",
        help="Field to order by (e.g., date_filed, -date_filed, entry_number)"
    ),
    pages: int = typer.Option(
        1,
        "--pages",
        "-p",
        help="Number of pages to fetch"
    ),
    page_size: int = typer.Option(
        20,
        "--page-size",
        help="Results per page (max 20 recommended due to nested docs)"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    api_mode: str = typer.Option(
        "standard",
        "--api-mode",
        "-m",
        help="API mode to use (standard or recap)"
    )
):
    """Fetch docket entries with nested RECAP documents."""
    # Load configuration
    config = load_config()
    
    # Validate API mode
    if api_mode not in ["standard", "recap"]:
        logger.error(f"Invalid API mode: {api_mode}. Must be 'standard' or 'recap'")
        raise typer.Exit(1)
    
    # Run docket entries processing
    try:
        process_docket_entries(
            config=config,
            docket_id=docket_id,
            query=query,
            order_by=order_by,
            pages=pages,
            page_size=page_size,
            output_dir=output_dir,
            api_mode=api_mode
        )
    except Exception as e:
        logger.exception("Error during docket entries processing")
        raise typer.Exit(1)

@app.command()
def documents(
    docket_id: Optional[int] = typer.Option(
        None,
        "--docket-id",
        "-d",
        help="Docket ID to fetch all documents for"
    ),
    docket_entry_id: Optional[int] = typer.Option(
        None,
        "--entry-id",
        "-e",
        help="Specific docket entry ID to fetch documents for"
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Search query for RECAP documents"
    ),
    order_by: str = typer.Option(
        "-date_created",
        "--order-by",
        help="Field to order by (e.g., date_created, -date_created)"
    ),
    pages: int = typer.Option(
        1,
        "--pages",
        "-p",
        help="Number of pages to fetch"
    ),
    page_size: int = typer.Option(
        100,
        "--page-size",
        help="Results per page"
    ),
    include_text: bool = typer.Option(
        True,
        "--include-text",
        help="Include plain text content (can be large)"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    api_mode: str = typer.Option(
        "standard",
        "--api-mode",
        "-m",
        help="API mode to use (standard or recap)"
    )
):
    """Fetch RECAP documents with full text content."""
    # Load configuration
    config = load_config()
    
    # Validate API mode
    if api_mode not in ["standard", "recap"]:
        logger.error(f"Invalid API mode: {api_mode}. Must be 'standard' or 'recap'")
        raise typer.Exit(1)
    
    # Run RECAP documents processing
    try:
        process_recap_documents(
            config=config,
            docket_id=docket_id,
            docket_entry_id=docket_entry_id,
            query=query,
            order_by=order_by,
            pages=pages,
            page_size=page_size,
            include_plain_text=include_text,
            output_dir=output_dir,
            api_mode=api_mode
        )
    except Exception as e:
        logger.exception("Error during RECAP documents processing")
        raise typer.Exit(1)

@app.command()
def full_docket(
    docket_id: int = typer.Argument(
        ...,
        help="The docket ID to fetch"
    ),
    include_documents: bool = typer.Option(
        True,
        "--include-documents",
        help="Include full document text"
    ),
    order_by: str = typer.Option(
        "-date_filed",
        "--order-by",
        help="How to order docket entries"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    api_mode: str = typer.Option(
        "standard",
        "--api-mode",
        "-m",
        help="API mode to use (standard or recap)"
    )
):
    """Fetch a complete docket with all entries and documents."""
    # Load configuration
    config = load_config()
    
    # Validate API mode
    if api_mode not in ["standard", "recap"]:
        logger.error(f"Invalid API mode: {api_mode}. Must be 'standard' or 'recap'")
        raise typer.Exit(1)
    
    # Run full docket processing
    try:
        process_full_docket(
            config=config,
            docket_id=docket_id,
            include_documents=include_documents,
            order_by=order_by,
            output_dir=output_dir,
            api_mode=api_mode
        )
    except Exception as e:
        logger.exception("Error during full docket processing")
        raise typer.Exit(1)

@app.command()
def fetch(
    resource_type: str = typer.Argument(..., help="Resource type (opinions, dockets, docket_entries, recap_docs, recap)"),
    output_dir: str = typer.Option("data/", help="Output directory"),
    param: List[str] = typer.Option(None, help="Query params as key=value"),
    limit: int = typer.Option(10, help="Maximum number of results to save (default: 10)"),
    show_fields: bool = typer.Option(False, help="Show available fields for the resource type and exit")
):
    """Fetch any resource from CourtListener."""
    from corp_speech_risk_dataset.api.courtlistener import CourtListenerClient
    from corp_speech_risk_dataset.config import load_config
    import json

    if show_fields:
        fields = RESOURCE_FIELDS.get(resource_type)
        if fields:
            print(f"Available fields for {resource_type}:\n  " + ", ".join(fields))
        else:
            print(f"No field info for resource type: {resource_type}")
        raise typer.Exit(0)

    params = dict(p.split("=", 1) for p in param) if param else {}
    config = load_config()
    client = CourtListenerClient(config)
    try:
        results = client.fetch_resource(resource_type, params, limit=limit)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.reason_phrase}\nURL: {e.request.url}\nMessage: {e.response.text}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)
    outdir = Path(output_dir) / resource_type
    outdir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(results):
        with open(outdir / f"{resource_type}_{i}.json", "w") as f:
            json.dump(item, f, indent=2)
    print(f"Saved {len(results)} {resource_type} to {outdir}")

@app.command()
def search_api(
    param: List[str] = typer.Option(None, help="Query params as key=value, e.g. q=foo type=o order_by=dateFiled"),
    output_dir: Path = typer.Option(None, help="Optional output directory to save results as JSON"),
    limit: int = typer.Option(None, help="Limit number of results (client-side, not API param)"),
    show_url: bool = typer.Option(False, help="Print the full API URL used and exit"),
    token: Optional[str] = typer.Option(None, help="CourtListener API token (overrides config if set)")
):
    """Directly query the /api/rest/v4/search/ endpoint with arbitrary parameters."""
    # Load configuration
    config = load_config()
    params = dict(p.split("=", 1) for p in param) if param else {}
    try:
        process_search_api(
            config=config,
            params=params,
            output_dir=output_dir,
            limit=limit,
            show_url=show_url,
            token=token
        )
    except Exception as e:
        logger.exception("Error during search API processing")
        raise typer.Exit(1)

@app.command()
def recap_fetch(
    post_param: List[str] = typer.Option(None, help="POST params as key=value, e.g. request_type=1 docket_number=5:16-cv-00432 court=okwd"),
    show_url: bool = typer.Option(False, help="Print the POST URL and data, do not send request"),
    token: Optional[str] = typer.Option(None, help="CourtListener API token (overrides config if set)")
):
    """[TEST/FALLBACK] POST to /api/rest/v4/recap-fetch/ to trigger a PACER fetch. Will not actually purchase anything."""
    config = load_config()
    post_data = dict(p.split("=", 1) for p in post_param) if post_param else {}
    try:
        process_recap_fetch(
            config=config,
            post_data=post_data,
            show_url=show_url,
            token=token
        )
    except Exception as e:
        logger.exception("Error during recap-fetch POST")
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