"""CourtListener API client and batch processing functionality.

This module provides a client for interacting with the CourtListener API to fetch
dockets, opinions, and clusters related to corporate speech cases, along with
batch processing capabilities.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import httpx
from loguru import logger

from corp_speech_risk_dataset.api.base_api_client import BaseAPIClient
from corp_speech_risk_dataset.custom_types.base_types import APIConfig
from corp_speech_risk_dataset.api.courtlistener.queries import STATUTE_QUERIES

# API endpoints
BASE_DOCKETS_URL = "https://www.courtlistener.com/api/rest/v4/dockets/"
BASE_CLUSTERS_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
BASE_OPINIONS_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
BASE_DOCKET_ENTRIES_URL = "https://www.courtlistener.com/api/rest/v4/docket-entries/"
BASE_RECAP_DOCS_URL = "https://www.courtlistener.com/api/rest/v4/recap-documents/"
BASE_RECAP_URL = "https://www.courtlistener.com/api/rest/v4/recap/"

# API endpoint configurations
API_ENDPOINTS = {
    "standard": {
        "base_url": "https://www.courtlistener.com/api/rest/v4",
        "dockets": BASE_DOCKETS_URL,
        "clusters": BASE_CLUSTERS_URL,
        "opinions": BASE_OPINIONS_URL,
        "docket_entries": BASE_DOCKET_ENTRIES_URL,
        "recap_docs": BASE_RECAP_DOCS_URL,
    },
    "recap": {
        "base_url": "https://www.courtlistener.com/api/rest/v4",
        "dockets": BASE_DOCKETS_URL,
        "clusters": BASE_CLUSTERS_URL,
        "opinions": BASE_OPINIONS_URL,
        "docket_entries": BASE_DOCKET_ENTRIES_URL,
        "recap_docs": BASE_RECAP_DOCS_URL,
        "recap": BASE_RECAP_URL,
    }
}

def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

class CourtListenerClient(BaseAPIClient):
    """Client for interacting with the CourtListener API."""

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    BASE_OPINIONS_URL = f"{BASE_URL}/opinions/"

    def __init__(self, config: APIConfig, api_mode: str = "standard"):
        """Initialize the client with API configuration.
        
        Args:
            config: API configuration containing token and settings
            api_mode: API mode to use ("standard" or "recap")
        """
        super().__init__(config)
        self.logger = logger.bind(client="courtlistener")
        self.api_mode = api_mode
        self.endpoints = API_ENDPOINTS.get(api_mode, API_ENDPOINTS["standard"])
        self._session = httpx.Client(
            headers=self._build_headers(),
            follow_redirects=True,
            timeout=30.0
        )
        self._sleep = config.rate_limit or 0.25
        
    def _build_headers(self) -> Dict[str, str]:
        """Build headers for API requests."""
        return {
            "Accept": "application/json",
            "Authorization": f"Token {getattr(self.config, 'api_token', getattr(self.config, 'api_key', None))}"
        }

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the API."""
        self.logger.debug(f"Making GET request to {url} with params {params}")
        self.logger.debug(f"Headers: {self._build_headers()}")
        response = self._session.get(url, params=params)
        self.logger.debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        time.sleep(self._sleep)  # Rate limiting
        return response.json()

    def fetch_resource(self, resource_type: str, params: dict = None, limit: int = None) -> list[dict]:
        """Fetch any resource from CourtListener by type, with optional result limit."""
        endpoint = self.endpoints.get(resource_type)
        if not endpoint:
            raise ValueError(f"Unknown resource type: {resource_type}")
        url = endpoint
        results = []
        while url:
            data = self._get(url, params)
            batch = data.get("results", [])
            if limit is not None and len(results) + len(batch) > limit:
                results.extend(batch[:limit - len(results)])
                break
            results.extend(batch)
            if limit is not None and len(results) >= limit:
                break
            url = data.get("next")
            params = None  # Only use params on first request
        return results

def process_and_save(
    client: CourtListenerClient,
    resource_type: str,
    params: dict,
    output_dir: Path,
    limit: int = 10
):
    """Fetch resource and save results to output_dir."""
    results = client.fetch_resource(resource_type, params, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(results):
        with open(output_dir / f"{resource_type}_{i}.json", "w") as f:
            json.dump(item, f, indent=2)
    logger.info(f"Saved {len(results)} {resource_type} to {output_dir}")

def process_statutes(
    statutes: List[str],
    config: APIConfig,
    pages: int = 1,
    page_size: int = 50,
    date_min: Optional[str] = None,
    output_dir: Optional[str] = None,
    api_mode: str = "standard",
    company_file: Optional[str] = None,
    chunk_size: int = 10,
) -> None:
    """Process a list of statutes and save the results.
    
    Args:
        statutes: List of statute names to process
        config: API configuration containing token and settings
        pages: Number of pages to fetch
        page_size: Number of results per page
        date_min: Minimum date to fetch from (YYYY-MM-DD)
        output_dir: Directory to save results to
        api_mode: API mode to use ("standard" or "recap")
        company_file: Optional path to CSV file with company names to filter by
        chunk_size: Number of companies per query chunk (default 50)
    """
    import math
    import csv
    from itertools import islice

    client = CourtListenerClient(config, api_mode=api_mode)

    for statute in statutes:
        base_query = STATUTE_QUERIES[statute].strip()
        logger.info(f"Base query for {statute}: {base_query}")

        if company_file:
            # Read all company names
            names = []
            with open(company_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    names.append(row["official_name"].strip())
            # Chunk the names
            total_chunks = math.ceil(len(names) / chunk_size)
            for chunk_idx in range(total_chunks):
                chunk = names[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                company_filter = "(" + " OR ".join(f'"{n}"' for n in sorted(chunk)) + ")"
                query = base_query + "\nAND\n" + company_filter
                logger.info(f"Searching {statute} (chunk {chunk_idx+1}/{total_chunks}): {query}")
                opinions = client.fetch_resource("opinions", {
                    "search": query,
                    "date_filed_min": date_min
                })
                logger.info(f"Retrieved {len(opinions)} opinions for chunk {chunk_idx+1}/{total_chunks}")
                # Save results in subdirectory or with chunk info
                if output_dir is None:
                    chunk_output_dir = Path("data") / "raw" / "courtlistener" / slugify(statute) / f"chunk_{chunk_idx+1}"
                else:
                    chunk_output_dir = Path(output_dir) / slugify(statute) / f"chunk_{chunk_idx+1}"
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
                for opinion in opinions:
                    metadata_path = chunk_output_dir / f"opinion_{opinion['id']}_metadata.json"
                    with open(metadata_path, "w") as f:
                        json.dump(opinion, f, indent=2)
                    if opinion.get("plain_text"):
                        text_path = chunk_output_dir / f"opinion_{opinion['id']}_text.txt"
                        with open(text_path, "w") as f:
                            f.write(opinion["plain_text"])
                logger.info(f"Saved {len(opinions)} opinions to {chunk_output_dir}")
        else:
            query = base_query
            logger.info(f"Searching {statute}: {query}")
            opinions = client.fetch_resource("opinions", {
                "search": query,
                "date_filed_min": date_min
            })
            logger.info(f"Retrieved {len(opinions)} opinions")
            if output_dir is None:
                output_dir_final = Path("data") / "raw" / "courtlistener" / slugify(statute)
            else:
                output_dir_final = Path(output_dir) / slugify(statute)
            output_dir_final.mkdir(parents=True, exist_ok=True)
            for opinion in opinions:
                metadata_path = output_dir_final / f"opinion_{opinion['id']}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(opinion, f, indent=2)
                if opinion.get("plain_text"):
                    text_path = output_dir_final / f"opinion_{opinion['id']}_text.txt"
                    with open(text_path, "w") as f:
                        f.write(opinion["plain_text"])
            logger.info(f"Saved {len(opinions)} opinions to {output_dir_final}")

    # NOTE: Company name chunking is required to avoid exceeding the CourtListener server's URL length limit (8,190 bytes). Each chunk is queried separately to ensure all company names are included without hitting the server's 413 error.

def process_recap_data(
    config: APIConfig,
    query: str = None,
    docket_id: int = None,
    pages: int = 1,
    page_size: int = 50,
    output_dir: Optional[str] = None,
) -> None:
    """Process RECAP data and save the results.
    
    Args:
        config: API configuration containing token and settings
        query: Optional search query string
        docket_id: Optional specific docket ID to fetch
        pages: Number of pages to fetch
        page_size: Number of results per page
        output_dir: Directory to save results to
    """
    client = CourtListenerClient(config, api_mode="recap")
    
    logger.info(f"Fetching RECAP data with query: {query or 'all'}")
    
    # Fetch RECAP data
    recap_data = client.fetch_resource("recap", {
        "page_size": page_size
    })
    
    logger.info(f"Retrieved {len(recap_data)} RECAP records")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path("data") / "raw" / "courtlistener" / "recap"
    else:
        output_dir = Path(output_dir) / "recap"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save RECAP data
    for i, record in enumerate(recap_data):
        # Save metadata
        metadata_path = output_dir / f"recap_{i}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(record, f, indent=2)
    
    logger.info(f"Saved {len(recap_data)} RECAP records to {output_dir}")

def process_docket_entries(
    config: APIConfig,
    docket_id: int = None,
    query: str = None,
    order_by: str = "-date_filed",
    pages: int = 1,
    page_size: int = 20,
    output_dir: Optional[str] = None,
    api_mode: str = "standard"
) -> None:
    """Process docket entries and save the results.
    
    Args:
        config: API configuration containing token and settings
        docket_id: Specific docket ID to fetch entries for
        query: Optional search query string
        order_by: Field to order by
        pages: Number of pages to fetch
        page_size: Number of results per page
        output_dir: Directory to save results to
        api_mode: API mode to use
    """
    client = CourtListenerClient(config, api_mode=api_mode)
    
    logger.info(f"Fetching docket entries with docket_id: {docket_id}, query: {query or 'all'}")
    
    # Fetch docket entries
    entries = client.fetch_resource("docket_entries", {
        "docket": docket_id,
        "order_by": order_by,
        "page_size": page_size
    })
    
    logger.info(f"Retrieved {len(entries)} docket entries")
    
    # Create output directory
    if output_dir is None:
        if docket_id:
            output_dir = Path("data") / "raw" / "courtlistener" / "docket_entries" / f"docket_{docket_id}"
        else:
            output_dir = Path("data") / "raw" / "courtlistener" / "docket_entries" / "search"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save docket entries
    for i, entry in enumerate(entries):
        # Save entry metadata
        entry_path = output_dir / f"entry_{entry.get('id', i)}_metadata.json"
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)
        
        # Save nested RECAP documents if present
        if entry.get("recap_documents"):
            docs_dir = output_dir / f"entry_{entry.get('id', i)}_documents"
            docs_dir.mkdir(exist_ok=True)
            
            for j, doc in enumerate(entry["recap_documents"]):
                # Save document metadata
                doc_meta_path = docs_dir / f"doc_{doc.get('id', j)}_metadata.json"
                with open(doc_meta_path, "w") as f:
                    json.dump(doc, f, indent=2)
                
                # Save plain text if available
                if doc.get("plain_text"):
                    doc_text_path = docs_dir / f"doc_{doc.get('id', j)}_text.txt"
                    with open(doc_text_path, "w") as f:
                        f.write(doc["plain_text"])
    
    logger.info(f"Saved {len(entries)} docket entries to {output_dir}")

def process_recap_documents(
    config: APIConfig,
    docket_id: int = None,
    docket_entry_id: int = None,
    query: str = None,
    order_by: str = "-date_created",
    pages: int = 1,
    page_size: int = 100,
    include_plain_text: bool = True,
    output_dir: Optional[str] = None,
    api_mode: str = "standard"
) -> None:
    """Process RECAP documents with full text content and save the results.
    
    Args:
        config: API configuration containing token and settings
        docket_id: Docket ID to fetch all documents for
        docket_entry_id: Specific docket entry ID to fetch documents for
        query: Optional search query string
        order_by: Field to order by
        pages: Number of pages to fetch
        page_size: Number of results per page
        include_plain_text: Whether to include plain text content
        output_dir: Directory to save results to
        api_mode: API mode to use
    """
    client = CourtListenerClient(config, api_mode=api_mode)
    
    logger.info(f"Fetching RECAP documents with docket_id: {docket_id}, entry_id: {docket_entry_id}, query: {query or 'all'}")
    
    # Fetch RECAP documents
    documents = client.fetch_resource("recap_docs", {
        "docket_entry": docket_entry_id,
        "docket_entry__docket": docket_id,
        "search": query,
        "order_by": order_by,
        "page_size": page_size
    })
    
    logger.info(f"Retrieved {len(documents)} RECAP documents")
    
    # Create output directory
    if output_dir is None:
        if docket_id:
            output_dir = Path("data") / "raw" / "courtlistener" / "recap_documents" / f"docket_{docket_id}"
        elif docket_entry_id:
            output_dir = Path("data") / "raw" / "courtlistener" / "recap_documents" / f"entry_{docket_entry_id}"
        else:
            output_dir = Path("data") / "raw" / "courtlistener" / "recap_documents" / "search"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save documents
    for i, doc in enumerate(documents):
        # Save document metadata
        doc_meta_path = output_dir / f"doc_{doc.get('id', i)}_metadata.json"
        with open(doc_meta_path, "w") as f:
            json.dump(doc, f, indent=2)
        
        # Save plain text if available and requested
        if include_plain_text and doc.get("plain_text"):
            doc_text_path = output_dir / f"doc_{doc.get('id', i)}_text.txt"
            with open(doc_text_path, "w") as f:
                f.write(doc["plain_text"])
    
    logger.info(f"Saved {len(documents)} RECAP documents to {output_dir}")

def process_full_docket(
    config: APIConfig,
    docket_id: int,
    include_documents: bool = True,
    order_by: str = "-date_filed",
    output_dir: Optional[str] = None,
    api_mode: str = "standard"
) -> None:
    """Process a complete docket with all entries and documents.
    
    Args:
        config: API configuration containing token and settings
        docket_id: The docket ID to fetch
        include_documents: Whether to include full document text
        order_by: How to order docket entries
        output_dir: Directory to save results to
        api_mode: API mode to use
    """
    client = CourtListenerClient(config, api_mode=api_mode)
    
    logger.info(f"Fetching complete docket {docket_id} with documents: {include_documents}")
    
    # Fetch complete docket
    docket_data = client.fetch_resource("dockets", {
        "docket": docket_id
    })
    
    logger.info(f"Retrieved docket with {len(docket_data)} entries and {len(docket_data[0].get('documents', []))} documents")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path("data") / "raw" / "courtlistener" / "full_dockets" / f"docket_{docket_id}"
    else:
        output_dir = Path(output_dir) / f"docket_{docket_id}"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save docket info
    docket_path = output_dir / "docket_info.json"
    with open(docket_path, "w") as f:
        json.dump(docket_data[0], f, indent=2)
    
    # Save entries
    entries_dir = output_dir / "entries"
    entries_dir.mkdir(exist_ok=True)
    
    for entry in docket_data:
        entry_path = entries_dir / f"entry_{entry.get('id')}_metadata.json"
        with open(entry_path, "w") as f:
            json.dump(entry, f, indent=2)
    
    # Save documents if included
    if include_documents and docket_data[0].get("documents"):
        docs_dir = output_dir / "documents"
        docs_dir.mkdir(exist_ok=True)
        
        for doc in docket_data[0]["documents"]:
            # Save document metadata
            doc_meta_path = docs_dir / f"doc_{doc.get('id')}_metadata.json"
            with open(doc_meta_path, "w") as f:
                json.dump(doc, f, indent=2)
            
            # Save plain text if available
            if doc.get("plain_text"):
                doc_text_path = docs_dir / f"doc_{doc.get('id')}_text.txt"
                with open(doc_text_path, "w") as f:
                    f.write(doc["plain_text"])
    
    # Save summary
    summary = {
        "docket_id": docket_id,
        "entry_count": len(docket_data),
        "document_count": len(docket_data[0].get("documents", [])),
        "include_documents": include_documents,
        "order_by": order_by
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved complete docket {docket_id} to {output_dir}")

def process_search_api(
    config,
    params,
    output_dir=None,
    limit=None,
    show_url=False,
    token=None
):
    """Process a direct search API query and print or save results."""
    import json
    import httpx
    from urllib.parse import urlencode
    from pathlib import Path

    base_url = "https://www.courtlistener.com/api/rest/v4/search/"
    url = base_url + ("?" + urlencode(params) if params else "")

    # Use token from config if not provided
    if not token:
        token = getattr(config, "api_token", None)
    headers = {"Authorization": f"Token {token}"} if token else {}

    if show_url:
        print(f"API URL: {url}")
        return

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Optionally limit results
    if limit is not None and "results" in data:
        data["results"] = data["results"][:limit]

    # Print or save
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outpath = Path(output_dir) / "search_api_results.json"
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {outpath}")
    else:
        print(json.dumps(data, indent=2))

def process_recap_fetch(
    config,
    post_data,
    show_url=False,
    token=None
):
    """POST to /api/rest/v4/recap-fetch/ to trigger a PACER fetch. Allows safe, credentialed, free attachment fetch (type=3)."""
    import json
    import httpx
    base_url = "https://www.courtlistener.com/api/rest/v4/recap-fetch/"
    url = base_url
    # Use token from config if not provided
    if not token:
        token = getattr(config, "api_token", None)
    headers = {"Authorization": f"Token {token}"} if token else {}
    if show_url:
        print(f"POST URL: {url}")
        print(f"POST data: {post_data}")
        return
    # Only allow real PACER credentials for request_type=3 (free attachment pages)
    if str(post_data.get("request_type")) != "3" and (
        "pacer_username" in post_data or "pacer_password" in post_data
    ):
        print("[TEST MODE] Not sending real PACER credentials or purchase request except for request_type=3 (free attachment pages).")
        return
    if str(post_data.get("request_type")) == "3" and ("pacer_username" in post_data and "pacer_password" in post_data):
        print("[WARNING] You are sending PACER credentials to fetch free attachment pages. This will NOT purchase anything, but your credentials are required for authentication. They are not stored by CourtListener. Proceeding...")
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, data=post_data, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"Error: {e}")
        raise
    print(json.dumps(data, indent=2)) 