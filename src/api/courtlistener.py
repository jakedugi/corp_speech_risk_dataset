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

from src.api.base_api_client import BaseAPIClient
from src.custom_types.base_types import APIConfig

# API endpoints
BASE_DOCKETS_URL = "https://www.courtlistener.com/api/rest/v4/dockets/"
BASE_CLUSTERS_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
BASE_OPINIONS_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
BASE_DOCKET_ENTRIES_URL = "https://www.courtlistener.com/api/rest/v4/docket-entries/"
BASE_RECAP_DOCS_URL = "https://www.courtlistener.com/api/rest/v4/recap-documents/"

# Statute queries
STATUTE_QUERIES: Dict[str, str] = {
    "FTC Section 5": '''(
        ("FTC Act" OR "Section 5" OR "15 U.S.C. § 45" OR "unfair methods of competition")
        AND
        (
            tweet OR "Twitter post" OR "X post" OR Facebook OR Instagram OR website OR blog
            OR TikTok OR YouTube OR LinkedIn OR "social media" OR "online advertising"
        )
        AND
        (
            deceptive OR misleading OR "unfair practice" OR fraudulent OR "false claim"
            OR "unfair methods"
        )
        AND
        (
            "corporate speech" OR "press release" OR "company statement" OR "internal memo"
            OR "marketing materials" OR "executive statement" OR advertisement OR promotion OR claim
        )
    )''',
    "FTC Section 12": '(("Section 12" OR "15 USC 52" OR "15 U.S.C. § 52") AND (Instagram OR TikTok OR YouTube OR tweet OR website OR "landing page") AND (efficacy OR health OR "performance claim"))',
    "Lanham Act § 43(a)": '(("Lanham Act" OR "Section 43(a)" OR "15 USC 1125(a)" OR "15 U.S.C. § 1125(a)") AND (Twitter OR "X post" OR influencer OR hashtag OR TikTok OR "sponsored post" OR website) AND ("false advertising" OR "false endorsement" OR misrepresentation))',
    "SEC Rule 10b-5": '(("10b-5" OR "Rule 10b-5" OR "17 CFR 240.10b-5") AND ("press release" OR tweet OR blog OR "CEO post" OR Reddit OR Discord) AND ("material misstatement" OR fraud OR "stock price" OR "market manipulation"))',
    "SEC Regulation FD": '(("Regulation FD" OR "Reg FD" OR "17 CFR 243.100" OR "Rule FD") AND (CEO OR CFO OR executive) AND ("Facebook post" OR tweet OR "LinkedIn post" OR webcast OR blog) AND ("material information" OR disclosure OR "selective disclosure"))',
    "NLRA § 8(a)(1)": '(("Section 8(a)(1)" OR "29 USC 158(a)(1)" OR "29 U.S.C. § 158(a)(1)") AND (tweet OR "X post" OR Facebook OR website OR memo OR blog) AND (union OR unionize OR collective-bargaining OR organizing) AND (threat OR promise OR coercive))',
    "CFPA UDAAP": '((UDAAP OR "12 USC 5531" OR "12 U.S.C. § 5531" OR "Consumer Financial Protection Act") AND (loan OR fintech OR credit OR "BNPL") AND (website OR app OR tweet OR "Instagram ad" OR webinar) AND (deceptive OR misleading OR unfair))',
    "California § 17200 / 17500": '(("Business and Professions Code § 17200" OR "Bus & Prof Code 17200" OR "§ 17500") AND (site OR newsletter OR email OR Instagram OR TikTok OR tweet) AND (claim OR representation OR advertisement) AND (misleading OR deceptive OR untrue))',
    "NY GBL §§ 349–350": '(("GBL § 349" OR "General Business Law § 349" OR "GBL § 350") AND (webinar OR "landing page" OR infomercial OR website OR tweet OR Facebook) AND (misleading OR deceptive OR fraud OR "false advertising"))',
    "FD&C Act § 331": '(("21 USC 331" OR "21 U.S.C. § 331" OR "FD&C Act") AND (marketing OR promo OR blog OR website OR Facebook OR tweet OR "YouTube video") AND (misbranding OR "risk disclosure" OR "omitted risk"))'
}

def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

class CourtListenerClient(BaseAPIClient):
    """Client for interacting with the CourtListener API."""

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    BASE_OPINIONS_URL = f"{BASE_URL}/opinions/"

    def __init__(self, config: APIConfig):
        """Initialize the client with API configuration."""
        super().__init__(config)
        self.logger = logger.bind(client="courtlistener")
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
            "Authorization": f"Token {self.config.api_token}"
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

    def fetch_dockets(
        self,
        query: str,
        pages: int = 2,
        page_size: int = 100,
        jurisdiction: str = "F",
        date_filed_min: str = "2019-01-01"
    ) -> List[Dict[str, Any]]:
        """Fetch dockets matching the search query."""
        params = {
            "free_text": query,
            "court__jurisdiction": jurisdiction,
            "date_filed_min": date_filed_min,
            "page_size": page_size
        }
        
        url = BASE_DOCKETS_URL
        all_results: List[Dict[str, Any]] = []
        
        for _ in range(pages):
            data = self._get(url, params)
            all_results.extend(data.get("results", []))
            
            if not data.get("next"):
                break
            url, params = data["next"], None
            
        return all_results
        
    def fetch_opinions(
        self,
        query: str,
        pages: int = 2,
        page_size: int = 100,
        jurisdiction: str = "F",
        date_filed_min: str = "2019-01-01"
    ) -> List[Dict[str, Any]]:
        """Fetch opinions matching the search query.
        
        Args:
            query: Search query string
            pages: Number of pages to fetch
            page_size: Results per page
            jurisdiction: Court jurisdiction (default: Federal)
            date_filed_min: Minimum filing date (YYYY-MM-DD)
        """
        url = self.BASE_OPINIONS_URL
        params = {
            "search": query,
            "court__jurisdiction": jurisdiction,
            "date_filed_min": date_filed_min,
            "page_size": page_size,
            "ordering": "-date_filed",  # Newest first
            "highlight": "off",  # Enable highlighting
            "fields": "id,caseName,dateFiled,snippet,plain_text,html,html_lawbox,cluster,citations,court,judges,type,absolute_url"  # Include all relevant fields
        }
        
        all_opinions = []
        for _ in range(pages):
            data = self._get(url, params)
            opinions = data.get("results", [])
            all_opinions.extend(opinions)
            
            if not data.get("next"):
                break
                
            url = data["next"]
            params = None  # Next URL already includes params
            
        return all_opinions

    def fetch_all_docket_entries(
        self, docket_id: int, page_size: int = 1000
    ) -> list[dict]:
        """
        Fetch *every* docket entry for a docket, including nested recap_documents
        (each of which has plain_text).
        """
        url = BASE_DOCKET_ENTRIES_URL
        params = {"docket": docket_id, "page_size": page_size}
        all_entries: list[dict] = []

        while True:
            data = self._get(url, params)
            all_entries.extend(data.get("results", []))
            nxt = data.get("next")
            if not nxt:
                break
            url, params = nxt, None

        return all_entries

    def fetch_all_recap_documents(self, docket_id: int, page_size: int = 1000) -> list[dict]:
        """Fetch all RECAP documents for a docket, including their plain text."""
        url = BASE_RECAP_DOCS_URL
        params = {"docket_entry__docket": docket_id, "page_size": page_size}
        docs: list[dict] = []
        
        while True:
            data = self._get(url, params)
            docs.extend(data.get("results", []))
            nxt = data.get("next")
            if not nxt:
                break
            url, params = nxt, None
            
        return docs

def process_statutes(
    statutes: List[str],
    config: APIConfig,
    pages: int = 1,
    page_size: int = 50,
    date_min: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Process a list of statutes and save the results.
    
    Args:
        statutes: List of statute names to process
        config: API configuration containing token and settings
        pages: Number of pages to fetch
        page_size: Number of results per page
        date_min: Minimum date to fetch from (YYYY-MM-DD)
        output_dir: Directory to save results to
    """
    client = CourtListenerClient(config)
    
    for statute in statutes:
        logger.info(f"Searching {statute}: {STATUTE_QUERIES[statute]}")
        
        # Fetch opinions using the new method
        opinions = client.fetch_opinions(
            query=STATUTE_QUERIES[statute],
            pages=pages,
            page_size=page_size,
            date_filed_min=date_min
        )
        
        logger.info(f"Retrieved {len(opinions)} opinions")
        
        # Create output directory with correct structure
        if output_dir is None:
            output_dir = Path("data") / "raw" / "courtlistener" / slugify(statute)
        else:
            output_dir = Path(output_dir) / slugify(statute)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each opinion
        for opinion in opinions:
            # Save metadata
            metadata_path = output_dir / f"opinion_{opinion['id']}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(opinion, f, indent=2)
            
            # Save plain text if available
            if opinion.get("plain_text"):
                text_path = output_dir / f"opinion_{opinion['id']}_text.txt"
                with open(text_path, "w") as f:
                    f.write(opinion["plain_text"])
        
        logger.info(f"Saved {len(opinions)} opinions to {output_dir}") 