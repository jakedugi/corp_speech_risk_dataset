"""
CourtListener HTTP/JSON client (modularized).

This module will contain only the low-level HTTP/JSON client logic for CourtListener.
"""

from __future__ import annotations

import httpx
from loguru import logger
import time
from typing import Any, Dict, List, Optional
import asyncio
import random
from asyncio import Semaphore
from httpx import AsyncClient, Limits, AsyncHTTPTransport

from corp_speech_risk_dataset.api.base_api_client import BaseAPIClient
from corp_speech_risk_dataset.custom_types.base_types import APIConfig
from corp_speech_risk_dataset.utils.http_utils import safe_sync_get, safe_async_get

# API endpoint configurations
API_ENDPOINTS = {
    "standard": {
        "base_url": "https://www.courtlistener.com/api/rest/v4",
        "dockets": "https://www.courtlistener.com/api/rest/v4/dockets/",
        "clusters": "https://www.courtlistener.com/api/rest/v4/clusters/",
        "opinions": "https://www.courtlistener.com/api/rest/v4/opinions/",
        "docket_entries": "https://www.courtlistener.com/api/rest/v4/docket-entries/",
        "recap_docs": "https://www.courtlistener.com/api/rest/v4/recap-documents/",
    },
    "recap": {
        "base_url": "https://www.courtlistener.com/api/rest/v4",
        "dockets": "https://www.courtlistener.com/api/rest/v4/dockets/",
        "clusters": "https://www.courtlistener.com/api/rest/v4/clusters/",
        "opinions": "https://www.courtlistener.com/api/rest/v4/opinions/",
        "docket_entries": "https://www.courtlistener.com/api/rest/v4/docket-entries/",
        "recap_docs": "https://www.courtlistener.com/api/rest/v4/recap-documents/",
        "recap": "https://www.courtlistener.com/api/rest/v4/recap/",
    }
}


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
            timeout=httpx.Timeout(
                connect=5.0,    # still quick to connect
                read=120.0,     # give slower endpoints up to 2 minutes
                write=5.0,
                pool=5.0
            )
        )
        self._sleep = config.rate_limit or 3.0
        
    def _build_headers(self) -> Dict[str, str]:
        """Build headers for API requests."""
        return {
            "Accept": "application/json",
            "Authorization": f"Token {getattr(self.config, 'api_token', getattr(self.config, 'api_key', None))}"
        }

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = safe_sync_get(
            session=self._session,
            url=url,
            params=params,
            max_attempts=5,
            rate_limit=self._sleep,
        )
        return result or {"results": [], "next": None}

    def fetch_resource(self, resource_type: str, params: dict = None, limit: int = None) -> list[dict]:
        """
        Fetch any resource from CourtListener by type, with optional result limit.
        Uses API's next-link for pagination; never manually increments page numbers.
        Only the first request uses params; subsequent requests follow the 'next' URL.
        If a 404 is encountered (e.g., due to deep pagination), stops gracefully.
        """
        endpoint = self.endpoints.get(resource_type)
        if not endpoint:
            raise ValueError(f"Unknown resource type: {resource_type}")
        url = endpoint
        results = []
        while url:
            try:
                data = self._get(url, params)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    self.logger.warning(f"404 Not Found for {url}; stopping pagination loop.")
                    break
                raise
            batch = data.get("results", [])
            if limit is not None and len(results) + len(batch) > limit:
                results.extend(batch[:limit - len(results)])
                break
            results.extend(batch)
            if limit is not None and len(results) >= limit:
                break
            url = data.get("next")
            params = None  # Only use params on first request; never set page manually
        return results 

class AsyncCourtListenerClient:
    """
    Asynchronous CourtListener API client with bounded concurrency, robust retry/backoff,
    and respect for server rate limits and Retry-After headers.
    Uses HTTPX's built-in AsyncHTTPTransport for automatic retries.
    """
    def __init__(self, token: str, max_concurrency: int = 2, rate_limit: float = 3.0):
        # Use HTTPX built-in AsyncHTTPTransport with retries
        transport = AsyncHTTPTransport(retries=5)
        self.client = AsyncClient(
            transport=transport,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
            limits=Limits(max_connections=max_concurrency),
            timeout=60.0,
        )
        self.sem = asyncio.Semaphore(max_concurrency)
        self.rate_limit = rate_limit

    async def _get(self, url, params=None, retries=5):
        data = await safe_async_get(
            client=self.client,
            url=url,
            params=params,
            max_attempts=retries,
            rate_limit=self.rate_limit,
            semaphore=self.sem,
        )
        return data or {}

    async def fetch_docs(self, doc_uris: list[str]):
        """
        Fire off all doc GETs in parallel, return list of results or exceptions.
        """
        tasks = [self._get(uri) for uri in doc_uris]
        return await asyncio.gather(*tasks, return_exceptions=True) 