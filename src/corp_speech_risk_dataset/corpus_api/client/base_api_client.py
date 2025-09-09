"""Base API client for making HTTP requests."""

from typing import Any, Dict, Optional
from corp_speech_risk_dataset.types.schemas.models import APIConfig


class BaseAPIClient:
    """Base class for all API clients."""

    def __init__(self, config: APIConfig):
        self.config = config

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Token {self.config.api_key}"
        return headers
