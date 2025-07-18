# src/corp_speech_risk_dataset/custom_types/base_types.py

from typing import Optional


class APIConfig:
    """Generic API configuration container."""

    def __init__(self, api_token: Optional[str] = None, rate_limit: float = 0.25):
        self.api_token = api_token
        self.rate_limit = rate_limit
        # alias if code expects api_key
        self.api_key = api_token
