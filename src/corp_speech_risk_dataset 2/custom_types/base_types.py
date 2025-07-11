from typing import Optional

class APIConfig:
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 0.25):
        self.api_key = api_key
        self.rate_limit = rate_limit
