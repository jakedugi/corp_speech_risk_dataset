"""FTC API client (placeholder implementation).

This module will contain the FTC API client when implemented.
Currently serves as a placeholder to maintain import compatibility.
"""

from typing import Optional
from loguru import logger

from .base_api_client import BaseAPIClient


class FTCClient(BaseAPIClient):
    """Placeholder FTC API client.

    TODO: Implement actual FTC API integration when requirements are defined.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize FTC client.

        Args:
            api_key: FTC API key (currently unused)
        """
        logger.info("FTC client initialized (placeholder implementation)")
        self.api_key = api_key

    def fetch_data(self, **kwargs):
        """Placeholder method for fetching FTC data.

        TODO: Implement actual data fetching logic.
        """
        logger.warning("FTC client fetch_data called but not implemented")
        return []
