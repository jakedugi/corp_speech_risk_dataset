"""SEC API client (placeholder implementation).

This module will contain the SEC API client when implemented.
Currently serves as a placeholder to maintain import compatibility.
"""

from typing import Optional
from loguru import logger

from .base_api_client import BaseAPIClient


class SECClient(BaseAPIClient):
    """Placeholder SEC API client.
    
    TODO: Implement actual SEC API integration when requirements are defined.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize SEC client.
        
        Args:
            api_key: SEC API key (currently unused)
        """
        logger.info("SEC client initialized (placeholder implementation)")
        self.api_key = api_key
    
    def fetch_data(self, **kwargs):
        """Placeholder method for fetching SEC data.
        
        TODO: Implement actual data fetching logic.
        """
        logger.warning("SEC client fetch_data called but not implemented")
        return []
