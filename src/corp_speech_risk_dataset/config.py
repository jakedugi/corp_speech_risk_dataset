"""Configuration management for the CourtListener API client.

This module handles loading configuration from environment variables and config files,
with proper fallbacks and validation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

@dataclass
class CourtListenerConfig:
    """Configuration settings for CourtListener API."""
    
    # API settings
    api_token: Optional[str] = None
    rate_limit: float = 0.25  # seconds between API requests
    
    # Output settings
    output_dir: Path = Path("data/raw/courtlistener")
    
    # Default processing settings
    default_pages: int = 1
    default_page_size: int = 50
    default_date_min: Optional[str] = None
    api_mode: str = "standard"

def load_config() -> CourtListenerConfig:
    """Load configuration with proper fallbacks.
    
    Priority:
    1. Environment variables
    2. .env file
    3. Default values
    
    Returns:
        CourtListenerConfig: Loaded configuration
    """
    try:
        # Load from environment variables
        api_token = os.getenv("COURTLISTENER_API_TOKEN")
        rate_limit = float(os.getenv("COURTLISTENER_RATE_LIMIT", "0.25"))
        output_dir = Path(os.getenv("COURTLISTENER_OUTPUT_DIR", "data/raw/courtlistener"))
        default_pages = int(os.getenv("COURTLISTENER_DEFAULT_PAGES", "1"))
        default_page_size = int(os.getenv("COURTLISTENER_DEFAULT_PAGE_SIZE", "50"))
        default_date_min = os.getenv("COURTLISTENER_DEFAULT_DATE_MIN")
        api_mode = os.getenv("COURTLISTENER_API_MODE", "standard")
        
        config = CourtListenerConfig(
            api_token=api_token,
            rate_limit=rate_limit,
            output_dir=output_dir,
            default_pages=default_pages,
            default_page_size=default_page_size,
            default_date_min=default_date_min,
            api_mode=api_mode
        )
        
        # Validate API token
        if not config.api_token:
            logger.warning(
                "No API token found. Set COURTLISTENER_API_TOKEN environment variable "
                "or create a .env file with COURTLISTENER_API_TOKEN=your_token"
            )
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise 