"""Configuration management for the CourtListener API client.

This module handles loading configuration from environment variables and config files,
with proper fallbacks and validation.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field

class CourtListenerConfig(BaseSettings):
    """Configuration settings for CourtListener API."""
    
    # API settings
    api_token: Optional[str] = Field(
        default=None,
        description="CourtListener API token"
    )
    rate_limit: float = Field(
        default=0.25,  # seconds between API requests
        description="Rate limit in seconds between requests"
    )
    
    # Output settings
    output_dir: Path = Field(
        default=Path("data/raw/courtlistener"),
        description="Base directory for output files"
    )
    
    class Config:
        env_prefix = "COURTLISTENER_"  # Will look for COURTLISTENER_API_TOKEN etc.
        case_sensitive = False
        env_file = ".env"  # Load environment variables from .env file

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
        config = CourtListenerConfig()
        
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