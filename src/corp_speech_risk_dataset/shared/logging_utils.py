"""Logging utilities for the corporate speech risk dataset."""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "1 day"
) -> None:
    """Configure logging for the application.

    Args:
        log_file: Optional path to log file
        level: Logging level
        rotation: Log rotation policy
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation
        )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: The name for the logger

    Returns:
        A logger instance
    """
    return logging.getLogger(name)
