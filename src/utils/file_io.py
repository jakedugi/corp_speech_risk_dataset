"""File I/O utilities for the corporate speech risk dataset."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save to
        indent: JSON indentation level
    """
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.debug(f"Saved JSON to {path}")

def load_json(path: Path) -> Any:
    """Load data from a JSON file.
    
    Args:
        path: Path to load from
        
    Returns:
        Loaded data
    """
    with open(path) as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON from {path}")
    return data

def list_json_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    """List all JSON files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of matching file paths
    """
    return sorted(directory.glob(pattern))

def merge_json_files(files: List[Path], output_path: Path) -> None:
    """Merge multiple JSON files into one.
    
    Args:
        files: List of JSON files to merge
        output_path: Path to save merged data
    """
    merged_data = []
    for file in files:
        data = load_json(file)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)
    
    save_json(merged_data, output_path)
    logger.info(f"Merged {len(files)} files into {output_path}")

