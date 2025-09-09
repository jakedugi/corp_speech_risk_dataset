"""File I/O utilities for the corporate speech risk dataset."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import time
import random

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
    with open(path, "w") as f:
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


# --- New helpers for legacy workflow support ---
def download(url: str, path: Path, max_attempts: int = 3, sleep: float = 0.5) -> None:
    """
    Download a file from a URL to a local path with retry and logging.
    Follows redirects automatically.
    For /recap/ URLs, only try once (no retry on 429).
    Args:
        url: The URL to download from
        path: The local file path to save to
        max_attempts: Number of retry attempts (ignored for /recap/)
        sleep: Seconds to wait between retries
    """
    ensure_dir(path.parent)
    effective_retries = 1 if "/recap/" in url else max_attempts
    for attempt in range(1, effective_retries + 1):
        try:
            with httpx.stream("GET", url, timeout=30.0, follow_redirects=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            logger.info(f"Downloaded {url} to {path}")
            return
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            # Treat any 4xx/5xx as skip errors
            if 400 <= code < 600:
                logger.warning(f"HTTP {code} for {url}; skipping download.")
                return
            raise
        except Exception as e:
            logger.warning(
                f"[{attempt}/{effective_retries}] Failed to download {url}: {e}"
            )
            time.sleep(sleep * attempt)
    if effective_retries == 1:
        logger.warning(f"Giving up on {url} after single attempt (recap PDF policy)")
    else:
        raise RuntimeError(
            f"Failed to download {url} after {effective_retries} attempts"
        )


def needs_recap_fetch(ia_json_path: Path) -> bool:
    """
    Returns True if any entry in the IA dump has empty recap_documents or missing description.
    Args:
        ia_json_path: Path to the IA JSON dump
    Returns:
        True if RECAP fetch is needed, False otherwise
    """
    if not ia_json_path.exists():
        logger.warning(f"IA JSON not found: {ia_json_path}")
        return False
    try:
        data = load_json(ia_json_path)
        entries = data.get("entries") or data.get("docket_entries") or []
        for entry in entries:
            if not entry.get("recap_documents") or not entry.get("description"):
                return True
        return False
    except Exception as e:
        logger.error(f"Error reading {ia_json_path}: {e}")
        return False


def download_missing_pdfs(entry_json: Path, filings_dir: Path) -> None:
    """
    Download PDFs for recap_documents in an entry if plain_text is missing.
    Args:
        entry_json: Path to the entry JSON file
        filings_dir: Directory to save PDFs to
    """
    ensure_dir(filings_dir)
    try:
        entry = load_json(entry_json)
        docs = entry.get("recap_documents") or []
        for doc in docs:
            pdf_url = doc.get("filepath_local")
            doc_id = doc.get("id")
            has_text = bool(doc.get("plain_text"))
            if pdf_url and not has_text:
                pdf_path = filings_dir / f"doc_{doc_id}.pdf"
                if not pdf_path.exists():
                    try:
                        download(pdf_url, pdf_path)
                    except Exception as e:
                        logger.warning(f"Failed to download PDF for doc {doc_id}: {e}")
                else:
                    logger.debug(f"PDF already exists: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing {entry_json}: {e}")
