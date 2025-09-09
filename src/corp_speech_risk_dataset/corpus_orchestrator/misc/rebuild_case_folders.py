#!/usr/bin/env python3
"""
Rebuild Case Folder Structure

This script reorganizes the flat directory of tokenized stage8 .jsonl files
into case-based subfolders based on the case names extracted from the _src field.

Usage: python scripts/rebuild_case_folders.py

Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_case_name_from_src(src_path: str) -> Optional[str]:
    """
    Extract case name from the _src field.

    Example:
    "results/courtlistener_v11/14-2274_ca6/entries/entry_783690_documents/doc_639746_text.txt"
    Returns: "14-2274_ca6"

    Args:
        src_path: The source path from the _src field

    Returns:
        Case name if found, None otherwise
    """
    try:
        # Split by '/' and find the case name after 'courtlistener_v11'
        path_parts = src_path.split("/")

        # Find the index of the courtlistener version directory
        courtlistener_idx = None
        for i, part in enumerate(path_parts):
            if part.startswith("courtlistener_v"):
                courtlistener_idx = i
                break

        if courtlistener_idx is not None and courtlistener_idx + 1 < len(path_parts):
            case_name = path_parts[courtlistener_idx + 1]
            return case_name

        logger.warning(f"Could not extract case name from: {src_path}")
        return None

    except Exception as e:
        logger.error(f"Error extracting case name from {src_path}: {e}")
        return None


def get_case_name_from_file(file_path: Path) -> Optional[str]:
    """
    Read the first line of a .jsonl file and extract the case name from _src field.

    Args:
        file_path: Path to the .jsonl file

    Returns:
        Case name if successfully extracted, None otherwise
    """
    try:
        # Skip empty files
        if file_path.stat().st_size == 0:
            logger.warning(f"Skipping empty file: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if not first_line:
            logger.warning(f"Empty first line in file: {file_path}")
            return None

        # Parse JSON from first line
        data = json.loads(first_line)
        src_path = data.get("_src")

        if not src_path:
            logger.warning(f"No _src field found in: {file_path}")
            return None

        return extract_case_name_from_src(src_path)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def reorganize_files(source_dir: str, dry_run: bool = False) -> None:
    """
    Reorganize files from flat structure into case-based folders.

    Args:
        source_dir: Directory containing the flat .jsonl files
        dry_run: If True, only print what would be done without moving files
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return

    # Find all .jsonl files with stage8 in the name
    jsonl_files = list(source_path.glob("*_stage8.jsonl"))

    if not jsonl_files:
        logger.warning(f"No *_stage8.jsonl files found in {source_dir}")
        return

    logger.info(f"Found {len(jsonl_files)} stage8 files to process")

    # Track statistics
    case_counts = {}
    moved_files = 0
    skipped_files = 0
    error_files = 0

    for file_path in jsonl_files:
        logger.debug(f"Processing: {file_path.name}")

        # Extract case name
        case_name = get_case_name_from_file(file_path)

        if not case_name:
            logger.warning(f"Skipping file with no case name: {file_path.name}")
            skipped_files += 1
            continue

        # Track case counts
        case_counts[case_name] = case_counts.get(case_name, 0) + 1

        # Create case directory
        case_dir = source_path / case_name

        if dry_run:
            logger.info(f"[DRY RUN] Would move {file_path.name} -> {case_name}/")
        else:
            try:
                # Create case directory if it doesn't exist
                case_dir.mkdir(exist_ok=True)

                # Move file to case directory
                dest_path = case_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))

                logger.debug(f"Moved {file_path.name} -> {case_name}/")
                moved_files += 1

            except Exception as e:
                logger.error(f"Error moving {file_path.name}: {e}")
                error_files += 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("REORGANIZATION SUMMARY")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN MODE - No files were actually moved")

    logger.info(f"Total files processed: {len(jsonl_files)}")
    logger.info(f"Files moved: {moved_files}")
    logger.info(f"Files skipped: {skipped_files}")
    logger.info(f"Files with errors: {error_files}")
    logger.info(f"Total cases identified: {len(case_counts)}")

    # Show case distribution
    logger.info("\nCASE DISTRIBUTION:")
    for case_name, count in sorted(case_counts.items()):
        logger.info(f"  {case_name}: {count} files")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild case folder structure from flat tokenized files"
    )
    parser.add_argument(
        "--source-dir",
        default="data/tokenized/courtlistener_v5_final",
        help="Source directory containing flat .jsonl files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting reorganization of {args.source_dir}")
    logger.info(f"Dry run mode: {args.dry_run}")

    reorganize_files(args.source_dir, args.dry_run)

    logger.info("Reorganization complete!")


if __name__ == "__main__":
    main()
