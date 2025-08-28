#!/usr/bin/env python3
"""
Script to add CourtListener provenance fields to the final dataset.

Maps existing dataset entries to their corresponding CourtListener metadata
to populate required provenance fields.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_doc_info_from_path(src_path: str) -> Dict[str, str]:
    """
    Extract case_id, entry_id, and doc_id from the metadata source path.

    Args:
        src_path: Path like "results/courtlistener_v11/1:19-cv-02184_dcd/entries/entry_95995261_documents/doc_100208922_text.txt"

    Returns:
        Dictionary with case_id, entry_id, doc_id
    """
    # Extract case_id (e.g., "1:19-cv-02184_dcd")
    case_match = re.search(r"([0-9]+:[0-9]+-[a-z]+-[0-9]+_[a-z]+)", src_path)
    if not case_match:
        raise ValueError(f"Could not extract case_id from path: {src_path}")
    case_id = case_match.group(1)

    # Extract entry_id (e.g., "95995261")
    entry_match = re.search(r"entry_([0-9]+)_documents", src_path)
    if not entry_match:
        raise ValueError(f"Could not extract entry_id from path: {src_path}")
    entry_id = entry_match.group(1)

    # Extract doc_id (e.g., "100208922")
    doc_match = re.search(r"doc_([0-9]+)_text\.txt", src_path)
    if not doc_match:
        raise ValueError(f"Could not extract doc_id from path: {src_path}")
    doc_id = doc_match.group(1)

    return {"case_id": case_id, "entry_id": entry_id, "doc_id": doc_id}


def load_document_metadata(
    raw_data_dir: Path, case_id: str, entry_id: str, doc_id: str
) -> Optional[Dict]:
    """
    Load document metadata from CourtListener raw data.

    Args:
        raw_data_dir: Path to raw CourtListener data directory
        case_id: Case identifier (e.g., "1:19-cv-02184_dcd")
        entry_id: Entry identifier (e.g., "95995261")
        doc_id: Document identifier (e.g., "100208922")

    Returns:
        Document metadata dictionary or None if not found
    """
    metadata_path = (
        raw_data_dir
        / case_id
        / "entries"
        / f"entry_{entry_id}_documents"
        / f"doc_{doc_id}_metadata.json"
    )

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return None


def extract_docket_id_from_url(absolute_url: str) -> Optional[str]:
    """
    Extract docket_id from absolute_url.

    Args:
        absolute_url: URL like "/docket/15959672/1/2/united-states-v-facebook-inc/"

    Returns:
        Docket ID string or None if not found
    """
    match = re.search(r"/docket/([0-9]+)/", absolute_url)
    if match:
        return match.group(1)
    return None


def create_provenance_fields(doc_metadata: Dict) -> Dict[str, str]:
    """
    Create provenance fields from document metadata.

    Args:
        doc_metadata: Document metadata dictionary from CourtListener

    Returns:
        Dictionary with provenance fields
    """
    # Extract docket_id from absolute_url
    docket_id = extract_docket_id_from_url(doc_metadata.get("absolute_url", ""))

    # Construct source_url
    source_url = f"https://www.courtlistener.com{doc_metadata.get('absolute_url', '')}"

    provenance = {
        "courtlistener_id": str(doc_metadata.get("id", "")),
        "docket_id": docket_id or "",
        "source_url": source_url,
        "doc_type": doc_metadata.get("document_type", ""),
        "retrieved_date": "2025-06-07",  # As specified by user
    }

    return provenance


def process_dataset(input_file: Path, output_file: Path, raw_data_dir: Path):
    """
    Process the dataset to add CourtListener provenance fields.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        raw_data_dir: Path to raw CourtListener data directory
    """
    logger.info(f"Processing dataset from {input_file}")
    logger.info(f"Output will be written to {output_file}")
    logger.info(f"Using raw data from {raw_data_dir}")

    total_entries = 0
    successful_mappings = 0
    failed_mappings = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            if line_num % 1000 == 0:
                logger.info(f"Processed {line_num} entries...")

            try:
                entry = json.loads(line.strip())
                total_entries += 1

                # Extract document info from source path
                src_path = entry.get("_metadata_src_path", "")
                if not src_path:
                    logger.warning(f"Line {line_num}: No _metadata_src_path found")
                    failed_mappings += 1
                    continue

                try:
                    doc_info = extract_doc_info_from_path(src_path)
                except ValueError as e:
                    logger.warning(f"Line {line_num}: {e}")
                    failed_mappings += 1
                    continue

                # Load corresponding metadata
                doc_metadata = load_document_metadata(
                    raw_data_dir,
                    doc_info["case_id"],
                    doc_info["entry_id"],
                    doc_info["doc_id"],
                )

                if doc_metadata is None:
                    logger.warning(
                        f"Line {line_num}: Could not load metadata for {doc_info}"
                    )
                    failed_mappings += 1
                    continue

                # Create provenance fields
                provenance = create_provenance_fields(doc_metadata)

                # Add provenance fields to entry
                entry.update(provenance)

                # Write enhanced entry
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                successful_mappings += 1

            except Exception as e:
                logger.error(f"Line {line_num}: Error processing entry: {e}")
                failed_mappings += 1
                continue

    logger.info(f"Processing complete!")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Successful mappings: {successful_mappings}")
    logger.info(f"Failed mappings: {failed_mappings}")
    logger.info(f"Success rate: {successful_mappings/total_entries*100:.1f}%")


def main():
    """Main function to run the provenance enhancement."""

    # File paths
    input_file = Path(
        "/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/enhanced_combined_FINAL/final_clean_dataset_with_interpretable_features.jsonl"
    )
    output_file = Path(
        "/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/enhanced_combined_FINAL/final_clean_dataset_with_interpretable_features_with_provenance.jsonl"
    )
    raw_data_dir = Path(
        "/Users/jakedugan/Projects/corporate_media_risk/corp_speech_risk_dataset/data/raw/courtlistener"
    )

    # Validate paths
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return

    # Process dataset
    process_dataset(input_file, output_file, raw_data_dir)


if __name__ == "__main__":
    main()
