#!/usr/bin/env python3
"""
CLI interface for corpus-cleaner module.

Usage:
    python -m corpus_cleaner.cli.normalize --input docs.jsonl --output docs.norm.jsonl --config normalize.yaml
"""

import json
from pathlib import Path
from typing import Optional
import typer
import logging

from ..cleaner import TextCleaner

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def normalize(
    input_file: Path = typer.Option(..., "--input", help="Input documents JSONL file"),
    output_file: Path = typer.Option(
        ..., "--output", help="Output normalized documents JSONL file"
    ),
    offset_file: Optional[Path] = typer.Option(
        None, "--offsets", help="Output offset mapping JSONL file"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", help="Normalization configuration YAML"
    ),
):
    """
    Normalize text documents and preserve offset mappings.

    Takes raw documents and produces normalized documents with optional offset mappings
    for maintaining span alignments.
    """
    logger.info(f"Normalizing documents from {input_file}")
    logger.info(f"Output: {output_file}")
    if offset_file:
        logger.info(f"Offsets: {offset_file}")

    # Load configuration
    config = {}
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

    # Initialize text cleaner
    cleaner = TextCleaner(config)

    # Process documents
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        offset_mappings = []

        for i, line in enumerate(infile):
            if i % 1000 == 0:
                logger.info(f"Processed {i} documents")

            doc = json.loads(line.strip())

            # Clean the text
            original_text = doc.get("raw_text", "")
            cleaned_text, offset_map = cleaner.normalize_text_with_offsets(
                original_text
            )

            # Update document
            normalized_doc = doc.copy()
            normalized_doc["raw_text"] = cleaned_text
            normalized_doc["normalized"] = True
            normalized_doc["original_length"] = len(original_text)
            normalized_doc["normalized_length"] = len(cleaned_text)

            # Store offset mapping if requested
            if offset_file:
                offset_mappings.append(
                    {
                        "doc_id": doc.get("doc_id"),
                        "offset_map": offset_map,
                        "original_length": len(original_text),
                        "normalized_length": len(cleaned_text),
                    }
                )

            # Write normalized document
            outfile.write(json.dumps(normalized_doc, ensure_ascii=False) + "\n")

    # Write offset mappings if requested
    if offset_file and offset_mappings:
        with open(offset_file, "w") as f:
            for mapping in offset_mappings:
                f.write(json.dumps(mapping, ensure_ascii=False) + "\n")

    logger.info(f"Normalization complete. Output: {output_file}")
    if offset_file:
        logger.info(f"Offsets saved to: {offset_file}")


@app.command()
def validate(
    input_file: Path = typer.Option(..., "--input", help="Input documents JSONL file"),
    golden_file: Path = typer.Option(
        ..., "--golden", help="Golden reference JSONL file"
    ),
):
    """
    Validate normalization results against golden references.
    """
    logger.info(f"Validating normalization against golden file: {golden_file}")

    # Load golden references
    golden_docs = {}
    with open(golden_file, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            golden_docs[doc["doc_id"]] = doc

    # Validate input documents
    mismatches = []
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            doc = json.loads(line.strip())
            doc_id = doc.get("doc_id")

            if doc_id in golden_docs:
                golden_doc = golden_docs[doc_id]
                if doc.get("raw_text") != golden_doc.get("raw_text"):
                    mismatches.append(
                        {
                            "line": line_num,
                            "doc_id": doc_id,
                            "expected": golden_doc.get("raw_text", "")[:100] + "...",
                            "actual": doc.get("raw_text", "")[:100] + "...",
                        }
                    )

    if mismatches:
        logger.error(f"Found {len(mismatches)} mismatches:")
        for mismatch in mismatches[:5]:  # Show first 5
            logger.error(f"Line {mismatch['line']}: {mismatch['doc_id']}")
        return False
    else:
        logger.info("✅ All documents match golden references!")
        return True


if __name__ == "__main__":
    app()
