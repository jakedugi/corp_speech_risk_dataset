#!/usr/bin/env python3
"""
Extract Interpretable Features Script

This script:
1. Loads JSONL data from the balanced splits
2. Extracts interpretable features using the fully interpretable feature extractor
3. Appends features as new fields to each record
4. Saves enhanced data with features

Key features:
- Batch processing for memory efficiency
- Progress tracking with detailed logging
- Feature name prefixing to avoid conflicts
- Context-aware feature extraction
- Robust error handling

Usage:
    python scripts/extract_interpretable_features.py \
        --input data/balanced_case_splits/train.jsonl \
        --output data/balanced_case_splits/train_with_features.jsonl \
        --text-field text \
        --context-field context \
        --batch-size 1000 \
        --feature-prefix interpretable
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)


class FeatureProcessor:
    """Process JSONL files and append interpretable features."""

    def __init__(
        self,
        text_field: str = "text",
        context_field: str = "context",
        feature_prefix: str = "interpretable",
        batch_size: int = 1000,
        include_lexicons: bool = True,
        include_sequence: bool = True,
        include_linguistic: bool = True,
        include_structural: bool = True,
    ):
        """Initialize the feature processor.

        Args:
            text_field: Name of the field containing the main text
            context_field: Name of the field containing context (optional)
            feature_prefix: Prefix to add to all feature names
            batch_size: Number of records to process in each batch
            include_lexicons: Whether to include lexicon-based features
            include_sequence: Whether to include sequence features
            include_linguistic: Whether to include linguistic features
            include_structural: Whether to include structural features
        """
        self.text_field = text_field
        self.context_field = context_field
        self.feature_prefix = feature_prefix
        self.batch_size = batch_size

        # Initialize feature extractor
        self.extractor = InterpretableFeatureExtractor(
            include_lexicons=include_lexicons,
            include_sequence=include_sequence,
            include_linguistic=include_linguistic,
            include_structural=include_structural,
        )

        logger.info(f"Initialized FeatureProcessor with:")
        logger.info(f"  Text field: {text_field}")
        logger.info(f"  Context field: {context_field}")
        logger.info(f"  Feature prefix: {feature_prefix}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(
            f"  Feature types: lexicons={include_lexicons}, sequence={include_sequence}, "
            f"linguistic={include_linguistic}, structural={include_structural}"
        )

    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load records from JSONL file."""
        records = []

        logger.info(f"Loading data from {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records

    def save_jsonl(self, records: List[Dict[str, Any]], file_path: Path) -> None:
        """Save records to JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(records)} records to {file_path}...")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise

        logger.success(f"Successfully saved records to {file_path}")

    def extract_features_for_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interpretable features for a single record."""
        try:
            # Get text and context
            text = record.get(self.text_field, "")
            context = record.get(self.context_field, "")

            if not text:
                logger.warning(
                    f"Empty text field for record: {record.get('doc_id', 'unknown')}"
                )
                text = ""

            # Extract features
            features = self.extractor.extract_features(
                text, context if context else None
            )

            # Add prefix to feature names to avoid conflicts
            prefixed_features = {}
            for name, value in features.items():
                prefixed_name = (
                    f"{self.feature_prefix}_{name}" if self.feature_prefix else name
                )
                prefixed_features[prefixed_name] = value

            return prefixed_features

        except Exception as e:
            logger.error(
                f"Error extracting features for record {record.get('doc_id', 'unknown')}: {e}"
            )
            return {}

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records and append features."""
        enhanced_records = []

        for i, record in enumerate(records):
            try:
                # Create enhanced record with original data
                enhanced_record = record.copy()

                # Extract and add features
                features = self.extract_features_for_record(record)
                enhanced_record.update(features)

                enhanced_records.append(enhanced_record)

                # Log progress within batch
                if (i + 1) % 100 == 0:
                    logger.debug(
                        f"Processed {i + 1}/{len(records)} records in current batch"
                    )

            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                # Include original record without features rather than failing
                enhanced_records.append(record)

        return enhanced_records

    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process entire file with batch processing."""
        logger.info(f"Starting feature extraction from {input_path} to {output_path}")

        # Load all records
        records = self.load_jsonl(input_path)

        if not records:
            logger.warning("No records to process")
            return

        # Process in batches
        all_enhanced_records = []
        total_batches = (len(records) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(records), self.batch_size):
            batch_num = batch_idx // self.batch_size + 1
            batch_records = records[batch_idx : batch_idx + self.batch_size]

            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch_records)} records)"
            )

            try:
                enhanced_batch = self.process_batch(batch_records)
                all_enhanced_records.extend(enhanced_batch)

                logger.info(f"Completed batch {batch_num}/{total_batches}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add original records without features
                all_enhanced_records.extend(batch_records)

        # Save enhanced records
        self.save_jsonl(all_enhanced_records, output_path)

        # Log summary
        if all_enhanced_records:
            # Check feature extraction success rate
            sample_record = all_enhanced_records[0]
            feature_count = sum(
                1
                for key in sample_record.keys()
                if key.startswith(self.feature_prefix + "_")
            )

            logger.success(f"Feature extraction complete!")
            logger.info(f"  Processed: {len(all_enhanced_records)} records")
            logger.info(f"  Features per record: {feature_count}")
            logger.info(f"  Output saved to: {output_path}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract interpretable features from JSONL data"
    )

    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--text-field", default="text", help="Name of text field (default: text)"
    )
    parser.add_argument(
        "--context-field",
        default="context",
        help="Name of context field (default: context)",
    )
    parser.add_argument(
        "--feature-prefix",
        default="interpretable",
        help="Prefix for feature names (default: interpretable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )

    # Feature type flags
    parser.add_argument(
        "--no-lexicons", action="store_true", help="Disable lexicon features"
    )
    parser.add_argument(
        "--no-sequence", action="store_true", help="Disable sequence features"
    )
    parser.add_argument(
        "--no-linguistic", action="store_true", help="Disable linguistic features"
    )
    parser.add_argument(
        "--no-structural", action="store_true", help="Disable structural features"
    )

    # Processing options
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    # Initialize processor
    processor = FeatureProcessor(
        text_field=args.text_field,
        context_field=args.context_field,
        feature_prefix=args.feature_prefix,
        batch_size=args.batch_size,
        include_lexicons=not args.no_lexicons,
        include_sequence=not args.no_sequence,
        include_linguistic=not args.no_linguistic,
        include_structural=not args.no_structural,
    )

    # Process file
    try:
        processor.process_file(input_path, output_path)
        logger.success("Feature extraction completed successfully!")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
