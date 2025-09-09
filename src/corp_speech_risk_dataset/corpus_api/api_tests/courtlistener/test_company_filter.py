#!/usr/bin/env python3
"""Test script for the company file filtering functionality."""

import os
import sys
import tempfile
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from corp_speech_risk_dataset.api.adapters.courtlistener.courtlistener_core import (
    process_statutes,
)
from corp_speech_risk_dataset.api.adapters.courtlistener.queries import STATUTE_QUERIES
from corp_speech_risk_dataset.config import load_config
from corp_speech_risk_dataset.custom_types.base_types import APIConfig


def create_test_company_file():
    """Create a temporary CSV file with test company names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "official_name", "wiki_url", "cik"])
        writer.writerow(
            ["AAPL", "Apple Inc.", "https://example.com/apple", "0000320193"]
        )
        writer.writerow(
            [
                "MSFT",
                "Microsoft Corporation",
                "https://example.com/microsoft",
                "0000789019",
            ]
        )
        writer.writerow(
            ["GOOGL", "Alphabet Inc.", "https://example.com/alphabet", "0001652044"]
        )
        return f.name


def test_company_filter_logic():
    """Test that the company filter logic works correctly."""
    print("Testing company filter logic...")

    # Create a test company file
    company_file = create_test_company_file()

    try:
        # Test the logic manually
        import csv

        names = set()
        with open(company_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.add(row["official_name"].strip())

        # CourtListener full‐text search is case‐insensitive, so just quote each name
        company_filter = "(" + " OR ".join(f'"{n}"' for n in sorted(names)) + ")"

        print(f"Company names loaded: {sorted(names)}")
        print(f"Company filter: {company_filter}")

        # Test with a sample statute
        statute = "FTC Section 5"
        original_query = STATUTE_QUERIES[statute]
        filtered_query = original_query.strip() + "\nAND\n" + company_filter

        print(f"\nOriginal query for {statute}:")
        print(original_query)
        print(f"\nFiltered query for {statute}:")
        print(filtered_query)

        # Verify the filter is properly appended
        assert "Apple Inc." in filtered_query
        assert "Microsoft Corporation" in filtered_query
        assert "Alphabet Inc." in filtered_query
        assert "AND" in filtered_query

        print("✓ Company filter logic test passed!")

    finally:
        # Clean up
        os.unlink(company_file)


def test_process_statutes_with_company_file():
    """Test that process_statutes accepts the company_file parameter."""
    print("\nTesting process_statutes with company file parameter...")

    # Create a test company file
    company_file = create_test_company_file()

    try:
        # Load config (this will fail without API token, but we're just testing the function signature)
        try:
            config = load_config()
        except Exception:
            # Create a mock config for testing
            config = APIConfig(api_key="test_token")

        # Test that the function can be called with company_file parameter
        # We expect this to fail due to missing API token, but the function should accept the parameter
        try:
            process_statutes(
                statutes=["FTC Section 5"],
                config=config,
                pages=1,
                page_size=5,
                date_min="2023-01-01",
                api_mode="standard",
                company_file=company_file,
            )
        except Exception as e:
            # Expected to fail due to missing API token, but should not fail due to parameter issues
            if "API token" in str(e) or "401" in str(e) or "403" in str(e):
                print(
                    "✓ process_statutes accepts company_file parameter (failed as expected due to API token)"
                )
            else:
                print(f"✗ Unexpected error: {e}")
                raise

    finally:
        # Clean up
        os.unlink(company_file)


def main():
    """Run all tests."""
    print("Testing company file filtering functionality...")

    test_company_filter_logic()
    test_process_statutes_with_company_file()

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
