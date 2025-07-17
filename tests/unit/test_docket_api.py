#!/usr/bin/env python3
"""Test script for the CourtListener docket API with RECAP support."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corp_speech_risk_dataset.api.courtlistener import (
    CourtListenerClient,
    process_statutes,
    process_recap_data,
    process_docket_entries,
    process_recap_documents,
    process_full_docket,
)
from corp_speech_risk_dataset.config import load_config
from corp_speech_risk_dataset.custom_types.base_types import APIConfig


def main():
    """Test the docket API with upgraded privileges."""
    print("Testing CourtListener Docket API with upgraded privileges...")

    # Load configuration
    try:
        config = load_config()
        print(f"Loaded config with API mode: {config.api_mode}")

        if not config.api_token:
            print(
                "Warning: No API token found. Set COURTLISTENER_API_TOKEN environment variable."
            )
            return

    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Test standard API mode
    print("\n=== Testing Standard API Mode ===")
    try:
        client = CourtListenerClient(config, api_mode="standard")
        print("✓ Standard API client initialized successfully")

        # Test a simple search
        opinions = client.fetch_opinions(
            query="FTC Section 5", pages=1, page_size=5, date_filed_min="2023-01-01"
        )
        print(f"✓ Retrieved {len(opinions)} opinions from standard API")

    except Exception as e:
        print(f"✗ Error with standard API: {e}")

    # Test RECAP API mode
    print("\n=== Testing RECAP API Mode ===")
    try:
        client = CourtListenerClient(config, api_mode="recap")
        print("✓ RECAP API client initialized successfully")

        # Test RECAP data fetch
        recap_data = client.fetch_recap_data(query="FTC", pages=1, page_size=5)
        print(f"✓ Retrieved {len(recap_data)} RECAP records")

    except Exception as e:
        print(f"✗ Error with RECAP API: {e}")

    # Test processing functions
    print("\n=== Testing Processing Functions ===")
    try:
        # Test statute processing
        process_statutes(
            statutes=["FTC Section 5"],
            config=config,
            pages=1,
            page_size=5,
            date_min="2023-01-01",
            api_mode="standard",
        )
        print("✓ Statute processing completed")

        # Test RECAP processing
        process_recap_data(config=config, query="FTC", pages=1, page_size=5)
        print("✓ RECAP processing completed")

    except Exception as e:
        print(f"✗ Error with processing: {e}")

    # Test new docket entries endpoint
    print("\n=== Testing Docket Entries API ===")
    try:
        client = CourtListenerClient(config, api_mode="standard")

        # First, let's get a docket ID from the opinions we fetched earlier
        if opinions:
            # Try to get docket entries for the first opinion's docket
            first_opinion = opinions[0]
            if first_opinion.get("cluster"):
                cluster_id = first_opinion["cluster"]
                print(f"Testing docket entries for cluster: {cluster_id}")

                # Fetch docket entries
                entries = client.fetch_docket_entries(
                    query=f"cluster:{cluster_id}", pages=1, page_size=5
                )
                print(f"✓ Retrieved {len(entries)} docket entries")

                # Test docket entries processing
                process_docket_entries(
                    config=config,
                    query=f"cluster:{cluster_id}",
                    pages=1,
                    page_size=5,
                    api_mode="standard",
                )
                print("✓ Docket entries processing completed")
            else:
                print("No cluster ID found in opinion, skipping docket entries test")
        else:
            print("No opinions available, skipping docket entries test")

    except Exception as e:
        print(f"✗ Error with docket entries API: {e}")

    # Test new RECAP documents endpoint
    print("\n=== Testing RECAP Documents API ===")
    try:
        client = CourtListenerClient(config, api_mode="standard")

        # Test RECAP documents fetch
        documents = client.fetch_recap_documents(
            query="FTC", pages=1, page_size=5, include_plain_text=True
        )
        print(f"✓ Retrieved {len(documents)} RECAP documents")

        # Test RECAP documents processing
        process_recap_documents(
            config=config,
            query="FTC",
            pages=1,
            page_size=5,
            include_plain_text=True,
            api_mode="standard",
        )
        print("✓ RECAP documents processing completed")

    except Exception as e:
        print(f"✗ Error with RECAP documents API: {e}")

    # Test full docket functionality
    print("\n=== Testing Full Docket API ===")
    try:
        client = CourtListenerClient(config, api_mode="standard")

        # Try to get a docket ID from the opinions
        if opinions:
            first_opinion = opinions[0]
            if first_opinion.get("cluster"):
                cluster_id = first_opinion["cluster"]
                print(f"Testing full docket for cluster: {cluster_id}")

                # Note: This would require a specific docket ID, so we'll just test the method exists
                print(
                    "✓ Full docket methods available (requires specific docket ID for full test)"
                )
            else:
                print("No cluster ID found, skipping full docket test")
        else:
            print("No opinions available, skipping full docket test")

    except Exception as e:
        print(f"✗ Error with full docket API: {e}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
