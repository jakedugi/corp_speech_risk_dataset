#!/usr/bin/env python3
"""
Example script demonstrating the enhanced position-based extraction functionality.

This script shows how to use the new chronological position-based voting system
that gives additional votes to candidates found in the latter portions of cases
and dockets, which are more likely to contain final judgment amounts.
"""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.case_outcome.case_outcome_imputer import (
    AmountSelector,
    scan_stage1,
)


def demonstrate_enhanced_extraction():
    """Demonstrate the enhanced extraction with position-based voting."""

    print("🔍 Enhanced Cash Amount Extraction with Position-Based Voting")
    print("=" * 60)

    # Example case directory
    case_dir = Path("data/extracted/courtlistener/0:12-cv-62086_flsd")

    if not case_dir.exists():
        print(f"❌ Example case directory not found: {case_dir}")
        print("   Please ensure you have extracted courtlistener data.")
        return

    print(f"📁 Analyzing case: {case_dir.name}")

    # Test with different threshold combinations
    test_configs = [
        {"case_thresh": 0.3, "docket_thresh": 0.3, "name": "Lenient"},
        {"case_thresh": 0.5, "docket_thresh": 0.5, "name": "Balanced"},
        {"case_thresh": 0.8, "docket_thresh": 0.8, "name": "Strict"},
    ]

    selector = AmountSelector()

    for config in test_configs:
        print(
            f"\n🎯 {config['name']} Thresholds (Case: {config['case_thresh']}, Docket: {config['docket_thresh']}):"
        )

        # Scan with enhanced position-based voting
        candidates = scan_stage1(
            case_dir,
            min_amount=10000,  # $10k minimum
            context_chars=100,
            min_features=2,  # Require at least 2 feature votes
            case_position_threshold=float(config["case_thresh"]),
            docket_position_threshold=float(config["docket_thresh"]),
        )

        print(f"   📊 Found {len(candidates)} candidates")

        if candidates:
            # Show top candidates
            sorted_candidates = sorted(
                candidates, key=lambda c: (c.feature_votes, c.value), reverse=True
            )

            print("   🏆 Top candidates:")
            for i, candidate in enumerate(sorted_candidates[:3], 1):
                print(
                    f"      {i}. ${candidate.value:,.0f} (votes: {candidate.feature_votes})"
                )
                print(f"         Context: ...{candidate.context[:80]}...")

            # Get the selected amount
            selected = selector.choose(candidates)
            print(f"   ✅ Selected amount: ${selected:,.0f}")
        else:
            print("   ❌ No candidates found")


def show_position_calculation_example():
    """Show how position calculation works."""

    print("\n📍 Position Calculation Example")
    print("-" * 40)

    # Example file paths in chronological order
    example_files = [
        "doc_125129554_text_stage8.jsonl",  # Position 0/5 = 0.0
        "doc_170098238_text_stage8.jsonl",  # Position 1/5 = 0.2
        "doc_170098243_text_stage8.jsonl",  # Position 2/5 = 0.4
        "doc_170098260_text_stage8.jsonl",  # Position 3/5 = 0.6
        "doc_170098267_text_stage8.jsonl",  # Position 4/5 = 0.8
        "doc_170098275_text_stage8.jsonl",  # Position 5/5 = 1.0
    ]

    print("📄 Files in chronological order (earliest to latest):")
    for i, filename in enumerate(example_files):
        position = i / (len(example_files) - 1) if len(example_files) > 1 else 0

        # Calculate votes for different thresholds
        votes_lenient = 2 if position >= 0.3 else 0  # Both case and docket position
        votes_balanced = 2 if position >= 0.5 else 0
        votes_strict = 2 if position >= 0.8 else 0

        print(f"   {i+1}. {filename}")
        print(f"      Position: {position:.1f}")
        print(
            f"      Votes → Lenient: {votes_lenient}, Balanced: {votes_balanced}, Strict: {votes_strict}"
        )


if __name__ == "__main__":
    demonstrate_enhanced_extraction()
    show_position_calculation_example()
    print("\n✅ Example completed!")
    print("\n💡 Key Benefits:")
    print("   • Candidates in later documents get additional votes")
    print("   • Configurable thresholds allow fine-tuning")
    print("   • Helps prioritize final judgment amounts over procedural mentions")
    print("   • Maintains backward compatibility with existing features")
