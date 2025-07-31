#!/usr/bin/env python3
"""
focus_missing_cases.py

Focused analysis on the 9 cases where actual amounts were not found in any candidates.
Shows actual spans from CSV vs top 10 candidates from predictor.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from local modules
from case_outcome_imputer import (
    scan_stage1,
    VotingWeights,
    AmountSelector,
    Candidate,
)


class MissingCasesAnalyzer:
    """Analyze cases where actual amounts were not found in candidates."""

    def __init__(self):
        # Best parameters from optimization
        self.best_params = {
            "min_amount": 976,
            "context_chars": 510,
            "min_features": 7,
            "header_chars": 1686,
            "case_position_threshold": 0.634,
            "docket_position_threshold": 0.832,
            "proximity_pattern_weight": 1.000,
            "judgment_verbs_weight": 0.690,
            "case_position_weight": 1.752,
            "docket_position_weight": 2.609,
            "all_caps_titles_weight": 1.741,
            "document_titles_weight": 1.027,
        }

        self.voting_weights = VotingWeights(
            proximity_pattern_weight=self.best_params["proximity_pattern_weight"],
            judgment_verbs_weight=self.best_params["judgment_verbs_weight"],
            case_position_weight=self.best_params["case_position_weight"],
            docket_position_weight=self.best_params["docket_position_weight"],
            all_caps_titles_weight=self.best_params["all_caps_titles_weight"],
            document_titles_weight=self.best_params["document_titles_weight"],
        )

        self.selector = AmountSelector()

    def _load_gold_standard(self) -> pd.DataFrame:
        """Load the gold standard dataset."""
        project_root = Path(__file__).parent.parent.parent.parent
        gold_standard_path = (
            project_root / "data/gold_standard/case_outcome_amounts_hand_annotated.csv"
        )

        df = pd.read_csv(gold_standard_path)

        # Convert final_amount to numeric, handling commas
        df["final_amount"] = (
            df["final_amount"].astype(str).str.replace(",", "").str.replace("$", "")
        )
        df["final_amount"] = pd.to_numeric(df["final_amount"], errors="coerce")

        # Filter out rows with empty case_id or null final_amount
        df = df[df["case_id"].notna() & (df["case_id"] != "")]
        df = df[df["final_amount"].notna() & (df["final_amount"] != "null")]
        df = df.dropna(subset=["final_amount"])

        return df

    def _get_case_data_path(self, case_id: str) -> Optional[Path]:
        """Get the path to case data."""
        project_root = Path(__file__).parent.parent.parent.parent
        extracted_root = project_root / "data/extracted"

        if case_id.startswith("data/extracted/"):
            case_id = case_id.replace("data/extracted/", "")

        case_path = extracted_root / case_id
        return case_path if case_path.exists() else None

    def _find_actual_candidate(
        self, candidates: List[Candidate], actual_amount: float
    ) -> Optional[Candidate]:
        """Find the actual amount in candidates with tolerance."""
        tolerance = 1.0
        for candidate in candidates:
            if abs(candidate.value - actual_amount) <= tolerance:
                return candidate
        return None

    def analyze_missing_case(
        self, case_id: str, actual_amount: float, actual_span: str, actual_context: str
    ):
        """Analyze a case where actual amount was not found."""
        case_path = self._get_case_data_path(case_id)
        if not case_path:
            print(f"❌ Case data not found: {case_id}")
            return

        # Get candidates
        candidates = scan_stage1(
            case_path,
            min_amount=self.best_params["min_amount"],
            context_chars=self.best_params["context_chars"],
            min_features=self.best_params["min_features"],
            case_position_threshold=self.best_params["case_position_threshold"],
            docket_position_threshold=self.best_params["docket_position_threshold"],
            voting_weights=self.voting_weights,
            header_chars=self.best_params["header_chars"],
            fast_mode=True,
        )

        # Sort candidates by votes
        sorted_candidates = sorted(
            candidates, key=lambda c: c.feature_votes, reverse=True
        )

        # Choose best candidate
        chosen_candidate = self.selector.choose(candidates)
        predicted_amount = chosen_candidate if chosen_candidate else None

        # Find actual candidate
        actual_candidate = self._find_actual_candidate(candidates, actual_amount)

        print(f"\n{'='*120}")
        print(f"🔍 MISSING CASE ANALYSIS: {case_id}")
        print(f"💰 ACTUAL AMOUNT: ${actual_amount:,.0f}")
        print(f"{'='*120}")

        print(f"\n📊 ACTUAL FROM CSV:")
        print(f"   Amount: ${actual_amount:,.0f}")
        print(f"   Span: {actual_span}")
        print(f"   Context: {actual_context[:300]}...")

        print(f"\n🎯 PREDICTED:")
        if predicted_amount:
            error = abs(predicted_amount - actual_amount)
            print(f"   Amount: ${predicted_amount:,.0f}")
            print(f"   Error: ${error:,.0f}")
        else:
            print(f"   Amount: None")
            print(f"   Error: N/A")

        print(f"\n🏆 TOP 10 CANDIDATES (Total: {len(candidates)}):")
        for i, candidate in enumerate(sorted_candidates[:10], 1):
            is_chosen = (
                chosen_candidate and abs(candidate.value - chosen_candidate) < 1.0
            )
            chosen_marker = "✅ CHOSEN" if is_chosen else ""

            print(
                f"   {i:2d}. ${candidate.value:>15,.0f} ({candidate.feature_votes:2d} votes) {chosen_marker}"
            )
            print(f"       Span: {candidate.raw_text}")
            print(f"       Context: {candidate.context[:100]}...")

        # Check if actual was found anywhere
        if actual_candidate:
            rank = sorted_candidates.index(actual_candidate) + 1
            print(f"\n⚠️  ACTUAL FOUND BUT RANKED LOW:")
            print(f"   Rank: #{rank} out of {len(candidates)}")
            print(f"   Votes: {actual_candidate.feature_votes}")
            print(f"   Span: {actual_candidate.raw_text}")
            print(f"   Context: {actual_candidate.context[:100]}...")
        else:
            print(f"\n❌ ACTUAL AMOUNT NOT FOUND IN ANY CANDIDATES")
            print(
                f"   This suggests the extraction or filtering is missing the actual amount"
            )

        # Analyze why it might be missing
        print(f"\n🔍 POSSIBLE REASONS FOR MISSING:")

        # Check if actual amount is too small
        if actual_amount < self.best_params["min_amount"]:
            print(
                f"   • Actual amount (${actual_amount:,.0f}) is below min_amount threshold (${self.best_params['min_amount']:,.0f})"
            )

        # Check if it's in a different format
        if actual_span and actual_span != "nan":
            print(f"   • Actual span format: '{actual_span}'")
            print(
                f"   • Check if this format is being captured by the extraction patterns"
            )

        # Check context for clues
        if actual_context and actual_context != "nan":
            print(f"   • Actual context mentions: '{actual_context[:100]}...'")
            print(f"   • Check if this context is being processed correctly")

    def run_missing_cases_analysis(self):
        """Run analysis on cases where actual amounts were not found."""
        df = self._load_gold_standard()

        print(f"🔍 FOCUSED ANALYSIS ON MISSING CASES")
        print(f"📋 Analyzing cases where actual amounts were not found in candidates")
        print(f"{'='*120}")

        missing_cases = []

        for idx, row in df.iterrows():
            case_id = str(row["case_id"])
            actual_amount = float(row["final_amount"])
            actual_span = str(row.get("amount_text", "N/A"))
            actual_context = str(row.get("candidate_text", "N/A"))

            case_path = self._get_case_data_path(case_id)
            if not case_path:
                continue

            # Get candidates
            candidates = scan_stage1(
                case_path,
                min_amount=self.best_params["min_amount"],
                context_chars=self.best_params["context_chars"],
                min_features=self.best_params["min_features"],
                case_position_threshold=self.best_params["case_position_threshold"],
                docket_position_threshold=self.best_params["docket_position_threshold"],
                voting_weights=self.voting_weights,
                header_chars=self.best_params["header_chars"],
                fast_mode=True,
            )

            # Check if actual was found
            actual_candidate = self._find_actual_candidate(candidates, actual_amount)

            if not actual_candidate:
                missing_cases.append(
                    {
                        "case_id": case_id,
                        "actual_amount": actual_amount,
                        "actual_span": actual_span,
                        "actual_context": actual_context,
                        "total_candidates": len(candidates),
                    }
                )

        print(f"\n📊 FOUND {len(missing_cases)} CASES WHERE ACTUAL AMOUNT WAS MISSING:")
        for case in missing_cases:
            print(
                f"   • {case['case_id']}: ${case['actual_amount']:,.0f} ({case['total_candidates']} candidates)"
            )

        # Analyze each missing case
        for case in missing_cases:
            self.analyze_missing_case(
                case["case_id"],
                case["actual_amount"],
                case["actual_span"],
                case["actual_context"],
            )

        return missing_cases


def main():
    """Run focused analysis on missing cases."""
    analyzer = MissingCasesAnalyzer()

    # Run analysis
    missing_cases = analyzer.run_missing_cases_analysis()

    print(f"\n🎉 Focused analysis completed!")
    print(f"📊 Summary: {len(missing_cases)} cases where actual amounts were not found")


if __name__ == "__main__":
    main()
