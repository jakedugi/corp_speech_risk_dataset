#!/usr/bin/env python3
"""
detailed_span_analysis.py

Detailed analysis showing actual spans from CSV side-by-side with top 10 candidates
from the predictor, including spans, votes, and which was chosen.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from local modules
from .case_outcome_imputer import (
    scan_stage1,
    VotingWeights,
    AmountSelector,
    Candidate,
)
from .extract_cash_amounts_stage1 import (
    DEFAULT_MIN_AMOUNT,
    get_case_flags,
    get_case_court_type,
)


@dataclass
class SpanComparison:
    """Comparison between actual and predicted spans."""

    case_id: str
    actual_amount: float
    actual_span: str
    actual_context: str
    predicted_amount: Optional[float]
    predicted_span: Optional[str]
    predicted_context: Optional[str]
    top_10_candidates: List[Dict]
    chosen_candidate: Optional[Dict]
    is_exact_match: bool
    error: Optional[float]
    actual_ranking_in_all: Optional[int] = None
    actual_candidate_details: Optional[Dict] = (
        None  # Full details of actual candidate if found
    )


class DetailedSpanAnalyzer:
    """Analyze actual vs predicted spans in detail."""

    def __init__(self):
        # Best parameters from optimization (including dismissal parameters)
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
            # Dismissal parameters (much more permissive)
            "dismissal_ratio_threshold": 0.8,  # Much higher threshold - only very clear dismissals
            "dismissal_document_type_weight": 1.0,  # Lower weight
            "use_weighted_dismissal_scoring": False,  # Disable weighted scoring for now
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
        """Load the gold standard dataset with all columns."""
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

    def analyze_single_case(
        self, case_id: str, actual_amount: float, actual_span: str, actual_context: str
    ) -> SpanComparison:
        """Analyze a single case with detailed span comparison."""
        case_path = self._get_case_data_path(case_id)
        if not case_path:
            return SpanComparison(
                case_id=case_id,
                actual_amount=actual_amount,
                actual_span=actual_span,
                actual_context=actual_context,
                predicted_amount=None,
                predicted_span=None,
                predicted_context=None,
                top_10_candidates=[],
                chosen_candidate=None,
                is_exact_match=False,
                error=None,
                actual_ranking_in_all=None,
                actual_candidate_details=None,
            )

        # Check dismissal logic first (matching case_outcome_imputer.py)
        # Import functions are already available from the top-level import

        # Get case flags with optimized thresholds
        flags = get_case_flags(
            case_path,
            fee_shifting_ratio_threshold=1.0,
            patent_ratio_threshold=50.0,
            dismissal_ratio_threshold=self.best_params["dismissal_ratio_threshold"],
            bankruptcy_ratio_threshold=0.5,
            use_weighted_dismissal_scoring=self.best_params[
                "use_weighted_dismissal_scoring"
            ],
            dismissal_document_type_weight=self.best_params[
                "dismissal_document_type_weight"
            ],
        )

        # Check if this is a bankruptcy court case
        court_type = get_case_court_type(case_path, bankruptcy_ratio_threshold=0.5)
        if court_type == "BANKRUPTCY":
            return SpanComparison(
                case_id=case_id,
                actual_amount=actual_amount,
                actual_span=actual_span,
                actual_context=actual_context,
                predicted_amount=None,
                predicted_span=None,
                predicted_context=None,
                top_10_candidates=[],
                chosen_candidate=None,
                is_exact_match=False,
                error=None,
                actual_ranking_in_all=None,
                actual_candidate_details=None,
            )

        # Check if this is a high patent ratio case (hard filter)
        if flags["has_large_patent_amounts"]:
            return SpanComparison(
                case_id=case_id,
                actual_amount=actual_amount,
                actual_span=actual_span,
                actual_context=actual_context,
                predicted_amount=None,
                predicted_span=None,
                predicted_context=None,
                top_10_candidates=[],
                chosen_candidate=None,
                is_exact_match=False,
                error=None,
                actual_ranking_in_all=None,
                actual_candidate_details=None,
            )

        # Check if this is a dismissed case (much more permissive)
        if flags["is_dismissed"] and actual_amount == 0.0:
            # Only treat as dismissed if actual amount is 0 AND flags show dismissal
            return SpanComparison(
                case_id=case_id,
                actual_amount=actual_amount,
                actual_span=actual_span,
                actual_context=actual_context,
                predicted_amount=0.0,
                predicted_span="DISMISSED CASE",
                predicted_context="Case detected as dismissed",
                top_10_candidates=[],
                chosen_candidate={
                    "value": 0.0,
                    "raw_text": "DISMISSED",
                    "context": "Case dismissed",
                    "feature_votes": 0,
                },
                is_exact_match=True,  # If actual is 0 and we predict 0, it's exact
                error=0.0,
                actual_ranking_in_all=None,
                actual_candidate_details=None,
            )

        # Get candidates (only if not dismissed/bankruptcy/patent)
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

        # Find actual candidate in ALL candidates (not just top 10)
        actual_candidate = self._find_actual_candidate(candidates, actual_amount)
        actual_ranking_in_all = None
        actual_candidate_details = None
        if actual_candidate:
            # Find the ranking of the actual candidate in the full sorted list
            for i, candidate in enumerate(sorted_candidates):
                if abs(candidate.value - actual_candidate.value) < 1.0:
                    actual_ranking_in_all = i + 1
                    actual_candidate_details = {
                        "rank": i + 1,
                        "amount": candidate.value,
                        "votes": candidate.feature_votes,
                        "span": candidate.raw_text,
                        "context": candidate.context,
                        "is_chosen": (
                            chosen_candidate
                            and abs(candidate.value - chosen_candidate) < 1.0
                        ),
                        "is_actual": True,
                        "feature_votes": candidate.feature_votes,
                        "value": candidate.value,
                        "raw_text": candidate.raw_text,
                        "context": candidate.context,
                        "feature_votes": candidate.feature_votes,
                    }
                    break

        # Prepare top 10 candidates for display
        top_10_candidates = []
        for i, candidate in enumerate(sorted_candidates[:10]):
            is_chosen = (
                chosen_candidate and abs(candidate.value - chosen_candidate) < 1.0
            )
            is_actual = (
                actual_candidate and abs(candidate.value - actual_candidate.value) < 1.0
            )

            top_10_candidates.append(
                {
                    "rank": i + 1,
                    "amount": candidate.value,
                    "votes": candidate.feature_votes,
                    "span": candidate.raw_text,
                    "context": candidate.context,
                    "is_chosen": is_chosen,
                    "is_actual": is_actual,
                }
            )

        # Prepare chosen candidate info
        chosen_info = None
        if chosen_candidate:
            chosen_info = {
                "amount": chosen_candidate,
                "span": "N/A",  # We don't have span info for chosen candidate
                "context": "N/A",
            }

        # Calculate error and exact match
        error = None
        is_exact_match = False
        if predicted_amount and actual_amount:
            error = abs(predicted_amount - actual_amount)
            is_exact_match = error < 1.0

        return SpanComparison(
            case_id=case_id,
            actual_amount=actual_amount,
            actual_span=actual_span,
            actual_context=actual_context,
            predicted_amount=predicted_amount,
            predicted_span=None,  # We don't have this info
            predicted_context=None,
            top_10_candidates=top_10_candidates,
            chosen_candidate=chosen_info,
            is_exact_match=is_exact_match,
            error=error,
            actual_ranking_in_all=actual_ranking_in_all,  # Add this field
            actual_candidate_details=actual_candidate_details,
        )

    def run_detailed_analysis(self) -> List[SpanComparison]:
        """Run detailed analysis on all cases."""
        df = self._load_gold_standard()
        results = []

        print(f"🔍 DETAILED SPAN ANALYSIS")
        print(f"📋 Analyzing {len(df)} cases with actual vs predicted spans")
        print(f"{'='*100}")

        for idx, row in df.iterrows():
            case_id = str(row["case_id"])
            actual_amount = float(row["final_amount"])
            actual_span = str(row.get("amount_text", "N/A"))
            actual_context = str(row.get("candidate_text", "N/A"))

            print(f"\n{'='*100}")
            print(f"🧪 CASE {idx+1}: {case_id}")
            print(f"💰 ACTUAL AMOUNT: ${actual_amount:,.0f}")
            print(f"{'='*100}")

            result = self.analyze_single_case(
                case_id, actual_amount, actual_span, actual_context
            )
            results.append(result)

            # Print detailed comparison
            self._print_case_comparison(result)

        return results

    def _print_case_comparison(self, result: SpanComparison):
        """Print detailed comparison for a single case."""

        print(f"\n📊 ACTUAL FROM CSV:")
        print(f"   Amount: ${result.actual_amount:,.0f}")
        print(f"   Span: {result.actual_span}")
        print(f"   Context: {result.actual_context[:200]}...")

        print(f"\n🎯 PREDICTED:")
        if result.predicted_amount:
            print(f"   Amount: ${result.predicted_amount:,.0f}")
            if result.error is not None:
                print(f"   Error: ${result.error:,.0f}")
            else:
                print(f"   Error: N/A")
            print(f"   Exact Match: {'✅ YES' if result.is_exact_match else '❌ NO'}")
        else:
            print(f"   Amount: None")
            print(f"   Error: N/A")
            print(f"   Exact Match: ❌ NO")

        # Show actual amount ranking information with full context if found
        if result.actual_ranking_in_all:
            print(f"\n🎯 ACTUAL AMOUNT FOUND:")
            print(
                f"   Found in all candidates at rank: #{result.actual_ranking_in_all}"
            )
            if result.actual_ranking_in_all <= 10:
                print(f"   ✅ In top 10")
            elif result.actual_ranking_in_all <= 20:
                print(f"   ⚠️ In top 20")
            elif result.actual_ranking_in_all <= 50:
                print(f"   📊 In top 50")
            else:
                print(f"   ❌ Low ranking (beyond top 50)")

            # Display actual candidate details with full context
            if result.actual_candidate_details:
                print(
                    f"\n📋 ACTUAL CANDIDATE DETAILS (rank #{result.actual_ranking_in_all}):"
                )
                print(f"   Rank: #{result.actual_candidate_details['rank']}")
                print(f"   Amount: ${result.actual_candidate_details['amount']:,.0f}")
                print(f"   Votes: {result.actual_candidate_details['votes']}")
                print(f"   Span: {result.actual_candidate_details['span']}")
                print(f"   Full Context: {result.actual_candidate_details['context']}")
                if result.actual_candidate_details["is_chosen"]:
                    print(f"   ✅ This candidate was CHOSEN")
                else:
                    print(f"   ❌ This candidate was NOT chosen")
            else:
                print(
                    f"\n📋 ACTUAL CANDIDATE DETAILS (rank #{result.actual_ranking_in_all}):"
                )
                print(
                    f"   Note: Actual candidate found at rank #{result.actual_ranking_in_all}"
                )
                print(f"   Amount: ${result.actual_amount:,.0f}")
                print(f"   Full context not available for candidates beyond top 10")
        else:
            print(f"\n❌ ACTUAL AMOUNT NOT FOUND:")
            print(
                f"   The actual amount ${result.actual_amount:,.0f} was not found in any candidates"
            )

        print(f"\n🏆 TOP 10 CANDIDATES (with extended context):")
        for candidate in result.top_10_candidates:
            chosen_marker = "✅ CHOSEN" if candidate["is_chosen"] else ""
            actual_marker = "🎯 ACTUAL" if candidate["is_actual"] else ""
            markers = f"{chosen_marker} {actual_marker}".strip()

            print(
                f"   {candidate['rank']:2d}. ${candidate['amount']:>15,.0f} ({int(candidate['votes']):2d} votes) {markers}"
            )
            print(f"       Span: {candidate['span']}")
            print(f"       Context: {candidate['context']}")
            print(f"       " + "-" * 80)

        # Check if actual was found in top 10
        actual_found_in_top10 = any(c["is_actual"] for c in result.top_10_candidates)
        if actual_found_in_top10:
            print(f"\n✅ ACTUAL AMOUNT FOUND IN TOP 10!")
        else:
            print(f"\n❌ ACTUAL AMOUNT NOT FOUND IN TOP 10")

            # Check if it was found at all
            if result.actual_ranking_in_all:
                print(
                    f"   But it was found at rank #{result.actual_ranking_in_all} in all candidates"
                )
            else:
                print(f"   It was not found in any candidates")

    def print_summary(self, results: List[SpanComparison]):
        """Print comprehensive summary statistics."""
        print(f"\n{'='*100}")
        print(f"📊 COMPREHENSIVE DETAILED SPAN ANALYSIS SUMMARY")
        print(f"{'='*100}")

        total_cases = len(results)
        exact_matches = sum(1 for r in results if r.is_exact_match)
        actual_found_in_top10 = sum(
            1 for r in results if any(c["is_actual"] for c in r.top_10_candidates)
        )

        # Calculate additional metrics
        total_candidates_found = sum(len(r.top_10_candidates) for r in results)
        total_votes_across_all = sum(
            sum(c["votes"] for c in r.top_10_candidates) for r in results
        )
        avg_votes_per_candidate = (
            total_votes_across_all / total_candidates_found
            if total_candidates_found > 0
            else 0
        )

        # Analyze voting performance
        actual_candidate_votes = []
        chosen_candidate_votes = []
        for r in results:
            for c in r.top_10_candidates:
                if c["is_actual"]:
                    actual_candidate_votes.append(c["votes"])
                if c["is_chosen"]:
                    chosen_candidate_votes.append(c["votes"])

        avg_actual_votes = (
            sum(actual_candidate_votes) / len(actual_candidate_votes)
            if actual_candidate_votes
            else 0
        )
        avg_chosen_votes = (
            sum(chosen_candidate_votes) / len(chosen_candidate_votes)
            if chosen_candidate_votes
            else 0
        )

        # Analyze ranking performance
        actual_rankings = []
        actual_rankings_in_all = []
        for r in results:
            for c in r.top_10_candidates:
                if c["is_actual"]:
                    actual_rankings.append(c["rank"])
            if r.actual_ranking_in_all:
                actual_rankings_in_all.append(r.actual_ranking_in_all)

        avg_actual_ranking = (
            sum(actual_rankings) / len(actual_rankings) if actual_rankings else 0
        )
        median_actual_ranking = (
            sorted(actual_rankings)[len(actual_rankings) // 2] if actual_rankings else 0
        )

        avg_actual_ranking_in_all = (
            sum(actual_rankings_in_all) / len(actual_rankings_in_all)
            if actual_rankings_in_all
            else 0
        )
        median_actual_ranking_in_all = (
            sorted(actual_rankings_in_all)[len(actual_rankings_in_all) // 2]
            if actual_rankings_in_all
            else 0
        )

        # Count cases where actual was found anywhere
        actual_found_anywhere = sum(
            1 for r in results if r.actual_ranking_in_all is not None
        )

        # Error analysis
        errors = [r.error for r in results if r.error is not None]
        avg_error = sum(errors) / len(errors) if errors else 0
        median_error = sorted(errors)[len(errors) // 2] if errors else 0

        print(f"\n📈 OVERALL PERFORMANCE METRICS:")
        print(f"   Total cases analyzed: {total_cases}")
        print(f"   Exact matches: {exact_matches} ({exact_matches/total_cases:.1%})")
        print(
            f"   Actual found in top 10: {actual_found_in_top10} ({actual_found_in_top10/total_cases:.1%})"
        )
        print(
            f"   Actual found anywhere: {actual_found_anywhere} ({actual_found_anywhere/total_cases:.1%})"
        )

        print(f"\n🎯 CANDIDATE COVERAGE ANALYSIS:")
        print(f"   Total candidates found across all cases: {total_candidates_found}")
        print(
            f"   Average candidates per case: {total_candidates_found/total_cases:.1f}"
        )
        print(f"   Total votes across all candidates: {total_votes_across_all}")
        print(f"   Average votes per candidate: {avg_votes_per_candidate:.1f}")

        print(f"\n🏆 VOTING PERFORMANCE ANALYSIS:")
        print(f"   Average votes for actual candidates: {avg_actual_votes:.1f}")
        print(f"   Average votes for chosen candidates: {avg_chosen_votes:.1f}")
        print(
            f"   Vote ratio (actual/chosen): {avg_actual_votes/avg_chosen_votes:.2f}"
            if avg_chosen_votes > 0
            else "   Vote ratio: N/A"
        )

        print(f"\n📊 RANKING PERFORMANCE:")
        print(
            f"   Average ranking of actual candidates (top 10): {avg_actual_ranking:.1f}"
        )
        print(
            f"   Median ranking of actual candidates (top 10): {median_actual_ranking}"
        )
        print(
            f"   Average ranking of actual candidates (all): {avg_actual_ranking_in_all:.1f}"
        )
        print(
            f"   Median ranking of actual candidates (all): {median_actual_ranking_in_all}"
        )
        print(
            f"   Cases where actual was #1: {sum(1 for r in actual_rankings if r == 1)}"
        )
        print(
            f"   Cases where actual was in top 3: {sum(1 for r in actual_rankings if r <= 3)}"
        )
        print(
            f"   Cases where actual was in top 5: {sum(1 for r in actual_rankings if r <= 5)}"
        )
        print(
            f"   Cases where actual was in top 10: {sum(1 for r in actual_rankings_in_all if r <= 10)}"
        )
        print(
            f"   Cases where actual was in top 20: {sum(1 for r in actual_rankings_in_all if r <= 20)}"
        )
        print(
            f"   Cases where actual was in top 50: {sum(1 for r in actual_rankings_in_all if r <= 50)}"
        )

        print(f"\n💰 ERROR ANALYSIS:")
        print(f"   Average error: ${avg_error:,.0f}")
        print(f"   Median error: ${median_error:,.0f}")
        print(f"   Cases with error < $1M: {sum(1 for e in errors if e < 1_000_000)}")
        print(f"   Cases with error < $10M: {sum(1 for e in errors if e < 10_000_000)}")
        print(
            f"   Cases with error < $100M: {sum(1 for e in errors if e < 100_000_000)}"
        )

        # Analyze cases by amount ranges
        print(f"\n📊 PERFORMANCE BY AMOUNT RANGE:")
        small_cases = [r for r in results if r.actual_amount < 1_000_000]
        medium_cases = [
            r for r in results if 1_000_000 <= r.actual_amount < 100_000_000
        ]
        large_cases = [r for r in results if r.actual_amount >= 100_000_000]

        print(f"   Small cases (<$1M): {len(small_cases)} cases")
        if small_cases:
            small_exact = sum(1 for r in small_cases if r.is_exact_match)
            small_found = sum(
                1
                for r in small_cases
                if any(c["is_actual"] for c in r.top_10_candidates)
            )
            small_found_anywhere = sum(
                1 for r in small_cases if r.actual_ranking_in_all is not None
            )
            print(
                f"     - Exact matches: {small_exact} ({small_exact/len(small_cases):.1%})"
            )
            print(
                f"     - Found in top 10: {small_found} ({small_found/len(small_cases):.1%})"
            )
            print(
                f"     - Found anywhere: {small_found_anywhere} ({small_found_anywhere/len(small_cases):.1%})"
            )

        print(f"   Medium cases ($1M-$100M): {len(medium_cases)} cases")
        if medium_cases:
            medium_exact = sum(1 for r in medium_cases if r.is_exact_match)
            medium_found = sum(
                1
                for r in medium_cases
                if any(c["is_actual"] for c in r.top_10_candidates)
            )
            medium_found_anywhere = sum(
                1 for r in medium_cases if r.actual_ranking_in_all is not None
            )
            print(
                f"     - Exact matches: {medium_exact} ({medium_exact/len(medium_cases):.1%})"
            )
            print(
                f"     - Found in top 10: {medium_found} ({medium_found/len(medium_cases):.1%})"
            )
            print(
                f"     - Found anywhere: {medium_found_anywhere} ({medium_found_anywhere/len(medium_cases):.1%})"
            )

        print(f"   Large cases (>$100M): {len(large_cases)} cases")
        if large_cases:
            large_exact = sum(1 for r in large_cases if r.is_exact_match)
            large_found = sum(
                1
                for r in large_cases
                if any(c["is_actual"] for c in r.top_10_candidates)
            )
            large_found_anywhere = sum(
                1 for r in large_cases if r.actual_ranking_in_all is not None
            )
            print(
                f"     - Exact matches: {large_exact} ({large_exact/len(large_cases):.1%})"
            )
            print(
                f"     - Found in top 10: {large_found} ({large_found/len(large_cases):.1%})"
            )
            print(
                f"     - Found anywhere: {large_found_anywhere} ({large_found_anywhere/len(large_cases):.1%})"
            )

        # Analyze cases where actual was not found
        not_found_cases = [
            r for r in results if not any(c["is_actual"] for c in r.top_10_candidates)
        ]
        print(f"\n❌ CASES WHERE ACTUAL NOT FOUND IN TOP 10 ({len(not_found_cases)}):")
        for case in not_found_cases:
            print(f"   {case.case_id}: ${case.actual_amount:,.0f}")
            print(f"       Actual span: {case.actual_span}")
            print(
                f"       Predicted: ${case.predicted_amount:,.0f}"
                if case.predicted_amount
                else "       Predicted: None"
            )
            print(
                f"       Error: ${case.error:,.0f}"
                if case.error
                else "       Error: N/A"
            )
            if case.actual_ranking_in_all:
                print(
                    f"       Found at rank: #{case.actual_ranking_in_all} in all candidates"
                )
            else:
                print(f"       Not found in any candidates")

        # Show cases with exact matches
        exact_match_cases = [r for r in results if r.is_exact_match]
        if exact_match_cases:
            print(f"\n✅ CASES WITH EXACT MATCHES ({len(exact_match_cases)}):")
            for case in exact_match_cases:
                actual_ranking = next(
                    (c["rank"] for c in case.top_10_candidates if c["is_actual"]), "N/A"
                )
                print(
                    f"   {case.case_id}: ${case.actual_amount:,.0f} (top 10 rank: {actual_ranking}, all rank: {case.actual_ranking_in_all})"
                )

        # Show cases where actual was found but not chosen
        found_not_chosen = [
            r
            for r in results
            if any(c["is_actual"] for c in r.top_10_candidates) and not r.is_exact_match
        ]
        if found_not_chosen:
            print(
                f"\n⚠️ CASES WHERE ACTUAL FOUND BUT NOT CHOSEN ({len(found_not_chosen)}):"
            )
            for case in found_not_chosen:
                actual_ranking = next(
                    (c["rank"] for c in case.top_10_candidates if c["is_actual"]), "N/A"
                )
                print(
                    f"   {case.case_id}: ${case.actual_amount:,.0f} (top 10 rank: {actual_ranking}, all rank: {case.actual_ranking_in_all})"
                )
                print(
                    f"       Chosen: ${case.predicted_amount:,.0f}"
                    if case.predicted_amount
                    else "       Chosen: None"
                )

        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if exact_matches / total_cases >= 0.5:
            print(
                f"   ✅ EXCELLENT: High exact match rate ({exact_matches/total_cases:.1%})"
            )
        elif exact_matches / total_cases >= 0.25:
            print(
                f"   ⚠️ GOOD: Moderate exact match rate ({exact_matches/total_cases:.1%})"
            )
        else:
            print(
                f"   ❌ NEEDS IMPROVEMENT: Low exact match rate ({exact_matches/total_cases:.1%})"
            )

        if actual_found_in_top10 / total_cases >= 0.7:
            print(
                f"   ✅ EXCELLENT: High top-10 coverage ({actual_found_in_top10/total_cases:.1%})"
            )
        elif actual_found_in_top10 / total_cases >= 0.5:
            print(
                f"   ⚠️ GOOD: Moderate top-10 coverage ({actual_found_in_top10/total_cases:.1%})"
            )
        else:
            print(
                f"   ❌ NEEDS IMPROVEMENT: Low top-10 coverage ({actual_found_in_top10/total_cases:.1%})"
            )

        if avg_actual_ranking <= 3:
            print(
                f"   ✅ EXCELLENT: Actual candidates rank high (avg: {avg_actual_ranking:.1f})"
            )
        elif avg_actual_ranking <= 5:
            print(
                f"   ⚠️ GOOD: Actual candidates rank moderately (avg: {avg_actual_ranking:.1f})"
            )
        else:
            print(
                f"   ❌ NEEDS IMPROVEMENT: Actual candidates rank low (avg: {avg_actual_ranking:.1f})"
            )

        if actual_found_anywhere / total_cases >= 0.8:
            print(
                f"   ✅ EXCELLENT: High overall coverage ({actual_found_anywhere/total_cases:.1%})"
            )
        elif actual_found_anywhere / total_cases >= 0.6:
            print(
                f"   ⚠️ GOOD: Moderate overall coverage ({actual_found_anywhere/total_cases:.1%})"
            )
        else:
            print(
                f"   ❌ NEEDS IMPROVEMENT: Low overall coverage ({actual_found_anywhere/total_cases:.1%})"
            )

    def save_detailed_results(self, results: List[SpanComparison], output_path: str):
        """Save detailed results to JSON file."""
        output_data = {
            "summary": {
                "total_cases": len(results),
                "exact_matches": sum(1 for r in results if r.is_exact_match),
                "actual_found_in_top10": sum(
                    1
                    for r in results
                    if any(c["is_actual"] for c in r.top_10_candidates)
                ),
            },
            "detailed_results": [
                {
                    "case_id": r.case_id,
                    "actual_amount": r.actual_amount,
                    "actual_span": r.actual_span,
                    "actual_context": r.actual_context,
                    "predicted_amount": r.predicted_amount,
                    "error": r.error,
                    "is_exact_match": r.is_exact_match,
                    "top_10_candidates": r.top_10_candidates,
                    "chosen_candidate": r.chosen_candidate,
                    "actual_ranking_in_all": r.actual_ranking_in_all,
                    "actual_candidate_details": r.actual_candidate_details,  # Add this field
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    """Run detailed span analysis."""
    analyzer = DetailedSpanAnalyzer()

    # Run detailed analysis
    results = analyzer.run_detailed_analysis()

    # Print summary
    analyzer.print_summary(results)

    # Save results
    analyzer.save_detailed_results(results, "detailed_span_analysis.json")

    print(f"\n🎉 Detailed span analysis completed!")


if __name__ == "__main__":
    main()
