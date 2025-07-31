#!/usr/bin/env python3
"""
analyze_candidate_coverage.py

Analyze if actual amounts are in candidates and identify missing features.
Focus on understanding why actual amounts are missing or not chosen.
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
from case_outcome_imputer import (
    scan_stage1,
    VotingWeights,
    AmountSelector,
    Candidate,
)
from extract_cash_amounts_stage1 import (
    DEFAULT_MIN_AMOUNT,
)


@dataclass
class CaseAnalysis:
    """Analysis result for a single case."""

    case_id: str
    actual_amount: float
    predicted_amount: Optional[float]
    total_candidates: int
    actual_in_candidates: bool
    actual_in_top_5: bool
    actual_votes: Optional[int]
    top_candidate_votes: Optional[int]
    error: Optional[float]
    is_exact_match: bool
    missing_reason: Optional[str]


class CandidateCoverageAnalyzer:
    """Analyze candidate coverage and identify missing features."""

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

        # Print initial stats
        print(f"📊 CSV ANALYSIS:")
        print(f"   Total rows: {len(df)}")
        print(f"   Rows with case_id: {df['case_id'].notna().sum()}")
        print(f"   Rows with final_amount: {df['final_amount'].notna().sum()}")
        print(
            f"   Rows with non-null final_amount: {(df['final_amount'] != 'null').sum()}"
        )

        # Filter out rows with empty case_id or null final_amount
        df = df[df["case_id"].notna() & (df["case_id"] != "")]
        df = df[df["final_amount"].notna() & (df["final_amount"] != "null")]

        # Convert final_amount to numeric, handling commas
        df["final_amount"] = (
            df["final_amount"].astype(str).str.replace(",", "").str.replace("$", "")
        )
        df["final_amount"] = pd.to_numeric(df["final_amount"], errors="coerce")
        df = df.dropna(subset=["final_amount"])

        print(f"   After filtering: {len(df)} valid cases")
        print(f"   Valid cases with amounts:")
        for idx, row in df.iterrows():
            print(f"     {row['case_id']}: ${row['final_amount']:,.0f}")

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

    def analyze_single_case(self, case_id: str, actual_amount: float) -> CaseAnalysis:
        """Analyze a single case for candidate coverage."""
        case_path = self._get_case_data_path(case_id)
        if not case_path:
            return CaseAnalysis(
                case_id=case_id,
                actual_amount=actual_amount,
                predicted_amount=None,
                total_candidates=0,
                actual_in_candidates=False,
                actual_in_top_5=False,
                actual_votes=None,
                top_candidate_votes=None,
                error=None,
                is_exact_match=False,
                missing_reason="Case data not found",
            )

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

        # Find actual candidate
        actual_candidate = self._find_actual_candidate(candidates, actual_amount)
        actual_in_candidates = actual_candidate is not None

        # Check if actual is in top 5
        actual_in_top_5 = False
        if actual_candidate and sorted_candidates:
            actual_in_top_5 = actual_candidate in sorted_candidates[:5]

        # Choose best candidate
        chosen_candidate = self.selector.choose(candidates)
        predicted_amount = chosen_candidate if chosen_candidate else None

        # Calculate error
        error = None
        if predicted_amount:
            error = abs(predicted_amount - actual_amount)

        # Check exact match
        is_exact_match = (
            predicted_amount is not None and abs(predicted_amount - actual_amount) < 1.0
        )

        # Determine why actual might be missing
        missing_reason = None
        if not actual_in_candidates:
            missing_reason = "Actual amount not found in any candidates"
        elif actual_candidate.feature_votes < self.best_params["min_features"]:
            missing_reason = f"Actual found but failed min_features filter ({actual_candidate.feature_votes} < {self.best_params['min_features']})"
        elif not actual_in_top_5:
            missing_reason = f"Actual found but not in top 5 (ranked #{sorted_candidates.index(actual_candidate) + 1})"

        return CaseAnalysis(
            case_id=case_id,
            actual_amount=actual_amount,
            predicted_amount=predicted_amount,
            total_candidates=len(candidates),
            actual_in_candidates=actual_in_candidates,
            actual_in_top_5=actual_in_top_5,
            actual_votes=actual_candidate.feature_votes if actual_candidate else None,
            top_candidate_votes=(
                sorted_candidates[0].feature_votes if sorted_candidates else None
            ),
            error=error,
            is_exact_match=is_exact_match,
            missing_reason=missing_reason,
        )

    def run_analysis(self) -> List[CaseAnalysis]:
        """Run analysis on all cases."""
        df = self._load_gold_standard()
        results = []

        print(f"🔍 Analyzing {len(df)} cases for candidate coverage...")

        for idx, row in df.iterrows():
            case_id = str(row["case_id"])
            actual_amount = float(row["final_amount"])

            result = self.analyze_single_case(case_id, actual_amount)
            results.append(result)

            # Print progress
            status = "✅" if result.actual_in_candidates else "❌"
            print(
                f"   {idx+1:2d}. {status} {case_id}: {result.total_candidates} candidates"
            )

        return results

    def analyze_results(self, results: List[CaseAnalysis]) -> Dict:
        """Analyze results to identify patterns and missing features."""

        # Basic statistics
        total_cases = len(results)
        actual_found = sum(1 for r in results if r.actual_in_candidates)
        actual_in_top_5 = sum(1 for r in results if r.actual_in_top_5)
        exact_matches = sum(1 for r in results if r.is_exact_match)

        # Analyze missing reasons
        missing_reasons = {}
        for result in results:
            if result.missing_reason:
                missing_reasons[result.missing_reason] = (
                    missing_reasons.get(result.missing_reason, 0) + 1
                )

        # Analyze vote patterns
        cases_with_actual = [r for r in results if r.actual_in_candidates]
        vote_analysis = {
            "avg_actual_votes": (
                sum(r.actual_votes for r in cases_with_actual) / len(cases_with_actual)
                if cases_with_actual
                else 0
            ),
            "avg_top_votes": sum(
                r.top_candidate_votes for r in results if r.top_candidate_votes
            )
            / len([r for r in results if r.top_candidate_votes]),
            "actual_vote_range": (
                (
                    min(r.actual_votes for r in cases_with_actual),
                    max(r.actual_votes for r in cases_with_actual),
                )
                if cases_with_actual
                else (0, 0)
            ),
        }

        # Find cases with most/least candidates
        sorted_by_candidates = sorted(
            results, key=lambda r: r.total_candidates, reverse=True
        )
        cases_with_most_candidates = sorted_by_candidates[:5]
        cases_with_least_candidates = sorted_by_candidates[-5:]

        # Find positive vs negative cases for comparison
        positive_cases = [r for r in results if r.is_exact_match]
        negative_cases = [r for r in results if not r.is_exact_match]

        return {
            "total_cases": total_cases,
            "actual_found": actual_found,
            "actual_found_percentage": actual_found / total_cases,
            "actual_in_top_5": actual_in_top_5,
            "actual_in_top_5_percentage": actual_in_top_5 / total_cases,
            "exact_matches": exact_matches,
            "exact_match_percentage": exact_matches / total_cases,
            "missing_reasons": missing_reasons,
            "vote_analysis": vote_analysis,
            "cases_with_most_candidates": cases_with_most_candidates,
            "cases_with_least_candidates": cases_with_least_candidates,
            "positive_cases": positive_cases,
            "negative_cases": negative_cases,
        }

    def suggest_new_features(self, analysis: Dict) -> List[str]:
        """Suggest new features based on analysis."""
        suggestions = []

        # If many actuals are not found at all
        if analysis["actual_found_percentage"] < 0.5:
            suggestions.append(
                "Consider adding features to capture amounts in different formats (e.g., spelled out numbers)"
            )
            suggestions.append("Add features for amounts in tables or structured data")
            suggestions.append(
                "Consider OCR quality features for poorly scanned documents"
            )

        # If actuals are found but not in top 5
        if analysis["actual_in_top_5_percentage"] < analysis["actual_found_percentage"]:
            suggestions.append(
                "Add features for document structure (e.g., headers, footers)"
            )
            suggestions.append(
                "Consider semantic similarity features for amount context"
            )
            suggestions.append(
                "Add features for legal document sections (e.g., damages, settlements)"
            )

        # If actuals have low votes compared to top candidates
        vote_analysis = analysis["vote_analysis"]
        if vote_analysis["avg_actual_votes"] < vote_analysis["avg_top_votes"] * 0.7:
            suggestions.append("Add features for proximity to key legal terms")
            suggestions.append("Consider features for document type classification")
            suggestions.append("Add features for temporal context (dates near amounts)")

        # If many cases have few candidates
        avg_candidates = (
            sum(r.total_candidates for r in analysis["cases_with_most_candidates"]) / 5
        )
        if avg_candidates < 10:
            suggestions.append("Add features for different amount extraction patterns")
            suggestions.append("Consider features for multi-currency amounts")
            suggestions.append(
                "Add features for amounts in different units (millions, thousands)"
            )

        return suggestions

    def print_detailed_analysis(self, results: List[CaseAnalysis], analysis: Dict):
        """Print detailed analysis results."""
        print(f"\n{'='*80}")
        print(f"📊 CANDIDATE COVERAGE ANALYSIS")
        print(f"{'='*80}")

        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total cases: {analysis['total_cases']}")
        print(
            f"   Actual amounts found in candidates: {analysis['actual_found']} ({analysis['actual_found_percentage']:.1%})"
        )
        print(
            f"   Actual amounts in top 5: {analysis['actual_in_top_5']} ({analysis['actual_in_top_5_percentage']:.1%})"
        )
        print(
            f"   Exact matches: {analysis['exact_matches']} ({analysis['exact_match_percentage']:.1%})"
        )

        print(f"\n🔍 VOTE ANALYSIS:")
        vote_analysis = analysis["vote_analysis"]
        print(f"   Average actual votes: {vote_analysis['avg_actual_votes']:.1f}")
        print(f"   Average top candidate votes: {vote_analysis['avg_top_votes']:.1f}")
        print(
            f"   Actual vote range: {vote_analysis['actual_vote_range'][0]} - {vote_analysis['actual_vote_range'][1]}"
        )

        print(f"\n❌ MISSING REASONS:")
        for reason, count in analysis["missing_reasons"].items():
            print(f"   {reason}: {count} cases")

        print(f"\n🏆 CASES WITH MOST CANDIDATES:")
        for i, case in enumerate(analysis["cases_with_most_candidates"], 1):
            status = "✅" if case.actual_in_candidates else "❌"
            print(
                f"   {i}. {case.case_id}: {case.total_candidates} candidates {status}"
            )

        print(f"\n⚠️  CASES WITH LEAST CANDIDATES:")
        for i, case in enumerate(analysis["cases_with_least_candidates"], 1):
            status = "✅" if case.actual_in_candidates else "❌"
            print(
                f"   {i}. {case.case_id}: {case.total_candidates} candidates {status}"
            )

        # Compare positive vs negative cases
        print(f"\n🎯 POSITIVE VS NEGATIVE CASE COMPARISON:")
        positive = analysis["positive_cases"]
        negative = analysis["negative_cases"]

        if positive:
            avg_positive_candidates = sum(r.total_candidates for r in positive) / len(
                positive
            )
            avg_positive_votes = sum(
                r.actual_votes for r in positive if r.actual_votes
            ) / len([r for r in positive if r.actual_votes])
            print(
                f"   Positive cases ({len(positive)}): avg {avg_positive_candidates:.1f} candidates, avg {avg_positive_votes:.1f} votes"
            )

        if negative:
            avg_negative_candidates = sum(r.total_candidates for r in negative) / len(
                negative
            )
            avg_negative_votes = sum(
                r.actual_votes for r in negative if r.actual_votes
            ) / len([r for r in negative if r.actual_votes])
            print(
                f"   Negative cases ({len(negative)}): avg {avg_negative_candidates:.1f} candidates, avg {avg_negative_votes:.1f} votes"
            )

        # Suggest new features
        suggestions = self.suggest_new_features(analysis)
        if suggestions:
            print(f"\n💡 SUGGESTED NEW FEATURES:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")

    def save_results(
        self, results: List[CaseAnalysis], analysis: Dict, output_path: str
    ):
        """Save results to JSON file."""
        output_data = {
            "analysis": analysis,
            "detailed_results": [
                {
                    "case_id": r.case_id,
                    "actual_amount": r.actual_amount,
                    "predicted_amount": r.predicted_amount,
                    "total_candidates": r.total_candidates,
                    "actual_in_candidates": r.actual_in_candidates,
                    "actual_in_top_5": r.actual_in_top_5,
                    "actual_votes": r.actual_votes,
                    "top_candidate_votes": r.top_candidate_votes,
                    "error": r.error,
                    "is_exact_match": r.is_exact_match,
                    "missing_reason": r.missing_reason,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Run the candidate coverage analysis."""
    analyzer = CandidateCoverageAnalyzer()

    # Run analysis
    results = analyzer.run_analysis()

    # Analyze results
    analysis = analyzer.analyze_results(results)

    # Print detailed analysis
    analyzer.print_detailed_analysis(results, analysis)

    # Save results
    analyzer.save_results(results, analysis, "candidate_coverage_analysis.json")

    print(f"\n🎉 Analysis completed!")


if __name__ == "__main__":
    main()
