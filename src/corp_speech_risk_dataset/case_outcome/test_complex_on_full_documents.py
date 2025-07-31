#!/usr/bin/env python3
"""
test_complex_on_full_documents.py

Test complex extraction patterns on full case documents to find missing amounts.
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
from extract_cash_amounts_stage1 import (
    extract_complex_amount_candidates,
    compute_eligible_count,
)


class FullDocumentComplexTester:
    """Test complex extraction on full case documents."""

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

    def _scan_full_documents_for_complex_patterns(
        self, case_path: Path, actual_amount: float
    ) -> List[Dict]:
        """Scan all stage1 documents in a case for complex patterns."""
        complex_candidates = []

        for stage1_file in case_path.rglob("*_stage1.jsonl"):
            try:
                with open(stage1_file, "r", encoding="utf8") as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if not text:
                            continue

                        # Test complex extraction on this document
                        candidates = extract_complex_amount_candidates(
                            text, min_amount=100
                        )

                        for candidate in candidates:
                            # Check if this matches the actual amount
                            if abs(candidate["value"] - actual_amount) <= 1.0:
                                complex_candidates.append(
                                    {
                                        **candidate,
                                        "file": str(stage1_file),
                                        "is_exact_match": True,
                                    }
                                )
                            elif (
                                abs(candidate["value"] - actual_amount) / actual_amount
                                <= 0.1
                            ):
                                complex_candidates.append(
                                    {
                                        **candidate,
                                        "file": str(stage1_file),
                                        "is_exact_match": False,
                                    }
                                )

            except Exception as e:
                print(f"Error processing {stage1_file}: {e}")
                continue

        return complex_candidates

    def test_complex_extraction_on_full_documents(self):
        """Test complex extraction on full case documents."""
        df = self._load_gold_standard()

        print("🧪 Testing Complex Extraction on Full Case Documents")
        print("=" * 80)

        missing_cases = []

        for idx, row in df.iterrows():
            case_id = str(row["case_id"])
            actual_amount = float(row["final_amount"])
            actual_span = str(row.get("amount_text", "N/A"))
            actual_context = str(row.get("candidate_text", "N/A"))

            case_path = self._get_case_data_path(case_id)
            if not case_path:
                continue

            # Get standard candidates
            standard_candidates = scan_stage1(
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

            # Check if actual was found in standard candidates
            actual_candidate = self._find_actual_candidate(
                standard_candidates, actual_amount
            )

            if not actual_candidate:
                missing_cases.append(
                    {
                        "case_id": case_id,
                        "actual_amount": actual_amount,
                        "actual_span": actual_span,
                        "actual_context": actual_context,
                        "standard_candidates": len(standard_candidates),
                        "case_path": case_path,
                    }
                )

        print(
            f"\n📊 Found {len(missing_cases)} cases where actual amounts were missing from standard extraction"
        )

        # Test complex extraction on full documents for each missing case
        for case in missing_cases:
            print(f"\n{'='*80}")
            print(f"🔍 Testing: {case['case_id']}")
            print(f"💰 Actual Amount: ${case['actual_amount']:,.0f}")
            print(f"📝 Actual Span: {case['actual_span']}")
            print(f"📄 Actual Context: {case['actual_context'][:200]}...")

            # Scan full documents for complex patterns
            complex_candidates = self._scan_full_documents_for_complex_patterns(
                case["case_path"], case["actual_amount"]
            )

            if complex_candidates:
                print(
                    f"✅ Complex extraction found {len(complex_candidates)} candidates in full documents:"
                )
                for i, candidate in enumerate(complex_candidates, 1):
                    match_type = (
                        "🎯 EXACT MATCH"
                        if candidate["is_exact_match"]
                        else "🎯 CLOSE MATCH"
                    )
                    print(
                        f"   {i}. ${candidate['value']:,.0f} ({candidate['type']}) - {match_type}"
                    )
                    print(f"      Raw: '{candidate['amount']}'")
                    print(f"      File: {candidate['file']}")
                    print(f"      Context: ...{candidate['context'][:100]}...")
            else:
                print("❌ Complex extraction found no candidates in full documents")

            # Also test the eligible count function on the actual context
            try:
                eligible_count = compute_eligible_count(case["actual_context"])
                print(f"   📊 Eligible count: {eligible_count:,}")
            except ValueError as e:
                print(f"   📊 Eligible count: {e}")

        return missing_cases


def main():
    """Run complex extraction test on full documents."""
    tester = FullDocumentComplexTester()

    # Run test
    missing_cases = tester.test_complex_extraction_on_full_documents()

    print(f"\n🎉 Complex extraction on full documents test completed!")
    print(f"📊 Summary: {len(missing_cases)} cases tested")


if __name__ == "__main__":
    main()
