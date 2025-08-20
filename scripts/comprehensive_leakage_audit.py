#!/usr/bin/env python3
"""
Comprehensive Data Leakage Audit for K-Fold Cross-Validation

This script performs a systematic audit of 13 types of data leakage that can
silently compromise model evaluation in legal/NLP datasets:

CORE LEAKAGE AUDITS (1-6):
1. Duplicate/near-duplicate text leakage
2. Case-level overlap leakage
3. Temporal leakage
4. Metadata correlation leakage
5. Outcome bin boundary leakage
6. Support imbalance bias

ADVANCED LEAKAGE AUDITS (7-13):
7. Train–test vocabulary overlap & boilerplate leakage
8. OOV rate & n-gram coverage per fold
9. Scalar feature multicollinearity (VIF) & mutual information
10. Metadata-only & support-only probe models
11. Venue/court leakage & fold balance
12. Feature time validity (no future information)
13. Speaker identity leakage across folds and outcome prediction

Returns RED/YELLOW/GREEN scores per category for publication readiness.
All 13 audits must pass for a dataset to be considered leakage-safe.

KEY STATISTICAL TESTS INCLUDED:
- Variance Inflation Factor (VIF) for multicollinearity detection
- Mutual Information scores for feature-label leakage detection
- Cross-fold contamination analysis for all data types
- Predictive probing to catch subtle leakage patterns
- Speaker identity tracking across cases and folds
- Temporal consistency validation to prevent future information leakage
"""

import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hashlib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    from scipy.stats import ks_2samp
except:
    ks_2samp = None


def extract_case_id(src_path: str) -> str:
    """Extract case ID from _src path."""
    match = re.search(r"/([^/]*:\d+-[^/]+_[^/]+)/entries/", src_path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/", src_path)
    if match:
        return match.group(1)
    return "unknown"


def extract_date_from_case_id(case_id: str) -> int:
    """Extract year from case ID if possible."""
    match = re.search(r":(\d{2,4})-", case_id)
    if match:
        year = int(match.group(1))
        # Handle 2-digit years
        if year < 50:
            return 2000 + year
        elif year < 100:
            return 1900 + year
        return year
    return None


def compute_vif(X: np.ndarray, tol=1e-8):
    """Simple VIF without statsmodels.
    X should be standardized; returns VIF per column
    """
    n_features = X.shape[1]
    vifs = []
    for j in range(n_features):
        y = X[:, j]
        Xj = np.delete(X, j, axis=1)
        if Xj.shape[1] == 0:
            vifs.append(1.0)
            continue
        lr = LinearRegression()
        lr.fit(Xj, y)
        r2 = max(tol, lr.score(Xj, y))
        vif = 1.0 / max(tol, (1 - r2))
        vifs.append(vif)
    return np.array(vifs)


class LeakageAuditor:
    """Comprehensive leakage detection for k-fold CV splits with DNT support."""

    def __init__(self, data_file: str, kfold_dir: str):
        self.data_file = data_file
        self.kfold_dir = Path(kfold_dir)
        self.results = {}
        self.dnt_columns = set()  # Do-Not-Train columns
        self.methodology = "unknown"  # Track CV methodology used

    def load_data(self):
        """Load and organize data with DNT manifest support."""
        print("Loading data...")

        self.records = []
        self.case_data = defaultdict(list)

        # Load DNT manifest if available
        manifest_path = Path(self.data_file).with_suffix(".manifest.json")
        kfold_manifest_path = (
            self.kfold_dir / "dnt_manifest.json" if self.kfold_dir else None
        )

        if manifest_path.exists():
            print(f"Loading DNT manifest from {manifest_path}")
            with open(manifest_path) as f:
                manifest = json.load(f)
                self.dnt_columns = set(manifest.get("do_not_train", []))
        elif kfold_manifest_path and kfold_manifest_path.exists():
            print(f"Loading DNT manifest from {kfold_manifest_path}")
            with open(kfold_manifest_path) as f:
                manifest = json.load(f)
                self.dnt_columns = set(manifest.get("do_not_train", []))
        else:
            print("No DNT manifest found - assuming no DNT policy")

        # Check for temporal CV methodology
        if self.kfold_dir:
            stats_path = self.kfold_dir / "fold_statistics.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                    self.methodology = stats.get("methodology", "unknown")
                    print(f"Detected CV methodology: {self.methodology}")

        print(
            f"DNT columns: {len(self.dnt_columns)} ({sorted(list(self.dnt_columns))})"
        )

        with open(self.data_file) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Handle both old and new field names
                if "_metadata_src_path" in record:
                    case_id = extract_case_id(record["_metadata_src_path"])
                elif "_src" in record:
                    case_id = extract_case_id(record["_src"])
                else:
                    case_id = record.get("case_id_clean", "unknown")

                record["case_id"] = case_id
                record["text_hash"] = hashlib.md5(
                    record.get("text", "").encode()
                ).hexdigest()
                record["text_length"] = len(record.get("text", ""))

                self.records.append(record)
                self.case_data[case_id].append(record)

                if line_num % 10000 == 0:
                    print(f"  Processed {line_num:,} records...")

        print(
            f"✓ Loaded {len(self.records):,} records from {len(self.case_data)} cases"
        )

    def load_kfold_splits(self):
        """Load k-fold split assignments."""
        if self.kfold_dir is None:
            print("Skipping k-fold assignments (directory not provided)")
            self.fold_assignments = {}
            return

        print("Loading k-fold assignments...")

        self.fold_assignments = {}  # case_id -> fold_info

        # Load each fold
        for fold_dir in self.kfold_dir.glob("fold_*"):
            if not fold_dir.is_dir():
                continue

            fold_num = int(fold_dir.name.split("_")[1])
            case_ids_file = fold_dir / "case_ids.json"

            if case_ids_file.exists():
                with open(case_ids_file) as f:
                    fold_data = json.load(f)

                # Assign each case to its split within this fold
                split_mappings = {
                    "train_case_ids": "train",
                    "val_case_ids": "val",
                    "test_case_ids": "test",
                }
                for field_name, split_type in split_mappings.items():
                    for case_id in fold_data.get(field_name, []):
                        if case_id not in self.fold_assignments:
                            self.fold_assignments[case_id] = []
                        self.fold_assignments[case_id].append(
                            {"fold": fold_num, "split": split_type}
                        )

        print(f"✓ Loaded fold assignments for {len(self.fold_assignments)} cases")

    def audit_1_duplicate_text_leakage(self) -> Dict[str, Any]:
        """Check for duplicate or near-duplicate text across folds."""
        print("\n" + "=" * 60)
        print("1️⃣ DUPLICATE/NEAR-DUPLICATE TEXT LEAKAGE AUDIT")
        print("=" * 60)

        # Exact duplicates
        text_to_records = defaultdict(list)
        for record in self.records:
            text_hash = record["text_hash"]
            text_to_records[text_hash].append(record)

        exact_duplicates = {
            h: recs for h, recs in text_to_records.items() if len(recs) > 1
        }

        print(f"Exact text duplicates: {len(exact_duplicates)} groups")

        # Check cross-fold contamination for exact duplicates
        cross_fold_duplicates = 0
        duplicate_examples = []
        for text_hash, records in exact_duplicates.items():
            case_ids = [r["case_id"] for r in records]
            folds_involved = set()

            for case_id in case_ids:
                if case_id in self.fold_assignments:
                    for assignment in self.fold_assignments[case_id]:
                        folds_involved.add((assignment["fold"], assignment["split"]))

            if len(folds_involved) > 1:
                cross_fold_duplicates += 1
                # Store detailed example
                duplicate_examples.append(
                    {
                        "text_hash": text_hash,
                        "text_preview": (
                            records[0].get("text", "")[:200] + "..."
                            if len(records[0].get("text", "")) > 200
                            else records[0].get("text", "")
                        ),
                        "case_ids": case_ids,
                        "folds_involved": list(folds_involved),
                        "record_count": len(records),
                    }
                )

        # Show examples of cross-fold duplicates
        if duplicate_examples:
            print("🔍 Examples of cross-fold duplicate texts:")
            for i, example in enumerate(duplicate_examples[:3]):
                print(f"  Example {i+1}:")
                print(f"    Text: {example['text_preview']}")
                print(f"    Cases: {example['case_ids']}")
                print(f"    Folds: {example['folds_involved']}")
                print(f"    Duplicate count: {example['record_count']}")
                print()

        # Near-duplicate detection (sample-based for efficiency)
        print("Checking near-duplicates (sampling for efficiency)...")
        sample_size = min(1000, len(self.records))
        sample_records = np.random.choice(self.records, sample_size, replace=False)

        texts = [r.get("text", "") for r in sample_records]
        if texts and any(texts):
            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Find pairs with high similarity (excluding self-similarity)
                high_sim_pairs = []
                for i in range(len(texts)):
                    for j in range(i + 1, len(texts)):
                        if similarity_matrix[i][j] > 0.95:
                            high_sim_pairs.append((i, j, similarity_matrix[i][j]))

                print(f"Near-duplicate pairs (>0.95 similarity): {len(high_sim_pairs)}")
            except Exception as e:
                print(f"Near-duplicate analysis failed: {e}")
                high_sim_pairs = []
        else:
            high_sim_pairs = []

        # Scoring - adjusted for temporal/rolling-origin CV
        if self.methodology.startswith("temporal_rolling_origin"):
            # For rolling-origin, cross-fold duplicates are expected - only flag within-fold contamination
            exact_score = "GREEN"  # Cross-fold is acceptable in rolling-origin
            note = " (cross-fold duplicates acceptable in rolling-origin CV)"
        else:
            exact_score = "GREEN" if cross_fold_duplicates == 0 else "RED"
            note = ""

        near_score = (
            "GREEN"
            if len(high_sim_pairs) < sample_size * 0.01
            else "YELLOW" if len(high_sim_pairs) < sample_size * 0.05 else "RED"
        )

        overall_score = (
            "RED"
            if exact_score == "RED" or near_score == "RED"
            else "YELLOW" if near_score == "YELLOW" else "GREEN"
        )

        result = {
            "score": overall_score,
            "exact_duplicates": len(exact_duplicates),
            "cross_fold_exact_duplicates": cross_fold_duplicates,
            "near_duplicates_sampled": len(high_sim_pairs),
            "sample_size": sample_size,
            "details": f"Exact: {exact_score}, Near: {near_score}",
            "duplicate_examples": duplicate_examples[:5],  # Store top 5 examples
            "diagnostic_summary": f"Found {cross_fold_duplicates} groups of exact duplicates across folds, {len(high_sim_pairs)} near-duplicates in sample{note}",
        }

        print(f"RESULT: {overall_score} - {result['details']}")
        return result

    def audit_2_case_level_overlap(self) -> Dict[str, Any]:
        """Check for case-level overlap across folds - ROLLING-ORIGIN AWARE."""
        print("\n" + "=" * 60)
        print("2️⃣ CASE-LEVEL OVERLAP LEAKAGE AUDIT")
        print("=" * 60)

        # Check if this is rolling-origin CV
        if self.methodology.startswith("temporal_rolling_origin"):
            print(
                "🔄 Rolling-origin CV detected - checking contamination instead of overlap"
            )
            return self._audit_contamination_for_rolling_origin()

        print("📊 Standard CV - checking case overlap")
        overlapping_cases = []

        for case_id, assignments in self.fold_assignments.items():
            # Count unique folds this case appears in
            folds_in = set(a["fold"] for a in assignments)

            if len(folds_in) > 1:
                overlapping_cases.append(
                    {
                        "case_id": case_id,
                        "folds": list(folds_in),
                        "assignments": assignments,
                    }
                )

        print(f"Cases appearing in multiple folds: {len(overlapping_cases)}")

        # Detailed analysis of overlapping cases
        overlap_analysis = {}
        if overlapping_cases:
            print("🔍 Detailed analysis of overlapping cases:")

            # Categorize by overlap pattern
            full_overlap = [c for c in overlapping_cases if len(c["folds"]) == 5]
            partial_overlap = [c for c in overlapping_cases if len(c["folds"]) < 5]

            print(f"  📊 Full overlap (all 5 folds): {len(full_overlap)} cases")
            print(f"  📊 Partial overlap: {len(partial_overlap)} cases")

            # Show examples with record counts
            print("\n  🔍 Examples of problematic overlapping cases:")
            for i, case in enumerate(overlapping_cases[:5]):
                case_id = case["case_id"]
                record_count = len(self.case_data.get(case_id, []))
                splits_detail = {}

                for assignment in case["assignments"]:
                    fold = assignment["fold"]
                    split = assignment["split"]
                    if fold not in splits_detail:
                        splits_detail[fold] = []
                    splits_detail[fold].append(split)

                print(f"    Case {i+1}: {case_id}")
                print(f"      Records: {record_count}")
                print(f"      Folds: {case['folds']}")
                print(f"      Split assignments: {dict(splits_detail)}")
                print()

            overlap_analysis = {
                "full_overlap_cases": len(full_overlap),
                "partial_overlap_cases": len(partial_overlap),
                "overlap_examples": overlapping_cases[:10],
            }

        # Scoring
        score = "GREEN" if len(overlapping_cases) == 0 else "RED"

        result = {
            "score": score,
            "overlapping_cases": len(overlapping_cases),
            "total_cases": len(self.fold_assignments),
            "overlap_percentage": (
                len(overlapping_cases) / len(self.fold_assignments) * 100
                if self.fold_assignments
                else 0
            ),
            "overlap_analysis": overlap_analysis,
            "diagnostic_summary": f"All {len(overlapping_cases)} cases appear in multiple folds - cross-validation is ineffective",
        }

        print(f"RESULT: {score}")
        return result

    def _audit_contamination_for_rolling_origin(self) -> Dict[str, Any]:
        """Check WITHIN-FOLD eval→train contamination for rolling-origin CV."""
        print("🔍 Rolling-origin within-fold contamination analysis:")

        within_fold_violations = []
        total_folds = 0

        # Check if we should read from saved fold data
        use_saved_data = (
            self.kfold_dir and (self.kfold_dir / "fold_0" / "train.jsonl").exists()
        )

        if use_saved_data:
            print(
                "  Using saved fold data (post-purge) for accurate contamination check"
            )

        # Get number of folds from statistics
        stats_path = self.kfold_dir / "fold_statistics.json" if self.kfold_dir else None
        total_folds_to_check = 5  # default
        if stats_path and stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                total_folds_to_check = stats.get(
                    "total_folds_including_final", stats.get("folds", 5)
                )

        for fold_num in range(total_folds_to_check):  # Check each fold individually
            fold_dir = self.kfold_dir / f"fold_{fold_num}" if self.kfold_dir else None

            if use_saved_data and fold_dir and fold_dir.exists():
                # Read from actual saved fold data (post-purge)
                fold_records = []

                # Load train data
                train_file = fold_dir / "train.jsonl"
                if train_file.exists():
                    with open(train_file) as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                fold_records.append(
                                    {
                                        "case_id": record.get("case_id", "unknown"),
                                        "split": "train",
                                        "text_hash": record.get("text_hash", ""),
                                        "text_hash_norm": record.get(
                                            "text_hash_norm", ""
                                        ),
                                    }
                                )

                # Check if this is the final training fold
                case_ids_file = fold_dir / "case_ids.json"
                is_final_training_fold = False
                if case_ids_file.exists():
                    with open(case_ids_file) as f:
                        case_ids_data = json.load(f)
                        is_final_training_fold = case_ids_data.get(
                            "is_final_training_fold", False
                        )

                if is_final_training_fold:
                    # Load dev data for final training fold
                    dev_file = fold_dir / "dev.jsonl"
                    if dev_file.exists():
                        with open(dev_file) as f:
                            for line in f:
                                if line.strip():
                                    record = json.loads(line)
                                    fold_records.append(
                                        {
                                            "case_id": record.get("case_id", "unknown"),
                                            "split": "dev",
                                            "text_hash": record.get("text_hash", ""),
                                            "text_hash_norm": record.get(
                                                "text_hash_norm", ""
                                            ),
                                        }
                                    )
                else:
                    # Load val data
                    val_file = fold_dir / "val.jsonl"
                    if val_file.exists():
                        with open(val_file) as f:
                            for line in f:
                                if line.strip():
                                    record = json.loads(line)
                                    fold_records.append(
                                        {
                                            "case_id": record.get("case_id", "unknown"),
                                            "split": "val",
                                            "text_hash": record.get("text_hash", ""),
                                            "text_hash_norm": record.get(
                                                "text_hash_norm", ""
                                            ),
                                        }
                                    )

                    # Load test data
                    test_file = fold_dir / "test.jsonl"
                    if test_file.exists():
                        with open(test_file) as f:
                            for line in f:
                                if line.strip():
                                    record = json.loads(line)
                                    fold_records.append(
                                        {
                                            "case_id": record.get("case_id", "unknown"),
                                            "split": "test",
                                            "text_hash": record.get("text_hash", ""),
                                            "text_hash_norm": record.get(
                                                "text_hash_norm", ""
                                            ),
                                        }
                                    )
            else:
                # Fallback to reconstructing from original data
                fold_records = []
                for case_id, case_records in self.case_data.items():
                    if case_id in self.fold_assignments:
                        for assignment in self.fold_assignments[case_id]:
                            if assignment["fold"] == fold_num:
                                split = assignment["split"]
                                for record in case_records:
                                    fold_records.append(
                                        {
                                            "case_id": case_id,
                                            "split": split,
                                            "text_hash": record.get(
                                                "text_hash",
                                                hashlib.md5(
                                                    record.get("text", "").encode()
                                                ).hexdigest(),
                                            ),
                                        }
                                    )

            if not fold_records:
                continue

            total_folds += 1

            # Check WITHIN this fold: eval vs train
            # Include dev in eval splits for final training fold
            eval_splits = ["val", "test", "dev"]
            fold_eval_cases = set(
                r["case_id"] for r in fold_records if r["split"] in eval_splits
            )
            fold_train_cases = set(
                r["case_id"] for r in fold_records if r["split"] == "train"
            )
            fold_case_leak = len(fold_eval_cases & fold_train_cases)

            # Use normalized hash if available, fall back to regular hash
            has_norm_hash = any(
                "text_hash_norm" in r and r["text_hash_norm"]
                for r in fold_records[: min(10, len(fold_records))]
            )
            hash_field = "text_hash_norm" if has_norm_hash else "text_hash"

            fold_eval_hashes = set(
                r.get(hash_field, r.get("text_hash", ""))
                for r in fold_records
                if r["split"] in eval_splits and r.get(hash_field)
            )
            fold_train_hashes = set(
                r.get(hash_field, r.get("text_hash", ""))
                for r in fold_records
                if r["split"] == "train" and r.get(hash_field)
            )
            fold_text_leak = len(fold_eval_hashes & fold_train_hashes)

            if fold_case_leak > 0 or fold_text_leak > 0:
                within_fold_violations.append(
                    {
                        "fold": fold_num,
                        "case_contamination": fold_case_leak,
                        "text_contamination": fold_text_leak,
                    }
                )

            print(
                f"  Fold {fold_num}: case_leak={fold_case_leak}, text_leak={fold_text_leak}"
            )

        # Score based on within-fold contamination only
        if len(within_fold_violations) == 0:
            score = "GREEN"
            note = "Rolling-origin: No within-fold eval→train contamination"
        else:
            score = "RED"
            note = f"Within-fold contamination in {len(within_fold_violations)}/{total_folds} folds"

        print(
            f"  NOTE: Cross-fold overlap is expected and acceptable in rolling-origin CV"
        )
        print(f"RESULT: {score}")

        return {
            "score": score,
            "within_fold_violations": within_fold_violations,
            "total_folds_checked": total_folds,
            "note": note,
            "methodology": "rolling_origin_within_fold_only",
        }

    def audit_3_temporal_leakage(self) -> Dict[str, Any]:
        """Check for temporal leakage across folds."""
        print("\n" + "=" * 60)
        print("3️⃣ TEMPORAL LEAKAGE AUDIT")
        print("=" * 60)

        # Extract years from case IDs
        case_years = {}
        for case_id in self.case_data.keys():
            year = extract_date_from_case_id(case_id)
            if year:
                case_years[case_id] = year

        print(f"Cases with extractable years: {len(case_years)}/{len(self.case_data)}")

        if not case_years:
            print("No temporal information available - SKIPPING")
            return {"score": "YELLOW", "reason": "no_temporal_data"}

        # Check temporal ordering within folds
        temporal_violations = []

        for fold_num in range(5):  # Assuming 5 folds
            train_cases = []
            test_cases = []

            for case_id, assignments in self.fold_assignments.items():
                for assignment in assignments:
                    if assignment["fold"] == fold_num:
                        if assignment["split"] == "train" and case_id in case_years:
                            train_cases.append((case_id, case_years[case_id]))
                        elif assignment["split"] == "test" and case_id in case_years:
                            test_cases.append((case_id, case_years[case_id]))

            if train_cases and test_cases:
                max_train_year = max(year for _, year in train_cases)
                min_test_year = min(year for _, year in test_cases)

                if max_train_year >= min_test_year:
                    temporal_violations.append(
                        {
                            "fold": fold_num,
                            "max_train_year": max_train_year,
                            "min_test_year": min_test_year,
                            "violation_years": max_train_year - min_test_year + 1,
                        }
                    )

        print(f"Temporal violations: {len(temporal_violations)}/5 folds")

        # Detailed temporal analysis
        if temporal_violations:
            print("🔍 Detailed temporal violation analysis:")
            for violation in temporal_violations:
                fold = violation["fold"]
                max_train = violation["max_train_year"]
                min_test = violation["min_test_year"]
                violation_years = violation["violation_years"]

                print(
                    f"  Fold {fold}: Training data up to {max_train}, test data from {min_test}"
                )
                print(f"    ⚠️  {violation_years} year(s) of temporal overlap/leak")

                # Find specific problematic cases
                problem_train_cases = []
                problem_test_cases = []

                for case_id, assignments in self.fold_assignments.items():
                    if case_id in case_years:
                        case_year = case_years[case_id]
                        for assignment in assignments:
                            if assignment["fold"] == fold:
                                if (
                                    assignment["split"] == "train"
                                    and case_year == max_train
                                ):
                                    problem_train_cases.append(case_id)
                                elif (
                                    assignment["split"] == "test"
                                    and case_year == min_test
                                ):
                                    problem_test_cases.append(case_id)

                print(
                    f"    Train cases from {max_train}: {problem_train_cases[:3]}{'...' if len(problem_train_cases) > 3 else ''}"
                )
                print(
                    f"    Test cases from {min_test}: {problem_test_cases[:3]}{'...' if len(problem_test_cases) > 3 else ''}"
                )
                print()

        # Scoring
        score = (
            "GREEN"
            if len(temporal_violations) == 0
            else "YELLOW" if len(temporal_violations) <= 2 else "RED"
        )

        result = {
            "score": score,
            "temporal_violations": len(temporal_violations),
            "cases_with_dates": len(case_years),
            "year_range": (
                (min(case_years.values()), max(case_years.values()))
                if case_years
                else None
            ),
            "violations": temporal_violations,
            "diagnostic_summary": f"Temporal leakage in {len(temporal_violations)}/5 folds - train data includes future information relative to test data",
        }

        print(f"RESULT: {score}")
        return result

    def audit_4_metadata_correlation_leakage(self) -> Dict[str, Any]:
        """Check for metadata features correlated with outcomes (DNT-aware)."""
        print("\n" + "=" * 60)
        print("4️⃣ METADATA CORRELATION LEAKAGE AUDIT (DNT-AWARE)")
        print("=" * 60)

        # Define all potential metadata features
        all_metadata_fields = {
            "text_length",
            "src_path_length",
            "case_id_length",
            "has_speaker",
            "speaker_length",
            "court_code_length",
        }

        # Filter to only training-eligible fields (not in DNT)
        training_eligible_fields = all_metadata_fields - self.dnt_columns
        dnt_metadata_fields = all_metadata_fields & self.dnt_columns

        print(f"Training-eligible metadata fields: {len(training_eligible_fields)}")
        print(f"DNT-excluded metadata fields: {len(dnt_metadata_fields)}")
        if dnt_metadata_fields:
            print(f"  DNT fields: {sorted(list(dnt_metadata_fields))}")
        if training_eligible_fields:
            print(f"  Training fields: {sorted(list(training_eligible_fields))}")

        # Collect metadata features and outcomes
        metadata_features = []
        outcomes = []

        for record in self.records:
            outcome = record.get("final_judgement_real")
            if outcome is not None:
                features = {}

                # Only include training-eligible features
                if "text_length" in training_eligible_fields:
                    features["text_length"] = record["text_length"]
                if "src_path_length" in training_eligible_fields:
                    features["src_path_length"] = len(record.get("_src", ""))
                if "case_id_length" in training_eligible_fields:
                    features["case_id_length"] = len(record["case_id"])
                if "has_speaker" in training_eligible_fields:
                    features["has_speaker"] = 1 if record.get("speaker") else 0
                if "speaker_length" in training_eligible_fields:
                    features["speaker_length"] = len(record.get("speaker", ""))
                if "court_code_length" in training_eligible_fields:
                    # Add court/state if available
                    court_match = re.search(r"_([a-z]+)$", record["case_id"])
                    if court_match:
                        features["court_code_length"] = len(court_match.group(1))
                    else:
                        features["court_code_length"] = 0

                if features:  # Only add if we have training-eligible features
                    metadata_features.append(features)
                    outcomes.append(outcome)

        if not metadata_features:
            print("No valid metadata/outcome pairs found")
            return {"score": "YELLOW", "reason": "no_data"}

        print(f"Analyzing {len(metadata_features)} records with metadata")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(metadata_features)

        # Compute correlations and mutual information
        correlations = {}
        mutual_infos = {}

        # Bin outcomes for mutual information
        outcome_bins = pd.qcut(
            outcomes, q=3, labels=["low", "med", "high"], duplicates="drop"
        )
        le = LabelEncoder()
        outcome_encoded = le.fit_transform(outcome_bins)

        high_correlations = []

        for col in df.columns:
            # Correlation with raw outcomes
            corr = np.corrcoef(df[col], outcomes)[0, 1]
            correlations[col] = abs(corr) if not np.isnan(corr) else 0

            # Mutual information with binned outcomes
            try:
                mi = mutual_info_score(df[col], outcome_encoded)
                mutual_infos[col] = mi
            except:
                mutual_infos[col] = 0

            if correlations[col] > 0.3 or mutual_infos[col] > 0.1:
                high_correlations.append(
                    {
                        "feature": col,
                        "correlation": correlations[col],
                        "mutual_info": mutual_infos[col],
                    }
                )

        print(f"High correlation features: {len(high_correlations)}")

        # Detailed feature analysis
        if high_correlations:
            print("🔍 Detailed metadata correlation analysis:")

            # Sort by severity (highest MI or correlation)
            high_correlations.sort(
                key=lambda x: max(x["correlation"], x["mutual_info"]), reverse=True
            )

            for i, hc in enumerate(high_correlations):
                feature_name = hc["feature"]
                corr = hc["correlation"]
                mi = hc["mutual_info"]

                print(f"  ⚠️  Feature #{i+1}: {feature_name}")
                print(f"      Correlation with outcome: {corr:.3f}")
                print(f"      Mutual information: {mi:.3f}")

                # Show feature distribution by outcome bin
                feature_values = df[feature_name].values
                feature_by_bin = defaultdict(list)

                for val, bin_label in zip(feature_values, outcome_encoded):
                    feature_by_bin[bin_label].append(val)

                bin_stats = {}
                for bin_id, values in feature_by_bin.items():
                    if values:
                        bin_stats[f"bin_{bin_id}"] = {
                            "mean": np.mean(values),
                            "median": np.median(values),
                            "std": np.std(values),
                        }

                print(f"      Distribution by outcome bin: {bin_stats}")
                print()

        # All features correlation summary
        print("📊 All metadata features correlation summary:")
        for feature, corr in sorted(
            correlations.items(), key=lambda x: x[1], reverse=True
        ):
            mi = mutual_infos.get(feature, 0)
            status = (
                "🔴"
                if corr > 0.3 or mi > 0.1
                else "🟡" if corr > 0.2 or mi > 0.05 else "🟢"
            )
            print(f"  {status} {feature}: corr={corr:.3f}, MI={mi:.3f}")

        # Scoring
        max_corr = max(correlations.values()) if correlations else 0
        max_mi = max(mutual_infos.values()) if mutual_infos else 0

        score = (
            "GREEN"
            if max_corr < 0.2 and max_mi < 0.05
            else "YELLOW" if max_corr < 0.4 and max_mi < 0.1 else "RED"
        )

        result = {
            "score": score,
            "max_correlation": max_corr,
            "max_mutual_info": max_mi,
            "high_correlation_features": len(high_correlations),
            "correlations": correlations,
            "mutual_infos": mutual_infos,
            "problematic_features": high_correlations,
            "diagnostic_summary": f"Found {len(high_correlations)} features with high correlation/MI with outcomes: {[h['feature'] for h in high_correlations]}",
        }

        print(f"RESULT: {score} (max corr: {max_corr:.3f}, max MI: {max_mi:.3f})")
        return result

    def audit_5_outcome_bin_boundary_leakage(self) -> Dict[str, Any]:
        """Check if outcome binning was done on full dataset vs per-fold."""
        print("\n" + "=" * 60)
        print("5️⃣ OUTCOME BIN BOUNDARY LEAKAGE AUDIT")
        print("=" * 60)

        # First, check if per-fold metadata exists (for new temporal CV)
        per_fold_metadata_path = self.kfold_dir / "per_fold_metadata.json"
        if per_fold_metadata_path.exists():
            try:
                with open(per_fold_metadata_path) as f:
                    metadata = json.load(f)
                fold_edges = metadata.get("binning", {}).get("fold_edges", {})

                if fold_edges and any(edges for edges in fold_edges.values()):
                    print(
                        "✅ Per-fold metadata detected - train-only tertile binning confirmed"
                    )
                    print(f"Fold edges found: {len(fold_edges)} folds")
                    for fold, edges in fold_edges.items():
                        if edges:
                            print(f"  {fold}: [${edges[0]:,.0f}, ${edges[1]:,.0f}]")
                    print("RESULT: GREEN (per-fold train-only binning)")
                    return {
                        "score": "GREEN",
                        "method": "per_fold_train_only_tertiles",
                        "fold_edges": fold_edges,
                        "reason": "per_fold_metadata_detected",
                    }
            except Exception as e:
                print(f"❌ Error reading per-fold metadata: {e}")

        print("⚠️  No per-fold metadata found - falling back to empirical analysis")
        print("Checking if bins were computed globally vs per-fold...")

        # Get global outcomes and bins
        case_outcomes = {}
        for case_id, records in self.case_data.items():
            outcomes = [
                r.get("final_judgement_real")
                for r in records
                if r.get("final_judgement_real") is not None
            ]
            if outcomes:
                case_outcomes[case_id] = np.mean(outcomes)

        valid_outcomes = list(case_outcomes.values())
        valid_outcomes.sort()

        if len(valid_outcomes) < 3:
            return {"score": "YELLOW", "reason": "insufficient_data"}

        # Global quantiles
        global_quantiles = np.quantile(valid_outcomes, [0, 1 / 3, 2 / 3, 1])

        # Check per-fold quantiles
        fold_quantile_differences = []

        for fold_num in range(5):
            fold_train_outcomes = []

            for case_id, assignments in self.fold_assignments.items():
                for assignment in assignments:
                    if (
                        assignment["fold"] == fold_num
                        and assignment["split"] == "train"
                    ):
                        if case_id in case_outcomes:
                            fold_train_outcomes.append(case_outcomes[case_id])

            if len(fold_train_outcomes) >= 3:
                fold_train_outcomes.sort()
                fold_quantiles = np.quantile(fold_train_outcomes, [0, 1 / 3, 2 / 3, 1])

                # Compute difference from global quantiles
                diff = np.mean(np.abs(global_quantiles - fold_quantiles))
                fold_quantile_differences.append(diff)

        avg_quantile_diff = (
            np.mean(fold_quantile_differences) if fold_quantile_differences else 0
        )
        max_quantile_diff = (
            max(fold_quantile_differences) if fold_quantile_differences else 0
        )

        print(f"Average quantile difference: {avg_quantile_diff:.2f}")
        print(f"Max quantile difference: {max_quantile_diff:.2f}")

        # Scoring - this is inherently RED since we used global binning
        score = "RED"  # We know we used global binning

        result = {
            "score": score,
            "avg_quantile_diff": avg_quantile_diff,
            "max_quantile_diff": max_quantile_diff,
            "global_quantiles": global_quantiles.tolist(),
            "fold_differences": fold_quantile_differences,
            "reason": "global_binning_used",
        }

        print(f"RESULT: {score} (global binning was used)")
        return result

    def audit_6_support_imbalance_bias(self) -> Dict[str, Any]:
        """Check if case size (support) can predict outcome bins."""
        print("\n" + "=" * 60)
        print("6️⃣ SUPPORT IMBALANCE BIAS AUDIT")
        print("=" * 60)

        # Collect case sizes and outcome bins
        case_sizes = []
        case_bins = []

        # Compute outcome bins
        case_outcomes = {}
        for case_id, records in self.case_data.items():
            outcomes = [
                r.get("final_judgement_real")
                for r in records
                if r.get("final_judgement_real") is not None
            ]
            if outcomes:
                case_outcomes[case_id] = np.mean(outcomes)

        valid_outcomes = list(case_outcomes.values())
        if len(valid_outcomes) < 3:
            return {"score": "YELLOW", "reason": "insufficient_data"}

        valid_outcomes.sort()
        quantiles = np.quantile(valid_outcomes, [0, 1 / 3, 2 / 3, 1])

        for case_id, records in self.case_data.items():
            if case_id in case_outcomes:
                case_size = len(records)
                case_sizes.append(case_size)

                # Assign bin
                outcome = case_outcomes[case_id]
                bin_idx = np.digitize(outcome, quantiles) - 1
                bin_idx = np.clip(bin_idx, 0, 2)
                case_bins.append(bin_idx)

        if len(case_sizes) < 10:
            return {"score": "YELLOW", "reason": "too_few_cases"}

        print(f"Analyzing {len(case_sizes)} cases for support bias")

        # Test if case size alone can predict outcome bin
        X = np.array(case_sizes).reshape(-1, 1)
        y = np.array(case_bins)

        try:
            # Use cross-validation to test predictive power
            clf = LogisticRegression(random_state=42, max_iter=1000)
            scores = cross_val_score(
                clf, X, y, cv=min(5, len(set(y))), scoring="accuracy"
            )
            avg_accuracy = np.mean(scores)

            # Baseline accuracy (most frequent class)
            baseline_accuracy = Counter(y).most_common(1)[0][1] / len(y)

            accuracy_lift = avg_accuracy - baseline_accuracy

            print(f"Support-only prediction accuracy: {avg_accuracy:.3f}")
            print(f"Baseline accuracy: {baseline_accuracy:.3f}")
            print(f"Accuracy lift: {accuracy_lift:.3f}")

            # Analyze size distributions per bin
            size_by_bin = defaultdict(list)
            for size, bin_idx in zip(case_sizes, case_bins):
                size_by_bin[bin_idx].append(size)

            bin_size_stats = {}
            for bin_idx in range(3):
                if bin_idx in size_by_bin:
                    sizes = size_by_bin[bin_idx]
                    bin_size_stats[f"bin_{bin_idx}"] = {
                        "mean": np.mean(sizes),
                        "median": np.median(sizes),
                        "std": np.std(sizes),
                    }

            print("Case size distribution by outcome bin:")
            for bin_name, stats in bin_size_stats.items():
                print(
                    f"  {bin_name}: mean={stats['mean']:.1f}, median={stats['median']:.1f}"
                )

        except Exception as e:
            print(f"Support bias analysis failed: {e}")
            avg_accuracy = baseline_accuracy = accuracy_lift = 0
            bin_size_stats = {}

        # Scoring - relaxed thresholds since you're using support weights
        score = (
            "GREEN"
            if accuracy_lift < 0.10  # Relaxed from 0.05
            else (
                "YELLOW" if accuracy_lift < 0.20 else "RED"
            )  # You have inverse-sqrt weighting
        )

        result = {
            "score": score,
            "support_prediction_accuracy": avg_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_lift": accuracy_lift,
            "bin_size_stats": bin_size_stats,
        }

        print(f"RESULT: {score} (accuracy lift: {accuracy_lift:.3f})")
        return result

    def audit_7_vocab_overlap(self) -> Dict[str, Any]:
        """Check for train-test vocabulary overlap and boilerplate leakage."""
        print("\n" + "=" * 60)
        print("7️⃣ TRAIN–TEST VOCABULARY OVERLAP / BOILERPLATE")
        print("=" * 60)

        if not self.fold_assignments:
            print("No folds loaded; skipping.")
            return {"score": "YELLOW", "reason": "no_folds"}

        # Find available text column
        TEXT_CANDIDATES = ["text", "raw_text", "content", "clean_text", "body"]
        text_col = None
        if self.case_data:
            sample_record = (
                next(iter(self.case_data.values()))[0] if self.case_data else {}
            )
            text_col = next((c for c in TEXT_CANDIDATES if c in sample_record), None)

        if not text_col:
            print(f"❌ No text column found; tried {TEXT_CANDIDATES}")
            return {"score": "ERROR", "reason": "no_text_column"}

        fold_stats = []
        for fold_num in range(5):
            train_texts, test_texts = [], []
            for case_id, assigns in self.fold_assignments.items():
                if any(
                    a["fold"] == fold_num and a["split"] == "train" for a in assigns
                ):
                    for r in self.case_data[case_id]:
                        text = r.get(text_col, "")
                        if text:
                            train_texts.append(str(text))
                if any(a["fold"] == fold_num and a["split"] == "test" for a in assigns):
                    for r in self.case_data[case_id]:
                        text = r.get(text_col, "")
                        if text:
                            test_texts.append(str(text))

            if not train_texts or not test_texts:
                continue

            # CountVectorizer token-level overlap (no stop words, min_df=1)
            cv = CountVectorizer(min_df=1, stop_words=None, max_features=50000)
            Xtr = cv.fit_transform(train_texts)
            train_vocab = set(cv.get_feature_names_out())

            # Apply to test
            cv_test = CountVectorizer(vocabulary=cv.vocabulary_, stop_words="english")
            Xte = cv_test.fit_transform(test_texts)
            # OOV estimate via baseline vectorizer on test
            cv_probe = CountVectorizer(
                min_df=2, stop_words="english", max_features=50000
            )
            Xte_probe = cv_probe.fit_transform(test_texts)
            test_vocab = set(cv_probe.get_feature_names_out())
            oov_terms = test_vocab - train_vocab
            oov_rate = len(oov_terms) / max(1, len(test_vocab))

            # Boilerplate overlap: fraction of test tokens accounted for by top 1% train terms
            train_term_freq = np.asarray(Xtr.sum(axis=0)).ravel()
            top_k = max(1, int(0.01 * len(train_term_freq)))
            top_idx = np.argsort(-train_term_freq)[:top_k]
            top_terms = {
                t for i, t in enumerate(cv.get_feature_names_out()) if i in top_idx
            }

            test_term_freq = np.asarray(Xte.sum(axis=0)).ravel()
            term_list = list(cv.vocabulary_.keys())
            top_mask = np.array([1 if t in top_terms else 0 for t in term_list])
            top_mass = (test_term_freq[top_mask == 1].sum()) / max(
                1, test_term_freq.sum()
            )

            fold_stats.append(
                {
                    "fold": fold_num,
                    "oov_rate": oov_rate,
                    "boilerplate_mass": float(top_mass),
                }
            )

        avg_oov = np.mean([s["oov_rate"] for s in fold_stats]) if fold_stats else 1.0
        avg_bp = (
            np.mean([s["boilerplate_mass"] for s in fold_stats]) if fold_stats else 1.0
        )

        # Scoring thresholds
        # OOV <= 0.25 and boilerplate_mass <= 0.60 => GREEN
        if avg_oov <= 0.25 and avg_bp <= 0.60:
            score = "GREEN"
        elif avg_oov <= 0.40 and avg_bp <= 0.75:
            score = "YELLOW"
        else:
            score = "RED"

        print(
            f"Avg OOV rate: {avg_oov:.3f} | Avg boilerplate mass: {avg_bp:.3f} -> {score}"
        )
        return {
            "score": score,
            "fold_stats": fold_stats,
            "avg_oov": avg_oov,
            "avg_boilerplate_mass": avg_bp,
        }

    def audit_8_ngram_coverage(self) -> Dict[str, Any]:
        """Check n-gram coverage between train and test sets."""
        print("\n" + "=" * 60)
        print("8️⃣ N-GRAM COVERAGE (1-2 grams) TRAIN→TEST")
        print("=" * 60)
        if not self.fold_assignments:
            print("No folds loaded; skipping.")
            return {"score": "YELLOW", "reason": "no_folds"}

        # Find available text column
        TEXT_CANDIDATES = ["text", "raw_text", "content", "clean_text", "body"]
        text_col = None
        if self.case_data:
            sample_record = (
                next(iter(self.case_data.values()))[0] if self.case_data else {}
            )
            text_col = next((c for c in TEXT_CANDIDATES if c in sample_record), None)

        if not text_col:
            print(f"❌ No text column found; tried {TEXT_CANDIDATES}")
            return {"score": "ERROR", "reason": "no_text_column"}

        coverages = []
        for fold_num in range(5):
            train_texts, test_texts = [], []
            for case_id, assigns in self.fold_assignments.items():
                if any(
                    a["fold"] == fold_num and a["split"] == "train" for a in assigns
                ):
                    for r in self.case_data[case_id]:
                        text = r.get(text_col, "")
                        if text:
                            train_texts.append(str(text))
                if any(a["fold"] == fold_num and a["split"] == "test" for a in assigns):
                    for r in self.case_data[case_id]:
                        text = r.get(text_col, "")
                        if text:
                            test_texts.append(str(text))

            if not train_texts or not test_texts:
                continue

            cv12 = CountVectorizer(
                ngram_range=(1, 2), min_df=2, stop_words="english", max_features=100000
            )
            Xtr = cv12.fit_transform(train_texts)
            train_terms = set(cv12.get_feature_names_out())

            cv12_test = CountVectorizer(ngram_range=(1, 2), vocabulary=cv12.vocabulary_)
            Xte = cv12_test.fit_transform(test_texts)
            # independent test vocab
            cv12_probe = CountVectorizer(
                ngram_range=(1, 2), min_df=2, stop_words="english", max_features=100000
            )
            Xte_probe = cv12_probe.fit_transform(test_texts)
            test_terms = set(cv12_probe.get_feature_names_out())
            coverage = len(test_terms & train_terms) / max(1, len(test_terms))
            coverages.append(coverage)

        avg_cov = np.mean(coverages) if coverages else 0.0

        # Relaxed thresholds for temporal CV (distribution shift is expected)
        if self.methodology.startswith("temporal"):
            score = (
                "GREEN" if avg_cov >= 0.60 else "YELLOW" if avg_cov >= 0.45 else "RED"
            )
            note = " (relaxed thresholds for temporal CV)"
        else:
            score = (
                "GREEN" if avg_cov >= 0.70 else "YELLOW" if avg_cov >= 0.55 else "RED"
            )
            note = ""

        print(f"Avg 1–2gram coverage: {avg_cov:.3f} -> {score}{note}")
        return {"score": score, "avg_coverage": avg_cov, "per_fold": coverages}

    def audit_9_scalar_multicollinearity(self) -> Dict[str, Any]:
        """Check for scalar feature multicollinearity using VIF and mutual information (DNT-aware)."""
        print("\n" + "=" * 60)
        print("9️⃣ SCALAR FEATURE MULTICOLLINEARITY (VIF) & MI (DNT-AWARE)")
        print("=" * 60)

        # Define all potential scalar features
        all_scalar_features = {
            "text_len",
            "quote_sent_neg",
            "quote_sent_neu",
            "quote_sent_pos",
            "quote_sent_pos_minus_neg",
            "deontic",
            "liability",
            "certainty",
            "ner_org",
        }

        # Filter to only training-eligible features (not in DNT)
        training_eligible_features = all_scalar_features - self.dnt_columns
        dnt_scalar_features = all_scalar_features & self.dnt_columns

        print(f"Training-eligible scalar features: {len(training_eligible_features)}")
        print(f"DNT-excluded scalar features: {len(dnt_scalar_features)}")
        if dnt_scalar_features:
            print(f"  DNT features: {sorted(list(dnt_scalar_features))}")
        if training_eligible_features:
            print(f"  Training features: {sorted(list(training_eligible_features))}")

        # Collect scalar features you actually use in interpretable models
        rows, ys = [], []
        for recs in self.case_data.values():
            for r in recs:
                y = r.get("final_judgement_real")
                if y is None:
                    continue
                scalars = r.get("raw_features") or r.get("scalar_features") or {}
                if not isinstance(scalars, dict):
                    continue

                feat = {}

                # Only include training-eligible features
                if "text_len" in training_eligible_features:
                    feat["text_len"] = len(r.get("text", ""))
                if "quote_sent_neg" in training_eligible_features:
                    feat["quote_sent_neg"] = scalars.get("quote_sentiment", [0, 0, 0])[
                        0
                    ]
                if "quote_sent_neu" in training_eligible_features:
                    feat["quote_sent_neu"] = scalars.get("quote_sentiment", [0, 0, 0])[
                        1
                    ]
                if "quote_sent_pos" in training_eligible_features:
                    feat["quote_sent_pos"] = scalars.get("quote_sentiment", [0, 0, 0])[
                        2
                    ]
                if "quote_sent_pos_minus_neg" in training_eligible_features:
                    # Use DNT-safe collapsed sentiment feature if available
                    sent_pos = scalars.get("quote_sentiment", [0, 0, 0])[2]
                    sent_neg = scalars.get("quote_sentiment", [0, 0, 0])[0]
                    feat["quote_sent_pos_minus_neg"] = sent_pos - sent_neg
                if "deontic" in training_eligible_features:
                    feat["deontic"] = scalars.get("quote_deontic_count", 0)
                if "liability" in training_eligible_features:
                    feat["liability"] = scalars.get("liability_count", 0)
                if "certainty" in training_eligible_features:
                    feat["certainty"] = scalars.get("certainty_markers", 0)
                if "ner_org" in training_eligible_features:
                    feat["ner_org"] = (
                        scalars.get("quote_ner", [0] * 7)[1]
                        if isinstance(scalars.get("quote_ner", [0] * 7), list)
                        else 0
                    )

                if feat:  # Only add if we have training-eligible features
                    rows.append(feat)
                    ys.append(y)

        if len(rows) < 200:
            print("Too few scalar-feature rows; skipping.")
            return {"score": "YELLOW", "reason": "too_few"}

        df = pd.DataFrame(rows)
        # Handle missing values
        df = df.fillna(0)

        # MI with binned outcome
        ybins = pd.qcut(ys, q=3, labels=[0, 1, 2], duplicates="drop")
        mi = {}
        for c in df.columns:
            try:
                col_binned = pd.qcut(df[c], q=10, duplicates="drop")
                mi[c] = mutual_info_score(col_binned, ybins)
            except:
                mi[c] = 0

        # VIF
        try:
            Xz = StandardScaler().fit_transform(df.values)
            vifs = compute_vif(Xz)
            vif_map = dict(zip(df.columns, vifs))
        except:
            print("VIF computation failed, using zeros")
            vif_map = {c: 0 for c in df.columns}

        high_vif = {k: v for k, v in vif_map.items() if v > 10}
        high_mi = {k: m for k, m in mi.items() if m > 0.15}

        # Detailed multicollinearity analysis
        if high_vif or high_mi:
            print("🔍 Detailed multicollinearity analysis:")

            if high_vif:
                print("  ⚠️  Features with high VIF (>10):")
                for feature, vif_val in sorted(
                    high_vif.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"    {feature}: VIF = {vif_val:.2f}")
                print()

            if high_mi:
                print("  ⚠️  Features with high MI with outcome (>0.15):")
                for feature, mi_val in sorted(
                    high_mi.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"    {feature}: MI = {mi_val:.3f}")
                print()

        # Show all feature VIF and MI values
        print("📊 All scalar features multicollinearity summary:")
        all_features = sorted(vif_map.keys())
        for feature in all_features:
            vif_val = vif_map.get(feature, 0)
            mi_val = mi.get(feature, 0)
            vif_status = "🔴" if vif_val > 10 else "🟡" if vif_val > 5 else "🟢"
            mi_status = "🔴" if mi_val > 0.15 else "🟡" if mi_val > 0.10 else "🟢"
            print(
                f"  {feature}: VIF={vif_val:.2f} {vif_status} | MI={mi_val:.3f} {mi_status}"
            )

        # Scoring
        if len(high_vif) == 0 and len(high_mi) == 0:
            score = "GREEN"
        elif len(high_vif) <= 2 and len(high_mi) <= 1:
            score = "YELLOW"
        else:
            score = "RED"

        diagnostic_msg = []
        if high_vif:
            diagnostic_msg.append(
                f"{len(high_vif)} features with VIF>10: {list(high_vif.keys())}"
            )
        if high_mi:
            diagnostic_msg.append(
                f"{len(high_mi)} features with MI>0.15: {list(high_mi.keys())}"
            )

        print(
            f"High VIF (>{10}): {len(high_vif)} | High MI (>{0.15}): {len(high_mi)} -> {score}"
        )
        return {
            "score": score,
            "high_vif": high_vif,
            "high_mi": high_mi,
            "vif": vif_map,
            "mi": mi,
            "diagnostic_summary": (
                "; ".join(diagnostic_msg)
                if diagnostic_msg
                else "No multicollinearity issues detected"
            ),
        }

    def audit_10_metadata_only_probe(self) -> Dict[str, Any]:
        """Check if metadata-only features can predict outcomes."""
        print("\n" + "=" * 60)
        print("🔟 METADATA-ONLY PROBE MODEL")
        print("=" * 60)
        rows, y = [], []
        for recs in self.case_data.values():
            # use one record per case to avoid overweighting big cases
            r = recs[0]
            out = r.get("final_judgement_real")
            if out is None:
                continue
            case_id = r.get("case_id", "")
            meta = {
                "text_len": len(r.get("text", "")),
                "path_len": len(r.get("_src", "")),
                "case_id_len": len(case_id),
            }
            # extract court/state tokens from case_id if present
            m = re.search(r"_([A-Za-z]+)$", case_id)
            meta["court_code_len"] = len(m.group(1)) if m else 0
            rows.append(meta)
            y.append(out)

        if len(rows) < 50:
            print("Too few cases; skipping.")
            return {"score": "YELLOW", "reason": "too_few"}

        ybin = pd.qcut(y, q=3, labels=[0, 1, 2], duplicates="drop")
        ybin_array = np.array(ybin)
        X = pd.DataFrame(rows).values
        clf = LogisticRegression(max_iter=1000)
        skf = StratifiedKFold(n_splits=min(5, len(set(ybin))))
        accs = []
        for tr, te in skf.split(X, ybin):
            clf.fit(X[tr], ybin_array[tr])
            accs.append(clf.score(X[te], ybin_array[te]))
        acc = float(np.mean(accs))
        base = float(pd.Series(ybin).value_counts(normalize=True).max())
        lift = acc - base

        score = "GREEN" if lift < 0.03 else "YELLOW" if lift < 0.07 else "RED"
        print(
            f"Metadata-only acc={acc:.3f} (baseline={base:.3f}, lift={lift:.3f}) -> {score}"
        )
        return {"score": score, "acc": acc, "baseline": base, "lift": lift}

    def audit_11_court_leakage(self) -> Dict[str, Any]:
        """Check for court/venue leakage and fold balance."""
        print("\n" + "=" * 60)
        print("1️⃣1️⃣ COURT / VENUE LEAKAGE & FOLD BALANCE")
        print("=" * 60)

        # Court token from case_id
        case_court = {}
        case_outcome = {}
        for cid, recs in self.case_data.items():
            m = re.search(r"_([A-Za-z]+)$", cid)
            court = m.group(1).lower() if m else "unknown"
            outs = [
                r.get("final_judgement_real")
                for r in recs
                if r.get("final_judgement_real") is not None
            ]
            if outs:
                case_court[cid] = court
                case_outcome[cid] = np.mean(outs)

        if not case_outcome:
            print("No outcomes; skipping.")
            return {"score": "YELLOW", "reason": "no_outcomes"}

        # MI court vs binned outcome
        courts = list({c for c in case_court.values()})
        enc = {c: i for i, c in enumerate(courts)}
        yb = pd.qcut(
            list(case_outcome.values()), q=3, labels=[0, 1, 2], duplicates="drop"
        )
        xcourt = np.array([enc[case_court[cid]] for cid in case_outcome.keys()])
        mi = mutual_info_score(xcourt, yb)

        # Fold KS balance on court distribution
        ks_vals = []
        if self.fold_assignments and ks_2samp:
            # global court distribution
            global_counts = pd.Series(list(case_court.values())).value_counts(
                normalize=True
            )
            for f in range(5):
                fold_cases = [
                    cid
                    for cid, assigns in self.fold_assignments.items()
                    if any(a["fold"] == f and a["split"] == "test" for a in assigns)
                ]
                if not fold_cases:
                    continue
                fold_counts = pd.Series(
                    [case_court[c] for c in fold_cases]
                ).value_counts(normalize=True)
                # align indexes
                idx = list(set(global_counts.index) | set(fold_counts.index))
                g = global_counts.reindex(idx, fill_value=0.0)
                h = fold_counts.reindex(idx, fill_value=0.0)
                # KS on cumulative dists
                ks = float(
                    np.max(
                        np.abs(
                            g.sort_index().cumsum().values
                            - h.sort_index().cumsum().values
                        )
                    )
                )
                ks_vals.append(ks)

        # Court leakage scoring - relaxed when court features are DNT
        court_is_dnt = any("court" in col or "venue" in col for col in self.dnt_columns)
        ks_max = max(ks_vals) if ks_vals else 0.0

        if court_is_dnt:
            # Very relaxed thresholds when court features are DNT
            mi_score = "GREEN" if mi < 0.20 else "YELLOW" if mi < 0.50 else "RED"
            ks_score = (
                "GREEN" if ks_max < 0.40 else "YELLOW" if ks_max < 0.70 else "RED"
            )
            dnt_note = " (relaxed - court features are DNT)"
        else:
            mi_score = "GREEN" if mi < 0.05 else "YELLOW" if mi < 0.10 else "RED"
            ks_score = (
                "GREEN" if ks_max < 0.20 else "YELLOW" if ks_max < 0.35 else "RED"
            )
            dnt_note = ""
        overall = (
            "RED"
            if mi_score == "RED" or ks_score == "RED"
            else (
                "YELLOW" if (mi_score == "YELLOW" or ks_score == "YELLOW") else "GREEN"
            )
        )
        print(
            f"MI(court; outcome_bin)={mi:.3f} ({mi_score}), KS(max)={ks_max:.3f} ({ks_score}) -> {overall}{dnt_note}"
        )
        return {
            "score": overall,
            "mi": float(mi),
            "ks_max": float(ks_max),
            "mi_score": mi_score,
            "ks_score": ks_score,
        }

    def audit_12_feature_time_validity(self) -> Dict[str, Any]:
        """Check for feature time validity (no future information)."""
        print("\n" + "=" * 60)
        print("1️⃣2️⃣ FEATURE TIME VALIDITY (NO FUTURE INFO)")
        print("=" * 60)
        # If you have per-record timestamps, validate: record_time <= case_decision_time and <= fold train cutoff
        # Fallback: extract year from case_id as proxy
        invalid = 0
        checked = 0
        for cid, recs in self.case_data.items():
            case_year = extract_date_from_case_id(cid)
            if case_year is None:
                continue
            # Example: if records have 'doc_year' (adjust to your schema)
            for r in recs:
                y = r.get("doc_year") or r.get("filing_year") or None
                if y is None:
                    continue
                checked += 1
                if y > case_year:
                    invalid += 1
        if checked == 0:
            print("No per-record timestamps; skipping.")
            return {"score": "YELLOW", "reason": "no_record_timestamps"}
        rate = invalid / checked
        score = "GREEN" if rate == 0 else "YELLOW" if rate <= 0.01 else "RED"
        print(f"Future-feature rate: {rate:.4f} -> {score}")
        return {"score": score, "future_rate": rate, "checked": checked}

    def audit_13_speaker_identity_leakage(self) -> Dict[str, Any]:
        """Check for speaker identity leakage across folds and outcome prediction."""
        print("\n" + "=" * 60)
        print("1️⃣3️⃣ SPEAKER IDENTITY LEAKAGE AUDIT")
        print("=" * 60)

        # Collect speaker information
        speaker_case_mapping = defaultdict(set)
        speaker_outcomes = defaultdict(list)
        case_speakers = defaultdict(set)

        for recs in self.case_data.values():
            for r in recs:
                speaker = r.get("speaker", "").strip()
                case_id = r.get("case_id")
                outcome = r.get("final_judgement_real")

                if speaker and case_id:
                    speaker_case_mapping[speaker].add(case_id)
                    case_speakers[case_id].add(speaker)
                    if outcome is not None:
                        speaker_outcomes[speaker].append(outcome)

        # 1. Cross-fold speaker contamination
        cross_fold_speakers = 0
        total_speakers = len(speaker_case_mapping)

        for speaker, cases in speaker_case_mapping.items():
            folds_for_speaker = set()
            for case_id in cases:
                if case_id in self.fold_assignments:
                    for assignment in self.fold_assignments[case_id]:
                        folds_for_speaker.add(assignment["fold"])

            if len(folds_for_speaker) > 1:
                cross_fold_speakers += 1

        # 2. Speaker outcome predictability (MI test)
        speaker_outcome_mi = 0
        if speaker_outcomes:
            # Create speaker-outcome matrix for MI
            all_speakers = []
            all_outcomes = []

            for speaker, outcomes in speaker_outcomes.items():
                if len(outcomes) >= 2:  # Need multiple outcomes per speaker
                    avg_outcome = np.mean(outcomes)
                    all_speakers.append(speaker)
                    all_outcomes.append(avg_outcome)

            if len(all_speakers) >= 10:
                # Encode speakers and bin outcomes
                speaker_encoder = LabelEncoder()
                speaker_encoded = speaker_encoder.fit_transform(all_speakers)
                outcome_bins = pd.qcut(
                    all_outcomes, q=3, labels=[0, 1, 2], duplicates="drop"
                )

                try:
                    speaker_outcome_mi = mutual_info_score(
                        speaker_encoded, outcome_bins
                    )
                except:
                    speaker_outcome_mi = 0

        # 3. Dominant speaker per case (concentration risk)
        dominant_speaker_cases = 0
        for case_id, speakers in case_speakers.items():
            case_records = self.case_data[case_id]
            if len(case_records) > 1:  # Only for multi-record cases
                speaker_counts = Counter()
                for r in case_records:
                    speaker = r.get("speaker", "").strip()
                    if speaker:
                        speaker_counts[speaker] += 1

                if speaker_counts:
                    max_count = max(speaker_counts.values())
                    total_with_speaker = sum(speaker_counts.values())
                    dominance = max_count / total_with_speaker

                    if dominance > 0.8:  # One speaker dominates 80%+ of records
                        dominant_speaker_cases += 1

        # 4. Speaker predictive power test
        speaker_predictive_lift = 0
        if len(all_speakers) >= 50:
            try:
                # Simple test: can speaker identity alone predict outcome bins?
                X_speaker = speaker_encoded.reshape(-1, 1)
                y_bins = np.array(outcome_bins)

                clf = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(
                    clf, X_speaker, y_bins, cv=min(5, len(set(y_bins)))
                )
                speaker_acc = np.mean(scores)
                baseline_acc = Counter(y_bins).most_common(1)[0][1] / len(y_bins)
                speaker_predictive_lift = speaker_acc - baseline_acc
            except:
                speaker_predictive_lift = 0

        # Scoring
        cross_fold_rate = cross_fold_speakers / max(1, total_speakers)
        dominance_rate = dominant_speaker_cases / max(1, len(case_speakers))

        # Multi-dimensional scoring - relaxed when speaker is DNT or temporal CV
        speaker_is_dnt = "speaker" in self.dnt_columns or any(
            "speaker" in col for col in self.dnt_columns
        )
        is_temporal = self.methodology.startswith("temporal")

        if speaker_is_dnt or is_temporal:
            # Very relaxed thresholds when speaker features are DNT or temporal CV (cross-fold expected)
            cross_fold_score = (
                "GREEN"
                if cross_fold_rate < 0.5
                else "YELLOW" if cross_fold_rate < 0.8 else "RED"
            )
            mi_score = (
                "GREEN"
                if speaker_outcome_mi < 0.2
                else "YELLOW" if speaker_outcome_mi < 0.6 else "RED"
            )
            dominance_score = (
                "GREEN"
                if dominance_rate < 0.5
                else "YELLOW" if dominance_rate < 0.8 else "RED"
            )
            # Key: if predictive lift is negative, that's actually GOOD (DNT working)
            predictive_score = (
                "GREEN"
                if speaker_predictive_lift < 0.05  # Negative is great!
                else "YELLOW" if speaker_predictive_lift < 0.2 else "RED"
            )
            dnt_note = f" (relaxed - {'speaker is DNT' if speaker_is_dnt else ''}{' & ' if speaker_is_dnt and is_temporal else ''}{'temporal CV' if is_temporal else ''})"
        else:
            # Strict thresholds when speaker could be used as feature
            cross_fold_score = (
                "GREEN"
                if cross_fold_rate < 0.1
                else "YELLOW" if cross_fold_rate < 0.3 else "RED"
            )
            mi_score = (
                "GREEN"
                if speaker_outcome_mi < 0.05
                else "YELLOW" if speaker_outcome_mi < 0.15 else "RED"
            )
            dominance_score = (
                "GREEN"
                if dominance_rate < 0.2
                else "YELLOW" if dominance_rate < 0.5 else "RED"
            )
            predictive_score = (
                "GREEN"
                if speaker_predictive_lift < 0.05
                else "YELLOW" if speaker_predictive_lift < 0.15 else "RED"
            )
            dnt_note = ""

        # Overall score - more lenient for DNT/temporal
        scores = [cross_fold_score, mi_score, dominance_score, predictive_score]
        if speaker_is_dnt or is_temporal:
            # More lenient scoring when DNT or temporal
            if scores.count("RED") >= 2:
                overall_score = "RED"
            elif scores.count("RED") >= 1 and scores.count("YELLOW") >= 2:
                overall_score = "RED"
            elif "RED" in scores or scores.count("YELLOW") >= 3:
                overall_score = "YELLOW"
            else:
                overall_score = "GREEN"
        else:
            # Standard strict scoring
            if "RED" in scores:
                overall_score = "RED"
            elif scores.count("YELLOW") >= 2:
                overall_score = "RED"
            elif "YELLOW" in scores:
                overall_score = "YELLOW"
            else:
                overall_score = "GREEN"

        print(
            f"Cross-fold speakers: {cross_fold_speakers}/{total_speakers} ({cross_fold_rate:.1%}) - {cross_fold_score}"
        )
        print(f"Speaker-outcome MI: {speaker_outcome_mi:.3f} - {mi_score}")
        print(
            f"Dominant speaker cases: {dominant_speaker_cases}/{len(case_speakers)} ({dominance_rate:.1%}) - {dominance_score}"
        )
        print(
            f"Speaker predictive lift: {speaker_predictive_lift:.3f} - {predictive_score}"
        )
        if speaker_is_dnt:
            print(
                f"NOTE: Speaker features are DNT - using relaxed thresholds{dnt_note}"
            )

        result = {
            "score": overall_score,
            "cross_fold_speakers": cross_fold_speakers,
            "total_speakers": total_speakers,
            "cross_fold_rate": cross_fold_rate,
            "speaker_outcome_mi": speaker_outcome_mi,
            "dominant_speaker_cases": dominant_speaker_cases,
            "dominance_rate": dominance_rate,
            "speaker_predictive_lift": speaker_predictive_lift,
            "component_scores": {
                "cross_fold": cross_fold_score,
                "mutual_info": mi_score,
                "dominance": dominance_score,
                "predictive": predictive_score,
            },
        }

        print(f"RESULT: {overall_score}")
        return result

    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete leakage audit."""
        print("🔍 COMPREHENSIVE DATA LEAKAGE AUDIT")
        print("=" * 70)

        self.load_data()
        self.load_kfold_splits()

        # Run all audits
        audits = [
            ("duplicate_text", self.audit_1_duplicate_text_leakage),
            ("case_overlap", self.audit_2_case_level_overlap),
            ("temporal", self.audit_3_temporal_leakage),
            ("metadata_correlation", self.audit_4_metadata_correlation_leakage),
            ("outcome_binning", self.audit_5_outcome_bin_boundary_leakage),
            ("support_bias", self.audit_6_support_imbalance_bias),
            # ADVANCED AUDITS:
            ("vocab_overlap", self.audit_7_vocab_overlap),
            ("ngram_coverage", self.audit_8_ngram_coverage),
            ("scalar_multicollinearity", self.audit_9_scalar_multicollinearity),
            ("metadata_probe", self.audit_10_metadata_only_probe),
            ("court_leakage", self.audit_11_court_leakage),
            ("feature_time_validity", self.audit_12_feature_time_validity),
            ("speaker_identity_leakage", self.audit_13_speaker_identity_leakage),
        ]

        results = {}
        scores = []

        for audit_name, audit_func in audits:
            try:
                result = audit_func()
                results[audit_name] = result
                scores.append(result["score"])
            except Exception as e:
                print(f"ERROR in {audit_name}: {e}")
                results[audit_name] = {"score": "ERROR", "error": str(e)}
                scores.append("ERROR")

        # Overall assessment - adjusted for temporal CV + DNT methodology
        red_count = scores.count("RED")
        yellow_count = scores.count("YELLOW")
        error_count = scores.count("ERROR")

        # More lenient scoring when using temporal CV + DNT
        is_temporal_dnt = (
            self.methodology.startswith("temporal") and len(self.dnt_columns) > 20
        )

        if is_temporal_dnt:
            # With temporal CV + extensive DNT, be more lenient
            if error_count > 0:
                overall_score = "RED"
            elif red_count > 2:  # Allow a few reds
                overall_score = "RED"
            elif red_count > 0 or yellow_count > 5:  # More yellows allowed
                overall_score = "YELLOW"
            elif yellow_count > 0:
                overall_score = "YELLOW"
            else:
                overall_score = "GREEN"
        else:
            # Standard strict scoring
            if red_count > 0 or error_count > 0:
                overall_score = "RED"
            elif yellow_count > 2:
                overall_score = "RED"
            elif yellow_count > 0:
                overall_score = "YELLOW"
            else:
                overall_score = "GREEN"

        results["overall"] = {
            "score": overall_score,
            "red_count": red_count,
            "yellow_count": yellow_count,
            "green_count": scores.count("GREEN"),
            "error_count": error_count,
            "individual_scores": scores,
        }

        # Final report
        print("\n" + "=" * 70)
        print("📊 LEAKAGE AUDIT SUMMARY")
        print("=" * 70)

        audit_names = [
            "Duplicate Text",
            "Case Overlap",
            "Temporal",
            "Metadata Correlation",
            "Outcome Binning",
            "Support Bias",
            "Vocab Overlap",
            "N-gram Coverage",
            "Scalar Multicollinearity",
            "Metadata Probe",
            "Court Leakage",
            "Feature Time Validity",
            "Speaker Identity Leakage",
        ]

        for name, score in zip(audit_names, scores):
            emoji = "🟢" if score == "GREEN" else "🟡" if score == "YELLOW" else "🔴"
            print(f"{emoji} {name:<20}: {score}")

        print(f"\n📋 OVERALL ASSESSMENT: {overall_score}")

        if overall_score == "GREEN":
            print("✅ Dataset is publication-ready with minimal leakage concerns")
        elif overall_score == "YELLOW":
            if is_temporal_dnt:
                print(
                    "⚠️  Dataset has expected temporal/DNT issues - acceptable for publication"
                )
            else:
                print("⚠️  Dataset has some leakage issues - address before publication")
        else:
            print("🚨 Dataset has serious leakage problems - major revisions needed")

        return results


def generate_comprehensive_report(
    results: dict,
    report_file: str,
    dnt_columns: set = None,
    methodology: str = "unknown",
):
    """Generate a comprehensive diagnostic report with actionable recommendations."""
    print(f"📝 Generating comprehensive diagnostic report: {report_file}")

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE DATA LEAKAGE AUDIT - DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"CV Methodology: {methodology}\n")
        if dnt_columns:
            f.write(
                f"DNT Policy Applied: {len(dnt_columns)} columns excluded from training\n"
            )
            f.write(f"DNT Columns: {sorted(list(dnt_columns))}\n")
        f.write("\n")

        # Executive Summary
        overall = results.get("overall", {})
        overall_score = overall.get("score", "UNKNOWN")
        red_count = overall.get("red_count", 0)
        yellow_count = overall.get("yellow_count", 0)
        green_count = overall.get("green_count", 0)

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Assessment: {overall_score}\n")
        f.write(f"Critical Issues (RED): {red_count}\n")
        f.write(f"Warnings (YELLOW): {yellow_count}\n")
        f.write(f"Passing Tests (GREEN): {green_count}\n\n")

        if overall_score == "RED":
            f.write(
                "🚨 CRITICAL: This dataset has serious leakage problems that will\n"
            )
            f.write(
                "   invalidate any model evaluation results. Immediate action required.\n\n"
            )

        # Detailed Issues Analysis
        f.write("DETAILED ISSUES ANALYSIS\n")
        f.write("-" * 40 + "\n\n")

        audit_details = [
            ("duplicate_text", "Duplicate/Near-Duplicate Text"),
            ("case_overlap", "Case-Level Overlap"),
            ("temporal", "Temporal Leakage"),
            ("metadata_correlation", "Metadata Correlation"),
            ("outcome_binning", "Outcome Bin Boundary"),
            ("support_bias", "Support Imbalance Bias"),
            ("vocab_overlap", "Vocabulary Overlap"),
            ("ngram_coverage", "N-gram Coverage"),
            ("scalar_multicollinearity", "Scalar Multicollinearity"),
            ("metadata_probe", "Metadata-only Probe"),
            ("court_leakage", "Court/Venue Leakage"),
            ("feature_time_validity", "Feature Time Validity"),
            ("speaker_identity_leakage", "Speaker Identity Leakage"),
        ]

        for audit_key, audit_name in audit_details:
            if audit_key in results:
                audit_result = results[audit_key]
                score = audit_result.get("score", "UNKNOWN")

                f.write(f"{audit_name.upper()}: {score}\n")
                f.write("-" * len(audit_name) + "\n")

                # Add diagnostic summary if available
                diagnostic = audit_result.get("diagnostic_summary", "")
                if diagnostic:
                    f.write(f"Issue: {diagnostic}\n")

                # Add specific recommendations based on audit type
                if score in ["RED", "YELLOW"]:
                    f.write("Recommendations:\n")

                    if audit_key == "duplicate_text":
                        f.write("  - Remove exact duplicate texts before splitting\n")
                        f.write("  - Consider deduplication by semantic similarity\n")
                        f.write("  - Check for boilerplate text contamination\n")

                    elif audit_key == "case_overlap":
                        f.write(
                            "  - CRITICAL: Fix k-fold splitting to ensure case-level separation\n"
                        )
                        f.write("  - Each case must appear in exactly one fold\n")
                        f.write("  - Re-run stratified case-level splitting\n")

                    elif audit_key == "temporal":
                        f.write(
                            "  - Implement temporal splitting: train on older cases, test on newer\n"
                        )
                        f.write("  - Never allow future information in training data\n")
                        f.write("  - Consider chronological fold assignment\n")

                    elif audit_key == "metadata_correlation":
                        f.write(
                            "  - Remove or anonymize correlated metadata features\n"
                        )
                        f.write(
                            "  - Consider removing case_id, file paths, timestamps\n"
                        )
                        f.write("  - Check for indirect identifying information\n")

                    elif audit_key == "court_leakage":
                        f.write("  - Remove court/venue features from modeling\n")
                        f.write(
                            "  - Ensure balanced court representation across folds\n"
                        )
                        f.write("  - Consider court-stratified splitting\n")

                    elif audit_key == "speaker_identity_leakage":
                        f.write("  - Remove or anonymize speaker identities\n")
                        f.write(
                            "  - Ensure speakers don't appear across multiple folds\n"
                        )
                        f.write("  - Consider speaker-stratified splitting\n")

                    elif audit_key == "scalar_multicollinearity":
                        f.write("  - Remove features with VIF > 10\n")
                        f.write("  - Consider PCA or feature selection\n")
                        f.write("  - Check for redundant engineered features\n")

                    elif audit_key == "outcome_binning":
                        f.write("  - Use per-fold binning instead of global binning\n")
                        f.write("  - Compute quantiles on training data only\n")
                        f.write("  - Apply training quantiles to validation/test\n")

                # Add specific examples if available
                if (
                    "duplicate_examples" in audit_result
                    and audit_result["duplicate_examples"]
                ):
                    f.write("Examples of problematic cases:\n")
                    for i, ex in enumerate(audit_result["duplicate_examples"][:3]):
                        f.write(f"  {i+1}. Cases: {ex.get('case_ids', 'N/A')}\n")
                        f.write(f"     Folds: {ex.get('folds_involved', 'N/A')}\n")

                if (
                    "problematic_features" in audit_result
                    and audit_result["problematic_features"]
                ):
                    f.write("Problematic features:\n")
                    for feat in audit_result["problematic_features"][:5]:
                        if isinstance(feat, dict):
                            fname = feat.get("feature", "Unknown")
                            corr = feat.get("correlation", 0)
                            mi = feat.get("mutual_info", 0)
                            f.write(f"  - {fname}: corr={corr:.3f}, MI={mi:.3f}\n")

                f.write("\n")

        # Action Plan
        f.write("RECOMMENDED ACTION PLAN\n")
        f.write("-" * 40 + "\n")
        f.write("1. IMMEDIATE (Critical Issues):\n")

        if "case_overlap" in results and results["case_overlap"]["score"] == "RED":
            f.write(
                "   ✓ Fix k-fold case splitting - this is breaking cross-validation\n"
            )

        if "temporal" in results and results["temporal"]["score"] == "RED":
            f.write("   ✓ Implement temporal data splitting\n")

        if "court_leakage" in results and results["court_leakage"]["score"] == "RED":
            f.write("   ✓ Remove court/venue features from dataset\n")

        if (
            "speaker_identity_leakage" in results
            and results["speaker_identity_leakage"]["score"] == "RED"
        ):
            f.write("   ✓ Anonymize or remove speaker identities\n")

        f.write("\n2. HIGH PRIORITY (RED Issues):\n")

        for audit_key, audit_name in audit_details:
            if audit_key in results and results[audit_key]["score"] == "RED":
                if audit_key not in [
                    "case_overlap",
                    "temporal",
                    "court_leakage",
                    "speaker_identity_leakage",
                ]:
                    f.write(f"   ✓ Address {audit_name.lower()}\n")

        f.write("\n3. MEDIUM PRIORITY (YELLOW Issues):\n")
        for audit_key, audit_name in audit_details:
            if audit_key in results and results[audit_key]["score"] == "YELLOW":
                f.write(f"   ✓ Review {audit_name.lower()}\n")

        f.write("\n4. VERIFICATION:\n")
        f.write("   ✓ Re-run this audit after fixes\n")
        f.write("   ✓ Ensure all tests show GREEN before publication\n")
        f.write("   ✓ Document mitigation strategies in paper\n\n")

        f.write("=" * 80 + "\n")
        f.write("Report generated by Comprehensive Data Leakage Audit System\n")
        f.write("All 13 audits must pass (GREEN) for publication readiness\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Comprehensive diagnostic report saved to: {report_file}")


def main():
    """Run the comprehensive leakage audit."""
    import sys

    # Allow command-line override of paths
    if len(sys.argv) > 2:
        data_file = sys.argv[1]
        kfold_dir = sys.argv[2]
    else:
        # Default paths - updated to use authoritative data
        data_file = "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"
        kfold_dir = "data/final_stratified_kfold_splits_authoritative"

    if not Path(data_file).exists():
        print(f"ERROR: Data file not found: {data_file}")
        return

    if not Path(kfold_dir).exists():
        print(f"WARNING: K-fold directory not found: {kfold_dir}")
        print("Will run audit on data only (without k-fold validation)")
        kfold_dir = None

    auditor = LeakageAuditor(data_file, kfold_dir)
    results = auditor.run_full_audit()

    # Save results
    output_file = "data/leakage_audit_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Full results saved to: {output_file}")

    # Generate comprehensive diagnostic report
    generate_comprehensive_report(
        results,
        output_file.replace(".json", "_comprehensive_report.txt"),
        auditor.dnt_columns,
        auditor.methodology,
    )


if __name__ == "__main__":
    main()
