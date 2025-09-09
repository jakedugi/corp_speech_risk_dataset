#!/usr/bin/env python3
"""
Unified Authoritative Feature Extraction and Pruning Pipeline

This script combines all interpretable feature extraction, testing, and pruning
into a single comprehensive pipeline that:

1. Extracts all interpretable features (base + derived)
2. Tests features for discriminative power
3. Analyzes redundancy and multicollinearity
4. Prunes features based on established criteria
5. Generates final feature set for training

Usage:
    python scripts/extract_and_prune_features_pipeline.py \
        --input "data/final_destination/courtlistener_v6_fused_raw_coral_pred/doc_*_text_stage15.jsonl" \
        --output-dir data/final_destination/courtlistener_v6_fused_raw_coral_pred_with_features \
        --analysis-output-dir docs/feature_analysis \
        --text-field text \
        --context-field context \
        --batch-size 1000 \
        --sample-size 50000 \
        --run-pruning
"""

import argparse
import json
import glob
import sys
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.features import (
    InterpretableFeatureExtractor,
)

# ============================================================
# PART 1: FEATURE EXTRACTION
# ============================================================


def load_jsonl_batch(file_path: str, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    """Load JSONL file in batches for memory efficiency."""
    batch = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    batch.append(record)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}"
                    )
                    continue

            # Yield remaining records
            if batch:
                yield batch

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return


def extract_text_and_context(
    record: Dict[str, Any], text_field: str, context_field: Optional[str]
) -> tuple[str, str]:
    """Extract text and context from a record."""
    text = record.get(text_field, "")
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    context = ""
    if context_field and context_field in record:
        context = record.get(context_field, "")
        if not isinstance(context, str):
            context = str(context) if context is not None else ""

    return text, context


def enhance_records_with_features(
    records: List[Dict[str, Any]],
    text_field: str,
    context_field: Optional[str],
    feature_prefix: str,
    extractor: InterpretableFeatureExtractor,
) -> List[Dict[str, Any]]:
    """Enhance records with interpretable features."""
    enhanced_records = []

    for record in records:
        try:
            # Extract text and context
            text, context = extract_text_and_context(record, text_field, context_field)

            # Skip if no text
            if not text.strip():
                enhanced_records.append(record)
                continue

            # Extract features using the class-based extractor
            features_dict = extractor.extract_features(text, context)

            # Create enhanced record
            enhanced_record = record.copy()

            # Add features with prefix
            for feature_name, feature_value in features_dict.items():
                prefixed_name = f"{feature_prefix}_{feature_name}"
                enhanced_record[prefixed_name] = float(feature_value)

            # Add metadata
            enhanced_record[f"{feature_prefix}_feature_count"] = len(features_dict)
            enhanced_record[f"{feature_prefix}_text_length"] = len(text)
            enhanced_record[f"{feature_prefix}_context_length"] = len(context)

            enhanced_records.append(enhanced_record)

        except Exception as e:
            print(f"Warning: Failed to extract features for record: {e}")
            enhanced_records.append(record)
            continue

    return enhanced_records


def process_single_file_extraction(
    input_path: str,
    output_path: str,
    text_field: str,
    context_field: Optional[str],
    feature_prefix: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Process a single JSONL file and enhance with features."""
    print(f"Processing: {input_path}")

    # Initialize feature extractor
    extractor = InterpretableFeatureExtractor()

    stats = {
        "total_records": 0,
        "processed_records": 0,
        "failed_records": 0,
        "feature_count": 0,
    }

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Process file in batches
    with open(output_path, "w", encoding="utf-8") as output_file:
        for batch_records in tqdm(
            load_jsonl_batch(input_path, batch_size), desc="Extracting features"
        ):
            stats["total_records"] += len(batch_records)

            try:
                # Enhance batch with features
                enhanced_records = enhance_records_with_features(
                    batch_records, text_field, context_field, feature_prefix, extractor
                )

                # Set feature count from first successful extraction
                if stats["feature_count"] == 0 and enhanced_records:
                    # Count features from first enhanced record
                    feature_fields = [
                        k
                        for k in enhanced_records[0].keys()
                        if k.startswith(f"{feature_prefix}_")
                        and not k.endswith(
                            ("_feature_count", "_text_length", "_context_length")
                        )
                    ]
                    stats["feature_count"] = len(feature_fields)

                # Write enhanced records
                for record in enhanced_records:
                    json.dump(record, output_file, ensure_ascii=False)
                    output_file.write("\n")

                stats["processed_records"] += len(enhanced_records)

            except Exception as e:
                print(f"Error processing batch: {e}")
                stats["failed_records"] += len(batch_records)

                # Write original records if enhancement fails
                for record in batch_records:
                    json.dump(record, output_file, ensure_ascii=False)
                    output_file.write("\n")

    return stats


# ============================================================
# PART 2: FEATURE ANALYSIS AND PRUNING
# ============================================================


def load_sample_for_analysis(
    file_pattern: str, sample_size: int = 50000
) -> pd.DataFrame:
    """Load a sample of data for feature analysis."""
    print(f"📊 Loading sample data for analysis (up to {sample_size:,} records)...")

    files = sorted(glob.glob(file_pattern))[:5]  # Limit to first 5 files
    all_records = []

    for file_path in files:
        if len(all_records) >= sample_size:
            break

        with open(file_path, "r") as f:
            for line in f:
                if len(all_records) >= sample_size:
                    break
                try:
                    all_records.append(json.loads(line))
                except:
                    continue

    df = pd.DataFrame(all_records)
    print(f"✓ Loaded {len(df):,} records for analysis")

    return df


def get_interpretable_features(
    df: pd.DataFrame, prefix: str = "interpretable"
) -> List[str]:
    """Get list of interpretable feature columns."""
    return [
        col
        for col in df.columns
        if col.startswith(f"{prefix}_")
        and not col.endswith(("_feature_count", "_text_length", "_context_length"))
    ]


def group_features_by_concept(features: List[str]) -> Dict[str, List[str]]:
    """Group features by their base concept."""
    concepts = {}

    for feature in features:
        # Extract concept name
        if feature.startswith("interpretable_"):
            parts = feature.replace("interpretable_", "").split("_")

            # Group by main concept
            if parts[0] == "lex":
                concept = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "ling":
                if "certainty" in feature:
                    concept = "ling_certainty"
                else:
                    concept = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "seq":
                concept = f"{parts[0]}_{parts[1]}"
            elif parts[0] in ["ratio", "interact"]:
                concept = parts[0]
            else:
                concept = parts[0]

            if concept not in concepts:
                concepts[concept] = []
            concepts[concept].append(feature)

    return concepts


def compute_redundancy_analysis(
    df: pd.DataFrame, concepts: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """Compute redundancy metrics within each concept group."""
    print("🔍 Computing redundancy within concept groups...")

    redundancy_results = {}

    for concept, concept_features in concepts.items():
        if len(concept_features) <= 1:
            continue

        print(f"  Analyzing concept: {concept} ({len(concept_features)} features)")

        # Filter to features that exist in data
        available_features = [f for f in concept_features if f in df.columns]
        if len(available_features) <= 1:
            continue

        concept_data = df[available_features].fillna(0)

        # Compute VIF if we have multiple features
        vif_scores = {}
        if len(available_features) > 1:
            try:
                X = concept_data.values
                if X.std() > 0:
                    for i, feature in enumerate(available_features):
                        if X[:, i].std() > 0:
                            vif = variance_inflation_factor(X, i)
                            vif_scores[feature] = vif
            except Exception as e:
                print(f"    Warning: VIF calculation failed for {concept}: {e}")

        # Compute correlation matrix
        corr_matrix = concept_data.corr()

        redundancy_results[concept] = {
            "features": available_features,
            "vif_scores": vif_scores,
            "correlation_matrix": corr_matrix,
        }

    return redundancy_results


def compute_discriminative_power(
    df: pd.DataFrame, features: List[str], target_field: str = "outcome_bin"
) -> pd.DataFrame:
    """Compute discriminative power metrics for features."""
    print("📊 Computing discriminative power analysis...")

    if target_field not in df.columns and "final_judgement_real" in df.columns:
        # Create simple tertiles if outcome_bin not available
        values = df["final_judgement_real"].values
        tertiles = pd.qcut(values, q=3, labels=[0, 1, 2])
        df["outcome_bin"] = tertiles.astype(int)
        target_field = "outcome_bin"

    results = []

    for feature in tqdm(features, desc="Analyzing features"):
        if feature not in df.columns:
            continue

        result = {"feature": feature}

        # Basic stats
        data = df[feature].dropna()
        result["mean"] = data.mean()
        result["std"] = data.std()
        result["zero_pct"] = (data == 0).mean() * 100
        result["missing_pct"] = df[feature].isna().mean() * 100

        # Skip if no variance
        if data.std() == 0:
            result["mutual_info"] = 0
            result["kw_pvalue"] = 1.0
            results.append(result)
            continue

        # Mutual information with target
        try:
            # Bin continuous features for MI
            if data.nunique() > 10:
                feature_binned = pd.qcut(data, q=5, duplicates="drop", labels=False)
            else:
                feature_binned = data

            data_clean = df[[feature, target_field]].dropna()
            if len(data_clean) > 0:
                mi_score = mutual_info_score(
                    data_clean[target_field],
                    pd.qcut(data_clean[feature], q=5, duplicates="drop", labels=False),
                )
                result["mutual_info"] = mi_score
            else:
                result["mutual_info"] = 0
        except:
            result["mutual_info"] = 0

        # Kruskal-Wallis test
        try:
            groups = []
            for k in sorted(df[target_field].unique()):
                group_data = df[df[target_field] == k][feature].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)

            if len(groups) >= 2:
                kw_stat, kw_p = kruskal(*groups)
                result["kw_statistic"] = kw_stat
                result["kw_pvalue"] = kw_p
            else:
                result["kw_pvalue"] = 1.0
        except:
            result["kw_pvalue"] = 1.0

        results.append(result)

    return pd.DataFrame(results)


def recommend_feature_pruning(
    redundancy_results: Dict,
    discriminative_df: pd.DataFrame,
    mi_threshold: float = 0.005,
    p_threshold: float = 0.1,
) -> Dict[str, List[str]]:
    """Recommend which features to keep vs drop based on analysis."""
    print("✂️ Generating feature pruning recommendations...")

    keep_features = []
    drop_features = []
    drop_reasons = {}

    # Process each concept group
    for concept, concept_data in redundancy_results.items():
        features = concept_data["features"]

        if len(features) == 1:
            keep_features.extend(features)
            continue

        # Prioritize by type: norm > present > count
        def feature_priority(f):
            if "_norm" in f and "_quote_norm" not in f:
                return 1
            elif "_present" in f:
                return 2
            elif "_quote_norm" in f:
                return 3
            elif "_count" in f and "_quote_count" not in f:
                return 4
            elif "_quote_count" in f:
                return 5
            else:
                return 0

        # Sort features by priority
        sorted_features = sorted(features, key=feature_priority)

        # Keep the best normalized version + optionally presence
        kept_for_concept = []

        # Look for norm version first
        norm_features = [
            f for f in sorted_features if "_norm" in f and "_quote_norm" not in f
        ]
        if norm_features:
            kept_for_concept.append(norm_features[0])

        # Look for presence version
        present_features = [f for f in sorted_features if "_present" in f]
        if present_features:
            kept_for_concept.append(present_features[0])

        # If we didn't keep anything yet, keep the highest priority feature
        if not kept_for_concept and sorted_features:
            kept_for_concept.append(sorted_features[0])

        # Add kept features
        keep_features.extend(kept_for_concept)

        # Mark others for dropping
        for feature in features:
            if feature not in kept_for_concept:
                drop_features.append(feature)
                drop_reasons[feature] = "redundant_in_concept"

    # Check discriminative power and drop weak features
    if len(discriminative_df) > 0:
        weak_features = discriminative_df[
            (discriminative_df["kw_pvalue"] > p_threshold)
            & (discriminative_df["mutual_info"] < mi_threshold)
        ]["feature"].tolist()

        for feature in weak_features:
            if feature in keep_features:
                keep_features.remove(feature)
                drop_features.append(feature)
                drop_reasons[feature] = "weak_discriminative_power"

    # Apply final approved feature list if available
    approved_features = {
        "interpretable_lex_deception_norm",
        "interpretable_lex_deception_present",
        "interpretable_lex_guarantee_norm",
        "interpretable_lex_guarantee_present",
        "interpretable_lex_hedges_norm",
        "interpretable_lex_hedges_present",
        "interpretable_lex_pricing_claims_present",
        "interpretable_lex_superlatives_present",
        "interpretable_ling_high_certainty",
        "interpretable_seq_discourse_additive",
        # Add derived features
        "interpretable_ratio_guarantee_vs_hedge",
        "interpretable_ratio_deception_vs_hedge",
        "interpretable_ratio_guarantee_vs_superlative",
        "interpretable_interact_guarantee_x_cert",
        "interpretable_interact_superlative_x_cert",
        "interpretable_interact_hedge_x_guarantee",
    }

    # Filter to only approved features
    final_keep = [f for f in keep_features if f in approved_features]
    for feature in keep_features:
        if feature not in approved_features and feature not in drop_features:
            drop_features.append(feature)
            drop_reasons[feature] = "not_in_approved_list"

    print(f"  Recommendations: Keep {len(final_keep)}, Drop {len(drop_features)}")

    return {"keep": final_keep, "drop": drop_features, "drop_reasons": drop_reasons}


def save_analysis_results(
    keep_features: List[str],
    drop_features: List[str],
    drop_reasons: Dict[str, str],
    discriminative_df: pd.DataFrame,
    output_dir: Path,
):
    """Save feature analysis and pruning results."""
    print("💾 Saving analysis results...")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final_feature_set"
    final_dir.mkdir(exist_ok=True)

    # Save kept features
    with open(final_dir / "final_kept_features.txt", "w") as f:
        f.write("# Final Pruned Feature Set for Training\n")
        f.write(f"# Total: {len(keep_features)} features\n")
        f.write("# Applied pruning:\n")
        f.write("# - Removed redundant scales within concepts\n")
        f.write("# - Kept normalized + presence versions\n")
        f.write("# - Dropped weak separation features\n")
        f.write("# - Applied approved feature list filter\n\n")
        for feature in sorted(keep_features):
            f.write(f"{feature}\n")

    # Save pruned features with reasons
    if drop_features:
        pruned_df = pd.DataFrame(
            [{"feature": f, "pruning_reason": drop_reasons[f]} for f in drop_features]
        )
        pruned_df.to_csv(final_dir / "pruned_features.csv", index=False)

    # Save discriminative power analysis
    discriminative_df.to_csv(
        final_dir / "discriminative_power_analysis.csv", index=False
    )

    # Create feature card for kept features
    kept_df = discriminative_df[discriminative_df["feature"].isin(keep_features)].copy()
    kept_df = kept_df.sort_values("mutual_info", ascending=False)
    kept_df.to_csv(final_dir / "final_feature_card.csv", index=False)

    print(f"✓ Results saved to: {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Unified feature extraction and pruning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file pattern (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for enhanced files"
    )
    parser.add_argument(
        "--analysis-output-dir",
        default="docs/feature_analysis",
        help="Output directory for analysis results (default: docs/feature_analysis)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing main text (default: text)",
    )
    parser.add_argument(
        "--context-field", help="Field name containing context (optional)"
    )
    parser.add_argument(
        "--feature-prefix",
        default="interpretable",
        help="Prefix for feature field names (default: interpretable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing (default: 500)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Sample size for feature analysis (default: 50000)",
    )
    parser.add_argument(
        "--run-pruning",
        action="store_true",
        help="Run feature analysis and pruning after extraction",
    )
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=0.005,
        help="Mutual information threshold for pruning (default: 0.005)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.1,
        help="P-value threshold for pruning (default: 0.1)",
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        print(f"Error: No files found matching pattern: {args.input}")
        return 1

    print("🚀 UNIFIED FEATURE EXTRACTION AND PRUNING PIPELINE")
    print("=" * 60)
    print(f"Found {len(input_files)} files to process")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature extraction settings:")
    print(f"  Text field: {args.text_field}")
    print(f"  Context field: {args.context_field or 'None'}")
    print(f"  Feature prefix: {args.feature_prefix}")
    print(f"  Batch size: {args.batch_size}")
    print(f"Analysis settings:")
    print(f"  Run pruning: {args.run_pruning}")
    print(f"  Sample size: {args.sample_size:,}")
    print(f"  MI threshold: {args.mi_threshold}")
    print(f"  P-value threshold: {args.p_threshold}")
    print()

    # ============================================================
    # PHASE 1: FEATURE EXTRACTION
    # ============================================================
    print("PHASE 1: FEATURE EXTRACTION")
    print("-" * 60)

    total_stats = {
        "total_files": len(input_files),
        "processed_files": 0,
        "total_records": 0,
        "processed_records": 0,
        "failed_records": 0,
    }

    for input_file in input_files:
        try:
            # Generate output path
            input_path = Path(input_file)
            output_path = Path(args.output_dir) / input_path.name

            # Process file
            file_stats = process_single_file_extraction(
                input_file,
                str(output_path),
                args.text_field,
                args.context_field,
                args.feature_prefix,
                args.batch_size,
            )

            # Update totals
            total_stats["processed_files"] += 1
            total_stats["total_records"] += file_stats["total_records"]
            total_stats["processed_records"] += file_stats["processed_records"]
            total_stats["failed_records"] += file_stats["failed_records"]

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            continue

    # Print extraction summary
    print()
    print("FEATURE EXTRACTION SUMMARY")
    print("-" * 60)
    print(
        f"Total files processed: {total_stats['processed_files']}/{total_stats['total_files']}"
    )
    print(f"Total records: {total_stats['total_records']:,}")
    print(f"Successfully enhanced: {total_stats['processed_records']:,}")
    print(f"Failed enhancements: {total_stats['failed_records']:,}")
    success_rate = (
        100 * total_stats["processed_records"] / max(1, total_stats["total_records"])
    )
    print(f"Success rate: {success_rate:.1f}%")
    print()

    # ============================================================
    # PHASE 2: FEATURE ANALYSIS AND PRUNING
    # ============================================================
    if args.run_pruning:
        print("PHASE 2: FEATURE ANALYSIS AND PRUNING")
        print("-" * 60)

        # Load sample data from enhanced files
        output_pattern = str(Path(args.output_dir) / "*.jsonl")
        sample_df = load_sample_for_analysis(output_pattern, args.sample_size)

        # Get interpretable features
        features = get_interpretable_features(sample_df, args.feature_prefix)
        print(f"📋 Found {len(features)} interpretable features")

        # Group features by concept
        concepts = group_features_by_concept(features)
        print(f"🔍 Identified {len(concepts)} feature concepts")

        # Compute redundancy analysis
        redundancy_results = compute_redundancy_analysis(sample_df, concepts)

        # Compute discriminative power
        discriminative_df = compute_discriminative_power(sample_df, features)

        # Generate pruning recommendations
        pruning_recommendations = recommend_feature_pruning(
            redundancy_results, discriminative_df, args.mi_threshold, args.p_threshold
        )

        keep_features = pruning_recommendations["keep"]
        drop_features = pruning_recommendations["drop"]
        drop_reasons = pruning_recommendations["drop_reasons"]

        print()
        print("PRUNING RESULTS")
        print("-" * 60)
        print(f"✅ Final keep: {len(keep_features)} features")
        print(f"❌ Final drop: {len(drop_features)} features")

        # Save analysis results
        save_analysis_results(
            keep_features,
            drop_features,
            drop_reasons,
            discriminative_df,
            Path(args.analysis_output_dir),
        )

        # Print final feature list
        print()
        print(f"🏆 FINAL FEATURES ({len(keep_features)}):")
        for concept in ["lex", "ling", "seq", "ratio", "interact"]:
            concept_features = [
                f
                for f in keep_features
                if f.startswith(f"{args.feature_prefix}_{concept}")
            ]
            if concept_features:
                feature_names = [f.split("_", 2)[-1] for f in concept_features]
                print(
                    f"  {concept.upper()} ({len(concept_features)}): {', '.join(feature_names[:5])}"
                )

    print()
    print("✅ PIPELINE COMPLETE!")
    print(f"📁 Enhanced files saved to: {args.output_dir}")
    if args.run_pruning:
        print(f"📊 Analysis results saved to: {args.analysis_output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
