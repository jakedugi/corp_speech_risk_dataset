#!/usr/bin/env python3
"""Final feature pruning and quality checks for interpretable features.

This script implements the final analysis steps:
1. Redundancy & size-bias checks
2. Per-bucket separation analysis
3. VIF within concept groups
4. Final pruned feature set generation
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.column_governance import (
    validate_columns,
)


def load_final_features() -> List[str]:
    """Load the current kept features list."""
    with open("docs/feature_analysis/final_feature_set/kept_features.txt", "r") as f:
        lines = f.readlines()

    features = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            features.append(line)

    return features


def load_fold3_train_data() -> pd.DataFrame:
    """Load fold 3 training data for analysis."""
    print("📊 Loading fold 3 training data...")

    # Load fold metadata
    with open(
        "data/final_stratified_kfold_splits_adaptive_oof/per_fold_metadata.json", "r"
    ) as f:
        fold_metadata = json.load(f)

    # Load fold 3 train data
    train_path = Path(
        "data/final_stratified_kfold_splits_adaptive_oof/fold_3/train.jsonl"
    )
    train_data = []
    with open(train_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))

    train_df = pd.DataFrame(train_data)

    # Apply tertile labels
    cutpoints_list = fold_metadata["binning"]["fold_edges"]["fold_3"]
    cutpoints = {"q1": cutpoints_list[0], "q2": cutpoints_list[1]}

    # Simple tertile labeling
    values = train_df["final_judgement_real"].values
    labels = np.zeros(len(values), dtype=int)
    labels[values > cutpoints["q1"]] = 1
    labels[values > cutpoints["q2"]] = 2
    train_df["bin"] = labels

    # Add case size
    train_df["case_size"] = train_df.groupby("case_id")["case_id"].transform("count")

    print(
        f"✓ Loaded {len(train_df):,} quotes from {train_df['case_id'].nunique()} cases"
    )

    return train_df


def group_features_by_concept(features: List[str]) -> Dict[str, List[str]]:
    """Group features by their base concept."""
    concepts = {}

    for feature in features:
        # Extract concept name (everything before the last underscore for scale)
        if feature.startswith("interpretable_"):
            parts = feature.replace("interpretable_", "").split("_")

            # Group by main concept
            if parts[0] == "lex":
                # lex_deception_count -> concept: lex_deception
                concept = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "ling":
                # ling_high_certainty -> concept: ling_certainty
                if "certainty" in feature:
                    concept = "ling_certainty"
                else:
                    concept = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "seq":
                # seq_discourse_additive -> concept: seq_discourse
                # seq_trans_neutral_to_neutral -> concept: seq_trans
                concept = f"{parts[0]}_{parts[1]}"
            else:
                concept = parts[0]

            if concept not in concepts:
                concepts[concept] = []
            concepts[concept].append(feature)

    return concepts


def compute_redundancy_within_concepts(
    train_df: pd.DataFrame, concepts: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """Compute redundancy metrics within each concept group."""
    print("🔍 Computing redundancy within concept groups...")

    redundancy_results = {}

    for concept, concept_features in concepts.items():
        if len(concept_features) <= 1:
            continue

        print(f"  Analyzing concept: {concept} ({len(concept_features)} features)")

        # Filter to features that exist in data
        available_features = [f for f in concept_features if f in train_df.columns]
        if len(available_features) <= 1:
            continue

        concept_data = train_df[available_features].fillna(0)

        # Compute VIF if we have multiple features
        vif_scores = {}
        if len(available_features) > 1:
            try:
                # Add constant for VIF calculation
                X = concept_data.values
                if X.std() > 0:  # Check for non-zero variance
                    for i, feature in enumerate(available_features):
                        if X[:, i].std() > 0:  # Feature has variance
                            vif = variance_inflation_factor(X, i)
                            vif_scores[feature] = vif
            except Exception as e:
                print(f"    Warning: VIF calculation failed for {concept}: {e}")

        # Compute correlation matrix
        corr_matrix = concept_data.corr()

        # Compute size bias for each feature
        size_bias = {}
        for feature in available_features:
            if feature in train_df.columns and "case_size" in train_df.columns:
                data = train_df[[feature, "case_size"]].dropna()
                if len(data) > 10 and data[feature].std() > 0:
                    try:
                        corr_size, _ = spearmanr(data[feature], data["case_size"])
                        size_bias[feature] = abs(corr_size)
                    except:
                        size_bias[feature] = 0
                else:
                    size_bias[feature] = 0

        redundancy_results[concept] = {
            "features": available_features,
            "vif_scores": vif_scores,
            "correlation_matrix": corr_matrix,
            "size_bias": size_bias,
        }

    return redundancy_results


def compute_per_bucket_separation(
    train_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Compute per-bucket separation metrics."""
    print("🪣 Computing per-bucket separation analysis...")

    separation_results = []

    for feature in features:
        if feature not in train_df.columns or "bin" not in train_df.columns:
            continue

        data = train_df[[feature, "bin"]].dropna()
        if len(data) == 0:
            continue

        result = {"feature": feature}

        # Kruskal-Wallis test
        groups = [data[data["bin"] == k][feature] for k in [0, 1, 2]]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            try:
                kw_stat, kw_p = kruskal(*groups)
                result["kw_statistic"] = kw_stat
                result["kw_pvalue"] = kw_p
            except:
                result["kw_statistic"] = np.nan
                result["kw_pvalue"] = np.nan
        else:
            result["kw_statistic"] = np.nan
            result["kw_pvalue"] = np.nan

        # Cliff's delta for adjacent pairs
        if len(groups) >= 2:
            # 0→1 comparison
            if len(groups[0]) > 0 and len(groups[1]) > 0:
                cliff_01 = cliffs_delta(groups[0], groups[1])
                result["cliff_delta_01"] = cliff_01
            else:
                result["cliff_delta_01"] = np.nan

            # 1→2 comparison
            if len(groups) >= 3 and len(groups[1]) > 0 and len(groups[2]) > 0:
                cliff_12 = cliffs_delta(groups[1], groups[2])
                result["cliff_delta_12"] = cliff_12
            else:
                result["cliff_delta_12"] = np.nan

        # Mean values per bucket
        for bucket in [0, 1, 2]:
            bucket_data = data[data["bin"] == bucket][feature]
            if len(bucket_data) > 0:
                result[f"mean_bucket_{bucket}"] = bucket_data.mean()
            else:
                result[f"mean_bucket_{bucket}"] = np.nan

        # Mutual information
        try:
            if data[feature].std() > 0:
                # Bin the feature for MI calculation
                feature_binned = pd.qcut(
                    data[feature], q=5, duplicates="drop", labels=False
                )
                mi_score = mutual_info_score(data["bin"], feature_binned)
                result["mutual_info"] = mi_score
            else:
                result["mutual_info"] = 0
        except:
            result["mutual_info"] = np.nan

        separation_results.append(result)

    return pd.DataFrame(separation_results)


def cliffs_delta(group1, group2):
    """Compute Cliff's delta effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan

    # Count comparisons
    greater = sum(x > y for x in group1 for y in group2)
    lesser = sum(x < y for x in group1 for y in group2)

    delta = (greater - lesser) / (n1 * n2)
    return delta


def recommend_feature_pruning(
    redundancy_results: Dict, separation_results: pd.DataFrame
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
            # Only one feature in concept - keep it
            keep_features.extend(features)
            continue

        # Prioritize by type: norm > present > count > quote_*
        def feature_priority(f):
            if "_norm" in f and "_quote_norm" not in f:
                return 1  # Highest priority - normalized per token
            elif "_present" in f:
                return 2  # Second priority - binary presence
            elif "_quote_norm" in f:
                return 3  # Third priority - normalized per quote
            elif "_count" in f and "_quote_count" not in f:
                return 4  # Lower priority - raw count
            elif "_quote_count" in f:
                return 5  # Lowest priority - quote-level count
            else:
                return 0  # Unknown - keep

        # Sort features by priority
        sorted_features = sorted(features, key=feature_priority)

        # Strategy: Keep the best normalized version + optionally presence
        kept_for_concept = []

        # Look for norm version first
        norm_features = [
            f for f in sorted_features if "_norm" in f and "_quote_norm" not in f
        ]
        if norm_features:
            # Keep the first (best) norm feature
            kept_for_concept.append(norm_features[0])
        else:
            # No norm version, look for quote_norm
            quote_norm_features = [f for f in sorted_features if "_quote_norm" in f]
            if quote_norm_features:
                kept_for_concept.append(quote_norm_features[0])

        # Look for presence version
        present_features = [f for f in sorted_features if "_present" in f]
        if present_features:
            # Add presence feature (be more permissive)
            present_feature = present_features[0]
            kept_for_concept.append(present_feature)

        # If we didn't keep anything yet, keep the highest priority feature
        if not kept_for_concept and sorted_features:
            kept_for_concept.append(sorted_features[0])

        # Add kept features
        keep_features.extend(kept_for_concept)

        # Mark others for dropping
        for feature in features:
            if feature not in kept_for_concept:
                drop_features.append(feature)
                if "_count" in feature or "_quote_count" in feature:
                    drop_reasons[feature] = "redundant_raw_count"
                elif "_quote_norm" in feature and any(
                    "_norm" in k for k in kept_for_concept
                ):
                    drop_reasons[feature] = "redundant_normalization"
                else:
                    drop_reasons[feature] = "redundant_in_concept"

    # Check separation quality and drop weak features (be more conservative)
    # Only drop features with very weak separation (p > 0.1 AND low MI)
    if len(separation_results) > 0:
        weak_separation = separation_results[
            (separation_results["kw_pvalue"] > 0.1)
            & (separation_results["mutual_info"] < 0.005)
        ]["feature"].tolist()

        for feature in weak_separation:
            if feature in keep_features:
                keep_features.remove(feature)
                drop_features.append(feature)
                drop_reasons[feature] = "weak_separation"

    print(f"  Recommendations: Keep {len(keep_features)}, Drop {len(drop_features)}")

    return {"keep": keep_features, "drop": drop_features, "drop_reasons": drop_reasons}


def create_final_feature_card(
    final_features: List[str],
    separation_results: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: Path,
):
    """Create final feature card with comprehensive metrics."""
    print("📋 Creating final feature card...")

    card_data = []

    for feature in final_features:
        if feature not in train_df.columns:
            continue

        # Get separation metrics
        sep_data = separation_results[separation_results["feature"] == feature]

        # Basic stats
        data = train_df[feature].dropna()

        # Determine concept and expected direction
        concept = "unknown"
        expected_direction = "?"

        if "deception" in feature:
            concept = "deception"
            expected_direction = "↑ with risk"
        elif "guarantee" in feature:
            concept = "guarantee"
            expected_direction = "↓ with risk"
        elif "hedges" in feature:
            concept = "hedging"
            expected_direction = "↑ with risk"
        elif "pricing" in feature:
            concept = "pricing_claims"
            expected_direction = "↑ with risk"
        elif "superlatives" in feature:
            concept = "superlatives"
            expected_direction = "↑ with risk"
        elif "certainty" in feature:
            concept = "certainty"
            if "low" in feature:
                expected_direction = "↑ with risk"
            else:
                expected_direction = "↓ with risk"
        elif "negation" in feature:
            concept = "negation"
            expected_direction = "↑ with risk"
        elif "discourse" in feature:
            concept = "discourse"
            expected_direction = "varies"
        elif "trans" in feature:
            concept = "transitions"
            expected_direction = "varies"

        # Transform recommendation
        transform = "none"
        if data.std() > 0:
            if (data == 0).mean() > 0.7:  # Very sparse
                transform = "binarize"
            elif data.skew() > 2:  # Highly skewed
                transform = "log1p"

        card_entry = {
            "feature": feature,
            "concept": concept,
            "expected_direction": expected_direction,
            "transform": transform,
            "zero_pct": (data == 0).mean() * 100,
            "missing_pct": train_df[feature].isna().mean() * 100,
            "kw_pvalue": sep_data["kw_pvalue"].iloc[0] if len(sep_data) > 0 else np.nan,
            "mutual_info": (
                sep_data["mutual_info"].iloc[0] if len(sep_data) > 0 else np.nan
            ),
            "cliff_delta_01": (
                sep_data["cliff_delta_01"].iloc[0] if len(sep_data) > 0 else np.nan
            ),
            "cliff_delta_12": (
                sep_data["cliff_delta_12"].iloc[0] if len(sep_data) > 0 else np.nan
            ),
        }

        card_data.append(card_entry)

    card_df = pd.DataFrame(card_data)

    # Save feature card
    final_dir = output_dir / "final_feature_set"
    card_df.to_csv(final_dir / "final_feature_card.csv", index=False)

    # Create LaTeX feature card table
    latex_dir = final_dir / "latex"
    create_feature_card_latex(card_df, latex_dir)

    print(f"✓ Final feature card saved with {len(card_df)} features")

    return card_df


def create_feature_card_latex(card_df: pd.DataFrame, latex_dir: Path):
    """Create LaTeX table for feature card."""

    # Sort by mutual information descending if column exists
    if "mutual_info" in card_df.columns and len(card_df) > 0:
        card_sorted = card_df.sort_values(
            "mutual_info", ascending=False, na_position="last"
        )
    else:
        card_sorted = card_df

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(
        "\\caption{Final Feature Card: Interpretable Features for Training}"
    )
    latex_lines.append("\\label{tab:feature_card}")
    latex_lines.append("\\begin{tabular}{lllrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append(
        "Feature & Concept & Direction & Zero\\% & MI & δ(0→1) & δ(1→2) \\\\"
    )
    latex_lines.append("\\midrule")

    for _, row in card_sorted.iterrows():
        feature_short = row["feature"].replace("interpretable_", "").replace("_", "\\_")
        concept = row["concept"].replace("_", "\\_")
        direction = row["expected_direction"]

        line = f"{feature_short} & {concept} & {direction} & "
        line += f"{row['zero_pct']:.1f} & "
        line += f"{row['mutual_info']:.3f} & "
        line += (
            f"{row['cliff_delta_01']:.2f} & "
            if pd.notna(row["cliff_delta_01"])
            else "-- & "
        )
        line += (
            f"{row['cliff_delta_12']:.2f} \\\\"
            if pd.notna(row["cliff_delta_12"])
            else "-- \\\\"
        )

        latex_lines.append(line)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    with open(latex_dir / "final_feature_card.tex", "w") as f:
        f.write("\n".join(latex_lines))


def update_column_governance_final(
    drop_features: List[str], drop_reasons: Dict[str, str]
):
    """Update column governance with final pruned features."""
    print("🔄 Updating column governance with final pruning...")

    if not drop_features:
        print("  No additional features to block")
        return

    # Read current column governance
    governance_file = Path(
        "src/corp_speech_risk_dataset/fully_interpretable/column_governance.py"
    )
    with open(governance_file, "r") as f:
        content = f.read()

    # Find the end of BLOCKLIST_PATTERNS
    patterns_start = content.find("BLOCKLIST_PATTERNS = [")
    patterns_end = content.find("]", patterns_start)

    # Create new patterns for dropped features
    new_patterns = []
    for feature in drop_features:
        escaped_feature = feature.replace("_", r"\_")
        pattern = f'    r"^{escaped_feature}$",  # {drop_reasons[feature]}'
        new_patterns.append(pattern)

    # Insert before the closing bracket
    insert_point = content.rfind("\n", patterns_start, patterns_end)
    new_content = (
        content[:insert_point]
        + "\n    # FINAL PRUNING (redundancy removal)\n"
        + "\n".join(new_patterns)
        + content[insert_point:]
    )

    # Write back
    with open(governance_file, "w") as f:
        f.write(new_content)

    print(f"✅ Added {len(drop_features)} pruned features to column governance")


def main():
    """Main function for final feature pruning analysis."""
    print("🚀 Starting final feature pruning analysis...")
    print("=" * 60)

    output_dir = Path("docs/feature_analysis")

    # Load current features and data
    current_features = load_final_features()
    train_df = load_fold3_train_data()

    print(f"📋 Starting with {len(current_features)} features")

    # Group features by concept
    concepts = group_features_by_concept(current_features)
    print(f"🔍 Identified {len(concepts)} feature concepts")

    # Compute redundancy within concepts
    redundancy_results = compute_redundancy_within_concepts(train_df, concepts)

    # Compute per-bucket separation
    separation_results = compute_per_bucket_separation(train_df, current_features)

    # Generate pruning recommendations
    pruning_recommendations = recommend_feature_pruning(
        redundancy_results, separation_results
    )

    final_keep = pruning_recommendations["keep"]
    final_drop = pruning_recommendations["drop"]
    drop_reasons = pruning_recommendations["drop_reasons"]

    print(f"\n📊 PRUNING RESULTS:")
    print(f"   ✅ Final keep: {len(final_keep)} features")
    print(f"   ❌ Final drop: {len(final_drop)} features")

    # Create final feature card
    feature_card = create_final_feature_card(
        final_keep, separation_results, train_df, output_dir
    )

    # Save final results
    final_dir = output_dir / "final_feature_set"

    # Update kept features file
    with open(final_dir / "final_kept_features.txt", "w") as f:
        f.write("# Final Pruned Feature Set for Training\n")
        f.write(f"# Total: {len(final_keep)} features\n")
        f.write("# Applied pruning:\n")
        f.write("# - Removed redundant scales within concepts\n")
        f.write("# - Kept normalized + presence versions\n")
        f.write("# - Dropped weak separation features\n\n")
        for feature in sorted(final_keep):
            f.write(f"{feature}\n")

    # Save pruned features with reasons
    if final_drop:
        pruned_df = pd.DataFrame(
            [{"feature": f, "pruning_reason": drop_reasons[f]} for f in final_drop]
        )
        pruned_df.to_csv(final_dir / "pruned_features.csv", index=False)

        # Update column governance
        update_column_governance_final(final_drop, drop_reasons)

    # Save analysis results
    separation_results.to_csv(final_dir / "separation_analysis.csv", index=False)

    print(f"\n✅ FINAL PRUNING COMPLETE!")
    print(f"📁 Results saved to: {final_dir}")
    print(f"📋 Final feature set: {len(final_keep)} interpretable features")
    print(f"🎯 Ready for training pipeline!")

    # Print final feature list
    print(f"\n🏆 FINAL FEATURES ({len(final_keep)}):")
    for concept in ["lex", "ling", "seq"]:
        concept_features = [
            f for f in final_keep if f.startswith(f"interpretable_{concept}")
        ]
        if concept_features:
            print(
                f"  {concept.upper()} ({len(concept_features)}): {', '.join([f.split('_')[-1] for f in concept_features])}"
            )


if __name__ == "__main__":
    main()
