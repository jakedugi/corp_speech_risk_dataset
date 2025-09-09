#!/usr/bin/env python3
"""Fix Quote-Level Class Imbalance

This script provides several strategies to address the extreme quote-level
class imbalance while maintaining proper weight inheritance from authoritative data.

Strategies:
1. Case-balanced weighting: Stronger weights for underrepresented classes
2. Quote subsampling: Subsample quotes from overrepresented classes
3. Enhanced class weights: Adjust class weights based on quote-level imbalance
4. Hybrid approach: Combine multiple strategies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
from collections import defaultdict


class ImbalanceCorrector:
    """Tools to correct quote-level class imbalance."""

    def __init__(
        self, data_dir: str = "data/final_stratified_kfold_splits_authoritative"
    ):
        self.data_dir = Path(data_dir)

    def analyze_imbalance(self) -> Dict[str, Any]:
        """Analyze the current imbalance."""
        print("🔍 Analyzing current class imbalance...")

        # Load all training data to understand the full scope
        all_train_data = []

        for fold_name in ["fold_0", "fold_1", "fold_2", "fold_3"]:
            train_file = self.data_dir / fold_name / "train.jsonl"
            if train_file.exists():
                df = pd.read_json(train_file, lines=True)
                df["source_fold"] = fold_name
                all_train_data.append(df)

        if not all_train_data:
            return {"error": "No training data found"}

        combined_df = pd.concat(all_train_data, ignore_index=True)

        # Compute case-level and quote-level statistics
        case_counts = (
            combined_df.groupby("case_id")["outcome_bin"]
            .first()
            .value_counts()
            .sort_index()
        )
        quote_counts = combined_df["outcome_bin"].value_counts().sort_index()

        # Quotes per case by class
        qpc_by_class = {}
        for class_val in sorted(combined_df["outcome_bin"].unique()):
            class_data = combined_df[combined_df["outcome_bin"] == class_val]
            qpc = class_data.groupby("case_id").size()
            qpc_by_class[int(class_val)] = {
                "mean": float(qpc.mean()),
                "std": float(qpc.std()),
                "min": int(qpc.min()),
                "max": int(qpc.max()),
                "median": float(qpc.median()),
                "total_quotes": int(qpc.sum()),
                "total_cases": int(len(qpc)),
            }

        # Imbalance ratios
        total_quotes = quote_counts.sum()
        total_cases = case_counts.sum()

        quote_proportions = {
            int(k): float(v / total_quotes) for k, v in quote_counts.items()
        }
        case_proportions = {
            int(k): float(v / total_cases) for k, v in case_counts.items()
        }

        # Calculate imbalance severity
        quote_max_prop = max(quote_proportions.values())
        quote_min_prop = min(quote_proportions.values())
        quote_imbalance_ratio = (
            quote_max_prop / quote_min_prop if quote_min_prop > 0 else float("inf")
        )

        case_max_prop = max(case_proportions.values())
        case_min_prop = min(case_proportions.values())
        case_imbalance_ratio = (
            case_max_prop / case_min_prop if case_min_prop > 0 else float("inf")
        )

        analysis = {
            "case_level": {
                "counts": {int(k): int(v) for k, v in case_counts.items()},
                "proportions": case_proportions,
                "imbalance_ratio": case_imbalance_ratio,
                "balanced": case_imbalance_ratio < 2.0,
            },
            "quote_level": {
                "counts": {int(k): int(v) for k, v in quote_counts.items()},
                "proportions": quote_proportions,
                "imbalance_ratio": quote_imbalance_ratio,
                "balanced": quote_imbalance_ratio < 2.0,
            },
            "quotes_per_case_by_class": qpc_by_class,
            "severity": (
                "severe"
                if quote_imbalance_ratio > 5.0
                else "moderate" if quote_imbalance_ratio > 2.0 else "mild"
            ),
        }

        # Print analysis
        print(f"\n📊 IMBALANCE ANALYSIS RESULTS")
        print("-" * 50)
        print(
            f"Quote-level imbalance ratio: {quote_imbalance_ratio:.1f}x (severity: {analysis['severity']})"
        )
        print(f"Case-level imbalance ratio: {case_imbalance_ratio:.1f}x")

        print(f"\nQuotes per case by class:")
        for class_val, stats in qpc_by_class.items():
            print(
                f"  Class {class_val}: {stats['mean']:.1f} ± {stats['std']:.1f} quotes/case (range: {stats['min']}-{stats['max']})"
            )

        return analysis

    def suggest_correction_strategies(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest correction strategies based on analysis."""
        strategies = []

        qpc_by_class = analysis.get("quotes_per_case_by_class", {})
        severity = analysis.get("severity", "mild")

        if severity == "severe":
            # Strategy 1: Enhanced class weights (quote-level rebalancing)
            quote_props = analysis.get("quote_level", {}).get("proportions", {})
            if quote_props:
                # Calculate inverse-frequency weights for quote-level balance
                target_prop = 1.0 / len(quote_props)  # Equal representation
                enhanced_weights = {}
                for class_val, prop in quote_props.items():
                    enhanced_weights[class_val] = (
                        target_prop / prop if prop > 0 else 1.0
                    )

                strategies.append(
                    {
                        "name": "enhanced_class_weights",
                        "description": "Apply stronger class weights to counteract quote-level imbalance",
                        "weights": enhanced_weights,
                        "implementation": "Multiply existing class weights by these factors",
                    }
                )

            # Strategy 2: Case-proportional quote weighting
            if qpc_by_class:
                # Weight quotes inversely to case size
                mean_qpc = np.mean([stats["mean"] for stats in qpc_by_class.values()])
                qpc_weights = {}
                for class_val, stats in qpc_by_class.items():
                    qpc_weights[class_val] = (
                        mean_qpc / stats["mean"] if stats["mean"] > 0 else 1.0
                    )

                strategies.append(
                    {
                        "name": "case_proportional_weighting",
                        "description": "Weight quotes inversely proportional to quotes-per-case",
                        "weights": qpc_weights,
                        "implementation": "Apply per-quote weights based on case size",
                    }
                )

            # Strategy 3: Quote subsampling
            if qpc_by_class:
                # Find minimum quotes per case
                min_mean_qpc = min([stats["mean"] for stats in qpc_by_class.values()])
                max_quotes_per_case = min_mean_qpc * 2  # Allow 2x the minimum

                strategies.append(
                    {
                        "name": "quote_subsampling",
                        "description": f"Subsample quotes to max {max_quotes_per_case:.0f} per case",
                        "max_quotes_per_case": max_quotes_per_case,
                        "implementation": "Randomly subsample quotes from large cases",
                    }
                )

        # Strategy 4: Hybrid approach (always applicable)
        strategies.append(
            {
                "name": "hybrid_approach",
                "description": "Combine moderate class reweighting with regularization",
                "class_weight_multiplier": 2.0,  # Moderate boost for rare classes
                "regularization_boost": True,  # Use lower C values
                "implementation": "Safe combination of techniques",
            }
        )

        return strategies

    def apply_enhanced_class_weights(
        self, fold_name: str, enhanced_weights: Dict[int, float]
    ) -> Dict[str, Any]:
        """Apply enhanced class weights to a specific fold."""
        print(f"🔧 Applying enhanced class weights to {fold_name}...")

        # Load existing metadata
        metadata_file = self.data_dir / "per_fold_metadata.json"
        if not metadata_file.exists():
            return {"error": "Metadata file not found"}

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Get existing class weights for this fold
        fold_weights = (
            metadata.get("weights", {}).get(fold_name, {}).get("class_weights", {})
        )
        if not fold_weights:
            return {"error": f"No existing class weights found for {fold_name}"}

        # Apply enhancement multipliers
        enhanced_fold_weights = {}
        for class_str, base_weight in fold_weights.items():
            class_int = int(class_str)
            multiplier = enhanced_weights.get(class_int, 1.0)
            enhanced_fold_weights[class_str] = float(base_weight * multiplier)

        result = {
            "fold": fold_name,
            "original_weights": fold_weights,
            "enhancement_multipliers": enhanced_weights,
            "enhanced_weights": enhanced_fold_weights,
            "change_summary": {},
        }

        # Calculate changes
        for class_str, original in fold_weights.items():
            enhanced = enhanced_fold_weights[class_str]
            change_factor = enhanced / original if original != 0 else float("inf")
            result["change_summary"][class_str] = {
                "original": original,
                "enhanced": enhanced,
                "change_factor": change_factor,
            }

        return result

    def generate_corrected_hyperparams(
        self, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate hyperparameter suggestions to address imbalance."""
        severity = analysis.get("severity", "mild")

        if severity == "severe":
            # More aggressive regularization to prevent majority class overfitting
            corrected_params = {
                "C": [0.001, 0.01, 0.1],  # Lower C values
                "solver": ["lbfgs"],
                "max_iter": [500, 1000],  # More iterations for convergence
                "tol": [1e-5],  # Tighter tolerance
                "class_weight": "balanced",  # Use sklearn's balanced weighting
            }
        elif severity == "moderate":
            corrected_params = {
                "C": [0.01, 0.1, 1.0],  # Moderate regularization
                "solver": ["lbfgs"],
                "max_iter": [200, 500],
                "tol": [1e-4],
                "class_weight": "balanced",
            }
        else:
            corrected_params = {
                "C": [0.1, 1.0, 10.0],  # Standard range
                "solver": ["lbfgs"],
                "max_iter": [200],
                "tol": [1e-4],
            }

        return {
            "severity": severity,
            "recommended_hyperparams": corrected_params,
            "rationale": {
                "lower_C": "Stronger regularization prevents overfitting to majority class",
                "more_iterations": "Allows model to find better balance between classes",
                "balanced_weights": "Sklearn's balanced weighting helps with class imbalance",
            },
        }


def create_updated_config_file(
    analysis: Dict[str, Any], strategies: List[Dict[str, Any]], output_path: Path
):
    """Create an updated configuration file to address imbalance."""

    print("📝 Creating updated configuration recommendations...")

    # Select best strategy
    best_strategy = None
    for strategy in strategies:
        if strategy["name"] == "enhanced_class_weights":
            best_strategy = strategy
            break

    if not best_strategy and strategies:
        best_strategy = strategies[0]  # Fallback to first strategy

    config_update = {
        "problem_diagnosis": {
            "issue": "Quote-level class imbalance causing prediction collapse",
            "quote_level_imbalance_ratio": analysis.get("quote_level", {}).get(
                "imbalance_ratio", 0
            ),
            "case_level_balanced": analysis.get("case_level", {}).get(
                "balanced", False
            ),
            "severity": analysis.get("severity", "unknown"),
        },
        "recommended_solution": best_strategy,
        "implementation_steps": [
            "1. Use enhanced class weights to counteract quote-level imbalance",
            "2. Apply stronger regularization (lower C values)",
            "3. Use more iterations for better convergence",
            "4. Monitor prediction distribution during training",
        ],
        "hyperparameter_adjustments": {
            "C": [0.001, 0.01, 0.1],  # Much stronger regularization
            "max_iter": [500, 1000],
            "class_weight": "enhanced",
        },
    }

    with open(output_path, "w") as f:
        json.dump(config_update, f, indent=2)

    print(f"📄 Configuration recommendations saved to: {output_path}")
    return config_update


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix quote-level class imbalance")
    parser.add_argument(
        "--data-dir", default="data/final_stratified_kfold_splits_authoritative"
    )
    parser.add_argument(
        "--output-config", help="Save configuration recommendations to file"
    )
    parser.add_argument("--save-analysis", help="Save full analysis to JSON file")

    args = parser.parse_args()

    corrector = ImbalanceCorrector(args.data_dir)

    # Analyze current imbalance
    analysis = corrector.analyze_imbalance()

    # Get correction strategies
    strategies = corrector.suggest_correction_strategies(analysis)

    # Generate hyperparameter suggestions
    hyperparam_suggestions = corrector.generate_corrected_hyperparams(analysis)

    # Print strategies
    print(f"\n💡 RECOMMENDED CORRECTION STRATEGIES")
    print("=" * 60)

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name'].replace('_', ' ').title()}")
        print(f"   Description: {strategy['description']}")
        print(f"   Implementation: {strategy['implementation']}")

        if "weights" in strategy:
            weights_str = ", ".join(
                [f"Class {k}: {v:.2f}x" for k, v in strategy["weights"].items()]
            )
            print(f"   Weights: {weights_str}")

    # Print hyperparameter recommendations
    print(f"\n⚙️ HYPERPARAMETER RECOMMENDATIONS")
    print("-" * 40)
    print(f"Severity: {hyperparam_suggestions['severity']}")
    print(f"Recommended hyperparameters:")
    for param, values in hyperparam_suggestions["recommended_hyperparams"].items():
        print(f"  {param}: {values}")

    print(f"\nRationale:")
    for reason, explanation in hyperparam_suggestions["rationale"].items():
        print(f"  - {explanation}")

    # Create configuration file
    if args.output_config:
        config_update = create_updated_config_file(
            analysis, strategies, Path(args.output_config)
        )

    # Save full analysis
    if args.save_analysis:
        full_results = {
            "analysis": analysis,
            "strategies": strategies,
            "hyperparameter_suggestions": hyperparam_suggestions,
        }

        with open(args.save_analysis, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"📄 Full analysis saved to: {args.save_analysis}")

    # Provide immediate action items
    print(f"\n🎯 IMMEDIATE ACTION ITEMS")
    print("=" * 60)
    print("1. Use stronger regularization in hyperparameter search:")
    print(
        '   --custom-hyperparameters \'{"C": [0.001, 0.01, 0.1], "max_iter": [500]}\' '
    )
    print()
    print("2. Consider running with enhanced class weights:")
    print(
        "   (This would require modifying the polar_pipeline.py to use the enhanced weights)"
    )
    print()
    print("3. Monitor prediction distribution during training:")
    print("   Check for prediction collapse in CV logs")
    print()
    print("4. If issue persists, consider quote subsampling approach")


if __name__ == "__main__":
    main()
