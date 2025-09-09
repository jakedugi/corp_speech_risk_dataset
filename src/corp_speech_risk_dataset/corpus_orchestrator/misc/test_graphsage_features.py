#!/usr/bin/env python3
"""
Test GraphSAGE 111-dimensional Input Features Through Unified Pipeline

This script extracts the 111 individual GraphSAGE input features and tests them
through the same validation framework used for interpretable features.

GraphSAGE Input Dimensions (111 total):
- BASE_NODE_DIM (16): degree + POS one-hot (12) + position + bias
- GLOBAL_FEATURE_DIM (92): quote/context features (46 each)
- NODE_TYPE_DIM (3): is_token, is_quote_node, is_context_node

Usage:
    python scripts/test_graphsage_features.py \
        --data-dir data/final_stratified_kfold_splits_binary_quote_balanced \
        --output-dir results/graphsage_feature_validation \
        --fold 4 \
        --sample-size 10000
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.encoding.graphembedder import (
    _extract_raw_features,
    _nx_to_pyg,
    BASE_NODE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_TYPE_DIM,
)
from corp_speech_risk_dataset.encoding.parser import to_dependency_graph

# Import the test framework from unified pipeline
from scripts.unified_binary_feature_pipeline import BinaryFeaturePipeline

warnings.filterwarnings("ignore")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)


class GraphSAGEFeatureTester:
    """Test individual GraphSAGE input features through unified validation."""

    def __init__(
        self, data_dir: Path, output_dir: Path, fold: int = 4, sample_size: int = 10000
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fold = fold
        self.sample_size = sample_size

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir = self.output_dir / "extracted_features"
        self.extracted_dir.mkdir(exist_ok=True)

        # GraphSAGE feature definitions
        self.feature_names = self._define_graphsage_features()

        logger.info(f"Initialized GraphSAGE Feature Tester")
        logger.info(f"Total GraphSAGE features to test: {len(self.feature_names)}")

    def _define_graphsage_features(self) -> List[str]:
        """Define all 111 GraphSAGE input feature names."""
        features = []

        # BASE_NODE_DIM (16): degree + POS one-hot (12) + position + bias
        features.append("graphsage_degree")

        pos_tags = [
            "NOUN",
            "VERB",
            "ADJ",
            "ADV",
            "PRON",
            "DET",
            "ADP",
            "CONJ",
            "NUM",
            "PART",
            "INTJ",
            "X",
        ]
        for pos in pos_tags:
            features.append(f"graphsage_pos_{pos}")

        features.append("graphsage_position")
        features.append("graphsage_bias")

        # GLOBAL_FEATURE_DIM (92): quote (46) + context (46)
        global_feature_types = [
            ("sentiment", 3),
            ("deontic_count", 1),
            ("pos", 11),
            ("ner", 7),
            ("deps", 23),
            ("wl", 1),
        ]

        for scope in ["quote", "context"]:
            for feat_type, dims in global_feature_types:
                if dims == 1:
                    features.append(f"graphsage_{scope}_{feat_type}")
                else:
                    for i in range(dims):
                        features.append(f"graphsage_{scope}_{feat_type}_{i}")

        # NODE_TYPE_DIM (3): node type indicators
        for node_type in ["is_token", "is_quote_node", "is_context_node"]:
            features.append(f"graphsage_{node_type}")

        assert len(features) == 111, f"Expected 111 features, got {len(features)}"
        return features

    def extract_graphsage_features(self) -> pd.DataFrame:
        """Extract GraphSAGE features from data and save as DataFrame."""
        logger.info("Extracting GraphSAGE input features...")

        # Load data
        fold_dir = self.data_dir / f"fold_{self.fold}"
        train_path = fold_dir / "train.jsonl"

        extracted_features = []

        with open(train_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc="Processing records")):
                if i >= self.sample_size:
                    break

                record = json.loads(line)
                features = self._extract_single_record_features(record)

                if features is not None:
                    # Add metadata
                    features["outcome_bin"] = record.get("outcome_bin", 0)
                    features["case_id"] = record.get("case_id", f"case_{i}")
                    features["text"] = record.get("text", "")

                    # Add temporal info if available
                    if "date" in record:
                        features["date"] = record["date"]
                    elif "year" in record:
                        features["year"] = record["year"]

                    extracted_features.append(features)

        df = pd.DataFrame(extracted_features)

        # Save extracted features
        output_file = (
            self.extracted_dir / f"graphsage_features_fold_{self.fold}.parquet"
        )
        df.to_parquet(output_file)
        logger.info(f"Saved {len(df)} records with GraphSAGE features to {output_file}")

        return df

    def _extract_single_record_features(
        self, record: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Extract GraphSAGE input features for a single record."""
        try:
            # Get text and build dependency graph
            text = record.get("text", "")
            if not text:
                return None

            g_nx = to_dependency_graph(text)

            # Get raw features (the 92-dim global features)
            raw_features = record.get("raw_features", {})
            quote_vec46, context_vec46 = _extract_raw_features(raw_features)
            global_vec92 = np.concatenate([quote_vec46.numpy(), context_vec46.numpy()])

            # Convert to PyG format to get the actual node features
            pyg = _nx_to_pyg(g_nx, raw_features=raw_features)

            if pyg.num_nodes == 0 or pyg.x is None:
                return None

            # Average node features across all nodes (mean pooling)
            # This gives us the "typical" feature values for this text
            mean_features = pyg.x.mean(dim=0).numpy()

            # Map to feature names
            feature_dict = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_dict[feature_name] = float(mean_features[i])

            return feature_dict

        except Exception as e:
            logger.warning(f"Failed to extract features for record: {e}")
            return None

    def run_unified_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run unified pipeline tests on GraphSAGE features."""
        logger.info("Running unified pipeline tests on GraphSAGE features...")

        # Create a modified dataset with GraphSAGE features prefixed as 'interpretable_'
        # so they get picked up by the unified pipeline
        test_df = df.copy()

        # Rename GraphSAGE features to match pipeline expectations
        rename_mapping = {}
        for feature in self.feature_names:
            if feature in test_df.columns:
                rename_mapping[feature] = f"interpretable_{feature}"

        test_df = test_df.rename(columns=rename_mapping)

        # Create temporary files for the pipeline
        temp_dir = self.output_dir / "temp_pipeline_data"
        temp_dir.mkdir(exist_ok=True)

        fold_dir = temp_dir / f"fold_{self.fold}"
        fold_dir.mkdir(exist_ok=True)

        # Save as JSONL for pipeline compatibility
        train_file = fold_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for _, row in test_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

        # Create dev file (use subset of train for testing)
        dev_file = fold_dir / "dev.jsonl"
        dev_df = test_df.sample(min(1000, len(test_df) // 10), random_state=42)
        with open(dev_file, "w") as f:
            for _, row in dev_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

        # Create empty DNT manifest
        dnt_file = temp_dir / "dnt_manifest.json"
        with open(dnt_file, "w") as f:
            json.dump({"do_not_train": []}, f)

        # Create test directory
        test_dir = temp_dir / "oof_test"
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.jsonl"
        with open(test_file, "w") as f:
            for _, row in dev_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

        # Initialize and run pipeline
        pipeline = BinaryFeaturePipeline(
            data_dir=temp_dir,
            output_dir=self.output_dir / "pipeline_results",
            fold=self.fold,
            sample_size=len(test_df),
            auto_update_governance=False,
        )

        # Load data through pipeline
        train_df_pipeline, dev_df_pipeline, test_df_pipeline = (
            pipeline.load_binary_data()
        )

        # Extract GraphSAGE features for testing
        graphsage_features = [
            col
            for col in train_df_pipeline.columns
            if col.startswith("interpretable_graphsage_")
        ]

        logger.info(f"Testing {len(graphsage_features)} GraphSAGE features")

        # Run comprehensive tests
        test_results = pipeline.run_comprehensive_tests(
            train_df_pipeline, graphsage_features, iteration=1
        )

        # Clean up temp files
        import shutil

        shutil.rmtree(temp_dir)

        return test_results

    def analyze_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results and categorize GraphSAGE features."""
        logger.info("Analyzing GraphSAGE feature test results...")

        # Categorize features by type
        categories = {
            "base_node": [],  # degree, POS, position, bias (16)
            "quote_global": [],  # quote-specific global features (46)
            "context_global": [],  # context-specific global features (46)
            "node_type": [],  # node type indicators (3)
        }

        passed_by_category = {cat: [] for cat in categories}
        failed_by_category = {cat: [] for cat in categories}

        for feature, result in test_results["feature_results"].items():
            # Remove 'interpretable_graphsage_' prefix for analysis
            clean_feature = feature.replace("interpretable_graphsage_", "")

            # Categorize
            if any(x in clean_feature for x in ["degree", "pos_", "position", "bias"]):
                category = "base_node"
            elif clean_feature.startswith("quote_"):
                category = "quote_global"
            elif clean_feature.startswith("context_"):
                category = "context_global"
            elif any(
                x in clean_feature for x in ["is_token", "is_quote", "is_context"]
            ):
                category = "node_type"
            else:
                category = "base_node"  # fallback

            categories[category].append(clean_feature)

            if result["overall_pass"]:
                passed_by_category[category].append(clean_feature)
            else:
                failed_by_category[category].append(clean_feature)

        # Analysis summary
        analysis = {
            "total_tested": len(test_results["feature_results"]),
            "total_passed": len(test_results["summary"]["passed_features"]),
            "total_failed": len(test_results["summary"]["failed_features"]),
            "pass_rate": len(test_results["summary"]["passed_features"])
            / len(test_results["feature_results"]),
            "categories": categories,
            "passed_by_category": passed_by_category,
            "failed_by_category": failed_by_category,
            "failure_analysis": test_results["summary"]["failure_reasons"],
        }

        # Category-wise pass rates
        for cat in categories:
            total_in_cat = len(categories[cat])
            passed_in_cat = len(passed_by_category[cat])
            analysis[f"{cat}_pass_rate"] = (
                passed_in_cat / total_in_cat if total_in_cat > 0 else 0
            )

        return analysis

    def generate_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive report on GraphSAGE feature validation."""
        report_file = self.output_dir / "GRAPHSAGE_FEATURE_VALIDATION_REPORT.md"

        with open(report_file, "w") as f:
            f.write("# GraphSAGE Feature Validation Report\n\n")
            f.write(
                f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(
                f"- **Total GraphSAGE features tested**: {analysis['total_tested']}\n"
            )
            f.write(f"- **Features passed validation**: {analysis['total_passed']}\n")
            f.write(f"- **Features failed validation**: {analysis['total_failed']}\n")
            f.write(f"- **Overall pass rate**: {analysis['pass_rate']:.1%}\n\n")

            # Category Breakdown
            f.write("## Results by Feature Category\n\n")

            categories = {
                "base_node": "Base Node Features (16 expected)",
                "quote_global": "Quote Global Features (46 expected)",
                "context_global": "Context Global Features (46 expected)",
                "node_type": "Node Type Indicators (3 expected)",
            }

            for cat, desc in categories.items():
                pass_rate = analysis.get(f"{cat}_pass_rate", 0)
                passed = len(analysis["passed_by_category"][cat])
                total = len(analysis["categories"][cat])

                f.write(f"### {desc}\n")
                f.write(f"- **Pass rate**: {pass_rate:.1%} ({passed}/{total})\n")

                if analysis["passed_by_category"][cat]:
                    f.write(
                        f"- **Passed**: {', '.join(analysis['passed_by_category'][cat])}\n"
                    )
                if analysis["failed_by_category"][cat]:
                    f.write(
                        f"- **Failed**: {', '.join(analysis['failed_by_category'][cat][:10])}{'...' if len(analysis['failed_by_category'][cat]) > 10 else ''}\n"
                    )
                f.write("\n")

            # Failure Analysis
            f.write("## Failure Analysis\n\n")
            for reason, count in sorted(
                analysis["failure_analysis"].items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"- **{reason.replace('_', ' ').title()}**: {count} features\n")

            # Predictions vs Reality
            f.write("\n## Validation of Predictions\n\n")
            f.write("### Expected Failures:\n")
            f.write("- **bias feature**: Should fail sparsity test (all 1.0 values)\n")
            f.write("- **node type indicators**: Should fail coverage tests (sparse)\n")
            f.write(
                "- **POS features**: Should show temporal drift in legal language\n"
            )
            f.write(
                "- **distribution features**: Should show correlation with case complexity\n\n"
            )

            f.write("### Expected Passes:\n")
            f.write("- **position feature**: Genuine positional signal\n")
            f.write(
                "- **degree feature**: Structural complexity (if not length-biased)\n"
            )
            f.write("- **quote_wl**: Structural hash feature\n")

        logger.success(f"Report generated: {report_file}")

    def run_full_validation(self):
        """Run complete GraphSAGE feature validation pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING GRAPHSAGE FEATURE VALIDATION")
        logger.info("=" * 80)

        # Step 1: Extract features
        df = self.extract_graphsage_features()

        # Step 2: Run unified tests
        test_results = self.run_unified_tests(df)

        # Step 3: Analyze results
        analysis = self.analyze_results(test_results)

        # Step 4: Generate report
        self.generate_report(analysis)

        # Step 5: Save detailed results
        results_file = self.output_dir / "detailed_test_results.json"
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        logger.success("GraphSAGE feature validation completed!")
        return analysis


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test GraphSAGE Input Features Through Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/final_stratified_kfold_splits_binary_quote_balanced",
        help="Directory containing binary k-fold splits",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/graphsage_feature_validation",
        help="Output directory for validation results",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=4,
        help="Fold number to use for testing (default: 4)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Sample size for feature extraction (default: 10000)",
    )

    args = parser.parse_args()

    # Initialize and run tester
    tester = GraphSAGEFeatureTester(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        fold=args.fold,
        sample_size=args.sample_size,
    )

    # Run full validation
    analysis = tester.run_full_validation()

    logger.success("GraphSAGE feature validation completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Overall pass rate: {analysis['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
