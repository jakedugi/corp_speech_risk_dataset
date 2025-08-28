#!/usr/bin/env python3
"""Binary POLR Pipeline Command Helper

This script provides easy access to common binary POLR pipeline commands and configurations.
All commands ensure proper weight inheritance from the binary dataset and use exactly 10 features.

Usage:
    python scripts/binary_polr_commands.py --help
    python scripts/binary_polr_commands.py list
    python scripts/binary_polr_commands.py run <command_name>
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime


class BinaryPOLRCommands:
    """Helper class for common binary POLR pipeline commands."""

    def __init__(self):
        self.commands = {
            "validate": {
                "description": "Quick validation of binary data integrity and weight inheritance",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "validate",
                    "--no-confirmation",
                ],
                "estimated_time": "30 seconds",
                "features": "Binary data validation with 10 features",
            },
            "quick_test": {
                "description": "Fast test run with minimal hyperparameters for binary development",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "fast",
                    "--no-confirmation",
                    "--output-dir",
                    f"runs/binary_polr_quick_test_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "5-8 minutes",
                "features": "Binary classification with 10 features, 4-fold CV + fold_4 final training",
            },
            "cv_only": {
                "description": "Run binary cross-validation for hyperparameter search only (folds 0,1,2,3)",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "cv-only",
                    "--hyperparameters",
                    "default",
                    "--output-dir",
                    f"runs/binary_polr_cv_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "12-18 minutes",
                "features": "4-fold CV on binary data with 10 features",
            },
            "production_polr": {
                "description": "Full binary production pipeline with POLR and thorough hyperparameter search",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "polr",
                    "--hyperparameters",
                    "thorough",
                    "--output-dir",
                    f"runs/binary_polr_production_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "45-75 minutes",
                "features": "Binary POLR with comprehensive hyperparameter search",
            },
            "production_mlr": {
                "description": "Full binary production pipeline with MLR and thorough hyperparameter search",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "mlr",
                    "--hyperparameters",
                    "thorough",
                    "--output-dir",
                    f"runs/binary_mlr_production_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "35-60 minutes",
                "features": "Binary MLR with comprehensive hyperparameter search",
            },
            "paper_ready_polr": {
                "description": "Binary POLR production run optimized for paper publication",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "polr",
                    "--hyperparameters",
                    "default",
                    "--output-dir",
                    f"runs/binary_polr_paper_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "25-40 minutes",
                "features": "Binary POLR with paper-ready outputs and polr_ prefixed predictions",
            },
            "paper_ready_mlr": {
                "description": "Binary MLR production run optimized for paper publication",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "mlr",
                    "--hyperparameters",
                    "default",
                    "--output-dir",
                    f"runs/binary_mlr_paper_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "20-35 minutes",
                "features": "Binary MLR with paper-ready outputs and polr_ prefixed predictions",
            },
            "final_only": {
                "description": "Train final binary model only (requires existing CV results)",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "final-only",
                ],
                "estimated_time": "8-12 minutes",
                "features": "Final training on fold_4 with OOF evaluation",
                "note": "Requires existing CV results in output directory",
            },
            "custom_fast_polr": {
                "description": "Custom binary hyperparameters for very fast POLR testing",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "polr",
                    "--hyperparameters",
                    "custom",
                    "--custom-hyperparameters",
                    '{"C": [1.0], "solver": ["lbfgs"], "max_iter": [100], "tol": [1e-3]}',
                    "--no-confirmation",
                    "--output-dir",
                    f"runs/binary_polr_custom_fast_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "4-6 minutes",
                "features": "Very fast binary POLR testing",
            },
            "custom_fast_mlr": {
                "description": "Custom binary hyperparameters for very fast MLR testing",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--model",
                    "mlr",
                    "--hyperparameters",
                    "custom",
                    "--custom-hyperparameters",
                    '{"C": [1.0], "solver": ["lbfgs"], "max_iter": [100], "tol": [1e-3], "class_weight": [null]}',
                    "--no-confirmation",
                    "--output-dir",
                    f"runs/binary_mlr_custom_fast_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "3-5 minutes",
                "features": "Very fast binary MLR testing",
            },
            "debug_mode": {
                "description": "Binary debug run with extensive validation and logging",
                "cmd": [
                    "python",
                    "scripts/run_binary_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "fast",
                    "--output-dir",
                    f"runs/binary_polr_debug_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "8-12 minutes",
                "features": "Comprehensive validation and debug logging for binary pipeline",
            },
        }

    def list_commands(self):
        """List all available binary commands."""
        print("Available Binary POLR Pipeline Commands:")
        print("=" * 60)
        print("🔥 BINARY CLASSIFICATION: 2 classes (low/high risk)")
        print("📊 FEATURES: Exactly 10 features from importance analysis")
        print("🎯 STRUCTURE: 4-fold CV (0,1,2,3) + fold_4 final training")
        print("⚖️ WEIGHTS: Inherited from binary dataset")
        print("🏷️ OUTPUTS: polr_ prefixed predictions")
        print("=" * 60)

        for name, info in self.commands.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Features: {info['features']}")
            print(f"  Estimated time: {info['estimated_time']}")
            if "note" in info:
                print(f"  Note: {info['note']}")

        print("\nUsage:")
        print("  python scripts/binary_polr_commands.py run <command_name>")
        print("  python scripts/binary_polr_commands.py run production_polr")

        print(f"\n🌟 RECOMMENDED COMMANDS:")
        print("  • validate          - Always start here")
        print("  • quick_test        - Fast development testing")
        print("  • paper_ready_polr  - Production POLR for publication")
        print("  • paper_ready_mlr   - Production MLR for publication")

    def run_command(self, command_name: str, dry_run: bool = False):
        """Run a specific binary command."""
        if command_name not in self.commands:
            print(f"Error: Command '{command_name}' not found.")
            print(
                "Use 'python scripts/binary_polr_commands.py list' to see available commands."
            )
            return False

        cmd_info = self.commands[command_name]
        cmd = cmd_info["cmd"]

        print(f"Running binary command: {command_name}")
        print(f"Description: {cmd_info['description']}")
        print(f"Features: {cmd_info['features']}")
        print(f"Estimated time: {cmd_info['estimated_time']}")

        if "note" in cmd_info:
            print(f"Note: {cmd_info['note']}")

        print(f"\nCommand: {' '.join(cmd)}")

        if dry_run:
            print("\n[DRY RUN] Command would be executed but --dry-run flag is set")
            return True

        # Get user confirmation
        response = input("\nProceed with binary pipeline? [y/N]: ").lower().strip()
        if response not in ["y", "yes"]:
            print("Binary command cancelled.")
            return False

        # Run the command
        try:
            print(
                f"\n🚀 Starting binary '{command_name}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("=" * 80)

            result = subprocess.run(cmd, cwd=Path.cwd())

            if result.returncode == 0:
                print("\n" + "=" * 80)
                print(f"✅ Binary command '{command_name}' completed successfully!")
                print("📊 Check output directory for:")
                print("  • Binary classification results (2 classes)")
                print("  • 10 feature importance analysis")
                print("  • polr_ prefixed predictions")
                print("  • Comprehensive logs and metrics")
            else:
                print("\n" + "=" * 80)
                print(
                    f"❌ Binary command '{command_name}' failed with return code {result.returncode}"
                )
                return False

        except KeyboardInterrupt:
            print("\n❌ Binary command interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Binary command failed with error: {e}")
            return False

        return True

    def get_custom_command(self) -> List[str]:
        """Interactive binary command builder."""
        print("Binary Custom Command Builder")
        print("=" * 40)
        print("🔥 Building command for BINARY CLASSIFICATION")

        # Mode selection
        modes = ["cv-only", "final-only", "full", "validate"]
        print("\nAvailable modes:")
        for i, mode in enumerate(modes, 1):
            print(f"  {i}. {mode}")

        while True:
            try:
                mode_choice = input("Select mode [1-4]: ").strip()
                mode_idx = int(mode_choice) - 1
                if 0 <= mode_idx < len(modes):
                    selected_mode = modes[mode_idx]
                    break
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Model selection
        models = ["polr", "mlr"]
        print(f"\nModel types:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")

        while True:
            try:
                model_choice = input("Select model [1-2]: ").strip()
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(models):
                    selected_model = models[model_idx]
                    break
                else:
                    print("Invalid choice. Please select 1-2.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Hyperparameter preset
        presets = ["default", "fast", "thorough", "custom"]
        print(f"\nHyperparameter presets:")
        for i, preset in enumerate(presets, 1):
            print(f"  {i}. {preset}")

        while True:
            try:
                preset_choice = input("Select preset [1-4]: ").strip()
                preset_idx = int(preset_choice) - 1
                if 0 <= preset_idx < len(presets):
                    selected_preset = presets[preset_idx]
                    break
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Output directory
        output_dir = input(
            "Output directory (leave empty for auto-generated): "
        ).strip()
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = f"runs/binary_polr_custom_{timestamp}"

        # Build command
        cmd = [
            "python",
            "scripts/run_binary_polr_comprehensive.py",
            "--mode",
            selected_mode,
            "--model",
            selected_model,
            "--hyperparameters",
            selected_preset,
            "--output-dir",
            output_dir,
        ]

        # Additional options
        print("\nAdditional options:")
        if input("Skip confirmation prompts? [y/N]: ").lower().strip() in ["y", "yes"]:
            cmd.extend(["--no-confirmation"])

        if input("Skip validation? [y/N]: ").lower().strip() in ["y", "yes"]:
            cmd.extend(["--skip-validation"])

        # Custom hyperparameters if needed
        if selected_preset == "custom":
            print("\nEnter custom hyperparameters as JSON:")
            if selected_model == "mlr":
                print(
                    'MLR Example: {"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [200], "class_weight": ["balanced", null]}'
                )
            else:
                print(
                    'POLR Example: {"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [200]}'
                )
            custom_params = input("Custom hyperparameters: ").strip()
            if custom_params:
                cmd.extend(["--custom-hyperparameters", custom_params])

        print(f"\nGenerated binary command:")
        print(" ".join(cmd))

        return cmd


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Binary POLR Pipeline Command Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="action", help="Available actions")

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List all available binary commands"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a specific binary command")
    run_parser.add_argument("command", help="Binary command name to run")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Show command without executing"
    )

    # Custom command
    custom_parser = subparsers.add_parser(
        "custom", help="Build custom binary command interactively"
    )
    custom_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the custom binary command immediately",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show detailed information about the binary pipeline"
    )

    return parser


def show_binary_pipeline_info():
    """Show detailed binary pipeline information."""
    print("Binary POLR Pipeline Information")
    print("=" * 60)

    print("\n🔥 BINARY CLASSIFICATION:")
    print("  - Classes: 2 (low risk = 0, high risk = 1)")
    print("  - Target: outcome_bin field")
    print("  - Bucket names: ['low', 'high']")

    print("\n📊 DATA REQUIREMENTS:")
    print(
        "  - Uses binary data from: data/final_stratified_kfold_splits_binary_quote_balanced/"
    )
    print("  - Requires precomputed outcome_bin labels (0/1)")
    print("  - Requires precomputed sample_weight values")
    print(
        "  - Must have per_fold_metadata.json with fold-specific boundaries and class weights"
    )

    print("\n📋 FEATURE GOVERNANCE:")
    print("  - Exactly 10 features from importance analysis:")
    print("    1. feat_new_attribution_verb_density")
    print("    2. feat_new2_deception_cluster_density")
    print("    3. interpretable_lex_deception_count")
    print("    4. interpretable_seq_trans_neutral_to_neutral")
    print("    5. feat_new3_neutral_edge_coverage")
    print("    6. feat_new2_attribution_verb_clustering_score")
    print("    7. feat_new5_deesc_half_life")
    print("    8. feat_new2_attr_verb_near_neutral_transition")
    print("    9. feat_new2_neutral_run_mean")
    print("    10. feat_new2_neutral_to_deception_transition_rate")

    print("\n🔄 PIPELINE STRUCTURE:")
    print("  - cv-only: Hyperparameter search on folds 0,1,2,3 (4-fold CV)")
    print("  - final-only: Train final model on fold_4 + evaluate on OOF test")
    print("  - full: Complete pipeline (4-fold CV + final + evaluation)")
    print("  - validate: Check binary data integrity and weight inheritance")

    print("\n⚙️ MODEL OPTIONS:")
    print("  - POLR: Proportional Odds Logistic Regression (ordered logistic)")
    print("  - MLR: Multinomial Logistic Regression (standard multiclass)")

    print("\n⚙️ HYPERPARAMETER PRESETS:")
    print("  - fast: Minimal search for quick testing (C=[1.0])")
    print("  - default: Balanced search (C=[0.01, 1, 100])")
    print("  - thorough: Extensive search (C=[0.001...100], multiple solvers)")
    print("  - custom: User-defined JSON configuration")

    print("\n📈 GENERATED OUTPUTS:")
    print(
        "  - Binary classification metrics (accuracy, precision, recall, F1, ROC-AUC)"
    )
    print("  - OOF predictions in JSONL format with polr_ prefix")
    print("  - Model artifacts (final_binary_*_model.joblib, etc.)")
    print("  - Comprehensive logs with timestamps")

    print("\n🛡️ SAFETY FEATURES:")
    print("  - Binary data integrity validation before training")
    print("  - User confirmation prompts for critical operations")
    print("  - Feature governance (exactly 10 features enforced)")
    print("  - Weight inheritance verification")

    print("\n💡 BEST PRACTICES:")
    print("  1. Always start with 'validate' mode to check binary data")
    print("  2. Use 'quick_test' for development and debugging")
    print("  3. Use 'production_polr' or 'production_mlr' for final results")
    print("  4. Check logs in output_dir/binary_polr_comprehensive_*.log for issues")

    print(f"\n🔧 Example Usage:")
    print("  python scripts/binary_polr_commands.py list")
    print("  python scripts/binary_polr_commands.py run validate")
    print("  python scripts/binary_polr_commands.py run quick_test")
    print("  python scripts/binary_polr_commands.py run paper_ready_polr")
    print("  python scripts/binary_polr_commands.py run paper_ready_mlr")

    print(f"\n📊 EXPECTED OUTPUTS:")
    print("  • Binary classification with 2 classes (low=0, high=1)")
    print("  • 10 features from importance analysis")
    print("  • polr_ prefixed predictions")
    print("  • 4-fold CV + fold_4 final training structure")
    print("  • Inherited sample and class weights")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    commands = BinaryPOLRCommands()

    if args.action == "list":
        commands.list_commands()
    elif args.action == "run":
        commands.run_command(args.command, dry_run=args.dry_run)
    elif args.action == "custom":
        custom_cmd = commands.get_custom_command()

        if args.execute:
            # Execute the custom command
            try:
                print(f"\n🚀 Executing custom binary command...")
                result = subprocess.run(custom_cmd, cwd=Path.cwd())
                if result.returncode == 0:
                    print("✅ Custom binary command completed successfully!")
                else:
                    print(
                        f"❌ Custom binary command failed with return code {result.returncode}"
                    )
            except KeyboardInterrupt:
                print("\n❌ Binary command interrupted by user")
            except Exception as e:
                print(f"\n❌ Binary command failed with error: {e}")
        else:
            print("\nTo execute this binary command, run:")
            print(" ".join(custom_cmd))
            print("\nOr use: python scripts/binary_polr_commands.py custom --execute")

    elif args.action == "info":
        show_binary_pipeline_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
