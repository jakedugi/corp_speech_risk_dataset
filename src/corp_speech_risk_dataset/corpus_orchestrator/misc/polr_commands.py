#!/usr/bin/env python3
"""POLR Pipeline Command Helper

This script provides easy access to common POLR pipeline commands and configurations.
All commands ensure proper weight inheritance from the authoritative dataset.

Usage:
    python scripts/polr_commands.py --help
    python scripts/polr_commands.py list
    python scripts/polr_commands.py run <command_name>
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime


class POLRCommands:
    """Helper class for common POLR pipeline commands."""

    def __init__(self):
        self.commands = {
            "validate": {
                "description": "Quick validation of data integrity and weight inheritance",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "validate",
                    "--no-confirmation",
                ],
                "estimated_time": "30 seconds",
            },
            "quick_test": {
                "description": "Fast test run with minimal hyperparameters for development",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "fast",
                    "--no-confirmation",
                    "--output-dir",
                    f"runs/polr_quick_test_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "3-5 minutes",
            },
            "cv_only": {
                "description": "Run cross-validation for hyperparameter search only (folds 0,1,2)",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "cv-only",
                    "--hyperparameters",
                    "default",
                    "--output-dir",
                    f"runs/polr_cv_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "10-15 minutes",
            },
            "production": {
                "description": "Full production pipeline with thorough hyperparameter search",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "thorough",
                    "--output-dir",
                    f"runs/polr_production_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "30-60 minutes",
            },
            "paper_ready": {
                "description": "Production run optimized for paper publication",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "default",
                    "--generate-assets",
                    "--output-dir",
                    f"runs/polr_paper_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "20-35 minutes",
            },
            "final_only": {
                "description": "Train final model only (requires existing CV results)",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "final-only",
                ],
                "estimated_time": "5-10 minutes",
                "note": "Requires existing CV results in output directory",
            },
            "assets_only": {
                "description": "Generate paper assets from existing results",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "assets",
                ],
                "estimated_time": "2-5 minutes",
                "note": "Requires existing model results in output directory",
            },
            "custom_fast": {
                "description": "Custom hyperparameters for very fast testing",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "custom",
                    "--custom-hyperparameters",
                    '{"C": [1.0], "solver": ["lbfgs"], "max_iter": [100], "tol": [1e-3]}',
                    "--no-confirmation",
                    "--output-dir",
                    f"runs/polr_custom_fast_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "2-3 minutes",
            },
            "debug_mode": {
                "description": "Debug run with extensive validation and logging",
                "cmd": [
                    "python",
                    "scripts/run_polr_comprehensive.py",
                    "--mode",
                    "full",
                    "--hyperparameters",
                    "fast",
                    "--output-dir",
                    f"runs/polr_debug_{datetime.now().strftime('%Y%m%d_%H%M')}",
                ],
                "estimated_time": "5-8 minutes",
            },
        }

    def list_commands(self):
        """List all available commands."""
        print("Available POLR Pipeline Commands:")
        print("=" * 50)

        for name, info in self.commands.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Estimated time: {info['estimated_time']}")
            if "note" in info:
                print(f"  Note: {info['note']}")

        print("\nUsage:")
        print("  python scripts/polr_commands.py run <command_name>")
        print("  python scripts/polr_commands.py run production")

    def run_command(self, command_name: str, dry_run: bool = False):
        """Run a specific command."""
        if command_name not in self.commands:
            print(f"Error: Command '{command_name}' not found.")
            print(
                "Use 'python scripts/polr_commands.py list' to see available commands."
            )
            return False

        cmd_info = self.commands[command_name]
        cmd = cmd_info["cmd"]

        print(f"Running command: {command_name}")
        print(f"Description: {cmd_info['description']}")
        print(f"Estimated time: {cmd_info['estimated_time']}")

        if "note" in cmd_info:
            print(f"Note: {cmd_info['note']}")

        print(f"\nCommand: {' '.join(cmd)}")

        if dry_run:
            print("\n[DRY RUN] Command would be executed but --dry-run flag is set")
            return True

        # Get user confirmation
        response = input("\nProceed? [y/N]: ").lower().strip()
        if response not in ["y", "yes"]:
            print("Command cancelled.")
            return False

        # Run the command
        try:
            print(
                f"\n🚀 Starting '{command_name}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print("=" * 60)

            result = subprocess.run(cmd, cwd=Path.cwd())

            if result.returncode == 0:
                print("\n" + "=" * 60)
                print(f"✅ Command '{command_name}' completed successfully!")
            else:
                print("\n" + "=" * 60)
                print(
                    f"❌ Command '{command_name}' failed with return code {result.returncode}"
                )
                return False

        except KeyboardInterrupt:
            print("\n❌ Command interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Command failed with error: {e}")
            return False

        return True

    def get_custom_command(self) -> List[str]:
        """Interactive command builder."""
        print("Custom Command Builder")
        print("=" * 30)

        # Mode selection
        modes = ["cv-only", "final-only", "full", "validate", "assets"]
        print("\nAvailable modes:")
        for i, mode in enumerate(modes, 1):
            print(f"  {i}. {mode}")

        while True:
            try:
                mode_choice = input("Select mode [1-5]: ").strip()
                mode_idx = int(mode_choice) - 1
                if 0 <= mode_idx < len(modes):
                    selected_mode = modes[mode_idx]
                    break
                else:
                    print("Invalid choice. Please select 1-5.")
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
            output_dir = f"runs/polr_custom_{timestamp}"

        # Build command
        cmd = [
            "python",
            "scripts/run_polr_comprehensive.py",
            "--mode",
            selected_mode,
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

        if selected_mode in ["full", "assets"]:
            if input("Skip asset generation? [y/N]: ").lower().strip() in ["y", "yes"]:
                cmd.extend(["--no-assets"])

        # Custom hyperparameters if needed
        if selected_preset == "custom":
            print("\nEnter custom hyperparameters as JSON:")
            print('Example: {"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [200]}')
            custom_params = input("Custom hyperparameters: ").strip()
            if custom_params:
                cmd.extend(["--custom-hyperparameters", custom_params])

        print(f"\nGenerated command:")
        print(" ".join(cmd))

        return cmd


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="POLR Pipeline Command Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="action", help="Available actions")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a specific command")
    run_parser.add_argument("command", help="Command name to run")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Show command without executing"
    )

    # Custom command
    custom_parser = subparsers.add_parser(
        "custom", help="Build custom command interactively"
    )
    custom_parser.add_argument(
        "--execute", action="store_true", help="Execute the custom command immediately"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show detailed information about the pipeline"
    )

    return parser


def show_pipeline_info():
    """Show detailed pipeline information."""
    print("POLR Pipeline Information")
    print("=" * 50)

    print("\n📊 Data Requirements:")
    print(
        "  - Uses authoritative data from: data/final_stratified_kfold_splits_authoritative/"
    )
    print("  - Requires precomputed outcome_bin labels")
    print("  - Requires precomputed sample_weight values")
    print(
        "  - Must have per_fold_metadata.json with fold-specific boundaries and class weights"
    )

    print("\n🔄 Pipeline Modes:")
    print("  - cv-only: Hyperparameter search on folds 0,1,2")
    print("  - final-only: Train final model on fold 3 + evaluate on OOF test")
    print("  - full: Complete pipeline (CV + final + evaluation + assets)")
    print("  - validate: Check data integrity and weight inheritance")
    print("  - assets: Generate paper assets from existing results")

    print("\n⚙️ Hyperparameter Presets:")
    print("  - fast: Minimal search for quick testing (C=[1.0])")
    print("  - default: Balanced search (C=[0.01, 1, 100])")
    print("  - thorough: Extensive search (C=[0.001...100], multiple solvers)")
    print("  - custom: User-defined JSON configuration")

    print("\n📈 Generated Assets:")
    print("  - LaTeX tables (T1-T10) with comprehensive metrics")
    print("  - PDF figures (F1-F10) for paper inclusion")
    print("  - OOF predictions in JSONL format with polr_ prefix")
    print("  - Model artifacts (final_polar_model.joblib, etc.)")

    print("\n🛡️ Safety Features:")
    print("  - Data integrity validation before training")
    print("  - User confirmation prompts for critical operations")
    print("  - Comprehensive logging with timestamps")
    print("  - Weight inheritance verification")

    print("\n💡 Best Practices:")
    print("  1. Always start with 'validate' mode to check data")
    print("  2. Use 'quick_test' for development and debugging")
    print("  3. Use 'production' or 'paper_ready' for final results")
    print("  4. Check logs in output_dir/polr_comprehensive_*.log for issues")

    print(f"\n🔧 Example Usage:")
    print("  python scripts/polr_commands.py list")
    print("  python scripts/polr_commands.py run validate")
    print("  python scripts/polr_commands.py run quick_test")
    print("  python scripts/polr_commands.py run paper_ready")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    commands = POLRCommands()

    if args.action == "list":
        commands.list_commands()
    elif args.action == "run":
        commands.run_command(args.command, dry_run=args.dry_run)
    elif args.action == "custom":
        custom_cmd = commands.get_custom_command()

        if args.execute:
            # Execute the custom command
            try:
                print(f"\n🚀 Executing custom command...")
                result = subprocess.run(custom_cmd, cwd=Path.cwd())
                if result.returncode == 0:
                    print("✅ Custom command completed successfully!")
                else:
                    print(
                        f"❌ Custom command failed with return code {result.returncode}"
                    )
            except KeyboardInterrupt:
                print("\n❌ Command interrupted by user")
            except Exception as e:
                print(f"\n❌ Command failed with error: {e}")
        else:
            print("\nTo execute this command, run:")
            print(" ".join(custom_cmd))
            print("\nOr use: python scripts/polr_commands.py custom --execute")

    elif args.action == "info":
        show_pipeline_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
