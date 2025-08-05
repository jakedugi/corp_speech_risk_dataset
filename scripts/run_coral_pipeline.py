#!/usr/bin/env python3
"""
Complete CORAL pipeline runner script.

This script demonstrates the complete workflow:
1. Prepare CORAL training data from fused embeddings
2. Train CORAL ordinal regression model
3. Run inference on new data
4. Generate comprehensive reports

Usage:
    python scripts/run_coral_pipeline.py \
        --input "data/outcomes/courtlistener_v1/*/doc_*_text_stage9.jsonl" \
        --output-dir runs/coral_complete_pipeline \
        --max-threshold 15500000000 \
        --exclude-speakers "Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA" \
        --epochs 50 \
        --batch-size 64 \
        --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger
import json
import time


def run_command(cmd: list, description: str) -> bool:
    """Run a command and log the results."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.success(f"Completed: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Exit code: {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete CORAL pipeline")

    # Data arguments
    parser.add_argument("--input", required=True, help="Input JSONL pattern")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for all results"
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=15500000000,
        help="Maximum outcome threshold",
    )
    parser.add_argument(
        "--exclude-speakers",
        default="Unknown,Court,FTC,Fed,Plaintiff,State,Commission,Congress,Circuit,FDA",
        help="Comma-separated list of speakers to exclude",
    )

    # Training arguments
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["low", "medium", "high"],
        help="Ordinal buckets",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[768, 512, 256],
        help="Hidden layer dimensions (wider & deeper with residual connections)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/mps/auto)")

    # Pipeline control
    parser.add_argument(
        "--skip-prep", action="store_true", help="Skip data preparation"
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include missing outcome data in training",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "pipeline.log"
    logger.add(log_file, rotation="100 MB")

    logger.info("=" * 60)
    logger.info("CORAL PIPELINE STARTING")
    logger.info("=" * 60)

    # Save pipeline configuration
    config = vars(args)
    config["start_time"] = time.time()

    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Define file paths
    prepared_data_path = output_dir / "coral_training_data.jsonl"
    model_dir = output_dir / "model"
    predictions_path = output_dir / "predictions" / "coral_predictions.jsonl"

    success_steps = []

    # Step 1: Data Preparation
    if not args.skip_prep:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 40)

        prep_cmd = [
            "uv",
            "run",
            "python",
            "scripts/prepare_coral_data.py",
            "--input",
            args.input,
            "--output",
            str(prepared_data_path),
            "--max-threshold",
            str(args.max_threshold),
            "--exclude-speakers",
            args.exclude_speakers,
        ]

        if args.include_missing:
            prep_cmd.append("--include-missing")

        if run_command(prep_cmd, "Data preparation"):
            success_steps.append("data_preparation")
            logger.info(f"Prepared data saved to: {prepared_data_path}")
        else:
            logger.error("Data preparation failed!")
            return 1
    else:
        logger.info("Skipping data preparation")
        if not prepared_data_path.exists():
            logger.error(f"Prepared data not found: {prepared_data_path}")
            return 1
        success_steps.append("data_preparation")

    # Step 2: Model Training
    if not args.skip_train:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 40)

        train_cmd = (
            [
                "uv",
                "run",
                "python",
                "scripts/train_coral_model.py",
                "--data",
                str(prepared_data_path),
                "--output",
                str(model_dir),
                "--buckets",
            ]
            + args.buckets
            + ["--hidden-dims"]
            + [str(d) for d in args.hidden_dims]
            + [
                "--dropout",
                str(args.dropout),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--val-split",
                str(args.val_split),
                "--test-split",
                str(args.test_split),
                "--threshold",
                str(args.threshold),
                "--seed",
                str(args.seed),
            ]
        )

        if args.device:
            train_cmd.extend(["--device", args.device])

        if run_command(train_cmd, "Model training"):
            success_steps.append("model_training")
            logger.info(f"Model saved to: {model_dir}")
        else:
            logger.error("Model training failed!")
            return 1
    else:
        logger.info("Skipping model training")
        if not (model_dir / "best_model.pt").exists():
            logger.error(f"Trained model not found: {model_dir / 'best_model.pt'}")
            return 1
        success_steps.append("model_training")

    # Step 3: Inference
    if not args.skip_inference:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: INFERENCE")
        logger.info("=" * 40)

        # Run inference on the same input data (in practice, you'd use new data)
        inference_cmd = [
            "uv",
            "run",
            "python",
            "scripts/coral_inference.py",
            "--model",
            str(model_dir / "best_model.pt"),
            "--input",
            args.input,
            "--output",
            str(predictions_path),
            "--batch-size",
            str(args.batch_size),
        ]

        if args.device:
            inference_cmd.extend(["--device", args.device])

        if run_command(inference_cmd, "Model inference"):
            success_steps.append("inference")
            logger.info(f"Predictions saved to: {predictions_path}")
        else:
            logger.error("Inference failed!")
            return 1
    else:
        logger.info("Skipping inference")
        success_steps.append("inference")

    # Step 4: Generate Final Report
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: FINAL REPORT")
    logger.info("=" * 40)

    # Collect results
    results = {
        "pipeline_config": config,
        "success_steps": success_steps,
        "end_time": time.time(),
        "output_files": {},
    }

    # Check for output files
    if prepared_data_path.exists():
        results["output_files"]["prepared_data"] = str(prepared_data_path)

    if (model_dir / "best_model.pt").exists():
        results["output_files"]["best_model"] = str(model_dir / "best_model.pt")
        results["output_files"]["model_config"] = str(model_dir / "config.json")

    if (model_dir / "test_results.json").exists():
        with open(model_dir / "test_results.json", "r") as f:
            results["test_results"] = json.load(f)
        results["output_files"]["test_results"] = str(model_dir / "test_results.json")

    if predictions_path.exists():
        results["output_files"]["predictions"] = str(predictions_path)

    if (predictions_path.parent / f"{predictions_path.stem}_analysis.json").exists():
        analysis_path = (
            predictions_path.parent / f"{predictions_path.stem}_analysis.json"
        )
        with open(analysis_path, "r") as f:
            results["prediction_analysis"] = json.load(f)
        results["output_files"]["prediction_analysis"] = str(analysis_path)

    # Calculate runtime
    runtime = results["end_time"] - results["start_time"]
    results["runtime_seconds"] = runtime
    results["runtime_formatted"] = (
        f"{runtime // 3600:.0f}h {(runtime % 3600) // 60:.0f}m {runtime % 60:.1f}s"
    )

    # Save final report
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Final report saved to: {report_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {results['runtime_formatted']}")
    logger.info(f"Successful steps: {', '.join(success_steps)}")

    if "test_results" in results:
        test_metrics = results["test_results"]["test_metrics"]
        logger.info(f"Model performance:")
        logger.info(f"  Exact accuracy: {test_metrics['exact']:.3f}")
        logger.info(f"  Off-by-one accuracy: {test_metrics['off_by_one']:.3f}")
        logger.info(f"  Spearman correlation: {test_metrics['spearman_r']:.3f}")

    if "prediction_analysis" in results:
        pred_analysis = results["prediction_analysis"]
        logger.info(f"Predictions made: {pred_analysis['total_predictions']}")
        logger.info(f"Prediction distribution:")
        for bucket, count in pred_analysis["bucket_distribution"]["counts"].items():
            pct = pred_analysis["bucket_distribution"]["percentages"][bucket]
            logger.info(f"  {bucket}: {count} ({pct:.1f}%)")

    logger.info("\nOutput files:")
    for file_type, file_path in results["output_files"].items():
        logger.info(f"  {file_type}: {file_path}")

    logger.success("CORAL pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
