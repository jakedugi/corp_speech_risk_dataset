"""CLI for interpretable baselines.

Subcommands:
- train: fit and save a model
- predict-dir: recursively predict and write enriched jsonl files mirroring input structure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from loguru import logger

from .pipeline import (
    InterpretableConfig,
    train_and_eval,
    save_model,
    load_model,
    predict_directory,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpretable baselines (TF‑IDF + scalars)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train subcommand
    pt = sub.add_parser("train", help="train a model and save it")
    pt.add_argument("--data", required=True, help="path to jsonl dataset")
    pt.add_argument("--label-key", default="bucket")
    pt.add_argument("--buckets", nargs="+", default=["low", "medium", "high"])
    pt.add_argument(
        "--text-keys",
        nargs="+",
        default=["text"],
        help="text fields to use (e.g., text context)",
    )
    pt.add_argument(
        "--no-text", action="store_true", help="disable text TF‑IDF branch entirely"
    )
    pt.add_argument(
        "--include-scalars",
        action="store_true",
        help="append flattened raw_features scalars into numeric branch",
    )
    pt.add_argument(
        "--no-numeric",
        action="store_true",
        help="exclude simple numeric count features",
    )
    pt.add_argument(
        "--no-keywords", action="store_true", help="exclude keywords features"
    )
    pt.add_argument("--no-speaker", action="store_true", help="exclude speaker feature")
    pt.add_argument(
        "--numeric-keys",
        nargs="+",
        default=None,
        help="explicit list of numeric keys to include (overrides defaults)",
    )
    # Enhanced features
    pt.add_argument(
        "--no-lexicons", action="store_true", help="disable risk lexicon features"
    )
    pt.add_argument(
        "--no-sequence", action="store_true", help="disable sequence modeling features"
    )
    pt.add_argument(
        "--no-linguistic", action="store_true", help="disable linguistic features"
    )
    pt.add_argument(
        "--no-structural", action="store_true", help="disable structural features"
    )
    pt.add_argument("--val-split", type=float, default=0.2)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument(
        "--model",
        choices=[
            "ridge",
            "nb",
            "tree",
            "linreg",
            "polr",
            "ebm",
            "logistic",
            "ensemble",
        ],
        default="polr",
        help="model type (default: polr for ordinal regression)",
    )
    pt.add_argument(
        "--no-calibration", action="store_true", help="disable probability calibration"
    )
    pt.add_argument(
        "--feature-selection",
        action="store_true",
        default=True,
        help="enable feature selection (default: True)",
    )
    pt.add_argument(
        "--n-features", type=int, default=5000, help="max features for selection"
    )
    pt.add_argument("--output-dir", help="directory for reports and figures")
    pt.add_argument(
        "--no-validation", action="store_true", help="skip validation experiments"
    )
    pt.add_argument(
        "--no-interpretability",
        action="store_true",
        help="skip interpretability report generation",
    )
    pt.add_argument(
        "--n-jobs", type=int, default=-1, help="parallel jobs (-1 for all cores)"
    )
    pt.add_argument(
        "--out",
        required=True,
        help="path to save the trained model (e.g., runs/fi_polr.joblib)",
    )

    # predict-dir subcommand
    pp = sub.add_parser(
        "predict-dir", help="predict recursively over a directory of jsonl files"
    )
    pp.add_argument("--model", required=True, help="path to saved model (joblib)")
    pp.add_argument(
        "--input-root",
        required=True,
        help="input directory root to search for .jsonl files",
    )
    pp.add_argument(
        "--output-root",
        default=None,
        help="optional explicit output directory root; defaults to <input>_fi_<model>",
    )

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "train":
        cfg = InterpretableConfig(
            data_path=args.data,
            label_key=args.label_key,
            buckets=tuple(args.buckets),
            text_keys=tuple(args.text_keys),
            include_text=not args.no_text,
            include_scalars=bool(args.include_scalars),
            include_numeric=not args.no_numeric,
            include_keywords=not args.no_keywords,
            include_speaker=not args.no_speaker,
            numeric_keys=tuple(args.numeric_keys) if args.numeric_keys else None,
            # Enhanced features
            include_lexicons=not args.no_lexicons,
            include_sequence=not args.no_sequence,
            include_linguistic=not args.no_linguistic,
            include_structural=not args.no_structural,
            val_split=args.val_split,
            seed=args.seed,
            model=args.model,
            calibrate=not args.no_calibration,
            feature_selection=args.feature_selection,
            n_features=args.n_features,
            n_jobs=args.n_jobs,
            generate_report=bool(args.output_dir),
            output_dir=args.output_dir,
        )

        logger.info(f"Training enhanced interpretable model: {args.model}")

        # Run enhanced training and evaluation
        results = train_and_eval(
            cfg,
            run_validation=not args.no_validation,
            run_interpretability=not args.no_interpretability,
            case_outcomes=None,  # Could load from file if available
        )

        # Save final model trained on full dataset
        # Rebuild with full dataset
        from .pipeline import build_dataset, make_model

        rows, y, labels, feature_extractor = build_dataset(cfg)

        # Determine which branches to enable
        enable_text_branch = bool(cfg.include_text)
        enable_scalar_branch = (
            cfg.include_scalars
            or cfg.include_numeric
            or any(
                [
                    cfg.include_lexicons,
                    cfg.include_sequence,
                    cfg.include_linguistic,
                    cfg.include_structural,
                ]
            )
        )

        model = make_model(
            cfg,
            enable_text_branch=enable_text_branch,
            enable_scalar_branch=enable_scalar_branch,
            enable_keywords=cfg.include_keywords,
            enable_speaker=cfg.include_speaker,
        )

        logger.info("Training final model on full dataset...")
        model.fit(rows, y)

        # Save with feature extractor
        save_model(model, labels, cfg, args.out, feature_extractor)

        # Print summary metrics
        metrics_summary = {
            "model": args.model,
            "accuracy": results["metrics"]["accuracy"],
            "qwk": results["metrics"].get("qwk", "N/A"),
            "mae": results["metrics"].get("mae", "N/A"),
            "model_path": args.out,
        }

        if args.output_dir:
            metrics_summary["reports_dir"] = args.output_dir

        logger.info(f"Training complete: {metrics_summary}")
        print(json.dumps(metrics_summary, indent=2))
        return

    if args.cmd == "predict-dir":
        out_dir = predict_directory(args.model, args.input_root, args.output_root)
        print({"output_root": str(out_dir)})
        return


if __name__ == "__main__":
    main()
