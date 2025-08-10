"""CLI for interpretable baselines.

Subcommands:
- train: fit and save a model
- predict-dir: recursively predict and write enriched jsonl files mirroring input structure
"""

from __future__ import annotations

import argparse
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
    pt.add_argument("--buckets", nargs="+", default=["Low", "Medium", "High"])
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
    pt.add_argument("--val-split", type=float, default=0.2)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--model", choices=["ridge", "nb", "tree"], default="ridge")
    pt.add_argument(
        "--out",
        required=True,
        help="path to save the trained model (e.g., runs/fi_ridge.joblib)",
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
            val_split=args.val_split,
            seed=args.seed,
            model=args.model,
        )

        logger.info(f"Training interpretable baseline: {cfg}")
        results = train_and_eval(cfg)
        # Save model
        # Rebuild the model with full dataset for final fit
        from .pipeline import build_dataset, make_model

        rows, y, labels = build_dataset(cfg)
        enable_text_branch = bool(cfg.include_text)
        enable_scalar_branch = cfg.include_scalars or cfg.include_numeric
        model = make_model(
            cfg,
            enable_text_branch=enable_text_branch,
            enable_scalar_branch=enable_scalar_branch,
            enable_keywords=cfg.include_keywords,
            enable_speaker=cfg.include_speaker,
        )
        model.fit(rows, y)
        save_model(model, labels, cfg, args.out)
        logger.info({k: results["metrics"][k] for k in results["metrics"]})
        print(
            {
                "accuracy": results["metrics"]["accuracy"],
                "model": args.model,
                "model_path": args.out,
            }
        )
        return

    if args.cmd == "predict-dir":
        out_dir = predict_directory(args.model, args.input_root, args.output_root)
        print({"output_root": str(out_dir)})
        return


if __name__ == "__main__":
    main()
