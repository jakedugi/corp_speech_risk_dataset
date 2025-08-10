"""CLI for case-level modeling from quote-level risk predictions.

Example usage:
    uv run python -m corp_speech_risk_dataset.case_aggregation.modeling.cli \
        --quotes-dir /Users/.../data/final_destination/courtlistener_v6_fused_raw_coral_pred \
        --output-dir /Users/.../data/reports/case_modeling \
        --thresholds docket_half token_2500

Outputs:
- For each threshold: JSON of evaluation metrics per model
- Combined CSV of per-case features and labels per threshold (optional)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from loguru import logger

from .utils import load_quotes_dir
from .dataset import (
    DEFAULT_THRESHOLDS,
    ThresholdSpec,
    build_datasets,
    FeatureConfig,
    DatasetBundle,
)
from .models import evaluate_models, evaluate_regressors
from . import reporting


def _parse_thresholds(names: List[str]) -> List[ThresholdSpec]:
    defaults = {t.name: t for t in DEFAULT_THRESHOLDS}
    specs: List[ThresholdSpec] = []
    for n in names:
        if n in defaults:
            specs.append(defaults[n])
        else:
            raise SystemExit(f"Unknown threshold name: {n}. Known: {sorted(defaults)}")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval case-level outcome models")
    parser.add_argument(
        "--quotes-dir",
        required=True,
        help="Directory of quotes JSONL with risk probs and positions",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write evaluation artifacts"
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        default=[t.name for t in DEFAULT_THRESHOLDS],
        help="Subset of thresholds to evaluate (by name)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    # Feature toggles
    parser.add_argument(
        "--with-counts", action="store_true", help="Include label count features"
    )
    parser.add_argument(
        "--with-mean-probs",
        action="store_true",
        help="Include mean probability features",
    )
    parser.add_argument(
        "--with-confidence",
        action="store_true",
        help="Include mean confidence feature if present",
    )
    parser.add_argument(
        "--with-pred-class",
        action="store_true",
        help="Include mean predicted class feature if present",
    )
    parser.add_argument(
        "--with-scores",
        action="store_true",
        help="Include mean coral_scores[0:2] if present",
    )
    parser.add_argument(
        "--with-position-stats",
        action="store_true",
        help="Include position median features",
    )
    parser.add_argument(
        "--with-model-threshold",
        action="store_true",
        help="Include mean model threshold if present",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Write per-threshold features CSV alongside metrics",
    )
    # Experiment controls
    parser.add_argument(
        "--feature-subset",
        choices=[
            "counts",
            "mean_probs",
            "max_severity",
            "density_high",
            "all",
        ],
        default="all",
        help="Run models using only a specific subset of features (or all)",
    )
    parser.add_argument(
        "--only-threshold",
        nargs="*",
        default=None,
        help="Optionally restrict to one or more thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quotes_dir = os.path.abspath(args.quotes_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.add(os.path.join(output_dir, "modeling.log"), rotation="1 day")

    print("Loading quotes (fast orjson loader)...")
    rows = load_quotes_dir(quotes_dir)
    print(f"Loaded {len(rows)} quotes.")
    specs = _parse_thresholds(
        args.thresholds if args.only_threshold is None else args.only_threshold
    )
    feature_cfg = FeatureConfig(
        include_mean_probabilities=args.with_mean_probs,
        include_counts=args.with_counts,
        include_confidence=args.with_confidence,
        include_pred_class=args.with_pred_class,
        include_scores=args.with_scores,
        include_position_stats=args.with_position_stats,
        include_model_threshold=args.with_model_threshold,
    )
    # Apply feature subset selection for experiment mode
    if args.feature_subset != "all":
        feature_cfg = FeatureConfig(
            include_mean_probabilities=(args.feature_subset == "mean_probs"),
            include_counts=(args.feature_subset in {"counts", "density_high"}),
            include_confidence=False,
            include_pred_class=False,
            include_scores=False,
            include_position_stats=False,
            include_model_threshold=False,
        )
    datasets = build_datasets(rows, thresholds=specs, features=feature_cfg)

    if not datasets:
        logger.warning("No datasets produced; check inputs")
        return

    summary = {}
    for name, bundle in datasets.items():
        X, y = bundle.X, bundle.y
        y_reg = bundle.y_reg
        covered = set(bundle.covered_case_ids)
        total = set(bundle.total_case_ids)
        coverage = (len(covered) / len(total)) if total else 0.0
        print(
            f"Training/evaluating models for threshold={name} ... cases={X.height} | coverage={coverage:.2%} ({len(covered)}/{len(total)})"
        )
        results = evaluate_models(X, y, test_size=args.test_size)
        reg_results = evaluate_regressors(X, y_reg, test_size=args.test_size)
        # Persist results as JSON
        out_path = os.path.join(output_dir, f"eval_{name}.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        # Friendlier console summary
        if results:
            rows_print: List[tuple[str, float]] = []
            for mname, info in results.items():
                acc_raw = info.get("accuracy")
                acc_val = float(acc_raw) if isinstance(acc_raw, (int, float)) else 0.0
                rows_print.append((mname, acc_val))
            rows_print.sort(key=lambda x: x[1], reverse=True)
            pretty = ", ".join([f"{mn}={acc:.3f}" for mn, acc in rows_print])
            print(f"Results (accuracy) @coverage={coverage:.2%}: {pretty}")
        logger.info("Wrote evaluation", threshold=name, path=out_path)
        if args.save_features:
            # Write features and labels for further analysis
            X.write_csv(os.path.join(output_dir, f"features_{name}.csv"))
            y.write_csv(os.path.join(output_dir, f"labels_{name}.csv"))
            # Figures per threshold (class distribution)
            thr_dir = os.path.join(output_dir, f"figures_{name}")
            reporting.save_class_distribution(
                y, thr_dir, title=f"Class distribution ({name})"
            )
            # Model comparison plot
            reporting.plot_model_accuracies(
                results,
                os.path.join(thr_dir, "model_accuracies.png"),
                title=f"Model accuracies ({name})",
            )
            # Box plots: outcome_bucket vs mean risk score and density of high risk
            joined = X.join(y, on="case_id", how="inner")
            reporting.boxplot_by_outcome(
                joined,
                "f_mean_risk_score",
                os.path.join(thr_dir, "box_mean_risk_score.png"),
                title=f"Mean risk score by outcome ({name})",
            )
            reporting.boxplot_by_outcome(
                joined,
                "f_density_high",
                os.path.join(thr_dir, "box_density_high.png"),
                title=f"High-risk quote density by outcome ({name})",
            )
            reporting.violin_by_outcome(
                joined,
                "f_mean_risk_score",
                os.path.join(thr_dir, "violin_mean_risk_score.png"),
                title=f"Mean risk score by outcome (violin, {name})",
            )
            reporting.violin_by_outcome(
                joined,
                "f_density_high",
                os.path.join(thr_dir, "violin_density_high.png"),
                title=f"High-risk quote density by outcome (violin, {name})",
            )
            # Spearman correlations with numeric outcome, if available
            joined_reg = X.join(y_reg, on="case_id", how="inner")
            rho1 = reporting.spearman_correlation(
                joined_reg, "f_mean_risk_score", "final_judgement_real"
            )
            rho2 = reporting.spearman_correlation(
                joined_reg, "f_density_high", "final_judgement_real"
            )
            with open(
                os.path.join(thr_dir, "spearman.txt"), "w", encoding="utf-8"
            ) as fp:
                fp.write(
                    f"Spearman(f_mean_risk_score, final_judgement_real) = {rho1}\n"
                )
                fp.write(f"Spearman(f_density_high, final_judgement_real) = {rho2}\n")
        # Small console line
        if results:

            def _acc_of(item: tuple[str, dict[str, object]]) -> float:
                _, info = item
                val = info.get("accuracy")
                return float(val) if isinstance(val, (int, float)) else 0.0

            best_item = max(results.items(), key=_acc_of)  # type: ignore[arg-type]
            best = best_item[0]
            best_acc = _acc_of(best_item)
            # Confusion matrix and per-class metrics for best model
            best_info = best_item[1]
            if isinstance(best_info, dict):
                y_true = best_info.get("y_true")
                y_pred = best_info.get("y_pred")
                report = best_info.get("report")
                if (
                    isinstance(y_true, list)
                    and isinstance(y_pred, list)
                    and isinstance(report, dict)
                ):
                    labels_sorted = sorted(set(y_true) | set(y_pred))
                    thr_dir = os.path.join(output_dir, f"figures_{name}")
                    reporting.plot_confusion_matrix(
                        y_true,
                        y_pred,
                        labels_sorted,
                        out_path_counts=os.path.join(thr_dir, "confusion_counts.png"),
                        out_path_norm=os.path.join(thr_dir, "confusion_norm.png"),
                    )
                    reporting.plot_per_class_metrics(
                        report,
                        labels_sorted,
                        os.path.join(thr_dir, "per_class_metrics.png"),
                        title=f"Per-class metrics ({name}, {best})",
                    )
        else:
            best = None
            best_acc = 0.0
        print(
            f"Best: threshold={name} model={best} accuracy={best_acc:.3f} | coverage={coverage:.2%}"
        )
        # Add regression summary (best r2)
        best_r2 = 0.0
        best_reg = None
        if reg_results:
            for rn, rinfo in reg_results.items():
                r2 = rinfo.get("r2") if isinstance(rinfo, dict) else 0.0
                r2f = float(r2) if isinstance(r2, (int, float)) else 0.0
                if r2f > best_r2:
                    best_r2 = r2f
                    best_reg = rn
        summary[name] = {
            "best_model": best,
            "accuracy": best_acc,
            "coverage": coverage,
            "best_regressor": best_reg if best_reg else "",
            "best_r2": best_r2,
        }

    # Write a combined summary file
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    # Cross-threshold summary figure
    reporting.plot_threshold_summary(
        summary,
        os.path.join(output_dir, "threshold_summary.png"),
        title="Best accuracy by threshold",
    )
    # Executive summary text
    lines: List[str] = []
    lines.append(f"Quotes loaded: {len(rows)}\n")
    lines.append("Per-threshold results:")
    for thr, info in summary.items():
        acc_obj = info.get("accuracy")
        cov_obj = info.get("coverage")
        r2_obj = info.get("best_r2")
        acc = float(acc_obj) if isinstance(acc_obj, (int, float)) else 0.0
        cov = float(cov_obj) if isinstance(cov_obj, (int, float)) else 0.0
        best_model = str(info.get("best_model", ""))
        best_reg = str(info.get("best_regressor", ""))
        r2 = float(r2_obj) if isinstance(r2_obj, (int, float)) else 0.0
        line = f"- {thr}: best_cls={best_model}, acc={acc:.3f}, coverage={cov:.1%}, best_reg={best_reg}, r2={r2:.3f}"
        lines.append(line)
    lines.append("")
    lines.append(
        "Interpretation guide:\n"
        "- Coverage: fraction of cases with at least one quote within the early threshold (cases excluded have no qualifying quotes).\n"
        "- Accuracy: how well early risk signals predict final outcomes; consider class balance via per-class metrics.\n"
        "- Thresholds: docket-based = chronology; token-based = content length control across variable case sizes.\n"
        "- Inputs: quote-level predicted risk probabilities/buckets plus positional features; labels: case-level final_judgement_real bucketed (33/66 quantiles)."
    )
    with open(
        os.path.join(output_dir, "executive_summary.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write("\n".join(lines))


if __name__ == "__main__":
    main()
