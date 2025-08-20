#!/usr/bin/env python3
"""
Calculate actual economic impact within each tertile zone for all folds.
This will generate comprehensive statistics for the LaTeX document.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_case_outcomes(data_file):
    """Load case outcomes from the main dataset."""
    case_outcomes = {}
    with open(data_file) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                case_id = record.get("case_id_clean")
                outcome = record.get("final_judgement_real")
                if case_id and outcome is not None:
                    case_outcomes[case_id] = float(outcome)
    return case_outcomes


def calculate_fold_economic_impact(fold_dir, case_outcomes, tertile_boundaries):
    """Calculate economic impact for each tertile in a fold."""
    case_ids_file = fold_dir / "case_ids.json"

    if not case_ids_file.exists():
        return None

    with open(case_ids_file) as f:
        case_ids = json.load(f)

    # Get training cases for this fold
    train_case_ids = case_ids.get("train_case_ids", [])

    # Calculate economic impact by tertile
    low_total = medium_total = high_total = 0.0
    low_count = medium_count = high_count = 0
    low_cases = []
    medium_cases = []
    high_cases = []

    for case_id in train_case_ids:
        if case_id in case_outcomes:
            outcome = case_outcomes[case_id]
            if outcome <= tertile_boundaries[0]:
                low_total += outcome
                low_count += 1
                low_cases.append((case_id, outcome))
            elif outcome <= tertile_boundaries[1]:
                medium_total += outcome
                medium_count += 1
                medium_cases.append((case_id, outcome))
            else:
                high_total += outcome
                high_count += 1
                high_cases.append((case_id, outcome))

    return {
        "low": {"total": low_total, "count": low_count, "cases": low_cases},
        "medium": {"total": medium_total, "count": medium_count, "cases": medium_cases},
        "high": {"total": high_total, "count": high_count, "cases": high_cases},
        "boundaries": tertile_boundaries,
        "train_cases_total": len(train_case_ids),
    }


def main():
    # Paths
    data_file = "data/enhanced_combined/final_clean_dataset_no_bankruptcy.jsonl"
    kfold_dir = Path("data/final_stratified_kfold_splits_adaptive_oof")
    metadata_file = kfold_dir / "per_fold_metadata.json"

    print("Loading case outcomes...")
    case_outcomes = load_case_outcomes(data_file)
    print(f"Loaded {len(case_outcomes)} case outcomes")

    print("Loading fold metadata...")
    with open(metadata_file) as f:
        metadata = json.load(f)

    fold_boundaries = metadata["binning"]["fold_edges"]

    print("Calculating economic impact by tertiles for each fold...")

    results = {}

    for fold_name, boundaries in fold_boundaries.items():
        fold_num = fold_name.split("_")[1]
        fold_dir = kfold_dir / f"fold_{fold_num}"

        print(f"Processing {fold_name}...")
        impact = calculate_fold_economic_impact(fold_dir, case_outcomes, boundaries)

        if impact:
            results[fold_name] = impact

            # Print summary
            print(f"  {fold_name}:")
            print(f"    Boundaries: ${boundaries[0]:,.0f} | ${boundaries[1]:,.0f}")
            print(
                f"    Low tertile: {impact['low']['count']} cases, ${impact['low']['total']:,.0f} total"
            )
            print(
                f"    Medium tertile: {impact['medium']['count']} cases, ${impact['medium']['total']:,.0f} total"
            )
            print(
                f"    High tertile: {impact['high']['count']} cases, ${impact['high']['total']:,.0f} total"
            )
            print(
                f"    Grand total: ${impact['low']['total'] + impact['medium']['total'] + impact['high']['total']:,.0f}"
            )
            print()

    # Save results
    output_file = "tertile_economic_impact_analysis.json"
    with open(output_file, "w") as f:
        # Remove case details for JSON serialization
        clean_results = {}
        for fold_name, data in results.items():
            clean_results[fold_name] = {
                "low": {"total": data["low"]["total"], "count": data["low"]["count"]},
                "medium": {
                    "total": data["medium"]["total"],
                    "count": data["medium"]["count"],
                },
                "high": {
                    "total": data["high"]["total"],
                    "count": data["high"]["count"],
                },
                "boundaries": data["boundaries"],
                "train_cases_total": data["train_cases_total"],
            }
        json.dump(clean_results, f, indent=2)

    print(f"Results saved to {output_file}")

    # Generate LaTeX table content
    print("\n" + "=" * 60)
    print("LATEX TABLE CONTENT FOR ECONOMIC IMPACT BY TERTILES")
    print("=" * 60)

    latex_content = """
\\subsection{Economic Impact Analysis by Tertiles Across Folds}

The following table shows the actual total economic value within each tertile zone for every fold, demonstrating how the dynamic tertile boundaries capture different economic impacts as the training data grows.

\\begin{table}[H]
\\centering
\\caption{Total Economic Impact by Tertile Zone Across Folds}
\\begin{tabular}{lrrrrrr}
\\toprule
\\textbf{Fold} & \\textbf{Low Boundary} & \\textbf{High Boundary} & \\textbf{Low Tertile} & \\textbf{Medium Tertile} & \\textbf{High Tertile} & \\textbf{Total Impact} \\\\
& \\textbf{(USD)} & \\textbf{(USD)} & \\textbf{(USD)} & \\textbf{(USD)} & \\textbf{(USD)} & \\textbf{(USD)} \\\\
\\midrule
"""

    for fold_name in sorted(results.keys()):
        data = results[fold_name]
        fold_num = fold_name.split("_")[1]
        boundaries = data["boundaries"]
        low_total = data["low"]["total"]
        medium_total = data["medium"]["total"]
        high_total = data["high"]["total"]
        grand_total = low_total + medium_total + high_total

        latex_content += f"Fold {fold_num} & \\${boundaries[0]:,.0f} & \\${boundaries[1]:,.0f} & \\${low_total:,.0f} & \\${medium_total:,.0f} & \\${high_total:,.0f} & \\${grand_total:,.0f} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

\\subsection{Case Distribution by Tertiles Across Folds}

\\begin{table}[H]
\\centering
\\caption{Number of Cases by Tertile Zone Across Folds}
\\begin{tabular}{lrrrr}
\\toprule
\\textbf{Fold} & \\textbf{Low Tertile} & \\textbf{Medium Tertile} & \\textbf{High Tertile} & \\textbf{Total Cases} \\\\
& \\textbf{Cases} & \\textbf{Cases} & \\textbf{Cases} & \\textbf{Cases} \\\\
\\midrule
"""

    for fold_name in sorted(results.keys()):
        data = results[fold_name]
        fold_num = fold_name.split("_")[1]
        low_count = data["low"]["count"]
        medium_count = data["medium"]["count"]
        high_count = data["high"]["count"]
        total_count = data["train_cases_total"]

        latex_content += f"Fold {fold_num} & {low_count} & {medium_count} & {high_count} & {total_count} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

\\subsection{Economic Impact Analysis Summary}

\\begin{itemize}
\\item \\textbf{Dynamic Boundaries}: Tertile boundaries adjust across folds as more training data becomes available, ensuring optimal case-wise balance within each fold's training set.
\\item \\textbf{Economic Concentration}: High tertile consistently captures the majority of economic value despite representing approximately one-third of cases, reflecting the natural skew in corporate litigation outcomes.
\\item \\textbf{Temporal Robustness}: The rolling-origin methodology ensures each fold's boundaries are calculated only from temporally preceding cases, preventing future leakage.
\\item \\textbf{Progressive Learning}: Later folds incorporate larger training sets with more comprehensive coverage of the economic outcome space.
\\end{itemize}
"""

    print(latex_content)

    # Save LaTeX content
    with open("tertile_economic_impact_latex.txt", "w") as f:
        f.write(latex_content)

    print(f"\nLaTeX content saved to tertile_economic_impact_latex.txt")


if __name__ == "__main__":
    main()
