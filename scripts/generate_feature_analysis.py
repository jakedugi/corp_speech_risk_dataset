#!/usr/bin/env python3
"""Generate comprehensive feature analysis for academic paper.

This script creates:
1. LaTeX summary statistics tables for all interpretable features
2. PDF figures showing feature distributions and correlations
3. Feature explanations and examples for academic writing
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")
from scipy import stats
from scipy.stats import kruskal, spearmanr

try:
    import scikit_posthocs as sp

    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print(
        "Warning: scikit-posthocs not installed. Some post-hoc tests will be skipped."
    )
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Warning: VIF computation unavailable")
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corp_speech_risk_dataset.fully_interpretable.column_governance import (
    _is_blocked,
    NUMERIC_WHITELIST,
    META_KEYS,
)

# Removed tertile computation imports - inherit precomputed labels from authoritative data

# Set up matplotlib for PDF output
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,  # Ensure fonts are embedded
        "ps.fonttype": 42,
    }
)


def get_feature_explanations() -> Dict[str, Dict[str, str]]:
    """Return comprehensive explanations for each feature category."""
    return {
        "lex_deception": {
            "description": "Lexical markers of potentially deceptive language",
            "explanation": "Counts and ratios of words/phrases that may indicate misleading statements, such as 'allegedly', 'supposedly', or hedging language that distances the speaker from claims.",
            "example": "High values when quotes contain phrases like 'it is believed that' or 'supposedly guarantees'",
        },
        "lex_disclaimers": {
            "description": "Legal disclaimer and cautionary language",
            "explanation": "Frequency of risk warnings, safe harbor statements, and legal disclaimers that companies use to limit liability.",
            "example": "Contains phrases like 'forward-looking statements', 'no guarantee', 'subject to risks'",
        },
        "lex_guarantee": {
            "description": "Strong commitment and guarantee language",
            "explanation": "Words and phrases expressing certainty, promises, or guarantees about future performance or outcomes.",
            "example": "Uses words like 'guarantee', 'ensure', 'promise', 'will definitely'",
        },
        "lex_hedges": {
            "description": "Linguistic hedging and uncertainty markers",
            "explanation": "Language that softens claims or expresses uncertainty, often used to avoid making definitive statements.",
            "example": "Phrases like 'may', 'might', 'possibly', 'to some extent', 'generally'",
        },
        "lex_pricing_claims": {
            "description": "Financial and pricing-related assertions",
            "explanation": "Language related to cost savings, pricing advantages, financial benefits, or economic claims.",
            "example": "Terms like 'cost-effective', 'best price', 'savings', 'value proposition'",
        },
        "lex_scienter": {
            "description": "Knowledge and awareness indicators",
            "explanation": "Language suggesting knowledge, awareness, or intent, which is legally significant for establishing scienter in fraud cases.",
            "example": "Phrases indicating knowledge: 'we knew', 'aware that', 'understood', 'realized'",
        },
        "lex_superlatives": {
            "description": "Superlative and hyperbolic language",
            "explanation": "Extreme positive language and superlatives that may overstate capabilities or performance.",
            "example": "Words like 'best', 'greatest', 'revolutionary', 'unprecedented', 'breakthrough'",
        },
        "ling_certainty": {
            "description": "Linguistic certainty markers",
            "explanation": "High certainty: definitive language (definitely, certainly). Low certainty: tentative language (maybe, perhaps).",
            "example": "High: 'will definitely increase'. Low: 'might possibly improve'",
        },
        "ling_negation": {
            "description": "Negation and denial patterns",
            "explanation": "Frequency of negative constructions and denial statements, which can indicate defensive communication.",
            "example": "Uses 'not', 'never', 'no', 'deny', 'refuse', 'reject'",
        },
        "seq_discourse": {
            "description": "Discourse structure markers",
            "explanation": "Additive: building arguments (also, furthermore). Causal: cause-effect (because, therefore). Conditional: if-then logic. Contrast: opposing ideas (however, but). Temporal: time sequencing (then, after).",
            "example": "Causal: 'therefore profits increased'. Contrast: 'however, risks remain'",
        },
        "seq_risk": {
            "description": "Risk positioning in discourse",
            "explanation": "Where risk-related content appears in the quote (beginning, middle, end) and how varied this positioning is.",
            "example": "Mean position 0.1 = risk mentioned early; 0.9 = risk mentioned late",
        },
        "seq_trans": {
            "description": "Sequential transitions between language types",
            "explanation": "How speakers transition between different types of language (e.g., from hedging to guarantees), indicating rhetorical strategy.",
            "example": "High 'hedges_to_guarantee' = speaker starts uncertain then becomes confident",
        },
    }


def load_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Load a sample of the training data for analysis."""
    print(f"📊 Loading {n_samples:,} samples for feature analysis...")

    fold0_train = Path(
        "data/final_stratified_kfold_splits_authoritative/fold_0/train.jsonl"
    )

    # Load samples
    samples = []
    with open(fold0_train, "r") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))

    df = pd.DataFrame(samples)
    print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} columns")

    return df


def get_approved_features(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """Get approved features and categorize them."""
    all_columns = df.columns.tolist()

    # Get approved features
    blocked = [c for c in all_columns if (c not in META_KEYS) and _is_blocked(c)]
    interpretable = [
        c for c in all_columns if c.startswith("interpretable_") and not _is_blocked(c)
    ]
    numerics = [c for c in all_columns if c in NUMERIC_WHITELIST]
    approved = numerics + interpretable

    # Categorize interpretable features
    categories = {}
    for feature in interpretable:
        if feature.startswith("interpretable_"):
            category = feature.split("_")[1]  # First word after interpretable_
            if category not in categories:
                categories[category] = []
            categories[category].append(feature)

    print(f"✓ Found {len(approved)} approved features in {len(categories)} categories")
    print(f"  Categories: {list(categories.keys())}")

    return approved, categories


def compute_feature_statistics(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute comprehensive statistics for features."""
    print("📈 Computing feature statistics...")

    stats_list = []

    for feature in features:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) == 0:
            continue

        # Convert to numeric if needed
        if data.dtype == "object":
            data = pd.to_numeric(data, errors="coerce").dropna()

        if len(data) == 0:
            continue

        stats = {
            "feature": feature,
            "category": (
                feature.split("_")[1]
                if feature.startswith("interpretable_")
                else "other"
            ),
            "count": len(data),
            "missing": df[feature].isna().sum(),
            "mean": data.mean(),
            "std": data.std(),
            "min": data.min(),
            "q25": data.quantile(0.25),
            "median": data.median(),
            "q75": data.quantile(0.75),
            "max": data.max(),
            "skewness": data.skew(),
            "kurtosis": data.kurtosis(),
            "zeros": (data == 0).sum(),
            "zero_pct": (data == 0).mean() * 100,
            "nonzero_mean": data[data > 0].mean() if (data > 0).any() else 0,
        }

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    print(f"✓ Computed statistics for {len(stats_df)} features")

    return stats_df


def create_latex_tables(
    stats_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    explanations: Dict[str, Dict[str, str]],
    output_dir: Path,
):
    """Generate LaTeX tables for feature statistics."""
    print("📝 Generating LaTeX tables...")

    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Main summary table
    main_table = []
    main_table.append("\\begin{table}[htbp]")
    main_table.append("\\centering")
    main_table.append(
        "\\caption{Summary Statistics of Interpretable Features by Category}"
    )
    main_table.append("\\label{tab:feature_summary}")
    main_table.append("\\begin{tabular}{lrrrrrr}")
    main_table.append("\\toprule")
    main_table.append("Category & Count & Mean & Std & Median & Zero\\% & Skew \\\\")
    main_table.append("\\midrule")

    # Aggregate by category
    for category in sorted(categories.keys()):
        cat_features = [
            f for f in categories[category] if f in stats_df["feature"].values
        ]
        if not cat_features:
            continue

        cat_stats = stats_df[stats_df["feature"].isin(cat_features)]

        row = f"{category.replace('_', ' ').title()} & "
        row += f"{len(cat_features)} & "
        row += f"{cat_stats['mean'].mean():.3f} & "
        row += f"{cat_stats['std'].mean():.3f} & "
        row += f"{cat_stats['median'].mean():.3f} & "
        row += f"{cat_stats['zero_pct'].mean():.1f} & "
        row += f"{cat_stats['skewness'].mean():.2f} \\\\"

        main_table.append(row)

    main_table.append("\\bottomrule")
    main_table.append("\\end{tabular}")
    main_table.append("\\end{table}")

    with open(latex_dir / "feature_summary.tex", "w") as f:
        f.write("\n".join(main_table))

    # Detailed table for each category
    for category, features in categories.items():
        # Map category to explanation key
        exp_key = category
        if category == "lex":
            exp_keys = [k for k in explanations.keys() if k.startswith("lex_")]
        elif category == "ling":
            exp_keys = [k for k in explanations.keys() if k.startswith("ling_")]
        elif category == "seq":
            exp_keys = [k for k in explanations.keys() if k.startswith("seq_")]
        else:
            exp_keys = []

        cat_stats = stats_df[stats_df["category"] == category].copy()
        if len(cat_stats) == 0:
            continue

        # Create detailed table
        detailed_table = []
        detailed_table.append("\\begin{table}[htbp]")
        detailed_table.append("\\centering")
        detailed_table.append(
            f"\\caption{{Detailed Statistics: {category.title()} Features}}"
        )
        detailed_table.append(f"\\label{{tab:{category}_details}}")
        detailed_table.append("\\begin{tabular}{lrrrrr}")
        detailed_table.append("\\toprule")
        detailed_table.append("Feature & Mean & Std & Median & Zero\\% & Skew \\\\")
        detailed_table.append("\\midrule")

        # Sort by mean value
        cat_stats = cat_stats.sort_values("mean", ascending=False)

        for _, row in cat_stats.iterrows():
            feature_name = (
                row["feature"].replace("interpretable_", "").replace("_", "\\_")
            )
            table_row = f"{feature_name} & "
            table_row += f"{row['mean']:.3f} & "
            table_row += f"{row['std']:.3f} & "
            table_row += f"{row['median']:.3f} & "
            table_row += f"{row['zero_pct']:.1f} & "
            table_row += f"{row['skewness']:.2f} \\\\"
            detailed_table.append(table_row)

        detailed_table.append("\\bottomrule")
        detailed_table.append("\\end{tabular}")
        detailed_table.append("\\end{table}")

        with open(latex_dir / f"{category}_details.tex", "w") as f:
            f.write("\n".join(detailed_table))

    # Feature explanations document
    explanations_doc = []
    explanations_doc.append("\\section{Interpretable Feature Definitions}")
    explanations_doc.append("")

    for category, info in explanations.items():
        if category in categories:
            explanations_doc.append(f"\\subsection{{{info['description']}}}")
            explanations_doc.append(f"\\label{{sec:{category}}}")
            explanations_doc.append("")
            explanations_doc.append(info["explanation"])
            explanations_doc.append("")
            explanations_doc.append(f"\\textbf{{Example:}} {info['example']}")
            explanations_doc.append("")

    with open(latex_dir / "feature_explanations.tex", "w") as f:
        f.write("\n".join(explanations_doc))

    print(f"✓ Generated LaTeX tables in {latex_dir}")


def create_pdf_figures(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    output_dir: Path,
):
    """Generate PDF figures for the academic paper."""
    print("📊 Generating PDF figures...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature distribution overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Interpretable Features: Distribution Overview", fontsize=14, fontweight="bold"
    )

    # Distribution of means by category
    cat_means = []
    cat_names = []
    for category in sorted(categories.keys()):
        cat_stats = stats_df[stats_df["category"] == category]
        if len(cat_stats) > 0:
            cat_means.extend(cat_stats["mean"].tolist())
            cat_names.extend([category.replace("_", " ").title()] * len(cat_stats))

    if cat_means:
        axes[0, 0].boxplot(
            [
                stats_df[stats_df["category"] == cat]["mean"].tolist()
                for cat in sorted(categories.keys())
            ],
            labels=[cat.replace("_", " ").title() for cat in sorted(categories.keys())],
        )
        axes[0, 0].set_title("Mean Values by Category")
        axes[0, 0].set_ylabel("Mean Value")
        axes[0, 0].tick_params(axis="x", rotation=45)

    # Zero percentage distribution
    axes[0, 1].hist(
        stats_df["zero_pct"], bins=20, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0, 1].set_title("Distribution of Zero Percentages")
    axes[0, 1].set_xlabel("Percentage of Zero Values")
    axes[0, 1].set_ylabel("Number of Features")

    # Skewness distribution
    axes[1, 0].hist(
        stats_df["skewness"].dropna(),
        bins=20,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    axes[1, 0].set_title("Distribution of Skewness")
    axes[1, 0].set_xlabel("Skewness")
    axes[1, 0].set_ylabel("Number of Features")
    axes[1, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)

    # Feature count by category
    cat_counts = [len(categories[cat]) for cat in sorted(categories.keys())]
    axes[1, 1].bar(range(len(cat_counts)), cat_counts, color="lightgreen", alpha=0.7)
    axes[1, 1].set_title("Number of Features by Category")
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Number of Features")
    axes[1, 1].set_xticks(range(len(categories)))
    axes[1, 1].set_xticklabels(
        [cat.replace("_", " ").title() for cat in sorted(categories.keys())],
        rotation=45,
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "feature_distributions.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Category-specific distributions (top 3 categories by feature count)
    top_categories = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[
        :3
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Feature Value Distributions by Category (Top 3 Categories)",
        fontsize=14,
        fontweight="bold",
    )

    for i, (category, features) in enumerate(top_categories):
        # Get a representative feature from this category
        cat_features = [f for f in features if f in df.columns]
        if not cat_features:
            continue

        # Use the feature with highest variance for visualization
        feature_vars = {
            f: df[f].var() for f in cat_features[:5]
        }  # Limit to avoid memory issues
        if feature_vars:
            top_feature = max(feature_vars, key=feature_vars.get)
            data = df[top_feature].dropna()

            if len(data) > 0:
                # Handle potential object types
                if data.dtype == "object":
                    data = pd.to_numeric(data, errors="coerce").dropna()

                if len(data) > 0:
                    axes[i].hist(
                        data,
                        bins=30,
                        alpha=0.7,
                        color=plt.cm.Set3(i),
                        edgecolor="black",
                    )
                    axes[i].set_title(
                        f'{category.replace("_", " ").title()}\n({top_feature.split("_")[-1]})'
                    )
                    axes[i].set_xlabel("Value")
                    axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "category_distributions.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Correlation heatmap (sample of features)
    # Select representative features from each category
    sample_features = []
    for category, features in categories.items():
        cat_features = [f for f in features if f in df.columns]
        if cat_features:
            # Take first 2 features from each category to keep heatmap readable
            sample_features.extend(cat_features[:2])

    if len(sample_features) > 5:  # Only create if we have enough features
        sample_features = sample_features[:20]  # Limit to 20 for readability

        # Compute correlation matrix
        corr_data = df[sample_features].select_dtypes(include=[np.number])
        if len(corr_data.columns) > 1:
            corr_matrix = corr_data.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=False,
                cmap="RdBu_r",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(
                "Feature Correlation Matrix (Sample)", fontsize=14, fontweight="bold"
            )

            # Rotate labels for better readability
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

            plt.tight_layout()
            plt.savefig(
                figures_dir / "feature_correlations.pdf", dpi=300, bbox_inches="tight"
            )
            plt.close()

    print(f"✓ Generated PDF figures in {figures_dir}")


def create_summary_document(
    stats_df: pd.DataFrame,
    categories: Dict[str, List[str]],
    explanations: Dict[str, Dict[str, str]],
    output_dir: Path,
):
    """Create a comprehensive summary document."""
    print("📄 Creating summary document...")

    summary_lines = []
    summary_lines.append("# Interpretable Features Analysis Summary")
    summary_lines.append("")
    summary_lines.append(
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    summary_lines.append(f"**Total Features Analyzed:** {len(stats_df)}")
    summary_lines.append(f"**Feature Categories:** {len(categories)}")
    summary_lines.append("")

    summary_lines.append("## Overview")
    summary_lines.append("")
    summary_lines.append(
        "This analysis covers the interpretable features used for training the POLAR model."
    )
    summary_lines.append(
        "All features have been filtered through column governance to ensure they are:"
    )
    summary_lines.append("- Interpretable and auditable")
    summary_lines.append("- Free from data leakage")
    summary_lines.append("- Legally and ethically appropriate")
    summary_lines.append("")

    summary_lines.append("## Feature Categories")
    summary_lines.append("")
    for category, info in explanations.items():
        if category in categories:
            count = len(categories[category])
            summary_lines.append(f"### {info['description']} ({count} features)")
            summary_lines.append("")
            summary_lines.append(info["explanation"])
            summary_lines.append("")
            summary_lines.append(f"**Example:** {info['example']}")
            summary_lines.append("")

    summary_lines.append("## Statistical Summary")
    summary_lines.append("")
    summary_lines.append(
        "| Category | Features | Avg Mean | Avg Std | Avg Zero% | Avg Skew |"
    )
    summary_lines.append(
        "|----------|----------|----------|---------|-----------|----------|"
    )

    for category in sorted(categories.keys()):
        cat_stats = stats_df[stats_df["category"] == category]
        if len(cat_stats) > 0:
            row = f"| {category.replace('_', ' ').title()} | "
            row += f"{len(cat_stats)} | "
            row += f"{cat_stats['mean'].mean():.3f} | "
            row += f"{cat_stats['std'].mean():.3f} | "
            row += f"{cat_stats['zero_pct'].mean():.1f}% | "
            row += f"{cat_stats['skewness'].mean():.2f} |"
            summary_lines.append(row)

    summary_lines.append("")
    summary_lines.append("## Key Findings")
    summary_lines.append("")

    # Add some key insights
    high_zero = stats_df[stats_df["zero_pct"] > 80]
    if len(high_zero) > 0:
        summary_lines.append(
            f"- **Sparse Features:** {len(high_zero)} features have >80% zero values"
        )

    high_skew = stats_df[abs(stats_df["skewness"]) > 2]
    if len(high_skew) > 0:
        summary_lines.append(
            f"- **Highly Skewed:** {len(high_skew)} features show high skewness (|skew| > 2)"
        )

    summary_lines.append(
        f"- **Most Informative Category:** {stats_df.groupby('category')['nonzero_mean'].mean().idxmax().replace('_', ' ').title()}"
    )
    summary_lines.append("")

    summary_lines.append("## Files Generated")
    summary_lines.append("")
    summary_lines.append("- `latex/feature_summary.tex` - Main summary table")
    summary_lines.append("- `latex/feature_explanations.tex` - Feature definitions")
    summary_lines.append("- `latex/*_details.tex` - Detailed tables by category")
    summary_lines.append(
        "- `figures/feature_distributions.pdf` - Distribution overview"
    )
    summary_lines.append(
        "- `figures/category_distributions.pdf` - Category-specific distributions"
    )
    summary_lines.append("- `figures/feature_correlations.pdf` - Correlation heatmap")
    summary_lines.append("")

    with open(output_dir / "FEATURE_ANALYSIS_SUMMARY.md", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"✓ Created summary document")


def load_fold3_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load fold 3 data including train, dev, and OOF test sets."""
    print("📊 Loading fold 3 data for statistical analysis...")

    # Load fold metadata from authoritative data
    with open(
        "data/final_stratified_kfold_splits_authoritative/per_fold_metadata.json", "r"
    ) as f:
        fold_metadata = json.load(f)

    # Load fold 3 train and dev from authoritative data
    train_path = Path(
        "data/final_stratified_kfold_splits_authoritative/fold_3/train.jsonl"
    )
    dev_path = Path("data/final_stratified_kfold_splits_authoritative/fold_3/dev.jsonl")
    test_path = Path(
        "data/final_stratified_kfold_splits_authoritative/oof_test/test.jsonl"
    )

    train_data = []
    with open(train_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    dev_data = []
    with open(dev_path, "r") as f:
        for line in f:
            dev_data.append(json.loads(line))
    dev_df = pd.DataFrame(dev_data)

    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_df = pd.DataFrame(test_data)

    print(
        f"✓ Loaded fold 3: train={len(train_df):,}, dev={len(dev_df):,}, test={len(test_df):,}"
    )

    # Use precomputed outcome_bin labels from authoritative data (no recomputation)
    for df in [train_df, dev_df, test_df]:
        if "outcome_bin" in df.columns:
            # Use authoritative precomputed labels
            df["bin"] = df["outcome_bin"].astype(int)
            df["y"] = df["bin"]  # For compatibility
        else:
            # Fallback for missing labels (should not happen with authoritative data)
            df["bin"] = 1  # Default to medium risk
            df["y"] = 1

    # Use precomputed sample weights from authoritative data
    if "sample_weight" in train_df.columns:
        weights = train_df["sample_weight"].values
        weight_stats = {"method": "inherited_from_authoritative_data"}
    else:
        # Fallback to uniform weights
        weights = np.ones(len(train_df))
        weight_stats = {"method": "uniform_fallback"}
    train_df["sample_weight"] = weights

    return train_df, dev_df, test_df, fold_metadata


def compute_dataset_health(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """Compute dataset health metrics."""
    print("🏥 Computing dataset health metrics...")

    health_metrics = {"metric": [], "overall": [], "train": [], "dev": [], "test": []}

    # Basic counts
    health_metrics["metric"].append("Cases")
    health_metrics["overall"].append(
        len(set(train_df["case_id"]) | set(dev_df["case_id"]) | set(test_df["case_id"]))
    )
    health_metrics["train"].append(train_df["case_id"].nunique())
    health_metrics["dev"].append(dev_df["case_id"].nunique())
    health_metrics["test"].append(test_df["case_id"].nunique())

    health_metrics["metric"].append("Quotes")
    health_metrics["overall"].append(len(train_df) + len(dev_df) + len(test_df))
    health_metrics["train"].append(len(train_df))
    health_metrics["dev"].append(len(dev_df))
    health_metrics["test"].append(len(test_df))

    # Quotes per case stats
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        qpc = df.groupby("case_id").size()
        health_metrics["metric"].extend(
            [f"Quotes/Case (mean)", f"Quotes/Case (median)", f"Quotes/Case (p90)"]
        )
        health_metrics["overall"].extend([np.nan, np.nan, np.nan])
        for split in ["train", "dev", "test"]:
            if split == name:
                health_metrics[split].extend(
                    [qpc.mean(), qpc.median(), qpc.quantile(0.9)]
                )
            else:
                health_metrics[split].extend([np.nan, np.nan, np.nan])

    # Missing data
    health_metrics["metric"].append("Missing (%)")
    all_missing = (
        pd.concat([train_df[features], dev_df[features], test_df[features]])
        .isna()
        .mean()
        .mean()
        * 100
    )
    health_metrics["overall"].append(all_missing)
    health_metrics["train"].append(train_df[features].isna().mean().mean() * 100)
    health_metrics["dev"].append(dev_df[features].isna().mean().mean() * 100)
    health_metrics["test"].append(test_df[features].isna().mean().mean() * 100)

    return pd.DataFrame(health_metrics)


def compute_feature_hygiene(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Compute detailed feature hygiene metrics."""
    print("🧹 Computing feature hygiene metrics...")

    hygiene_data = []

    for feature in features:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) == 0:
            continue

        # Convert to numeric if needed
        if data.dtype == "object":
            data = pd.to_numeric(data, errors="coerce").dropna()

        if len(data) == 0:
            continue

        # Compute metrics
        q = data.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        mad = (data - data.median()).abs().median()

        hygiene = {
            "feature": feature,
            "mean": data.mean(),
            "median": data.median(),
            "sd": data.std(),
            "mad": mad,
            "p01": q.loc[0.01],
            "p05": q.loc[0.05],
            "p50": q.loc[0.5],
            "p95": q.loc[0.95],
            "p99": q.loc[0.99],
            "zeros_pct": (data == 0).mean() * 100,
            "missing_pct": df[feature].isna().mean() * 100,
            "skew": data.skew(),
            "kurt": data.kurtosis(),
            "transform": "none",  # Will be updated based on skew
        }

        # Suggest transformation
        if abs(hygiene["skew"]) > 2:
            if hygiene["zeros_pct"] < 50:
                hygiene["transform"] = "log1p"
            else:
                hygiene["transform"] = "binarize"
        elif hygiene["p99"] > hygiene["p95"] * 2:
            hygiene["transform"] = "winsorize"

        hygiene_data.append(hygiene)

    return pd.DataFrame(hygiene_data)


def compute_univariate_predictiveness(
    train_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Compute univariate predictiveness metrics."""
    print("📈 Computing univariate predictiveness...")

    predictiveness_data = []

    for feature in features:
        if feature not in train_df.columns or "bin" not in train_df.columns:
            continue

        data = train_df[[feature, "bin"]].dropna()
        if len(data) == 0:
            continue

        X = data[feature].values
        y = data["bin"].values

        # Skip if feature is constant
        if X.std() == 0:
            continue

        pred = {
            "feature": feature,
            "auc_macro": np.nan,
            "mi": np.nan,
            "smd_01": np.nan,
            "smd_12": np.nan,
            "monotonic": False,
            "kw_pvalue": np.nan,
        }

        try:
            # Macro AUC (one-vs-rest)
            if len(np.unique(y)) == 3:
                auc_scores = []
                for class_val in [0, 1, 2]:
                    y_binary = (y == class_val).astype(int)
                    if len(np.unique(y_binary)) == 2:
                        auc = roc_auc_score(y_binary, X)
                        auc_scores.append(
                            max(auc, 1 - auc)
                        )  # Handle inverse relationships
                pred["auc_macro"] = np.mean(auc_scores)

            # Mutual information
            X_binned = pd.qcut(X, q=10, duplicates="drop", labels=False)
            pred["mi"] = mutual_info_score(y, X_binned)

            # Standardized mean differences
            groups = [X[y == k] for k in [0, 1, 2]]
            if len(groups[0]) > 0 and len(groups[1]) > 0:
                pooled_std = np.sqrt((groups[0].var() + groups[1].var()) / 2)
                if pooled_std > 0:
                    pred["smd_01"] = (groups[1].mean() - groups[0].mean()) / pooled_std

            if len(groups[1]) > 0 and len(groups[2]) > 0:
                pooled_std = np.sqrt((groups[1].var() + groups[2].var()) / 2)
                if pooled_std > 0:
                    pred["smd_12"] = (groups[2].mean() - groups[1].mean()) / pooled_std

            # Monotonicity check
            means = [g.mean() for g in groups if len(g) > 0]
            if len(means) == 3:
                pred["monotonic"] = (means[0] <= means[1] <= means[2]) or (
                    means[0] >= means[1] >= means[2]
                )

            # Kruskal-Wallis test
            groups_nonempty = [g for g in groups if len(g) > 0]
            if len(groups_nonempty) >= 2:
                _, pred["kw_pvalue"] = kruskal(*groups_nonempty)

        except Exception as e:
            print(f"  Warning: Failed to compute metrics for {feature}: {e}")

        predictiveness_data.append(pred)

    return pd.DataFrame(predictiveness_data)


def compute_temporal_stability(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Compute temporal stability metrics."""
    print("⏱️ Computing temporal stability...")

    stability_data = []

    for feature in features:
        if feature not in train_df.columns:
            continue

        stab = {"feature": feature, "spearman_year": np.nan, "psi_train_dev": np.nan}

        try:
            # Correlation with year (if year column exists)
            if "year" in train_df.columns:
                data = train_df[[feature, "year"]].dropna()
                if len(data) > 10:
                    stab["spearman_year"], _ = spearmanr(data[feature], data["year"])

            # PSI between train and dev
            train_vals = train_df[feature].dropna()
            dev_vals = dev_df[feature].dropna()

            if len(train_vals) > 0 and len(dev_vals) > 0:
                # Compute PSI
                q = np.quantile(train_vals, np.linspace(0, 1, 11))
                q[0], q[-1] = -np.inf, np.inf

                train_hist = np.histogram(train_vals, q)[0] / len(train_vals)
                dev_hist = np.histogram(dev_vals, q)[0] / len(dev_vals)

                # Add small epsilon to avoid log(0)
                eps = 1e-6
                train_hist = np.clip(train_hist, eps, 1)
                dev_hist = np.clip(dev_hist, eps, 1)

                stab["psi_train_dev"] = np.sum(
                    (train_hist - dev_hist) * np.log(train_hist / dev_hist)
                )

        except Exception as e:
            print(f"  Warning: Failed to compute stability for {feature}: {e}")

        stability_data.append(stab)

    return pd.DataFrame(stability_data)


def compute_ordered_logit_baseline(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, features: List[str], output_dir: Path
) -> Dict:
    """Fit ordered logit baseline model."""
    print("📊 Fitting ordered logit baseline...")

    # Prepare case-level data
    train_case = train_df.groupby("case_id").first().reset_index()
    dev_case = dev_df.groupby("case_id").first().reset_index()

    # Filter to features that exist
    feat_cols = [f for f in features if f in train_case.columns]

    # Remove features with no variance
    valid_features = []
    for f in feat_cols:
        if train_case[f].std() > 0:
            valid_features.append(f)

    if len(valid_features) == 0:
        print("  Warning: No valid features for ordered logit")
        return {}

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_case[valid_features].fillna(0))
    X_train = sm.add_constant(X_train)
    y_train = train_case["bin"]

    try:
        # Fit ordered logit
        model = OrderedModel(y_train, X_train, distr="logit")
        result = model.fit(method="bfgs", disp=False, maxiter=100)

        # Get odds ratios and CIs
        or_table = pd.DataFrame(
            {
                "feature": ["const"] + valid_features,
                "coef": result.params,
                "or": np.exp(result.params),
                "or_lower": np.exp(result.conf_int()[0]),
                "or_upper": np.exp(result.conf_int()[1]),
                "pvalue": result.pvalues,
            }
        )

        # Evaluate on dev
        X_dev = scaler.transform(dev_case[valid_features].fillna(0))
        X_dev = sm.add_constant(X_dev)
        y_dev = dev_case["bin"]

        # Predict probabilities
        pred_probs = result.predict(X_dev)
        pred_class = pred_probs.argmax(axis=1)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score

        metrics = {
            "accuracy": accuracy_score(y_dev, pred_class),
            "macro_f1": f1_score(y_dev, pred_class, average="macro"),
            "pseudo_r2": result.prsquared,
        }

        # Save results
        or_table.to_csv(output_dir / "ordered_logit_results.csv", index=False)

        print(
            f"✓ Ordered logit complete: accuracy={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}"
        )

        return {"or_table": or_table, "metrics": metrics}

    except Exception as e:
        print(f"  Warning: Ordered logit failed: {e}")
        return {}


def compute_per_bucket_profiles(
    train_df: pd.DataFrame, features: List[str], top_n: int = 10
) -> pd.DataFrame:
    """Compute per-bucket feature profiles."""
    print("🪣 Computing per-bucket profiles...")

    # Get top features by SMD
    predictiveness = compute_univariate_predictiveness(train_df, features)
    predictiveness["smd_avg"] = predictiveness[["smd_01", "smd_12"]].abs().mean(axis=1)
    top_features = predictiveness.nlargest(top_n, "smd_avg")["feature"].tolist()

    profile_data = []

    for feature in top_features:
        if feature not in train_df.columns:
            continue

        row = {"feature": feature}

        # Compute stats per bucket
        for bucket in [0, 1, 2]:
            data = train_df[train_df["bin"] == bucket][feature].dropna()
            if len(data) > 0:
                row[
                    (
                        f"low_med"
                        if bucket == 0
                        else f"med_med" if bucket == 1 else "high_med"
                    )
                ] = data.median()
                row[
                    (
                        f"low_iqr"
                        if bucket == 0
                        else f"med_iqr" if bucket == 1 else "high_iqr"
                    )
                ] = data.quantile(0.75) - data.quantile(0.25)

        # KW test
        groups = [train_df[train_df["bin"] == k][feature].dropna() for k in [0, 1, 2]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            _, row["kw_p"] = kruskal(*groups)

        profile_data.append(row)

    return pd.DataFrame(profile_data)


def create_statistical_latex_tables(
    health_df: pd.DataFrame,
    hygiene_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    ordered_logit: Dict,
    output_dir: Path,
):
    """Generate LaTeX tables for statistical analysis."""
    print("📝 Generating statistical LaTeX tables...")

    latex_dir = output_dir / "latex"

    # Dataset health table
    health_table = []
    health_table.append("\\begin{table}[htbp]")
    health_table.append("\\centering")
    health_table.append("\\caption{Dataset Health Metrics}")
    health_table.append("\\label{tab:dataset_health}")
    health_table.append("\\begin{tabular}{lrrrr}")
    health_table.append("\\toprule")
    health_table.append("Metric & Overall & Train & Dev & Test \\\\")
    health_table.append("\\midrule")

    for _, row in health_df.iterrows():
        line = f"{row['metric']} & "
        for col in ["overall", "train", "dev", "test"]:
            val = row[col]
            if pd.isna(val):
                line += "-- & "
            elif isinstance(val, float):
                line += f"{val:.1f} & "
            else:
                line += f"{val:,} & "
        line = line.rstrip(" & ") + " \\\\"
        health_table.append(line)

    health_table.append("\\bottomrule")
    health_table.append("\\end{tabular}")
    health_table.append("\\end{table}")

    with open(latex_dir / "dataset_health.tex", "w") as f:
        f.write("\n".join(health_table))

    # Top predictive features table
    top_predict = predict_df.nlargest(15, "auc_macro")[
        ["feature", "auc_macro", "smd_01", "smd_12", "monotonic", "kw_pvalue"]
    ]

    predict_table = []
    predict_table.append("\\begin{table}[htbp]")
    predict_table.append("\\centering")
    predict_table.append("\\caption{Top 15 Features by Univariate Predictiveness}")
    predict_table.append("\\label{tab:top_predictive}")
    predict_table.append("\\begin{tabular}{lrrrrr}")
    predict_table.append("\\toprule")
    predict_table.append("Feature & AUC & SMD(0→1) & SMD(1→2) & Mono. & KW p \\\\")
    predict_table.append("\\midrule")

    for _, row in top_predict.iterrows():
        feature_name = row["feature"].replace("interpretable_", "").replace("_", "\\_")
        line = f"{feature_name} & "
        line += f"{row['auc_macro']:.3f} & "
        line += f"{row['smd_01']:.2f} & "
        line += f"{row['smd_12']:.2f} & "
        line += f"{'✓' if row['monotonic'] else '✗'} & "
        line += (
            f"{row['kw_pvalue']:.3f}"
            if row["kw_pvalue"] < 0.001
            else f"{row['kw_pvalue']:.4f}"
        )
        line += " \\\\"
        predict_table.append(line)

    predict_table.append("\\bottomrule")
    predict_table.append("\\end{tabular}")
    predict_table.append("\\end{table}")

    with open(latex_dir / "top_predictive_features.tex", "w") as f:
        f.write("\n".join(predict_table))

    # Per-bucket profiles table
    if len(profile_df) > 0:
        bucket_table = []
        bucket_table.append("\\begin{table}[htbp]")
        bucket_table.append("\\centering")
        bucket_table.append("\\caption{Feature Profiles by Outcome Bucket}")
        bucket_table.append("\\label{tab:bucket_profiles}")
        bucket_table.append("\\begin{tabular}{lcccc}")
        bucket_table.append("\\toprule")
        bucket_table.append(
            "Feature & Low (Med[IQR]) & Med (Med[IQR]) & High (Med[IQR]) & KW p \\\\"
        )
        bucket_table.append("\\midrule")

        for _, row in profile_df.iterrows():
            feature_name = (
                row["feature"].replace("interpretable_", "").replace("_", "\\_")
            )
            line = f"{feature_name} & "

            for bucket in ["low", "med", "high"]:
                med_col = f"{bucket}_med"
                iqr_col = f"{bucket}_iqr"
                if med_col in row and not pd.isna(row[med_col]):
                    line += f"{row[med_col]:.2f}[{row[iqr_col]:.2f}] & "
                else:
                    line += "-- & "

            if "kw_p" in row and not pd.isna(row["kw_p"]):
                line += (
                    f"{row['kw_p']:.3f}"
                    if row["kw_p"] < 0.001
                    else f"{row['kw_p']:.4f}"
                )
            else:
                line += "--"
            line += " \\\\"
            bucket_table.append(line)

        bucket_table.append("\\bottomrule")
        bucket_table.append("\\end{tabular}")
        bucket_table.append("\\end{table}")

        with open(latex_dir / "bucket_profiles.tex", "w") as f:
            f.write("\n".join(bucket_table))

    print(f"✓ Generated statistical LaTeX tables")


def create_statistical_figures(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    features: List[str],
    predict_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate statistical analysis figures."""
    print("📊 Generating statistical figures...")

    figures_dir = output_dir / "figures"

    # 1. Per-bucket violin plots for top features
    top_features = predict_df.nlargest(6, "auc_macro")["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Feature Distributions by Outcome Bucket (Top 6 Features)",
        fontsize=14,
        fontweight="bold",
    )
    axes = axes.flatten()

    for i, feature in enumerate(top_features[:6]):
        if feature not in train_df.columns:
            continue

        # Prepare data
        plot_data = []
        for bucket in [0, 1, 2]:
            data = train_df[train_df["bin"] == bucket][feature].dropna()
            plot_data.extend([(bucket, val) for val in data])

        plot_df = pd.DataFrame(plot_data, columns=["Bucket", "Value"])

        # Create violin plot
        sns.violinplot(data=plot_df, x="Bucket", y="Value", ax=axes[i])
        axes[i].set_title(
            feature.replace("interpretable_", "").replace("_", " ").title()
        )
        axes[i].set_xlabel("Outcome Bucket")
        axes[i].set_xticklabels(["Low", "Medium", "High"])

    plt.tight_layout()
    plt.savefig(figures_dir / "bucket_violins.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Feature importance heatmap (based on AUC and SMD)
    top_n = 20
    top_pred = predict_df.nlargest(top_n, "auc_macro")

    # Create importance matrix
    importance_matrix = pd.DataFrame(
        {
            "AUC": top_pred["auc_macro"],
            "SMD 0→1": top_pred["smd_01"].abs(),
            "SMD 1→2": top_pred["smd_12"].abs(),
            "Monotonic": top_pred["monotonic"].astype(int),
        }
    )
    importance_matrix.index = [
        f.replace("interpretable_", "").replace("_", " ") for f in top_pred["feature"]
    ]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(importance_matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title(
        f"Feature Importance Metrics (Top {top_n})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / "feature_importance_heatmap.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Temporal stability plot
    if "spearman_year" in predict_df.columns:
        # Filter features with significant temporal correlation
        temporal_features = predict_df[
            predict_df["spearman_year"].abs() > 0.1
        ].nlargest(10, "spearman_year", key=abs)

        if len(temporal_features) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))

            features_short = [
                f.replace("interpretable_", "").replace("_", " ")
                for f in temporal_features["feature"]
            ]
            y_pos = np.arange(len(features_short))

            ax.barh(
                y_pos,
                temporal_features["spearman_year"].values,
                color=[
                    "red" if x < 0 else "green"
                    for x in temporal_features["spearman_year"].values
                ],
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_short)
            ax.set_xlabel("Spearman Correlation with Year")
            ax.set_title(
                "Features with Temporal Trends", fontsize=14, fontweight="bold"
            )
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

            plt.tight_layout()
            plt.savefig(
                figures_dir / "temporal_trends.pdf", dpi=300, bbox_inches="tight"
            )
            plt.close()

    print(f"✓ Generated statistical figures")


def create_comprehensive_feature_csv(
    stats_df: pd.DataFrame,
    hygiene_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: Path,
):
    """Create comprehensive CSV with all feature metrics."""
    print("📊 Creating comprehensive feature metrics CSV...")

    # Merge all dataframes on feature
    comprehensive = stats_df[
        ["feature", "category", "mean", "std", "median", "zero_pct", "skewness"]
    ]

    # Add hygiene metrics
    if len(hygiene_df) > 0:
        hygiene_cols = [
            "feature",
            "mad",
            "p01",
            "p05",
            "p95",
            "p99",
            "missing_pct",
            "kurt",
            "transform",
        ]
        comprehensive = comprehensive.merge(
            hygiene_df[hygiene_cols], on="feature", how="left"
        )

    # Add predictiveness metrics
    if len(predict_df) > 0:
        predict_cols = [
            "feature",
            "auc_macro",
            "mi",
            "smd_01",
            "smd_12",
            "monotonic",
            "kw_pvalue",
        ]
        comprehensive = comprehensive.merge(
            predict_df[predict_cols], on="feature", how="left"
        )

    # Add stability metrics
    if len(stability_df) > 0:
        stability_cols = ["feature", "spearman_year", "psi_train_dev"]
        comprehensive = comprehensive.merge(
            stability_df[stability_cols], on="feature", how="left"
        )

    # Add flags based on thresholds
    comprehensive["flag_missing"] = comprehensive["missing_pct"] > 20
    comprehensive["flag_sparse"] = comprehensive["zero_pct"] > 95
    comprehensive["flag_drift"] = comprehensive["psi_train_dev"] > 0.25
    comprehensive["flag_temporal"] = comprehensive["spearman_year"].abs() > 0.3

    # Save comprehensive CSV
    comprehensive.to_csv(output_dir / "comprehensive_feature_metrics.csv", index=False)
    print(f"✓ Saved comprehensive metrics for {len(comprehensive)} features")

    return comprehensive


def apply_feature_filtering_rules(
    comprehensive_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Apply feature filtering rules to determine which features to keep/drop."""
    print("🔍 Applying feature filtering rules...")

    # Define thresholds
    MISSING_THRESHOLD = 20.0
    SPARSITY_THRESHOLD = 95.0
    PSI_THRESHOLD = 0.25

    # Initialize results
    keep_features = []
    drop_features = []
    drop_reasons = {}

    for _, row in comprehensive_df.iterrows():
        feature = row["feature"]
        drop_reason = []

        # Rule 1: Missing percentage > 20%
        if pd.notna(row.get("missing_pct")) and row["missing_pct"] > MISSING_THRESHOLD:
            drop_reason.append(f"missing_pct={row['missing_pct']:.1f}%")

        # Rule 2: Zero percentage > 95% (extreme sparsity)
        if pd.notna(row.get("zero_pct")) and row["zero_pct"] > SPARSITY_THRESHOLD:
            drop_reason.append(f"zero_pct={row['zero_pct']:.1f}%")

        # Rule 3: PSI train->dev > 0.25 (population shift)
        if pd.notna(row.get("psi_train_dev")) and row["psi_train_dev"] > PSI_THRESHOLD:
            drop_reason.append(f"psi_train_dev={row['psi_train_dev']:.3f}")

        # Rule 4: Temporal drift flag
        if pd.notna(row.get("flag_temporal")) and row["flag_temporal"]:
            drop_reason.append("temporal_drift")

        if drop_reason:
            drop_features.append(feature)
            drop_reasons[feature] = "; ".join(drop_reason)
        else:
            keep_features.append(feature)

    print(f"✓ Feature filtering complete:")
    print(f"   - Keep: {len(keep_features)} features")
    print(f"   - Drop: {len(drop_features)} features")

    return {"keep": keep_features, "drop": drop_features, "drop_reasons": drop_reasons}


def compute_court_proxy_probe(
    train_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Compute court/venue proxy analysis (if court data available)."""
    print("🏛️ Computing court proxy probe...")

    court_data = []

    # Check if we have court information (may not be available)
    court_cols = [
        c for c in train_df.columns if "court" in c.lower() or "venue" in c.lower()
    ]

    if not court_cols:
        print("   No court/venue columns found - skipping court proxy analysis")
        return pd.DataFrame()

    # Use first available court column
    court_col = court_cols[0]
    print(f"   Using {court_col} for court proxy analysis")

    for feature in features:
        if feature not in train_df.columns:
            continue

        data = train_df[[feature, court_col]].dropna()
        if len(data) == 0:
            continue

        try:
            # Mutual information with court
            feature_vals = data[feature].values
            court_vals = data[court_col].values

            # Bin feature values for MI computation
            feature_binned = pd.qcut(feature_vals, q=5, duplicates="drop", labels=False)
            mi_court = mutual_info_score(court_vals, feature_binned)

            # AUC for reference
            auc = np.nan
            if "bin" in train_df.columns:
                feature_outcome_data = train_df[[feature, "bin"]].dropna()
                if len(feature_outcome_data) > 0:
                    X = feature_outcome_data[feature].values
                    y = feature_outcome_data["bin"].values
                    if len(np.unique(y)) > 1 and X.std() > 0:
                        auc_scores = []
                        for class_val in np.unique(y):
                            y_binary = (y == class_val).astype(int)
                            if len(np.unique(y_binary)) == 2:
                                auc = roc_auc_score(y_binary, X)
                                auc_scores.append(max(auc, 1 - auc))
                        auc = np.mean(auc_scores)

            court_data.append(
                {"feature": feature, "mi_court": mi_court, "auc_outcome": auc}
            )

        except Exception as e:
            print(f"   Warning: Failed court probe for {feature}: {e}")

    return pd.DataFrame(court_data)


def compute_size_bias_probe(
    train_df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Compute case size bias analysis."""
    print("📏 Computing size bias probe...")

    size_data = []

    # Create case size if not exists
    if "case_size" not in train_df.columns:
        train_df["case_size"] = train_df.groupby("case_id")["case_id"].transform(
            "count"
        )

    for feature in features:
        if feature not in train_df.columns:
            continue

        data = train_df[[feature, "case_size"]].dropna()
        if len(data) == 0 or data[feature].std() == 0:
            continue

        try:
            # Correlation with case size
            corr_size, _ = spearmanr(data[feature], data["case_size"])

            # Partial correlation with outcome given case size
            partial_corr_outcome = np.nan
            if "bin" in train_df.columns:
                full_data = train_df[[feature, "case_size", "bin"]].dropna()
                if len(full_data) > 10:
                    # Simple partial correlation approximation
                    from scipy.stats import pearsonr

                    # Residualize feature against case_size
                    from sklearn.linear_model import LinearRegression

                    reg = LinearRegression()
                    reg.fit(full_data[["case_size"]], full_data[feature])
                    feature_resid = full_data[feature] - reg.predict(
                        full_data[["case_size"]]
                    )

                    # Correlation of residualized feature with outcome
                    if feature_resid.std() > 0:
                        partial_corr_outcome, _ = pearsonr(
                            feature_resid, full_data["bin"]
                        )

            size_data.append(
                {
                    "feature": feature,
                    "corr_case_size": corr_size,
                    "partial_corr_outcome": partial_corr_outcome,
                    "size_bias_flag": abs(corr_size) > 0.3
                    and abs(partial_corr_outcome) < 0.1,
                }
            )

        except Exception as e:
            print(f"   Warning: Failed size probe for {feature}: {e}")

    return pd.DataFrame(size_data)


def create_final_feature_set_analysis(
    keep_features: List[str],
    drop_features: List[str],
    drop_reasons: Dict[str, str],
    court_probe_df: pd.DataFrame,
    size_probe_df: pd.DataFrame,
    comprehensive_df: pd.DataFrame,
    output_dir: Path,
):
    """Create analysis of final kept feature set."""
    print("📋 Creating final feature set analysis...")

    # Create final feature set directory
    final_dir = output_dir / "final_feature_set"
    final_dir.mkdir(exist_ok=True)

    # 1. Save kept and dropped features
    with open(final_dir / "kept_features.txt", "w") as f:
        f.write("# Final Kept Features for Training\n")
        f.write(f"# Total: {len(keep_features)} features\n")
        f.write("# Applied filtering rules:\n")
        f.write("# - Drop if missing_pct > 20%\n")
        f.write("# - Drop if zero_pct > 95%\n")
        f.write("# - Drop if psi_train_dev > 0.25\n")
        f.write("# - Drop if temporal drift flag\n\n")
        for feature in sorted(keep_features):
            f.write(f"{feature}\n")

    # 2. Save dropped features with reasons
    dropped_df = pd.DataFrame(
        [{"feature": f, "drop_reason": drop_reasons[f]} for f in drop_features]
    )
    dropped_df.to_csv(final_dir / "dropped_features.csv", index=False)

    # 3. Final feature statistics
    final_stats = comprehensive_df[
        comprehensive_df["feature"].isin(keep_features)
    ].copy()
    final_stats.to_csv(final_dir / "final_feature_statistics.csv", index=False)

    # 4. Court probe results (if available)
    if len(court_probe_df) > 0:
        court_probe_df.to_csv(final_dir / "court_proxy_probe.csv", index=False)

    # 5. Size bias probe results
    if len(size_probe_df) > 0:
        size_probe_df.to_csv(final_dir / "size_bias_probe.csv", index=False)

    print(f"✓ Final feature set analysis saved to {final_dir}")

    return final_stats


def create_final_latex_tables(
    final_stats: pd.DataFrame,
    dropped_df: pd.DataFrame,
    court_probe_df: pd.DataFrame,
    size_probe_df: pd.DataFrame,
    output_dir: Path,
):
    """Create LaTeX tables for final feature analysis."""
    print("📝 Generating final feature set LaTeX tables...")

    latex_dir = output_dir / "final_feature_set" / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final feature summary table
    final_table = []
    final_table.append("\\begin{table}[htbp]")
    final_table.append("\\centering")
    final_table.append("\\caption{Final Feature Set Summary Statistics}")
    final_table.append("\\label{tab:final_features}")
    final_table.append("\\begin{tabular}{lrrrrrr}")
    final_table.append("\\toprule")
    final_table.append(
        "Feature & AUC & Zero\\% & Miss\\% & SMD(0→1) & SMD(1→2) & Transform \\\\"
    )
    final_table.append("\\midrule")

    # Sort by AUC descending
    final_sorted = final_stats.sort_values(
        "auc_macro", ascending=False, na_position="last"
    )

    for _, row in final_sorted.head(20).iterrows():  # Top 20
        feature_name = row["feature"].replace("interpretable_", "").replace("_", "\\_")
        line = f"{feature_name} & "
        line += f"{row.get('auc_macro', 0):.3f} & "
        line += f"{row.get('zero_pct', 0):.1f} & "
        line += f"{row.get('missing_pct', 0):.1f} & "
        line += f"{row.get('smd_01', 0):.2f} & "
        line += f"{row.get('smd_12', 0):.2f} & "
        line += f"{row.get('transform', 'none')} \\\\"
        final_table.append(line)

    final_table.append("\\bottomrule")
    final_table.append("\\end{tabular}")
    final_table.append("\\end{table}")

    with open(latex_dir / "final_feature_summary.tex", "w") as f:
        f.write("\n".join(final_table))

    # 2. Dropped features table
    drop_table = []
    drop_table.append("\\begin{table}[htbp]")
    drop_table.append("\\centering")
    drop_table.append("\\caption{Dropped Features with Reasons}")
    drop_table.append("\\label{tab:dropped_features}")
    drop_table.append("\\begin{tabular}{lp{8cm}}")
    drop_table.append("\\toprule")
    drop_table.append("Feature & Drop Reason \\\\")
    drop_table.append("\\midrule")

    for _, row in dropped_df.head(20).iterrows():  # First 20 dropped
        feature_name = row["feature"].replace("interpretable_", "").replace("_", "\\_")
        reason = row["drop_reason"].replace("_", "\\_").replace("%", "\\%")
        line = f"{feature_name} & {reason} \\\\"
        drop_table.append(line)

    drop_table.append("\\bottomrule")
    drop_table.append("\\end{tabular}")
    drop_table.append("\\end{table}")

    with open(latex_dir / "dropped_features.tex", "w") as f:
        f.write("\n".join(drop_table))

    print(f"✓ Generated final feature set LaTeX tables")


def create_final_figures(
    final_stats: pd.DataFrame, train_df: pd.DataFrame, output_dir: Path
):
    """Create figures for final feature set."""
    print("📊 Generating final feature set figures...")

    figures_dir = output_dir / "final_feature_set" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final feature importance ranking
    top_features = final_stats.nlargest(15, "auc_macro")

    fig, ax = plt.subplots(figsize=(10, 8))

    feature_names = [
        f.replace("interpretable_", "").replace("_", " ").title()
        for f in top_features["feature"]
    ]
    y_pos = np.arange(len(feature_names))

    ax.barh(y_pos, top_features["auc_macro"], color="skyblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Macro AUC")
    ax.set_title(
        "Top 15 Final Features by Predictiveness", fontsize=14, fontweight="bold"
    )
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "final_feature_ranking.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Feature category distribution in final set
    category_counts = final_stats["category"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
        colors=["lightcoral", "lightblue", "lightgreen"],
    )
    ax.set_title("Final Feature Set by Category", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "final_category_distribution.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Generated final feature set figures")


def main():
    """Main function to generate all analysis outputs."""
    print("🚀 Starting comprehensive feature analysis for academic paper...")
    print("=" * 70)

    # Create output directory
    output_dir = Path("docs/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # PART 1: Original feature analysis
    print("\n" + "=" * 50)
    print("PART 1: Basic Feature Analysis")
    print("=" * 50)

    # Load data and get features
    df = load_sample_data(n_samples=10000)  # Use 10k samples for efficiency
    approved_features, categories = get_approved_features(df)

    # Get feature explanations
    explanations = get_feature_explanations()

    # Compute statistics
    stats_df = compute_feature_statistics(df, approved_features)

    # Generate outputs
    create_latex_tables(stats_df, categories, explanations, output_dir)
    create_pdf_figures(df, stats_df, categories, output_dir)
    create_summary_document(stats_df, categories, explanations, output_dir)

    # PART 2: Statistical audit (fold 3)
    print("\n" + "=" * 50)
    print("PART 2: Statistical Audit (Fold 3)")
    print("=" * 50)

    # Load fold 3 data
    train_df, dev_df, test_df, fold_metadata = load_fold3_data()

    # Get interpretable features only
    interpretable_features = [
        f for f in approved_features if f.startswith("interpretable_")
    ]

    # Compute all statistical metrics
    health_df = compute_dataset_health(
        train_df, dev_df, test_df, interpretable_features
    )
    hygiene_df = compute_feature_hygiene(train_df, interpretable_features)
    predict_df = compute_univariate_predictiveness(train_df, interpretable_features)
    stability_df = compute_temporal_stability(train_df, dev_df, interpretable_features)
    profile_df = compute_per_bucket_profiles(train_df, interpretable_features)

    # Fit ordered logit baseline
    ordered_logit = compute_ordered_logit_baseline(
        train_df, dev_df, interpretable_features, output_dir
    )

    # Generate statistical outputs
    create_statistical_latex_tables(
        health_df,
        hygiene_df,
        predict_df,
        stability_df,
        profile_df,
        ordered_logit,
        output_dir,
    )
    create_statistical_figures(
        train_df, dev_df, interpretable_features, predict_df, output_dir
    )

    # Create comprehensive CSV
    comprehensive_df = create_comprehensive_feature_csv(
        stats_df, hygiene_df, predict_df, stability_df, output_dir
    )

    # PART 3: Feature filtering and final set analysis
    print("\n" + "=" * 50)
    print("PART 3: Feature Filtering & Final Set")
    print("=" * 50)

    # Apply filtering rules
    filtering_results = apply_feature_filtering_rules(comprehensive_df)
    keep_features = filtering_results["keep"]
    drop_features = filtering_results["drop"]
    drop_reasons = filtering_results["drop_reasons"]

    # Run additional probes on kept features
    court_probe_df = compute_court_proxy_probe(train_df, keep_features)
    size_probe_df = compute_size_bias_probe(train_df, keep_features)

    # Check for additional drops from probes
    additional_drops = []
    if len(court_probe_df) > 0:
        # Drop features with high court MI but low outcome signal
        court_issues = court_probe_df[
            (court_probe_df["mi_court"] > court_probe_df["mi_court"].quantile(0.95))
            & (court_probe_df["auc_outcome"] < 0.55)
        ]
        additional_drops.extend(court_issues["feature"].tolist())

    if len(size_probe_df) > 0:
        # Drop features flagged for size bias
        size_issues = size_probe_df[size_probe_df["size_bias_flag"] == True]
        additional_drops.extend(size_issues["feature"].tolist())

    # Update final keep/drop lists
    for feature in additional_drops:
        if feature in keep_features:
            keep_features.remove(feature)
            drop_features.append(feature)
            if feature in court_issues["feature"].values:
                drop_reasons[feature] = "court_proxy"
            if feature in size_issues["feature"].values:
                drop_reasons[feature] = "size_bias"

    print(f"   - Additional drops from probes: {len(additional_drops)}")
    print(f"   - Final keep: {len(keep_features)} features")
    print(f"   - Final drop: {len(drop_features)} features")

    # Create final analysis outputs
    dropped_df = pd.DataFrame(
        [{"feature": f, "drop_reason": drop_reasons[f]} for f in drop_features]
    )

    final_stats = create_final_feature_set_analysis(
        keep_features,
        drop_features,
        drop_reasons,
        court_probe_df,
        size_probe_df,
        comprehensive_df,
        output_dir,
    )

    create_final_latex_tables(
        final_stats, dropped_df, court_probe_df, size_probe_df, output_dir
    )
    create_final_figures(final_stats, train_df, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📁 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("📊 LaTeX Tables:")
    for tex_file in sorted((output_dir / "latex").glob("*.tex")):
        print(f"   - {tex_file.name}")
    print("📈 PDF Figures:")
    for pdf_file in sorted((output_dir / "figures").glob("*.pdf")):
        print(f"   - {pdf_file.name}")
    print("📄 Documentation & Data:")
    print("   - FEATURE_ANALYSIS_SUMMARY.md")
    print("   - comprehensive_feature_metrics.csv")
    if (output_dir / "ordered_logit_results.csv").exists():
        print("   - ordered_logit_results.csv")

    # Print key findings
    print("\n🔍 KEY FINDINGS:")
    print(f"   - Total features analyzed: {len(interpretable_features)}")
    print(f"   - Features kept for training: {len(keep_features)}")
    print(f"   - Features dropped: {len(drop_features)}")
    print(f"   - Best AUC in final set: {final_stats['auc_macro'].max():.3f}")
    print(f"   - Monotonic in final set: {final_stats['monotonic'].sum()}")

    if ordered_logit and "metrics" in ordered_logit:
        print(f"\n📊 Ordered Logit Baseline Performance:")
        print(f"   - Accuracy: {ordered_logit['metrics']['accuracy']:.3f}")
        print(f"   - Macro F1: {ordered_logit['metrics']['macro_f1']:.3f}")
        print(f"   - Pseudo R²: {ordered_logit['metrics']['pseudo_r2']:.3f}")

    print(f"\n📋 FINAL FEATURE SET:")
    print(f"   - Saved to: docs/feature_analysis/final_feature_set/")
    print(f"   - kept_features.txt ({len(keep_features)} features)")
    print(f"   - dropped_features.csv ({len(drop_features)} features)")
    print(f"   - Ready for training pipeline!")

    print("\n💡 Ready for academic paper inclusion!")


if __name__ == "__main__":
    main()
