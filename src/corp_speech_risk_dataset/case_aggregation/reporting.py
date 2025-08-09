"""Case-level reporting: summary statistics and figures for positional features.

This module aggregates quote-level positional features (produced by the
positional extraction step) into case-level summary statistics suitable for
academic reporting, and generates basic figures.

Inputs: a directory of JSONL files containing quote dictionaries augmented with
fields like `docket_number`, `docket_token_start`, `global_token_start`,
`num_tokens`, and optionally `feature_error`. Quotes should retain `_src` to
derive `case_id` via the `.../<case_id>/entries/...` pattern.

Outputs:
- Per-quote Polars DataFrame with derived `case_id` and flags
- Per-case Polars DataFrame with summary metrics (match_rate, early fractions,
  etc.) and an optional `final_judgement_real` if present in the quotes
- Optional PNG figures saved to a figures directory

Note: We use Polars for tabular processing for performance and clarity. Figures
are generated with matplotlib if available; otherwise figure generation is
skipped.
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

import time
from .positional_features import load_jsonl
from .progress import Progress


CASE_ID_RE = re.compile(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/")


def _extract_case_id_from_src(src: str) -> Optional[str]:
    m = CASE_ID_RE.search(src)
    return m.group(1) if m else None


def load_positions_dir(positions_dir: str, pattern: str = "**/*.jsonl") -> pl.DataFrame:
    """Load augmented quotes JSONL files from a directory into a Polars DataFrame.

    Adds `case_id` derived from `_src` when available.
    """
    files = [
        f
        for f in glob(os.path.join(positions_dir, pattern), recursive=True)
        if os.path.isfile(f)
    ]
    rows: List[Dict[str, Any]] = []
    prog = Progress(label="load_positions", total=len(files))
    t0 = time.time()
    for i, f in enumerate(files, start=1):
        for r in load_jsonl(f):
            if "case_id" not in r:
                src = r.get("_src") or r.get("src") or ""
                if isinstance(src, str) and src:
                    cid = _extract_case_id_from_src(src)
                    if cid:
                        r["case_id"] = cid
            rows.append(r)
        prog.update(i, extra=os.path.basename(f))
    prog.finish()
    print(f"Loaded {len(files)} files in {time.time()-t0:.2f}s | {len(rows)} rows")
    if not rows:
        return pl.DataFrame([])
    # Allow mixed-types across rows; we'll coerce columns below
    df = pl.DataFrame(rows, strict=False)
    # Ensure expected columns exist with sane dtypes
    for col, dtype in (
        ("case_id", pl.Utf8),
        ("doc_id", pl.Utf8),
        ("docket_number", pl.Int64),
        ("docket_token_start", pl.Int64),
        ("global_token_start", pl.Int64),
        ("num_tokens", pl.Int64),
        ("feature_error", pl.Utf8),
        ("final_judgement_real", pl.Float64),
    ):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))

    # Matched flag
    df = df.with_columns((pl.col("feature_error").is_null()).alias("matched"))
    return df


def compute_case_summaries(
    df_quotes: pl.DataFrame, cases_root: Optional[str] = None
) -> pl.DataFrame:
    """Compute per-case summary statistics from quote-level DataFrame.

    If `cases_root` is provided and accessible, it is not currently used here,
    because accurate total docket counts are not necessary for summary stats; we
    estimate per-case total dockets as max observed `docket_number` among matched
    quotes. This provides conservative early/third fractions for matched quotes.
    """
    if df_quotes.is_empty():
        return pl.DataFrame([])

    # Max docket per case from matched quotes
    df_case_props = (
        df_quotes.filter(pl.col("matched"))
        .group_by("case_id")
        .agg(
            pl.col("docket_number").max().alias("max_docket"),
            pl.count().alias("n_matched"),
        )
    )

    # Total quotes (matched + unmatched) per case
    df_case_counts = df_quotes.group_by("case_id").agg(
        pl.count().alias("n_quotes_total")
    )

    # Join props back to quotes to label early fractions
    df_joined = df_quotes.join(df_case_props, on="case_id", how="left")
    df_joined = df_joined.with_columns(
        (
            (pl.col("docket_number") <= (pl.col("max_docket") / 2).floor())
            & pl.col("matched")
        ).alias("in_first_half_dockets"),
        (
            (pl.col("docket_number") <= (pl.col("max_docket") / 3).floor())
            & pl.col("matched")
        ).alias("in_first_third_dockets"),
        (
            ((pl.col("global_token_start") + pl.col("num_tokens")) <= 2500)
            & pl.col("matched")
        ).alias("within_2500_tokens"),
    )

    # Per-case judgement value (if present); flag conflicts
    # Use a row filter rather than drop_nulls on a single Series to avoid
    # mismatched lengths with case_id.
    df_judgement = (
        df_joined.filter(pl.col("final_judgement_real").is_not_null())
        .select(["case_id", "final_judgement_real"])
        .group_by("case_id")
        .agg(
            pl.col("final_judgement_real").n_unique().alias("judgement_unique_count"),
            pl.col("final_judgement_real").first().alias("final_judgement_real"),
        )
    )
    df_judgement = df_judgement.with_columns(
        (pl.col("judgement_unique_count") > 1).alias("judgement_conflict")
    )

    # Aggregate per-case summary metrics
    t1 = time.time()
    df_case_summary = (
        df_joined.group_by("case_id")
        .agg(
            pl.max("max_docket").alias("max_docket"),
            pl.col("matched").sum().alias("n_matched"),
            pl.count().alias("n_quotes_total"),
            (pl.col("matched").sum() / pl.count()).alias("match_rate"),
            (pl.count() - pl.col("matched").sum()).alias("n_unmatched"),
            # Centrality of positions among matched
            pl.col("docket_number")
            .filter(pl.col("matched"))
            .median()
            .alias("median_docket_number"),
            pl.col("docket_token_start")
            .filter(pl.col("matched"))
            .median()
            .alias("median_docket_token_start"),
            pl.col("global_token_start")
            .filter(pl.col("matched"))
            .median()
            .alias("median_global_token_start"),
            # Early coverage fractions among matched
            pl.col("in_first_half_dockets").mean().alias("fraction_first_half_dockets"),
            pl.col("in_first_third_dockets")
            .mean()
            .alias("fraction_first_third_dockets"),
            pl.col("within_2500_tokens").mean().alias("fraction_within_2500_tokens"),
        )
        .join(df_case_counts, on="case_id", how="left")
        .join(
            df_judgement.select(
                "case_id", "final_judgement_real", "judgement_conflict"
            ),
            on="case_id",
            how="left",
        )
    )
    print(
        f"Computed case summaries in {time.time()-t1:.2f}s for {df_case_summary.height} cases"
    )

    return df_case_summary


def generate_figures(
    df_quotes: pl.DataFrame, df_cases: pl.DataFrame, figures_dir: str
) -> None:
    """Generate basic figures for academic reporting. Saves PNG files.

    If matplotlib is unavailable, this function is a no-op.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    os.makedirs(figures_dir, exist_ok=True)

    # 1) Distribution of match rate across cases
    if not df_cases.is_empty() and "match_rate" in df_cases.columns:
        vals = df_cases.select("match_rate").to_series().drop_nans().to_list()
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=20, color="#4C78A8")
        plt.xlabel("Match rate per case")
        plt.ylabel("Count of cases")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "match_rate_hist.png"), dpi=150)
        plt.close()

    # 2) Scatter of match_rate vs final_judgement_real (log-scale x)
    if set(["match_rate", "final_judgement_real"]).issubset(df_cases.columns):
        sub = df_cases.select(["final_judgement_real", "match_rate"]).drop_nulls()
        if sub.height > 0:
            x = sub["final_judgement_real"].to_list()
            y = sub["match_rate"].to_list()
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, alpha=0.6)
            plt.xscale("symlog")
            plt.xlabel("Final judgement (symlog)")
            plt.ylabel("Match rate")
            plt.tight_layout()
            plt.savefig(
                os.path.join(figures_dir, "match_rate_vs_judgement.png"), dpi=150
            )
            plt.close()

    # 3) Scatter of early fraction vs final judgement
    if set(["fraction_first_half_dockets", "final_judgement_real"]).issubset(
        df_cases.columns
    ):
        sub = df_cases.select(
            ["final_judgement_real", "fraction_first_half_dockets"]
        ).drop_nulls()
        if sub.height > 0:
            x = sub["final_judgement_real"].to_list()
            y = sub["fraction_first_half_dockets"].to_list()
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, alpha=0.6, color="#F58518")
            plt.xscale("symlog")
            plt.xlabel("Final judgement (symlog)")
            plt.ylabel("Fraction quotes in 1st half of dockets")
            plt.tight_layout()
            plt.savefig(
                os.path.join(figures_dir, "early_fraction_vs_judgement.png"), dpi=150
            )
            plt.close()

    # 4) Global token start distribution (matched quotes)
    if not df_quotes.is_empty():
        gts = (
            df_quotes.filter(pl.col("matched"))
            .select("global_token_start")
            .drop_nulls()
            .to_series()
            .to_list()
        )
        if gts:
            plt.figure(figsize=(6, 4))
            plt.hist(gts, bins=50, color="#54A24B")
            plt.xlabel("Global token start (matched quotes)")
            plt.ylabel("Count of quotes")
            plt.tight_layout()
            plt.savefig(
                os.path.join(figures_dir, "global_token_start_hist.png"), dpi=150
            )
            plt.close()


def save_tables(
    df_quotes: pl.DataFrame, df_cases: pl.DataFrame, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df_quotes.write_csv(os.path.join(output_dir, "quotes_with_case_id.csv"))
    df_cases.write_csv(os.path.join(output_dir, "case_level_summary.csv"))


def summarize_positions(
    positions_dir: str,
    output_dir: str,
    figures_dir: Optional[str] = None,
    pattern: str = "**/*.jsonl",
    cases_root: Optional[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """High-level API: load positions, compute summaries, save outputs and figures.

    Returns the pair (df_quotes, df_cases).
    """
    df_quotes = load_positions_dir(positions_dir, pattern)
    df_cases = compute_case_summaries(df_quotes, cases_root=cases_root)
    save_tables(df_quotes, df_cases, output_dir)
    if figures_dir:
        generate_figures(df_quotes, df_cases, figures_dir)
    return df_quotes, df_cases
