#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX summary tables/macros from POLAR CV outputs.

Inputs:
  --oof  : path to oof_predictions.jsonl (from train_polar_cv)
  --cv   : path to cv_results.json (from train_polar_cv)
  --out  : output directory for .tex files

Writes:
  tables/
    tbl_dataset_summary.tex
    tbl_cv_performance.tex
    tbl_calibration.tex
    tbl_temporal_integrity.tex
    tbl_class_distribution.tex
    tbl_robustness_quotes_per_case.tex
    tbl_hyperparams.tex
  macros/
    paper_macros.tex  (e.g., \\Nquotes, \\Ncases, \\QWKmean, \\QWKci, ...)
"""
import json, argparse, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict


# ----------------- small helpers -----------------
def _ci_mean(series, alpha=0.05):
    # normal approx
    s = pd.Series(series).dropna().astype(float)
    n = len(s)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    z = 1.96  # 95%
    half = z * (sd / np.sqrt(max(n, 1)))
    return (m, m - half, m + half)


def _bootstrap_ci_by_case(
    df, prob_cols, y_col="y", case_col="case_id", n_boot=1000, seed=42, metric="qwk"
):
    """
    Bootstrap CI over unique case_ids to respect grouping.
    metric in {"qwk","macro_f1","mae","brier","ece"}
    Expects per-quote rows containing case_id, y, and calibrated probs.
    """
    rng = np.random.default_rng(seed)
    cases = df[case_col].dropna().unique().tolist()
    if len(cases) == 0:
        return np.nan, np.nan, np.nan

    def safe_qwk(y_true, y_pred):
        present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
        if len(present) < 2:
            return 0.0
        idx = {c: i for i, c in enumerate(present)}
        yt = np.array([idx[int(y)] for y in y_true])
        yp = np.array([idx[int(y)] for y in y_pred])
        k = len(present)
        w = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
        from sklearn.metrics import confusion_matrix

        O = confusion_matrix(yt, yp, labels=range(k))
        if O.sum() == 0:
            return 0.0
        row = O.sum(1, keepdims=True)
        col = O.sum(0, keepdims=True)
        E = row @ col / O.sum()
        denom = (w * E).sum()
        if denom == 0:
            return 0.0
        return 1.0 - (w * O).sum() / denom

    def macro_f1_present(y_true, y_pred):
        from sklearn.metrics import f1_score

        present = sorted(set(y_true))
        if len(present) < 2:
            return 0.0
        return f1_score(
            y_true, y_pred, average="macro", labels=present, zero_division=0
        )

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

    def brier(y_true, P):
        y = np.array(y_true, int)
        onehot = np.zeros_like(P)
        onehot[np.arange(len(P)), y] = 1.0
        return float(np.mean(((P - onehot) ** 2).sum(axis=1)))

    def ece(y_true, P, n_bins=10):
        # per-class ECE mean
        y = np.array(y_true, int)
        C = P.shape[1]
        acc = []
        for c in range(C):
            p = P[:, c]
            t = (y == c).astype(float)
            bins = np.linspace(0, 1, n_bins + 1)
            e = 0.0
            for i in range(n_bins):
                left, right = bins[i], bins[i + 1]
                mask = (p >= left) & (p < (right if i < n_bins - 1 else right + 1e-12))
                if mask.sum() > 0:
                    e += (mask.mean()) * abs(p[mask].mean() - t[mask].mean())
            acc.append(e)
        return float(np.mean(acc))

    stats = []
    for _ in range(n_boot):
        samp_cases = rng.choice(cases, size=len(cases), replace=True)
        sub = df[df[case_col].isin(samp_cases)]
        P = sub[prob_cols].to_numpy()
        y = sub[y_col].to_numpy()

        # Guard against empty arrays
        if P.size == 0 or len(y) == 0:
            stats.append(np.nan)
            continue

        yhat = P.argmax(axis=1)
        if metric == "qwk":
            v = safe_qwk(y, yhat)
        elif metric == "macro_f1":
            v = macro_f1_present(y, yhat)
        elif metric == "mae":
            v = mae(y, yhat)
        elif metric == "brier":
            v = brier(y, P)
        elif metric == "ece":
            v = ece(y, P)
        else:
            v = np.nan
        stats.append(v)
    m = np.mean(stats)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(m), float(lo), float(hi)


def _fmt_mean_ci(m, lo, hi, decimals=3):
    if any(np.isnan([m, lo, hi])):
        return r"--"
    return f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _to_tex(df, path, caption, label, index=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            df.to_latex(
                index=index,
                escape=True,
                float_format="%.3f",
                caption=caption,
                label=label,
                longtable=False,
                bold_rows=False,
                column_format="l" + "r" * (df.shape[1] - 1),
            )
        )


# ----------------- main build -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    tdir = out_dir / "tables"
    mdir = out_dir / "macros"
    tdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    oof = pd.read_json(args.oof, lines=True)
    with open(args.cv, "r") as f:
        cv = json.load(f)

    # Ensure expected fields exist - check for both polar_ and polr_ prefixes
    prob_cols = []
    # First try polr_ prefix (current output format)
    for k in ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]:
        if k in oof.columns:
            prob_cols.append(k)
    # Fallback to polar_ prefix (legacy)
    if len(prob_cols) != 3:
        for k in ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]:
            if k in oof.columns:
                prob_cols.append(k)

    # Try expanding from class_probs dict
    if len(prob_cols) != 3 and "polr_class_probs" in oof.columns:
        # expand dict to columns
        tmp = oof["polr_class_probs"].apply(lambda d: pd.Series(d))
        tmp.columns = [f"polr_prob_{c}" for c in tmp.columns]
        oof = pd.concat([oof.drop(columns=["polr_class_probs"]), tmp], axis=1)
        prob_cols = ["polr_prob_low", "polr_prob_medium", "polr_prob_high"]
    elif len(prob_cols) != 3 and "polar_class_probs" in oof.columns:
        # expand dict to columns (legacy)
        tmp = oof["polar_class_probs"].apply(lambda d: pd.Series(d))
        tmp.columns = [f"polar_prob_{c}" for c in tmp.columns]
        oof = pd.concat([oof.drop(columns=["polar_class_probs"]), tmp], axis=1)
        prob_cols = ["polar_prob_low", "polar_prob_medium", "polar_prob_high"]

    # Dataset summary ------------------------------------------------
    N_quotes = len(oof)
    N_cases = oof["case_id"].nunique() if "case_id" in oof.columns else np.nan
    quotes_per_case = (
        oof.groupby("case_id").size()
        if "case_id" in oof.columns
        else pd.Series(dtype=int)
    )
    qpc_stats = {
        "Mean quotes/case": quotes_per_case.mean() if len(quotes_per_case) else np.nan,
        "Median quotes/case": (
            quotes_per_case.median() if len(quotes_per_case) else np.nan
        ),
        "Min quotes/case": quotes_per_case.min() if len(quotes_per_case) else np.nan,
        "Max quotes/case": quotes_per_case.max() if len(quotes_per_case) else np.nan,
    }
    y_counts = oof["y"].value_counts().reindex([0, 1, 2], fill_value=0)
    ds = pd.DataFrame(
        {
            "Statistic": [
                "# Quotes",
                "# Cases",
                "Mean quotes/case",
                "Median quotes/case",
                "Min quotes/case",
                "Max quotes/case",
                "Class 0 (low)",
                "Class 1 (med)",
                "Class 2 (high)",
            ],
            "Value": [
                N_quotes,
                N_cases,
                qpc_stats["Mean quotes/case"],
                qpc_stats["Median quotes/case"],
                qpc_stats["Min quotes/case"],
                qpc_stats["Max quotes/case"],
                int(y_counts.get(0, 0)),
                int(y_counts.get(1, 0)),
                int(y_counts.get(2, 0)),
            ],
        }
    )
    _to_tex(
        ds,
        tdir / "tbl_dataset_summary.tex",
        "Dataset overview for out-of-fold evaluation set.",
        "tab:dataset-summary",
    )

    # Per-fold performance (DEV in your current logging) -------------
    rows = []
    for k, v in cv["folds"].items():
        dm = v["dev_metrics"]
        rows.append(
            {
                "Fold": int(k),
                "QWK": dm.get("qwk", np.nan),
                "Macro-F1": dm.get("macro_f1", np.nan),
                "MAE": dm.get("mae", np.nan),
                "Brier": dm.get("brier", np.nan),
                "ECE": dm.get("ece", np.nan),
            }
        )
    perf = pd.DataFrame(rows).sort_values("Fold")
    # add mean±sd row
    mean_row = {
        "Fold": "Mean",
        "QWK": perf["QWK"].mean(),
        "Macro-F1": perf["Macro-F1"].mean(),
        "MAE": perf["MAE"].mean(),
        "Brier": perf["Brier"].mean(),
        "ECE": perf["ECE"].mean(),
    }
    perf = pd.concat([perf, pd.DataFrame([mean_row])], ignore_index=True)
    _to_tex(
        perf,
        tdir / "tbl_cv_performance.tex",
        "Per-fold validation performance (temporal DEV) and mean.",
        "tab:cv-performance",
        index=False,
    )

    # OOF (DEV) bootstrap CIs by case --------------------------------
    # If you also emit TEST OOF in the future, point this at that file similarly.
    if (
        len(prob_cols) == 3
        and all(c in oof.columns for c in prob_cols + ["y", "case_id"])
        and len(oof) > 0
    ):
        m_qwk, lo_qwk, hi_qwk = _bootstrap_ci_by_case(oof, prob_cols, metric="qwk")
        m_f1, lo_f1, hi_f1 = _bootstrap_ci_by_case(oof, prob_cols, metric="macro_f1")
        m_mae, lo_mae, hi_mae = _bootstrap_ci_by_case(oof, prob_cols, metric="mae")
        m_br, lo_br, hi_br = _bootstrap_ci_by_case(oof, prob_cols, metric="brier")
        m_ece, lo_ece, hi_ece = _bootstrap_ci_by_case(oof, prob_cols, metric="ece")

        cal_tbl = pd.DataFrame(
            {
                "Metric": ["QWK ↑", "Macro-F1 ↑", "MAE ↓", "Brier ↓", "ECE ↓"],
                "Mean [95% CI]": [
                    _fmt_mean_ci(m_qwk, lo_qwk, hi_qwk),
                    _fmt_mean_ci(m_f1, lo_f1, hi_f1),
                    _fmt_mean_ci(m_mae, lo_mae, hi_mae),
                    _fmt_mean_ci(m_br, lo_br, hi_br),
                    _fmt_mean_ci(m_ece, lo_ece, hi_ece),
                ],
            }
        )
        _to_tex(
            cal_tbl,
            tdir / "tbl_calibration.tex",
            "Out-of-fold performance with 95\\% case-level bootstrap CIs.",
            "tab:oof-performance",
            index=False,
        )
    else:
        # still write an empty shell
        pd.DataFrame({"Metric": [], "Mean [95% CI]": []}).to_latex(
            tdir / "tbl_calibration.tex"
        )

    # Temporal integrity table (date ranges) --------------------------
    # If you logged split metadata per fold, you can load it from cv; we'll summarize DEV characteristics.
    trows = []
    for k, v in cv["folds"].items():
        dm = v.get("dev_metadata", {})
        tr = v.get("weight_stats", {}).get("train_core", {}).get("quotes_per_case", {})
        trows.append(
            {
                "Fold": int(k),
                "DEV cases": dm.get("n_cases", np.nan),
                "DEV quotes": dm.get("n_quotes", np.nan),
                "DEV classes": dm.get("n_classes", np.nan),
                "Frac used": dm.get("frac_used", np.nan),
                "Fallback?": dm.get("used_fallback", dm.get("fallback", False)),
            }
        )
    tmp = pd.DataFrame(trows).sort_values("Fold")
    _to_tex(
        tmp,
        tdir / "tbl_temporal_integrity.tex",
        "Temporal DEV characteristics per fold (contiguous tails; embargo enforced).",
        "tab:temporal-integrity",
        index=False,
    )

    # Class distribution table ---------------------------------------
    class_tbl = pd.DataFrame(
        {
            "Class": ["Low (0)", "Medium (1)", "High (2)"],
            "Count": [
                int(y_counts.get(0, 0)),
                int(y_counts.get(1, 0)),
                int(y_counts.get(2, 0)),
            ],
        }
    )
    _to_tex(
        class_tbl,
        tdir / "tbl_class_distribution.tex",
        "Class distribution in the out-of-fold set.",
        "tab:class-dist",
        index=False,
    )

    # Robustness: performance by quotes-per-case buckets --------------
    if "case_id" in oof.columns:
        qpc = oof.groupby("case_id").size().rename("qpc")
        oof2 = oof.merge(qpc, left_on="case_id", right_index=True, how="left")

        # buckets: 1, 2–3, 4–7, 8+
        def buck(n):
            if n <= 1:
                return "1"
            if n <= 3:
                return "2–3"
            if n <= 7:
                return "4–7"
            return "8+"

        oof2["qpc_bucket"] = oof2["qpc"].apply(buck)
        rows = []
        for b, sub in oof2.groupby("qpc_bucket"):
            if all(c in sub.columns for c in prob_cols + ["y"]):
                P = sub[prob_cols].to_numpy()
                y = sub["y"].to_numpy()
                yhat = P.argmax(axis=1)

                # quick stats (no bootstrap here to keep it light)
                def safe_qwk(y_true, y_pred):
                    present = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
                    if len(present) < 2:
                        return 0.0
                    idx = {c: i for i, c in enumerate(present)}
                    yt = np.array([idx[int(y)] for y in y_true])
                    yp = np.array([idx[int(y)] for y in y_pred])
                    k = len(present)
                    w = np.zeros((k, k))
                    for i in range(k):
                        for j in range(k):
                            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
                    from sklearn.metrics import confusion_matrix

                    O = confusion_matrix(yt, yp, labels=range(k))
                    if O.sum() == 0:
                        return 0.0
                    row = O.sum(1, keepdims=True)
                    col = O.sum(0, keepdims=True)
                    E = row @ col / O.sum()
                    denom = (w * E).sum()
                    if denom == 0:
                        return 0.0
                    return 1.0 - (w * O).sum() / denom

                from sklearn.metrics import f1_score

                qwk = safe_qwk(y, yhat)
                macro_f1 = f1_score(
                    y, yhat, average="macro", labels=sorted(set(y)), zero_division=0
                )
                mae = float(np.mean(np.abs(y - yhat)))
                onehot = np.zeros_like(P)
                onehot[np.arange(len(y)), y] = 1
                brier = float(np.mean(((P - onehot) ** 2).sum(axis=1)))
                rows.append(
                    {
                        "Quotes/case": b,
                        "QWK": qwk,
                        "Macro-F1": macro_f1,
                        "MAE": mae,
                        "Brier": brier,
                        "n": len(sub),
                    }
                )
        rob = pd.DataFrame(rows).sort_values(["Quotes/case"])
        _to_tex(
            rob,
            tdir / "tbl_robustness_quotes_per_case.tex",
            "Performance stratified by quotes-per-case bucket (OOF).",
            "tab:robustness-qpc",
            index=False,
        )

    # Hyperparameters table ------------------------------------------
    hp_rows = []
    for k, v in cv["folds"].items():
        p = v.get("best_params", {})
        hp_rows.append({"Fold": int(k), **p})
    hp = pd.DataFrame(hp_rows).sort_values("Fold")
    _to_tex(
        hp,
        tdir / "tbl_hyperparams.tex",
        "Best hyperparameters per fold.",
        "tab:hyperparams",
        index=False,
    )

    # Macros ----------------------------------------------------------
    m_qwk, lo_qwk, hi_qwk = (np.nan, np.nan, np.nan)
    if all(c in oof.columns for c in prob_cols + ["y", "case_id"]):
        m_qwk, lo_qwk, hi_qwk = _bootstrap_ci_by_case(oof, prob_cols, metric="qwk")
    macros = [
        rf"\newcommand{{\Nquotes}}{{{N_quotes}}}",
        rf"\newcommand{{\Ncases}}{{{N_cases}}}",
        rf"\newcommand{{\QWKmean}}{{{(0 if np.isnan(m_qwk) else round(m_qwk,3))}}}",
        rf"\newcommand{{\QWKciLo}}{{{(0 if np.isnan(lo_qwk) else round(lo_qwk,3))}}}",
        rf"\newcommand{{\QWKciHi}}{{{(0 if np.isnan(hi_qwk) else round(hi_qwk,3))}}}",
    ]
    with open(mdir / "paper_macros.tex", "w") as f:
        f.write("% Auto-generated macros\n")
        for line in macros:
            f.write(line + "\n")

    print(f"Wrote LaTeX tables to: {tdir}")
    print(f"Wrote LaTeX macros to: {mdir/'paper_macros.tex'}")


if __name__ == "__main__":
    main()
