#!/usr/bin/env python3
import os, sys, json, glob, logging, math
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ---------------------------- logging ----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("case_lr")

# ------------------------- fast jsonl load -----------------------
try:
    import orjson as _json

    def _loads_bytes(b):
        return _json.loads(b)

    def _dumps(o):
        return _json.dumps(o, option=_json.OPT_SERIALIZE_NUMPY)

    ORJ = True
except Exception:
    import json as _json

    def _loads_bytes(b):
        return json.loads(b.decode("utf-8"))

    def _dumps(o):
        return json.dumps(o, default=str).encode()

    ORJ = False


def load_jsonl_dir(dir_path: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(dir_path / "*.jsonl")))
    if not files:
        return pd.DataFrame()
    rows = []
    for fp in files:
        with open(fp, "rb") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(_loads_bytes(ln))
                except Exception:
                    try:
                        rows.append(json.loads(ln.decode("utf-8", errors="ignore")))
                    except Exception as e:
                        log.warning(f"Bad line in {fp}: {e}")
    return pd.DataFrame(rows)


# -------------------- per-case feature builder -------------------
def _clusters_from_binary(b: np.ndarray) -> Tuple[int, int, float]:
    """Number of positive runs, max run length, avg run length (positives only)."""
    if len(b) == 0 or b.max() == 0:
        return 0, 0, 0.0
    # find run starts/ends on 1s
    xb = np.r_[0, b, 0]
    starts = np.where((xb[1:-1] == 1) & (xb[:-2] == 0))[0]
    ends = np.where((xb[1:-1] == 1) & (xb[2:] == 0))[0]
    lens = (ends - starts + 1) if len(starts) else np.array([0])
    return len(lens), int(lens.max()), float(lens.mean())


# ----------- Helper: robust sort for per-case quote order ---------
def _sort_case_naturally(g: pd.DataFrame) -> pd.DataFrame:
    # Prefer docket structure, then global positions, then docket char positions; stable where ties.
    preferred_orders = [
        ["docket_number", "docket_token_start"],
        ["global_token_start"],
        ["global_char_start"],
        ["docket_char_start"],
    ]
    for cols in preferred_orders:
        if all(c in g.columns for c in cols):
            # ensure numeric for sorting
            g = g.copy()
            for c in cols:
                g[c] = pd.to_numeric(g[c], errors="coerce")
            return g.sort_values(cols, kind="mergesort")
    return g  # fallback: original order


# --------------- Early cutoff filtering helper --------------------
def _filter_early(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Return a per-case prefix subset according to `mode`:
      - 'first_doc': keep only quotes from the earliest docket_number per case; if missing, fall back to earliest global_token_start, else first row.
      - 'p10'/'p30'/'p50': keep quotes whose token-range end is within the first {10,30,50}% of the case by tokens.
    """
    if df.empty:
        return df

    def _first_doc(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        if "docket_number" in g.columns and g["docket_number"].notna().any():
            mn = pd.to_numeric(g["docket_number"], errors="coerce").min()
            sub = g[pd.to_numeric(g["docket_number"], errors="coerce") == mn]
            if len(sub) > 0:
                return sub
        # fallback: earliest global_token_start, else docket/global_char_start
        for col in [
            "global_token_start",
            "docket_token_start",
            "global_char_start",
            "docket_char_start",
        ]:
            if col in g.columns and g[col].notna().any():
                mn = pd.to_numeric(g[col], errors="coerce").min()
                sub = g[pd.to_numeric(g[col], errors="coerce") == mn]
                if len(sub) > 0:
                    return sub
        # final fallback: first row only
        return g.iloc[:1]

    def _prefix_by_percent(group: pd.DataFrame, frac: float) -> pd.DataFrame:
        g = group.copy()
        n = len(g)
        if n == 0:
            return g

        # Choose token-based range if available; fallback to char, then index
        if "global_token_start" in g.columns:
            start = (
                pd.to_numeric(g["global_token_start"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            # try to use end position with num_tokens if available
            if "num_tokens" in g.columns:
                toks = (
                    pd.to_numeric(g["num_tokens"], errors="coerce")
                    .fillna(1.0)
                    .clip(lower=1.0)
                    .to_numpy(dtype=float)
                )
            else:
                toks = np.ones_like(start)
            end = start + toks
        elif "global_char_start" in g.columns:
            start = (
                pd.to_numeric(g["global_char_start"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            end = start + 1.0
        else:
            # fallback: normalized index
            idx = np.arange(n, dtype=float)
            start = idx
            end = idx + 1.0

        s0 = np.nanmin(start) if np.isfinite(start).any() else 0.0
        e1 = np.nanmax(end) if np.isfinite(end).any() else float(n)
        span = max(e1 - s0, 1.0)
        cutoff = s0 + frac * span
        mask = end <= cutoff
        if not mask.any():
            # ensure at least one row (earliest)
            mask = end <= (s0 + 0.01 * span)
        return g.iloc[np.where(mask)[0]]

    mode = (mode or "").strip().lower()
    if mode in ("first_doc", "first", "doc1"):
        return (
            df.groupby("case_id", sort=False, group_keys=False)
            .apply(_first_doc)
            .reset_index(drop=True)
        )
    elif mode in ("p10", "p30", "p50"):
        frac = {"p10": 0.10, "p30": 0.30, "p50": 0.50}[mode]
        return (
            df.groupby("case_id", sort=False, group_keys=False)
            .apply(lambda g: _prefix_by_percent(g, frac))
            .reset_index(drop=True)
        )
    else:
        return df


def build_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: quotes with mlp_probability, mlp_pred_strict, mlp_pred_recallT, case_id, case_id_clean, and a sortable order."""
    need_cols = [
        "case_id",
        "case_id_clean",
        "mlp_probability",
        "mlp_pred_strict",
        "mlp_pred_recallT",
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    feats = []
    for cid, g in df.groupby("case_id", sort=False):
        g = _sort_case_naturally(g)
        p = g["mlp_probability"].astype(float).to_numpy()
        s = g["mlp_pred_strict"].astype(int).to_numpy()
        r = g["mlp_pred_recallT"].astype(int).to_numpy()
        n = len(g)
        # normalized position ∈ [0,1]
        if "global_token_start" in g.columns:
            pos_raw = pd.to_numeric(g["global_token_start"], errors="coerce").to_numpy(
                dtype=float
            )
        elif "global_char_start" in g.columns:
            pos_raw = pd.to_numeric(g["global_char_start"], errors="coerce").to_numpy(
                dtype=float
            )
        elif "docket_token_start" in g.columns:
            pos_raw = pd.to_numeric(g["docket_token_start"], errors="coerce").to_numpy(
                dtype=float
            )
        elif "docket_char_start" in g.columns:
            pos_raw = pd.to_numeric(g["docket_char_start"], errors="coerce").to_numpy(
                dtype=float
            )
        else:
            pos_raw = np.arange(n, dtype=float)
        if n > 0:
            pr = np.nan_to_num(pos_raw, nan=0.0, posinf=0.0, neginf=0.0)
            span = pr.max() - pr.min()
            pos = (pr - pr.min()) / (span if span > 0 else 1.0)
        else:
            pos = np.array([])
        # token weights (optional)
        if "num_tokens" in g.columns:
            w = (
                pd.to_numeric(g["num_tokens"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            w = np.clip(w, 1.0, np.inf)
        else:
            w = np.ones(n, dtype=float)
        # Extract court from case_id_clean (e.g., "0:21-md-03015_flsd" -> "flsd")
        case_id_clean = str(g["case_id_clean"].iloc[0]) if len(g) > 0 else ""
        court = case_id_clean.split("_")[-1] if "_" in case_id_clean else "unknown"

        # density
        prop_strict = float(s.mean()) if n else 0.0
        prop_recallT = float(r.mean()) if n else 0.0
        prop_p80 = float((p >= 0.80).mean()) if n else 0.0
        prop_p90 = float((p >= 0.90).mean()) if n else 0.0
        prop_p95 = float((p >= 0.95).mean()) if n else 0.0
        mean_p = float(p.mean()) if n else 0.0
        std_p = float(p.std()) if n else 0.0
        max_p = float(p.max()) if n else 0.0
        top3_mean = float(np.mean(np.sort(p)[-3:])) if n >= 3 else mean_p

        # positional (on strict)
        if n:
            q1, q2 = np.quantile(pos, [0.30, 0.70])
            early_mask = pos <= q1
            mid_mask = (pos > q1) & (pos <= q2)
            late_mask = pos > q2
            early_prop = float(s[early_mask].mean()) if early_mask.any() else 0.0
            mid_prop = float(s[mid_mask].mean()) if mid_mask.any() else 0.0
            late_prop = float(s[late_mask].mean()) if late_mask.any() else 0.0
        else:
            early_prop = mid_prop = late_prop = 0.0
        pos_com = float(((p * pos) * w).sum() / max((p * w).sum(), 1e-9)) if n else 0.0

        # clustering on strict
        n_clusters, max_len, avg_len = _clusters_from_binary(s)
        first_pos = float(pos[s.argmax()]) if n and s.any() else 1.0
        last_pos = float(pos[n - 1 - np.argmax(s[::-1])]) if n and s.any() else 0.0

        # quantiles of p
        if n:
            p_q25, p_q50, p_q75, p_q90 = (
                float(np.quantile(p, q)) for q in (0.25, 0.50, 0.75, 0.90)
            )
        else:
            p_q25 = p_q50 = p_q75 = p_q90 = 0.0

        feats.append(
            {
                "case_id": cid,
                "court_id": court,
                "n_quotes": n,
                "prop_strict": prop_strict,
                "prop_recallT": prop_recallT,
                "prop_p80": prop_p80,
                "prop_p90": prop_p90,
                "prop_p95": prop_p95,
                "mean_p": mean_p,
                "std_p": std_p,
                "max_p": max_p,
                "top3_mean_p": top3_mean,
                "early_prop_strict": early_prop,
                "mid_prop_strict": mid_prop,
                "late_prop_strict": late_prop,
                "pos_center_of_mass": pos_com,
                "n_clusters": n_clusters,
                "max_cluster_len": max_len,
                "avg_cluster_len": avg_len,
                "first_pos": first_pos,
                "last_pos": last_pos,
                "p_q25": p_q25,
                "p_q50": p_q50,
                "p_q75": p_q75,
                "p_q90": p_q90,
            }
        )
    return pd.DataFrame(feats)


# ---------------- identity suppression on case features -----------
def _letters_only(x: str) -> str:
    return "".join([c for c in str(x).lower() if c.isalpha()])


def fit_suppression_means(X: np.ndarray, courts: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-court and per-circuit means + global in SCALED feature space."""
    means = {"court": {}, "circuit": {}, "global": X.mean(axis=0)}
    # court means (min size 10)
    court_vals = pd.Series(courts).fillna("").values
    for c in np.unique(court_vals):
        if c == "":
            continue
        idx = court_vals == c
        if int(idx.sum()) >= 10:
            means["court"][c] = X[idx].mean(axis=0)
    # circuit means (min size 10)
    circ_sum, circ_cnt = {}, {}
    for i, c in enumerate(court_vals):
        if c == "":
            continue
        circ = _letters_only(c)
        if not circ:
            continue
        if circ not in circ_sum:
            circ_sum[circ] = X[i].copy()
            circ_cnt[circ] = 1
        else:
            circ_sum[circ] += X[i]
            circ_cnt[circ] += 1
    for circ, cnt in circ_cnt.items():
        if cnt >= 10:
            means["circuit"][circ] = circ_sum[circ] / cnt
    return means


def apply_suppression(
    X: np.ndarray, courts: np.ndarray, means: Dict[str, np.ndarray]
) -> np.ndarray:
    courts = pd.Series(courts).astype(str).fillna("unknown").values
    mu = []
    fallback = {"court": 0, "circuit": 0, "global": 0}
    for c in courts:
        if c in means["court"]:
            mu.append(means["court"][c])
            fallback["court"] += 1
        else:
            circ = _letters_only(c)
            if circ in means["circuit"]:
                mu.append(means["circuit"][circ])
                fallback["circuit"] += 1
            else:
                mu.append(means["global"])
                fallback["global"] += 1
    Xs = X - np.stack(mu, axis=0)
    tot = len(courts)
    if tot:
        log.info(
            f"Suppression fallback: Court={fallback['court']/tot:.1%} | Circuit={fallback['circuit']/tot:.1%} | Global={fallback['global']/tot:.1%}"
        )
    return Xs


# ----------------------------- metrics ---------------------------
def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.any():
            acc = y_true[m].mean()
            conf = y_prob[m].mean()
            ece += abs(conf - acc) * (m.mean())
    return float(ece)


def mcc_opt_threshold(y, p, steps=201) -> Tuple[float, float]:
    ths = np.linspace(0, 1, steps)
    best = (-1.0, 0.5)
    for t in ths:
        yhat = (p >= t).astype(int)
        if len(np.unique(yhat)) < 2:
            continue
        m = matthews_corrcoef(y, yhat)
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]


def threshold_for_recall(y, p, target=0.20, steps=401) -> float:
    ths = np.linspace(0, 1, steps)
    best_t, best_gap = 0.5, 1e9
    for t in ths:
        yhat = (p >= t).astype(int)
        rec = recall_score(y, yhat, zero_division=0.0)
        gap = abs(rec - target)
        if rec >= target and gap < best_gap:
            best_gap, best_t = gap, float(t)
    if best_gap == 1e9:
        for t in ths:
            yhat = (p >= t).astype(int)
            gap = abs(recall_score(y, yhat, zero_division=0.0) - target)
            if gap < best_gap:
                best_gap, best_t = gap, float(t)
    return best_t


def op_metrics(y, p, t) -> Dict[str, Any]:
    yhat = (p >= t).astype(int)
    prec = precision_score(y, yhat, zero_division=0.0)
    rec = recall_score(y, yhat, zero_division=0.0)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "mcc": float(matthews_corrcoef(y, yhat)) if len(np.unique(yhat)) > 1 else 0.0,
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        },
        "threshold": float(t),
    }


# ----------------------------- main ------------------------------
def main(
    mirror_root: str,
    feature_config: str,
    output_dir: str,
    target_recall: float = 0.20,
    topk_percent: Optional[float] = None,
    fold: int = 4,
    early: Optional[str] = None,
    refit_on_train_plus_dev: bool = False,
    try_elasticnet: bool = False,
    bagging_n: int = 0,
):
    mirror = Path(mirror_root)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load fold_X train/dev and oof_test/test
    fold_dir = f"fold_{int(fold)}"
    train_dir = mirror / feature_config / fold_dir
    dev_dir = mirror / feature_config / fold_dir
    test_dir = mirror / feature_config / "oof_test"
    log.info(
        f"Loading: train={train_dir}/train.jsonl | dev={dev_dir}/dev.jsonl | test={test_dir}/test.jsonl"
    )

    tr = load_jsonl_dir(train_dir)
    dv = load_jsonl_dir(dev_dir)
    ts = load_jsonl_dir(test_dir)
    for name, df in [("train", tr), ("dev", dv), ("test", ts)]:
        log.info(f"{name}: {len(df)} rows")

    # sanity
    need_cols = [
        "case_id",
        "case_id_clean",
        "outcome_bin",
        "mlp_probability",
        "mlp_pred_strict",
        "mlp_pred_recallT",
    ]
    for c in need_cols:
        for name, df in [("train", tr), ("dev", dv), ("test", ts)]:
            if c not in df.columns:
                raise ValueError(f"{name} missing {c}")

    # 2) Aggregate to case features
    trF = build_case_features(tr)
    dvF = build_case_features(dv)
    tsF = build_case_features(ts)

    # Attach case outcomes (majority vote or given per-quote col? here we take any row label)
    def attach_y(orig: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
        y_map = (
            orig.groupby("case_id")["outcome_bin"]
            .agg(lambda x: int(round(x.mean())))
            .to_dict()
        )
        cf["y"] = cf["case_id"].map(y_map).astype(int)
        return cf

    trF = attach_y(tr, trF)
    dvF = attach_y(dv, dvF)
    tsF = attach_y(ts, tsF)

    # 3) Scale → fit suppression means on TRAIN (case-space)
    feat_cols = [c for c in trF.columns if c not in ("case_id", "court_id", "y")]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(trF[feat_cols].to_numpy(dtype=np.float32))
    Xdv = scaler.transform(dvF[feat_cols].to_numpy(dtype=np.float32))
    Xts = scaler.transform(tsF[feat_cols].to_numpy(dtype=np.float32))

    means = fit_suppression_means(Xtr, trF["court_id"].to_numpy())
    Xdv_sup = apply_suppression(Xdv, dvF["court_id"].to_numpy(), means)
    Xts_sup = apply_suppression(Xts, tsF["court_id"].to_numpy(), means)

    ytr, ydv, yts = trF["y"].to_numpy(), dvF["y"].to_numpy(), tsF["y"].to_numpy()

    # 4) Train LR (tiny grid), select by dev-suppressed MCC
    gridC = [0.01, 0.1, 1.0]
    best = {
        "mcc": -1,
        "C": None,
        "penalty": None,
        "l1_ratio": None,
        "solver": None,
        "clf": None,
        "cal": None,
        "cal_name": None,
    }

    def metrics(y, p) -> Dict[str, float]:
        return {
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5,
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "ece": ece_score(y, p),
        }

    # Try L2 LR grid as before
    for C in gridC:
        base = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=2000,
            n_jobs=None,
        )
        base.fit(Xtr, ytr)
        # two calibrators trained ONLY on train
        cal_sig = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal_iso = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal_sig.fit(Xtr, ytr)
        cal_iso.fit(Xtr, ytr)

        # evaluate on DEV (suppressed) to choose calibrator & C
        ps_sig = cal_sig.predict_proba(Xdv_sup)[:, 1]
        ps_iso = cal_iso.predict_proba(Xdv_sup)[:, 1]
        ece_sig = ece_score(ydv, ps_sig)
        ece_iso = ece_score(ydv, ps_iso)
        if ece_iso <= ece_sig:
            ps, cal_used, cal_name = ps_iso, cal_iso, "isotonic"
        else:
            ps, cal_used, cal_name = ps_sig, cal_sig, "sigmoid"

        thr_mcc, mcc_dev = mcc_opt_threshold(ydv, ps)
        if mcc_dev > best["mcc"]:
            best.update(
                {
                    "mcc": mcc_dev,
                    "C": C,
                    "penalty": "l2",
                    "l1_ratio": None,
                    "solver": "lbfgs",
                    "clf": base,
                    "cal": cal_used,
                    "cal_name": cal_name,
                    "thr_mcc": thr_mcc,
                }
            )

    # Optionally try Elastic-Net LR if requested
    if try_elasticnet:
        from sklearn.exceptions import ConvergenceWarning
        import warnings

        gridC_en = [0.1, 1.0]
        grid_l1 = [0.1, 0.5]
        for C in gridC_en:
            for l1r in grid_l1:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    base = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        class_weight="balanced",
                        C=C,
                        l1_ratio=l1r,
                        max_iter=2000,
                        n_jobs=None,
                    )
                    try:
                        base.fit(Xtr, ytr)
                    except Exception as e:
                        log.warning(
                            f"ElasticNet LR failed for C={C}, l1_ratio={l1r}: {e}"
                        )
                        continue
                cal_sig = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                cal_iso = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal_sig.fit(Xtr, ytr)
                cal_iso.fit(Xtr, ytr)
                ps_sig = cal_sig.predict_proba(Xdv_sup)[:, 1]
                ps_iso = cal_iso.predict_proba(Xdv_sup)[:, 1]
                ece_sig = ece_score(ydv, ps_sig)
                ece_iso = ece_score(ydv, ps_iso)
                if ece_iso <= ece_sig:
                    ps, cal_used, cal_name = ps_iso, cal_iso, "isotonic"
                else:
                    ps, cal_used, cal_name = ps_sig, cal_sig, "sigmoid"
                thr_mcc, mcc_dev = mcc_opt_threshold(ydv, ps)
                if mcc_dev > best["mcc"]:
                    best.update(
                        {
                            "mcc": mcc_dev,
                            "C": C,
                            "penalty": "elasticnet",
                            "l1_ratio": l1r,
                            "solver": "saga",
                            "clf": base,
                            "cal": cal_used,
                            "cal_name": cal_name,
                            "thr_mcc": thr_mcc,
                        }
                    )

    # 5) Refit best base on train (already fit), calibrator already fit; compute metrics
    log.info(
        f"Best model: penalty={best['penalty']}, solver={best['solver']}, C={best['C']}, l1_ratio={best.get('l1_ratio', None)} | calibrator={best['cal_name']} | dev_suppr_MCC={best['mcc']:.4f}"
    )

    # DEV RAW & SUPPRESSED metrics (using chosen cal)
    dev_raw_prob = best["cal"].predict_proba(Xdv)[:, 1]
    dev_sup_prob = best["cal"].predict_proba(Xdv_sup)[:, 1]
    dev_thr_mcc = best["thr_mcc"]
    dev_thr_recall = threshold_for_recall(ydv, dev_sup_prob, target=target_recall)
    dev_thr_topk = None
    if topk_percent is not None:
        q = 1.0 - float(topk_percent)
        dev_thr_topk = float(np.quantile(dev_sup_prob, q))

    dev = {
        "raw": {
            **metrics(ydv, dev_raw_prob),
            "thr_mcc": dev_thr_mcc,
            "ops_mcc": op_metrics(ydv, dev_sup_prob, dev_thr_mcc),
        },
        "supp": {
            **metrics(ydv, dev_sup_prob),
            "thr_mcc": dev_thr_mcc,
            "ops_mcc": op_metrics(ydv, dev_sup_prob, dev_thr_mcc),
        },
    }
    dev["supp"]["thr_recallT"] = dev_thr_recall
    dev["supp"]["ops_recallT"] = op_metrics(ydv, dev_sup_prob, dev_thr_recall)
    if dev_thr_topk is not None:
        dev["supp"]["thr_topk"] = dev_thr_topk
        dev["supp"]["ops_topk"] = op_metrics(ydv, dev_sup_prob, dev_thr_topk)

    # TEST RAW & SUPPRESSED (fixed thresholds from dev-suppressed)
    ts_raw_prob = best["cal"].predict_proba(Xts)[:, 1]
    ts_sup_prob = best["cal"].predict_proba(Xts_sup)[:, 1]
    thr_mcc = dev_thr_mcc
    thr_recall = dev_thr_recall
    thr_topk = dev_thr_topk

    # --------- Optional refit on train+dev ---------
    if refit_on_train_plus_dev:
        # Refit scaler on TRAIN+DEV case features
        F = pd.concat([trF, dvF], ignore_index=True)
        ytrdv = np.concatenate([ytr, ydv])
        courts_trdv = F["court_id"].to_numpy()

        scaler = StandardScaler()
        Xtrdv = scaler.fit_transform(F[feat_cols].to_numpy(dtype=np.float32))

        # Recompute suppression means on TRAIN+DEV (in scaled space)
        means = fit_suppression_means(Xtrdv, courts_trdv)

        # Refit base LR with same hyperparameters on TRAIN+DEV
        lr_kwargs = {
            "penalty": best["penalty"],
            "solver": best["solver"],
            "class_weight": "balanced",
            "C": best["C"],
            "max_iter": 2000,
            "n_jobs": None,
        }
        if best["penalty"] == "elasticnet" and best.get("l1_ratio") is not None:
            lr_kwargs["l1_ratio"] = best["l1_ratio"]
        base = LogisticRegression(**lr_kwargs)
        base.fit(Xtrdv, ytrdv)

        # Refit calibrator (same method) on TRAIN+DEV
        cal = CalibratedClassifierCV(base, method=best["cal_name"], cv=3)
        cal.fit(Xtrdv, ytrdv)

        # Update best references
        best["clf"] = base
        best["cal"] = cal

        # Recompute dev/test features under the NEW scaler and NEW means
        Xdv = scaler.transform(dvF[feat_cols].to_numpy(dtype=np.float32))
        Xts = scaler.transform(tsF[feat_cols].to_numpy(dtype=np.float32))
        dev_raw_prob = cal.predict_proba(Xdv)[:, 1]
        Xdv_sup = apply_suppression(Xdv, dvF["court_id"].to_numpy(), means)
        dev_sup_prob = cal.predict_proba(Xdv_sup)[:, 1]
        ts_raw_prob = cal.predict_proba(Xts)[:, 1]
        Xts_sup = apply_suppression(Xts, tsF["court_id"].to_numpy(), means)
        ts_sup_prob = cal.predict_proba(Xts_sup)[:, 1]

        # Re-derive thresholds on DEV (suppressed)
        dev_thr_mcc, _ = mcc_opt_threshold(ydv, dev_sup_prob)
        dev_thr_recall = threshold_for_recall(ydv, dev_sup_prob, target=target_recall)
        dev_thr_topk = None
        if topk_percent is not None:
            q = 1.0 - float(topk_percent)
            dev_thr_topk = float(np.quantile(dev_sup_prob, q))

        thr_mcc = dev_thr_mcc
        thr_recall = dev_thr_recall
        thr_topk = dev_thr_topk

        log.info(
            "Refit on train+dev enabled: re-fitted scaler, suppression means, LR, and calibrator; re-derived dev thresholds."
        )

    test = {
        "raw": {
            **metrics(yts, ts_raw_prob),
            "ops_mcc": op_metrics(yts, ts_sup_prob, thr_mcc),
        },
        "supp": {
            **metrics(yts, ts_sup_prob),
            "ops_mcc": op_metrics(yts, ts_sup_prob, thr_mcc),
        },
    }
    test["supp"]["ops_recallT"] = op_metrics(yts, ts_sup_prob, thr_recall)
    if thr_topk is not None:
        test["supp"]["ops_topk"] = op_metrics(yts, ts_sup_prob, thr_topk)

    # ----- Early-cutoff evaluation (optional) -----
    early_results = {}

    def _eval_early(tag: str, df_dev: pd.DataFrame, df_test: pd.DataFrame):
        # 1) rebuild case features on the early subset
        dvE = build_case_features(df_dev)
        tsE = build_case_features(df_test)
        # attach labels
        dvE = attach_y(dv, dvE)
        tsE = attach_y(ts, tsE)
        # scale with the SAME scaler; suppress with SAME means
        XdvE = scaler.transform(dvE[feat_cols].to_numpy(dtype=np.float32))
        XtsE = scaler.transform(tsE[feat_cols].to_numpy(dtype=np.float32))
        XdvE_sup = apply_suppression(XdvE, dvE["court_id"].to_numpy(), means)
        XtsE_sup = apply_suppression(XtsE, tsE["court_id"].to_numpy(), means)
        ydvE, ytsE = dvE["y"].to_numpy(), tsE["y"].to_numpy()
        # probabilities
        ps_dv_raw = best["cal"].predict_proba(XdvE)[:, 1]
        ps_dv_sup = best["cal"].predict_proba(XdvE_sup)[:, 1]
        ps_ts_raw = best["cal"].predict_proba(XtsE)[:, 1]
        ps_ts_sup = best["cal"].predict_proba(XtsE_sup)[:, 1]
        # metrics
        devE = {
            "raw": {**metrics(ydvE, ps_dv_raw)},
            "supp": {**metrics(ydvE, ps_dv_sup)},
        }
        testE = {
            "raw": {**metrics(ytsE, ps_ts_raw)},
            "supp": {**metrics(ytsE, ps_ts_sup)},
        }
        # operating points (use thresholds recomputed on early dev-suppressed)
        thr_mcc_loc, mcc_dev_loc = mcc_opt_threshold(ydvE, ps_dv_sup)
        thr_recall_loc = threshold_for_recall(ydvE, ps_dv_sup, target=target_recall)
        devE["supp"]["ops_mcc"] = op_metrics(ydvE, ps_dv_sup, thr_mcc_loc)
        devE["supp"]["ops_recallT"] = op_metrics(ydvE, ps_dv_sup, thr_recall_loc)
        testE["supp"]["ops_mcc"] = op_metrics(ytsE, ps_ts_sup, thr_mcc_loc)
        testE["supp"]["ops_recallT"] = op_metrics(ytsE, ps_ts_sup, thr_recall_loc)
        # record local thresholds inside the objects (handy for inspection)
        devE["supp"]["thr_mcc"] = float(thr_mcc_loc)
        devE["supp"]["thr_recallT"] = float(thr_recall_loc)
        testE["supp"]["thr_mcc"] = float(thr_mcc_loc)
        testE["supp"]["thr_recallT"] = float(thr_recall_loc)
        # positive rates
        devE["supp"]["pos_rate_mcc"] = float((ps_dv_sup >= thr_mcc_loc).mean())
        devE["supp"]["pos_rate_recallT"] = float((ps_dv_sup >= thr_recall_loc).mean())
        testE["supp"]["pos_rate_mcc"] = float((ps_ts_sup >= thr_mcc_loc).mean())
        testE["supp"]["pos_rate_recallT"] = float((ps_ts_sup >= thr_recall_loc).mean())
        # Store local thresholds
        early_results[tag] = {
            "dev": devE,
            "test": testE,
            "supp_thresholds": {"mcc": thr_mcc_loc, "recallT": thr_recall_loc},
        }

        # Console summary (compact)
        log.info(
            f"[EARLY {tag}] τ_MCC={thr_mcc_loc:.3f} τ_rec={thr_recall_loc:.3f} | DEV supp: MCC@τ_MCC={devE['supp']['ops_mcc']['mcc']:.3f} | MCC@τ_rec={devE['supp']['ops_recallT']['mcc']:.3f} | Pos% (τ_MCC/τ_rec) = {devE['supp']['pos_rate_mcc']:.1%}/{devE['supp']['pos_rate_recallT']:.1%}"
        )
        log.info(
            f"[EARLY {tag}] τ_MCC={thr_mcc_loc:.3f} τ_rec={thr_recall_loc:.3f} | TEST supp: MCC@τ_MCC={testE['supp']['ops_mcc']['mcc']:.3f} | MCC@τ_rec={testE['supp']['ops_recallT']['mcc']:.3f} | Pos% (τ_MCC/τ_rec) = {testE['supp']['pos_rate_mcc']:.1%}/{testE['supp']['pos_rate_recallT']:.1%}"
        )

    if early:
        modes = []
        if early.lower() == "all":
            modes = ["first_doc", "p10", "p30", "p50"]
        else:
            modes = [m.strip() for m in early.split(",") if m.strip()]
        for m in modes:
            # filter dev/test quotes to early subset per case
            dv_early = _filter_early(dv, m)
            ts_early = _filter_early(ts, m)
            _eval_early(m, dv_early, ts_early)

    # ----------- Optional bagging step -----------
    if bagging_n and bagging_n > 0:
        bag_n = int(bagging_n)
        # Use the same train set as the final model
        if refit_on_train_plus_dev:
            Xtrain = np.vstack([Xtr, Xdv])
            ytrain = np.concatenate([ytr, ydv])
            courts_train = np.concatenate(
                [trF["court_id"].to_numpy(), dvF["court_id"].to_numpy()]
            )
        else:
            Xtrain = Xtr
            ytrain = ytr
            courts_train = trF["court_id"].to_numpy()
        # Use the same scaler (fit on TRAIN) and suppression means (fit on train+dev if refit)
        dev_probs_raw = []
        dev_probs_sup = []
        ts_probs_raw = []
        ts_probs_sup = []
        # For early: keep per-replica dev/test early probabilities
        early_probs = {}  # tag -> {dev_raw:[], dev_sup:[], ts_raw:[], ts_sup:[]}
        for i in range(bag_n):
            idx = np.random.choice(len(Xtrain), len(Xtrain), replace=True)
            Xb = Xtrain[idx]
            yb = ytrain[idx]
            # Fit base LR
            lr_kwargs = {
                "penalty": best["penalty"],
                "solver": best["solver"],
                "class_weight": "balanced",
                "C": best["C"],
                "max_iter": 2000,
                "n_jobs": None,
            }
            if best["penalty"] == "elasticnet":
                lr_kwargs["l1_ratio"] = best["l1_ratio"]
            base = LogisticRegression(**lr_kwargs)
            base.fit(Xb, yb)
            # Fit calibrator
            cal = CalibratedClassifierCV(base, method=best["cal_name"], cv=3)
            cal.fit(Xb, yb)
            # Probabilities on dev/test
            dev_probs_raw.append(cal.predict_proba(Xdv)[:, 1])
            Xdv_sup_bag = apply_suppression(Xdv, dvF["court_id"].to_numpy(), means)
            dev_probs_sup.append(cal.predict_proba(Xdv_sup_bag)[:, 1])
            ts_probs_raw.append(cal.predict_proba(Xts)[:, 1])
            Xts_sup_bag = apply_suppression(Xts, tsF["court_id"].to_numpy(), means)
            ts_probs_sup.append(cal.predict_proba(Xts_sup_bag)[:, 1])
            # For early
            for tag in early_results.keys():
                # Recompute early features for dev/test
                # Use the same early subset as before
                dv_early = _filter_early(dv, tag)
                ts_early = _filter_early(ts, tag)
                dvE = build_case_features(dv_early)
                tsE = build_case_features(ts_early)
                dvE = attach_y(dv, dvE)
                tsE = attach_y(ts, tsE)
                XdvE = scaler.transform(dvE[feat_cols].to_numpy(dtype=np.float32))
                XtsE = scaler.transform(tsE[feat_cols].to_numpy(dtype=np.float32))
                XdvE_sup = apply_suppression(XdvE, dvE["court_id"].to_numpy(), means)
                XtsE_sup = apply_suppression(XtsE, tsE["court_id"].to_numpy(), means)
                if tag not in early_probs:
                    early_probs[tag] = {
                        "dev_raw": [],
                        "dev_sup": [],
                        "ts_raw": [],
                        "ts_sup": [],
                        "ydvE": dvE["y"].to_numpy(),
                        "ytsE": tsE["y"].to_numpy(),
                    }
                early_probs[tag]["dev_raw"].append(cal.predict_proba(XdvE)[:, 1])
                early_probs[tag]["dev_sup"].append(cal.predict_proba(XdvE_sup)[:, 1])
                early_probs[tag]["ts_raw"].append(cal.predict_proba(XtsE)[:, 1])
                early_probs[tag]["ts_sup"].append(cal.predict_proba(XtsE_sup)[:, 1])
        # Average probabilities
        dev_raw_prob = np.mean(dev_probs_raw, axis=0)
        dev_sup_prob = np.mean(dev_probs_sup, axis=0)
        ts_raw_prob = np.mean(ts_probs_raw, axis=0)
        ts_sup_prob = np.mean(ts_probs_sup, axis=0)
        # Recompute dev thresholds on averaged dev-supp
        dev_thr_mcc, _ = mcc_opt_threshold(ydv, dev_sup_prob)
        dev_thr_recall = threshold_for_recall(ydv, dev_sup_prob, target=target_recall)
        dev_thr_topk = None
        if topk_percent is not None:
            q = 1.0 - float(topk_percent)
            dev_thr_topk = float(np.quantile(dev_sup_prob, q))
        thr_mcc = dev_thr_mcc
        thr_recall = dev_thr_recall
        thr_topk = dev_thr_topk
        log.info(
            f"Bagging enabled (n={bag_n}): thresholds re-derived on averaged dev-suppressed scores."
        )
        log.info(f"Bagging: averaged probabilities from {bag_n} replicas")
        # For early: recompute local thresholds and update early_results
        for tag in early_results.keys():
            ydvE = early_probs[tag]["ydvE"]
            ytsE = early_probs[tag]["ytsE"]
            ps_dv_raw = np.mean(early_probs[tag]["dev_raw"], axis=0)
            ps_dv_sup = np.mean(early_probs[tag]["dev_sup"], axis=0)
            ps_ts_raw = np.mean(early_probs[tag]["ts_raw"], axis=0)
            ps_ts_sup = np.mean(early_probs[tag]["ts_sup"], axis=0)
            # metrics
            devE = {
                "raw": {**metrics(ydvE, ps_dv_raw)},
                "supp": {**metrics(ydvE, ps_dv_sup)},
            }
            testE = {
                "raw": {**metrics(ytsE, ps_ts_raw)},
                "supp": {**metrics(ytsE, ps_ts_sup)},
            }
            # local thresholds
            thr_mcc_loc, mcc_dev_loc = mcc_opt_threshold(ydvE, ps_dv_sup)
            thr_recall_loc = threshold_for_recall(ydvE, ps_dv_sup, target=target_recall)
            devE["supp"]["ops_mcc"] = op_metrics(ydvE, ps_dv_sup, thr_mcc_loc)
            devE["supp"]["ops_recallT"] = op_metrics(ydvE, ps_dv_sup, thr_recall_loc)
            testE["supp"]["ops_mcc"] = op_metrics(ytsE, ps_ts_sup, thr_mcc_loc)
            testE["supp"]["ops_recallT"] = op_metrics(ytsE, ps_ts_sup, thr_recall_loc)
            devE["supp"]["pos_rate_mcc"] = float((ps_dv_sup >= thr_mcc_loc).mean())
            devE["supp"]["pos_rate_recallT"] = float(
                (ps_dv_sup >= thr_recall_loc).mean()
            )
            testE["supp"]["pos_rate_mcc"] = float((ps_ts_sup >= thr_mcc_loc).mean())
            testE["supp"]["pos_rate_recallT"] = float(
                (ps_ts_sup >= thr_recall_loc).mean()
            )
            early_results[tag] = {
                "dev": devE,
                "test": testE,
                "supp_thresholds": {"mcc": thr_mcc_loc, "recallT": thr_recall_loc},
            }
            # Console summary
            log.info(
                f"[EARLY {tag}] τ_MCC={thr_mcc_loc:.3f} τ_rec={thr_recall_loc:.3f} | DEV supp: MCC@τ_MCC={devE['supp']['ops_mcc']['mcc']:.3f} | MCC@τ_rec={devE['supp']['ops_recallT']['mcc']:.3f} | Pos% (τ_MCC/τ_rec) = {devE['supp']['pos_rate_mcc']:.1%}/{devE['supp']['pos_rate_recallT']:.1%}"
            )
            log.info(
                f"[EARLY {tag}] τ_MCC={thr_mcc_loc:.3f} τ_rec={thr_recall_loc:.3f} | TEST supp: MCC@τ_MCC={testE['supp']['ops_mcc']['mcc']:.3f} | MCC@τ_rec={testE['supp']['ops_recallT']['mcc']:.3f} | Pos% (τ_MCC/τ_rec) = {testE['supp']['pos_rate_mcc']:.1%}/{testE['supp']['pos_rate_recallT']:.1%}"
            )

    # 6) Save predictions (test set case-level)
    pred_df = pd.DataFrame(
        {
            "case_id": tsF["case_id"],
            "court_id": tsF["court_id"],
            "y_true": tsF["y"],
            "prob_cal": ts_sup_prob,  # suppressed-probabilities are for ranking under the robust view
            "pred_mcc": (ts_sup_prob >= thr_mcc).astype(int),
            "pred_recallT": (ts_sup_prob >= thr_recall).astype(int),
        }
    )
    if thr_topk is not None:
        pred_df["pred_topk"] = (ts_sup_prob >= thr_topk).astype(int)
    pred_out = outdir / "case_predictions.csv"
    pred_df.to_csv(pred_out, index=False)

    # 7) Save model card
    # pull LR coefficients from best["clf"]
    coef = best["clf"].coef_.ravel().tolist() if hasattr(best["clf"], "coef_") else None
    model_card = {
        "feature_config": feature_config,
        "n_case_features": len(feat_cols),
        "feature_names": feat_cols,
        "best_C": best["C"],
        "calibrator": best["cal_name"],
        "thresholds": {
            "mcc": float(thr_mcc),
            "recallT": float(thr_recall),
            "topk": float(thr_topk) if thr_topk is not None else None,
        },
        "dev_metrics": dev,
        "test_metrics": test,
        "early_evaluation": early_results,
        "coefficients": coef,
        "selected_model": {
            "penalty": best["penalty"],
            "solver": best["solver"],
            "C": best["C"],
            "l1_ratio": best.get("l1_ratio", None),
        },
    }
    if refit_on_train_plus_dev:
        model_card["refit_on_train_plus_dev"] = True
    if bagging_n and bagging_n > 0:
        model_card["bagging_n"] = bagging_n
    with open(outdir / "model_card.json", "wb") as f:
        f.write(_dumps(model_card))

    log.info(f"Saved: {pred_out}")
    log.info(f"Saved: {outdir/'model_card.json'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Case-level LR on hero MLP mirror (fold_X/dev + oof_test/test)"
    )
    ap.add_argument("--mirror-root", required=True, help=".../mirror_with_predictions")
    ap.add_argument("--feature-config", default="E+3", choices=["E", "E+3", "E_3"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-recall", type=float, default=0.20)
    ap.add_argument("--topk-percent", type=float, default=None)
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument(
        "--early",
        type=str,
        default=None,
        help="Early cutoff to evaluate: one of {first_doc,p10,p30,p50,all}. If 'all', evaluates all four.",
    )
    ap.add_argument(
        "--refit-on-train-plus-dev",
        action="store_true",
        help="Refit final model and calibrator on train+dev, recompute suppression means.",
    )
    ap.add_argument(
        "--try-elasticnet",
        action="store_true",
        help="Try ElasticNet LR grid in addition to standard L2 LR.",
    )
    ap.add_argument(
        "--bagging-n",
        type=int,
        default=0,
        help="Number of bagging replicas to train and average. If 0, no bagging.",
    )
    args = ap.parse_args()
    main(
        args.mirror_root,
        args.feature_config,
        args.output_dir,
        args.target_recall,
        args.topk_percent,
        args.fold,
        args.early,
        args.refit_on_train_plus_dev,
        args.try_elasticnet,
        args.bagging_n,
    )
