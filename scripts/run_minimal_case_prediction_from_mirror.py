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
    best = {"mcc": -1, "C": None, "clf": None, "cal": None, "cal_name": None}

    def metrics(y, p) -> Dict[str, float]:
        return {
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5,
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "ece": ece_score(y, p),
        }

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
                    "clf": base,
                    "cal": cal_used,
                    "cal_name": cal_name,
                    "thr_mcc": thr_mcc,
                }
            )

    # 5) Refit best base on train (already fit), calibrator already fit; compute metrics
    log.info(
        f"Best C={best['C']} | calibrator={best['cal_name']} | dev_suppr_MCC={best['mcc']:.4f}"
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
        "coefficients": coef,
    }
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
    ap.add_argument("--feature-config", default="E+3", choices=["E", "E+3"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-recall", type=float, default=0.20)
    ap.add_argument("--topk-percent", type=float, default=None)
    ap.add_argument("--fold", type=int, default=4)
    args = ap.parse_args()
    main(
        args.mirror_root,
        args.feature_config,
        args.output_dir,
        args.target_recall,
        args.topk_percent,
        args.fold,
    )
