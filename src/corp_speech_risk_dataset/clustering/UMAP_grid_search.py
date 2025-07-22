#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
from numpy import nanpercentile
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap


def parse_args():
    p = argparse.ArgumentParser(
        description="Final UMAP grid sweep: n_neighbors & spread"
    )
    p.add_argument("--vec", required=True, help="Path to vectors.npy")
    p.add_argument("--meta", required=True, help="Path to metadata.json")
    return p.parse_args()


def load_data(vec_path, meta_path):
    meta = json.loads(Path(meta_path).read_text())
    raw = np.load(vec_path).astype(np.float32)
    X = normalize(raw, norm="l2", axis=1)
    y_raw = np.array([m.get("final_judgement_real") for m in meta], dtype=object)
    is_nan = [v is None or (isinstance(v, float) and np.isnan(v)) for v in y_raw]
    vals = np.array([float(v) if not nan else np.nan for v, nan in zip(y_raw, is_nan)])
    q33, q66 = nanpercentile(vals[~np.isnan(vals)], [33.3, 66.6])
    buckets = np.where(
        is_nan, "missing", np.select([vals < q33, vals < q66], ["low", "med"], "high")
    )
    _, target = np.unique(buckets, return_inverse=True)
    return X, buckets, target


if __name__ == "__main__":
    args = parse_args()
    X, buckets, target = load_data(args.vec, args.meta)

    BASE = dict(
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        target_weight=0.9,
        target_metric="categorical",
        set_op_mix_ratio=0.5,
        local_connectivity=1,
    )
    n_neighbors_list = [10, 15, 20, 30, 50]
    spread_list = [0.5, 1.0, 1.5]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for nn in n_neighbors_list:
        for sp in spread_list:
            sils, aris = [], []
            for tr, te in skf.split(X, buckets):
                X_tr, X_te = X[tr], X[te]
                y_tr = target[tr]
                um = umap.UMAP(n_neighbors=nn, spread=sp, **BASE)
                emb_tr = um.fit_transform(X_tr, y_tr)
                emb_te = um.transform(X_te)
                km = KMeans(n_clusters=3, random_state=42)
                km.fit(emb_tr)
                labels_te = km.predict(emb_te)
                sils.append(silhouette_score(emb_te, labels_te))
                aris.append(adjusted_rand_score(buckets[te], labels_te))
            results.append(
                {
                    "n_neighbors": nn,
                    "spread": sp,
                    "silhouette": np.mean(sils),
                    "ARI": np.mean(aris),
                }
            )

    # Sort by ARI then silhouette
    results = sorted(results, key=lambda r: (r["ARI"], r["silhouette"]), reverse=True)
    print(f"{'n_nb':>8} {'spread':>7} {'sil':>6} {'ARI':>6}")
    for r in results:
        print(
            f"{r['n_neighbors']:>8} {r['spread']:7.2f} {r['silhouette']:6.3f} {r['ARI']:6.3f}"
        )
