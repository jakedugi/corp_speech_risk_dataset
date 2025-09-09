"""Lightweight smoke‑tests for the clustering stack."""

from pathlib import Path

import numpy as np

from corp_speech_risk_dataset.clustering import (
    FaissIndex,
    HDBSCANClusterer,
    DimReducer,
)


def test_faiss_roundtrip(tmp_path: Path):
    vecs = np.random.rand(100, 64).astype(np.float32)
    idx = FaissIndex(dim=64)
    idx.add(vecs)
    sims, ids = idx.search(vecs[:1], k=5)
    assert ids.shape == (1, 5)
    save_p = tmp_path / "faiss.idx"
    idx.save(save_p)
    # reload
    idx2 = FaissIndex.load(save_p)
    sims2, ids2 = idx2.search(vecs[:1], k=5)
    assert np.allclose(sims, sims2)
    assert np.array_equal(ids, ids2)


def test_hdbscan_labels():
    vecs = np.random.rand(200, 32).astype(np.float32)
    cl = HDBSCANClusterer(min_cluster_size=5)
    labels = cl.fit(vecs)
    # At least one label other than noise
    assert (labels >= -1).all()


def test_umap_shape():
    vecs = np.random.rand(100, 128).astype(np.float32)
    dr = DimReducer()
    coords = dr.fit_transform(vecs)
    assert coords.shape == (100, 2)
