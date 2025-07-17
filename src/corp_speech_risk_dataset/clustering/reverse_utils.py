# NEW FILE: src/corp_speech_risk_dataset/models/clustering/reverse_utils.py
from pathlib import Path
import json
import numpy as np

DIM_WL = 2048  # change if your WL size differs


def load_meta(meta_path: str | Path):
    return json.loads(Path(meta_path).read_text())


def wl_vector(indices, counts, dim=DIM_WL):
    """Rebuild the dense WL vector exactly as used in concat_vectors.npy."""
    v = np.zeros(dim, dtype=np.float32)
    for i, c in zip(indices, counts):
        v[i] = c
    return v


def entry_from_idx(idx: int, meta):
    """Recover the original JSON entry – text, deps, everything – from its position."""
    return meta[idx]
