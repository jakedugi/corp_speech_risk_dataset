# ----------------------------------
# coral_ordinal/evaluate.py
# ----------------------------------
from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from .metrics import compute_metrics
from .utils import choose_device, load_checkpoint


def evaluate(model_path: str | Path, data_loader, cfg):
    device = choose_device(cfg.device)
    model, _ = load_checkpoint(model_path, device)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            pred = model.predict(x, threshold=cfg.prob_threshold)
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return compute_metrics(y_true, y_pred), (y_true, y_pred)
