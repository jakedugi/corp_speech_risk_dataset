# =============================
# pplm_ordinal/viz.py
# =============================
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, out_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    return fig


def plot_bucket_trajectory(bucket_ids, labels, out_path=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(bucket_ids, marker="o")
    ax.set_ylabel("Bucket")
    ax.set_xlabel("Token step")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    return fig
