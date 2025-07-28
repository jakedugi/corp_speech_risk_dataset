# ----------------------------------
# coral_ordinal/__init__.py
# ----------------------------------
__all__ = [
    "Config",
    "load_config",
    "make_buckets",
    "bucketize",
    "CORALMLP",
    "coral_loss",
    "compute_metrics",
    "ExactMatch",
    "OffByOne",
    "SpearmanR",
    "train",
    "evaluate",
    "plot_confusion_matrix",
]

from .config import Config, load_config
from .buckets import make_buckets, bucketize
from .model import CORALMLP
from .losses import coral_loss
from .metrics import compute_metrics, ExactMatch, OffByOne, SpearmanR
from .train import train
from .evaluate import evaluate
from .viz import plot_confusion_matrix
