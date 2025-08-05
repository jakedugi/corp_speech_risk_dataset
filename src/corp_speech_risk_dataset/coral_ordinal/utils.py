# ----------------------------------
# coral_ordinal/utils.py
# ----------------------------------
import torch
import numpy as np
import random
from pathlib import Path


def choose_device(requested: str | None):
    """Choose device with priority: GPU (CUDA) > Apple Silicon (MPS) > CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path: Path, cfg):
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": model.num_classes,
            "cfg": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path: str | Path, device):
    ckpt = torch.load(path, map_location=device)
    from .model import CORALMLP

    num_classes = ckpt["num_classes"]
    in_dim = None  # must be inferred at runtime; store or detect from model file path
    # We'll store in_dim in cfg when training, so:
    in_dim = ckpt["cfg"].get("input_dim", None)
    if in_dim is None:
        raise ValueError(
            "input_dim not stored in checkpoint. Please re-train with updated utils.save_checkpoint storing it."
        )
    model = CORALMLP(
        in_dim,
        num_classes,
        hidden_dims=tuple(ckpt["cfg"]["hidden_dims"]),
        dropout=ckpt["cfg"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, ckpt["cfg"]
