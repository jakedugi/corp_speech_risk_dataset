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


def save_checkpoint(model, path: Path, cfg, input_dim=None):
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "cfg": cfg.__dict__,
    }
    # Add input_dim if provided, or try to infer from model
    if input_dim is not None:
        checkpoint_data["input_dim"] = input_dim
    elif hasattr(model, "backbone") and hasattr(model.backbone, "0"):
        # Try to infer from first layer of backbone
        first_layer = model.backbone[0]
        if hasattr(first_layer, "in_features"):
            checkpoint_data["input_dim"] = first_layer.in_features

    torch.save(checkpoint_data, path)


def load_checkpoint(path: str | Path, device):
    ckpt = torch.load(path, map_location=device)
    from .model import CORALMLP

    num_classes = ckpt["num_classes"]
    # Try to get input_dim from checkpoint
    in_dim = ckpt.get("input_dim") or ckpt["cfg"].get("input_dim")
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
    # Use model_state_dict if available, fallback to model_state for backward compatibility
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_dict_key])
    model.to(device)
    return model, ckpt["cfg"]
