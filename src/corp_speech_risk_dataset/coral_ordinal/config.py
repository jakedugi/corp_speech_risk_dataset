# ----------------------------------
# coral_ordinal/config.py
# ----------------------------------
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    data_path: str
    feature_key: str = "fused_emb"  # name of the input vector field
    # Optional multi-feature support: when provided, concatenate these keys.
    feature_keys: list | None = None
    label_key: str = (
        "bucket"  # name of categorical ordinal label (e.g. High/Medium/Low)
    )
    buckets: list | None = (
        None  # explicit ordered names, e.g. ["Low", "Medium", "High"]
    )
    hidden_dims: tuple = (512, 128)
    dropout: float = 0.1
    lr: float = 1e-4  # Lower default learning rate
    weight_decay: float = 1e-3  # Higher regularization
    batch_size: int = 128
    num_epochs: int = 20
    val_split: float = 0.2
    seed: int = 42
    device: str | None = None  # "cpu", "cuda", "mps" or None = auto
    output_dir: str = "runs/coral"

    # Training improvements
    prob_threshold: float = 0.5
    patience: int = 10  # Early stopping patience
    warmup_epochs: int = 5  # LR warmup epochs
    label_smoothing: float = 0.03  # Label smoothing epsilon
    use_imbalance_weights: bool = True  # Use threshold reweighting
    use_onecycle_lr: bool = True  # Use OneCycle LR instead of cosine
    feature_noise: float = 0.01  # Input noise for regularization
    tune_threshold: bool = True  # Auto-tune threshold on validation

    # Add these fields to the Config dataclass
    model_type: str = "coral"  # "coral" or "hybrid"
    lambda_cls: float = 0.7  # Classification loss weight
    lambda_reg: float = 0.3  # Regression loss weight

    # Whether to include flattened scalar priors from raw_features alongside vectors
    include_scalars: bool = False

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2))


def load_config(path: str | Path) -> Config:
    data = json.loads(Path(path).read_text())
    return Config(**data)
