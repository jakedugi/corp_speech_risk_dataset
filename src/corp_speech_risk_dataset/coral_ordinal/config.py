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
    label_key: str = (
        "bucket"  # name of categorical ordinal label (e.g. High/Medium/Low)
    )
    buckets: list | None = (
        None  # explicit ordered names, e.g. ["Low", "Medium", "High"]
    )
    hidden_dims: tuple = (512, 128)
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 20
    val_split: float = 0.2
    seed: int = 42
    device: str | None = None  # "cpu", "cuda", "mps" or None = auto
    output_dir: str = "runs/coral"

    # threshold for decision
    prob_threshold: float = 0.5

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2))


def load_config(path: str | Path) -> Config:
    data = json.loads(Path(path).read_text())
    return Config(**data)
