# =============================
# pplm_ordinal/config.py
# =============================
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class PPLMConfig:
    # LM
    model_name: str = "gpt2"
    tokenizer_name: str | None = None
    device: str | None = None  # "cpu"|"cuda"|"mps" or auto

    # Steering
    class_id: int = 0  # target ordinal bucket index
    step_size: float = 0.04  # gradient ascent step on past
    num_steps: int = 3  # inner optimization steps per token
    grad_norm_threshold: float = 1.0  # clip grad norm
    kl_scale: float = 0.01  # KL to keep close to base LM distribution
    length: int = 60  # tokens to generate
    gm_scale: float = 0.95  # interpolation between perturbed and unperturbed probs
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9

    # Classifier
    classifier_path: str | None = None  # torch .pt or .pth with state_dict & meta
    input_rep: str = (
        "last_hidden_mean"  # how to form classifier input ("last_hidden", "mean_hidden")
    )
    num_classes: int = 3

    # Ordinal metrics / gate
    prob_threshold: float = 0.5

    # Misc
    seed: int = 42

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "PPLMConfig":
        return PPLMConfig(**json.loads(Path(path).read_text()))
