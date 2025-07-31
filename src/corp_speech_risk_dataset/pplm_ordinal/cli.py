# =============================
# pplm_ordinal/cli.py
# =============================
import argparse
from pathlib import Path
import json
import numpy as np

from .config import PPLMConfig
from .classifier_api import load_classifier
from .generation import pplm_generate
from .metrics import compute_metrics, go_no_go_gate
from .viz import plot_confusion_matrix


def main():
    p = argparse.ArgumentParser(description="PPLM Ordinal Steering CLI")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--class-id", type=int, required=True, help="target ordinal bucket index"
    )
    p.add_argument("--config", default=None, help="Optional path to JSON cfg")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--tokenizer-name", default=None)
    p.add_argument("--classifier-path", default=None)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--length", type=int, default=60)
    p.add_argument("--step-size", type=float, default=0.04)
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--gm-scale", type=float, default=0.95)
    p.add_argument("--kl-scale", type=float, default=0.01)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.config:
        cfg = PPLMConfig.load(args.config)
    else:
        cfg = PPLMConfig(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            classifier_path=args.classifier_path,
            num_classes=args.num_classes,
            class_id=args.class_id,
            length=args.length,
            step_size=args.step_size,
            num_steps=args.num_steps,
            gm_scale=args.gm_scale,
            kl_scale=args.kl_scale,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )

    classifier = (
        load_classifier(cfg.classifier_path, device=None)
        if cfg.classifier_path
        else None
    )
    text = pplm_generate(args.prompt, cfg, classifier)
    print(text)


if __name__ == "__main__":
    main()
