# ----------------------------------
# coral_ordinal/cli_coral.py
# ----------------------------------
import argparse
from pathlib import Path
import torch

from .config import Config
from .data import build_loaders
from .model import CORALMLP
from .train import train
from .evaluate import evaluate
from .viz import plot_confusion_matrix
from .utils import choose_device


def main():
    p = argparse.ArgumentParser(description="Train/Eval CORAL Ordinal MLP")
    p.add_argument("--data", required=True, help="path to jsonl dataset")
    p.add_argument("--feature-key", default="fused_emb")
    p.add_argument("--label-key", default="bucket")
    p.add_argument("--buckets", nargs="+", default=["Low", "Medium", "High"])
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 128])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/coral")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--plot-cm", action="store_true")
    args = p.parse_args()

    cfg = Config(
        data_path=args.data,
        feature_key=args.feature_key,
        label_key=args.label_key,
        buckets=args.buckets,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_split=args.val_split,
        seed=args.seed,
        device=args.device,
        output_dir=args.out,
        prob_threshold=args.threshold,
    )

    train_loader, val_loader, full_ds = build_loaders(cfg)
    in_dim = len(full_ds[0][0])

    if args.eval_only:
        assert args.model_path, "--model-path required for eval-only"
        metrics, (yt, yp) = evaluate(args.model_path, val_loader, cfg)
        print(metrics)
        if args.plot_cm:
            plot_confusion_matrix(yt, yp, cfg.buckets, Path(cfg.output_dir) / "cm.png")
        return

    model = CORALMLP(
        in_dim,
        num_classes=len(cfg.buckets),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    # store input_dim in cfg for later load
    cfg.__dict__["input_dim"] = in_dim
    train(model, train_loader, val_loader, cfg)

    # Final evaluation on validation split
    metrics, (yt, yp) = evaluate(Path(cfg.output_dir) / "best.pt", val_loader, cfg)
    print("Final metrics:", metrics)
    if args.plot_cm:
        plot_confusion_matrix(yt, yp, cfg.buckets, Path(cfg.output_dir) / "cm.png")


if __name__ == "__main__":
    main()
