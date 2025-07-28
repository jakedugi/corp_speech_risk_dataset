# ----------------------------------
# coral_ordinal/train.py
# ----------------------------------
from __future__ import annotations
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from .losses import coral_loss
from .metrics import compute_metrics
from .utils import set_seed, choose_device, save_checkpoint


def train(model, train_loader, val_loader, cfg):
    device = choose_device(cfg.device)
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(
        enabled=False
    )  # MPS doesn't support AMP well; keep False by default

    best_val = -1
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast(enabled=False):
                logits = model(x)
                loss = coral_loss(logits, y, model.num_classes)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = (torch.sigmoid(logits) > cfg.prob_threshold).sum(1)
                ys.append(y.numpy())
                ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        metrics = compute_metrics(y_true, y_pred)
        val_score = metrics["exact"].value  # can use composite metric if desired

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | loss={avg_loss:.4f} | exact={metrics['exact'].value:.3f} | off1={metrics['off_by_one'].value:.3f} | rho={metrics['spearman_r'].value:.3f}"
        )

        if val_score > best_val:
            best_val = val_score
            save_checkpoint(model, outdir / "best.pt", cfg)

    save_checkpoint(model, outdir / "last.pt", cfg)
    return model
