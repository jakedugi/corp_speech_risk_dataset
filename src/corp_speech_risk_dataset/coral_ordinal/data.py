# ----------------------------------
# coral_ordinal/data.py
# ----------------------------------
from __future__ import annotations
from typing import List, Dict
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class JsonDataset(Dataset):
    """Expect one JSON object per line, each containing at least feature_key and label_key.
    Feature must be a list/array of floats.
    """

    def __init__(
        self, path: str | Path, feature_key: str, label_key: str, buckets: List[str]
    ):
        self.feature_key = feature_key
        self.label_key = label_key
        self.buckets = buckets
        self.data: List[Dict] = []
        with Path(path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.data.append(obj)
        # map labels to indices
        self.label2idx = {b: i for i, b in enumerate(self.buckets)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        x = torch.tensor(obj[self.feature_key], dtype=torch.float32)
        y = torch.tensor(self.label2idx[obj[self.label_key]], dtype=torch.long)
        return x, y


def build_loaders(cfg: Config):
    ds = JsonDataset(cfg.data_path, cfg.feature_key, cfg.label_key, cfg.buckets)
    val_len = int(len(ds) * cfg.val_split)
    train_len = len(ds) - val_len
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, ds
