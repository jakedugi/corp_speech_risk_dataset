# ----------------------------------
# coral_ordinal/data.py
# ----------------------------------
from __future__ import annotations
from typing import List, Dict, Any
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class JsonDataset(Dataset):
    """JSONL dataset with support for single or multi-key feature assembly.

    - If ``feature_keys`` is provided, concatenate vectors (missing keys => zeros).
    - Else fall back to single ``feature_key`` behavior.
    - Optionally append flattened ``raw_features`` scalar priors (include_scalars).
    """

    def __init__(
        self,
        path: str | Path,
        feature_key: str,
        label_key: str,
        buckets: List[str],
        feature_keys: List[str] | None = None,
        include_scalars: bool = False,
    ):
        self.feature_key = feature_key
        self.feature_keys = feature_keys
        self.include_scalars = include_scalars
        self.label_key = label_key
        self.buckets = buckets
        self.data: List[Dict[str, Any]] = []
        with Path(path).open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.data.append(obj)
        # map labels to indices
        self.label2idx = {b: i for i, b in enumerate(self.buckets)}

        # Infer per-key dims for padding
        self.key_dims: Dict[str, int] = {}
        if self.feature_keys:
            for key in self.feature_keys:
                dim = 0
                for rec in self.data:
                    if key in rec and isinstance(rec[key], (list, tuple)):
                        dim = len(rec[key])
                        break
                self.key_dims[key] = dim

        # Determine scalar dim once if requested
        self.scalar_dim: int = 0
        if self.include_scalars:
            for rec in self.data:
                raw = rec.get("raw_features")
                if isinstance(raw, dict):
                    self.scalar_dim = len(self._flatten_scalars(raw))
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        if self.feature_keys:
            parts: List[torch.Tensor] = []
            for key in self.feature_keys:
                vals = obj.get(key)
                if isinstance(vals, (list, tuple)):
                    parts.append(torch.tensor(vals, dtype=torch.float32))
                else:
                    pad_dim = self.key_dims.get(key, 0)
                    if pad_dim > 0:
                        parts.append(torch.zeros(pad_dim, dtype=torch.float32))
            if self.include_scalars:
                raw_vec = self._flatten_scalars(obj.get("raw_features"))
                if raw_vec is not None:
                    parts.append(torch.tensor(raw_vec, dtype=torch.float32))
                elif self.scalar_dim > 0:
                    parts.append(torch.zeros(self.scalar_dim, dtype=torch.float32))
            x = torch.cat(parts, dim=0)
        else:
            x = torch.tensor(obj[self.feature_key], dtype=torch.float32)
        y = torch.tensor(self.label2idx[obj[self.label_key]], dtype=torch.long)
        return x, y

    @staticmethod
    def _pad_or_list(val: Any, expected_len: int) -> List[float]:
        if val is None:
            return [0.0] * expected_len
        if isinstance(val, (list, tuple)):
            arr = list(val)[:expected_len]
            if len(arr) < expected_len:
                arr += [0.0] * (expected_len - len(arr))
            return [float(x) for x in arr]
        return [float(val)] + [0.0] * (expected_len - 1)

    def _flatten_scalars(self, raw: Dict[str, Any] | None) -> List[float] | None:
        if not isinstance(raw, dict):
            return None

        def infer_len(key: str, default_len: int) -> int:
            v = raw.get(key)
            if isinstance(v, (list, tuple)):
                return len(v)
            return default_len

        q_sent_len = infer_len("quote_sentiment", 3)
        c_sent_len = infer_len("context_sentiment", 3)
        q_pos_len = infer_len("quote_pos", 11)
        c_pos_len = infer_len("context_pos", 11)
        q_ner_len = infer_len("quote_ner", 7)
        c_ner_len = infer_len("context_ner", 7)
        q_dep_len = infer_len("quote_deps", 23)
        c_dep_len = infer_len("context_deps", 23)

        parts: List[float] = []
        parts += self._pad_or_list(raw.get("quote_sentiment"), q_sent_len)
        parts += self._pad_or_list(raw.get("context_sentiment"), c_sent_len)
        parts += self._pad_or_list(raw.get("quote_deontic_count"), 1)
        parts += self._pad_or_list(raw.get("context_deontic_count"), 1)
        parts += self._pad_or_list(raw.get("quote_pos"), q_pos_len)
        parts += self._pad_or_list(raw.get("context_pos"), c_pos_len)
        parts += self._pad_or_list(raw.get("quote_ner"), q_ner_len)
        parts += self._pad_or_list(raw.get("context_ner"), c_ner_len)
        parts += self._pad_or_list(raw.get("quote_deps"), q_dep_len)
        parts += self._pad_or_list(raw.get("context_deps"), c_dep_len)
        parts += self._pad_or_list(raw.get("quote_wl"), 1)
        parts += self._pad_or_list(raw.get("context_wl"), 1)

        return parts


def build_loaders(cfg: Config):
    ds = JsonDataset(
        cfg.data_path,
        cfg.feature_key,
        cfg.label_key,
        cfg.buckets,
        feature_keys=cfg.feature_keys,
        include_scalars=cfg.include_scalars,
    )
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
