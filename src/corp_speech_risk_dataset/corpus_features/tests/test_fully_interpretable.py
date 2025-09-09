"""Unit tests for fully_interpretable module.

These are smoke tests that verify the pipeline runs end-to-end on a tiny
synthetic JSONL dataset and produces predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.corp_speech_risk_dataset.fully_interpretable.pipeline import (
    InterpretableConfig,
    train_and_eval,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_pipeline_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "toy.jsonl"
    rows = [
        {"text": "profit rose", "context": "earnings up", "bucket": "High"},
        {"text": "profit fell", "context": "earnings down", "bucket": "Low"},
        {"text": "revenue stable", "context": "flat outlook", "bucket": "Medium"},
        {"text": "profits surge", "context": "growth strong", "bucket": "High"},
        {"text": "profits slump", "context": "growth weak", "bucket": "Low"},
    ]
    _write_jsonl(data_path, rows)

    cfg = InterpretableConfig(
        data_path=str(data_path),
        buckets=("Low", "Medium", "High"),
        text_keys=("text", "context"),
        include_scalars=False,
        model="ridge",
        val_split=0.4,
        seed=0,
    )
    results = train_and_eval(cfg)
    assert "metrics" in results and "accuracy" in results["metrics"]
    assert len(results["y_pred"]) == len(results["y_true"]) > 0
