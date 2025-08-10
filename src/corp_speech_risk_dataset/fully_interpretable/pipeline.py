"""fully_interpretable.pipeline
--------------------------------

Sklearn-based pipelines that mirror the data ingestion style of
`coral_ordinal` but use interpretable features and models:

- Text vectorization via TF‑IDF
- Optional tabular scalars from `raw_features` flattened deterministically
- Models: Ridge (linear regression to ordinal classes), MultinomialNB, DecisionTree

All I/O is synchronous and minimal; we rely on scikit-learn tooling for
training and evaluation. The dataset schema expected mirrors
`coral_ordinal.data.JsonDataset` in structure, but without requiring tensor
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json

import numpy as np
from loguru import logger

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import joblib


@dataclass
class InterpretableConfig:
    """Configuration for interpretable baselines.

    This mirrors the important pieces of `coral_ordinal.config.Config` but is
    specific to sklearn training.
    """

    data_path: str
    label_key: str = "bucket"
    # text fields to vectorize via TF‑IDF; any subset of ["text", "context"] is supported
    text_keys: Tuple[str, ...] = ("text",)
    include_text: bool = True
    include_scalars: bool = False  # raw_features scalars
    include_numeric: bool = True  # numeric counts like omission/commission/etc.
    include_keywords: bool = True  # use quote/context keywords
    include_speaker: bool = True  # use speaker_raw
    numeric_keys: Tuple[str, ...] | None = None  # optional override list
    # ordered class labels (e.g., ["Low", "Medium", "High"])
    buckets: Tuple[str, ...] = ("Low", "Medium", "High")
    val_split: float = 0.2
    seed: int = 42
    model: str = "ridge"  # ridge | nb | tree


def _flatten_scalars(raw: Dict[str, Any] | None) -> List[float] | None:
    """Flatten `raw_features` dict into a fixed-length numeric vector.

    We mirror the rules in `coral_ordinal.data.JsonDataset._flatten_scalars` to
    ensure parity of scalar engineering across modules.
    """

    if not isinstance(raw, dict):
        return None

    def pad_or_list(val: Any, expected_len: int) -> List[float]:
        if val is None:
            return [0.0] * expected_len
        if isinstance(val, (list, tuple)):
            arr = list(val)[:expected_len]
            if len(arr) < expected_len:
                arr += [0.0] * (expected_len - len(arr))
            return [float(x) for x in arr]
        return [float(val)] + [0.0] * (expected_len - 1)

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
    parts += pad_or_list(raw.get("quote_sentiment"), q_sent_len)
    parts += pad_or_list(raw.get("context_sentiment"), c_sent_len)
    parts += pad_or_list(raw.get("quote_deontic_count"), 1)
    parts += pad_or_list(raw.get("context_deontic_count"), 1)
    parts += pad_or_list(raw.get("quote_pos"), q_pos_len)
    parts += pad_or_list(raw.get("context_pos"), c_pos_len)
    parts += pad_or_list(raw.get("quote_ner"), q_ner_len)
    parts += pad_or_list(raw.get("context_ner"), c_ner_len)
    parts += pad_or_list(raw.get("quote_deps"), q_dep_len)
    parts += pad_or_list(raw.get("context_deps"), c_dep_len)
    parts += pad_or_list(raw.get("quote_wl"), 1)
    parts += pad_or_list(raw.get("context_wl"), 1)

    return parts


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    This is intentionally simple and mirrors `coral_ordinal.data.JsonDataset`.
    """

    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                logger.error(f"Failed to parse JSON line in {path}: {exc}")
    return records


def _prepare_supervision(
    records: List[Dict[str, Any]], cfg: InterpretableConfig
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Map string labels to integer targets matching `cfg.buckets` order.

    Returns
    -------
    y : np.ndarray (N,)
        Integer class ids
    buckets : list[str]
        The label names in order
    """

    label2idx = {b: i for i, b in enumerate(cfg.buckets)}
    y = np.array([label2idx[r[cfg.label_key]] for r in records], dtype=np.int64)
    return y, np.array(cfg.buckets), list(cfg.buckets)


def _combine_text_fields(rec: Dict[str, Any], text_keys: Iterable[str]) -> str:
    """Concatenate selected text fields with simple separators for TF‑IDF."""

    parts: List[str] = []
    for k in text_keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " \n ".join(parts)


NUMERIC_KEYS: Tuple[str, ...] = (
    "quote_size",
    "quote_omission",
    "quote_commission",
    "context_omission",
    "context_commission",
    "quote_guilt",
    "context_guilt",
    "quote_lying",
    "context_lying",
    "quote_evidential_count",
    "context_evidential_count",
    "quote_causal_count",
    "context_causal_count",
    "quote_conditional_count",
    "context_conditional_count",
    "quote_temporal_count",
    "context_temporal_count",
    "quote_certainty_count",
    "context_certainty_count",
    "quote_discourse_count",
    "context_discourse_count",
    "quote_liability_count",
    "context_liability_count",
)


def _row_to_feature_dict(
    rec: Dict[str, Any], cfg: InterpretableConfig
) -> Dict[str, Any]:
    """Convert a raw record into a multi-branch feature row for the pipeline.

    Keys:
    - text: combined text per cfg.text_keys
    - keywords: space-joined keywords from both quote/context lists
    - speaker: speaker_raw string
    - scalars: numeric array combining selected numeric keys and optional raw_features
    """

    text_value = _combine_text_fields(rec, cfg.text_keys) if cfg.include_text else ""

    # Keywords branch
    kw_tokens: List[str] = []
    if cfg.include_keywords:
        for key in ("quote_top_keywords", "context_top_keywords"):
            vals = rec.get(key) or []
            if isinstance(vals, list):
                for t in vals:
                    if isinstance(t, str) and t:
                        kw_tokens.append(t)
    keywords_value = " ".join(kw_tokens)

    # Speaker branch
    speaker_value = rec.get("speaker_raw") or ""
    if not isinstance(speaker_value, str):
        speaker_value = str(speaker_value)

    # Numeric scalars
    numeric_values: List[float] = []
    if cfg.include_numeric:
        active_numeric_keys = cfg.numeric_keys if cfg.numeric_keys else NUMERIC_KEYS
        for key in active_numeric_keys:
            v = rec.get(key)
            try:
                numeric_values.append(float(v) if v is not None else 0.0)
            except Exception:
                numeric_values.append(0.0)
    if cfg.include_scalars:
        raw_vec = _flatten_scalars(rec.get("raw_features"))
        if raw_vec is None:
            raw_vec = []
        numeric_values.extend(raw_vec)

    return {
        "text": text_value,
        "keywords": keywords_value,
        "speaker": speaker_value,
        "scalars": np.asarray(numeric_values, dtype=np.float32),
    }


def build_dataset(
    cfg: InterpretableConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
    """Load records and prepare multi-branch rows and labels.

    Returns
    -------
    rows : list[dict]
        Feature rows with keys: text, keywords, speaker, scalars
    labels : list[str]
        Ordered label names
    """

    path = Path(cfg.data_path)
    logger.info(f"Loading dataset from {path}")
    records = _load_jsonl_records(path)
    y, _, labels = _prepare_supervision(records, cfg)

    rows: List[Dict[str, Any]] = [_row_to_feature_dict(r, cfg) for r in records]

    # Normalize scalar vector lengths across rows for batch transformers
    max_len = max((row["scalars"].shape[0] for row in rows), default=0)
    for row in rows:
        if row["scalars"].shape[0] < max_len:
            pad_width = max_len - row["scalars"].shape[0]
            row["scalars"] = np.pad(row["scalars"], (0, pad_width), mode="constant")

    return rows, y, labels


def make_model(
    cfg: InterpretableConfig,
    enable_text_branch: bool,
    enable_scalar_branch: bool,
    enable_keywords: bool,
    enable_speaker: bool,
) -> Pipeline:
    """Create a sklearn pipeline for the requested model type.

    - Text → TF‑IDF
    - Scalars → StandardScaler
    - Estimator → RidgeClassifier | MultinomialNB | DecisionTreeClassifier
    """

    if cfg.model == "ridge":
        estimator: BaseEstimator = RidgeClassifier(alpha=1.0)
    elif cfg.model == "nb":
        estimator = MultinomialNB(alpha=0.1)
    elif cfg.model == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=cfg.seed
        )
    else:
        raise ValueError("model must be one of: ridge | nb | tree")

    # Branch 1: TF‑IDF over concatenated text
    branches: List[Tuple[str, Any]] = []
    if enable_text_branch:
        text_branch = Pipeline(
            steps=[
                (
                    "select_text",
                    FunctionTransformer(
                        lambda X: [row["text"] for row in X], validate=False
                    ),
                ),
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        strip_accents="unicode",
                    ),
                ),
            ]
        )
        branches.append(("text", text_branch))

    # Branch 2: keywords
    if enable_keywords:
        keywords_branch = Pipeline(
            steps=[
                (
                    "select_keywords",
                    FunctionTransformer(
                        lambda X: [row["keywords"] for row in X], validate=False
                    ),
                ),
                (
                    "tfidf_kw",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=1.0,
                        strip_accents="unicode",
                        token_pattern=r"(?u)\\b\\w+\\b",  # keep numeric tokens
                    ),
                ),
            ]
        )
        branches.append(("keywords", keywords_branch))

    # Branch 3: speaker
    if enable_speaker:
        speaker_branch = Pipeline(
            steps=[
                (
                    "select_speaker",
                    FunctionTransformer(
                        lambda X: [row["speaker"] for row in X], validate=False
                    ),
                ),
                (
                    "tfidf_speaker",
                    TfidfVectorizer(
                        lowercase=True, ngram_range=(1, 1), min_df=1, max_df=1.0
                    ),
                ),
            ]
        )
        branches.append(("speaker", speaker_branch))

    # Branch 4: numeric scalars (optional). For MultinomialNB we exclude non-negative constraint violations.
    scalar_branch_enabled = enable_scalar_branch and cfg.model in {"ridge", "tree"}
    if scalar_branch_enabled:
        scalar_branch = Pipeline(
            steps=[
                (
                    "select_scalars",
                    FunctionTransformer(
                        lambda X: np.vstack([row["scalars"] for row in X]),
                        validate=False,
                    ),
                ),
                ("scale", StandardScaler()),
            ]
        )

        branches.append(("scalars", scalar_branch))

    features = FeatureUnion(transformer_list=branches)

    pipe = Pipeline(steps=[("features", features), ("clf", estimator)])
    return pipe


def train_and_eval(cfg: InterpretableConfig) -> Dict[str, Any]:
    """Train and evaluate according to `cfg` and return a metrics dict."""

    rows, y, labels = build_dataset(cfg)
    X = rows

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.val_split, random_state=cfg.seed, stratify=y
    )

    enable_text_branch = bool(cfg.include_text)
    enable_scalar_branch = cfg.include_scalars or cfg.include_numeric
    model = make_model(
        cfg,
        enable_text_branch=enable_text_branch,
        enable_scalar_branch=enable_scalar_branch,
        enable_keywords=cfg.include_keywords,
        enable_speaker=cfg.include_speaker,
    )
    logger.info(f"Training interpretable model: {cfg.model}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    report = classification_report(y_val, y_pred, target_names=labels, output_dict=True)
    metrics = {
        "accuracy": float(report["accuracy"]),
        # keep parity with coral metrics as much as possible
        "exact": float(report["accuracy"]),
    }
    return {
        "metrics": metrics,
        "report": report,
        "labels": labels,
        "y_true": y_val.tolist(),
        "y_pred": y_pred.tolist(),
    }


def save_model(
    model: Pipeline, labels: List[str], cfg: InterpretableConfig, path: Path | str
) -> None:
    """Persist the trained pipeline and metadata."""
    payload = {"model": model, "labels": labels, "cfg": asdict(cfg)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    logger.info(f"Saved model to {path}")


def load_model(path: Path | str) -> Tuple[Pipeline, List[str], InterpretableConfig]:
    """Load a trained pipeline and metadata."""
    path = Path(path)
    payload = joblib.load(path)
    cfg = (
        InterpretableConfig(**payload["cfg"])
        if isinstance(payload.get("cfg"), dict)
        else payload["cfg"]
    )
    return payload["model"], list(payload["labels"]), cfg


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def predict_records(
    model: Pipeline, labels: List[str], rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Predict for a batch of rows and return per-sample outputs.

    Returns a dict with keys: y_pred, probs (or None), scores (or None)
    """
    y_pred = model.predict(rows)
    probs = None
    scores = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        try:
            probs = model.predict_proba(rows)
        except Exception:
            probs = None
    if probs is None and hasattr(model.named_steps["clf"], "decision_function"):
        try:
            df = model.decision_function(rows)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T  # binary fallback
            scores = df
            # Convert to pseudo-probs
            probs = _softmax(df)
        except Exception:
            scores = None
            probs = None
    return {"y_pred": y_pred, "probs": probs, "scores": scores}


def enrich_record_with_prediction(
    rec: Dict[str, Any],
    pred_class: int,
    probs: np.ndarray | None,
    scores: np.ndarray | None,
    labels: List[str],
    model_tag: str,
) -> Dict[str, Any]:
    """Return a copy of record with appended prediction fields under model-specific prefix."""

    out = dict(rec)
    prefix = f"fi_{model_tag}_"
    out[prefix + "pred_class"] = int(pred_class)
    out[prefix + "pred_bucket"] = str(labels[int(pred_class)]).lower()
    # Confidence as max probability when available, else None
    if probs is not None:
        conf = float(np.max(probs))
        out[prefix + "confidence"] = conf
        # class probs dict
        prob_map = {labels[i].lower(): float(probs[i]) for i in range(len(labels))}
        out[prefix + "class_probs"] = prob_map
        # flattened convenience fields
        for i, name in enumerate(labels):
            out[prefix + f"prob_{name.lower()}"] = float(probs[i])
    else:
        out[prefix + "confidence"] = None
    if scores is not None:
        out[prefix + "scores"] = [float(s) for s in scores]
    out[prefix + "model_buckets"] = [str(x).lower() for x in labels]
    out[prefix + "model_threshold"] = None
    return out


def predict_directory(
    model_path: str | Path,
    input_root: str | Path,
    output_root: str | Path | None = None,
) -> Path:
    """Predict recursively over JSONL files, writing enriched mirrored outputs.

    - model_tag is inferred from config.model (ridge|nb|tree)
    - output_root defaults to f"{input_root}_fi_{model}"
    """

    pipe, labels, cfg = load_model(model_path)
    model_tag = cfg.model
    input_root = Path(input_root)
    if output_root is None:
        output_root = input_root.parent / f"{input_root.name}_fi_{model_tag}"
    output_root = Path(output_root)

    logger.info(f"Predicting {model_tag} over {input_root} → {output_root}")

    for in_path in input_root.rglob("*.jsonl"):
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            records = _load_jsonl_records(in_path)
        except Exception as exc:
            logger.error(f"Failed reading {in_path}: {exc}")
            continue

        rows = [_row_to_feature_dict(r, cfg) for r in records]
        # ensure equal scalar length
        max_len = max((row["scalars"].shape[0] for row in rows), default=0)
        for row in rows:
            if row["scalars"].shape[0] < max_len:
                pad_width = max_len - row["scalars"].shape[0]
                row["scalars"] = np.pad(row["scalars"], (0, pad_width), mode="constant")

        preds = predict_records(pipe, labels, rows)
        with out_path.open("w") as fout:
            for i, rec in enumerate(records):
                pred_cls = int(preds["y_pred"][i])
                probs_row = preds["probs"][i] if preds["probs"] is not None else None
                scores_row = (
                    preds["scores"][i].tolist() if preds["scores"] is not None else None
                )
                enriched = enrich_record_with_prediction(
                    rec,
                    pred_cls,
                    probs_row,
                    np.array(scores_row) if scores_row is not None else None,
                    labels,
                    model_tag,
                )
                fout.write(json.dumps(enriched) + "\n")

    return Path(output_root)
