"""fully_interpretable.pipeline
--------------------------------

Enhanced sklearn-based pipelines for interpretable legal risk classification
with state-of-the-art performance and publication-ready outputs.

Key features:
- Advanced interpretable models: POLR, EBM, calibrated classifiers
- Risk lexicons and sophisticated linguistic features
- Transparent sequence modeling without embeddings
- Comprehensive interpretation and validation experiments
- Optimized performance with sparse matrices and parallel processing

All models maintain full interpretability while achieving competitive performance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from operator import itemgetter

import numpy as np
from scipy import sparse
from loguru import logger

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import RidgeClassifier, LinearRegression, LogisticRegression


# Helper functions for FunctionTransformer (to avoid lambda pickling issues)
def _extract_text_field(X):
    """Extract text field from records."""
    return [row.get("text", "") for row in X]


def _extract_keywords_field(X):
    """Extract keywords field from records."""
    return [row.get("keywords", "") for row in X]


def _extract_speaker_field(X):
    """Extract speaker field from records."""
    return [row.get("speaker", "") for row in X]


def _extract_scalars_field(X):
    """Extract and stack scalar features from records."""
    return np.vstack([row.get("scalars", np.array([])) for row in X])


from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
import joblib

# Import enhanced components
from .models import (
    ProportionalOddsLogisticRegression,
    CalibratedInterpretableClassifier,
    TransparentEnsemble,
    create_ebm_classifier,
)
from .features import InterpretableFeatureExtractor, create_feature_matrix
from .interpretation import InterpretabilityReport
from .validation import ValidationExperiments

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class InterpretableConfig:
    """Enhanced configuration for interpretable baselines.

    Supports advanced models, risk lexicons, and comprehensive feature engineering.
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
    # Risk lexicons and enhanced features
    include_lexicons: bool = True
    include_sequence: bool = True
    include_linguistic: bool = True
    include_structural: bool = True
    lexicon_weights: Dict[str, float] | None = None
    # ordered class labels (e.g., ["low", "medium", "high"])
    buckets: Tuple[str, ...] = ("low", "medium", "high")
    val_split: float = 0.2
    seed: int = 42
    model: str = (
        "ridge"  # ridge | nb | tree | linreg | polr | ebm | logistic | ensemble
    )
    # Model-specific parameters
    model_params: Dict[str, Any] | None = None
    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # isotonic | sigmoid
    # Feature selection
    feature_selection: bool = True
    n_features: int = 5000  # for chi2 selection
    # Performance optimization
    use_sparse: bool = True
    n_jobs: int = -1  # for parallel processing
    # Output options
    generate_report: bool = True
    output_dir: str | None = None


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

    # Extract labels and check for missing mappings
    raw_labels = [r[cfg.label_key] for r in records]
    unique_labels_in_data = set(raw_labels)
    missing_labels = unique_labels_in_data - set(cfg.buckets)

    if missing_labels:
        raise ValueError(
            f"Found labels in data that are not in cfg.buckets: {missing_labels}. "
            f"Data labels: {sorted(unique_labels_in_data)}, "
            f"Expected buckets: {cfg.buckets}"
        )

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
    rec: Dict[str, Any],
    cfg: InterpretableConfig,
    feature_extractor: Optional[InterpretableFeatureExtractor] = None,
) -> Dict[str, Any]:
    """Convert a raw record into a multi-branch feature row for the pipeline.

    Enhanced with risk lexicons, sequence features, and linguistic analysis.
    """
    # Extract enhanced interpretable features if enabled
    enhanced_features = {}
    if any(
        [
            cfg.include_lexicons,
            cfg.include_sequence,
            cfg.include_linguistic,
            cfg.include_structural,
        ]
    ):
        if feature_extractor is None:
            feature_extractor = InterpretableFeatureExtractor(
                include_lexicons=cfg.include_lexicons,
                include_sequence=cfg.include_sequence,
                include_linguistic=cfg.include_linguistic,
                include_structural=cfg.include_structural,
                lexicon_weights=cfg.lexicon_weights,
            )
        enhanced_features = feature_extractor.extract_features(
            rec.get("text", ""), rec.get("context", "")
        )

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

    # Numeric scalars (combine original and enhanced features)
    numeric_values: List[float] = []

    # Original numeric features
    if cfg.include_numeric:
        active_numeric_keys = cfg.numeric_keys if cfg.numeric_keys else NUMERIC_KEYS
        for key in active_numeric_keys:
            v = rec.get(key)
            try:
                numeric_values.append(float(v) if v is not None else 0.0)
            except Exception:
                numeric_values.append(0.0)

    # Raw features scalars
    if cfg.include_scalars:
        raw_vec = _flatten_scalars(rec.get("raw_features"))
        if raw_vec is None:
            raw_vec = []
        numeric_values.extend(raw_vec)

    # Enhanced features as additional scalars
    if enhanced_features:
        # Convert enhanced features to ordered list
        enhanced_keys = sorted(enhanced_features.keys())
        for key in enhanced_keys:
            numeric_values.append(enhanced_features[key])

    return {
        "text": text_value,
        "keywords": keywords_value,
        "speaker": speaker_value,
        "scalars": np.asarray(numeric_values, dtype=np.float32),
        "enhanced_features": enhanced_features,  # Keep for reference
    }


def build_dataset(
    cfg: InterpretableConfig,
    use_parallel: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str], InterpretableFeatureExtractor]:
    """Load records and prepare multi-branch rows and labels.

    Enhanced with parallel processing for large datasets.

    Returns
    -------
    rows : list[dict]
        Feature rows with keys: text, keywords, speaker, scalars
    y : np.ndarray
        Integer labels
    labels : list[str]
        Ordered label names
    feature_extractor : InterpretableFeatureExtractor
        Feature extractor for consistency
    """
    path = Path(cfg.data_path)
    logger.info(f"Loading dataset from {path}")
    records = _load_jsonl_records(path)
    y, _, labels = _prepare_supervision(records, cfg)

    # Create feature extractor
    feature_extractor = InterpretableFeatureExtractor(
        include_lexicons=cfg.include_lexicons,
        include_sequence=cfg.include_sequence,
        include_linguistic=cfg.include_linguistic,
        include_structural=cfg.include_structural,
        lexicon_weights=cfg.lexicon_weights,
    )

    # Extract features (optionally in parallel)
    if use_parallel and cfg.n_jobs != 1 and len(records) > 100:
        rows = _parallel_feature_extraction(records, cfg, feature_extractor)
    else:
        rows = [_row_to_feature_dict(r, cfg, feature_extractor) for r in records]

    # Normalize scalar vector lengths across rows for batch transformers
    max_len = max((row["scalars"].shape[0] for row in rows), default=0)
    for row in rows:
        if row["scalars"].shape[0] < max_len:
            pad_width = max_len - row["scalars"].shape[0]
            row["scalars"] = np.pad(row["scalars"], (0, pad_width), mode="constant")

    return rows, y, labels, feature_extractor


def _parallel_feature_extraction(
    records: List[Dict[str, Any]],
    cfg: InterpretableConfig,
    feature_extractor: InterpretableFeatureExtractor,
) -> List[Dict[str, Any]]:
    """Extract features in parallel for better performance."""
    n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else None

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit jobs in batches
        batch_size = max(1, len(records) // (n_jobs or 4))
        futures = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            future = executor.submit(_process_batch, batch, cfg, feature_extractor)
            futures.append(future)

        # Collect results
        all_rows = []
        for future in as_completed(futures):
            all_rows.extend(future.result())

    return all_rows


def _process_batch(
    records: List[Dict[str, Any]],
    cfg: InterpretableConfig,
    feature_extractor: InterpretableFeatureExtractor,
) -> List[Dict[str, Any]]:
    """Process a batch of records."""
    return [_row_to_feature_dict(r, cfg, feature_extractor) for r in records]


def make_model(
    cfg: InterpretableConfig,
    enable_text_branch: bool,
    enable_scalar_branch: bool,
    enable_keywords: bool,
    enable_speaker: bool,
) -> Pipeline:
    """Create an enhanced sklearn pipeline for the requested model type.

    Supports:
    - Traditional: Ridge, MultinomialNB, DecisionTree, LinearRegression
    - Advanced: POLR, EBM, Calibrated models, Ensembles
    - Optimizations: Feature selection, sparse matrices
    """
    # Get model parameters
    model_params = cfg.model_params or {}

    # Create base estimator
    if cfg.model == "ridge":
        estimator: BaseEstimator = RidgeClassifier(
            alpha=model_params.get("alpha", 1.0),
            random_state=cfg.seed,
        )
    elif cfg.model == "nb":
        estimator = MultinomialNB(
            alpha=model_params.get("alpha", 0.1),
        )
    elif cfg.model == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=model_params.get("max_depth", 6),
            min_samples_leaf=model_params.get("min_samples_leaf", 5),
            min_samples_split=model_params.get("min_samples_split", 10),
            random_state=cfg.seed,
        )
    elif cfg.model == "linreg":
        estimator = LinearRegression()
    elif cfg.model == "logistic":
        estimator = LogisticRegression(
            penalty=model_params.get("penalty", "l2"),
            C=model_params.get("C", 1.0),
            solver=model_params.get("solver", "lbfgs"),
            max_iter=model_params.get("max_iter", 1000),
            multi_class="multinomial",
            random_state=cfg.seed,
        )
    elif cfg.model == "polr":
        estimator = ProportionalOddsLogisticRegression(
            penalty=model_params.get("penalty", "l2"),
            C=model_params.get("C", 1.0),
            solver=model_params.get("solver", "lbfgs"),
            max_iter=model_params.get("max_iter", 1000),
            random_state=cfg.seed,
        )
    elif cfg.model == "ebm":
        estimator = create_ebm_classifier(
            max_bins=model_params.get("max_bins", 256),
            interactions=model_params.get("interactions", 0),
            learning_rate=model_params.get("learning_rate", 0.01),
            max_rounds=model_params.get("max_rounds", 5000),
            random_state=cfg.seed,
        )
        if estimator is None:
            logger.warning("EBM not available, falling back to Ridge")
            estimator = RidgeClassifier(alpha=1.0, random_state=cfg.seed)
    elif cfg.model == "ensemble":
        # Create transparent ensemble
        base_estimators = []

        # Add diverse base models
        base_estimators.append(
            ("ridge", RidgeClassifier(alpha=1.0, random_state=cfg.seed))
        )
        base_estimators.append(
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    multi_class="multinomial",
                    random_state=cfg.seed,
                ),
            )
        )
        if enable_scalar_branch:  # Tree works better with scalars
            base_estimators.append(
                (
                    "tree",
                    DecisionTreeClassifier(
                        max_depth=6, min_samples_leaf=5, random_state=cfg.seed
                    ),
                )
            )

        estimator = TransparentEnsemble(
            estimators=base_estimators,
            voting=model_params.get("voting", "soft"),
            weights=model_params.get("weights", None),
        )
    else:
        raise ValueError(
            f"model must be one of: ridge | nb | tree | linreg | "
            f"logistic | polr | ebm | ensemble, got {cfg.model}"
        )

    # Apply calibration if requested
    if cfg.calibrate and cfg.model not in [
        "polr",
        "ebm",
        "linreg",
    ]:  # These are well-calibrated or don't support calibration
        estimator = CalibratedInterpretableClassifier(
            base_estimator=estimator,
            method=cfg.calibration_method,
            cv=3,
        )

    # Branch 1: TF‑IDF over concatenated text
    branches: List[Tuple[str, Any]] = []
    if enable_text_branch:
        # Optimize TF-IDF parameters
        tfidf_params = {
            "lowercase": True,
            "ngram_range": (1, 2),
            "min_df": model_params.get(
                "min_df", 1
            ),  # Reduced from 3 to handle small texts
            "max_df": model_params.get("max_df", 0.95),
            "strip_accents": "unicode",
            "use_idf": True,
            "sublinear_tf": True,  # Log normalization
            "dtype": np.float32,  # Memory efficiency
        }

        # Add max_features for feature selection
        if cfg.feature_selection:
            tfidf_params["max_features"] = cfg.n_features

        text_steps = [
            (
                "select_text",
                FunctionTransformer(_extract_text_field, validate=False),
            ),
            ("tfidf", TfidfVectorizer(**tfidf_params)),
        ]

        # Optional chi2 feature selection
        if (
            cfg.feature_selection and cfg.model != "nb"
        ):  # NB doesn't play well with chi2
            text_steps.append(
                ("select_best", SelectKBest(chi2, k=min(cfg.n_features, 2000)))
            )

        text_branch = Pipeline(steps=text_steps)
        branches.append(("text", text_branch))

    # Branch 2: keywords
    if enable_keywords:
        keywords_branch = Pipeline(
            steps=[
                (
                    "select_keywords",
                    FunctionTransformer(_extract_keywords_field, validate=False),
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
                    FunctionTransformer(_extract_speaker_field, validate=False),
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

    # Branch 4: numeric scalars (with proper handling for different models)
    if enable_scalar_branch:
        scalar_steps = [
            (
                "select_scalars",
                FunctionTransformer(_extract_scalars_field, validate=False),
            ),
        ]

        # Handle non-negative requirement for MultinomialNB
        if cfg.model == "nb":
            # MinMaxScaler to ensure non-negative values
            from sklearn.preprocessing import MinMaxScaler

            scalar_steps.append(("scale", MinMaxScaler(feature_range=(0, 1))))
        else:
            # StandardScaler for other models
            scalar_steps.append(("scale", StandardScaler()))

        scalar_branch = Pipeline(steps=scalar_steps)
        branches.append(("scalars", scalar_branch))

    features = FeatureUnion(transformer_list=branches)

    pipe = Pipeline(steps=[("features", features), ("clf", estimator)])
    return pipe


def train_and_eval(
    cfg: InterpretableConfig,
    run_validation: bool = True,
    run_interpretability: bool = True,
    case_outcomes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enhanced training and evaluation with comprehensive metrics and validation.

    Includes:
    - Advanced metrics (QWK, MAE, calibration)
    - Optional validation experiments
    - Interpretability reports
    - Publication-ready outputs
    """
    rows, y, labels, feature_extractor = build_dataset(cfg)
    X = rows

    # Stratified split maintaining class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.val_split, random_state=cfg.seed, stratify=y
    )

    # Check if we have enough enhanced features
    enable_text_branch = bool(cfg.include_text)
    enable_scalar_branch = (
        cfg.include_scalars
        or cfg.include_numeric
        or any(
            [
                cfg.include_lexicons,
                cfg.include_sequence,
                cfg.include_linguistic,
                cfg.include_structural,
            ]
        )
    )

    model = make_model(
        cfg,
        enable_text_branch=enable_text_branch,
        enable_scalar_branch=enable_scalar_branch,
        enable_keywords=cfg.include_keywords,
        enable_speaker=cfg.include_speaker,
    )

    logger.info(f"Training interpretable model: {cfg.model}")
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_val)
    y_proba = None

    if cfg.model == "linreg":
        # Convert continuous output to ordinal class ids
        y_pred = np.rint(y_pred).astype(int)
        y_pred = np.clip(y_pred, 0, len(labels) - 1)
    else:
        # Get probabilities if available
        try:
            y_proba = model.predict_proba(X_val)
        except:
            pass

    # Comprehensive metrics
    report = classification_report(y_val, y_pred, target_names=labels, output_dict=True)

    # Convert labels to ordinal for metrics
    label_to_ord = {label: i for i, label in enumerate(labels)}
    y_val_ord = np.array([label_to_ord.get(y, y) for y in y_val])
    y_pred_ord = np.array([label_to_ord.get(y, y) for y in y_pred])

    # Quadratic Weighted Kappa (primary metric for ordinal)
    qwk = cohen_kappa_score(y_val, y_pred, weights="quadratic")

    # Mean Absolute Error for ordinal
    mae = mean_absolute_error(y_val_ord, y_pred_ord)

    metrics = {
        "accuracy": float(report["accuracy"]),
        "exact": float(report["accuracy"]),  # for compatibility
        "qwk": float(qwk),
        "mae": float(mae),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    # Expected Calibration Error if probabilities available
    if y_proba is not None:
        from sklearn.calibration import calibration_curve

        ece = _compute_ece(y_val_ord, y_proba)
        metrics["ece"] = float(ece)

    results = {
        "metrics": metrics,
        "report": report,
        "labels": labels,
        "y_true": y_val.tolist(),
        "y_pred": y_pred.tolist(),
        "model": model,
    }

    # Generate interpretability report if requested
    if cfg.generate_report and cfg.output_dir and run_interpretability:
        logger.info("Generating interpretability report...")
        output_dir = Path(cfg.output_dir)

        # Get feature names from the pipeline
        feature_names = _extract_feature_names(model)

        interpreter = InterpretabilityReport(
            model=model,
            feature_names=feature_names,
            class_names=labels,
            output_dir=output_dir / "interpretability",
        )

        interp_report = interpreter.generate_full_report(
            X_test=X_val,
            y_test=y_val,
            X_train=X_train,
            y_train=y_train,
        )
        results["interpretability"] = interp_report

    # Run validation experiments if requested
    if run_validation and cfg.output_dir:
        logger.info("Running validation experiments...")
        validator = ValidationExperiments(
            model=model,
            feature_extractor=feature_extractor,
            output_dir=Path(cfg.output_dir) / "validation",
        )

        # Combine train and val for full dataset experiments
        all_data = X_train + X_val
        all_labels = np.concatenate([y_train, y_val])

        validation_results = validator.run_all_experiments(
            data=all_data,
            labels=all_labels,
            case_outcomes=case_outcomes,
        )
        results["validation"] = validation_results

    return results


def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    ece = 0.0
    for class_idx in range(y_proba.shape[1]):
        y_binary = (y_true == class_idx).astype(int)
        proba = y_proba[:, class_idx]

        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (proba > bin_lower) & (proba <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_binary[in_bin].mean()
                avg_confidence_in_bin = proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece / y_proba.shape[1]


def _extract_feature_names(model: Pipeline) -> List[str]:
    """Extract feature names from a fitted pipeline."""
    feature_names = []

    try:
        # Get the feature union
        feature_union = model.named_steps["features"]

        for name, transformer in feature_union.transformer_list:
            if name == "text" and hasattr(transformer, "named_steps"):
                # Get TF-IDF vocabulary
                tfidf = transformer.named_steps.get("tfidf")
                if tfidf and hasattr(tfidf, "get_feature_names_out"):
                    text_features = tfidf.get_feature_names_out()
                    feature_names.extend([f"text_{f}" for f in text_features])
            elif name == "keywords" and hasattr(transformer, "named_steps"):
                tfidf_kw = transformer.named_steps.get("tfidf_kw")
                if tfidf_kw and hasattr(tfidf_kw, "get_feature_names_out"):
                    kw_features = tfidf_kw.get_feature_names_out()
                    feature_names.extend([f"keywords_{f}" for f in kw_features])
            elif name == "speaker" and hasattr(transformer, "named_steps"):
                tfidf_sp = transformer.named_steps.get("tfidf_speaker")
                if tfidf_sp and hasattr(tfidf_sp, "get_feature_names_out"):
                    sp_features = tfidf_sp.get_feature_names_out()
                    feature_names.extend([f"speaker_{f}" for f in sp_features])
            elif name == "scalars":
                # Add scalar feature names
                # This would need to be enhanced to track actual names
                n_scalars = 100  # Placeholder
                feature_names.extend([f"scalar_{i}" for i in range(n_scalars)])
    except:
        logger.warning("Could not extract feature names, using generic names")

    return feature_names


def save_model(
    model: Pipeline,
    labels: List[str],
    cfg: InterpretableConfig,
    path: Path | str,
    feature_extractor: Optional[InterpretableFeatureExtractor] = None,
) -> None:
    """Persist the trained pipeline, metadata, and feature extractor."""
    payload = {
        "model": model,
        "labels": labels,
        "cfg": asdict(cfg),
        "feature_extractor": feature_extractor,
        "version": "2.0",  # Track enhanced version
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)  # Add compression
    logger.info(f"Saved enhanced model to {path}")


def load_model(
    path: Path | str,
) -> Tuple[
    Pipeline, List[str], InterpretableConfig, Optional[InterpretableFeatureExtractor]
]:
    """Load a trained pipeline, metadata, and feature extractor."""
    path = Path(path)
    payload = joblib.load(path)

    cfg = (
        InterpretableConfig(**payload["cfg"])
        if isinstance(payload.get("cfg"), dict)
        else payload["cfg"]
    )

    # Handle legacy models without feature extractor
    feature_extractor = payload.get("feature_extractor", None)
    if feature_extractor is None and any(
        [
            cfg.include_lexicons,
            cfg.include_sequence,
            cfg.include_linguistic,
            cfg.include_structural,
        ]
    ):
        # Recreate feature extractor for legacy models
        feature_extractor = InterpretableFeatureExtractor(
            include_lexicons=cfg.include_lexicons,
            include_sequence=cfg.include_sequence,
            include_linguistic=cfg.include_linguistic,
            include_structural=cfg.include_structural,
            lexicon_weights=cfg.lexicon_weights,
        )

    return payload["model"], list(payload["labels"]), cfg, feature_extractor


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
    if isinstance(model.named_steps["clf"], LinearRegression):
        y_pred = np.rint(y_pred).astype(int)
        y_pred = np.clip(y_pred, 0, len(labels) - 1)
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
    """Return a copy of record with appended prediction fields matching coral format.

    Uses model-specific prefixes like 'polr_', 'ebm_', 'ensemble_' similar to 'coral_'.
    """

    out = dict(rec)
    # Use model name as prefix (similar to coral_)
    prefix = f"{model_tag}_"

    # Core prediction fields
    out[prefix + "pred_bucket"] = str(labels[int(pred_class)]).lower()
    out[prefix + "pred_class"] = int(pred_class)

    # Confidence and probabilities
    if probs is not None:
        conf = float(np.max(probs))
        out[prefix + "confidence"] = conf

        # Class probabilities dict
        prob_map = {labels[i].lower(): float(probs[i]) for i in range(len(labels))}
        out[prefix + "class_probs"] = prob_map

        # Individual probability fields (matching coral format)
        for i, name in enumerate(labels):
            out[prefix + f"prob_{name.lower()}"] = float(probs[i])
    else:
        out[prefix + "confidence"] = None

    # Scores if available (matching coral format)
    if scores is not None:
        out[prefix + "scores"] = [float(s) for s in scores.tolist()]

    # Model metadata
    out[prefix + "model_threshold"] = 0.48  # Default threshold, can be customized
    out[prefix + "model_buckets"] = [str(x).lower() for x in labels]

    return out


def predict_directory(
    model_path: str | Path,
    input_root: str | Path,
    output_root: str | Path | None = None,
    batch_size: int = 100,
) -> Path:
    """Enhanced directory prediction with batch processing and parallel support.

    - Handles enhanced features
    - Processes in batches for efficiency
    - Preserves all metadata
    """
    pipe, labels, cfg, feature_extractor = load_model(model_path)
    model_tag = cfg.model
    input_root = Path(input_root)
    if output_root is None:
        output_root = input_root.parent / f"{input_root.name}_fi_{model_tag}"
    output_root = Path(output_root)

    logger.info(f"Predicting {model_tag} over {input_root} → {output_root}")

    # Track statistics
    total_files = 0
    total_records = 0

    for in_path in input_root.rglob("*.jsonl"):
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            records = _load_jsonl_records(in_path)
            if not records:
                continue

            total_files += 1
            total_records += len(records)

            # Process in batches for memory efficiency
            with out_path.open("w") as fout:
                for batch_start in range(0, len(records), batch_size):
                    batch_records = records[batch_start : batch_start + batch_size]

                    # Extract features
                    rows = [
                        _row_to_feature_dict(r, cfg, feature_extractor)
                        for r in batch_records
                    ]

                    # Normalize scalar lengths
                    max_len = max((row["scalars"].shape[0] for row in rows), default=0)
                    for row in rows:
                        if row["scalars"].shape[0] < max_len:
                            pad_width = max_len - row["scalars"].shape[0]
                            row["scalars"] = np.pad(
                                row["scalars"], (0, pad_width), mode="constant"
                            )

                    # Get predictions
                    preds = predict_records(pipe, labels, rows)

                    # Write enriched records
                    for i, rec in enumerate(batch_records):
                        pred_cls = int(preds["y_pred"][i])
                        probs_row = (
                            preds["probs"][i] if preds["probs"] is not None else None
                        )
                        scores_row = (
                            preds["scores"][i].tolist()
                            if preds["scores"] is not None
                            else None
                        )

                        enriched = enrich_record_with_prediction(
                            rec,
                            pred_cls,
                            probs_row,
                            np.array(scores_row) if scores_row is not None else None,
                            labels,
                            model_tag,
                        )

                        # Add interpretability info if available
                        if "enhanced_features" in rows[i]:
                            # Add top risk features
                            risk_features = _extract_top_risk_features(
                                rows[i]["enhanced_features"]
                            )
                            enriched[f"{model_tag}_risk_features"] = risk_features

                        fout.write(json.dumps(enriched) + "\n")

        except Exception as exc:
            logger.error(f"Failed processing {in_path}: {exc}")
            continue

    logger.info(f"Processed {total_files} files, {total_records} records")
    return Path(output_root)


def _extract_top_risk_features(
    enhanced_features: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Extract top risk-indicating features for interpretability."""
    # Focus on risk lexicon features
    risk_features = []

    for feature, value in enhanced_features.items():
        if value > 0 and any(
            risk_type in feature
            for risk_type in [
                "deception",
                "guarantee",
                "pricing",
                "scienter",
                "superlatives",
            ]
        ):
            risk_features.append(
                {
                    "feature": feature,
                    "value": float(value),
                }
            )

    # Sort by value and return top 5
    risk_features.sort(key=itemgetter("value"), reverse=True)
    return risk_features[:5]
