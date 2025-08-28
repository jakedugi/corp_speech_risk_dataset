"""Binary-specific column governance for POLR interpretable models.

This module enforces a strict feature policy for binary classification using exactly 10 features
from the feature importance analysis.
"""

from typing import List, Dict, Any
import re
import numpy as np
from loguru import logger

# ---------------- Binary-specific feature whitelist ----------------
# Exactly 10 features from feature importance analysis
BINARY_FEATURE_WHITELIST = {
    "feat_new_attribution_verb_density",
    "feat_new2_deception_cluster_density",
    "interpretable_lex_deception_count",
    "interpretable_seq_trans_neutral_to_neutral",
    "feat_new3_neutral_edge_coverage",
    "feat_new2_attribution_verb_clustering_score",
    "feat_new5_deesc_half_life",
    "feat_new2_attr_verb_near_neutral_transition",
    "feat_new2_neutral_run_mean",
    "feat_new2_neutral_to_deception_transition_rate",
}

# Metadata always permitted (not used as features directly)
META_KEYS = {
    "case_id",
    "quote_id",
    "doc_id",
    "_src",
    "timestamp",
    "case_time",
    "final_judgement_real",
    "outcome_bin",  # Binary target
    "sample_weight",  # Pre-computed weights
    "bin_weight",
    "support_weight",
    "support_tertile",
    "fold",
    "split",
    "record_id",
    "case_year",
}


def validate_binary_columns(df_columns: List[str]) -> Dict[str, Any]:
    """Validate columns for binary classification pipeline.

    Allowed feature columns are ONLY the 10 features from the whitelist.
    Everything else is treated as metadata or blocked.
    """

    # Find allowed features
    allowed_features = [c for c in df_columns if c in BINARY_FEATURE_WHITELIST]

    # Check if all 10 required features are present
    missing_features = BINARY_FEATURE_WHITELIST - set(allowed_features)
    if missing_features:
        logger.warning(f"Missing required binary features: {missing_features}")

    # Find extra features (not allowed, not metadata)
    extra_features = [
        c
        for c in df_columns
        if (c not in BINARY_FEATURE_WHITELIST) and (c not in META_KEYS)
    ]

    if extra_features:
        logger.info(
            f"Ignoring {len(extra_features)} extra features not in binary whitelist"
        )
        logger.debug(f"Extra features: {extra_features[:10]}...")  # Show first 10

    logger.info(f"Binary features found: {len(allowed_features)}/10 required")
    logger.info(f"Features: {sorted(allowed_features)}")

    return {
        "valid": len(allowed_features) > 0,
        "binary_features": allowed_features,
        "missing_features": list(missing_features),
        "extra_features": extra_features,
        "feature_count": len(allowed_features),
        "has_all_required": len(missing_features) == 0,
    }


def get_binary_feature_columns(df_columns: List[str]) -> List[str]:
    """Get only the allowed binary features from the column list."""
    return [c for c in df_columns if c in BINARY_FEATURE_WHITELIST]


def validate_binary_target(df_columns: List[str]) -> bool:
    """Validate that the binary target column is present."""
    return "outcome_bin" in df_columns


def validate_binary_weights(df_columns: List[str]) -> Dict[str, bool]:
    """Validate that required weight columns are present."""
    weight_columns = ["sample_weight", "bin_weight", "support_weight"]
    return {col: col in df_columns for col in weight_columns}


# Safe JSON helper (reused from original)
def pyify(o: Any) -> Any:
    """Convert numpy / non-JSON types to plain Python for json.dump."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)
            out[k] = pyify(v)
        return out
    if isinstance(o, (list, tuple, set)):
        return [pyify(x) for x in o]

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o
