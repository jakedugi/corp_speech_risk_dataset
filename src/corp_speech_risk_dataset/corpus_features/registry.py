"""
Feature registry for corpus-features module.

Manages different feature versions and their configurations.
"""

from typing import Dict, Any, Callable
from .features_case_agnostic import (
    extract_sentiment,
    extract_evidential_lexicons,
    embed_text,
)
from .encoders.legal_bert_embedder import LegalBertEmbedder


class FeatureRegistry:
    """Registry of available feature extractors by version."""

    def __init__(self):
        self._extractors: Dict[str, Dict[str, Any]] = {
            "v1": {
                "extractor_funcs": [extract_sentiment, extract_evidential_lexicons],
                "description": "Basic interpretable features",
                "vector_dim": None,  # Variable dimension
                "required_interpretable": ["compound", "neg", "neu", "pos"],
            },
            "v2": {
                "extractor_class": LegalBertEmbedder,
                "description": "Legal BERT embeddings with interpretable features",
                "vector_dim": 768,  # BERT base dimension
                "required_interpretable": ["sentiment", "confidence"],
            },
        }

    def get_extractor(self, version: str) -> Any:
        """Get feature extractor for the specified version."""
        if version not in self._extractors:
            raise ValueError(f"Unknown feature version: {version}")

        config = self._extractors[version]
        if "extractor_class" in config:
            extractor_class = config["extractor_class"]
            return extractor_class()
        elif "extractor_funcs" in config:
            return config["extractor_funcs"]
        else:
            raise ValueError(f"No extractor defined for version {version}")

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get metadata for a feature version."""
        if version not in self._extractors:
            raise ValueError(f"Unknown feature version: {version}")

        return self._extractors[version].copy()

    def validate_features(self, features: Dict[str, Any], version: str) -> bool:
        """Validate feature output against version requirements."""
        if version not in self._extractors:
            return False

        version_info = self._extractors[version]

        # Check vector dimension if specified
        if version_info["vector_dim"] is not None:
            if (
                "vector" not in features
                or len(features["vector"]) != version_info["vector_dim"]
            ):
                return False

        # Check required interpretable fields
        if "interpretable" in features:
            interpretable = features["interpretable"]
            for required_key in version_info["required_interpretable"]:
                if required_key not in interpretable:
                    return False

        return True

    def list_versions(self) -> Dict[str, str]:
        """List all available feature versions with descriptions."""
        return {
            version: info["description"] for version, info in self._extractors.items()
        }


# Global registry instance
registry = FeatureRegistry()
