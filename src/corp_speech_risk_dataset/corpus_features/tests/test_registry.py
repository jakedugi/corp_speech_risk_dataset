"""Tests for the feature registry."""

import pytest
from ..registry import FeatureRegistry


class TestFeatureRegistry:
    """Test the FeatureRegistry class."""

    def test_get_extractor_v1(self):
        """Test getting v1 extractor."""
        registry = FeatureRegistry()
        extractor = registry.get_extractor("v1")
        assert isinstance(extractor, list)
        assert len(extractor) > 0

    def test_get_extractor_v2(self):
        """Test getting v2 extractor."""
        registry = FeatureRegistry()
        extractor = registry.get_extractor("v2")
        assert extractor is not None

    def test_get_unknown_version(self):
        """Test getting unknown version raises error."""
        registry = FeatureRegistry()
        with pytest.raises(ValueError, match="Unknown feature version"):
            registry.get_extractor("v999")

    def test_get_version_info(self):
        """Test getting version info."""
        registry = FeatureRegistry()
        info = registry.get_version_info("v1")
        assert "description" in info
        assert "extractor_class" in info

    def test_validate_features_v1(self):
        """Test validating v1 features."""
        registry = FeatureRegistry()

        # Valid v1 features
        features = {
            "quote_id": "q_123",
            "compound": 0.5,
            "neg": 0.1,
            "neu": 0.7,
            "pos": 0.2,
        }
        assert registry.validate_features(features, "v1")

    def test_validate_features_v2(self):
        """Test validating v2 features."""
        registry = FeatureRegistry()

        # Valid v2 features
        features = {
            "quote_id": "q_123",
            "vector": [0.1] * 768,  # 768-dim vector for BERT
            "interpretable": {
                "sentiment": 0.8,
                "confidence": 0.9,
            },
        }
        assert registry.validate_features(features, "v2")

    def test_validate_invalid_features(self):
        """Test validating invalid features."""
        registry = FeatureRegistry()

        # Invalid v1 features (missing required interpretable fields)
        features = {
            "quote_id": "q_123",
            "compound": 0.5,
            # Missing neg, neu, pos
        }
        assert not registry.validate_features(features, "v1")

    def test_list_versions(self):
        """Test listing available versions."""
        registry = FeatureRegistry()
        versions = registry.list_versions()
        assert "v1" in versions
        assert "v2" in versions
        assert isinstance(versions["v1"], str)
