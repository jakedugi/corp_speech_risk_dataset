"""Tests for case aggregation functionality."""

import numpy as np
import pandas as pd
import pytest
from ..modeling.dataset import FeatureConfig, ThresholdSpec, DEFAULT_THRESHOLDS


class TestAggregation:
    """Test case-level aggregation logic."""

    @pytest.fixture
    def sample_quote_features(self):
        """Create sample quote features for testing."""
        np.random.seed(42)

        # Create sample data for 2 cases with 3-5 quotes each
        data = []

        # Case 1
        for i in range(4):
            data.append(
                {
                    "case_id": "case_001",
                    "doc_id": f"doc_{i:03d}",
                    "text": f"Sample quote {i}",
                    "docket_number": 1 + i,
                    "global_token_start": i * 100,
                    "num_tokens": 50,
                    "mlp_probability": np.random.beta(
                        2, 5
                    ),  # Skewed toward lower values
                    "mlp_pred_strict": np.random.binomial(1, 0.3),
                    "mlp_pred_recallT": np.random.binomial(1, 0.5),
                    "outcome_bin": 0,  # Case outcome
                }
            )

        # Case 2
        for i in range(3):
            data.append(
                {
                    "case_id": "case_002",
                    "doc_id": f"doc_{i+10:03d}",
                    "text": f"Another quote {i}",
                    "docket_number": 1 + i,
                    "global_token_start": i * 80,
                    "num_tokens": 40,
                    "mlp_probability": np.random.beta(
                        5, 2
                    ),  # Skewed toward higher values
                    "mlp_pred_strict": np.random.binomial(1, 0.7),
                    "mlp_pred_recallT": np.random.binomial(1, 0.8),
                    "outcome_bin": 1,  # Case outcome
                }
            )

        return pd.DataFrame(data)

    def test_feature_config_initialization(self):
        """Test FeatureConfig initialization."""
        config = FeatureConfig()

        assert config.include_counts is False
        assert config.include_mean_probabilities is False

        # Test with specific settings
        config_full = FeatureConfig(
            include_counts=True,
            include_mean_probabilities=True,
            include_confidence=True,
        )

        assert config_full.include_counts is True
        assert config_full.include_mean_probabilities is True
        assert config_full.include_confidence is True

    def test_threshold_spec(self):
        """Test ThresholdSpec functionality."""
        threshold = ThresholdSpec(
            name="test_threshold", kind="token_budget", value=1000.0
        )

        assert threshold.name == "test_threshold"
        assert threshold.kind == "token_budget"
        assert threshold.value == 1000.0

    def test_default_thresholds(self):
        """Test that default thresholds are properly defined."""
        assert "token_2500" in DEFAULT_THRESHOLDS
        assert "token_half" in DEFAULT_THRESHOLDS
        assert "complete_case" in DEFAULT_THRESHOLDS

        # Check structure
        token_2500 = DEFAULT_THRESHOLDS["token_2500"]
        assert token_2500.kind == "token_budget"
        assert token_2500.value == 2500.0

    def test_case_feature_aggregation_basic(self, sample_quote_features):
        """Test basic case feature aggregation."""
        # Group by case_id and compute basic statistics
        case_features = []

        for case_id, group in sample_quote_features.groupby("case_id"):
            features = {
                "case_id": case_id,
                "n_quotes": len(group),
                "mean_probability": group["mlp_probability"].mean(),
                "std_probability": group["mlp_probability"].std(),
                "max_probability": group["mlp_probability"].max(),
                "prop_strict": group["mlp_pred_strict"].mean(),
                "prop_recallT": group["mlp_pred_recallT"].mean(),
                "outcome": group["outcome_bin"].iloc[0],  # Same for all quotes in case
            }
            case_features.append(features)

        result_df = pd.DataFrame(case_features)

        # Verify we have features for both cases
        assert len(result_df) == 2
        assert set(result_df["case_id"]) == {"case_001", "case_002"}

        # Verify basic statistics
        for _, row in result_df.iterrows():
            assert row["n_quotes"] > 0
            assert 0 <= row["mean_probability"] <= 1
            assert 0 <= row["prop_strict"] <= 1
            assert 0 <= row["prop_recallT"] <= 1

    def test_density_feature_computation(self, sample_quote_features):
        """Test density-based feature computation."""
        # This would test density features like proportion above thresholds
        case_features = []

        for case_id, group in sample_quote_features.groupby("case_id"):
            probs = group["mlp_probability"].values

            density_features = {
                "case_id": case_id,
                "prop_p80": np.mean(probs >= 0.80),
                "prop_p90": np.mean(probs >= 0.90),
                "prop_p95": np.mean(probs >= 0.95),
            }
            case_features.append(density_features)

        result_df = pd.DataFrame(case_features)

        # Verify density features are between 0 and 1
        for _, row in result_df.iterrows():
            assert 0 <= row["prop_p80"] <= 1
            assert 0 <= row["prop_p90"] <= 1
            assert 0 <= row["prop_p95"] <= 1

            # p95 should be <= p90 should be <= p80
            assert row["prop_p95"] <= row["prop_p90"] <= row["prop_p80"]

    def test_quantile_feature_computation(self, sample_quote_features):
        """Test quantile-based feature computation."""
        case_features = []

        for case_id, group in sample_quote_features.groupby("case_id"):
            probs = group["mlp_probability"].values

            if len(probs) >= 3:  # Need enough data for quantiles
                quantile_features = {
                    "case_id": case_id,
                    "p25_probability": np.percentile(probs, 25),
                    "p50_probability": np.percentile(probs, 50),
                    "p75_probability": np.percentile(probs, 75),
                    "p90_probability": np.percentile(probs, 90),
                }
                case_features.append(quantile_features)

        if case_features:  # Only if we have enough data
            result_df = pd.DataFrame(case_features)

            for _, row in result_df.iterrows():
                # Check quantile ordering
                assert (
                    row["p25_probability"]
                    <= row["p50_probability"]
                    <= row["p75_probability"]
                    <= row["p90_probability"]
                )
                # All should be between 0 and 1
                for col in [
                    "p25_probability",
                    "p50_probability",
                    "p75_probability",
                    "p90_probability",
                ]:
                    assert 0 <= row[col] <= 1
