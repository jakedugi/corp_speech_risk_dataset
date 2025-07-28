#!/usr/bin/env python3
"""
Test voting weights system for case outcome imputation.
"""

import pytest
from pathlib import Path

from src.corp_speech_risk_dataset.case_outcome.extract_cash_amounts_stage1 import (
    VotingWeights,
    DEFAULT_VOTING_WEIGHTS,
    compute_feature_votes,
    compute_enhanced_feature_votes,
    compute_enhanced_feature_votes_with_titles,
)


class TestVotingWeights:
    """Test the voting weights system."""

    def test_default_weights(self):
        """Test that default weights are all 1.0."""
        weights = VotingWeights()
        assert weights.proximity_pattern_weight == 1.0
        assert weights.judgment_verbs_weight == 1.0
        assert weights.case_position_weight == 1.0
        assert weights.docket_position_weight == 1.0
        assert weights.all_caps_titles_weight == 1.0
        assert weights.document_titles_weight == 1.0

    def test_custom_weights(self):
        """Test that custom weights are properly set."""
        weights = VotingWeights(
            proximity_pattern_weight=2.0,
            judgment_verbs_weight=3.0,
            case_position_weight=1.5,
            docket_position_weight=0.5,
            all_caps_titles_weight=4.0,
            document_titles_weight=2.5,
        )
        assert weights.proximity_pattern_weight == 2.0
        assert weights.judgment_verbs_weight == 3.0
        assert weights.case_position_weight == 1.5
        assert weights.docket_position_weight == 0.5
        assert weights.all_caps_titles_weight == 4.0
        assert weights.document_titles_weight == 2.5

    def test_serialization(self):
        """Test that weights can be serialized and deserialized."""
        original = VotingWeights(
            proximity_pattern_weight=2.0,
            judgment_verbs_weight=3.0,
            case_position_weight=1.5,
            docket_position_weight=0.5,
            all_caps_titles_weight=4.0,
            document_titles_weight=2.5,
        )

        # Convert to dict and back
        weights_dict = original.to_dict()
        restored = VotingWeights.from_dict(weights_dict)

        assert restored.proximity_pattern_weight == original.proximity_pattern_weight
        assert restored.judgment_verbs_weight == original.judgment_verbs_weight
        assert restored.case_position_weight == original.case_position_weight
        assert restored.docket_position_weight == original.docket_position_weight
        assert restored.all_caps_titles_weight == original.all_caps_titles_weight
        assert restored.document_titles_weight == original.document_titles_weight


class TestVotingFunctions:
    """Test the voting functions with different weights."""

    def test_compute_feature_votes_with_weights(self):
        """Test that feature votes are properly weighted."""
        # Test context with both proximity patterns and judgment verbs
        context = "The court awarded damages of $1,000,000 in settlement of the claim."

        # Default weights (all 1.0)
        default_votes = compute_feature_votes(context, DEFAULT_VOTING_WEIGHTS)

        # Custom weights (emphasize judgment verbs)
        custom_weights = VotingWeights(
            proximity_pattern_weight=1.0,
            judgment_verbs_weight=5.0,  # 5x weight for judgment verbs
        )
        custom_votes = compute_feature_votes(context, custom_weights)

        # Should have more votes with higher judgment verbs weight
        assert custom_votes > default_votes

    def test_compute_feature_votes_proximity_emphasis(self):
        """Test emphasizing proximity patterns."""
        context = "The settlement amount was $500,000 with additional damages."

        # Default weights
        default_votes = compute_feature_votes(context, DEFAULT_VOTING_WEIGHTS)

        # Emphasize proximity patterns
        proximity_weights = VotingWeights(
            proximity_pattern_weight=3.0,  # 3x weight for proximity patterns
            judgment_verbs_weight=1.0,
        )
        proximity_votes = compute_feature_votes(context, proximity_weights)

        # Should have more votes with higher proximity pattern weight
        assert proximity_votes > default_votes

    def test_compute_feature_votes_zero_weights(self):
        """Test that zero weights result in zero votes for that component."""
        context = "The court awarded damages of $1,000,000 in settlement."

        # Zero out judgment verbs weight
        zero_judgment_weights = VotingWeights(
            proximity_pattern_weight=1.0,
            judgment_verbs_weight=0.0,  # Zero weight for judgment verbs
        )
        votes = compute_feature_votes(context, zero_judgment_weights)

        # Should still have votes from proximity patterns
        assert votes > 0

        # Zero out proximity patterns weight
        zero_proximity_weights = VotingWeights(
            proximity_pattern_weight=0.0,  # Zero weight for proximity patterns
            judgment_verbs_weight=1.0,
        )
        votes = compute_feature_votes(context, zero_proximity_weights)

        # Should still have votes from judgment verbs
        assert votes > 0


if __name__ == "__main__":
    pytest.main([__file__])
