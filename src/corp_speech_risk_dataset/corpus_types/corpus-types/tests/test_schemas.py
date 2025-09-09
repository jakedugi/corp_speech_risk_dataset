"""
Tests for corpus-types schema models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

import sys
from pathlib import Path

# Add the main project src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from corp_speech_risk_dataset.types.schemas.models import (
    Doc,
    Quote,
    Outcome,
    QuoteFeatures,
    CaseVector,
    Prediction,
    CasePrediction,
    APIConfig,
    QuoteCandidate,
    QuoteRow,
    Span,
    Meta,
    StrictBase,
    ExtensibleBase,
)


class TestBaseTypes:
    """Test base types and utilities."""

    def test_span_validation(self):
        """Test Span model validation."""
        # Valid span
        span = Span(start=10, end=20)
        assert span.start == 10
        assert span.end == 20

        # Invalid span (end before start)
        with pytest.raises(ValidationError):
            Span(start=20, end=10)

        # Invalid span (negative start)
        with pytest.raises(ValidationError):
            Span(start=-1, end=10)

    def test_meta_extensible(self):
        """Test Meta model allows extra fields."""
        meta = Meta(court="N.D.Cal", docket="1:24-cv-00001", custom_field="value")
        assert meta.court == "N.D.Cal"
        assert hasattr(meta, "custom_field")
        assert getattr(meta, "custom_field") == "value"

    def test_strict_base_forbids_extra(self):
        """Test StrictBase forbids extra fields."""

        class TestModel(StrictBase):
            name: str

        # Valid
        model = TestModel(name="test")
        assert model.name == "test"

        # Invalid - extra field
        with pytest.raises(ValidationError):
            TestModel(name="test", extra="field")

    def test_extensible_base_allows_extra(self):
        """Test ExtensibleBase allows extra fields."""

        class TestModel(ExtensibleBase):
            name: str

        # Valid with extra
        model = TestModel(name="test", extra="field")
        assert model.name == "test"
        assert hasattr(model, "extra")
        assert getattr(model, "extra") == "field"


class TestDoc:
    """Test Doc model."""

    def test_doc_basic(self):
        """Test basic Doc creation."""
        doc = Doc(
            doc_id="doc_AAAAAAAAAAAAAAAAAAAAAA",
            source_uri="https://example.com/doc.pdf",
            raw_text="This is document text",
            meta=Meta(court="N.D.Cal", docket="1:24-cv-00001"),
        )
        assert doc.doc_id == "doc_AAAAAAAAAAAAAAAAAAAAAA"
        assert doc.source_uri == "https://example.com/doc.pdf"
        assert doc.raw_text == "This is document text"
        assert doc.schema_version == "1.0"
        assert isinstance(doc.retrieved_at, datetime)
        assert doc.meta.court == "N.D.Cal"

    def test_doc_validation(self):
        """Test Doc validation."""
        # Empty doc_id
        with pytest.raises(ValidationError):
            Doc(
                doc_id="",
                source_uri="https://example.com/doc.pdf",
                raw_text="This is document text",
            )

        # Empty raw_text
        with pytest.raises(ValidationError):
            Doc(
                doc_id="doc_AAAAAAAAAAAAAAAAAAAAAA",
                source_uri="https://example.com/doc.pdf",
                raw_text="",
            )


class TestQuote:
    """Test Quote model."""

    def test_quote_basic(self):
        """Test basic Quote creation."""
        quote = Quote(
            quote_id="q_BBBBBBBBBBBBBBBBBBBBB",
            doc_id="doc_AAAAAAAAAAAAAAAAAAAAAA",
            span=Span(start=10, end=20),
            text="This is a quote",
            speaker="John Doe",
        )
        assert quote.quote_id == "q_BBBBBBBBBBBBBBBBBBBBB"
        assert quote.doc_id == "doc_AAAAAAAAAAAAAAAAAAAAAA"
        assert quote.span.start == 10
        assert quote.span.end == 20
        assert quote.text == "This is a quote"
        assert quote.speaker == "John Doe"
        assert quote.schema_version == "1.0"

    def test_quote_validation(self):
        """Test Quote validation."""
        # Empty quote_id
        with pytest.raises(ValidationError):
            Quote(
                quote_id="",
                doc_id="doc_AAAAAAAAAAAAAAAAAAAAAA",
                span=Span(start=10, end=20),
                text="This is a quote",
            )

        # Empty doc_id
        with pytest.raises(ValidationError):
            Quote(
                quote_id="q_BBBBBBBBBBBBBBBBBBBBB",
                doc_id="",
                span=Span(start=10, end=20),
                text="This is a quote",
            )


class TestOutcome:
    """Test Outcome model."""

    def test_outcome_basic(self):
        """Test basic Outcome creation."""
        outcome = Outcome(
            case_id="case_CCCCCCCCCCCCCCCCCCCCC",
            label="settlement",
            label_source="manual",
            date=datetime(2024, 3, 31),
            meta=Meta(court="N.D.Cal", docket="1:24-cv-00001"),
        )
        assert outcome.case_id == "case_CCCCCCCCCCCCCCCCCCCCC"
        assert outcome.label == "settlement"
        assert outcome.label_source == "manual"
        assert outcome.date == datetime(2024, 3, 31)
        assert outcome.schema_version == "1.0"

    def test_outcome_validation(self):
        """Test Outcome validation."""
        # Empty case_id
        with pytest.raises(ValidationError):
            Outcome(case_id="", label="settlement", label_source="manual")

        # Invalid label
        with pytest.raises(ValidationError):
            Outcome(
                case_id="case_CCCCCCCCCCCCCCCCCCCCC",
                label="invalid_label",
                label_source="manual",
            )


class TestQuoteFeatures:
    """Test QuoteFeatures model."""

    def test_quote_features_basic(self):
        """Test basic QuoteFeatures creation."""
        features = QuoteFeatures(
            quote_id="q_BBBBBBBBBBBBBBBBBBBBB",
            feature_version="v1.0.0",
            vector=[0.1, 0.2, 0.3],
            interpretable={"sentiment": 0.8, "confidence": 0.9},
        )
        assert features.quote_id == "q_BBBBBBBBBBBBBBBBBBBBB"
        assert features.feature_version == "v1.0.0"
        assert features.vector == [0.1, 0.2, 0.3]
        assert features.interpretable == {"sentiment": 0.8, "confidence": 0.9}
        assert features.schema_version == "1.0"

    def test_quote_features_validation(self):
        """Test QuoteFeatures validation."""
        # Empty quote_id
        with pytest.raises(ValidationError):
            QuoteFeatures(quote_id="", feature_version="v1.0.0", vector=[0.1, 0.2, 0.3])

        # Empty vector
        with pytest.raises(ValidationError):
            QuoteFeatures(
                quote_id="q_BBBBBBBBBBBBBBBBBBBBB", feature_version="v1.0.0", vector=[]
            )


class TestCaseVector:
    """Test CaseVector model."""

    def test_case_vector_basic(self):
        """Test basic CaseVector creation."""
        case_vector = CaseVector(
            case_id="case_CCCCCCCCCCCCCCCCCCCCC",
            agg_version="v1.0.0",
            stats={"mean": 0.5, "q90": 0.8},
            vector=[0.1, 0.2, 0.3],
        )
        assert case_vector.case_id == "case_CCCCCCCCCCCCCCCCCCCCC"
        assert case_vector.agg_version == "v1.0.0"
        assert case_vector.stats == {"mean": 0.5, "q90": 0.8}
        assert case_vector.vector == [0.1, 0.2, 0.3]
        assert case_vector.schema_version == "1.0"

    def test_case_vector_validation(self):
        """Test CaseVector validation."""
        # Empty case_id
        with pytest.raises(ValidationError):
            CaseVector(case_id="", agg_version="v1.0.0", stats={"mean": 0.5})


class TestPrediction:
    """Test Prediction model."""

    def test_prediction_basic(self):
        """Test basic Prediction creation."""
        prediction = Prediction(
            model_version="lr_v1.0",
            target="case/binary/settlement",
            split_id="temporal_2018_2020_fold3",
            case_id="case_CCCCCCCCCCCCCCCCCCCCC",
            proba=0.75,
            pred="settlement",
            calibrated=True,
        )
        assert prediction.model_version == "lr_v1.0"
        assert prediction.target == "case/binary/settlement"
        assert prediction.split_id == "temporal_2018_2020_fold3"
        assert prediction.case_id == "case_CCCCCCCCCCCCCCCCCCCCC"
        assert prediction.proba == 0.75
        assert prediction.pred == "settlement"
        assert prediction.calibrated == True
        assert prediction.schema_version == "1.0"

    def test_prediction_validation(self):
        """Test Prediction validation."""
        # Probability out of range
        with pytest.raises(ValidationError):
            Prediction(
                model_version="lr_v1.0",
                target="case/binary/settlement",
                split_id="temporal_2018_2020_fold3",
                case_id="case_CCCCCCCCCCCCCCCCCCCCC",
                proba=1.5,  # Invalid
                pred="settlement",
            )

        # Missing both quote_id and case_id
        with pytest.raises(ValidationError):
            Prediction(
                model_version="lr_v1.0",
                target="case/binary/settlement",
                split_id="temporal_2018_2020_fold3",
                proba=0.75,
                pred="settlement",
            )


class TestCasePrediction:
    """Test CasePrediction model."""

    def test_case_prediction_basic(self):
        """Test basic CasePrediction creation."""
        prediction = CasePrediction(
            model_version="lr_v1.0",
            target="case/binary/settlement",
            split_id="temporal_2018_2020_fold3",
            case_id="case_CCCCCCCCCCCCCCCCCCCCC",
            proba=0.75,
            pred="settlement",
            calibrated=True,
        )
        assert prediction.model_version == "lr_v1.0"
        assert prediction.target == "case/binary/settlement"
        assert prediction.split_id == "temporal_2018_2020_fold3"
        assert prediction.case_id == "case_CCCCCCCCCCCCCCCCCCCCC"
        assert prediction.proba == 0.75
        assert prediction.pred == "settlement"
        assert prediction.calibrated == True
        assert prediction.schema_version == "1.0"

    def test_case_prediction_validation(self):
        """Test CasePrediction validation."""
        # Empty case_id
        with pytest.raises(ValidationError):
            CasePrediction(
                model_version="lr_v1.0",
                target="case/binary/settlement",
                split_id="temporal_2018_2020_fold3",
                case_id="",
                proba=0.75,
                pred="settlement",
            )


class TestAPIConfig:
    """Test APIConfig model."""

    def test_api_config_basic(self):
        """Test basic APIConfig creation."""
        config = APIConfig(api_token="test_token", rate_limit=1.0)
        assert config.api_token == "test_token"
        assert config.rate_limit == 1.0
        assert config.api_key == "test_token"  # Test alias

    def test_api_config_defaults(self):
        """Test APIConfig with defaults."""
        config = APIConfig()
        assert config.api_token is None
        assert config.rate_limit == 0.25

    def test_api_config_extensible(self):
        """Test APIConfig allows extra fields."""
        config = APIConfig(api_token="test", custom_field="value")
        assert config.api_token == "test"
        assert hasattr(config, "custom_field")
        assert getattr(config, "custom_field") == "value"


class TestQuoteCandidate:
    """Test QuoteCandidate model (legacy compatibility)."""

    def test_quote_candidate_basic(self):
        """Test basic QuoteCandidate creation."""
        candidate = QuoteCandidate(
            quote="This is a quote",
            context="This is context",
            speaker="John Doe",
            score=0.9,
            urls=["https://example.com"],
        )
        assert candidate.quote == "This is a quote"
        assert candidate.context == "This is context"
        assert candidate.speaker == "John Doe"
        assert candidate.score == 0.9
        assert candidate.urls == ["https://example.com"]

    def test_quote_candidate_to_dict(self):
        """Test QuoteCandidate to_dict method."""
        candidate = QuoteCandidate(
            quote="This is a quote",
            context="This is context",
            speaker="John Doe",
            score=0.9,
            urls=["https://example.com"],
        )
        expected = {
            "text": "This is a quote",
            "speaker": "John Doe",
            "score": 0.9,
            "urls": ["https://example.com"],
            "context": "This is context",
        }
        assert candidate.to_dict() == expected


class TestQuoteRow:
    """Test QuoteRow model (legacy compatibility)."""

    def test_quote_row_basic(self):
        """Test basic QuoteRow creation."""
        row = QuoteRow(
            doc_id="doc_AAAAAAAAAAAAAAAAAAAAAA",
            stage=1,
            text="This is a quote",
            sp_ids=[1, 2, 3],
            deps=[(0, 1, "nsubj"), (1, 2, "dobj")],
            wl_indices=[10, 20, 30],
            wl_counts=[1, 2, 1],
        )
        assert row.doc_id == "doc_AAAAAAAAAAAAAAAAAAAAAA"
        assert row.stage == 1
        assert row.text == "This is a quote"
        assert row.sp_ids == [1, 2, 3]
        assert row.deps == [(0, 1, "nsubj"), (1, 2, "dobj")]
        assert row.wl_indices == [10, 20, 30]
        assert row.wl_counts == [1, 2, 1]
