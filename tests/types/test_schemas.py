"""
Tests for corpus-types schema models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from corp_speech_risk_dataset.corpus_types.schemas.models import (
    Doc,
    Quote,
    Outcome,
    QuoteFeatures,
    CaseVector,
    APIConfig,
    QuoteCandidate,
    QuoteRow,
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


class TestDoc:
    """Test Doc model."""

    def test_doc_basic(self):
        """Test basic Doc creation."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import (
            RequestProv,
            ResponseProv,
            AdapterProv,
            Provenance,
        )

        retrieved_at = datetime.now()
        doc = Doc(
            doc_id="doc_abc123",
            source_uri="https://example.com/doc.pdf",
            raw_text="This is document text",
            provenance=Provenance(
                source="courtlistener",
                source_uri="https://example.com/doc.pdf",
                retrieved_at=retrieved_at,
                request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
                response=ResponseProv(
                    http_status=200,
                    sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                ),
                adapter=AdapterProv(name="corpus-hydrator", version="1.0.0"),
            ),
        )
        assert doc.doc_id == "doc_abc123"
        assert doc.source_uri == "https://example.com/doc.pdf"
        assert doc.raw_text == "This is document text"
        assert isinstance(doc.retrieved_at, datetime)
        assert doc.provenance.source == "courtlistener"

    def test_doc_with_meta(self):
        """Test Doc with metadata."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import (
            RequestProv,
            ResponseProv,
            AdapterProv,
            Provenance,
        )

        meta = {"court": "scotus", "docket": "123-456"}
        retrieved_at = datetime.now()
        doc = Doc(
            doc_id="doc_abc123",
            source_uri="https://example.com/doc.pdf",
            raw_text="This is document text",
            meta=meta,
            provenance=Provenance(
                source="courtlistener",
                source_uri="https://example.com/doc.pdf",
                retrieved_at=retrieved_at,
                request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
                response=ResponseProv(
                    http_status=200,
                    sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                ),
                adapter=AdapterProv(name="corpus-hydrator", version="1.0.0"),
            ),
        )
        assert doc.meta.court == "scotus"
        assert doc.meta.docket == "123-456"
        assert doc.meta.source is None
        assert doc.meta.party is None

    def test_doc_empty_doc_id(self):
        """Test Doc with empty doc_id raises error."""
        with pytest.raises(ValidationError):
            Doc(
                doc_id="",
                source_uri="https://example.com/doc.pdf",
                raw_text="This is document text",
            )

    def test_doc_empty_raw_text(self):
        """Test Doc with empty raw_text raises error."""
        with pytest.raises(ValidationError):
            Doc(
                doc_id="doc:abc123",
                source_uri="https://example.com/doc.pdf",
                raw_text="",
            )


class TestQuote:
    """Test Quote model."""

    def test_quote_basic(self):
        """Test basic Quote creation."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import Span

        quote = Quote(
            quote_id="q_abc123",
            doc_id="doc_def456",
            span=Span(start=100, end=150),
            text="This is a quote",
        )
        assert quote.quote_id == "q_abc123"
        assert quote.doc_id == "doc_def456"
        assert quote.span.start == 100
        assert quote.span.end == 150
        assert quote.text == "This is a quote"

    def test_quote_with_speaker(self):
        """Test Quote with speaker."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import Span

        quote = Quote(
            quote_id="q_abc123",
            doc_id="doc_def456",
            span=Span(start=100, end=150),
            text="This is a quote",
            speaker="John Doe",
        )
        assert quote.speaker == "John Doe"

    def test_quote_empty_quote_id(self):
        """Test Quote with empty quote_id raises error."""
        with pytest.raises(ValidationError):
            Quote(
                quote_id="",
                doc_id="doc:def456",
                span=(100, 150),
                text="This is a quote",
            )

    def test_quote_empty_doc_id(self):
        """Test Quote with empty doc_id raises error."""
        with pytest.raises(ValidationError):
            Quote(
                quote_id="quote:abc123",
                doc_id="",
                span=(100, 150),
                text="This is a quote",
            )

    def test_quote_invalid_span(self):
        """Test Quote with invalid span raises error."""
        # Wrong tuple length
        with pytest.raises(ValidationError):
            Quote(
                quote_id="quote:abc123",
                doc_id="doc:def456",
                span=(100,),
                text="This is a quote",
            )

        # Negative start
        with pytest.raises(ValidationError):
            Quote(
                quote_id="quote:abc123",
                doc_id="doc:def456",
                span=(-1, 150),
                text="This is a quote",
            )

        # End before start
        with pytest.raises(ValidationError):
            Quote(
                quote_id="quote:abc123",
                doc_id="doc:def456",
                span=(150, 100),
                text="This is a quote",
            )


class TestOutcome:
    """Test Outcome model."""

    def test_outcome_basic(self):
        """Test basic Outcome creation."""
        outcome = Outcome(case_id="case_abc123", label="win", label_source="manual")
        assert outcome.case_id == "case_abc123"
        assert outcome.label == "win"
        assert outcome.label_source == "manual"

    def test_outcome_with_date(self):
        """Test Outcome with date."""
        date = datetime(2024, 1, 1)
        outcome = Outcome(
            case_id="case_abc123", label="win", label_source="manual", date=date
        )
        assert outcome.date == date

    def test_outcome_empty_case_id(self):
        """Test Outcome with empty case_id raises error."""
        with pytest.raises(ValidationError):
            Outcome(case_id="", label="won", label_source="manual_review")

    def test_outcome_empty_label(self):
        """Test Outcome with empty label raises error."""
        with pytest.raises(ValidationError):
            Outcome(case_id="case:abc123", label="", label_source="manual_review")


class TestQuoteFeatures:
    """Test QuoteFeatures model."""

    def test_quote_features_basic(self):
        """Test basic QuoteFeatures creation."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import Producer

        features = QuoteFeatures(
            quote_id="q_abc123",
            feature_version="1.0.0",
            vector=[0.1, 0.2, 0.3],
            producer=Producer(name="corpus-features", version="1.0.0"),
        )
        assert features.quote_id == "q_abc123"
        assert features.feature_version == "1.0.0"
        assert features.vector == [0.1, 0.2, 0.3]
        assert features.producer.name == "corpus-features"

    def test_quote_features_with_interpretable(self):
        """Test QuoteFeatures with interpretable features."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import Producer

        interpretable = {"sentiment": 0.8, "confidence": 0.9}
        features = QuoteFeatures(
            quote_id="q_abc123",
            feature_version="1.0.0",
            vector=[0.1, 0.2, 0.3],
            interpretable=interpretable,
            producer=Producer(name="corpus-features", version="1.0.0"),
        )
        assert features.interpretable == interpretable

    def test_quote_features_empty_quote_id(self):
        """Test QuoteFeatures with empty quote_id raises error."""
        with pytest.raises(ValidationError):
            QuoteFeatures(quote_id="", feature_version="1.0.0", vector=[0.1, 0.2, 0.3])

    def test_quote_features_empty_vector(self):
        """Test QuoteFeatures with empty vector raises error."""
        with pytest.raises(ValidationError):
            QuoteFeatures(quote_id="quote:abc123", feature_version="1.0.0", vector=[])


class TestCaseVector:
    """Test CaseVector model."""

    def test_case_vector_basic(self):
        """Test basic CaseVector creation."""
        from corp_speech_risk_dataset.corpus_types.schemas.models import Producer

        case_vector = CaseVector(
            case_id="case_abc123",
            agg_version="1.0.0",
            stats={"mean": 0.5, "q90": 0.8},
            producer=Producer(name="corpus-aggregator", version="1.0.0"),
        )
        assert case_vector.case_id == "case_abc123"
        assert case_vector.agg_version == "1.0.0"
        assert case_vector.stats == {"mean": 0.5, "q90": 0.8}
        assert case_vector.producer.name == "corpus-aggregator"

    def test_case_vector_empty_case_id(self):
        """Test CaseVector with empty case_id raises error."""
        with pytest.raises(ValidationError):
            CaseVector(case_id="", agg_version="1.0.0", stats={"mean": 0.5})


class TestQuoteCandidate:
    """Test QuoteCandidate model (legacy compatibility)."""

    def test_quote_candidate_basic(self):
        """Test basic QuoteCandidate creation."""
        candidate = QuoteCandidate(
            quote="This is a quote",
            context="This is context",
            speaker="John Doe",
            score=0.9,
        )
        assert candidate.quote == "This is a quote"
        assert candidate.context == "This is context"
        assert candidate.speaker == "John Doe"
        assert candidate.score == 0.9

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
            doc_id="doc:abc123",
            stage=1,
            text="This is a quote",
            sp_ids=[1, 2, 3],
            deps=[(0, 1, "nsubj"), (1, 2, "dobj")],
            wl_indices=[10, 20, 30],
            wl_counts=[1, 2, 1],
        )
        assert row.doc_id == "doc:abc123"
        assert row.stage == 1
        assert row.text == "This is a quote"
        assert row.sp_ids == [1, 2, 3]
        assert row.deps == [(0, 1, "nsubj"), (1, 2, "dobj")]
        assert row.wl_indices == [10, 20, 30]
        assert row.wl_counts == [1, 2, 1]
