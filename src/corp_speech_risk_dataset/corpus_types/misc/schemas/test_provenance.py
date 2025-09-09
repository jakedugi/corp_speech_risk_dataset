"""Tests for provenance models and functionality."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from corp_speech_risk_dataset.types.schemas.models import (
    RequestProv,
    ResponseProv,
    AdapterProv,
    Producer,
    CourtListenerProv,
    Provenance,
    Doc,
    QuoteFeatures,
    CaseVector,
    Prediction,
    CasePrediction,
)


class TestRequestProv:
    """Test RequestProv model."""

    def test_valid_request_prov(self):
        """Test creating a valid RequestProv."""
        req = RequestProv(endpoint="/api/opinions", params_hash="abc123def456")
        assert req.endpoint == "/api/opinions"
        assert req.params_hash == "abc123def456"

    def test_request_prov_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            RequestProv(endpoint="/api/opinions")  # missing params_hash

        with pytest.raises(ValidationError):
            RequestProv(params_hash="abc123")  # missing endpoint


class TestResponseProv:
    """Test ResponseProv model."""

    def test_valid_response_prov(self):
        """Test creating a valid ResponseProv."""
        resp = ResponseProv(
            http_status=200,
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            bytes=1024,
            content_type="application/json",
        )
        assert resp.http_status == 200
        assert (
            resp.sha256
            == "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
        )
        assert resp.bytes == 1024
        assert resp.content_type == "application/json"

    def test_sha256_validation(self):
        """Test SHA256 format validation."""
        # Valid 64-char hex
        ResponseProv(
            http_status=200,
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        )

        # Invalid: too short
        with pytest.raises(ValidationError):
            ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae",
            )

        # Invalid: non-hex characters
        with pytest.raises(ValidationError):
            ResponseProv(
                http_status=200,
                sha256="zzzzz45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            )

    def test_http_status_validation(self):
        """Test HTTP status code validation."""
        # Valid codes
        ResponseProv(
            http_status=200,
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        )
        ResponseProv(
            http_status=404,
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        )
        ResponseProv(
            http_status=500,
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        )

        # Invalid: out of range
        with pytest.raises(ValidationError):
            ResponseProv(
                http_status=99,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            )

        with pytest.raises(ValidationError):
            ResponseProv(
                http_status=600,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            )

    def test_response_prov_optional_fields(self):
        """Test that some fields are optional."""
        resp = ResponseProv(
            sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
        )
        assert resp.http_status is None
        assert resp.bytes is None
        assert resp.content_type is None


class TestAdapterProv:
    """Test AdapterProv model."""

    def test_valid_adapter_prov(self):
        """Test creating a valid AdapterProv."""
        adapter = AdapterProv(
            name="corpus-hydrator", version="1.2.3", git_sha="abc123def456"
        )
        assert adapter.name == "corpus-hydrator"
        assert adapter.version == "1.2.3"
        assert adapter.git_sha == "abc123def456"

    def test_adapter_prov_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            AdapterProv(name="corpus-hydrator")  # missing version

        with pytest.raises(ValidationError):
            AdapterProv(version="1.2.3")  # missing name

    def test_adapter_prov_optional_fields(self):
        """Test that git_sha is optional."""
        adapter = AdapterProv(name="corpus-hydrator", version="1.2.3")
        assert adapter.git_sha is None


class TestProducer:
    """Test Producer model."""

    def test_valid_producer(self):
        """Test creating a valid Producer."""
        producer = Producer(
            name="corpus-extractors",
            version="2.1.0",
            git_sha="def789ghi012",
            params_hash="param123hash",
            run_id="run_20250101_001",
        )
        assert producer.name == "corpus-extractors"
        assert producer.version == "2.1.0"
        assert producer.git_sha == "def789ghi012"
        assert producer.params_hash == "param123hash"
        assert producer.run_id == "run_20250101_001"
        assert isinstance(producer.timestamp, datetime)

    def test_producer_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            Producer(name="corpus-extractors")  # missing version

        with pytest.raises(ValidationError):
            Producer(version="2.1.0")  # missing name

    def test_producer_name_validation(self):
        """Test that producer name is from allowed values."""
        # Valid names
        Producer(name="corpus-cleaner", version="1.0.0")
        Producer(name="corpus-extractors", version="1.0.0")
        Producer(name="corpus-features", version="1.0.0")
        Producer(name="corpus-aggregator", version="1.0.0")
        Producer(name="corpus-temporal-cv", version="1.0.0")

        # Invalid name
        with pytest.raises(ValidationError):
            Producer(name="invalid-module", version="1.0.0")


class TestCourtListenerProv:
    """Test CourtListenerProv model."""

    def test_valid_courtlistener_prov(self):
        """Test creating a valid CourtListenerProv."""
        cl_prov = CourtListenerProv(
            opinion_id=12345,
            cluster_id=67890,
            docket_id=11111,
            absolute_url="https://www.courtlistener.com/docket/11111/test-case/",
            download_url="https://www.courtlistener.com/pdf/12345/download/",
            citation="123 F. Supp. 3d 456",
            docket_number="1:19-cv-02184",
            md5="abc123def456",
            sha1="def789ghi012",
        )
        assert cl_prov.source == "courtlistener"
        assert cl_prov.opinion_id == 12345
        assert cl_prov.cluster_id == 67890
        assert cl_prov.docket_id == 11111
        assert (
            cl_prov.absolute_url
            == "https://www.courtlistener.com/docket/11111/test-case/"
        )
        assert (
            cl_prov.download_url == "https://www.courtlistener.com/pdf/12345/download/"
        )
        assert cl_prov.citation == "123 F. Supp. 3d 456"
        assert cl_prov.docket_number == "1:19-cv-02184"
        assert cl_prov.md5 == "abc123def456"
        assert cl_prov.sha1 == "def789ghi012"

    def test_courtlistener_prov_all_optional(self):
        """Test that all fields except source are optional."""
        cl_prov = CourtListenerProv()
        assert cl_prov.source == "courtlistener"
        assert cl_prov.opinion_id is None
        assert cl_prov.cluster_id is None
        assert cl_prov.docket_id is None
        assert cl_prov.absolute_url is None
        assert cl_prov.download_url is None
        assert cl_prov.citation is None
        assert cl_prov.docket_number is None
        assert cl_prov.md5 is None
        assert cl_prov.sha1 is None


class TestProvenance:
    """Test Provenance model."""

    def test_valid_provenance_minimal(self):
        """Test creating a minimal valid Provenance."""
        retrieved_at = datetime.now()

        prov = Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(
                endpoint="/api/rest/v3/opinions/12345/", params_hash="hash123"
            ),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
        )

        assert prov.source == "courtlistener"
        assert (
            prov.source_uri
            == "https://www.courtlistener.com/api/rest/v3/opinions/12345/"
        )
        assert prov.retrieved_at == retrieved_at
        assert prov.api_version is None
        assert prov.license is None
        assert prov.provider is None

    def test_valid_provenance_with_provider(self):
        """Test creating Provenance with CourtListener provider."""
        retrieved_at = datetime.now()

        prov = Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(
                endpoint="/api/rest/v3/opinions/12345/", params_hash="hash123"
            ),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
            api_version="v3",
            license="public-domain",
            provider=CourtListenerProv(
                opinion_id=12345, cluster_id=67890, docket_id=11111
            ),
        )

        assert prov.source == "courtlistener"
        assert prov.api_version == "v3"
        assert prov.license == "public-domain"
        assert prov.provider is not None
        assert prov.provider.opinion_id == 12345
        assert prov.provider.cluster_id == 67890
        assert prov.provider.docket_id == 11111

    def test_provenance_provider_validation(self):
        """Test that CourtListener provider requires at least one ID."""
        retrieved_at = datetime.now()

        # Valid: has opinion_id
        Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
            provider=CourtListenerProv(opinion_id=12345),
        )

        # Valid: has cluster_id
        Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
            provider=CourtListenerProv(cluster_id=67890),
        )

        # Valid: has docket_id
        Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
            provider=CourtListenerProv(docket_id=11111),
        )

        # Invalid: CourtListener provider with no IDs
        with pytest.raises(ValidationError):
            Provenance(
                source="courtlistener",
                source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
                retrieved_at=retrieved_at,
                request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
                response=ResponseProv(
                    http_status=200,
                    sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                ),
                adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
                provider=CourtListenerProv(),  # No IDs
            )

    def test_provenance_source_validation(self):
        """Test that source matches provider type."""
        retrieved_at = datetime.now()

        # Valid: courtlistener source with courtlistener provider
        Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
            provider=CourtListenerProv(opinion_id=12345),
        )

        # Valid: no provider (None is allowed)
        Provenance(
            source="courtlistener",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            request=RequestProv(endpoint="/api/opinions", params_hash="hash123"),
            response=ResponseProv(
                http_status=200,
                sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            ),
            adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
        )


class TestDocWithProvenance:
    """Test Doc model with provenance."""

    def test_doc_with_provenance(self):
        """Test creating a Doc with provenance."""
        retrieved_at = datetime.now()

        doc = Doc(
            doc_id="doc_12345",
            source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
            retrieved_at=retrieved_at,
            raw_text="This is a test document.",
            meta={"court": "Test Court", "docket": "1:19-cv-02184"},
            provenance=Provenance(
                source="courtlistener",
                source_uri="https://www.courtlistener.com/api/rest/v3/opinions/12345/",
                retrieved_at=retrieved_at,
                request=RequestProv(
                    endpoint="/api/rest/v3/opinions/12345/", params_hash="hash123"
                ),
                response=ResponseProv(
                    http_status=200,
                    sha256="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                ),
                adapter=AdapterProv(name="corpus-hydrator", version="1.2.3"),
                provider=CourtListenerProv(
                    opinion_id=12345, cluster_id=67890, docket_id=11111
                ),
            ),
        )

        assert doc.doc_id == "doc_12345"
        assert doc.provenance.source == "courtlistener"
        assert doc.provenance.provider.opinion_id == 12345
        assert doc.provenance.provider.cluster_id == 67890
        assert doc.provenance.provider.docket_id == 11111


class TestDerivedArtifactsWithProducer:
    """Test derived artifacts with producer information."""

    def test_quote_features_with_producer(self):
        """Test QuoteFeatures with producer."""
        features = QuoteFeatures(
            feature_version="1.0.0",
            quote_id="q_123",
            vector=[0.1, 0.2, 0.3],
            interpretable={"sentiment": 0.8, "confidence": 0.9},
            producer=Producer(name="corpus-features", version="1.1.0"),
        )

        assert features.quote_id == "q_123"
        assert features.producer.name == "corpus-features"
        assert features.producer.version == "1.1.0"

    def test_case_vector_with_producer(self):
        """Test CaseVector with producer."""
        case_vector = CaseVector(
            agg_version="2.0.0",
            case_id="case_456",
            stats={"mean": 0.5, "std": 0.2},
            vector=[0.4, 0.6, 0.8],
            producer=Producer(name="corpus-aggregator", version="2.1.0"),
        )

        assert case_vector.case_id == "case_456"
        assert case_vector.producer.name == "corpus-aggregator"
        assert case_vector.producer.version == "2.1.0"

    def test_prediction_with_producer(self):
        """Test Prediction with producer."""
        prediction = Prediction(
            model_version="3.0.0",
            target="case/binary/settlement",
            split_id="fold_0",
            case_id="case_789",
            proba=0.75,
            pred="settlement",
            producer=Producer(name="corpus-temporal-cv", version="3.1.0"),
        )

        assert prediction.case_id == "case_789"
        assert prediction.proba == 0.75
        assert prediction.producer.name == "corpus-temporal-cv"
        assert prediction.producer.version == "3.1.0"

    def test_case_prediction_with_producer(self):
        """Test CasePrediction with producer."""
        case_pred = CasePrediction(
            model_version="3.0.0",
            target="case/binary/settlement",
            split_id="fold_0",
            case_id="case_101",
            proba=0.85,
            pred="win",
            producer=Producer(name="corpus-temporal-cv", version="3.1.0"),
        )

        assert case_pred.case_id == "case_101"
        assert case_pred.proba == 0.85
        assert case_pred.producer.name == "corpus-temporal-cv"
        assert case_pred.producer.version == "3.1.0"
