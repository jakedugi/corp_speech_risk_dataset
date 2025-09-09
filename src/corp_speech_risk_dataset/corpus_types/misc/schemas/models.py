"""
Pydantic models for corpus-types data contracts.

This module defines the authoritative data models used throughout the corpus processing pipeline.
These models serve as contracts for data exchange between different modules and provide
validation and serialization capabilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal, Union
from pydantic import BaseModel, Field, validator, ConfigDict


# --------------------------------------------------------------------------- #
# Schema Version and Base Types                                              #
# --------------------------------------------------------------------------- #

SchemaVersion = Literal["1.0"]


class StrictBase(BaseModel):
    """Base class for strict models that forbid extra fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExtensibleBase(BaseModel):
    """Base class for extensible models that allow extra fields."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class Meta(ExtensibleBase):
    """Metadata container for various entities."""

    source: Optional[str] = None
    court: Optional[str] = None
    docket: Optional[str] = None
    party: Optional[str] = None


# --------------------------------------------------------------------------- #
# Provenance Types                                                           #
# --------------------------------------------------------------------------- #


class RequestProv(StrictBase):
    """Request provenance information."""

    endpoint: str = Field(..., description="API endpoint used")
    params_hash: str = Field(..., description="Hash of normalized request parameters")


class ResponseProv(StrictBase):
    """Response provenance information."""

    http_status: Optional[int] = Field(None, description="HTTP status code")
    sha256: str = Field(..., description="SHA256 checksum of response payload")
    bytes: Optional[int] = Field(None, description="Response size in bytes")
    content_type: Optional[str] = Field(None, description="Response content type")

    @validator("sha256")
    def validate_sha256_format(cls, v: str) -> str:
        """Validate SHA256 is 64 hex characters."""
        import re

        if not re.match(r"^[a-f0-9]{64}$", v):
            raise ValueError("sha256 must be a valid 64-character hexadecimal string")
        return v

    @validator("http_status")
    def validate_http_status(cls, v: Optional[int]) -> Optional[int]:
        """Validate HTTP status code is in valid range."""
        if v is not None and not (100 <= v <= 599):
            raise ValueError("http_status must be between 100 and 599")
        return v


class AdapterProv(StrictBase):
    """Adapter provenance information."""

    name: Literal["corpus-hydrator"] = Field(..., description="Adapter name")
    version: str = Field(..., description="Adapter version")
    git_sha: Optional[str] = Field(None, description="Git SHA of adapter")


class Producer(StrictBase):
    """Producer information for derived artifacts."""

    name: Literal[
        "corpus-cleaner",
        "corpus-extractors",
        "corpus-features",
        "corpus-aggregator",
        "corpus-temporal-cv",
    ] = Field(..., description="Producer name")
    version: str = Field(..., description="Producer version")
    git_sha: Optional[str] = Field(None, description="Git SHA of producer")
    params_hash: Optional[str] = Field(None, description="Hash of producer parameters")
    run_id: Optional[str] = Field(None, description="Unique run identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Production timestamp"
    )


class CourtListenerProv(StrictBase):
    """CourtListener-specific provenance information."""

    source: Literal["courtlistener"] = Field(
        default="courtlistener", description="Source type"
    )
    opinion_id: Optional[int] = Field(None, description="CourtListener opinion ID")
    cluster_id: Optional[int] = Field(None, description="CourtListener cluster ID")
    docket_id: Optional[int] = Field(None, description="CourtListener docket ID")
    absolute_url: Optional[str] = Field(
        None, description="Absolute URL to document page"
    )
    download_url: Optional[str] = Field(None, description="Download URL for document")
    citation: Optional[str] = Field(None, description="Primary citation")
    docket_number: Optional[str] = Field(None, description="Docket number")
    md5: Optional[str] = Field(None, description="MD5 checksum if provided")
    sha1: Optional[str] = Field(None, description="SHA1 checksum if provided")


class Provenance(StrictBase):
    """Complete provenance information for a document."""

    # Generic fields (always present)
    source: Literal["courtlistener", "pacer", "scrape", "manual"] = Field(
        ..., description="Data source"
    )
    source_uri: str = Field(..., description="Canonical source URI")
    retrieved_at: datetime = Field(..., description="When document was retrieved")
    request: RequestProv = Field(..., description="Request provenance")
    response: ResponseProv = Field(..., description="Response provenance")
    adapter: AdapterProv = Field(..., description="Adapter provenance")

    # Optional generic fields
    api_version: Optional[str] = Field(None, description="Upstream API version")
    license: Optional[str] = Field(None, description="Data license")

    # Provider-specific fields (namespaced)
    provider: Optional[CourtListenerProv] = Field(
        None, description="Provider-specific provenance"
    )

    @validator("source")
    def validate_source_matches_provider(cls, v: str, values: Dict[str, Any]) -> str:
        """Ensure provider type matches source when provider is present."""
        if "provider" in values and values["provider"] is not None:
            if v == "courtlistener" and values["provider"].source != "courtlistener":
                raise ValueError("Provider source must match document source")
        return v

    @validator("provider")
    def validate_provider_consistency(
        cls, v: Optional[CourtListenerProv], values: Dict[str, Any]
    ) -> Optional[CourtListenerProv]:
        """Validate provider-specific requirements."""
        if v is None:
            return v

        # For CourtListener, ensure at least one ID is present
        if v.source == "courtlistener":
            has_id = any(
                [
                    v.opinion_id is not None,
                    v.cluster_id is not None,
                    v.docket_id is not None,
                ]
            )
            if not has_id:
                raise ValueError(
                    "CourtListener provider must have at least one of: opinion_id, cluster_id, docket_id"
                )

        return v


class Span(StrictBase):
    """Text span with start/end positions."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)  # exclusive
    # Optional mapping when normalization changes offsets
    offset_map_id: Optional[str] = None

    @validator("end")
    def validate_end_after_start(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure end >= start."""
        if "start" in values and v < values["start"]:
            raise ValueError("end must be >= start")
        return v


# --------------------------------------------------------------------------- #
# Base Configuration Types                                                    #
# --------------------------------------------------------------------------- #


class APIConfig(ExtensibleBase):
    """Generic API configuration container."""

    api_token: Optional[str] = None
    rate_limit: float = 0.25

    @property
    def api_key(self) -> Optional[str]:
        """Alias for api_token for backward compatibility."""
        return self.api_token


# --------------------------------------------------------------------------- #
# Document Types                                                              #
# --------------------------------------------------------------------------- #


class Doc(StrictBase):
    """
    Document model representing a raw document from any source.

    This is the primary data contract for documents fetched from APIs or scraped
    from web sources. It contains all the raw text and metadata needed for
    downstream processing.
    """

    schema_version: SchemaVersion = "1.0"
    doc_id: str = Field(..., description="Unique document identifier")
    source_uri: str = Field(..., description="Original source URI")
    retrieved_at: datetime = Field(
        default_factory=datetime.now, description="When document was retrieved"
    )
    raw_text: str = Field(..., description="Raw document text content")
    meta: Meta = Field(default_factory=Meta, description="Document metadata")
    provenance: Provenance = Field(..., description="Complete provenance information")

    @validator("doc_id")
    def validate_doc_id(cls, v: str) -> str:
        """Validate that doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()

    @validator("raw_text")
    def validate_raw_text(cls, v: str) -> str:
        """Validate that raw_text is not empty."""
        if not v or not v.strip():
            raise ValueError("raw_text cannot be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Quote Types                                                                 #
# --------------------------------------------------------------------------- #


class Quote(ExtensibleBase):
    """
    Quote model representing an extracted and processed quote from a document.

    This comprehensive model captures all the fields used in the actual pipeline,
    including processing stages, embeddings, features, and predictions.
    """

    schema_version: SchemaVersion = "1.0"
    quote_id: Optional[str] = Field(None, description="Unique quote identifier")
    doc_id: str = Field(..., description="Document this quote belongs to")

    # Core quote fields
    text: str = Field(..., description="Quote text content")
    context: Optional[str] = Field(None, description="Surrounding context")
    speaker: Optional[str] = Field(None, description="Speaker of the quote")
    score: Optional[float] = Field(None, description="Confidence/extraction score")
    urls: List[str] = Field(default_factory=list, description="Source URLs")

    # Processing fields
    stage: Optional[int] = Field(None, description="Processing stage")
    span: Optional[Span] = Field(None, description="Text span with start/end positions")

    # Encoding outputs
    sp_ids: List[int] = Field(default_factory=list, description="SentencePiece IDs")
    deps: List[Tuple[int, int, str]] = Field(
        default_factory=list, description="Dependency edges"
    )
    byte_fallback: Optional[bool] = Field(
        None, description="Whether byte fallback was used"
    )
    fallback_chars: List[str] = Field(
        default_factory=list, description="Fallback characters"
    )

    # Embeddings and features
    legal_bert_emb: List[float] = Field(
        default_factory=list, description="Legal BERT embeddings"
    )
    legal_bert_quote_top_keywords_emb: List[float] = Field(
        default_factory=list, description="Quote keyword embeddings"
    )
    legal_bert_context_top_keywords_emb: List[float] = Field(
        default_factory=list, description="Context keyword embeddings"
    )
    legal_bert_speaker_emb: List[float] = Field(
        default_factory=list, description="Speaker embeddings"
    )
    legal_bert_section_headers_emb: List[float] = Field(
        default_factory=list, description="Section header embeddings"
    )
    fused_emb: List[float] = Field(default_factory=list, description="Fused embeddings")
    gph_emb: List[float] = Field(default_factory=list, description="Graph embeddings")

    # Processing metadata
    gph_method: Optional[str] = Field(None, description="Graph processing method")
    text_embedder: Optional[str] = Field(None, description="Text embedder used")
    raw_features: Dict[str, Any] = Field(
        default_factory=dict, description="Raw extracted features"
    )

    # Prediction results
    final_judgement_real: Optional[float] = Field(
        None, description="Final judgment value"
    )
    coral_pred_bucket: Optional[str] = Field(
        None, description="Coral prediction bucket"
    )
    coral_pred_class: Optional[int] = Field(None, description="Coral prediction class")
    coral_confidence: Optional[float] = Field(
        None, description="Coral prediction confidence"
    )
    coral_class_probs: Dict[str, float] = Field(
        default_factory=dict, description="Coral class probabilities"
    )
    coral_scores: List[float] = Field(
        default_factory=list, description="Coral prediction scores"
    )
    coral_prob_low: Optional[float] = Field(
        None, description="Probability of low outcome"
    )
    coral_prob_medium: Optional[float] = Field(
        None, description="Probability of medium outcome"
    )
    coral_prob_high: Optional[float] = Field(
        None, description="Probability of high outcome"
    )
    coral_model_threshold: Optional[float] = Field(
        None, description="Coral model threshold"
    )
    coral_model_buckets: List[str] = Field(
        default_factory=list, description="Coral model buckets"
    )

    # Position and token information
    docket_number: Optional[float] = Field(None, description="Docket number")
    docket_token_start: Optional[float] = Field(
        None, description="Token start position in docket"
    )
    global_token_start: Optional[float] = Field(
        None, description="Global token start position"
    )
    docket_char_start: Optional[float] = Field(
        None, description="Character start in docket"
    )
    global_char_start: Optional[float] = Field(
        None, description="Global character start"
    )
    num_tokens: Optional[float] = Field(None, description="Number of tokens")

    # Case and metadata
    case_id: Optional[str] = Field(None, description="Case identifier")
    case_id_clean: Optional[str] = Field(None, description="Clean case identifier")
    case_year: Optional[int] = Field(None, description="Case year")

    # Hash and identifiers
    text_hash: Optional[str] = Field(None, description="Text hash")
    text_hash_norm: Optional[str] = Field(None, description="Normalized text hash")
    record_id: Optional[int] = Field(None, description="Record identifier")

    # Training metadata
    fold: Optional[int] = Field(None, description="Cross-validation fold")
    split: Optional[str] = Field(None, description="Train/validation/test split")
    outcome_bin: Optional[int] = Field(None, description="Binary outcome")
    support_tertile: Optional[float] = Field(None, description="Support tertile")
    bin_weight: Optional[float] = Field(None, description="Binary weight")
    support_weight: Optional[float] = Field(None, description="Support weight")
    sample_weight: Optional[float] = Field(None, description="Sample weight")

    # Metadata and evidence
    evidence_links: List[str] = Field(
        default_factory=list, description="Links to supporting evidence"
    )
    meta: Meta = Field(default_factory=Meta, description="Quote metadata")

    # Internal metadata fields (for data processing pipeline)
    _src: Optional[str] = Field(None, description="Source path")
    _metadata_src_path: Optional[str] = Field(None, description="Metadata source path")
    _metadata_wrapped: Optional[bool] = Field(
        None, description="Whether metadata is wrapped"
    )
    _leakage_prevention: Dict[str, Any] = Field(
        default_factory=dict, description="Leakage prevention metadata"
    )

    @validator("quote_id")
    def validate_quote_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate that quote_id is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("quote_id cannot be empty if provided")
        return v.strip() if v else None

    @validator("doc_id")
    def validate_doc_id(cls, v: str) -> str:
        """Validate that doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()

    @validator("text")
    def validate_text(cls, v: str) -> str:
        """Validate that text is not empty."""
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Outcome Types                                                               #
# --------------------------------------------------------------------------- #


class Outcome(StrictBase):
    """
    Outcome model representing case outcomes and labels.

    This model captures the labeled outcomes for legal cases, including
    the source of the label and any additional metadata.
    """

    schema_version: SchemaVersion = "1.0"
    case_id: str = Field(..., description="Unique case identifier")
    label: Literal["win", "loss", "settlement", "dismissal", "mixed", "unknown"] = (
        Field(..., description="Outcome label")
    )
    label_source: Literal["manual", "heuristic", "external", "inferred"] = Field(
        ..., description="Source of the label"
    )
    date: Optional[datetime] = Field(None, description="Date of the outcome")
    meta: Meta = Field(default_factory=Meta, description="Outcome metadata")

    @validator("case_id")
    def validate_case_id(cls, v: str) -> str:
        """Validate that case_id is not empty."""
        if not v or not v.strip():
            raise ValueError("case_id cannot be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Prediction Types                                                           #
# --------------------------------------------------------------------------- #


class Prediction(ExtensibleBase):
    """
    Prediction model for quote-level or case-level predictions.

    This model captures predictions from machine learning models, including
    probabilities, predicted classes, and calibration information.
    """

    schema_version: SchemaVersion = "1.0"
    model_version: str = Field(..., description="Version/fingerprint of the model")
    target: str = Field(
        ..., description="Prediction target (e.g., 'case/binary/settlement')"
    )
    split_id: str = Field(..., description="Cross-validation split identifier")
    quote_id: Optional[str] = Field(
        None, description="Quote ID for quote-level predictions"
    )
    case_id: Optional[str] = Field(
        None, description="Case ID for case-level predictions"
    )
    proba: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    pred: Union[str, int] = Field(..., description="Predicted class/label")
    calibrated: bool = Field(False, description="Whether prediction is calibrated")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )
    producer: Producer = Field(..., description="Who produced this prediction")

    @validator("proba")
    def validate_proba(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("proba must be between 0.0 and 1.0")
        return v

    @validator("case_id")
    def validate_at_least_one_id(
        cls, v: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        """Ensure at least one of quote_id or case_id is provided."""
        if "quote_id" in values and values["quote_id"] is None and v is None:
            raise ValueError("Either quote_id or case_id must be provided")
        return v


class CasePrediction(ExtensibleBase):
    """
    Case-level prediction model.

    This is a specialized prediction model for case-level outcomes that
    includes additional case-specific metadata.
    """

    schema_version: SchemaVersion = "1.0"
    model_version: str = Field(..., description="Version/fingerprint of the model")
    target: str = Field(..., description="Prediction target")
    split_id: str = Field(..., description="Cross-validation split identifier")
    case_id: str = Field(..., description="Case ID")
    proba: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    pred: Union[str, int] = Field(..., description="Predicted class/label")
    calibrated: bool = Field(False, description="Whether prediction is calibrated")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )
    meta: Meta = Field(default_factory=Meta, description="Prediction metadata")
    producer: Producer = Field(..., description="Who produced this case prediction")

    @validator("case_id")
    def validate_case_id(cls, v: str) -> str:
        """Validate that case_id is not empty."""
        if not v or not v.strip():
            raise ValueError("case_id cannot be empty")
        return v.strip()

    @validator("proba")
    def validate_proba(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("proba must be between 0.0 and 1.0")
        return v


# --------------------------------------------------------------------------- #
# Feature Types                                                               #
# --------------------------------------------------------------------------- #


class QuoteFeatures(ExtensibleBase):
    """
    Quote features model containing extracted features for a quote.

    This model captures the feature vector and interpretable features
    extracted from a quote for machine learning purposes.
    """

    schema_version: SchemaVersion = "1.0"
    feature_version: str = Field(
        ..., description="Version of feature extraction pipeline"
    )
    quote_id: str = Field(..., description="Quote this feature vector belongs to")
    vector: List[float] = Field(..., description="Feature vector")
    interpretable: Dict[str, float] = Field(
        default_factory=dict, description="Human-interpretable features"
    )
    producer: Producer = Field(..., description="Who produced these features")

    @validator("quote_id")
    def validate_quote_id(cls, v: str) -> str:
        """Validate that quote_id is not empty."""
        if not v or not v.strip():
            raise ValueError("quote_id cannot be empty")
        return v.strip()

    @validator("vector")
    def validate_vector(cls, v: List[float]) -> List[float]:
        """Validate vector is not empty."""
        if not v:
            raise ValueError("vector cannot be empty")
        return v


class CaseVector(ExtensibleBase):
    """
    Case vector model containing aggregated features for a case.

    This model captures the aggregated features and statistics for an entire
    legal case, computed from all quotes within the case.
    """

    schema_version: SchemaVersion = "1.0"
    agg_version: str = Field(..., description="Version of aggregation pipeline")
    case_id: str = Field(..., description="Case this vector belongs to")
    stats: Dict[str, float] = Field(..., description="Aggregated statistics")
    vector: List[float] = Field(
        default_factory=list, description="Aggregated feature vector"
    )
    producer: Producer = Field(..., description="Who produced this case vector")

    @validator("case_id")
    def validate_case_id(cls, v: str) -> str:
        """Validate that case_id is not empty."""
        if not v or not v.strip():
            raise ValueError("case_id cannot be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Legacy Types (for backward compatibility)                                   #
# --------------------------------------------------------------------------- #


class QuoteCandidate(ExtensibleBase):
    """
    Legacy quote candidate model for backward compatibility.

    This model represents a potential quote during the extraction process.
    """

    quote: str = Field(..., description="Quote text")
    context: str = Field(..., description="Surrounding context")
    urls: List[str] = Field(default_factory=list, description="Source URLs")
    speaker: Optional[str] = Field(None, description="Detected speaker")
    score: float = Field(default=0.0, description="Confidence score")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.quote,
            "speaker": self.speaker,
            "score": self.score,
            "urls": self.urls,
            "context": self.context,
        }


class QuoteRow(ExtensibleBase):
    """
    Legacy quote row model for backward compatibility.

    This model represents a processed quote with encoding outputs and
    should be aligned with the actual data structure used in the pipeline.
    """

    # Core fields
    doc_id: str = Field(..., description="Document identifier")
    stage: int = Field(..., description="Processing stage")
    text: str = Field(..., description="Quote text")
    context: Optional[str] = Field(None, description="Surrounding context")
    speaker: Optional[str] = Field(None, description="Speaker")
    score: Optional[float] = Field(None, description="Confidence score")
    urls: List[str] = Field(default_factory=list, description="Source URLs")

    # Encoding outputs
    sp_ids: List[int] = Field(default_factory=list, description="SentencePiece IDs")
    deps: List[Tuple[int, int, str]] = Field(
        default_factory=list, description="Dependency edges"
    )
    wl_indices: List[int] = Field(
        default_factory=list, description="WL feature vector indices"
    )
    wl_counts: List[int] = Field(
        default_factory=list, description="WL feature vector counts"
    )

    # Additional fields from actual data
    byte_fallback: Optional[bool] = Field(None, description="Byte fallback flag")
    fallback_chars: List[str] = Field(
        default_factory=list, description="Fallback characters"
    )
    gph_method: Optional[str] = Field(None, description="Graph processing method")

    # Internal metadata
    _src: Optional[str] = Field(None, description="Source path")

    @validator("doc_id")
    def validate_doc_id(cls, v: str) -> str:
        """Validate that doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()

    @validator("text")
    def validate_text(cls, v: str) -> str:
        """Validate that text is not empty."""
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()
