"""
Pydantic models for corpus-types data contracts.

This module defines the authoritative data models used throughout the corpus processing pipeline.
These models serve as contracts for data exchange between different modules and provide
validation and serialization capabilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


# --------------------------------------------------------------------------- #
# Schema Version and Base Types                                              #
# --------------------------------------------------------------------------- #

SchemaVersion = Literal["1.0"]


class StrictBase(BaseModel):
    """Base class for strict models that forbid extra fields."""

    class Config:
        extra = "forbid"
        validate_assignment = True


class ExtensibleBase(BaseModel):
    """Base class for extensible models that allow extra fields."""

    class Config:
        extra = "allow"
        validate_assignment = True


class Meta(ExtensibleBase):
    """Metadata container for various entities."""

    source: Optional[str] = None
    court: Optional[str] = None
    docket: Optional[str] = None
    party: Optional[str] = None


class Span(StrictBase):
    """Text span with start/end positions."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)  # exclusive
    # Optional mapping when normalization changes offsets
    offset_map_id: Optional[str] = None

    @validator("end")
    def validate_end_after_start(cls, v, values):
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

    @validator("doc_id")
    def validate_doc_id(cls, v):
        """Validate that doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()

    @validator("raw_text")
    def validate_raw_text(cls, v):
        """Validate that raw_text is not empty."""
        if not v or not v.strip():
            raise ValueError("raw_text cannot be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Quote Types                                                                 #
# --------------------------------------------------------------------------- #


class Quote(StrictBase):
    """
    Quote model representing an extracted quote from a document.

    This model captures quotes extracted from legal documents with their
    context, speaker information, and evidence links.
    """

    schema_version: SchemaVersion = "1.0"
    quote_id: str = Field(..., description="Unique quote identifier")
    doc_id: str = Field(..., description="Document this quote belongs to")
    span: Span = Field(..., description="Text span with start/end positions")
    text: str = Field(..., description="Quote text content")
    speaker: Optional[str] = Field(None, description="Speaker of the quote")
    evidence_links: List[str] = Field(
        default_factory=list, description="Links to supporting evidence"
    )
    meta: Meta = Field(default_factory=Meta, description="Quote metadata")

    @validator("quote_id")
    def validate_quote_id(cls, v):
        """Validate that quote_id is not empty."""
        if not v or not v.strip():
            raise ValueError("quote_id cannot be empty")
        return v.strip()

    @validator("doc_id")
    def validate_doc_id(cls, v):
        """Validate that doc_id is not empty."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
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
    def validate_case_id(cls, v):
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

    @validator("proba")
    def validate_proba(cls, v):
        """Validate probability is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("proba must be between 0.0 and 1.0")
        return v

    @validator("case_id")
    def validate_at_least_one_id(cls, v, values):
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

    @validator("case_id")
    def validate_case_id(cls, v):
        """Validate that case_id is not empty."""
        if not v or not v.strip():
            raise ValueError("case_id cannot be empty")
        return v.strip()

    @validator("proba")
    def validate_proba(cls, v):
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

    @validator("quote_id")
    def validate_quote_id(cls, v):
        """Validate that quote_id is not empty."""
        if not v or not v.strip():
            raise ValueError("quote_id cannot be empty")
        return v.strip()

    @validator("vector")
    def validate_vector(cls, v):
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

    @validator("case_id")
    def validate_case_id(cls, v):
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

    def to_dict(self) -> dict:
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

    This model represents a processed quote with encoding outputs.
    """

    # Original metadata fields
    doc_id: str
    stage: int
    text: str
    context: Optional[str] = None
    speaker: Optional[str] = None
    score: Optional[float] = None
    urls: List[str] = []
    _src: Optional[str] = None

    # Encoding outputs
    sp_ids: List[int]  # SentencePiece IDs (loss-less)
    deps: List[Tuple[int, int, str]]  # Dependency edges (head, child, label)
    wl_indices: List[int]  # CSR indices of WL feature vector
    wl_counts: List[int]  # CSR data of WL feature vector
