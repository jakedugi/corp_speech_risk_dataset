"""
JSON Schema export utilities for corpus-types.

This module provides utilities for generating JSON Schemas from Pydantic models
and exporting them to files for use by other systems.
"""

import json
from pathlib import Path
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel

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
)


def export_model_schema(
    model_cls: Type[BaseModel], version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Export a Pydantic model's JSON schema with additional metadata.

    Args:
        model_cls: The Pydantic model class to export
        version: Schema version string

    Returns:
        JSON schema dictionary with corpus-types metadata
    """
    schema = model_cls.schema()

    # Add corpus-types specific metadata
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://corpus-types.example.com/schemas/{model_cls.__name__.lower()}.json",
            "title": model_cls.__name__,
            "description": (
                model_cls.__doc__.strip()
                if model_cls.__doc__
                else f"{model_cls.__name__} model"
            ),
            "version": version,
            "corpus_types": {
                "model_type": model_cls.__name__,
                "module": "corp_speech_risk_dataset.types.schemas",
                "generated_at": "2024-01-01T00:00:00Z",  # Would be dynamic in real implementation
                "pydantic_version": "1.x",
            },
        }
    )

    return schema


def export_all_schemas(output_dir: Path, version: str = "1.0.0") -> None:
    """
    Export all corpus-types schemas to JSON files.

    Args:
        output_dir: Directory to write schema files
        version: Schema version to use
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Core models to export
    models = [
        (Doc, "doc"),
        (Quote, "quote"),
        (Outcome, "outcome"),
        (QuoteFeatures, "quote_features"),
        (CaseVector, "case_vector"),
        (Prediction, "prediction"),
        (CasePrediction, "case_prediction"),
        (APIConfig, "api_config"),
        (QuoteCandidate, "quote_candidate"),
        (QuoteRow, "quote_row"),
    ]

    for model_cls, filename_prefix in models:
        schema = export_model_schema(model_cls, version)

        output_file = output_dir / f"{filename_prefix}.schema.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        print(f"Exported schema: {output_file}")


def validate_against_schema(data: Dict[str, Any], model_cls: Type[BaseModel]) -> bool:
    """
    Validate data against a Pydantic model's schema.

    Args:
        data: Data dictionary to validate
        model_cls: Pydantic model class to validate against

    Returns:
        True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    # This will raise ValidationError if invalid
    model_cls(**data)
    return True


def get_model_by_name(name: str) -> Optional[Type[BaseModel]]:
    """
    Get a Pydantic model class by name.

    Args:
        name: Model name (e.g., 'Doc', 'Quote', 'Outcome')

    Returns:
        Model class or None if not found
    """
    models = {
        "Doc": Doc,
        "Quote": Quote,
        "Outcome": Outcome,
        "QuoteFeatures": QuoteFeatures,
        "CaseVector": CaseVector,
        "Prediction": Prediction,
        "CasePrediction": CasePrediction,
        "APIConfig": APIConfig,
        "QuoteCandidate": QuoteCandidate,
        "QuoteRow": QuoteRow,
    }

    return models.get(name)


if __name__ == "__main__":
    # Example usage: export all schemas
    output_dir = Path(__file__).parent.parent.parent / "schemas"
    export_all_schemas(output_dir)
