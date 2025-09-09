#!/usr/bin/env python3
"""
Validation CLI for corpus-types.

This CLI provides commands for validating JSONL files against corpus-types schemas
and generating schema files.
"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional, Type
import orjson

from corp_speech_risk_dataset.types.schemas.models import (
    Doc,
    Quote,
    Outcome,
    QuoteFeatures,
    CaseVector,
    Prediction,
    CasePrediction,
)
from corp_speech_risk_dataset.types.utils.export_schema import (
    export_all_schemas,
    get_model_by_name,
)

app = typer.Typer(no_args_is_help=True, help="corpus-types validation CLI")
console = Console()

MODEL_MAP = {
    "Doc": Doc,
    "Quote": Quote,
    "Outcome": Outcome,
    "QuoteFeatures": QuoteFeatures,
    "CaseVector": CaseVector,
    "Prediction": Prediction,
    "CasePrediction": CasePrediction,
}


def _iter_jsonl(path: Path):
    """Iterate over JSONL file, yielding (line_num, data) tuples."""
    with path.open("rb") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield i, orjson.loads(line)
            except Exception as e:
                raise ValueError(f"Line {i}: invalid JSON: {e}")


@app.command("jsonl")
def validate_jsonl(
    schema: str = typer.Argument(
        ..., help="Doc|Quote|Outcome|QuoteFeatures|CaseVector|Prediction|CasePrediction"
    ),
    path: Path = typer.Argument(..., help="JSONL file to validate"),
    limit: int = typer.Option(0, help="Validate first N records (0=all)"),
    quiet: bool = typer.Option(False, help="Suppress success messages"),
) -> None:
    """
    Validate JSONL file against a corpus-types schema.

    Examples:
        corpus-validate jsonl Doc data/docs.jsonl
        corpus-validate jsonl Quote data/quotes.jsonl --limit 100
    """
    Model: Type = MODEL_MAP.get(schema)
    if not Model:
        console.print(f"[red]❌ Unknown schema: {schema}[/red]")
        console.print(f"[dim]Available schemas: {', '.join(MODEL_MAP.keys())}[/dim]")
        raise typer.Exit(2)

    errors = 0
    valid_count = 0

    try:
        for i, obj in _iter_jsonl(path):
            try:
                Model(**obj)  # Pydantic v1 style validation
                valid_count += 1
                if limit and valid_count >= limit:
                    break
            except Exception as e:
                console.print(f"[red]Line {i}: Validation error - {e}[/red]")
                errors += 1

    except FileNotFoundError:
        console.print(f"[red]❌ File not found: {path}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)

    if errors == 0:
        if not quiet:
            console.print(
                f"[green]✅ Validation successful: {valid_count} records validated as {schema}[/green]"
            )
    else:
        console.print(
            f"[red]❌ Validation failed: {errors} errors out of {valid_count + errors} records[/red]"
        )
        raise typer.Exit(1)


@app.command("parquet")
def validate_parquet(
    schema: str = typer.Argument(
        ..., help="Doc|Quote|Outcome|QuoteFeatures|CaseVector|Prediction|CasePrediction"
    ),
    path: Path = typer.Argument(..., help="Parquet file to validate"),
    limit: int = typer.Option(0, help="Validate first N records (0=all)"),
    quiet: bool = typer.Option(False, help="Suppress success messages"),
) -> None:
    """
    Validate Parquet file against a corpus-types schema.

    Examples:
        corpus-validate parquet QuoteFeatures data/features.parquet
    """
    try:
        import polars as pl
    except ImportError:
        console.print(
            "[red]❌ polars not installed. Install with: pip install polars[/red]"
        )
        raise typer.Exit(1)

    Model: Type = MODEL_MAP.get(schema)
    if not Model:
        console.print(f"[red]❌ Unknown schema: {schema}[/red]")
        console.print(f"[dim]Available schemas: {', '.join(MODEL_MAP.keys())}[/dim]")
        raise typer.Exit(2)

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        console.print(f"[red]❌ Failed to read parquet file: {e}[/red]")
        raise typer.Exit(1)

    errors = 0
    valid_count = 0

    for i, row in enumerate(df.iter_rows(named=True)):
        try:
            Model(**row)  # Pydantic v1 style validation
            valid_count += 1
            if limit and valid_count >= limit:
                break
        except Exception as e:
            console.print(f"[red]Row {i}: Validation error - {e}[/red]")
            errors += 1

    if errors == 0:
        if not quiet:
            console.print(
                f"[green]✅ Validation successful: {valid_count} records validated as {schema}[/green]"
            )
    else:
        console.print(
            f"[red]❌ Validation failed: {errors} errors out of {valid_count + errors} records[/red]"
        )
        raise typer.Exit(1)


@app.command("generate-schemas")
def generate_schemas(
    output_dir: Path = typer.Argument(..., help="Directory to write schema files"),
    version: str = typer.Option("1.0.0", help="Schema version"),
) -> None:
    """
    Generate JSON Schema files for all corpus-types models.

    Examples:
        corpus-validate generate-schemas schemas/
        corpus-validate generate-schemas schemas/ --version 1.1.0
    """
    console.print(f"[blue]Generating schemas to: {output_dir}[/blue]")
    export_all_schemas(output_dir, version)
    console.print("[green]✅ Schema generation complete[/green]")


@app.command("list-models")
def list_models() -> None:
    """List all available models and their descriptions."""
    console.print("[bold]Available corpus-types models:[/bold]")
    descriptions = {
        "Doc": "Document with raw text and metadata",
        "Quote": "Extracted quote with span and speaker info",
        "Outcome": "Case outcome labels and metadata",
        "QuoteFeatures": "Feature vectors for quotes (versioned)",
        "CaseVector": "Aggregated case-level features",
        "Prediction": "ML predictions (quote or case level)",
        "CasePrediction": "Case-level predictions with metadata",
    }

    for name, desc in descriptions.items():
        console.print(f"  [cyan]{name:<15}[/cyan] {desc}")


@app.callback()
def callback():
    """
    corpus-types: Authoritative schemas, IDs, and validators for the corpus pipeline.

    Use this CLI to validate data files against corpus-types schemas,
    generate JSON Schema files, and manage data contracts.
    """


if __name__ == "__main__":
    app()
