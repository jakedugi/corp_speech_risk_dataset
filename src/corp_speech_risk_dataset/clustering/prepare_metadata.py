# Helper: Prepare metadata.json from raw tokenized JSONL
# Losslessly reversible: preserve all original fields
# Usage: python -m corp_speech_risk_dataset.models.clustering.prepare_metadata \
#            --input-dir data/tokenized \
#            --output-path data/clustering/metadata.json

from pathlib import Path
import json
from typing import List, Dict


def collect_jsonl_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all .jsonl files under root_dir.
    """
    return sorted(root_dir.rglob("*.jsonl"))


def load_entries(jsonl_paths: List[Path]) -> List[Dict]:
    """
    Read and parse all JSONL entries into a single list, preserving every field.
    """
    entries: List[Dict] = []
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
    return entries


def filter_speakers(entries: List[Dict], exclude: List[str]) -> List[Dict]:
    """Drop any entry whose 'speaker' matches one in `exclude`."""
    exclude_set = set(exclude)
    return [e for e in entries if e.get("speaker") not in exclude_set]


def write_metadata(entries: List[Dict], out_path: Path) -> None:
    """
    Write a single JSON file containing the list of entries with all fields.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare lossless metadata.json from raw tokenized JSONL entries."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing tokenized JSONL files",
    )
    parser.add_argument(
        "--output-path", required=True, help="File path to write metadata.json"
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    out = Path(args.output_path)

    jsonl_files = collect_jsonl_files(root)
    entries = load_entries(jsonl_files)
    write_metadata(entries, out)

    print(f"Wrote lossless metadata with {len(entries)} entries to {out}")
