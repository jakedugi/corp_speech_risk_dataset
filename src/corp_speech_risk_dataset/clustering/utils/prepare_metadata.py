# Helper: Prepare metadata.json from raw tokenized JSONL
# Losslessly reversible: preserve all original fields
# Usage: python -m corp_speech_risk_dataset.models.clustering.prepare_metadata \
#            --input-dir data/tokenized \
#            --output-path data/clustering/metadata.json

from pathlib import Path
import json
import re
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
    """Drop any entry whose 'speaker' matches exactly one in `exclude`."""
    exclude_set = set(exclude)
    return [e for e in entries if e.get("speaker") not in exclude_set]


def filter_heuristics(entries: List[Dict]) -> List[Dict]:
    """
    Apply the temporary heuristics:
      1. Drop entries whose speaker string contains (case‐insensitive) any excluded substrings.
      2. Drop entries whose text contains 'plaintiff' in any form.
      3. Drop entries whose text contains other banned tokens.
      4. Drop entries with fewer than 5 words in 'text'.
    """
    # compile once
    plaintiff_re = re.compile(r"\bplaintiff(s)?\b", re.I)

    bad_tokens = [
        "court",
        "ftc",
        "fed",
        "plaintiff",
        "state",
        "commission",
        "congress",
        "circuit",
        "fda",
        "federal law",
        "statutory",
        "juncture",
        "litigation",
        "counsel",
        "llc",  # newly added
        "supp",  # newly added (matches 'Supp.')
        "judge",  # newly added
        "litigant",  # newly added
    ]
    bad_token_res = [
        re.compile(r"\b{}\b".format(re.escape(w)), re.I) for w in bad_tokens
    ]

    exclude_speakers = {
        w.lower()
        for w in [
            "Unknown",
            "Court",
            "FTC",
            "Fed",
            "Plaintiff",
            "State",
            "Commission",
            "Congress",
            "Circuit",
            "FDA",
            "LLC",  # newly added
            "Supp.",  # newly added
            "Judge",  # newly added
            "Litigant",  # newly added
        ]
    }

    filtered = []
    for e in entries:
        text = e.get("text", "") or ""
        speaker = e.get("speaker", "") or ""
        sp_lower = speaker.lower()

        # A) speaker‐based substring exclusion
        if any(sub in sp_lower for sub in exclude_speakers):
            continue

        # B) 'plaintiff' anywhere in text
        if plaintiff_re.search(text):
            continue

        # C) any other banned token in text
        if any(rx.search(text) for rx in bad_token_res):
            continue

        # D) too short?
        if len(text.split()) < 5:
            continue

        filtered.append(e)

    return filtered


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
        "--output-path",
        required=True,
        help="File path to write metadata.json",
    )
    parser.add_argument(
        "--exclude-speakers",
        nargs="*",  # zero-or-more values
        default=[],
        metavar="NAME",
        help="List of speaker names to drop before writing metadata",
    )
    parser.add_argument(
        "--apply-heuristics",
        action="store_true",
        help=(
            "If set, also drop entries by substring‐based speaker exclusion, "
            "text containing 'plaintiff' or other banned tokens, or text <5 words."
        ),
    )

    args = parser.parse_args()

    root = Path(args.input_dir)
    out = Path(args.output_path)

    # 1) collect & load
    jsonl_files = collect_jsonl_files(root)
    entries = load_entries(jsonl_files)

    # 2) exact-speaker exclusion (existing behavior)
    if args.exclude_speakers:
        before = len(entries)
        entries = filter_speakers(entries, args.exclude_speakers)
        print(f"Dropped {before - len(entries)} entries for excluded speakers")

    # 3) optional heuristics
    if args.apply_heuristics:
        before = len(entries)
        entries = filter_heuristics(entries)
        print(f"Dropped {before - len(entries)} entries by heuristic filters")

    # 4) write out
    write_metadata(entries, out)
    print(f"Wrote lossless metadata with {len(entries)} entries to {out}")
