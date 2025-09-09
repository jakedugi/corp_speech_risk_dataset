"""
Helper script to import/merge positional features into fused JSONL files.

This script reads per-entry positional fields from files in a source directory
(`data/legal_bert_positions/`) and merges those fields into the corresponding
JSONL files in the destination directory (`data/legal_bert/courtlistener_v4_fused_raw/`).

Key properties:
- Matches entries robustly using a deterministic fingerprint based on
  `doc_id`, `stage`, `speaker`, `_src`, and a SHA1 hash of the entry `text`.
- Adds only the specified positional keys to each destination entry
  (default: `docket_number`, `docket_token_start`, `global_token_start`,
  `docket_char_start`, `global_char_start`, `num_tokens`).
- Never overwrites existing keys/values in the destination; it only adds
  missing positional fields.
- Validates that entries are matched correctly; supports strict mode.
- Writes atomically with an optional backup of the original destination file.
- Defaults to dry-run; use `--apply` to actually write changes.

Usage examples:
  uv run python scripts/import_positional_features.py --apply
  uv run python scripts/import_positional_features.py --positions-dir /abs/src --dest-dir /abs/dest --apply
  uv run python scripts/import_positional_features.py --file /abs/src/doc_639746_text_stage12.jsonl --apply --backup

Project: Corporate Speech Risk Dataset
Author: Jake Dugan <jake.dugan@ed.ac.uk>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict, deque
import re
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple


DEFAULT_POS_KEYS: Tuple[str, ...] = (
    "docket_number",
    "docket_token_start",
    "global_token_start",
    "docket_char_start",
    "global_char_start",
    "num_tokens",
)


@dataclasses.dataclass
class MergeStats:
    """Tracks per-file merge statistics for reporting.

    Attributes:
        total_destination_entries: Total number of entries read from the destination file.
        matched_entries: Number of destination entries that found a matching positions entry.
        entries_with_changes: Number of destination entries modified by adding at least one positional key.
        total_keys_added: Total number of positional keys added across all destination entries.
        unmatched_destination_entries: Number of destination entries without a match in positions file.
        unmatched_position_entries: Number of positions entries that remained unused after matching.
        conflicts_detected: Number of entries where a destination already had a positional key with a different value (not overwritten).
    """

    total_destination_entries: int = 0
    matched_entries: int = 0
    entries_with_changes: int = 0
    total_keys_added: int = 0
    unmatched_destination_entries: int = 0
    unmatched_position_entries: int = 0
    conflicts_detected: int = 0


def compute_entry_fingerprint(entry: Dict) -> str:
    """Compute a deterministic fingerprint for an entry to align across files.

    The fingerprint uses stable, discriminating attributes to minimize false
    positives while tolerating reorderings:
    - doc_id
    - stage
    - speaker
    - _src (source path of the snippet)
    - len(text) and SHA1(text)

    Args:
        entry: Parsed JSON object for one line.

    Returns:
        A string fingerprint.
    """

    doc_id = str(entry.get("doc_id", ""))
    stage = str(entry.get("stage", ""))
    speaker = str(entry.get("speaker", ""))
    src = str(entry.get("_src", ""))
    text = entry.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    text_len = len(text)
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{doc_id}|{stage}|{speaker}|{src}|{text_len}|{text_hash}"


def read_jsonl(path: Path) -> List[Dict]:
    """Read a JSONL file into a list of dictionaries.

    Skips empty lines. Raises ValueError on invalid JSON.
    """
    entries: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for idx, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path} line {idx}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path} line {idx}")
            entries.append(obj)
    return entries


def write_jsonl_atomic(path: Path, entries: Iterable[Dict]) -> None:
    """Write entries to `path` atomically.

    Writes to a temporary file in the same directory and then replaces the
    original via `os.replace` to ensure atomicity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
        for obj in entries:
            json.dump(obj, tmp, ensure_ascii=False, separators=(",", ":"))
            tmp.write("\n")
    os.replace(tmp_path, path)


def build_positions_index(positions_entries: List[Dict]) -> Dict[str, Deque[Dict]]:
    """Build a multimap from entry fingerprint to a queue of positions entries.

    Uses a deque per key to support FIFO consumption for duplicate fingerprints.
    """
    index: Dict[str, Deque[Dict]] = defaultdict(deque)
    for entry in positions_entries:
        index[compute_entry_fingerprint(entry)].append(entry)
    return index


def merge_positional_fields(
    positions_path: Path,
    destination_path: Path,
    keys_to_add: Tuple[str, ...],
    *,
    dry_run: bool,
    strict: bool,
    backup: bool,
) -> MergeStats:
    """Merge positional fields from positions file into destination JSONL file.

    This function reads both files, aligns entries using fingerprints, and adds
    only the missing keys from `keys_to_add`. It never overwrites existing
    values in the destination. It returns detailed statistics for reporting.

    Args:
        positions_path: Path to the source positions JSONL file.
        destination_path: Path to the destination JSONL file to augment.
        keys_to_add: Tuple of keys that should be copied if missing.
        dry_run: If True, do not write changes; only compute and report.
        strict: If True, raise on mismatched counts or unmatched entries.
        backup: If True and writing, create a timestamped backup of the destination.

    Returns:
        MergeStats with summary counts.
    """

    stats = MergeStats()

    if not destination_path.exists():
        # Destination missing; skip silently (not an error per requirements).
        return stats

    positions_entries = read_jsonl(positions_path)
    destination_entries = read_jsonl(destination_path)

    stats.total_destination_entries = len(destination_entries)

    positions_index = build_positions_index(positions_entries)

    any_changes = False
    conflicts_detected = 0
    entries_with_changes = 0
    total_keys_added = 0

    updated_entries: List[Dict] = []

    for dest_entry in destination_entries:
        fp = compute_entry_fingerprint(dest_entry)
        source_queue = positions_index.get(fp)
        if source_queue and len(source_queue) > 0:
            src_entry = source_queue.popleft()
            stats.matched_entries += 1

            keys_added_for_entry = 0
            for key in keys_to_add:
                if key in dest_entry:
                    # Do not overwrite; if values differ, record a conflict.
                    if key in src_entry and dest_entry.get(key) != src_entry.get(key):
                        conflicts_detected += 1
                    continue
                if key in src_entry:
                    dest_entry[key] = src_entry[key]
                    keys_added_for_entry += 1

            if keys_added_for_entry > 0:
                entries_with_changes += 1
                total_keys_added += keys_added_for_entry
                any_changes = True

            updated_entries.append(dest_entry)
        else:
            stats.unmatched_destination_entries += 1
            updated_entries.append(dest_entry)

    # Compute how many positions entries were left unused
    unused_positions = sum(len(q) for q in positions_index.values())
    stats.unmatched_position_entries = unused_positions

    stats.entries_with_changes = entries_with_changes
    stats.total_keys_added = total_keys_added
    stats.conflicts_detected = conflicts_detected

    if strict:
        # Strict matching requires zero unmatched on both sides.
        if (
            stats.unmatched_destination_entries != 0
            or stats.unmatched_position_entries != 0
        ):
            raise RuntimeError(
                (
                    f"Strict matching failed for {destination_path.name}: "
                    f"unmatched_destination_entries={stats.unmatched_destination_entries}, "
                    f"unmatched_position_entries={stats.unmatched_position_entries}"
                )
            )

    if any_changes and not dry_run:
        if backup:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = destination_path.with_suffix(
                destination_path.suffix + f".bak.{timestamp}"
            )
            shutil.copy2(destination_path, backup_path)
        write_jsonl_atomic(destination_path, updated_entries)

    return stats


def iter_position_files(
    positions_dir: Path, single_file: Path | None
) -> Iterable[Path]:
    """Yield positions JSONL files to process.

    If `single_file` is provided, yields only that file. Otherwise, yields all
    `*.jsonl` files under `positions_dir` (non-recursive).
    """
    if single_file is not None:
        yield single_file
        return
    for p in sorted(positions_dir.glob("*.jsonl")):
        if p.is_file():
            yield p


_FILENAME_RE = re.compile(r"^(?P<base>doc_\d+_text)_stage(?P<stage>\d+)\.jsonl$")


def parse_doc_and_stage(filename: str) -> Tuple[str, str] | None:
    """Parse a positions/destination filename to extract base doc token and stage.

    Expected pattern: `doc_<digits>_text_stage<digits>.jsonl`.
    Returns (base_without_stage, stage_str) or None if not matched.
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group("base"), m.group("stage")


def resolve_destination_path(
    dest_dir: Path, positions_file: Path, *, dest_stage: str | None
) -> Path | None:
    """Determine the correct destination file path for a given positions file.

    Logic:
    - If the filename follows the expected pattern, construct a candidate
      destination name by replacing the stage with `dest_stage` (if provided)
      or keeping the same stage as the positions file.
    - If the constructed candidate exists, return it.
    - Otherwise, fallback to glob for any file with the same `doc_*_text` base
      in `dest_dir`. If exactly one match is found, return it. If zero or more
      than one are found, return None (ambiguous or missing).
    """
    parsed = parse_doc_and_stage(positions_file.name)
    if parsed is None:
        # Fallback: try exact name first
        candidate = dest_dir / positions_file.name
        if candidate.exists():
            return candidate
        # Last resort: no pattern → unresolved
        return None

    base, pos_stage = parsed
    stage_to_use = dest_stage if dest_stage is not None else pos_stage
    candidate = dest_dir / f"{base}_stage{stage_to_use}.jsonl"
    if candidate.exists():
        return candidate

    # Fallback: search for any stage for this doc base
    matches = list(dest_dir.glob(f"{base}_stage*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    # Ambiguous or not found
    return None


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Import positional features from a positions directory into the mirrored "
            "fused_raw directory, aligning entries and adding only missing fields."
        )
    )
    default_positions_dir = Path.cwd() / "data" / "legal_bert_positions"
    default_dest_dir = Path.cwd() / "data" / "legal_bert" / "courtlistener_v4_fused_raw"

    parser.add_argument(
        "--positions-dir",
        type=Path,
        default=default_positions_dir,
        help="Absolute path to positions JSONL directory (default: ./data/legal_bert_positions)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=default_dest_dir,
        help=(
            "Absolute path to destination fused_raw JSONL directory "
            "(default: ./data/legal_bert/courtlistener_v4_fused_raw)"
        ),
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional: process a single positions JSONL file (absolute path).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to destination files (default is dry-run).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help=(
            "Disable strict matching. When set, the script will not raise on unmatched "
            "entries and will still write any additions for matched entries."
        ),
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a timestamped .bak copy of destination files before writing.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=",".join(DEFAULT_POS_KEYS),
        help=(
            "Comma-separated list of positional keys to add. "
            f"Default: {','.join(DEFAULT_POS_KEYS)}"
        ),
    )
    parser.add_argument(
        "--dest-stage",
        type=str,
        default=None,
        help=(
            "Optional: override the destination stage suffix used to map file names. "
            "For example, if positions are *_stage12.jsonl but destination files are "
            "*_stage15.jsonl, pass --dest-stage 15."
        ),
    )

    args = parser.parse_args(argv)

    # Resolve to absolute paths to align with project conventions.
    if args.positions_dir is not None:
        args.positions_dir = args.positions_dir.resolve()
    if args.dest_dir is not None:
        args.dest_dir = args.dest_dir.resolve()
    if args.file is not None:
        args.file = args.file.resolve()

    # Normalize keys
    args.keys = tuple(k.strip() for k in str(args.keys).split(",") if k.strip())
    return args


def main(argv: List[str]) -> int:
    """Entry point for the script."""
    args = parse_args(argv)

    positions_dir: Path = args.positions_dir
    dest_dir: Path = args.dest_dir
    single_file: Path | None = args.file
    keys: Tuple[str, ...] = args.keys
    dry_run: bool = not args.apply
    strict: bool = not args.no_strict
    backup: bool = args.backup

    if single_file is None and not positions_dir.exists():
        print(f"Positions directory not found: {positions_dir}", file=sys.stderr)
        return 2
    if not dest_dir.exists():
        print(f"Destination directory not found: {dest_dir}", file=sys.stderr)
        return 2

    total_files = 0
    total_changed_files = 0
    aggregate = MergeStats()

    for pos_path in iter_position_files(positions_dir, single_file):
        # Resolve destination path considering potential stage differences
        dest_path = resolve_destination_path(
            dest_dir, pos_path, dest_stage=args.dest_stage
        )
        if dest_path is None:
            print(
                (
                    f"[WARN] Could not resolve destination for positions file {pos_path.name}. "
                    f"Consider specifying --dest-stage or ensure matching filename exists in {dest_dir}."
                ),
                file=sys.stderr,
            )
            total_files += 1
            continue

        total_files += 1
        try:
            stats = merge_positional_fields(
                positions_path=pos_path,
                destination_path=dest_path,
                keys_to_add=keys,
                dry_run=dry_run,
                strict=strict,
                backup=backup,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[ERROR] {pos_path.name}: {exc}", file=sys.stderr)
            continue

        if stats.entries_with_changes > 0:
            total_changed_files += 1

        # Aggregate counters
        aggregate.total_destination_entries += stats.total_destination_entries
        aggregate.matched_entries += stats.matched_entries
        aggregate.entries_with_changes += stats.entries_with_changes
        aggregate.total_keys_added += stats.total_keys_added
        aggregate.unmatched_destination_entries += stats.unmatched_destination_entries
        aggregate.unmatched_position_entries += stats.unmatched_position_entries
        aggregate.conflicts_detected += stats.conflicts_detected

        # Per-file concise log
        print(
            (
                f"{pos_path.name} → {dest_path.name}: dest_entries={stats.total_destination_entries}, "
                f"matched={stats.matched_entries}, added={stats.total_keys_added}, "
                f"changed_entries={stats.entries_with_changes}, "
                f"unmatched_dest={stats.unmatched_destination_entries}, "
                f"unmatched_pos={stats.unmatched_position_entries}, "
                f"conflicts={stats.conflicts_detected}, "
                f"write={'no (dry-run)' if dry_run else 'yes'}"
            )
        )

    # Summary
    print(
        (
            f"Processed files: {total_files}, changed files: {total_changed_files}, "
            f"aggregate: dest_entries={aggregate.total_destination_entries}, matched={aggregate.matched_entries}, "
            f"added={aggregate.total_keys_added}, changed_entries={aggregate.entries_with_changes}, "
            f"unmatched_dest={aggregate.unmatched_destination_entries}, "
            f"unmatched_pos={aggregate.unmatched_position_entries}, conflicts={aggregate.conflicts_detected}, "
            f"mode={'dry-run' if dry_run else 'apply'}"
        )
    )

    # Exit code policy: 0 on success; 1 if strict mode and any unmatched present.
    if strict and (
        aggregate.unmatched_destination_entries != 0
        or aggregate.unmatched_position_entries != 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
