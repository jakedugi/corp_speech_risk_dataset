"""Batch CLI to append positional features to many quotes files.

This scans an input quotes directory (e.g., legal_bert outputs) for JSONL files
and augments each with positional features by locating the corresponding case
under a cases root (e.g., data/extracted/courtlistener).

Case discovery per quotes file:
- Prefer extracting `case_id` from the `_src` field of any quote in the file
  using the pattern `*/<case_id>/entries/...`.
- Build the case directory as `<cases_root>/<case_id>` and process quotes in the
  file using that docket index.
- If no `_src` yields a case_id, the file is skipped to ensure accuracy.

Outputs are written under the provided `--output-dir` with the same relative
file name as the input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Optional

import time
from .positional_features import append_positional_features, load_jsonl, dump_jsonl
from .progress import Progress


CASE_ID_RE = re.compile(r"/(\d[^/]*?_\w+|\d[^/]*)/entries/")


def extract_case_id_from_quotes(quotes: List[Dict]) -> Optional[str]:
    """Attempt to determine the case_id from quotes' `_src` fields.

    Returns the first consistent case_id found. If multiple different case_ids are
    present, returns None to avoid accidental mixing across cases.
    """
    case_id: Optional[str] = None
    for q in quotes:
        src = q.get("_src") or q.get("src") or ""
        m = CASE_ID_RE.search(src)
        if not m:
            continue
        cid = m.group(1)
        if case_id is None:
            case_id = cid
        elif case_id != cid:
            return None
    return case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch positional feature augmentation for quotes JSONL files."
    )
    parser.add_argument(
        "--cases-root",
        required=True,
        help="Absolute path to extracted cases root (contains many case folders with entries/).",
    )
    parser.add_argument(
        "--quotes-dir",
        required=True,
        help="Directory containing quotes JSONL files to process (recursively).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to mirror quotes-dir structure with augmented files.",
    )
    parser.add_argument(
        "--glob",
        default="**/*.jsonl",
        help="Glob pattern (relative to quotes-dir) to select quotes files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_root = os.path.abspath(args.cases_root)
    quotes_dir = os.path.abspath(args.quotes_dir)
    output_dir = os.path.abspath(args.output_dir)
    pattern = os.path.join(quotes_dir, args.glob)

    files = [f for f in glob(pattern, recursive=True) if os.path.isfile(f)]
    prog = Progress(label="case_aggregation", total=len(files))
    t0 = time.time()
    for i, fpath in enumerate(files, start=1):
        rel = os.path.relpath(fpath, quotes_dir)
        out_path = os.path.join(output_dir, rel)

        quotes = load_jsonl(fpath)
        # If the loader returned a dict (top-level object), try common fields
        if isinstance(quotes, dict):
            for key in ("quotes", "items", "records", "data"):
                if isinstance(quotes.get(key), list):
                    quotes = quotes[key]
                    break
        if not isinstance(quotes, list) or not quotes:
            # Nothing to process; write empty output for consistency
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fp:
                pass
            continue

        case_id = extract_case_id_from_quotes(quotes)
        if not case_id:
            # Skip to avoid incorrect case mapping
            continue

        case_dir = os.path.join(cases_root, case_id)
        if not os.path.isdir(case_dir):
            continue

        augmented = append_positional_features(case_dir, quotes)
        dump_jsonl(out_path, augmented)
        prog.update(i, extra=f"last={os.path.basename(rel)}")
    prog.finish()
    dt = time.time() - t0
    # Summary line
    print(
        f"Processed {len(files)} files in {dt:.2f}s | {len(files)/dt if dt>0 else 0:.2f} files/s"
    )


if __name__ == "__main__":
    main()
