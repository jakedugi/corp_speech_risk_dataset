"""Command-line interface for positional feature extraction.

Example usage:
    uv run python -m case_aggregation.cli \
        --cases-root /Users/.../data/extracted/courtlistener/14-2274_ca6 \
        --quotes-jsonl /Users/.../data/legal_bert/courtlistener_v4_fused_raw/doc_639752_text_stage12.jsonl \
        --output-jsonl /Users/.../outputs/case_positions/doc_639752_with_positions.jsonl

The CLI supports running over a single case directory at a time. If the quotes
span multiple cases, invoke the CLI per case.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import time
from .positional_features import append_positional_features, load_jsonl, dump_jsonl
from .progress import Progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append positional features to quotes for a case."
    )
    parser.add_argument(
        "--cases-root",
        required=True,
        help="Absolute path to a single case root containing entries/ with stage1.jsonl files.",
    )
    parser.add_argument(
        "--quotes-jsonl",
        required=True,
        help="Path to input quotes JSONL (or a single-line JSON array).",
    )
    parser.add_argument(
        "--quotes-field",
        default=None,
        help="Optional top-level field name if the quotes file is a JSON object with a list under this key.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Path to write augmented quotes JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = os.path.abspath(args.cases_root)
    quotes_path = os.path.abspath(args.quotes_jsonl)
    output_path = os.path.abspath(args.output_jsonl)

    t0 = time.time()
    quotes = load_jsonl(quotes_path)
    print(f"Loaded {len(quotes)} quotes in {time.time()-t0:.2f}s")
    # If a top-level object with a list field was provided, unwrap it
    if args.quotes_field and isinstance(quotes, dict):
        quotes = quotes.get(args.quotes_field, [])
    prog = Progress(label="pos_features", total=len(quotes))
    augmented = []
    # Process in a simple loop to update ETA; internally append_positional_features
    # already runs fast, but we show progress at high level.
    for idx, q in enumerate(quotes, start=1):
        augmented.extend(append_positional_features(case_dir, [q]))
        if idx % 10 == 0 or idx == len(quotes):
            prog.update(idx)
    prog.finish()
    dump_jsonl(output_path, augmented)
    print(f"Wrote {len(augmented)} records to {output_path}")


if __name__ == "__main__":
    main()
