#!/usr/bin/env python
"""
CLI to recursively filter all *_stageX.jsonl under an input directory,
mirroring the structure into an output directory, with full summary stats.
Supports parallel processing and thread-safe stats aggregation.
"""
import argparse

try:
    import ujson as json
except ImportError:
    import json

from pathlib import Path
import re
from collections import Counter
import statistics
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from .corpus_extractors import filter_speakers, filter_heuristics

# lock for stats accumulation
STATS_LOCK = Lock()

# global cumulative stats
CUM_STATS = {
    "before_n": 0,
    "before_word_counts": [],
    "before_context_lengths": [],
    "before_scores": [],
    "before_speakers": Counter(),
    "before_tokens": Counter(),
    "after_n": 0,
    "after_word_counts": [],
    "after_context_lengths": [],
    "after_scores": [],
    "after_speakers": Counter(),
    "after_tokens": Counter(),
}

# helper to compute percentiles


def percentile(data, percent):
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = min(int(percent / 100 * len(sorted_data)), len(sorted_data) - 1)
    return sorted_data[idx]


# merge local stats into global under lock


def merge_stats(local_stats):
    with STATS_LOCK:
        for key, val in local_stats.items():
            if isinstance(CUM_STATS[key], list):
                CUM_STATS[key].extend(val)
            elif isinstance(CUM_STATS[key], Counter):
                CUM_STATS[key].update(val)
            else:
                CUM_STATS[key] += val


# per-file work


def process_file(
    in_path: Path, out_path: Path, exclude, heuristics, stats, local_stats
):
    """Load JSONL, apply filters, write JSONL, update local_stats."""
    entries = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # BEFORE stats
    if stats:
        local_stats["before_n"] += len(entries)
        for e in entries:
            text = e.get("text", "") or ""
            tokens = text.split()
            local_stats["before_word_counts"].append(len(tokens))
            ctx = e.get("context", "") or ""
            local_stats["before_context_lengths"].append(len(ctx.split()))
            score = e.get("score")
            if isinstance(score, (int, float)):
                local_stats["before_scores"].append(score)
            local_stats["before_speakers"][e.get("speaker")] += 1
            local_stats["before_tokens"].update(tokens)

    # filters
    if exclude:
        entries = filter_speakers(entries, exclude)
    if heuristics:
        entries = filter_heuristics(entries)

    # write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # AFTER stats
    if stats:
        local_stats["after_n"] += len(entries)
        for e in entries:
            text = e.get("text", "") or ""
            tokens = text.split()
            local_stats["after_word_counts"].append(len(tokens))
            ctx = e.get("context", "") or ""
            local_stats["after_context_lengths"].append(len(ctx.split()))
            score = e.get("score")
            if isinstance(score, (int, float)):
                local_stats["after_scores"].append(score)
            local_stats["after_speakers"][e.get("speaker")] += 1
            local_stats["after_tokens"].update(tokens)


# worker wrapper to initialize, call process_file, and merge


def work_file(p, inp, out, args):
    # prepare local stats bucket
    local_stats = {
        "before_n": 0,
        "before_word_counts": [],
        "before_context_lengths": [],
        "before_scores": [],
        "before_speakers": Counter(),
        "before_tokens": Counter(),
        "after_n": 0,
        "after_word_counts": [],
        "after_context_lengths": [],
        "after_scores": [],
        "after_speakers": Counter(),
        "after_tokens": Counter(),
    }
    # bump stage in filename
    rel = p.relative_to(inp)
    stem = p.stem
    m = re.match(r"(.+_stage)(\d+)$", stem)
    new_stem = f"{m.group(1)}{int(m.group(2))+1}" if m else f"{stem}_stage1"
    out_p = out / rel.parent / f"{new_stem}.jsonl"

    process_file(
        p,
        out_p,
        exclude=args.exclude_speakers,
        heuristics=args.apply_heuristics,
        stats=not args.no_stats,
        local_stats=local_stats,
    )
    if not args.no_stats:
        merge_stats(local_stats)
    print(f"Filtered {p} → {out_p}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL with stats and parallelism"
    )
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-s", "--stage", type=int)
    parser.add_argument("--exclude-speakers", nargs="*", default=[])
    parser.add_argument("--apply-heuristics", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument(
        "--top-speakers", type=int, default=10, help="Number of top speakers to display"
    )
    parser.add_argument(
        "--top-tokens", type=int, default=10, help="Number of top tokens to display"
    )
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    # collect matching files
    files = [
        p
        for p in inp.rglob("*.jsonl")
        if p.name.endswith(".jsonl")
        and "_stage" in p.stem
        and (args.stage is None or p.stem.endswith(f"_stage{args.stage}"))
    ]

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            exe.map(lambda p: work_file(p, inp, out, args), files)
    else:
        for p in files:
            work_file(p, inp, out, args)

    # final stats display
    if not args.no_stats:
        bn = CUM_STATS["before_n"]
        an = CUM_STATS["after_n"]

        def summary(data):
            return (
                statistics.mean(data) if data else 0,
                statistics.median(data) if data else 0,
                percentile(data, 25),
                percentile(data, 75),
                percentile(data, 90),
            )

        bwc = CUM_STATS["before_word_counts"]
        awc = CUM_STATS["after_word_counts"]
        bcl = CUM_STATS["before_context_lengths"]
        acl = CUM_STATS["after_context_lengths"]
        bscores = CUM_STATS["before_scores"]
        ascores = CUM_STATS["after_scores"]
        bs = len(CUM_STATS["before_speakers"])
        as_ = len(CUM_STATS["after_speakers"])

        avg_b, med_b, p25_b, p75_b, p90_b = summary(bwc)
        avg_a, med_a, p25_a, p75_a, p90_a = summary(awc)
        cavg_b, cmed_b, cp25_b, cp75_b, cp90_b = summary(bcl)
        cavg_a, cmed_a, cp25_a, cp75_a, cp90_a = summary(acl)
        savg_b, smed_b, sp25_b, sp75_b, sp90_b = summary(bscores)
        savg_a, smed_a, sp25_a, sp75_a, sp90_a = summary(ascores)

        print("\nCUMULATIVE STATS SUMMARY:")
        print(f"BEFORE → entries={bn}")
        print(
            f"  WORDS: avg={avg_b:.1f}, med={med_b:.1f}, 25={p25_b}, 75={p75_b}, 90={p90_b}"
        )
        print(
            f"  CONTEXT: avg={cavg_b:.1f}, med={cmed_b:.1f}, 25={cp25_b}, 75={cp75_b}, 90={cp90_b}"
        )
        print(
            f"  SCORES: avg={savg_b:.2f}, med={smed_b:.2f}, 25={sp25_b}, 75={sp75_b}, 90={sp90_b}"
        )
        print(f"  UNIQUE SPEAKERS: {bs}")
        print(f"  TOP {args.top_speakers} SPEAKERS:")
        for sp, cnt in CUM_STATS["before_speakers"].most_common(args.top_speakers):
            print(f"    {sp}: {cnt}")
        print(f"  TOP {args.top_tokens} TOKENS:")
        for tok, cnt in CUM_STATS["before_tokens"].most_common(args.top_tokens):
            print(f"    {tok}: {cnt}")

        print(f"\nAFTER  → entries={an}")
        print(
            f"  WORDS: avg={avg_a:.1f}, med={med_a:.1f}, 25={p25_a}, 75={p75_a}, 90={p90_a}"
        )
        print(
            f"  CONTEXT: avg={cavg_a:.1f}, med={cmed_a:.1f}, 25={cp25_a}, 75={cp75_a}, 90={cp90_a}"
        )
        print(
            f"  SCORES: avg={savg_a:.2f}, med={smed_a:.2f}, 25={sp25_a}, 75={sp75_a}, 90={sp90_a}"
        )
        print(f"  UNIQUE SPEAKERS: {as_}")
        print(f"  TOP {args.top_speakers} SPEAKERS:")
        for sp, cnt in CUM_STATS["after_speakers"].most_common(args.top_speakers):
            print(f"    {sp}: {cnt}")
        print(f"  TOP {args.top_tokens} TOKENS:")
        for tok, cnt in CUM_STATS["after_tokens"].most_common(args.top_tokens):
            print(f"    {tok}: {cnt}")


if __name__ == "__main__":
    main()
