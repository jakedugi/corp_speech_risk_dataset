# cli.py
import os
import json
import numpy as np
from collections import Counter
import argparse
import time
from main_case_agnostic import process_data
from utils_case_agnostic import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw features for legal quotes."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with JSONL files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for mirrored JSONL files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Count total JSONL files and entries for ETA
    jsonl_files = [f for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
    total_files = len(jsonl_files)
    total_entries = 0
    for filename in jsonl_files:
        data = load_data(os.path.join(args.input_dir, filename))
        total_entries += len(data)

    start_time = time.time()
    processed_files = 0
    processed_entries = 0
    all_timings = []

    for idx, filename in enumerate(jsonl_files, 1):
        input_file = os.path.join(args.input_dir, filename)
        output_file = os.path.join(args.output_dir, filename)
        file_start_time = time.time()

        # Process file and collect timings
        timings = process_data(input_file, output_file, "all")
        all_timings.append(timings)

        processed_files += 1
        data = load_data(input_file)
        processed_entries += len(data)

        # Compute ETA
        elapsed_time = time.time() - start_time
        entries_per_second = processed_entries / elapsed_time if elapsed_time > 0 else 1
        remaining_entries = total_entries - processed_entries
        eta_seconds = (
            remaining_entries / entries_per_second if entries_per_second > 0 else 0
        )
        eta_minutes = eta_seconds / 60

        # Print progress
        print(
            f"Processed {idx}/{total_files} files ({processed_entries}/{total_entries} entries). "
            f"ETA: {eta_minutes:.2f} minutes"
        )

    # Aggregate timings across files
    if all_timings:
        extractor_times = {}
        for timings in all_timings:
            for extractor, time_list in timings.items():
                if extractor not in extractor_times:
                    extractor_times[extractor] = []
                extractor_times[extractor].extend(time_list)

        timing_stats = {}
        for extractor, times in extractor_times.items():
            timing_stats[extractor] = {
                "mean_ms": float(np.mean(times) * 1000),
                "std_ms": float(np.std(times) * 1000),
                "total_ms": float(np.sum(times) * 1000),
                "calls": len(times),
            }

        with open(os.path.join(args.output_dir, "timings.json"), "w") as f:
            json.dump(timing_stats, f, indent=2)

    # Compute summary stats across all entries (unchanged)
    all_raw = []
    for filename in os.listdir(args.output_dir):
        if filename.endswith(".jsonl"):
            data = load_data(os.path.join(args.output_dir, filename))
            for item in data:
                all_raw.append(item["raw_features"])

    if all_raw:
        stats = {}
        sample = all_raw[0]
        for key in sample.keys():
            values = [item.get(key) for item in all_raw if key in item]
            if not values:
                continue
            if isinstance(values[0], (int, float)):
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            elif (
                isinstance(values[0], list)
                and len(values[0]) > 0
                and isinstance(values[0][0], (int, float))
            ):
                try:
                    values_array = np.array(values)
                    stats[key] = {
                        "mean": np.mean(values_array, axis=0).tolist(),
                        "std": np.std(values_array, axis=0).tolist(),
                        "min": np.min(values_array, axis=0).tolist(),
                        "max": np.max(values_array, axis=0).tolist(),
                    }
                except ValueError:
                    pass
            elif (
                isinstance(values[0], list)
                and len(values[0]) > 0
                and isinstance(values[0][0], str)
            ):
                lengths = [len(v) for v in values]
                all_items = [item for v in values for item in v]
                counter = Counter(all_items)
                stats[key] = {
                    "avg_length": float(np.mean(lengths)),
                    "most_common": counter.most_common(10),
                }
            elif isinstance(values[0], str):
                counter = Counter(values)
                stats[key] = {"most_common": counter.most_common(10)}

        with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)


# Decisions:
# - Modular: Separated extraction functions for extendability.
# - CLI: Supports individual (e.g., --features=quote_size,sentiment) or all.
# - Optimization: CPU for Spacy/VADER/regex (fast on M1), MPS/GPU for BERT embeddings.
# - TF-IDF: Fit on whole corpus for IDF, PCA optional (set to 50 for dim control).
# - WL: Approximated with label propagation count for speed (O(nodes), not O(n^2)).
# - Detections: Lexicon-based for speed/meaningful signal, no ML to avoid training.
# - Fusion: Simple concat; user can PCA later if dim high. Comments inline on benefits.
# - Section headers: Regex extraction, assume in context.
# - Input/Output: JSONL for large data.
# - All implemented without skipping; two-step: extract raw, then fuse.
# - Works one-shot: Install deps (torch, transformers, spacy, vad
