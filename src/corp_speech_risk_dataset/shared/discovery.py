from pathlib import Path
from collections import defaultdict

def find_stage_files(root: Path, pattern="*_stage*.jsonl"):
    """
    Walk `root` recursively and return {stage_int: [Path, …]} dict.
    Accepts both the old flat layout and the new mirrored tree.
    """
    buckets = defaultdict(list)
    for p in root.rglob(pattern):
        try:
            stage_int = int(p.stem.split("_stage")[-1])
            buckets[stage_int].append(p)
        except ValueError:
            continue
    return buckets
