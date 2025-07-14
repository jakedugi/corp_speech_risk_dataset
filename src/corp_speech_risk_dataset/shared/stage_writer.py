from pathlib import Path
import json

class StageWriter:
    """
    Write per-stage JSONL beside the source file, preserving tree structure under a mirrored output root.
    """
    def __init__(self, src_root: Path, dst_root: Path):
        self.src_root = src_root
        self.dst_root = dst_root

    def _target(self, src: Path, stage: int) -> Path:
        rel_dir = src.parent.relative_to(self.src_root)
        out_dir = self.dst_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{src.stem}_stage{stage}.jsonl"

    def write(self, src: Path, stage: int, record: dict):
        with self._target(src, stage).open("a", encoding="utf8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
