import os
import json
from pathlib import Path
from typing import Iterator
from ..orchestrators import quote_extraction_config as config
from .models import Document

class DocumentLoader:
    def __init__(self, json_dir=None, txt_dir=None):
        self.json_dir = Path(json_dir) if json_dir is not None else Path(config.JSON_DIR)
        self.txt_dir = Path(txt_dir) if txt_dir is not None else Path(config.TXT_DIR)

    def load_json(self, path: Path) -> Document:
        obj = json.loads(path.read_text(encoding="utf8"))
        txt = obj.get("plain_text") or ""
        return Document(
            doc_id=path.stem,
            text=txt,
            source_path=str(path)
        )

    def load_txt(self, path: Path) -> Document:
        return Document(
            doc_id=path.stem,
            text=path.read_text(encoding="utf8"),
            source_path=str(path)
        )

    def __iter__(self) -> Iterator[Document]:
        print(f"[PRINT Loader] runtime JSON_DIR={self.json_dir}, TXT_DIR={self.txt_dir}")
        for p in self.json_dir.glob("*.json"):
            print(f"[PRINT Loader] yielding doc_id={p.stem}")
            yield self.load_json(p)
        for p in self.txt_dir.glob("*.txt"):
            print(f"[PRINT Loader] yielding doc_id={p.stem}")
            yield self.load_txt(p) 