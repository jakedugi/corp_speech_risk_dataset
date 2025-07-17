"""
Loads documents from the filesystem based on the configuration.
Supports multiple naming patterns for JSON/TXT pairs in the same folder.
"""

from types import SimpleNamespace
from pathlib import Path
import logging

from corp_speech_risk_dataset.orchestrators import quote_extraction_config as config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads .txt files recursively from a source root, yielding doc_id, text, and path (provenance).
    """

    def __init__(self, source_root: Path = config.DB_DIR):
        self.source_root = source_root

    def __iter__(self):
        for txt_path in self.source_root.rglob("*.txt"):
            yield SimpleNamespace(
                doc_id=txt_path.stem,
                text=txt_path.read_text(encoding="utf8"),
                path=txt_path,
            )
