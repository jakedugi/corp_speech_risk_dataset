"""
Loads documents from the filesystem based on the configuration.
Supports multiple naming patterns for JSON/TXT pairs in the same folder.
"""

from types import SimpleNamespace
from pathlib import Path
import logging

from corp_speech_risk_dataset.orchestrators import quote_extraction_config as config
from corp_speech_risk_dataset.extractors.rss_loader import RSSLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads .txt files recursively from a source root, yielding doc_id, text, and path (provenance).
    """

    def __init__(self, source_root: Path = config.DB_DIR):
        self.source_root = source_root

    def __iter__(self):
        # First: pick up any RSS JSON entries
        for json_path in self.source_root.rglob("*.json"):
            if "/rss/" in str(json_path):
                for rec in RSSLoader()._iter_records(json_path):
                    # wrap the dict into the doc‐object the pipeline expects
                    yield SimpleNamespace(
                        doc_id=rec["doc_id"],
                        text=rec["text"],
                        speaker=rec["speaker"],
                        urls=rec["urls"],
                        _rss_parts=rec.get("_rss_parts"),  # for your fallback
                        path=json_path,
                        _src=rec["_src"],
                    )
        # Then fall back to all the old .txt files (CourtListener pipeline)
        for txt_path in self.source_root.rglob("*.txt"):
            yield SimpleNamespace(
                doc_id=txt_path.stem,
                text=txt_path.read_text(encoding="utf8"),
                path=txt_path,
            )
