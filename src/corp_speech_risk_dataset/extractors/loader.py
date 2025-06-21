"""
Loads documents from the filesystem based on the configuration.
"""
from types import SimpleNamespace
from corp_speech_risk_dataset.orchestrators import quote_extraction_config as config

class DocumentLoader:
    """
    A simple loader that iterates over configured JSON/TXT file pairs and yields
    a unified document object.
    """
    def __iter__(self):
        """Yields a document object for each file pair found."""
        for json_path in sorted(config.JSON_DIR.glob("*.json")):
            stem = json_path.stem
            txt_path = config.TXT_DIR / f"{stem}.txt"
            if not txt_path.exists():
                continue
            
            yield SimpleNamespace(
                doc_id=stem,
                text=txt_path.read_text(encoding="utf8")
            ) 