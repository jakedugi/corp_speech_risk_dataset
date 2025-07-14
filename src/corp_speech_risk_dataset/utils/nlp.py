import spacy
from loguru import logger

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        # Try to add fastcoref but fall back gracefully if not available
        try:
            _nlp.add_pipe("fastcoref")
            logger.info("Added fastcoref pipeline component")
        except (ValueError, ImportError) as e:
            logger.warning(f"fastcoref not available, skipping: {e}")
    return _nlp
