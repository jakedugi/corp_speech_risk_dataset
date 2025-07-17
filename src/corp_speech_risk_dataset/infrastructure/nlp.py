import spacy
from loguru import logger

_nlp = None


def get_nlp():
    """Get spaCy NLP model with coreference resolution."""
    try:
        _nlp = spacy.load("en_core_web_sm")
        # Try to add fastcoref but fall back gracefully if not available
        try:
            _nlp.add_pipe("fastcoref")
            logger.info("Added fastcoref pipeline component")
        except (ValueError, ImportError) as e:
            logger.warning(f"fastcoref not available, skipping: {e}")
        return _nlp
    except OSError:
        logger.error(
            "spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm"
        )
        raise
