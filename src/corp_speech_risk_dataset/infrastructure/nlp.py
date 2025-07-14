import spacy
import fastcoref

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
        _nlp.add_pipe("fastcoref")
    return _nlp
