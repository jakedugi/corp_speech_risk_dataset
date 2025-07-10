import pytest
from src.corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer
from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph
from src.corp_speech_risk_dataset.encoding.wl_features import wl_vector

# TODO: Set the correct SentencePiece model path
SP_MODEL_PATH = "spiece.model"
tokenizer = SentencePieceTokenizer(SP_MODEL_PATH)

def test_wl_replay():
    """
    Test that the encoding pipeline is fully reversible and deterministic.
    """
    txt = "nutrient enhanced Water beverage"
    ids = tokenizer.encode(txt)
    g   = to_dependency_graph(txt)
    v   = wl_vector(txt)
    # byte-level round-trip
    assert tokenizer.decode(ids) == txt
    # replay WL labels
    g2  = to_dependency_graph(tokenizer.decode(ids))
    assert set(v.indices) == set(wl_vector(txt).indices) 