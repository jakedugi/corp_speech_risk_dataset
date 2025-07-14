"""Tests for extractor functionality."""

from corp_speech_risk_dataset.extractors.first_pass import FirstPassExtractor
from corp_speech_risk_dataset.extractors.rerank import SemanticReranker
from corp_speech_risk_dataset.extractors.cleaner import TextCleaner

def test_first_pass_hits_bullet():
    """
    Ensures the FirstPassExtractor correctly identifies a quote preceded
    by a bullet point.
    """
    text = '    •  "We won\'t sell your data," said WhatsApp.\\n'
    # The keyword "data" must be present for the window check to pass
    cleaner = TextCleaner()
    extractor = FirstPassExtractor(keywords=["data"], cleaner=cleaner)
    
    candidates = list(extractor.extract(text))
    assert len(candidates) >= 1
    assert any("We won't sell your data" in c.quote for c in candidates)


def test_reranker_threshold():
    """
    Ensures that the semantic reranker correctly filters out candidates
    below the threshold.
    """
    # Create mock quote candidates
    class MockQuoteCandidate:
        def __init__(self, quote):
            self.quote = quote
    
    seed_quotes = ["The company will protect user data"]
    reranker = SemanticReranker(seed_quotes, threshold=0.8)  # high threshold
    
    # Test with a low-similarity quote (should be filtered out)
    low_sim_candidate = MockQuoteCandidate("Weather is nice today")
    results = list(reranker.rerank([low_sim_candidate]))
    assert len(results) == 0  # Should be filtered out 