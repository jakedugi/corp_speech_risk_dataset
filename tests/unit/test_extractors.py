"""
Tests for the individual extractor components of the quote extraction pipeline.
"""
from corp_speech_risk_dataset.extractors.first_pass import FirstPassExtractor
from corp_speech_risk_dataset.extractors.rerank import SemanticReranker
from corp_speech_risk_dataset.models.quote_candidate import QuoteCandidate

def test_first_pass_hits_bullet():
    """
    Ensures the FirstPassExtractor correctly identifies a quote preceded
    by a bullet point.
    """
    text = '    •  "We won\'t sell your data," said WhatsApp.\\n'
    # The keyword "data" must be present for the window check to pass
    extractor = FirstPassExtractor(keywords=["data"])
    candidates = list(extractor.extract(text))
    assert len(candidates) == 1
    assert candidates[0].quote == '•  "We won\'t sell your data," said WhatsApp.\\n'

def test_reranker_threshold():
    """
    Ensures the SemanticReranker correctly drops a candidate with a score
    below the similarity threshold.
    """
    # This quote is semantically different enough from the seed to be dropped
    qc = QuoteCandidate(quote="This is a test.", context="", urls=[])
    # The seed quote is about data privacy
    reranker = SemanticReranker(seed_quotes=["we will not sell user data"], threshold=0.8)
    output = list(reranker.rerank([qc]))
    assert len(output) == 0 