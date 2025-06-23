import pytest
from corp_speech_risk_dataset.extractors.first_pass import FirstPassExtractor

# A minimal snippet of the exhibit text containing the "smoking-gun" quotes we must always capture.
GOLDEN_DOC = """
Case 1:19-cv-02184-TJK Document 5-9 Filed 07/26/19 Page 2 of 28

"We do not use your mobile phone number or other Personally Identifiable Information to send commercial or marketing messages without your consent."

"So first of all, let's set the record straight. We have not, we do not and we will not ever sell your personal information to anyone. Period. End of story."
"""

EXPECTED_QUOTES = [
    # Privacy-policy promise
    "\"We do not use your mobile phone number or other Personally Identifiable Information to send commercial or marketing messages without your consent.\"",
    # Founder blog pledge
    "\"So first of all, let's set the record straight. We have not, we do not and we will not ever sell your personal information to anyone. Period. End of story.\"",
]

@pytest.fixture
def extractor():
    # Use a representative keyword set for filtering; adjust as needed.
    keywords = [
        "privacy", "marketing", "consent", "disclosed",
        "posted", "says", "policy",
    ]
    return FirstPassExtractor(keywords=keywords)


def test_golden_quotes(extractor):
    quotes = [qc.quote for qc in extractor.extract(GOLDEN_DOC)]
    print("Extracted quotes:", quotes)
    # Each expected snippet should appear in the extracted quotes
    for expected in EXPECTED_QUOTES:
        assert expected in quotes, f"Missing expected quote: {expected}"
    # Ensure we extracted at least as many candidates as we have golden truths
    assert len(quotes) >= len(EXPECTED_QUOTES) 