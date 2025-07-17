import pytest
from corp_speech_risk_dataset.extractors.first_pass import FirstPassExtractor
from corp_speech_risk_dataset.extractors.cleaner import TextCleaner

# A minimal snippet of the exhibit text containing the "smoking-gun" quotes we must always capture.
GOLDEN_DOC = """
Case 1:19-cv-02184-TJK Document 5-9 Filed 07/26/19 Page 2 of 28

"We do not use your mobile phone number or other Personally Identifiable Information to send commercial or marketing messages without your consent."

"So first of all, let's set the record straight. We have not, we do not and we will not ever sell your personal information to anyone. Period. End of story."
"""

EXPECTED_QUOTES = [
    # Privacy-policy promise
    '"We do not use your mobile phone number or other Personally Identifiable Information to send commercial or marketing messages without your consent."',
    # Clear data-sale denial
    '"So first of all, let\'s set the record straight. We have not, we do not and we will not ever sell your personal information to anyone. Period. End of story."',
]


@pytest.fixture
def extractor():
    # Use a representative keyword set for filtering; adjust as needed.
    keywords = [
        "information",
        "personal",
        "data",
        "mobile",
        "phone",
        "consent",
        "sell",
        "marketing",
    ]
    cleaner = TextCleaner()
    return FirstPassExtractor(keywords, cleaner)


def test_golden_quotes(extractor):
    """
    Ensure these specific, critical quotes always pass through the first-pass filter.
    These represent the exact type of corporate speech that our system must capture.
    """
    candidates = list(extractor.extract(GOLDEN_DOC))
    candidate_texts = [c.quote for c in candidates]

    for expected in EXPECTED_QUOTES:
        assert any(
            expected in candidate for candidate in candidate_texts
        ), f"Golden quote not found: {expected}"
