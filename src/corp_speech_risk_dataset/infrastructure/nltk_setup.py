import nltk

REQUIRED_NLTK_PACKAGES = ["punkt", "punkt_tab"]


def ensure_nltk_resources():
    for package in REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            nltk.download(package)
