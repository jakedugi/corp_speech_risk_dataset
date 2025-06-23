import json
import pytest
import logging
import os
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
from corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline import QuoteExtractionPipeline
from corp_speech_risk_dataset.utils.nltk_setup import ensure_nltk_resources

@pytest.fixture(autouse=True)
def setup_pipeline_dirs_and_config(tmp_path, monkeypatch):
    # 1) Create isolated JSON/TXT dirs
    json_dir = tmp_path / "jsons"
    txt_dir = tmp_path / "txts"
    json_dir.mkdir()
    txt_dir.mkdir()

    # 2) Write multiple mock JSON and TXT files with known-good sentences
    mock_data = {
        "doc1": {
            "json": {"plain_text": 'Alice said, "This is a test quote."'},
            "txt":  'Bob said, "Here is another quote."'
        },
        "doc2": {
            "json": {"plain_text": (
                'Carol said, "Our product is safe." '
                'Dave said, "No deceptive practices here."'
            )},
            "txt":  'Eve said, "This seems suspicious."'
        },
        "doc3": {
            "json": {"plain_text": 'Frank said, "Transparency is our priority."'},
            "txt":  'Grace said, "We value your privacy."'
        }
    }

    for doc_id, texts in mock_data.items():
        # JSON file
        (json_dir / f"{doc_id}.json").write_text(
            json.dumps(texts["json"]), encoding="utf8"
        )
        # TXT file
        (txt_dir / f"{doc_id}.txt").write_text(
            texts["txt"], encoding="utf8"
        )

    # Patch the attributes on the config module directly
    import corp_speech_risk_dataset.orchestrators.quote_extraction_config as config
    monkeypatch.setattr(config, "JSON_DIR", str(json_dir), raising=False)
    monkeypatch.setattr(config, "TXT_DIR", str(txt_dir), raising=False)
    monkeypatch.setattr(config, "KEYWORDS", ["test", "quote", "safe", "deceptive", "suspicious", "transparency", "privacy"], raising=False)
    monkeypatch.setattr(config, "COMPANY_ALIASES", ["alice", "bob", "carol", "dave", "eve", "frank", "grace"], raising=False)
    monkeypatch.setattr(config, "SEED_QUOTES", [
        "This is a test quote.",
        "Here is another quote.",
        "Our product is safe.",
        "No deceptive practices here.",
        "This seems suspicious.",
        "Transparency is our priority.",
        "We value your privacy."
    ], raising=False)
    monkeypatch.setattr(config, "THRESHOLD", 0.0, raising=False)

    return json_dir, txt_dir

def test_quote_extraction_pipeline(setup_pipeline_dirs_and_config, capsys):
    # Ensure NLTK resources are present
    ensure_nltk_resources()

    # Reload config so the loader picks up our monkeypatched dirs & keywords
    import importlib
    import corp_speech_risk_dataset.orchestrators.quote_extraction_config as config_module
    importlib.reload(config_module)

    # Now import the pipeline (which will read from the fresh config on __init__)
    from corp_speech_risk_dataset.orchestrators.quote_extraction_pipeline import QuoteExtractionPipeline

    pipe = QuoteExtractionPipeline()
    results = list(pipe.run())

    # Debug print on failure
    if not results:
        pytest.fail("Pipeline returned no results")
    else:
        for doc_id, quotes in results:
            quote_texts = [q.quote for q in quotes]
            print(f"[DEBUG] {doc_id} → {quote_texts}")

    # Flatten all extracted quotes
    all_quotes = [q.quote.lower() for _, quotes in results for q in quotes]

    # Assert presence of expected substrings
    assert any("test quote" in qt for qt in all_quotes), (
        f"'test quote' not found in extracted quotes: {all_quotes}"
    )
    assert any("another quote" in qt for qt in all_quotes), (
        f"'another quote' not found in extracted quotes: {all_quotes}"
    )
    assert any("safe" in qt for qt in all_quotes), (
        f"'safe' not found in extracted quotes: {all_quotes}"
    )
    assert any("deceptive" in qt for qt in all_quotes), (
        f"'deceptive' not found in extracted quotes: {all_quotes}"
    )
    assert any("suspicious" in qt for qt in all_quotes), (
        f"'suspicious' not found in extracted quotes: {all_quotes}"
    )
    assert any("transparency" in qt for qt in all_quotes), (
        f"'transparency' not found in extracted quotes: {all_quotes}"
    )
    assert any("privacy" in qt for qt in all_quotes), (
        f"'privacy' not found in extracted quotes: {all_quotes}"
    )

    # Write to unique output file
    output_file = "test_extracted_quotes.jsonl"
    with open(output_file, "w", encoding="utf8") as out:
        for doc_id, quotes in results:
            rec = {
                "doc_id": doc_id,
                "quotes": [
                    {"text": q.quote, "speaker": q.speaker, "score": q.score, "urls": q.urls}
                    for q in quotes
                ]
            }
            out.write(json.dumps(rec) + "\n")

    # Cleanup
    for file in setup_pipeline_dirs_and_config[0].glob("*"):
        file.unlink()
    setup_pipeline_dirs_and_config[0].rmdir()
    os.remove(output_file)
    print("quote_extraction_pipeline integration test passed.") 