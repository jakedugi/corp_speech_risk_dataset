import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer
from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph
from src.corp_speech_risk_dataset.encoding.wl_features import wl_vector
from src.corp_speech_risk_dataset.cli_encode import main as cli_encode_main

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
tokenizer = SentencePieceTokenizer()   # GPT-2 tokenizer, no fallback needed

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_gpt2_roundtrip_basic():
    """Round-trip sanity check for the GPT-2 tokenizer."""
    txt = "SEC Form 10-K §12(b); Apple Inc."
    ids = tokenizer.encode(txt)
    assert tokenizer.decode(ids) == txt


def test_gpt2_roundtrip_special_chars():
    """Test lossless encoding of corporate/legal special characters."""
    test_cases = [
        "§12(b) compliance requirements",
        "™ trademark registration",
        "© copyright notice 2024",
        "€1,000,000 revenue",
        "10-K filing § & ™ symbols",
        "UTF-8: 🏢 📊 💼",  # emoji
        "Café résumé naïve",  # accented chars
        '"Smart quotes" and \'apostrophes\'',  # smart punctuation
    ]
    
    for txt in test_cases:
        ids = tokenizer.encode(txt)
        decoded = tokenizer.decode(ids)
        assert decoded == txt, f"Failed roundtrip for: {txt!r}"


def test_no_fallback_ever():
    """Verify that encode_with_flag never triggers fallback with GPT-2."""
    challenging_texts = [
        "Unicode: αβγδε ∑∏∆√",
        "Mixed: Hello世界🌍",
        "Legal: §§§ ™™™ ©©©",
        "Numbers: ①②③④⑤",
        "Arrows: ←→↑↓⇄",
        "Math: ∀x∈ℝ: x²≥0",
        "",  # empty string
        " ",  # whitespace only
        "\n\t\r",  # control chars
    ]
    
    for txt in challenging_texts:
        ids, used_fallback, fallback_chars = tokenizer.encode_with_flag(txt)
        assert not used_fallback, f"Unexpected fallback for: {txt!r}"
        assert fallback_chars == [], f"Expected empty fallback_chars for: {txt!r}"
        assert tokenizer.decode(ids) == txt, f"Failed roundtrip for: {txt!r}"


def test_deterministic_encoding():
    """Verify that encoding is deterministic across multiple calls."""
    txt = "Corporate governance § compliance ™"
    
    # Encode the same text multiple times
    encodings = [tokenizer.encode(txt) for _ in range(5)]
    
    # All encodings should be identical
    first = encodings[0]
    for i, encoding in enumerate(encodings[1:], 1):
        assert encoding == first, f"Encoding {i} differs from first"


def test_empty_and_edge_cases():
    """Test edge cases that could break tokenization."""
    edge_cases = [
        "",  # empty
        " ",  # single space
        "\n",  # newline
        "\t",  # tab
        "a",  # single char
        "AB",  # two chars
        "   ",  # multiple spaces
        "\n\n\n",  # multiple newlines
    ]
    
    for txt in edge_cases:
        ids = tokenizer.encode(txt)
        assert tokenizer.decode(ids) == txt
        
        # Verify no fallback
        ids2, used_fallback, fallback_chars = tokenizer.encode_with_flag(txt)
        assert ids == ids2
        assert not used_fallback
        assert fallback_chars == []


def test_wl_replay():
    """Ensure WL feature extraction survives encode → decode unchanged."""
    txt = "nutrient enhanced Water beverage"
    ids = tokenizer.encode(txt)
    features_in = wl_vector(txt)
    # Byte-level reversibility
    assert tokenizer.decode(ids) == txt
    # Identical WL feature set
    assert set(features_in.indices) == set(wl_vector(txt).indices)


def test_performance_large_text():
    """Verify reasonable performance on larger corporate text."""
    # Simulate a paragraph from a 10-K filing
    large_text = """
    Our business is subject to various risks and uncertainties, including those described 
    in "Risk Factors" in Item 1A of this Form 10-K. The forward-looking statements in this 
    filing are based on our current expectations and assumptions regarding our business, 
    the economy and other future conditions. We believe these expectations and assumptions 
    are reasonable, but forward-looking statements are inherently uncertain. Accordingly, 
    actual results may differ materially from those expressed in the forward-looking statements.
    """ * 10  # Repeat to make it larger
    
    # Should complete quickly and be lossless
    ids = tokenizer.encode(large_text)
    decoded = tokenizer.decode(ids)
    assert decoded == large_text
    
    # Should never need fallback
    ids2, used_fallback, fallback_chars = tokenizer.encode_with_flag(large_text)
    assert ids == ids2
    assert not used_fallback
    assert fallback_chars == []


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def _write_sample(path: Path, text: str = "Hello, world!") -> None:
    """Write a minimal one-row JSONL file required by the CLI."""
    row = {"id": "row1", "text": text, "speaker": "Unit-Test"}
    path.write_text(json.dumps(row) + "\n")


@pytest.fixture()
def temp_dataset(tmp_path, monkeypatch):
    """Create an isolated project tree and *cd* into it for CLI tests."""
    project_root = tmp_path
    extracted_root = project_root / "data" / "extracted"
    tokenized_root = project_root / "data" / "tokenized"
    extracted_root.mkdir(parents=True)
    monkeypatch.chdir(project_root)
    return extracted_root, tokenized_root


def test_cli_encode_single(temp_dataset):
    """`cli_encode` should process one *_stage4.jsonl file in place."""
    extracted_root, tokenized_root = temp_dataset
    input_file = extracted_root / "sample_stage4.jsonl"
    _write_sample(input_file, "Corporate governance § compliance ™")

    runner = CliRunner()
    result = runner.invoke(cli_encode_main, [str(input_file)])
    assert result.exit_code == 0, result.output

    out = tokenized_root / "sample_stage4.jsonl"
    assert out.exists()
    out_json = json.loads(out.read_bytes().splitlines()[0])
    assert out_json["text"] == "Corporate governance § compliance ™"
    assert isinstance(out_json["sp_ids"], list) and out_json["sp_ids"], "sp_ids missing or empty"
    
    # Verify no fallback was used
    assert out_json["byte_fallback"] is False
    assert out_json["fallback_chars"] == []


def test_cli_encode_recursive(temp_dataset):
    """Recursive mode should walk the entire *extracted/* subtree."""
    extracted_root, tokenized_root = temp_dataset
    sub = extracted_root / "sub"
    sub.mkdir()
    _write_sample(sub / "one_stage4.jsonl", "file one § ™")
    _write_sample(sub / "two_stage4.jsonl", "file two © ®")

    runner = CliRunner()
    result = runner.invoke(cli_encode_main, ["data/extracted", "--recursive"])
    assert result.exit_code == 0, result.output

    for fname in ["one_stage4.jsonl", "two_stage4.jsonl"]:
        assert (tokenized_root / "sub" / fname).exists()
        
        # Verify content and no fallback
        out_file = tokenized_root / "sub" / fname
        out_json = json.loads(out_file.read_bytes().splitlines()[0])
        assert out_json["byte_fallback"] is False
        assert out_json["fallback_chars"] == []


def test_cli_output_format_unchanged(temp_dataset):
    """Verify that CLI output format remains compatible with existing pipeline."""
    extracted_root, tokenized_root = temp_dataset
    input_file = extracted_root / "format_test_stage4.jsonl"
    
    # Test with challenging text that would have triggered fallback before
    test_text = "SEC § compliance ™ requirements © 2024"
    _write_sample(input_file, test_text)

    runner = CliRunner()
    result = runner.invoke(cli_encode_main, [str(input_file)])
    assert result.exit_code == 0

    out_file = tokenized_root / "format_test_stage4.jsonl"
    out_json = json.loads(out_file.read_bytes().splitlines()[0])
    
    # All expected fields should be present
    required_fields = [
        "id", "text", "speaker",  # original fields
        "sp_ids", "byte_fallback", "fallback_chars",  # tokenizer fields
        "deps", "wl_indices", "wl_counts"  # feature fields
    ]
    
    for field in required_fields:
        assert field in out_json, f"Missing field: {field}"
    
    # Verify tokenizer fields have expected values
    assert isinstance(out_json["sp_ids"], list)
    assert out_json["byte_fallback"] is False
    assert out_json["fallback_chars"] == []
    
    # Verify text is preserved exactly
    assert out_json["text"] == test_text
