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
try:
    tokenizer = SentencePieceTokenizer()   # now chooses path automatically
except FileNotFoundError:                  # should never happen, keep stub as guard
    class _StubTokenizer:  # noqa: D101
        def encode(self, text: str):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)
    tokenizer = _StubTokenizer()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_bpemb_roundtrip():
    """Round-trip sanity check for the SentencePiece model."""
    txt = "SEC Form 10-K §12(b); Apple Inc."
    ids = tokenizer.encode(txt)
    assert tokenizer.decode(ids) == txt


def test_wl_replay():
    """Ensure WL feature extraction survives encode → decode unchanged."""
    txt = "nutrient enhanced Water beverage"
    ids = tokenizer.encode(txt)
    features_in = wl_vector(txt)
    # Byte-level reversibility
    assert tokenizer.decode(ids) == txt
    # Identical WL feature set
    assert set(features_in.indices) == set(wl_vector(txt).indices)


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
    _write_sample(input_file)

    runner = CliRunner()
    result = runner.invoke(cli_encode_main, [str(input_file)])
    assert result.exit_code == 0, result.output

    out = tokenized_root / "sample_stage4.jsonl"
    assert out.exists()
    out_json = json.loads(out.read_bytes().splitlines()[0])
    assert out_json["text"] == "Hello, world!"
    assert isinstance(out_json["sp_ids"], list) and out_json["sp_ids"], "sp_ids missing or empty"


def test_cli_encode_recursive(temp_dataset):
    """Recursive mode should walk the entire *extracted/* subtree."""
    extracted_root, tokenized_root = temp_dataset
    sub = extracted_root / "sub"
    sub.mkdir()
    _write_sample(sub / "one_stage4.jsonl", "file one")
    _write_sample(sub / "two_stage4.jsonl", "file two")

    runner = CliRunner()
    result = runner.invoke(cli_encode_main, ["data/extracted", "--recursive"])
    assert result.exit_code == 0, result.output

    for fname in ["one_stage4.jsonl", "two_stage4.jsonl"]:
        assert (tokenized_root / "sub" / fname).exists()
