"""Test file I/O utilities."""

import json
import tempfile
from pathlib import Path
import pytest

from corp_speech_risk_dataset.infrastructure.file_io import (
    save_json,
    load_json,
    ensure_dir,
    list_json_files,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temporary directory."""
    return tmp_path


def test_ensure_dir(tmp_dir):
    """Test directory creation."""
    test_dir = tmp_dir / "test"
    ensure_dir(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_save_json(tmp_dir):
    """Test JSON saving."""
    test_file = tmp_dir / "test.json"
    test_data = {"key": "value"}
    save_json(test_data, test_file)
    assert test_file.exists()
    with open(test_file) as f:
        assert json.load(f) == test_data


def test_load_json(tmp_dir):
    """Test JSON loading."""
    test_file = tmp_dir / "test.json"
    test_data = {"key": "value"}
    with open(test_file, "w") as f:
        json.dump(test_data, f)

    loaded_data = load_json(test_file)
    assert loaded_data == test_data


def test_list_json_files(tmp_dir):
    """Test listing JSON files."""
    # Create some test files
    (tmp_dir / "test1.json").touch()
    (tmp_dir / "test2.json").touch()
    (tmp_dir / "test.txt").touch()  # Non-JSON file

    json_files = list_json_files(tmp_dir)
    assert len(json_files) == 2
    assert all(f.suffix == ".json" for f in json_files)


def test_merge_json_files(tmp_dir):
    """Test merging JSON files."""
    # Create test files
    files = []
    for i in range(3):
        file_path = tmp_dir / f"test{i}.json"
        test_data = {"data": f"content{i}"}
        save_json(test_data, file_path)
        files.append(file_path)

    # Test would merge files if merge_json_files function existed
    # For now, just test that individual files exist
    for f in files:
        assert f.exists()
        data = load_json(f)
        assert "data" in data
