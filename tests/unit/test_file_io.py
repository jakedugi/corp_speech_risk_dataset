"""Tests for file I/O utilities."""

import json
from pathlib import Path

import pytest

from corp_speech_risk_dataset.utils.file_io import (
    ensure_dir,
    save_json,
    load_json,
    list_json_files,
    merge_json_files
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
        saved_data = json.load(f)
    assert saved_data == test_data

def test_load_json(tmp_dir):
    """Test JSON loading."""
    test_file = tmp_dir / "test.json"
    test_data = {"key": "value"}
    
    with open(test_file, "w") as f:
        json.dump(test_data, f)
    
    loaded_data = load_json(test_file)
    assert loaded_data == test_data

def test_list_json_files(tmp_dir):
    """Test JSON file listing."""
    # Create test files
    for i in range(3):
        file = tmp_dir / f"test_{i}.json"
        file.write_text("{}")
    
    # List files
    files = list_json_files(tmp_dir)
    assert len(files) == 3
    assert all(f.suffix == ".json" for f in files)

def test_merge_json_files(tmp_dir):
    """Test JSON file merging."""
    # Create test files
    files = []
    for i in range(3):
        file = tmp_dir / f"test_{i}.json"
        data = [{"id": i, "value": f"test_{i}"}]
        save_json(data, file)
        files.append(file)
    
    # Merge files
    output_file = tmp_dir / "merged.json"
    merge_json_files(files, output_file)
    
    # Check merged data
    merged_data = load_json(output_file)
    assert len(merged_data) == 3
    assert all("id" in item for item in merged_data) 