"""Tests for the CourtListener API client."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.api.courtlistener import CourtListenerClient
from src.custom_types.base_types import APIConfig

@pytest.fixture
def mock_response():
    """Mock API response."""
    return {
        "count": 1,
        "next": None,
        "previous": None,
        "results": [{
            "id": 1,
            "case_name": "Test Case",
            "date_filed": "2020-01-01"
        }]
    }

@pytest.fixture
def client():
    """Create a test client."""
    config = APIConfig(api_key="test_token")
    return CourtListenerClient(config)

@pytest.fixture
def mock_resource_response():
    return {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {"id": 1, "type": "opinion", "value": "A"},
            {"id": 2, "type": "opinion", "value": "B"}
        ]
    }

def test_build_headers(client):
    """Test header construction."""
    headers = client._build_headers()
    assert headers["Authorization"] == "Token test_token"
    assert "Accept" in headers

@patch("requests.get")
def test_fetch_dockets(mock_get, client, mock_response):
    """Test docket fetching using fetch_resource."""
    # Patch _get to simulate API response
    client._get = MagicMock(return_value=mock_response)
    client.endpoints["dockets"] = "http://fake/dockets/"
    dockets = client.fetch_resource("dockets", {"search": "Test Statute"})
    assert len(dockets) == 1
    assert dockets[0]["case_name"] == "Test Case"

@patch("requests.get")
def test_fetch_opinions(mock_get, client, mock_response):
    """Test opinion fetching using fetch_resource."""
    client._get = MagicMock(return_value=mock_response)
    client.endpoints["opinions"] = "http://fake/opinions/"
    opinions = client.fetch_resource("opinions", {"search": "Test"})
    assert len(opinions) == 1
    assert opinions[0]["id"] == 1

def test_process_statutes(tmp_path, client):
    """Test statute processing."""
    # TODO: Implement full test with mocked API calls
    pass 

def test_fetch_resource(monkeypatch, client, mock_resource_response):
    """Test the unified fetch_resource method."""
    calls = []
    def fake_get(url, params=None):
        calls.append((url, params))
        return mock_resource_response
    monkeypatch.setattr(client, "_get", fake_get)
    client.endpoints["opinions"] = "http://fake/opinions/"
    results = client.fetch_resource("opinions", {"search": "test"})
    assert len(results) == 2
    assert results[0]["type"] == "opinion"
    assert calls[0][0] == "http://fake/opinions/"
    assert calls[0][1] == {"search": "test"} 