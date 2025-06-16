"""Tests for the CourtListener API client."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.api.courtlistener_client import CourtListenerClient

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
    return CourtListenerClient("test_token")

def test_build_headers(client):
    """Test header construction."""
    headers = client._build_headers()
    assert headers["Authorization"] == "Token test_token"
    assert "Accept" in headers
    assert "Content-Type" in headers

@patch("requests.get")
def test_fetch_dockets(mock_get, client, mock_response):
    """Test docket fetching."""
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    
    dockets = client.fetch_dockets(
        statute="Test Statute",
        page=1,
        page_size=100,
        date_filed_min="2020-01-01"
    )
    
    assert len(dockets) == 1
    assert dockets[0]["case_name"] == "Test Case"

@patch("requests.get")
def test_fetch_opinions(mock_get, client, mock_response):
    """Test opinion fetching."""
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200
    
    opinions = client.fetch_opinions(
        docket_id=1,
        page=1,
        page_size=100
    )
    
    assert len(opinions) == 1
    assert opinions[0]["id"] == 1

def test_process_statutes(tmp_path, client):
    """Test statute processing."""
    # TODO: Implement full test with mocked API calls
    pass 