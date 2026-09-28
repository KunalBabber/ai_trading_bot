"""
Integration tests for FastAPI application and GET /health endpoint.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns application information and Phase 1 metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert "Phase 1" in data["phase"]
    assert data["paper_trading"] is True
    assert data["live_trading_enabled"] is False


def test_health_endpoint_structure():
    """
    Verify GET /health returns all required fields:
    - application_status
    - paper_trading status
    - configured symbol
    - configured timeframe
    - exchange connectivity status
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Core required fields from requirements
    assert "application_status" in data
    assert "paper_trading" in data
    assert data["paper_trading"] is True
    assert "configured_symbol" in data
    assert data["configured_symbol"] == "XRPUSDT"
    assert "configured_timeframe" in data
    assert data["configured_timeframe"] == "5m"
    assert "exchange_connectivity" in data

    # Verify exchange connectivity details
    ex = data["exchange_connectivity"]
    assert "reachable" in ex
    assert "status" in ex
    assert "xrp_symbol_configured" in ex
    assert "xrp_symbol_valid" in ex

    # Verify database status details
    assert "database_status" in data
    assert data["database_status"]["status"] == "healthy"
    assert data["database_status"]["integrity"] == "ok"

    # Verify safety mode
    assert data["safety_mode"] == "READ_ONLY_PAPER_TRADING"
    assert data["live_trading_enabled"] is False
