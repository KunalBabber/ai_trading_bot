"""
Unit tests for DeltaExchangeClient and Exchange models.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.config import Settings
from app.exchange.delta_client import DeltaExchangeClient
from app.exchange.models import ExchangeHealthReport, ProductInfo, TickerData


def test_client_init_success():
    """Verify DeltaExchangeClient initializes with valid settings."""
    settings = Settings(DELTA_SYMBOL="XRPUSDT")
    client = DeltaExchangeClient(settings=settings)
    assert client.symbol == "XRPUSDT"
    assert client.base_url == "https://api.delta.exchange"


def test_signature_generation():
    """Verify signature generation computes valid HMAC-SHA256."""
    settings = Settings(
        DELTA_API_KEY="test_key",
        DELTA_API_SECRET="test_secret"
    )
    client = DeltaExchangeClient(settings=settings)
    sig, ts = client._generate_signature(
        method="GET",
        path="/v2/wallet/balances",
        timestamp="1700000000"
    )
    assert len(sig) == 64  # SHA256 hex string is 64 characters
    assert ts == "1700000000"


def test_signature_unconfigured_auth():
    """Verify signature returns empty when auth credentials are not set."""
    settings = Settings(DELTA_API_KEY=None, DELTA_API_SECRET=None)
    client = DeltaExchangeClient(settings=settings)
    sig, ts = client._generate_signature("GET", "/v2/wallet/balances")
    assert sig == ""
    assert ts == ""


def test_check_health_mocked_success():
    """Verify check_health parses successful responses into ExchangeHealthReport."""
    mock_products = [
        ProductInfo(id=1, symbol="BTCUSDT"),
        ProductInfo(id=2, symbol="XRPUSDT"),
        ProductInfo(id=3, symbol="XRP_USDT"),
    ]

    client = DeltaExchangeClient()
    with patch.object(client, "get_products", return_value=mock_products):
        with patch.object(client, "get_server_time", return_value="Mon, 28 Sep 2026 07:00:00 GMT"):
            report = client.check_health()
            assert isinstance(report, ExchangeHealthReport)
            assert report.status == "healthy"
            assert report.reachable is True
            assert report.xrp_symbol_valid is True
            assert "XRPUSDT" in report.available_xrp_products
            assert "XRP_USDT" in report.available_xrp_products
            assert len(report.errors) == 0


def test_check_health_symbol_not_found():
    """Verify check_health identifies when configured symbol is absent."""
    mock_products = [
        ProductInfo(id=1, symbol="BTCUSDT"),
        ProductInfo(id=2, symbol="ETHUSDT"),
    ]

    client = DeltaExchangeClient(settings=Settings(DELTA_SYMBOL="NON_EXISTENT_XRP"))
    with patch.object(client, "get_products", return_value=mock_products):
        with patch.object(client, "get_server_time", return_value="Mon, 28 Sep 2026 07:00:00 GMT"):
            report = client.check_health()
            assert report.status == "degraded"
            assert report.reachable is True
            assert report.xrp_symbol_valid is False
            assert any("NON_EXISTENT_XRP" in err for err in report.errors)


def test_get_ticker_model_mapping():
    """Verify get_ticker correctly maps API result to TickerData."""
    client = DeltaExchangeClient()
    mock_response = {
        "success": True,
        "result": {
            "symbol": "XRPUSDT",
            "mark_price": "1.4850",
            "spot_price": "1.4845",
            "bid": "1.4848",
            "ask": "1.4852",
            "volume_24h": "5000000",
            "timestamp": 1700000000000000  # microseconds
        }
    }

    with patch.object(client, "_request", return_value=mock_response):
        ticker = client.get_ticker("XRPUSDT")
        assert isinstance(ticker, TickerData)
        assert ticker.symbol == "XRPUSDT"
        assert ticker.mark_price == 1.4850
        assert ticker.spot_price == 1.4845
        assert ticker.timestamp == 1700000000
