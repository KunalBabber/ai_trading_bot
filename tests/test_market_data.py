"""
Unit tests for MarketDataService and WebSocket client payload generation.
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from app.data.market_data import DeltaWebSocketClient, MarketDataService
from app.exchange.models import CandleData, TickerData


def test_websocket_client_subscription_payload():
    """Verify WebSocket client builds correct subscription structure for Delta Exchange."""
    ws_client = DeltaWebSocketClient(symbol="XRPUSDT", timeframe="5m")
    payload = ws_client.build_subscription_payload()

    assert payload["type"] == "subscribe"
    channels = payload["payload"]["channels"]
    assert len(channels) == 3

    channel_names = [c["name"] for c in channels]
    assert "v2/ticker" in channel_names
    assert "candlestick_5m" in channel_names
    assert "l2_orderbook" in channel_names

    for c in channels:
        assert c["symbols"] == ["XRPUSDT"]


def test_market_data_service_dataframe_conversion():
    """Verify MarketDataService correctly converts candles to a pandas DataFrame."""
    mock_exchange = MagicMock()
    mock_candles = [
        CandleData(
            symbol="XRPUSDT",
            resolution="5m",
            timestamp=1700000000,
            open=0.50,
            high=0.55,
            low=0.48,
            close=0.52,
            volume=25000.0
        ),
        CandleData(
            symbol="XRPUSDT",
            resolution="5m",
            timestamp=1700000300,
            open=0.52,
            high=0.56,
            low=0.51,
            close=0.54,
            volume=30000.0
        ),
    ]
    mock_exchange.get_candles.return_value = mock_candles

    mock_db = MagicMock()
    service = MarketDataService(client=mock_exchange, db=mock_db)

    df = service.get_candles_dataframe(symbol="XRPUSDT", resolution="5m", from_exchange=True)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.loc[df["timestamp"] == 1700000000, "close"].iloc[0] == 0.52

    # Verify db.store_candles was called for persistence
    assert mock_db.store_candles.called
