"""
Market Data Service and WebSocket interface abstraction.

Phase 1 Invariant:
- Provides clean high-level market data access: ticker, OHLCV candles, order book, trades.
- Generates Pandas DataFrames for downstream quantitative and ML pipelines.
- Implements WebSocket subscription schema & message parsing for Delta Exchange streaming.
- Does NOT execute trading logic or automated execution loops.
"""

import json
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from app.config import Settings, get_settings
from app.database.db import DatabaseManager, get_db
from app.exchange.delta_client import DeltaExchangeClient
from app.exchange.models import CandleData, OrderBookData, TickerData, TradeData
from app.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataService:
    """
    Unified high-level market data interface for historical data, snapshots,
    and optional database persistence.
    """

    def __init__(
        self,
        client: Optional[DeltaExchangeClient] = None,
        db: Optional[DatabaseManager] = None,
        settings: Optional[Settings] = None
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or DeltaExchangeClient(settings=self.settings)
        self.db = db or get_db(self.settings.DB_PATH)
        self.symbol = self.settings.DELTA_SYMBOL
        self.timeframe = self.settings.TIMEFRAME

    def fetch_ticker(self, symbol: Optional[str] = None) -> TickerData:
        """Fetch current ticker price and volume snapshot."""
        target_symbol = symbol or self.symbol
        ticker = self.client.get_ticker(target_symbol)
        logger.debug(
            f"Fetched ticker for {target_symbol}: mark={ticker.mark_price}",
            extra={"event": "TICKER_FETCHED", "symbol": target_symbol}
        )
        return ticker

    def fetch_candles(
        self,
        symbol: Optional[str] = None,
        resolution: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 100,
        persist: bool = True
    ) -> List[CandleData]:
        """
        Fetch OHLCV historical candlestick data and optionally cache into SQLite database.
        """
        target_symbol = symbol or self.symbol
        target_res = resolution or self.timeframe

        candles = self.client.get_candles(
            symbol=target_symbol,
            resolution=target_res,
            start=start,
            end=end,
            limit=limit
        )

        if persist and candles and self.db:
            records = [
                {
                    "symbol": c.symbol,
                    "resolution": c.resolution,
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in candles
            ]
            stored_count = self.db.store_candles(records)
            logger.debug(
                f"Stored {stored_count} candles for {target_symbol} ({target_res}) into database",
                extra={"event": "CANDLES_PERSISTED", "symbol": target_symbol, "timeframe": target_res}
            )

        return candles

    def get_candles_dataframe(
        self,
        symbol: Optional[str] = None,
        resolution: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 100,
        from_exchange: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve candles as a structured pandas DataFrame with DatetimeIndex and float columns.
        """
        target_symbol = symbol or self.symbol
        target_res = resolution or self.timeframe

        if from_exchange:
            candles = self.fetch_candles(
                symbol=target_symbol,
                resolution=target_res,
                start=start,
                end=end,
                limit=limit,
                persist=True
            )
            if not candles:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

            data = [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in candles
            ]
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("datetime", inplace=True)
            return df
        else:
            return self.db.get_candles(
                symbol=target_symbol,
                resolution=target_res,
                start_ts=start,
                end_ts=end,
                limit=limit
            )

    def fetch_orderbook(self, symbol: Optional[str] = None, depth: int = 50) -> OrderBookData:
        """Fetch Layer 2 order book snapshot."""
        target_symbol = symbol or self.symbol
        return self.client.get_orderbook(target_symbol, depth=depth)

    def fetch_recent_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[TradeData]:
        """Fetch latest public executed trades."""
        target_symbol = symbol or self.symbol
        return self.client.get_trades(target_symbol, limit=limit)


class DeltaWebSocketClient:
    """
    WebSocket client interface abstraction for real-time market data streaming.
    Prepares channels and subscription payloads for Delta Exchange in Phase 2.
    """

    def __init__(
        self,
        ws_url: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> None:
        settings = get_settings()
        self.ws_url = ws_url or settings.DELTA_WS_URL
        self.symbol = symbol or settings.DELTA_SYMBOL
        self.timeframe = timeframe or settings.TIMEFRAME
        self._is_running = False

    def build_subscription_payload(
        self,
        channels: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Construct Delta Exchange WebSocket subscription message.
        Channels example: ['v2/ticker', 'candlestick_5m', 'l2_orderbook']
        """
        target_symbols = symbols or [self.symbol]
        channel_names = channels or [
            "v2/ticker",
            f"candlestick_{self.timeframe}",
            "l2_orderbook"
        ]

        channel_objects = [
            {"name": ch, "symbols": target_symbols}
            for ch in channel_names
        ]

        return {
            "type": "subscribe",
            "payload": {
                "channels": channel_objects
            }
        }

    def parse_message(self, raw_message: str) -> Dict[str, Any]:
        """Parse raw incoming WebSocket JSON messages."""
        try:
            return json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode WebSocket message: {e}", extra={"event": "WS_DECODE_ERROR"})
            return {"error": "Invalid JSON", "raw": raw_message}
