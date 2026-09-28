"""
SQLite Database interface and schema management.

Phase 1 Invariant:
- Provides robust persistent schemas for market_candles, trades, orders, positions,
  model_predictions, and agent_events.
- Does NOT seed or insert fake trading results.
- Supports health verification, WAL journaling, and structured event recording.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import pandas as pd

from app.logging_config import get_logger

logger = get_logger(__name__)

CREATE_TABLES_SQL = """
-- 1. Historical & Streaming Market Candles
CREATE TABLE IF NOT EXISTS market_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    resolution TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_candle UNIQUE (symbol, resolution, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON market_candles(symbol, resolution, timestamp);

-- 2. Raw Public & Executed Trades
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp);

-- 3. Orders (Paper trading in Phase 1 / Non-live)
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    client_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL,
    size REAL NOT NULL,
    status TEXT NOT NULL,
    filled_size REAL DEFAULT 0.0,
    average_fill_price REAL DEFAULT 0.0,
    paper_trading INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_orders_lookup ON orders(symbol, status);

-- 4. Portfolio Positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    side TEXT NOT NULL,
    size REAL NOT NULL,
    entry_price REAL NOT NULL,
    mark_price REAL NOT NULL,
    liquidation_price REAL,
    unrealized_pnl REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    paper_trading INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Machine Learning Model Predictions
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    prediction_signal TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_output TEXT,
    features_snapshot TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_predictions_lookup ON model_predictions(model_name, symbol, timestamp);

-- 6. Structured Agent Operational Events
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    symbol TEXT,
    timeframe TEXT,
    message TEXT NOT NULL,
    metadata_json TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_events ON agent_events(event_type, timestamp);
"""

EXPECTED_TABLES = {
    "market_candles",
    "trades",
    "orders",
    "positions",
    "model_predictions",
    "agent_events"
}


class DatabaseManager:
    """
    Manages SQLite connections, schema initialization, and transactional queries.
    """

    def __init__(self, db_path: str = "data/trading_agent.db") -> None:
        self.db_path = db_path
        self._ensure_dir()
        self.init_db()

    def _ensure_dir(self) -> None:
        if self.db_path != ":memory:":
            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields an optimized SQLite connection with row factory and WAL mode."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=15.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables and verify schema integrity."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(CREATE_TABLES_SQL)
            conn.commit()
        logger.info(
            f"Database initialized successfully at {self.db_path}",
            extra={"event": "DATABASE_INIT"}
        )

    def check_health(self) -> Dict[str, Any]:
        """
        Verify database readability, write integrity, and presence of all required tables.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                )
                existing_tables = {row["name"] for row in cursor.fetchall()}

                missing = EXPECTED_TABLES - existing_tables
                cursor.execute("PRAGMA quick_check;")
                integrity_row = cursor.fetchone()
                integrity = integrity_row[0] if integrity_row else "unknown"

                is_healthy = (len(missing) == 0) and (integrity == "ok")

                return {
                    "status": "healthy" if is_healthy else "degraded",
                    "db_path": self.db_path,
                    "existing_tables": sorted(list(existing_tables)),
                    "missing_tables": sorted(list(missing)),
                    "integrity": integrity,
                    "error": None
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}", extra={"event": "DATABASE_ERROR"})
            return {
                "status": "unhealthy",
                "db_path": self.db_path,
                "existing_tables": [],
                "missing_tables": sorted(list(EXPECTED_TABLES)),
                "integrity": "failed",
                "error": str(e)
            }

    def record_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store an operational or safety event into agent_events."""
        query = """
        INSERT INTO agent_events (event_type, severity, symbol, timeframe, message, metadata_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {}, default=str)

        try:
            with self.get_connection() as conn:
                conn.execute(query, (event_type, severity, symbol, timeframe, message, metadata_str, now_iso))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record event in database: {e}", extra={"event": "DATABASE_ERROR"})

    def store_candles(self, candles: List[Dict[str, Any]]) -> int:
        """
        Store candle records into market_candles using upsert.
        Expects list of dicts with: symbol, resolution, timestamp, open, high, low, close, volume.
        """
        if not candles:
            return 0

        query = """
        INSERT INTO market_candles (symbol, resolution, timestamp, open, high, low, close, volume)
        VALUES (:symbol, :resolution, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT(symbol, resolution, timestamp) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, candles)
            conn.commit()
            return cursor.rowcount

    def get_candles(
        self,
        symbol: str,
        resolution: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = 500
    ) -> pd.DataFrame:
        """
        Query stored candles into a pandas DataFrame indexed by timestamp.
        """
        query = "SELECT timestamp, open, high, low, close, volume FROM market_candles WHERE symbol = ? AND resolution = ?"
        params: List[Any] = [symbol, resolution]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.set_index("datetime", inplace=True)
        return df


_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: Optional[str] = None) -> DatabaseManager:
    """Singleton getter for DatabaseManager."""
    global _db_instance
    if _db_instance is None or (db_path and _db_instance.db_path != db_path):
        from app.config import get_settings
        settings = get_settings()
        target_path = db_path or settings.DB_PATH
        _db_instance = DatabaseManager(db_path=target_path)
    return _db_instance
