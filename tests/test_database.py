"""
Database unit tests for SQLite schema, health checks, candle persistence, and event logging.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from app.database.db import DatabaseManager, EXPECTED_TABLES


@pytest.fixture
def temp_db():
    """Fixture providing a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = DatabaseManager(db_path=db_path)
    yield db
    # Cleanup
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_database_initialization_creates_all_tables(temp_db):
    """Verify that all 6 required tables are created during initialization."""
    health = temp_db.check_health()
    assert health["status"] == "healthy"
    assert health["integrity"] == "ok"
    assert len(health["missing_tables"]) == 0

    existing_set = set(health["existing_tables"])
    for table in EXPECTED_TABLES:
        assert table in existing_set


def test_store_and_retrieve_candles(temp_db):
    """Verify upsert storing and querying of candlestick data into pandas DataFrame."""
    sample_candles = [
        {
            "symbol": "XRPUSDT",
            "resolution": "5m",
            "timestamp": 1700000000,
            "open": 0.50,
            "high": 0.52,
            "low": 0.49,
            "close": 0.51,
            "volume": 12000.0,
        },
        {
            "symbol": "XRPUSDT",
            "resolution": "5m",
            "timestamp": 1700000300,
            "open": 0.51,
            "high": 0.53,
            "low": 0.50,
            "close": 0.52,
            "volume": 15000.0,
        },
    ]

    stored = temp_db.store_candles(sample_candles)
    assert stored == 2

    # Query back as DataFrame
    df = temp_db.get_candles(symbol="XRPUSDT", resolution="5m")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "close" in df.columns
    assert "volume" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)

    # Test upsert update on conflict
    updated_candle = [
        {
            "symbol": "XRPUSDT",
            "resolution": "5m",
            "timestamp": 1700000000,
            "open": 0.50,
            "high": 0.55,  # updated high
            "low": 0.49,
            "close": 0.54,  # updated close
            "volume": 18000.0,
        }
    ]
    temp_db.store_candles(updated_candle)

    df_updated = temp_db.get_candles(symbol="XRPUSDT", resolution="5m")
    assert len(df_updated) == 2
    # Verify close price was updated
    assert df_updated.loc[df_updated["timestamp"] == 1700000000, "close"].iloc[0] == 0.54


def test_record_agent_event(temp_db):
    """Verify operational events are stored in agent_events table."""
    temp_db.record_event(
        event_type="HEALTH_CHECK",
        severity="INFO",
        message="System diagnostics completed successfully",
        symbol="XRPUSDT",
        timeframe="5m",
        metadata={"latency_ms": 42.5}
    )

    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_events WHERE event_type = 'HEALTH_CHECK';")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "XRPUSDT"
        assert rows[0]["severity"] == "INFO"
        assert "latency_ms" in rows[0]["metadata_json"]
