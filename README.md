# XRP AI Trading Agent (Delta Exchange) - Phase 1

An institutional-grade, modular crypto trading agent backend engineered specifically for XRP trading on [Delta Exchange](https://www.delta.exchange).

> **SAFETY INVARIANT:**  
> **Phase 1 is strictly limited to Infrastructure, Health Checks, and Read-Only Market Data.**  
> Order creation, order cancellation, leverage modification, and any form of automated or live order routing are **hard-disabled** at the code and network levels.  
> This system makes **no claims of guaranteed profits**, and strictly prohibits martingale, unlimited averaging, or uncontrolled leverage architectures.

---

## 1. Project Purpose

The purpose of Phase 1 is to establish a rock-solid, production-ready foundation for automated trading research:
- Safe, non-mutating interaction with Delta Exchange's REST API.
- Configurable contract symbol support (e.g. `XRPUSDT`, `XRP_USDT`, or perpetual futures).
- Reliable SQLite persistence for OHLCV candles, order states, positions, and operational audit events.
- Structured, sanitized logging ensuring credentials, private keys, and HMAC signatures are **never** leaked.
- Real-time health check diagnostics for exchange reachability, contract validity, and database integrity.
- Full compliance with Python 3.12+ type hints and Pydantic v2 schemas.

---

## 2. Architecture & Directory Structure

```
xrp_ai_trading_agent/
│
├── app/
│   ├── __init__.py                # Package metadata & phase version
│   ├── config.py                  # Pydantic Settings with strict safety validation
│   ├── logging_config.py          # Structured logging & credential redactor filter
│   ├── main.py                    # FastAPI application & /health endpoint
│   │
│   ├── exchange/
│   │   ├── __init__.py            # Exchange abstractions
│   │   ├── delta_client.py        # Delta Exchange REST client (Read-only + safety shields)
│   │   └── models.py              # Pydantic data schemas (Candle, Ticker, OrderBook, Health)
│   │
│   ├── data/
│   │   ├── __init__.py            # Market data abstractions
│   │   └── market_data.py         # MarketDataService (OHLCV, Pandas DataFrame, WS payload builder)
│   │
│   ├── strategy/                  # Strategy module (Reserved for Phase 3)
│   │   └── __init__.py
│   │
│   ├── ml/                        # Machine Learning models (Reserved for Phase 4)
│   │   └── __init__.py
│   │
│   ├── risk/                      # Risk management rules (Reserved for Phase 2/3)
│   │   └── __init__.py
│   │
│   ├── execution/                 # Order router & paper engine (Reserved for Phase 2)
│   │   └── __init__.py
│   │
│   └── database/
│       ├── __init__.py            # Database manager exports
│       └── db.py                  # SQLite schema creation (WAL mode), queries & event audit log
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py             # Config & env validation tests
│   ├── test_database.py           # SQLite schema & persistence tests
│   ├── test_exchange_client.py    # Delta API client & health check tests
│   ├── test_health_endpoint.py    # FastAPI /health endpoint tests
│   ├── test_market_data.py        # Data pipeline & WebSocket formatting tests
│   └── test_security_and_safety.py# Order blocking & secret protection tests
│
├── data/                          # Local SQLite database storage
├── models/                        # Serialized ML model weights (future phases)
├── logs/                          # Structured JSON log files
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules (secrets, databases, logs, caches)
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Documentation & operational guide
└── run.py                         # CLI runner for server & standalone diagnostics
```

---

## 3. Database Schema (SQLite)

The SQLite database (`data/trading_agent.db`) operates in Write-Ahead Logging (`WAL`) mode with foreign key checks and creates 6 dedicated tables:

1. **`market_candles`**: Historical and streaming OHLCV bars (`symbol`, `resolution`, `timestamp`, `open`, `high`, `low`, `close`, `volume`) with a `UNIQUE(symbol, resolution, timestamp)` constraint.
2. **`trades`**: Public and historical executed trades (`trade_id`, `symbol`, `side`, `price`, `size`, `timestamp`).
3. **`orders`**: Tracked orders (paper-trading mode) with execution states.
4. **`positions`**: Position sizing, entry price, liquidation price, mark price, and unrealized/realized PnL tracking.
5. **`model_predictions`**: Signal predictions, confidence scores, and feature snapshots for ML observability.
6. **`agent_events`**: Structured audit events (`event_type`, `severity`, `message`, `metadata_json`, `timestamp`).

---

## 4. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Configurable Environment Variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DELTA_BASE_URL` | `https://api.delta.exchange` | REST API URL (`https://testnet-api.delta.exchange` for testnet) |
| `DELTA_WS_URL` | `wss://socket.delta.exchange` | WebSocket URL for streaming |
| `DELTA_API_KEY` | *(empty)* | Optional in Phase 1 (required only for private read-only endpoints) |
| `DELTA_API_SECRET` | *(empty)* | Optional in Phase 1 (never logged or exposed) |
| `DELTA_SYMBOL` | `XRPUSDT` | Target contract symbol (e.g. `XRPUSDT`, `XRP_USDT`) |
| `TIMEFRAME` | `5m` | Supported: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `1d` |
| `PAPER_TRADING` | `true` | **Hard-locked to `true` in Phase 1** |
| `LIVE_TRADING_ENABLED` | `false` | **Hard-locked to `false` in Phase 1** |
| `DB_PATH` | `data/trading_agent.db` | Path to SQLite database file |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `HOST` | `127.0.0.1` | FastAPI server bind address |
| `PORT` | `8000` | FastAPI server bind port |

---

## 5. Installation

Ensure Python 3.12+ is installed.

```bash
# Optional: Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. How to Run

### Mode A: Standalone Health Check Diagnostics (CLI)

Run immediate terminal diagnostics verifying database schema, network reachability, Delta Exchange contract validation, and server time:

```bash
python run.py --health-check
```

Example Output:
```text
======================================================================
  XRP AI Trading Agent - Diagnostics (Phase 1 - Infrastructure Only)
======================================================================
Configured Symbol:     XRPUSDT
Configured Timeframe:  5m
Delta Base URL:        https://api.delta.exchange
Paper Trading:         True
Live Trading Enabled:  False
Auth Configured:       False
Database Path:         data/trading_agent.db
----------------------------------------------------------------------
Checking Database...
  DB Status:           HEALTHY
  DB Integrity:        ok
  Tables Found:        agent_events, market_candles, model_predictions, orders, positions, trades

Checking Delta Exchange Connectivity...
  Exchange Status:     HEALTHY
  Reachable:           True (185.4 ms)
  Symbol Valid:        True
  Available XRP Pairs: XRPUSDT, XRP_USDT
  Server Time:         Mon, 28 Sep 2026 07:29:23 GMT
======================================================================
DIAGNOSTICS PASSED: System ready for Phase 1 read-only operation.
```

### Mode B: FastAPI Server

Start the REST API server:

```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Health Check Endpoint: `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

Sample JSON response:
```json
{
  "application_status": "healthy",
  "paper_trading": true,
  "configured_symbol": "XRPUSDT",
  "configured_timeframe": "5m",
  "exchange_connectivity": {
    "status": "healthy",
    "base_url": "https://api.delta.exchange",
    "reachable": true,
    "latency_ms": 192.4,
    "auth_configured": false,
    "xrp_symbol_configured": "XRPUSDT",
    "xrp_symbol_valid": true,
    "available_xrp_products": ["XRPUSDT", "XRP_USDT"],
    "server_time": "Mon, 28 Sep 2026 07:29:23 GMT",
    "errors": []
  },
  "database_status": {
    "status": "healthy",
    "integrity": "ok",
    "table_count": 6
  },
  "safety_mode": "READ_ONLY_PAPER_TRADING",
  "live_trading_enabled": false,
  "timestamp": "2026-09-28T07:31:00.123456+00:00"
}
```

---

## 7. How to Run Tests

Run the full pytest suite:

```bash
python -m pytest -v
```

All 28 tests cover:
- Pydantic configuration validation, custom symbol support, and invalid timeframe rejection.
- Safety guards rejecting `PAPER_TRADING=false` or `LIVE_TRADING_ENABLED=true`.
- Secret masking in logs, exceptions, and representation strings.
- SQLite schema generation and upsert persistence.
- Delta Exchange client public API requests, signature calculations, and health reports.
- Hard block on order creation, cancellation, and mutating HTTP verbs (`Phase1SafetyViolationError`).
- FastAPI `GET /health` endpoint response verification.

---

## 8. Current Limitations & Safety Invariants

1. **NO Live Trading:** Phase 1 code contains hard shields (`Phase1SafetyViolationError`) blocking `create_order`, `cancel_order`, and `cancel_all_orders`.
2. **Read-Only / Paper Only:** The system is incapable of submitting transactions to the exchange.
3. **No Automated Strategy Loops:** Signal generators, ML forecasting, and backtesters are intentionally deferred to Phases 2, 3, and 4.
4. **No Unsound Financial Mechanisms:** The project architecture forbids martingale doubling, unlimited averaging, and uncontrolled leverage.
