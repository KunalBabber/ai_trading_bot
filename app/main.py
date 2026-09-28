"""
FastAPI application for XRP AI Trading Agent.

Phase 1 Invariant:
- Operates in strict read-only / paper-trading mode.
- Provides GET /health returning application status, paper trading status,
  configured symbol, configured timeframe, and exchange connectivity status.
- Never accepts or routes live orders.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database.db import get_db
from app.exchange.delta_client import DeltaExchangeClient
from app.logging_config import get_logger, setup_logging

settings = get_settings()
logger = get_logger(__name__, symbol=settings.DELTA_SYMBOL, timeframe=settings.TIMEFRAME)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown procedures."""
    # 1. Initialize logging
    setup_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
    logger.info(
        f"Starting {settings.APP_NAME} in [{settings.PHASE}] mode",
        extra={"event": "APP_STARTUP"}
    )

    # 2. Initialize database schema
    db = get_db(settings.DB_PATH)
    db_health = db.check_health()
    logger.info(
        f"Database initialized: status={db_health['status']} tables={len(db_health['existing_tables'])}",
        extra={"event": "DB_STARTUP"}
    )

    # 3. Log safety invariants
    logger.info(
        f"Safety Invariant: paper_trading={settings.PAPER_TRADING}, live_trading={settings.LIVE_TRADING_ENABLED}",
        extra={"event": "SAFETY_VERIFIED"}
    )

    yield

    # Clean shutdown
    logger.info("Application shutting down cleanly.", extra={"event": "APP_SHUTDOWN"})


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="XRP AI Trading Agent Infrastructure - Phase 1 Read-Only & Health Monitoring",
    lifespan=lifespan
)


@app.get("/", tags=["Info"])
def read_root() -> Dict[str, Any]:
    """Root endpoint returning basic agent identity and phase information."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "phase": settings.PHASE,
        "paper_trading": settings.PAPER_TRADING,
        "live_trading_enabled": settings.LIVE_TRADING_ENABLED,
        "symbol": settings.DELTA_SYMBOL,
        "timeframe": settings.TIMEFRAME,
        "documentation": "/docs"
    }


@app.get("/health", tags=["Monitoring"])
def get_health() -> Dict[str, Any]:
    """
    Comprehensive health status endpoint.
    Returns:
    - application_status
    - paper_trading status
    - configured symbol
    - configured timeframe
    - exchange connectivity status
    - database status
    """
    # 1. Verify database integrity
    db = get_db(settings.DB_PATH)
    db_health = db.check_health()

    # 2. Check Delta Exchange connectivity and symbol configuration
    client = DeltaExchangeClient(settings=settings)
    exchange_report = client.check_health()

    # 3. Assess overall system health
    app_is_healthy = (
        db_health["status"] == "healthy"
        and exchange_report.reachable
        and exchange_report.xrp_symbol_valid
    )

    application_status = "healthy" if app_is_healthy else (
        "degraded" if exchange_report.reachable else "unhealthy"
    )

    response_payload = {
        "application_status": application_status,
        "paper_trading": settings.PAPER_TRADING,
        "configured_symbol": settings.DELTA_SYMBOL,
        "configured_timeframe": settings.TIMEFRAME,
        "exchange_connectivity": exchange_report.model_dump(),
        "database_status": {
            "status": db_health["status"],
            "integrity": db_health["integrity"],
            "table_count": len(db_health["existing_tables"])
        },
        "safety_mode": "READ_ONLY_PAPER_TRADING",
        "live_trading_enabled": settings.LIVE_TRADING_ENABLED,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return response_payload
