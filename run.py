"""
Entry point to run the XRP AI Trading Agent FastAPI server or execute standalone diagnostics.

Usage:
  python run.py                   # Starts FastAPI server
  python run.py --health-check    # Runs immediate CLI health & connectivity check
"""

import argparse
import sys
import uvicorn

from app.config import get_settings
from app.database.db import get_db
from app.exchange.delta_client import DeltaExchangeClient
from app.logging_config import setup_logging


def run_standalone_health_check() -> int:
    """Run immediate health and connectivity diagnostics in terminal."""
    settings = get_settings()
    setup_logging(level=settings.LOG_LEVEL)

    print("=" * 70)
    print(f"  {settings.APP_NAME} - Diagnostics ({settings.PHASE})")
    print("=" * 70)
    print(f"Configured Symbol:     {settings.DELTA_SYMBOL}")
    print(f"Configured Timeframe:  {settings.TIMEFRAME}")
    print(f"Delta Base URL:        {settings.DELTA_BASE_URL}")
    print(f"Paper Trading:         {settings.PAPER_TRADING}")
    print(f"Live Trading Enabled:  {settings.LIVE_TRADING_ENABLED}")
    print(f"Auth Configured:       {settings.is_auth_configured}")
    print(f"Database Path:         {settings.DB_PATH}")
    print("-" * 70)

    # 1. Database Check
    print("Checking Database...")
    db = get_db(settings.DB_PATH)
    db_health = db.check_health()
    print(f"  DB Status:           {db_health['status'].upper()}")
    print(f"  DB Integrity:        {db_health['integrity']}")
    print(f"  Tables Found:        {', '.join(db_health['existing_tables'])}")

    # 2. Exchange Check
    print("\nChecking Delta Exchange Connectivity...")
    client = DeltaExchangeClient(settings=settings)
    ex_report = client.check_health()
    print(f"  Exchange Status:     {ex_report.status.upper()}")
    print(f"  Reachable:           {ex_report.reachable} ({ex_report.latency_ms} ms)")
    print(f"  Symbol Valid:        {ex_report.xrp_symbol_valid}")
    print(f"  Available XRP Pairs: {', '.join(ex_report.available_xrp_products)}")
    print(f"  Server Time:         {ex_report.server_time}")

    if ex_report.errors:
        print("\nWarnings / Errors:")
        for err in ex_report.errors:
            print(f"  - {err}")

    print("=" * 70)
    if db_health["status"] == "healthy" and ex_report.reachable:
        print("DIAGNOSTICS PASSED: System ready for Phase 1 read-only operation.")
        return 0
    else:
        print("DIAGNOSTICS FAILED: Connectivity or configuration issue detected.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XRP AI Trading Agent")
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run immediate health check and exit"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Custom bind host (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Custom bind port (overrides config)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for development"
    )

    args = parser.parse_args()

    if args.health_check:
        exit_code = run_standalone_health_check()
        sys.exit(exit_code)

    settings = get_settings()
    host = args.host or settings.HOST
    port = args.port or settings.PORT

    print(f"Starting {settings.APP_NAME} on http://{host}:{port} ...")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
