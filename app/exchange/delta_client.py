"""
Delta Exchange API Client Abstraction.

Phase 1 Invariant:
- Read-only & Public market data access ONLY.
- Account information & positions interfaces support read-only queries when credentials are provided.
- Order creation & order cancellation are STRICTLY DISABLED and blocked by hard safety guards.
- No live orders can be executed.
- Secrets are NEVER logged or printed.
"""

import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx

from app.config import Settings, get_settings
from app.exchange.models import (
    AccountBalance,
    CandleData,
    ExchangeHealthReport,
    OrderBookData,
    OrderBookLevel,
    OrderInfo,
    PositionInfo,
    ProductInfo,
    TickerData,
    TradeData,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


class Phase1SafetyViolationError(RuntimeError):
    """
    Raised when any code attempts to place or modify orders during Phase 1.
    """
    pass


class DeltaExchangeClient:
    """
    Delta Exchange REST Client.
    Configured for read-only market data and account inspection in Phase 1.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        timeout: float = 10.0
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.DELTA_BASE_URL.rstrip("/")
        self.timeout = timeout
        self.symbol = self.settings.DELTA_SYMBOL

        # Validate base safety requirement
        if not self.settings.PAPER_TRADING or self.settings.LIVE_TRADING_ENABLED:
            raise Phase1SafetyViolationError(
                "DeltaExchangeClient can only be initialized in Paper/Read-only mode during Phase 1."
            )

        logger.info(
            f"Initialized DeltaExchangeClient [Read-Only] for symbol={self.symbol} base_url={self.base_url}",
            extra={"event": "CLIENT_INIT", "symbol": self.symbol}
        )

    def _generate_signature(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = "",
        timestamp: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Generate HMAC-SHA256 signature for Delta Exchange API.
        Message = method + timestamp + path + query_string + body
        """
        if not self.settings.is_auth_configured:
            return "", ""

        ts = timestamp or str(int(time.time()))
        secret = self.settings.DELTA_API_SECRET.get_secret_value()  # type: ignore[union-attr]

        # Query string formatting: prepend '?' if non-empty and not already starting with '?'
        formatted_query = ""
        if query_string:
            formatted_query = query_string if query_string.startswith("?") else f"?{query_string}"

        message = f"{method.upper()}{ts}{path}{formatted_query}{body}"
        signature = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        return signature, ts

    def _build_headers(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = ""
    ) -> Dict[str, str]:
        """Construct HTTP headers with authentication if credentials are provided."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"XRP-AI-Trading-Agent/{self.settings.APP_VERSION}",
        }

        if self.settings.is_auth_configured:
            signature, ts = self._generate_signature(method, path, query_string, body)
            headers["api-key"] = self.settings.DELTA_API_KEY.get_secret_value()  # type: ignore[union-attr]
            headers["signature"] = signature
            headers["timestamp"] = ts

        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an HTTP request with error handling and safety checks.
        """
        method = method.upper()

        # Hard safety barrier: Only GET requests are permitted in Phase 1
        if method not in ("GET", "HEAD"):
            msg = (
                f"SAFETY SHIELD ACTIVATED: HTTP {method} requests to Delta Exchange "
                f"are strictly prohibited in Phase 1 (Read-Only Mode)."
            )
            logger.error(msg, extra={"event": "SAFETY_SHIELD_TRIGGERED", "symbol": self.symbol})
            raise Phase1SafetyViolationError(msg)

        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.base_url}{path}"

        # Construct query string for signature
        query_string = ""
        if params:
            from urllib.parse import urlencode
            query_string = urlencode(params)

        headers = self._build_headers(method=method, path=path, query_string=query_string)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers
                )

                if response.status_code >= 400:
                    logger.warning(
                        f"Delta API HTTP {response.status_code} at {path}: {response.text[:200]}",
                        extra={"event": "API_HTTP_ERROR", "symbol": self.symbol}
                    )
                    response.raise_for_status()

                return response.json()
        except httpx.HTTPError as e:
            logger.error(
                f"Network/HTTP exception during request to {path}: {e}",
                extra={"event": "API_NETWORK_ERROR", "symbol": self.symbol}
            )
            raise

    # =========================================================================
    # PUBLIC MARKET DATA INTERFACES
    # =========================================================================

    def get_products(self) -> List[ProductInfo]:
        """Fetch all tradable products from Delta Exchange."""
        data = self._request("GET", "/v2/products")
        results = data.get("result", [])
        return [
            ProductInfo(
                id=item["id"],
                symbol=item["symbol"],
                contract_type=item.get("contract_type"),
                tick_size=float(item.get("tick_size", 0.0)) if item.get("tick_size") else None,
                contract_value=float(item.get("contract_value", 0.0)) if item.get("contract_value") else None,
                state=item.get("state"),
                underlying_asset=item.get("underlying_asset"),
                raw_data=item
            )
            for item in results
        ]

    def get_product(self, symbol: Optional[str] = None) -> Optional[ProductInfo]:
        """Fetch details for a specific contract symbol."""
        target_symbol = (symbol or self.symbol).upper()
        products = self.get_products()
        for p in products:
            if p.symbol == target_symbol:
                return p
        return None

    def get_ticker(self, symbol: Optional[str] = None) -> TickerData:
        """Fetch ticker / price metrics for a symbol."""
        target_symbol = (symbol or self.symbol).upper()
        data = self._request("GET", f"/v2/tickers/{target_symbol}")
        result = data.get("result", {})
        if not result:
            raise ValueError(f"No ticker data found for symbol: {target_symbol}")

        # Timestamp can be in microseconds; normalize to seconds if needed
        raw_ts = int(result.get("timestamp", time.time()))
        ts_sec = raw_ts // 1_000_000 if raw_ts > 10_000_000_000 else raw_ts

        return TickerData(
            symbol=target_symbol,
            mark_price=float(result.get("mark_price", 0.0)),
            spot_price=float(result.get("spot_price", 0.0)) if result.get("spot_price") else None,
            bid=float(result.get("bid", 0.0)) if result.get("bid") else None,
            ask=float(result.get("ask", 0.0)) if result.get("ask") else None,
            high_24h=float(result.get("high_24h", 0.0)) if result.get("high_24h") else None,
            low_24h=float(result.get("low_24h", 0.0)) if result.get("low_24h") else None,
            volume_24h=float(result.get("volume_24h", 0.0)) if result.get("volume_24h") else None,
            timestamp=ts_sec
        )

    def get_candles(
        self,
        symbol: Optional[str] = None,
        resolution: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 100
    ) -> List[CandleData]:
        """
        Fetch historical candlestick (OHLCV) records.
        Delta Exchange requires 'resolution', 'symbol', 'start', and 'end' epoch seconds.
        """
        target_symbol = (symbol or self.symbol).upper()
        target_res = (resolution or self.settings.TIMEFRAME).lower()

        now = int(time.time())
        end_ts = end or now

        # Compute reasonable start timestamp based on limit and resolution if not provided
        if start is None:
            # Approximate timeframe in seconds
            seconds_per_candle = 300  # default 5m
            if target_res.endswith("m"):
                seconds_per_candle = int(target_res[:-1]) * 60
            elif target_res.endswith("h"):
                seconds_per_candle = int(target_res[:-1]) * 3600
            elif target_res.endswith("d"):
                seconds_per_candle = int(target_res[:-1]) * 86400

            start_ts = end_ts - (seconds_per_candle * limit)
        else:
            start_ts = start

        params = {
            "symbol": target_symbol,
            "resolution": target_res,
            "start": start_ts,
            "end": end_ts
        }

        data = self._request("GET", "/v2/history/candles", params=params)
        raw_candles = data.get("result", [])

        candles: List[CandleData] = []
        for c in raw_candles:
            candles.append(
                CandleData(
                    symbol=target_symbol,
                    resolution=target_res,
                    timestamp=int(c["time"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c.get("volume", 0.0))
                )
            )

        # Sort ascending by timestamp
        candles.sort(key=lambda x: x.timestamp)
        return candles

    def get_orderbook(self, symbol: Optional[str] = None, depth: int = 50) -> OrderBookData:
        """Fetch Layer 2 order book snapshot for symbol."""
        target_symbol = (symbol or self.symbol).upper()
        data = self._request("GET", f"/v2/l2orderbook/{target_symbol}")
        result = data.get("result", {})

        bids = [
            OrderBookLevel(price=float(item[0]), size=float(item[1]))
            for item in result.get("buy", [])[:depth]
        ]
        asks = [
            OrderBookLevel(price=float(item[0]), size=float(item[1]))
            for item in result.get("sell", [])[:depth]
        ]

        return OrderBookData(
            symbol=target_symbol,
            timestamp=int(time.time()),
            bids=bids,
            asks=asks
        )

    def get_trades(self, symbol: Optional[str] = None, limit: int = 50) -> List[TradeData]:
        """Fetch recent executed public trades for a symbol."""
        target_symbol = (symbol or self.symbol).upper()
        data = self._request("GET", f"/v2/trades/{target_symbol}")
        raw_trades = data.get("result", [])[:limit]

        trades: List[TradeData] = []
        for t in raw_trades:
            trades.append(
                TradeData(
                    trade_id=str(t.get("buyer_role_id", t.get("timestamp"))),
                    symbol=target_symbol,
                    price=float(t["price"]),
                    size=float(t["size"]),
                    side=t.get("seller_role", "unknown"),
                    timestamp=int(t["timestamp"])
                )
            )
        return trades

    def get_server_time(self) -> Optional[str]:
        """
        Retrieve exchange server time from HTTP response headers.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.head(f"{self.base_url}/v2/products")
                date_hdr = res.headers.get("Date")
                if date_hdr:
                    return date_hdr
        except Exception as e:
            logger.debug(f"Failed to fetch server time from header: {e}")

        # Fallback to ticker timestamp
        try:
            ticker = self.get_ticker(self.symbol)
            return datetime.fromtimestamp(ticker.timestamp, tz=timezone.utc).isoformat()
        except Exception:
            return None

    # =========================================================================
    # READ-ONLY PRIVATE ACCOUNT INTERFACES
    # =========================================================================

    def get_wallet_balances(self) -> List[AccountBalance]:
        """
        Fetch account wallet balances (Read-Only).
        Requires configured DELTA_API_KEY and DELTA_API_SECRET.
        """
        if not self.settings.is_auth_configured:
            logger.info("Auth not configured; wallet balances unavailable in unauthenticated mode.")
            return []

        data = self._request("GET", "/v2/wallet/balances")
        result = data.get("result", [])
        balances: List[AccountBalance] = []

        for item in result:
            balances.append(
                AccountBalance(
                    asset_id=item.get("asset_id"),
                    asset_symbol=item.get("asset_symbol", "UNKNOWN"),
                    balance=float(item.get("balance", 0.0)),
                    available_balance=float(item.get("available_balance", 0.0)),
                    reserved_balance=float(item.get("order_margin", 0.0))
                )
            )
        return balances

    def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """
        Fetch active positions (Read-Only).
        Requires configured DELTA_API_KEY and DELTA_API_SECRET.
        """
        if not self.settings.is_auth_configured:
            return []

        data = self._request("GET", "/v2/positions/margined")
        result = data.get("result", [])
        positions: List[PositionInfo] = []

        for item in result:
            pos_symbol = item.get("product_symbol", "")
            if symbol and pos_symbol.upper() != symbol.upper():
                continue

            size = float(item.get("size", 0.0))
            side = "flat"
            if size > 0:
                side = "long"
            elif size < 0:
                side = "short"

            positions.append(
                PositionInfo(
                    symbol=pos_symbol,
                    product_id=item.get("product_id"),
                    side=side,
                    size=abs(size),
                    entry_price=float(item.get("entry_price", 0.0)),
                    mark_price=float(item.get("mark_price", 0.0)),
                    liquidation_price=float(item.get("liquidation_price", 0.0)) if item.get("liquidation_price") else None,
                    unrealized_pnl=float(item.get("unrealized_pnl", 0.0)),
                    realized_pnl=float(item.get("realized_pnl", 0.0))
                )
            )
        return positions

    def get_orders(self, state: str = "open") -> List[OrderInfo]:
        """
        Fetch orders (Read-Only).
        Requires configured DELTA_API_KEY and DELTA_API_SECRET.
        """
        if not self.settings.is_auth_configured:
            return []

        params = {"state": state}
        data = self._request("GET", "/v2/orders", params=params)
        result = data.get("result", [])
        orders: List[OrderInfo] = []

        for item in result:
            orders.append(
                OrderInfo(
                    order_id=str(item.get("id")),
                    client_order_id=item.get("client_order_id"),
                    symbol=item.get("product_symbol", ""),
                    side=item.get("side", ""),
                    order_type=item.get("order_type", ""),
                    price=float(item.get("limit_price", 0.0)) if item.get("limit_price") else None,
                    size=float(item.get("size", 0.0)),
                    status=item.get("state", ""),
                    filled_size=float(item.get("unfilled_size", 0.0)),
                    created_at=item.get("created_at")
                )
            )
        return orders

    # =========================================================================
    # SAFETY GUARDS - STRICTLY DISABLED OPERATIONS IN PHASE 1
    # =========================================================================

    def create_order(self, *args: Any, **kwargs: Any) -> None:
        """
        EXPLICIT SAFETY BARRIER:
        Live order placement is strictly forbidden in Phase 1.
        """
        msg = (
            "CRITICAL SAFETY SHIELD: Order creation is completely disabled in Phase 1. "
            "Phase 1 is strictly limited to infrastructure, health checks, and read-only market data."
        )
        logger.critical(msg, extra={"event": "ILLEGAL_ORDER_ATTEMPT", "symbol": self.symbol})
        raise Phase1SafetyViolationError(msg)

    def cancel_order(self, *args: Any, **kwargs: Any) -> None:
        """
        EXPLICIT SAFETY BARRIER:
        Order cancellation is disabled in Phase 1.
        """
        msg = (
            "CRITICAL SAFETY SHIELD: Order cancellation is disabled in Phase 1. "
            "No order manipulation is allowed."
        )
        logger.critical(msg, extra={"event": "ILLEGAL_CANCEL_ATTEMPT", "symbol": self.symbol})
        raise Phase1SafetyViolationError(msg)

    def cancel_all_orders(self, *args: Any, **kwargs: Any) -> None:
        """
        EXPLICIT SAFETY BARRIER:
        Mass order cancellation is disabled in Phase 1.
        """
        msg = (
            "CRITICAL SAFETY SHIELD: Bulk order cancellation is disabled in Phase 1."
        )
        logger.critical(msg, extra={"event": "ILLEGAL_CANCEL_ALL_ATTEMPT", "symbol": self.symbol})
        raise Phase1SafetyViolationError(msg)

    # =========================================================================
    # HEALTH CHECK IMPLEMENTATION
    # =========================================================================

    def check_health(self) -> ExchangeHealthReport:
        """
        Perform a comprehensive health check verifying:
        1. API / Base URL is reachable
        2. Network latency
        3. Authentication configuration exists / valid status
        4. XRP contract configuration exists in Delta Exchange product catalog
        5. Current server time can be obtained
        6. Produce structured error and diagnostic messages
        """
        errors: List[str] = []
        is_reachable = False
        latency_ms: Optional[float] = None
        server_time: Optional[str] = None
        xrp_products: List[str] = []
        is_xrp_valid = False

        start_time = time.time()

        try:
            # 1. Connectivity & Product catalog
            products = self.get_products()
            latency_ms = round((time.time() - start_time) * 1000, 2)
            is_reachable = True

            # 2. Extract XRP contracts
            xrp_products = [
                p.symbol for p in products
                if "XRP" in p.symbol.upper()
            ]

            # 3. Verify configured symbol exists
            is_xrp_valid = self.symbol in [p.symbol for p in products]
            if not is_xrp_valid:
                errors.append(
                    f"Configured DELTA_SYMBOL '{self.symbol}' is not found in active Delta Exchange products. "
                    f"Available XRP symbols: {xrp_products}"
                )

            # 4. Exchange server time
            server_time = self.get_server_time()

        except Exception as e:
            is_reachable = False
            latency_ms = round((time.time() - start_time) * 1000, 2)
            errors.append(f"Connection to Delta Exchange base URL '{self.base_url}' failed: {e}")

        # 5. Check Auth configuration notice
        if not self.settings.is_auth_configured:
            # Not an error in public read-only Phase 1, but noted
            logger.info("Delta API authentication not configured. Operating in public read-only mode.")

        # Determine overall exchange health status
        if is_reachable and is_xrp_valid:
            status = "healthy"
        elif is_reachable and not is_xrp_valid:
            status = "degraded"
        else:
            status = "unreachable"

        return ExchangeHealthReport(
            status=status,
            base_url=self.base_url,
            reachable=is_reachable,
            latency_ms=latency_ms,
            auth_configured=self.settings.is_auth_configured,
            xrp_symbol_configured=self.symbol,
            xrp_symbol_valid=is_xrp_valid,
            available_xrp_products=xrp_products,
            server_time=server_time,
            errors=errors
        )
