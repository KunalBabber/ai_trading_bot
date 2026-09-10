"""
Delta Exchange API Client
Handles authentication, product metadata, leverage configuration,
order placement, position management, and live market candle retrieval.
"""

import os
import time
import hashlib
import hmac
import json
import socket
from urllib.parse import urlencode
from typing import Dict, Any, Optional, List
import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class DeltaClient:
    """
    Unified client for Delta Exchange REST API v2 (Production & Testnet).
    Handles HMAC-SHA256 authentication, candles, orders, wallet balances, and bracket targets.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://cdn-ind.testnet.deltaex.org",
        dry_run: Optional[bool] = None,
        timeout: int = 6,
    ):
        self.api_key = api_key or os.getenv("DELTA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("DELTA_API_SECRET", "")
        self.base_url = (
            base_url
            or os.getenv("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")
        ).rstrip("/")
        
        env_dry_run = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
        self.dry_run = dry_run if dry_run is not None else env_dry_run
        self.timeout = timeout
        self.session = requests.Session()

    # === Authentication & Request Dispatch ===

    def _sign_request(
        self,
        endpoint: str,
        method: str,
        query_params: Optional[Dict[str, Any]] = None,
        body_str: str = "",
    ) -> Dict[str, str]:
        """
        Generate Delta Exchange HMAC-SHA256 signature headers:
        auth_string = method + timestamp + path + query_string_or_payload
        """
        timestamp = str(int(time.time()))
        auth_payload = ""

        if method.upper() == "GET":
            if query_params:
                auth_payload = "?" + urlencode(query_params)
        else:
            auth_payload = body_str

        sign_str = method.upper() + timestamp + endpoint + auth_payload
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ai-trading-bot-delta/1.0",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> Dict[str, Any]:
        url = self.base_url + endpoint
        payload_json = json.dumps(data) if data and method.upper() != "GET" else ""

        if authenticated:
            if not self.api_key or not self.api_secret:
                raise ValueError("API Key and Secret required for authenticated requests.")
            headers = self._sign_request(endpoint, method, params, payload_json)
        else:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ai-trading-bot-delta/1.0",
            }

        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                data=payload_json if payload_json else None,
                headers=headers,
                timeout=self.timeout,
            )
            data_out = resp.json()
            if resp.status_code >= 400:
                err_msg = data_out.get("error", {}).get("code", resp.text)
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": err_msg,
                    "raw": data_out,
                }
            return data_out
        except Exception as e:
            return {
                "success": False,
                "status_code": -1,
                "error": str(e),
            }

    # === Public Endpoints ===

    def get_products(self) -> List[Dict[str, Any]]:
        """Fetch all tradable products on Delta Exchange."""
        res = self._request("GET", "/v2/products", authenticated=False)
        return res.get("result", []) if res.get("success", True) else []

    def get_product(self, symbol: str = "BTCUSD") -> Optional[Dict[str, Any]]:
        """Get product metadata by symbol (e.g. BTCUSD)."""
        products = self.get_products()
        for p in products:
            if p.get("symbol") == symbol:
                return p
        return None

    def fetch_candles(
        self,
        symbol: str = "BTCUSD",
        resolution: str = "5m",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from Delta Exchange.
        Returns a sorted DataFrame with:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        now = int(time.time())
        # Calculate start time based on resolution and limit
        res_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(resolution, 300)
        start = now - (limit * res_seconds + 3600)

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": start,
            "end": now,
        }

        res = self._request("GET", "/v2/history/candles", params=params, authenticated=False)
        raw_candles = res.get("result", [])
        if not raw_candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw_candles)
        # Rename 'time' to 'timestamp'
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.drop(columns=["time"])

        # Ensure correct column types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Sort ascending chronologically
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    # === Authenticated Endpoints ===

    def get_balances(self) -> List[Dict[str, Any]]:
        """Fetch all wallet balances."""
        res = self._request("GET", "/v2/wallet/balances", authenticated=True)
        return res.get("result", []) if res.get("success") else []

    def get_equity(self, asset: str = "USD") -> float:
        """Get available balance or equity for a specific asset currency."""
        balances = self.get_balances()
        for b in balances:
            if b.get("asset_symbol") == asset:
                return float(b.get("balance", 0.0))
        return 0.0

    def set_leverage(self, product_id: int, leverage: int = 10) -> bool:
        """Configure leverage for the specified product ID."""
        endpoint = f"/v2/products/{product_id}/leverage"
        payload = {"leverage": str(leverage)}
        res = self._request("POST", endpoint, data=payload, authenticated=True)
        return bool(res.get("success"))

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all currently open margined positions."""
        res = self._request("GET", "/v2/positions/margined", authenticated=True)
        return res.get("result", []) if res.get("success") else []

    def get_position_for_product(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Return position details for a specific product."""
        positions = self.get_positions()
        for pos in positions:
            if pos.get("product_id") == product_id:
                return pos
        return None

    def place_order(
        self,
        product_id: int,
        size: int,
        side: str,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an order on Delta Exchange.
        In dry_run mode, logs the order without sending it to the exchange.
        """
        payload = {
            "product_id": int(product_id),
            "size": int(size),
            "side": side.lower(),
            "order_type": order_type,
            "time_in_force": "ioc" if order_type == "market_order" else "gtc",
        }

        if limit_price is not None and order_type == "limit_order":
            payload["limit_price"] = str(limit_price)

        if stop_loss_price is not None:
            payload["bracket_stop_loss_price"] = str(stop_loss_price)

        if take_profit_price is not None:
            payload["bracket_take_profit_price"] = str(take_profit_price)

        if reduce_only:
            payload["reduce_only"] = True

        if self.dry_run:
            print(f"[DRY RUN] Order simulated: {payload}")
            return {
                "success": True,
                "dry_run": True,
                "result": payload,
            }

        return self._request("POST", "/v2/orders", data=payload, authenticated=True)

    def close_position(self, product_id: int) -> Dict[str, Any]:
        """Market close the active position for a given product."""
        pos = self.get_position_for_product(product_id)
        if not pos or float(pos.get("size", 0)) == 0:
            return {"success": True, "message": "No open position."}

        current_size = int(pos["size"])
        side = "sell" if current_size > 0 else "buy"
        abs_size = abs(current_size)

        return self.place_order(
            product_id=product_id,
            size=abs_size,
            side=side,
            order_type="market_order",
            reduce_only=True,
        )

    def cancel_all_orders(self, product_id: Optional[int] = None) -> Dict[str, Any]:
        """Cancel all open orders, optionally filtered by product_id."""
        if self.dry_run:
            print(f"[DRY RUN] Cancel all orders called for product_id={product_id}")
            return {"success": True, "dry_run": True}

        endpoint = "/v2/orders/all"
        payload = {"product_id": product_id} if product_id else None
        return self._request("DELETE", endpoint, data=payload, authenticated=True)
