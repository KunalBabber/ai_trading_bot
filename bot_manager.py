"""
Thread-Safe Trading Bot Manager for Streamlit
Controls the background execution thread, handles live market polling,
GRU inference, order placement, and provides atomic state to the Streamlit UI.
"""

import os
import time
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, Optional, List
import pandas as pd
import torch
import yaml
from dotenv import set_key

from delta_client import DeltaClient
from live_features import LiveFeatureEngine
from model import GRUTradingModel
from features import FEATURE_COLUMNS
from strategy import make_signal, position_fraction
from timezone_utils import IST_TZ, get_now


class TradingBotManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TradingBotManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()

        # Shared State for UI
        self.is_running = False
        self.mode = "dry_run"  # 'dry_run' or 'live'
        self.symbol = "BTCUSD"
        self.resolution = "5m"
        self.leverage = 10
        self.status_message = "Idle"

        self.current_price = 0.0
        self.candle_df = pd.DataFrame()
        self.metrics: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {
            "p_long": 0.5,
            "p_short": 0.5,
            "expected_return": 0.0,
            "signal_side": 0,
        }

        self.active_position: Dict[str, Any] = {
            "side": 0,
            "size": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
        }

        self.simulated_equity = 10000.0
        self.wallet_equity = 0.0
        self.balances_breakdown: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.recent_logs = deque(maxlen=200)
        self.cycle_count = 0
        self.last_heartbeat = time.time()

        # Model & Engines
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client: Optional[DeltaClient] = None
        self.model: Optional[GRUTradingModel] = None
        self.feature_engine: Optional[LiveFeatureEngine] = None
        self.product_specs: Dict[str, Any] = {}
        self.live_config: Dict[str, Any] = {}
        # Default Timezone: IST (+05:30)
        self.tz = IST_TZ
        self.tz_name = "IST"

    def set_timezone(self, tz, name: str = "IST"):
        """Update active timezone for bot logs and timestamps."""
        with self.state_lock:
            self.tz = tz
            self.tz_name = name

    def update_config(self, new_params: Dict[str, Any]):
        """Dynamically update strategy parameters while bot is running."""
        with self.state_lock:
            self.live_config.update(new_params)

    def now_dt(self) -> datetime:
        """Returns current datetime in bot's configured timezone."""
        return datetime.now(self.tz)

    def now_str(self, fmt: str = "%I:%M:%S %p") -> str:
        """Returns formatted current time in bot's configured timezone."""
        return self.now_dt().strftime(fmt)

    def log(self, message: str):
        """Append log message with localized timestamp."""
        ts = self.now_str("%I:%M:%S %p")
        entry = f"[{ts}] {message}"
        with self.state_lock:
            self.recent_logs.append(entry)

    def clear_logs(self):
        """Clear recent terminal logs."""
        with self.state_lock:
            self.recent_logs.clear()
        self.log("Terminal log cleared.")

    def get_state(self) -> Dict[str, Any]:
        """Thread-safe snapshot of bot state for UI rendering."""
        with self.state_lock:
            return {
                "is_running": self.is_running,
                "mode": self.mode,
                "symbol": self.symbol,
                "resolution": self.resolution,
                "leverage": self.leverage,
                "status_message": self.status_message,
                "current_price": self.current_price,
                "candle_df": self.candle_df.copy() if not self.candle_df.empty else pd.DataFrame(),
                "metrics": dict(self.metrics),
                "predictions": dict(self.predictions),
                "active_position": dict(self.active_position),
                "simulated_equity": self.simulated_equity,
                "wallet_equity": self.wallet_equity,
                "balances_breakdown": list(self.balances_breakdown),
                "trade_history": list(self.trade_history),
                "recent_logs": list(self.recent_logs),
                "product_specs": dict(self.product_specs),
                "cycle_count": self.cycle_count,
                "last_heartbeat": self.last_heartbeat,
            }

    def update_balance(self, api_key: str, api_secret: str, base_url: str) -> Dict[str, Any]:
        """Fetch and update live wallet balance and open positions from Delta Exchange."""
        if not api_key or not api_secret:
            return {"success": False, "error": "API Key and Secret required"}
        try:
            client = DeltaClient(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
                dry_run=False,
                timeout=8,
            )
            res = client._request("GET", "/v2/wallet/balances", authenticated=True)
            if res.get("success", False):
                balances = res.get("result", [])
                total_usd = sum(
                    float(b.get("balance", 0.0))
                    for b in balances
                    if b.get("asset_symbol") in ["USD", "USDT"] or float(b.get("balance", 0.0)) > 0
                )
                with self.state_lock:
                    self.balances_breakdown = balances
                    self.wallet_equity = total_usd

                # Sync live open positions from Delta Exchange
                try:
                    positions = client.get_positions()
                    matching_pos = None
                    for p in positions:
                        if p.get("product_symbol") == self.symbol or p.get("product_id") == self.product_specs.get("id"):
                            matching_pos = p
                            break
                    with self.state_lock:
                        if matching_pos and int(matching_pos.get("size", 0)) != 0:
                            sz = int(matching_pos["size"])
                            ep = float(matching_pos.get("entry_price", 0.0))
                            upnl = float(matching_pos.get("unrealized_pnl", 0.0))
                            self.active_position = {
                                "side": 1 if sz > 0 else -1,
                                "size": abs(sz),
                                "entry_price": ep,
                                "stop_loss": 0.0,
                                "take_profit": 0.0,
                                "unrealized_pnl": upnl,
                                "unrealized_pnl_pct": 0.0,
                            }
                        else:
                            self.active_position = {
                                "side": 0,
                                "size": 0,
                                "entry_price": 0.0,
                                "stop_loss": 0.0,
                                "take_profit": 0.0,
                                "unrealized_pnl": 0.0,
                                "unrealized_pnl_pct": 0.0,
                            }
                except Exception:
                    pass

                return {"success": True, "equity": self.wallet_equity, "balances": balances}
            else:
                return {"success": False, "error": res.get("error", "Failed to fetch balances")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_connection(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
    ) -> Dict[str, Any]:
        """Verify API keys and check IP whitelisting status."""
        test_client = DeltaClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            dry_run=False,
            timeout=8,
        )
        res = test_client._request("GET", "/v2/wallet/balances", authenticated=True)
        if res.get("success", False):
            balances = res.get("result", [])
            total_bal = sum(float(b.get("balance", 0)) for b in balances)
            return {
                "success": True,
                "message": f"Connected successfully! Found {len(balances)} asset wallets (Total: ${total_bal:.2f})",
                "balances": balances,
            }
        else:
            err = res.get("error", "Unknown error")
            raw_err = res.get("raw", {}).get("error", {})
            if "ip_not_whitelisted" in str(err):
                client_ip = raw_err.get("context", {}).get("client_ip", "Unknown")
                return {
                    "success": False,
                    "ip_needed": client_ip,
                    "error": f"IP Not Whitelisted on Delta Exchange!\n\nYour Streamlit Cloud Server IP is: {client_ip}",
                }
            return {
                "success": False,
                "error": f"Connection Failed: {err}",
            }

    def save_env_credentials(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        dry_run: bool = True,
    ):
        """Persist updated credentials to .env file."""
        env_path = os.path.abspath(".env")
        if not os.path.exists(env_path):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("# Delta Exchange Configuration\n")

        set_key(env_path, "DELTA_API_KEY", api_key)
        set_key(env_path, "DELTA_API_SECRET", api_secret)
        set_key(env_path, "DELTA_BASE_URL", base_url)
        set_key(env_path, "DRY_RUN", "true" if dry_run else "false")
        self.log("Credentials saved to .env file.")

    def start_bot(self, config: Dict[str, Any]) -> bool:
        """Start background execution worker."""
        if self.is_running:
            self.log("Bot is already running.")
            return True

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            args=(config,),
            daemon=True,
            name="DeltaAITraderThread",
        )
        self.thread.start()
        return True

    def stop_bot(self):
        """Stop background execution worker."""
        if not self.is_running:
            return
        self.log("Stopping bot background thread...")
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        with self.state_lock:
            self.is_running = False
            self.status_message = "Stopped"
        self.log("Bot stopped successfully.")

    def panic_close(self) -> Dict[str, Any]:
        """Emergency market close for any active position."""
        with self.state_lock:
            pos = dict(self.active_position)
            specs = dict(self.product_specs)
            client = self.client

        if pos["side"] == 0:
            self.log("Emergency Close: No active position to close.")
            return {"success": True, "message": "No open position."}

        product_id = specs.get("id")
        price = self.current_price
        self.log(f"[PANIC] Closing {pos['side']} position of {pos['size']} contracts at {price:.2f}...")

        if client and not client.dry_run and product_id:
            res = client.close_position(product_id)
            client.cancel_all_orders(product_id)
        else:
            res = {"success": True, "dry_run": True}

        # Update state
        with self.state_lock:
            pnl_dollar = pos["unrealized_pnl"]
            self.simulated_equity += pnl_dollar
            self.trade_history.append({
                "time": self.now_str("%Y-%m-%d %I:%M:%S %p"),
                "symbol": self.symbol,
                "side": "BUY (Long)" if pos["side"] == 1 else "SELL (Short)",
                "size": pos["size"],
                "entry": pos["entry_price"],
                "exit": price,
                "pnl": pnl_dollar,
                "reason": "PANIC_CLOSE",
            })
            self.active_position = {
                "side": 0,
                "size": 0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
            }
        self.log(f"[PANIC] Position closed. Realized PnL: ${pnl_dollar:+.2f}")
        return res

    def _run_loop(self, config: Dict[str, Any]):
        """Main background loop executed in worker thread."""
        with self.state_lock:
            self.is_running = True
            self.mode = "dry_run" if config.get("dry_run", True) else "live"
            self.symbol = config.get("symbol", "BTCUSD")
            self.resolution = config.get("resolution", "5m")
            self.leverage = config.get("leverage", 10)
            self.status_message = "Initializing..."

        self.log(f"Initializing bot: {self.symbol} ({self.resolution}) | Leverage: {self.leverage}x | Mode: {self.mode.upper()}")

        # 1. Initialize Client
        self.client = DeltaClient(
            api_key=config.get("api_key"),
            api_secret=config.get("api_secret"),
            base_url=config.get("base_url"),
            dry_run=config.get("dry_run", True),
        )

        # 2. Product Specs
        product = self.client.get_product(self.symbol)
        if not product:
            self.log(f"[ERROR] Product {self.symbol} not found on Delta Exchange.")
            with self.state_lock:
                self.is_running = False
                self.status_message = f"Product {self.symbol} not found"
            return

        with self.state_lock:
            self.product_specs = {
                "id": product["id"],
                "contract_value": float(product.get("contract_value", 0.001)),
                "tick_size": float(product.get("tick_size", 0.1)),
                "symbol": product["symbol"],
            }
        product_id = product["id"]
        contract_val = float(product.get("contract_value", 0.001))
        self.log(f"Product metadata: ID={product_id}, Contract={contract_val}, Tick={self.product_specs['tick_size']}")

        # 3. Query Wallet Balance
        if config.get("api_key") and config.get("api_secret"):
            try:
                bals_res = self.client._request("GET", "/v2/wallet/balances", authenticated=True)
                if bals_res.get("success"):
                    bals = bals_res.get("result", [])
                    total_bal = sum(
                        float(b.get("balance", 0.0))
                        for b in bals
                        if b.get("asset_symbol") in ["USD", "USDT"] or float(b.get("balance", 0.0)) > 0
                    )
                    with self.state_lock:
                        self.balances_breakdown = bals
                        self.wallet_equity = total_bal
                    self.log(f"Wallet balance loaded: ${total_bal:.2f}")
                else:
                    err = bals_res.get("error", "Unknown auth")
                    if "ip_not_whitelisted" in str(err):
                        raw_err = bals_res.get("raw", {}).get("error", {})
                        client_ip = raw_err.get("context", {}).get("client_ip", "")
                        ip_info = f" (Detected IP: {client_ip})" if client_ip else ""
                        self.log(f"[WARN] IP not whitelisted for API key on Delta Exchange!{ip_info}")
            except Exception as e:
                self.log(f"[WARN] Balance query error: {e}")

        # 4. Configure Leverage if live
        if not self.client.dry_run:
            lev_ok = self.client.set_leverage(product_id, self.leverage)
            if lev_ok:
                self.log(f"Leverage successfully set to {self.leverage}x on Delta Exchange.")
            else:
                self.log(f"[WARN] Could not set leverage on exchange. Using default.")

            # 4.5 Sync Initial Live Position from Delta Exchange
            try:
                live_pos = self.client.get_position_for_product(product_id)
                with self.state_lock:
                    if live_pos and int(live_pos.get("size", 0)) != 0:
                        sz = int(live_pos["size"])
                        ep = float(live_pos.get("entry_price", 0.0))
                        upnl = float(live_pos.get("unrealized_pnl", 0.0))
                        self.active_position = {
                            "side": 1 if sz > 0 else -1,
                            "size": abs(sz),
                            "entry_price": ep,
                            "stop_loss": 0.0,
                            "take_profit": 0.0,
                            "unrealized_pnl": upnl,
                            "unrealized_pnl_pct": 0.0,
                        }
                        self.log(f"Active position synced from Delta: {'LONG' if sz > 0 else 'SHORT'} {abs(sz)}x @ ${ep:,.1f}")
                    else:
                        self.active_position = {
                            "side": 0,
                            "size": 0,
                            "entry_price": 0.0,
                            "stop_loss": 0.0,
                            "take_profit": 0.0,
                            "unrealized_pnl": 0.0,
                            "unrealized_pnl_pct": 0.0,
                        }
                        self.log("Delta Exchange position: FLAT (No active positions).")
            except Exception as e:
                self.log(f"[WARN] Could not sync initial position: {e}")

        # 5. Load Models
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f)

            self.feature_engine = LiveFeatureEngine(
                scaler_path="artifacts/scaler.joblib",
                sequence_length=base_cfg["data"]["sequence_length"],
            )

            ckpt = torch.load("artifacts/gru.pt", map_location=self.device)
            self.model = GRUTradingModel(
                len(FEATURE_COLUMNS),
                base_cfg["model"]["hidden_size"],
                base_cfg["model"]["num_layers"],
                base_cfg["model"]["dropout"],
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.log("GRU Sequence Model & Scaler loaded successfully.")
        except Exception as e:
            self.log(f"[ERROR] Failed to load model: {str(e)}")
            with self.state_lock:
                self.is_running = False
                self.status_message = f"Model load error: {e}"
            return

        with self.state_lock:
            self.status_message = "Running"

        poll_sec = config.get("poll_interval", 15)
        last_bar_time = None

        with self.state_lock:
            self.live_config = dict(config)

        # Main Trading Loop
        while not self.stop_event.is_set():
            try:
                # Dynamic strategy parameters from live_config
                with self.state_lock:
                    long_prob_threshold = self.live_config.get("long_probability", 0.58)
                    short_prob_threshold = self.live_config.get("short_probability", 0.58)
                    min_expected_return = self.live_config.get("min_expected_return", 0.0015)
                    take_profit_target = self.live_config.get("take_profit", 0.016)
                    sizing_mode = self.live_config.get("sizing_mode", "risk_budget")
                    fixed_contracts = self.live_config.get("fixed_contracts", 1)
                    risk_per_trade = self.live_config.get("risk_per_trade", 0.003)
                # 1. Fetch Candles
                candles = self.client.fetch_candles(self.symbol, self.resolution, limit=350)
                if len(candles) < base_cfg["data"]["sequence_length"] + 60:
                    self.log(f"Waiting for candles history ({len(candles)}/156)...")
                    self.stop_event.wait(poll_sec)
                    continue

                # 2. Process features
                tensor, metrics = self.feature_engine.process_candles(candles)
                current_price = metrics["close"]
                bar_time = metrics["timestamp"]

                # 3. Model Inference
                tensor_dev = tensor.to(self.device)
                with torch.no_grad():
                    pred_ret, long_logit, short_logit = self.model(tensor_dev)

                expected_return = float(pred_ret.item())
                p_long = float(torch.sigmoid(long_logit).item())
                p_short = float(torch.sigmoid(short_logit).item())

                # 4. Generate Signal
                signal = make_signal(
                    expected_return,
                    p_long,
                    p_short,
                    metrics["trend_15m"],
                    metrics["trend_1h"],
                    metrics["atr_pct"],
                    long_prob_threshold,
                    short_prob_threshold,
                    min_expected_return,
                )

                # 5. Check Active Position & PnL
                with self.state_lock:
                    self.current_price = current_price
                    self.candle_df = candles
                    self.metrics = metrics
                    self.predictions = {
                        "p_long": p_long,
                        "p_short": p_short,
                        "expected_return": expected_return,
                        "signal_side": signal.side,
                    }

                # Sync live position directly from Delta Exchange on each scan cycle
                if not self.client.dry_run and product_id:
                    try:
                        live_pos = self.client.get_position_for_product(product_id)
                        with self.state_lock:
                            if live_pos and int(live_pos.get("size", 0)) != 0:
                                sz = int(live_pos["size"])
                                real_side = 1 if sz > 0 else -1
                                abs_sz = abs(sz)
                                ep = float(live_pos.get("entry_price", current_price))
                                un_pnl = float(live_pos.get("unrealized_pnl", 0.0))
                                self.active_position["side"] = real_side
                                self.active_position["size"] = abs_sz
                                self.active_position["entry_price"] = ep
                                self.active_position["unrealized_pnl"] = un_pnl
                                if ep > 0:
                                    self.active_position["unrealized_pnl_pct"] = real_side * (current_price / ep - 1.0) * 100.0
                            else:
                                # Exchange has no open position
                                if self.active_position["side"] != 0:
                                    realized_val = self.active_position["unrealized_pnl"]
                                    self.log(f"[SYNC] Delta Exchange position closed externally. Realized PnL: ${realized_val:+.2f}")
                                    self.trade_history.append({
                                        "time": self.now_str("%Y-%m-%d %I:%M:%S %p"),
                                        "symbol": self.symbol,
                                        "side": "BUY (Long)" if self.active_position["side"] == 1 else "SELL (Short)",
                                        "size": self.active_position["size"],
                                        "entry": self.active_position["entry_price"],
                                        "exit": current_price,
                                        "pnl": realized_val,
                                        "reason": "DELTA_MANUAL_CLOSE",
                                    })
                                    self.active_position = {
                                        "side": 0,
                                        "size": 0,
                                        "entry_price": 0.0,
                                        "stop_loss": 0.0,
                                        "take_profit": 0.0,
                                        "unrealized_pnl": 0.0,
                                        "unrealized_pnl_pct": 0.0,
                                    }
                    except Exception:
                        pass
                else:
                    # Dry run simulated position PnL calculation
                    with self.state_lock:
                        pos = self.active_position
                        if pos["side"] != 0:
                            pnl_pct = pos["side"] * (current_price / pos["entry_price"] - 1.0)
                            pos["unrealized_pnl_pct"] = pnl_pct * 100.0
                            pos["unrealized_pnl"] = (
                                pos["side"]
                                * (current_price / pos["entry_price"] - 1.0)
                                * (pos["size"] * contract_val * pos["entry_price"])
                            )

                with self.state_lock:
                    pos = dict(self.active_position)
                    self.cycle_count += 1
                    self.last_heartbeat = time.time()

                # Real-time Terminal Log Every Scan Cycle
                sig_desc = "BUY (+1) 🚀" if signal.side == 1 else ("SELL (-1) 🔻" if signal.side == -1 else "FLAT (Wait)")
                pos_desc = f"LONG ({pos['size']}x @ ${pos['entry_price']:,.1f})" if pos["side"] == 1 else (f"SHORT ({pos['size']}x @ ${pos['entry_price']:,.1f})" if pos["side"] == -1 else "FLAT")

                self.log(
                    f"SCAN #{self.cycle_count:03d} • {self.symbol}: ${current_price:,.2f} | "
                    f"P(L): {p_long*100:.1f}% | P(S): {p_short*100:.1f}% | E(R): {expected_return*100:+.2f}% | "
                    f"Signal: {sig_desc} | Pos: {pos_desc}"
                )

                # 6. Check Exits (Stop Loss / Take Profit)
                self._check_and_handle_exits(current_price, contract_val)

                # 7. Check Entries: on candle bar close OR immediately when FLAT with confirmed signal
                is_new_bar = (last_bar_time is None) or (bar_time > last_bar_time)
                is_flat_entry = (pos["side"] == 0 and signal.side != 0)

                if is_new_bar or is_flat_entry:
                    if signal.side == 0 and pos["side"] == 0 and is_new_bar:
                        reasons = []
                        if p_long < long_prob_threshold and p_short < short_prob_threshold:
                            reasons.append(f"AI Odds ({max(p_long, p_short)*100:.1f}%) < {long_prob_threshold*100:.0f}%")
                        if abs(expected_return) < min_expected_return:
                            reasons.append("Profit hurdle not met")
                        if (metrics.get("trend_15m", 0) <= 0 and p_long >= long_prob_threshold) or (metrics.get("trend_15m", 0) >= 0 and p_short >= short_prob_threshold):
                            reasons.append("15m Trend not aligned")
                        if (metrics.get("trend_1h", 0) <= 0 and p_long >= long_prob_threshold) or (metrics.get("trend_1h", 0) >= 0 and p_short >= short_prob_threshold):
                            reasons.append("1h Macro Trend not aligned")
                        reason_str = ", ".join(reasons) if reasons else "Waiting for high-conviction setup"
                        self.log(f"   ↳ [Bar Close] Standing by: {reason_str}")

                    self._check_and_handle_entries(
                        signal=signal,
                        current_price=current_price,
                        contract_val=contract_val,
                        product_id=product_id,
                        sizing_mode=sizing_mode,
                        fixed_contracts=fixed_contracts,
                        risk_per_trade=risk_per_trade,
                        take_profit_target=take_profit_target,
                    )
                    if is_new_bar:
                        last_bar_time = bar_time

            except Exception as e:
                self.log(f"[LOOP ERROR] {str(e)}")

            self.stop_event.wait(poll_sec)

        with self.state_lock:
            self.is_running = False
            self.status_message = "Stopped"

    def _check_and_handle_exits(self, current_price: float, contract_val: float):
        """Exits active position if Stop Loss or Take Profit hit."""
        with self.state_lock:
            pos = dict(self.active_position)
            specs = dict(self.product_specs)

        if pos["side"] == 0:
            return

        # Check if position was closed externally on Delta Exchange
        product_id = specs.get("id")
        if not self.client.dry_run and product_id:
            try:
                live_pos = self.client.get_position_for_product(product_id)
                if not live_pos or int(live_pos.get("size", 0)) == 0:
                    with self.state_lock:
                        pnl_dollar = pos["unrealized_pnl"]
                        self.trade_history.append({
                            "time": self.now_str("%Y-%m-%d %I:%M:%S %p"),
                            "symbol": self.symbol,
                            "side": "BUY (Long)" if pos["side"] == 1 else "SELL (Short)",
                            "size": pos["size"],
                            "entry": pos["entry_price"],
                            "exit": current_price,
                            "pnl": pnl_dollar,
                            "reason": "DELTA_EXCHANGE_CLOSE",
                        })
                        self.active_position = {
                            "side": 0,
                            "size": 0,
                            "entry_price": 0.0,
                            "stop_loss": 0.0,
                            "take_profit": 0.0,
                            "unrealized_pnl": 0.0,
                            "unrealized_pnl_pct": 0.0,
                        }
                    self.log(f"[SYNC] Position was closed directly on Delta Exchange. Status: FLAT (Realized PnL: ${pnl_dollar:+.2f}).")
                    return
            except Exception:
                pass

        exit_reason = None
        if pos["side"] == +1:
            if current_price <= pos["stop_loss"]:
                exit_reason = f"Stop Loss ({current_price:.1f} <= {pos['stop_loss']:.1f})"
            elif current_price >= pos["take_profit"]:
                exit_reason = f"Take Profit ({current_price:.1f} >= {pos['take_profit']:.1f})"
        elif pos["side"] == -1:
            if current_price >= pos["stop_loss"]:
                exit_reason = f"Stop Loss ({current_price:.1f} >= {pos['stop_loss']:.1f})"
            elif current_price <= pos["take_profit"]:
                exit_reason = f"Take Profit ({current_price:.1f} <= {pos['take_profit']:.1f})"

        if exit_reason:
            self.log(f"[EXIT] {exit_reason} triggered.")
            product_id = specs.get("id")
            if not self.client.dry_run and product_id:
                self.client.close_position(product_id)

            with self.state_lock:
                pnl_dollar = pos["unrealized_pnl"]
                self.simulated_equity += pnl_dollar
                self.trade_history.append({
                    "time": self.now_str("%Y-%m-%d %I:%M:%S %p"),
                    "symbol": self.symbol,
                    "side": "BUY (Long)" if pos["side"] == 1 else "SELL (Short)",
                    "size": pos["size"],
                    "entry": pos["entry_price"],
                    "exit": current_price,
                    "pnl": pnl_dollar,
                    "reason": exit_reason,
                })
                self.active_position = {
                    "side": 0,
                    "size": 0,
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                }
            self.log(f"[CLOSED] Realized PnL: ${pnl_dollar:+.2f}")

    def _check_and_handle_entries(
        self,
        signal,
        current_price: float,
        contract_val: float,
        product_id: int,
        sizing_mode: str,
        fixed_contracts: int,
        risk_per_trade: float,
        take_profit_target: float,
    ):
        """Opens position if a trade signal is confirmed."""
        if signal.side == 0:
            return

        with self.state_lock:
            pos = dict(self.active_position)

        # If already holding a position in the same direction, do not duplicate/pyramid
        if pos["side"] == signal.side:
            return

        # Handle reversal
        if pos["side"] != 0 and pos["side"] != signal.side:
            self.log("[REVERSAL] Closing existing position to enter opposite signal.")
            if not self.client.dry_run:
                self.client.close_position(product_id)
            with self.state_lock:
                self.active_position = {
                    "side": 0,
                    "size": 0,
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                }

        # Calculate contracts
        if sizing_mode == "fixed":
            contracts = max(1, int(fixed_contracts))
        else:
            equity = self.simulated_equity if self.client.dry_run else self.client.get_equity("USD")
            pos_frac = position_fraction(equity, signal.stop_distance, risk_per_trade, 0.20)
            notional = equity * pos_frac * self.leverage
            contracts = max(1, int(notional / (current_price * contract_val)))

        side_str = "buy" if signal.side == +1 else "sell"
        stop_dist = signal.stop_distance

        if signal.side == +1:
            sl_price = round(current_price * (1.0 - stop_dist), 1)
            tp_price = round(current_price * (1.0 + take_profit_target), 1)
        else:
            sl_price = round(current_price * (1.0 + stop_dist), 1)
            tp_price = round(current_price * (1.0 - take_profit_target), 1)

        self.log(f"[ORDER] Executing {side_str.upper()} {contracts} contracts @ {current_price:.1f} (SL: {sl_price}, TP: {tp_price})")

        order_res = self.client.place_order(
            product_id=product_id,
            size=contracts,
            side=side_str,
            order_type="market_order",
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )

        if order_res.get("success", False):
            with self.state_lock:
                self.active_position = {
                    "side": signal.side,
                    "size": contracts,
                    "entry_price": current_price,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                }
            self.log(f"[OK] Position established: {side_str.upper()} {contracts} contracts.")
        else:
            err = order_res.get('error')
            if "ip_not_whitelisted" in str(err):
                raw_err = order_res.get("raw", {}).get("error", {})
                client_ip = raw_err.get("context", {}).get("client_ip", "")
                ip_info = f" (Your IP detected by Delta: {client_ip})" if client_ip else ""
                self.log(f"[REJECTED] IP not whitelisted on Delta Exchange!{ip_info} Please remove IP restriction or add this IP in your Delta API key settings.")
            else:
                self.log(f"[REJECTED] {err}")


# Singleton instance
bot_manager = TradingBotManager()
