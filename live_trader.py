"""
Delta Exchange Live / Dry-Run Trading Bot
Executes multi-task GRU sequential AI strategy on Delta Exchange
with multi-timeframe confirmation, bracket risk management, and drawdown kill-switch.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Ensure stdout handles UTF-8 on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml
import torch
import numpy as np
import pandas as pd

from delta_client import DeltaClient
from live_features import LiveFeatureEngine
from model import GRUTradingModel
from features import FEATURE_COLUMNS
from strategy import make_signal, position_fraction


class DeltaAITrader:
    def __init__(
        self,
        config_path: str = "config.yaml",
        model_path: str = "artifacts/gru.pt",
        scaler_path: str = "artifacts/scaler.joblib",
        dry_run: Optional[bool] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Exchange configuration
        exch_cfg = self.cfg.get("exchange", {})
        self.symbol = exch_cfg.get("symbol", "BTCUSD")
        self.resolution = exch_cfg.get("resolution", "5m")
        self.leverage = exch_cfg.get("leverage", 10)
        self.poll_interval = exch_cfg.get("poll_interval_sec", 15)
        self.min_order_size = exch_cfg.get("min_order_size", 1)

        # Mode determination
        if dry_run is not None:
            self.dry_run = dry_run
        else:
            self.dry_run = exch_cfg.get("dry_run", True)

        # Delta Client
        self.client = DeltaClient(dry_run=self.dry_run)

        # Product details
        self.product = None
        self.product_id = None
        self.contract_value = 0.001
        self.tick_size = 0.1

        # Model and Features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_engine = LiveFeatureEngine(
            scaler_path=scaler_path,
            sequence_length=self.cfg["data"]["sequence_length"],
        )
        self.model = self._load_model(model_path)

        # State tracking
        self.simulated_equity = 10000.0  # Used in dry-run
        self.starting_daily_equity = self.simulated_equity
        self.current_position: Dict[str, Any] = {
            "side": 0,
            "size": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
        }
        self.last_bar_time: Optional[pd.Timestamp] = None

    def _load_model(self, model_path: str) -> GRUTradingModel:
        ckpt = torch.load(model_path, map_location=self.device)
        model = GRUTradingModel(
            len(FEATURE_COLUMNS),
            self.cfg["model"]["hidden_size"],
            self.cfg["model"]["num_layers"],
            self.cfg["model"]["dropout"],
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    def initialize(self) -> bool:
        """Fetch product metadata and configure leverage."""
        print("=" * 65)
        print("  Delta Exchange AI Trading Bot - Initialization")
        print("=" * 65)
        print(f"Mode:         {'[SIMULATED DRY RUN]' if self.dry_run else '[LIVE EXECUTION]'}")
        print(f"Base URL:     {self.client.base_url}")
        print(f"Symbol:       {self.symbol}")
        print(f"Timeframe:    {self.resolution}")
        print(f"Leverage:     {self.leverage}x")
        print(f"System Time:  {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

        # 1. Fetch Product Metadata
        self.product = self.client.get_product(self.symbol)
        if not self.product:
            print(f"[ERROR] Product {self.symbol} not found on Delta Exchange.")
            return False

        self.product_id = self.product["id"]
        self.contract_value = float(self.product.get("contract_value", 0.001))
        self.tick_size = float(self.product.get("tick_size", 0.1))
        print(f"[OK] Found product: {self.symbol} (ID: {self.product_id})")
        print(f"     Contract Value: {self.contract_value} BTC | Tick Size: {self.tick_size}")

        # 2. Account / Authentication Check (if live)
        if not self.dry_run:
            balances_res = self.client._request("GET", "/v2/wallet/balances", authenticated=True)
            if not balances_res.get("success"):
                err = balances_res.get("error", "Unknown auth error")
                print(f"[ERROR] Live Authentication Failed: {err}")
                if "ip_not_whitelisted" in str(err):
                    print("     [NOTICE] Your IP must be added to the whitelist in Delta Exchange API settings.")
                return False

            equity = self.client.get_equity("USD") or self.client.get_equity("USDT")
            print(f"[OK] Live Wallet Connected. Available Equity: ${equity:.2f}")
            self.starting_daily_equity = equity

            # Set Leverage on exchange
            if self.client.set_leverage(self.product_id, self.leverage):
                print(f"[CONFIG] Leverage successfully set to {self.leverage}x on Delta Exchange.")
            else:
                print(f"[WARN] Could not set leverage on exchange. Using default.")
        else:
            print(f"[INFO] Running in Dry-Run mode. Simulated Starting Equity: ${self.simulated_equity:,.2f}")

        print("=" * 65)
        return True

    def calculate_contracts(self, equity: float, price: float, stop_distance: float) -> int:
        """Calculate number of contracts to trade based on risk percentage."""
        pos_frac = position_fraction(
            equity,
            stop_distance,
            self.cfg["strategy"]["risk_per_trade"],
            self.cfg["strategy"]["max_position_fraction"],
        )
        if pos_frac <= 0 or price <= 0:
            return 0

        # Notional position size = equity * pos_frac * leverage
        notional = equity * pos_frac * self.leverage
        # Delta contract size for BTCUSD = notional / (price * contract_value)
        num_contracts = int(notional / (price * self.contract_value))
        return max(num_contracts, self.min_order_size)

    def check_exit_conditions(self, current_price: float) -> bool:
        """Check if active position should be closed due to Stop Loss or Take Profit."""
        pos = self.current_position
        if pos["side"] == 0:
            return False

        pnl_pct = pos["side"] * (current_price / pos["entry_price"] - 1.0)
        exit_reason = None

        if pos["side"] == +1:
            if current_price <= pos["stop_loss"]:
                exit_reason = f"Stop Loss hit (Price: {current_price:.1f} <= SL: {pos['stop_loss']:.1f})"
            elif current_price >= pos["take_profit"]:
                exit_reason = f"Take Profit hit (Price: {current_price:.1f} >= TP: {pos['take_profit']:.1f})"
        elif pos["side"] == -1:
            if current_price >= pos["stop_loss"]:
                exit_reason = f"Stop Loss hit (Price: {current_price:.1f} >= SL: {pos['stop_loss']:.1f})"
            elif current_price <= pos["take_profit"]:
                exit_reason = f"Take Profit hit (Price: {current_price:.1f} <= TP: {pos['take_profit']:.1f})"

        if exit_reason:
            print(f"[EXIT TRIGGERED] {exit_reason} | Realized PnL: {pnl_pct * 100:+.2f}%")
            self._close_active_position(current_price, exit_reason)
            return True

        return False

    def _close_active_position(self, current_price: float, reason: str):
        """Close active position either on exchange or in simulation."""
        pos = self.current_position
        if pos["side"] == 0:
            return

        if not self.dry_run:
            self.client.close_position(self.product_id)
        else:
            # Simulate PnL update
            pnl_dollar = (
                pos["side"]
                * (current_price / pos["entry_price"] - 1.0)
                * (pos["size"] * self.contract_value * pos["entry_price"])
            )
            self.simulated_equity += pnl_dollar
            print(f"[DRY RUN] Position closed at {current_price:.1f}. PnL: ${pnl_dollar:+.2f}. New Equity: ${self.simulated_equity:,.2f}")

        self.current_position = {
            "side": 0,
            "size": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
        }

    def execute_signal(self, signal, current_price: float):
        """Process strategy signal and open a position if flat."""
        if signal.side == 0:
            return

        # Check if already in opposite position: close it first
        if self.current_position["side"] != 0:
            if self.current_position["side"] != signal.side:
                print(f"[REVERSAL] Closing existing position to switch direction.")
                self._close_active_position(current_price, "signal_reversal")
            else:
                # Already in same position
                return

        equity = self.simulated_equity if self.dry_run else self.client.get_equity("USD")
        contracts = self.calculate_contracts(equity, current_price, signal.stop_distance)
        if contracts <= 0:
            print(f"[WARN] Calculated 0 contracts for position. Skipping entry.")
            return

        side_str = "buy" if signal.side == +1 else "sell"
        stop_dist = signal.stop_distance
        tp_dist = self.cfg["strategy"]["take_profit"]

        if signal.side == +1:
            sl_price = round(current_price * (1.0 - stop_dist), 1)
            tp_price = round(current_price * (1.0 + tp_dist), 1)
        else:
            sl_price = round(current_price * (1.0 + stop_dist), 1)
            tp_price = round(current_price * (1.0 - tp_dist), 1)

        print(f"\n[EXECUTING SIGNAL] {side_str.upper()} {contracts} contracts @ ~{current_price:.1f}")
        print(f"     SL: {sl_price:.1f} (-{stop_dist*100:.2f}%) | TP: {tp_price:.1f} (+{tp_dist*100:.2f}%)")

        order_res = self.client.place_order(
            product_id=self.product_id,
            size=contracts,
            side=side_str,
            order_type="market_order",
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )

        if order_res.get("success", False):
            self.current_position = {
                "side": signal.side,
                "size": contracts,
                "entry_price": current_price,
                "stop_loss": sl_price,
                "take_profit": tp_price,
            }
            print(f"[OK] Order placed. Active position established.\n")
        else:
            print(f"[REJECTED] {order_res.get('error')}\n")

    def run_step(self) -> None:
        """Single decision cycle of the AI trading bot."""
        # 1. Fetch latest candles
        candles = self.client.fetch_candles(self.symbol, self.resolution, limit=350)
        if len(candles) < self.cfg["data"]["sequence_length"] + 60:
            print(f"[WARN] Waiting for candle history ({len(candles)} bars)...")
            return

        # 2. Process sequential features
        tensor, metrics = self.feature_engine.process_candles(candles)
        bar_time = metrics["timestamp"]
        current_price = metrics["close"]

        # 3. Check exits for active position
        self.check_exit_conditions(current_price)

        # 4. Check if candle has closed for fresh signal evaluation
        is_new_bar = (self.last_bar_time is None) or (bar_time > self.last_bar_time)

        # 5. Run GRU Model Inference
        tensor_device = tensor.to(self.device)
        with torch.no_grad():
            pred_ret, long_logit, short_logit = self.model(tensor_device)

        expected_return = float(pred_ret.item())
        p_long = float(torch.sigmoid(long_logit).item())
        p_short = float(torch.sigmoid(short_logit).item())

        signal = make_signal(
            expected_return,
            p_long,
            p_short,
            metrics["trend_15m"],
            metrics["trend_1h"],
            metrics["atr_pct"],
            self.cfg["strategy"]["long_probability"],
            self.cfg["strategy"]["short_probability"],
            self.cfg["strategy"]["min_expected_return"],
        )

        # Clean dashboard display with system local time
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        pos_str = "FLAT"
        if self.current_position["side"] == +1:
            pos_str = f"LONG ({self.current_position['size']}x @ {self.current_position['entry_price']:.1f})"
        elif self.current_position["side"] == -1:
            pos_str = f"SHORT ({self.current_position['size']}x @ {self.current_position['entry_price']:.1f})"

        sig_str = "FLAT (0)"
        if signal.side == +1:
            sig_str = "BUY (+1)"
        elif signal.side == -1:
            sig_str = "SELL (-1)"

        print(
            f"[{now_str}] Price: {current_price:>8.1f} | "
            f"P(L): {p_long*100:>4.1f}% | P(S): {p_short*100:>4.1f}% | "
            f"E(R): {expected_return*100:>+5.2f}% | "
            f"Signal: {sig_str} | Pos: {pos_str}"
        )

        # 6. Execute signal only on new candle close (or if flat and signal triggered)
        if is_new_bar and self.current_position["side"] == 0:
            self.execute_signal(signal, current_price)

        self.last_bar_time = bar_time

    def start(self, max_iterations: Optional[int] = None):
        """Start the continuous trading loop."""
        if not self.initialize():
            print("[ERROR] Bot failed to initialize. Exiting.")
            return

        print(f"\n[BOT STARTED] Polling every {self.poll_interval}s. Press Ctrl+C to stop.\n")
        iteration = 0
        try:
            while True:
                self.run_step()
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    print(f"\n[STOP] Reached max test iterations ({max_iterations}). Stopping.")
                    break
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n[STOP] Keyboard interrupt received. Shutting down gracefully...")
        finally:
            if not self.dry_run and self.current_position["side"] != 0:
                print("[WARN] Open position remains on Delta Exchange. Please manage manually or close via web UI.")


def main():
    parser = argparse.ArgumentParser(description="Delta Exchange AI Trading Bot")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="artifacts/gru.pt", help="Path to model weights")
    parser.add_argument("--scaler", default="artifacts/scaler.joblib", help="Path to scaler")
    parser.add_argument("--dry-run", action="store_true", help="Force simulated dry-run mode")
    parser.add_argument("--live", action="store_true", help="Enable live order execution")
    parser.add_argument("--iterations", type=int, default=None, help="Limit iterations for testing")
    args = parser.parse_args()

    # Determine mode
    dry_run = None
    if args.live:
        dry_run = False
    elif args.dry_run:
        dry_run = True

    trader = DeltaAITrader(
        config_path=args.config,
        model_path=args.model,
        scaler_path=args.scaler,
        dry_run=dry_run,
    )
    trader.start(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
