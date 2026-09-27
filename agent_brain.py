"""
Autonomous Cognitive Trading Agent Brain
Implements multi-factor deliberate reasoning, market regime classification,
trade reflection, episodic experience memory, and continuous adaptive learning.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


MEMORY_FILE = "artifacts/agent_memory.json"


@dataclass
class AgentThought:
    timestamp: str
    cycle: int
    current_price: float
    regime: str
    active_direction: str
    neural_conviction: float
    expected_return: float
    action: int  # +1 BUY, -1 SELL, 0 FLAT
    action_label: str
    dynamic_tp_pct: float
    dynamic_sl_pct: float
    reasoning_summary: str
    market_thesis: str
    risk_evaluation: str
    learning_notes: str
    conditions_met: int
    total_conditions: int


class AgentMemory:
    """
    Episodic memory ledger that records trade reflections,
    tracks regime-specific win rates, and dynamically adjusts risk parameters.
    """

    def __init__(self, memory_path: str = MEMORY_FILE):
        self.memory_path = memory_path
        self.data = self._load()

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "rolling_win_rate": 0.0,
            "recent_pnls": [],
            "regime_stats": {
                "BULL_MOMENTUM": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0},
                "BEAR_MOMENTUM": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0},
                "CHOPPY_RANGE": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0},
                "HIGH_VOLATILITY": {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0},
            },
            "active_adaptations": {
                "defense_mode": False,
                "conviction_bias": 0.0,
                "sl_buffer_multiplier": 1.0,
                "tp_target_multiplier": 1.0,
            },
            "reflections": [],
        }

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                # Ensure all default keys exist
                defaults = self._default_memory()
                for k, v in defaults.items():
                    if k not in content:
                        content[k] = v
                return content
            except Exception:
                pass
        return self._default_memory()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def record_reflection(self, reflection: Dict[str, Any]):
        """Records a post-trade reflection and adapts behavioral parameters."""
        self.data["total_trades"] += 1
        pnl = float(reflection.get("pnl", 0.0))
        is_win = pnl > 0

        if is_win:
            self.data["wins"] += 1
        else:
            self.data["losses"] += 1

        self.data["win_rate"] = round(self.data["wins"] / self.data["total_trades"] * 100.0, 1)

        # Rolling win rate (last 10 trades)
        pnls = self.data.get("recent_pnls", [])
        pnls.append(pnl)
        if len(pnls) > 10:
            pnls.pop(0)
        self.data["recent_pnls"] = pnls

        rolling_wins = sum(1 for p in pnls if p > 0)
        self.data["rolling_win_rate"] = round(rolling_wins / len(pnls) * 100.0, 1) if pnls else 0.0

        # Regime-specific stats
        regime = reflection.get("regime", "CHOPPY_RANGE")
        if regime not in self.data["regime_stats"]:
            self.data["regime_stats"][regime] = {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        r_stat = self.data["regime_stats"][regime]
        r_stat["trades"] += 1
        if is_win:
            r_stat["wins"] += 1
        else:
            r_stat["losses"] += 1
        r_stat["win_rate"] = round(r_stat["wins"] / r_stat["trades"] * 100.0, 1)

        # Self-learning adjustments
        adaptations = self.data["active_adaptations"]
        if len(pnls) >= 4 and self.data["rolling_win_rate"] < 40.0:
            # Activate Defensive Capital Protection Mode
            adaptations["defense_mode"] = True
            adaptations["conviction_bias"] = +0.03  # Require +3% higher conviction before entering
            adaptations["sl_buffer_multiplier"] = 1.20  # Widen stops by 20% to prevent chop shakeouts
            adaptations["tp_target_multiplier"] = 0.90  # Take profits slightly earlier
        elif self.data["rolling_win_rate"] >= 60.0:
            # Favorable Regime: normal parameters
            adaptations["defense_mode"] = False
            adaptations["conviction_bias"] = 0.0
            adaptations["sl_buffer_multiplier"] = 1.0
            adaptations["tp_target_multiplier"] = 1.10  # Allow winners to run
        else:
            adaptations["defense_mode"] = False
            adaptations["conviction_bias"] = 0.0
            adaptations["sl_buffer_multiplier"] = 1.0
            adaptations["tp_target_multiplier"] = 1.0

        # Store reflection in history (capped at 50)
        reflections = self.data.get("reflections", [])
        reflections.insert(0, reflection)
        if len(reflections) > 50:
            reflections.pop()
        self.data["reflections"] = reflections

        self.save()


class CognitiveAgentBrain:
    """
    Cognitive Agent Brain that deliberates on market conditions,
    synthesizes multi-factor data, and learns continuously from trade outcomes.
    """

    def __init__(self, memory_path: str = MEMORY_FILE):
        self.memory = AgentMemory(memory_path)

    def classify_market_regime(self, metrics: Dict[str, Any]) -> str:
        """Classifies the market into a distinct behavioral regime."""
        t15 = metrics.get("trend_15m", 0.0)
        t1h = metrics.get("trend_1h", 0.0)
        atr_pct = metrics.get("atr_pct", 0.005)
        rsi = metrics.get("rsi_14", 0.50)

        # High Volatility
        if atr_pct >= 0.012:
            return "HIGH_VOLATILITY"

        # Trending regimes
        if t15 > 0.0008 and t1h >= -0.0005:
            return "BULL_MOMENTUM"
        if t15 < -0.0008 and t1h <= 0.0005:
            return "BEAR_MOMENTUM"

        # Overbought / Oversold Mean Reversion
        if rsi >= 0.72:
            return "OVERBOUGHT_STRETCH"
        if rsi <= 0.28:
            return "OVERSOLD_BOUNCE"

        return "CHOPPY_RANGE"

    def deliberate(
        self,
        metrics: Dict[str, Any],
        predictions: Dict[str, Any],
        current_price: float,
        active_position: Dict[str, Any],
        cycle_count: int = 1,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> AgentThought:
        """
        Executes a complete cognitive reasoning cycle:
        1. Analyzes market regime & indicators
        2. Inquires episodic memory & active adaptations
        3. Formulates a deliberate trade hypothesis
        4. Makes an executive decision (BUY, SELL, FLAT) with dynamic SL/TP
        """
        cfg = cfg or {}
        base_threshold = float(cfg.get("long_probability", 0.52))
        min_return_hurdle = float(cfg.get("min_expected_return", 0.0010))
        base_tp = float(cfg.get("take_profit", 0.016))
        require_1h_trend = bool(cfg.get("require_1h_trend", False))

        p_long = float(predictions.get("p_long", 0.5))
        p_short = float(predictions.get("p_short", 0.5))
        exp_ret = float(predictions.get("expected_return", 0.0))

        total_p = p_long + p_short + 1e-12
        rel_long = p_long / total_p
        rel_short = p_short / total_p
        best_conviction = max(rel_long, rel_short)
        active_direction = "BULLISH (Long)" if rel_long >= rel_short else "BEARISH (Short)"

        # 1. Market Regime
        regime = self.classify_market_regime(metrics)

        # 2. Episodic Memory Adaptations
        adaptations = self.memory.data.get("active_adaptations", {})
        defense_mode = adaptations.get("defense_mode", False)
        conviction_bias = adaptations.get("conviction_bias", 0.0)
        sl_multiplier = adaptations.get("sl_buffer_multiplier", 1.0)
        tp_multiplier = adaptations.get("tp_target_multiplier", 1.0)

        # Adjusted threshold from learning
        effective_threshold = base_threshold + conviction_bias
        effective_tp = base_tp * tp_multiplier
        atr_pct = metrics.get("atr_pct", 0.005)
        effective_sl = max(atr_pct * 1.5 * sl_multiplier, 0.008 * sl_multiplier)

        # 3. Multi-Factor Evaluation
        t15 = metrics.get("trend_15m", 0.0)
        t1h = metrics.get("trend_1h", 0.0)
        rsi = metrics.get("rsi_14", 0.50)

        c1_conviction_ok = (rel_long >= effective_threshold) or (rel_short >= effective_threshold)
        c2_return_ok = abs(exp_ret) >= min_return_hurdle
        c3_trend_ok = (t15 > 0 if rel_long >= rel_short else t15 < 0)
        c4_macro_ok = True if not require_1h_trend else (t1h > 0 if rel_long >= rel_short else t1h < 0)

        conditions_met = sum([c1_conviction_ok, c2_return_ok, c3_trend_ok, c4_macro_ok])

        # 4. Executive Decision
        action = 0
        action_label = "FLAT (Standby)"

        if conditions_met == 4:
            if rel_long > rel_short and exp_ret > 0 and t15 > 0:
                action = +1
                action_label = "BUY (+1) 🚀"
            elif rel_short > rel_long and exp_ret < 0 and t15 < 0:
                action = -1
                action_label = "SELL (-1) 🔻"

        # 5. Cognitive Narrative Synthesis
        # Market Thesis
        thesis_parts = []
        regime_desc = {
            "BULL_MOMENTUM": "Market is in an active Bull Momentum Expansion with constructive higher-timeframe flows.",
            "BEAR_MOMENTUM": "Market is in an active Bear Momentum Expansion with heavy selling pressure.",
            "HIGH_VOLATILITY": "Market is experiencing High Volatility; wider ATR swings require defensive risk control.",
            "CHOPPY_RANGE": "Market is in a Sideways Consolidation Range; price is searching for directional volume.",
            "OVERBOUGHT_STRETCH": "Market is technically Overbought on RSI; upside may encounter near-term resistance.",
            "OVERSOLD_BOUNCE": "Market is technically Oversold on RSI; potential mean-reversion bounce zone.",
        }.get(regime, "Consolidation phase.")

        thesis_parts.append(f"Regime: {regime_desc}")
        thesis_parts.append(
            f"Neural Analysis: GRU predicts {best_conviction*100:.1f}% conviction for {active_direction} "
            f"with expected return of {exp_ret*100:+.2f}%."
        )
        if t15 != 0:
            trend_dir = "Bullish" if t15 > 0 else "Bearish"
            thesis_parts.append(f"15m Trend: {trend_dir} (velocity: {t15*100:+.2f}%).")

        market_thesis = " ".join(thesis_parts)

        # Risk Evaluation
        risk_parts = []
        if defense_mode:
            risk_parts.append("🛡️ Defensive Mode Active: Higher conviction required due to recent adverse market chop.")
        else:
            risk_parts.append("Optimal Risk Posture: Standard capital allocation with 1:1.6 risk/reward.")

        if rsi > 0.70 or rsi < 0.30:
            risk_parts.append(f"RSI extreme ({rsi*100:.1f}%): Monitor for rapid momentum exhaustion.")
        else:
            risk_parts.append(f"RSI balanced ({rsi*100:.1f}%): Sustainable continuation environment.")

        risk_evaluation = " ".join(risk_parts)

        # Learning & Adaptation Notes
        regime_winrate = self.memory.data.get("regime_stats", {}).get(regime, {}).get("win_rate", 50.0)
        total_mem_trades = self.memory.data.get("total_trades", 0)
        learning_notes = (
            f"Experience: {total_mem_trades} trades analyzed across sessions. "
            f"Regime historical win-rate: {regime_winrate:.1f}%. "
            f"Effective conviction threshold: {effective_threshold*100:.1f}% (Base: {base_threshold*100:.1f}%)."
        )

        # Reasoning Summary
        if action == +1:
            reasoning_summary = (
                f"EXECUTING LONG: High-probability bullish alignment ({best_conviction*100:.1f}% conviction >= {effective_threshold*100:.0f}%), "
                f"expected move {exp_ret*100:+.2f}% covers costs, 15m trend confirmed. Bracket: TP {effective_tp*100:.1f}%, SL {effective_sl*100:.1f}%."
            )
        elif action == -1:
            reasoning_summary = (
                f"EXECUTING SHORT: High-probability bearish alignment ({best_conviction*100:.1f}% conviction >= {effective_threshold*100:.0f}%), "
                f"expected move {exp_ret*100:+.2f}% covers costs, 15m trend confirmed. Bracket: TP {effective_tp*100:.1f}%, SL {effective_sl*100:.1f}%."
            )
        else:
            missing = []
            if not c1_conviction_ok:
                missing.append(f"Conviction ({best_conviction*100:.1f}% < {effective_threshold*100:.0f}%)")
            if not c2_return_ok:
                missing.append(f"Hurdle ({abs(exp_ret)*100:.2f}% < {min_return_hurdle*100:.2f}%)")
            if not c3_trend_ok:
                missing.append("15m Trend Alignment")
            if not c4_macro_ok:
                missing.append("1h Macro Trend Alignment")
            missing_str = ", ".join(missing) if missing else "Confirmation"
            reasoning_summary = f"STANDBY: Waiting for {missing_str}. Maintaining capital preservation."

        now_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        return AgentThought(
            timestamp=now_str,
            cycle=cycle_count,
            current_price=current_price,
            regime=regime,
            active_direction=active_direction,
            neural_conviction=best_conviction,
            expected_return=exp_ret,
            action=action,
            action_label=action_label,
            dynamic_tp_pct=effective_tp,
            dynamic_sl_pct=effective_sl,
            reasoning_summary=reasoning_summary,
            market_thesis=market_thesis,
            risk_evaluation=risk_evaluation,
            learning_notes=learning_notes,
            conditions_met=conditions_met,
            total_conditions=4,
        )

    def reflect_on_trade(
        self,
        trade_data: Dict[str, Any],
        entry_regime: str = "CHOPPY_RANGE",
    ) -> Dict[str, Any]:
        """
        Post-Mortem Trade Reflection:
        Analyzes outcome, reasons why it won or lost, and extracts actionable lessons.
        """
        pnl = float(trade_data.get("pnl", 0.0))
        side = trade_data.get("side", "BUY")
        entry = float(trade_data.get("entry", 0.0))
        exit_price = float(trade_data.get("exit", 0.0))
        reason = trade_data.get("reason", "TAKE_PROFIT")

        is_win = pnl > 0
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

        if is_win:
            lesson = (
                f"Trade WIN (+${pnl:,.2f}) on {side} in {entry_regime}. "
                f"Take-Profit target achieved smoothly. Entry conviction was justified."
            )
        else:
            if "Stop Loss" in reason:
                lesson = (
                    f"Trade STOPPED OUT (-${abs(pnl):,.2f}) on {side} in {entry_regime}. "
                    f"Market volatility triggered stop. Adjusted: widening defensive buffer for future setups in this regime."
                )
            else:
                lesson = (
                    f"Trade CLOSED (-${abs(pnl):,.2f}) on {side}. "
                    f"Manual or external intervention observed. Documenting state for regime risk tracking."
                )

        reflection = {
            "time": now_str,
            "side": side,
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl,
            "is_win": is_win,
            "regime": entry_regime,
            "exit_reason": reason,
            "lesson_learned": lesson,
        }

        self.memory.record_reflection(reflection)
        return reflection


# Global singleton instance
agent_brain = CognitiveAgentBrain()
