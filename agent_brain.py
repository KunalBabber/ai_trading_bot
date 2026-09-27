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
    cognitive_score: float = 50.0
    setup_type: str = "Autonomous Synthesis"


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
        Executes a complete cognitive reasoning cycle using an Autonomous Multi-Factor Matrix.
        Instead of rigid binary rules, the agent holistically evaluates directional neural alpha,
        order flow velocity, discount/premium exhaustion, and regime payoff asymmetry.
        """
        cfg = cfg or {}
        temperament = cfg.get("temperament", "Autonomous Cognitive")

        p_long = float(predictions.get("p_long", 0.5))
        p_short = float(predictions.get("p_short", 0.5))
        exp_ret = float(predictions.get("expected_return", 0.0))

        total_p = p_long + p_short + 1e-12
        rel_long = p_long / total_p
        rel_short = p_short / total_p
        active_direction = "BULLISH (Long)" if rel_long >= rel_short else "BEARISH (Short)"
        best_conviction = max(rel_long, rel_short)

        # 1. Market Regime Classification
        regime = self.classify_market_regime(metrics)

        # 2. Episodic Memory & Learned Adaptations
        adaptations = self.memory.data.get("active_adaptations", {})
        defense_mode = adaptations.get("defense_mode", False)
        sl_multiplier = adaptations.get("sl_buffer_multiplier", 1.0)
        tp_multiplier = adaptations.get("tp_target_multiplier", 1.0)

        atr_pct = metrics.get("atr_pct", 0.005)
        t15 = metrics.get("trend_15m", 0.0)
        t1h = metrics.get("trend_1h", 0.0)
        rsi = metrics.get("rsi_14", 0.50)

        # Dynamic brackets: tailored to regime and volatility
        effective_sl = max(atr_pct * 1.5 * sl_multiplier, 0.008 * sl_multiplier)
        effective_tp = max(atr_pct * 2.2 * tp_multiplier, 0.016 * tp_multiplier)

        # 3. Autonomous Cognitive Scoring Matrix (0 to 100)
        bullish_score = 0.0
        bearish_score = 0.0
        setup_type = "Consolidation Scan"

        # A. Neural Alpha Component (0 to 50 points)
        bullish_score += rel_long * 50.0
        bearish_score += rel_short * 50.0

        # B. Expected Return Quality (0 to 20 points)
        if exp_ret > 0:
            bullish_score += min(20.0, 10.0 + (exp_ret * 200.0))
            bearish_score -= 8.0
        elif exp_ret < 0:
            bearish_score += min(20.0, 10.0 + (abs(exp_ret) * 200.0))
            bullish_score -= 8.0

        # C. Order Flow, Pullback & Discount Context (0 to 20 points)
        if rel_long >= rel_short:
            # Bullish context
            if t15 > 0:
                bullish_score += 15.0
                setup_type = "Bullish Momentum Expansion"
            elif rsi <= 0.46:
                # Intelligent pullback / discount accumulation
                bullish_score += 14.0
                setup_type = "Dip Accumulation at RSI Discount"
            else:
                bullish_score += 4.0
                setup_type = "Consolidation Drift"

            if t1h > 0:
                bullish_score += 8.0
            elif t1h < -0.002:
                bullish_score -= 5.0
        else:
            # Bearish context
            if t15 < 0:
                bearish_score += 15.0
                setup_type = "Bearish Momentum Breakdown"
            elif rsi >= 0.54:
                # Intelligent rally / premium shorting
                bearish_score += 14.0
                setup_type = "Premium Shorting at RSI Exhaustion"
            else:
                bearish_score += 4.0
                setup_type = "Consolidation Drift"

            if t1h < 0:
                bearish_score += 8.0
            elif t1h > 0.002:
                bearish_score -= 5.0

        # D. Regime Synergy (0 to 10 points)
        if regime == "BULL_MOMENTUM":
            bullish_score += 10.0
            bearish_score -= 5.0
        elif regime == "BEAR_MOMENTUM":
            bearish_score += 10.0
            bullish_score -= 5.0
        elif regime == "OVERSOLD_BOUNCE":
            bullish_score += 12.0
            setup_type = "Oversold Mean-Reversion Bounce"
        elif regime == "OVERBOUGHT_STRETCH":
            bearish_score += 12.0
            setup_type = "Overbought Mean-Reversion Short"

        # E. Episodic Memory Feedback
        regime_stats = self.memory.data.get("regime_stats", {}).get(regime, {})
        regime_winrate = regime_stats.get("win_rate", 50.0)
        regime_trades = regime_stats.get("trades", 0)
        if regime_trades >= 2:
            if regime_winrate >= 60.0:
                bullish_score += 4.0 if rel_long >= rel_short else 0.0
                bearish_score += 4.0 if rel_short > rel_long else 0.0
            elif regime_winrate <= 35.0:
                bullish_score -= 5.0
                bearish_score -= 5.0

        # 4. Decision Hurdle based on Agent Temperament & Learning
        if temperament == "Aggressive Edge Hunter":
            hurdle = 48.0
        elif temperament == "Defensive Capital Preserver":
            hurdle = 56.0
        else:
            hurdle = 50.5  # Balanced autonomous

        if defense_mode:
            hurdle += 3.5

        top_score = max(bullish_score, bearish_score)

        # 5. Executive Action
        action = 0
        action_label = "STANDBY (Preserving Capital)"

        if bullish_score > bearish_score and bullish_score >= hurdle:
            action = +1
            action_label = "AUTONOMOUS BUY 🚀"
        elif bearish_score > bullish_score and bearish_score >= hurdle:
            action = -1
            action_label = "AUTONOMOUS SELL 🔻"

        # 6. Natural Language Synthesis
        thesis_parts = [
            f"Regime: {regime}. Setup: {setup_type}.",
            f"Neural Analysis: {best_conviction*100:.1f}% conviction for {active_direction} (Expected Return: {exp_ret*100:+.2f}%).",
            f"Momentum & RSI: 15m trend velocity {t15*100:+.2f}%, RSI at {rsi*100:.1f}%.",
        ]
        market_thesis = " ".join(thesis_parts)

        risk_parts = []
        if defense_mode:
            risk_parts.append("🛡️ Defense Mode Active (+3.5pt hurdle & wider stops) due to recent market turbulence.")
        else:
            risk_parts.append(f"Autonomous Cognitive Posture: Risk-Reward 1:{effective_tp/effective_sl:.1f}.")
        risk_parts.append(f"Dynamic Protective Stop: {effective_sl*100:.1f}% | Dynamic Target: {effective_tp*100:.1f}%.")
        risk_evaluation = " ".join(risk_parts)

        learning_notes = (
            f"Total experience: {self.memory.data.get('total_trades', 0)} trades. "
            f"Regime historical win-rate: {regime_winrate:.1f}%. "
            f"Cognitive score: {top_score:.1f}/100 (Hurdle: {hurdle:.1f})."
        )

        if action == +1:
            reasoning_summary = (
                f"AUTONOMOUS BUY: Cognitive score {bullish_score:.1f} exceeds hurdle {hurdle:.1f}. "
                f"Brain identified {setup_type} with favorable {effective_tp/effective_sl:.1f}:1 reward-to-risk. "
                f"Targets: TP ${current_price*(1+effective_tp):,.1f} (+{effective_tp*100:.1f}%), SL ${current_price*(1-effective_sl):,.1f} (-{effective_sl*100:.1f}%)."
            )
        elif action == -1:
            reasoning_summary = (
                f"AUTONOMOUS SELL: Cognitive score {bearish_score:.1f} exceeds hurdle {hurdle:.1f}. "
                f"Brain identified {setup_type} with favorable {effective_tp/effective_sl:.1f}:1 reward-to-risk. "
                f"Targets: TP ${current_price*(1-effective_tp):,.1f} (-{effective_tp*100:.1f}%), SL ${current_price*(1+effective_sl):,.1f} (+{effective_sl*100:.1f}%)."
            )
        else:
            gap = hurdle - top_score
            reasoning_summary = (
                f"STANDBY: Edge score {top_score:.1f} is {gap:.1f} points below operational threshold ({hurdle:.1f}). "
                f"Maintaining capital preservation until high-asymmetry opportunity aligns."
            )

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
            cognitive_score=round(top_score, 1),
            setup_type=setup_type,
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
