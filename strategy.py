from dataclasses import dataclass


@dataclass
class Signal:
    side: int          # +1 long, -1 short, 0 flat
    expected_return: float
    probability: float
    stop_distance: float
    confidence: float = 0.0  # Directional AI conviction (0.0 to 1.0)


def make_signal(
    predicted_return: float,
    p_long: float,
    p_short: float,
    trend_15m: float,
    trend_1h: float,
    atr_pct: float,
    long_probability=0.52,
    short_probability=0.52,
    min_expected_return=0.0010,
    require_1h_trend=False,
):
    """
    Generate actionable trade signal using calibrated directional AI conviction
    and trend momentum filters.
    
    In multi-task dual-head binary classification, raw sigmoid outputs reflect the 
    underlying prior label frequency (~39% base rate). Calibrated directional 
    conviction normalizes long vs short edge:
        rel_long = p_long / (p_long + p_short)
        rel_short = p_short / (p_long + p_short)
    
    A trigger occurs when directional conviction (or raw probability) exceeds the 
    threshold, predicted return covers fees + slippage, and higher-timeframe trend aligns.
    """
    total_p = p_long + p_short + 1e-12
    rel_long = p_long / total_p
    rel_short = p_short / total_p
    best_conviction = max(rel_long, rel_short)

    long_conviction_ok = (rel_long >= long_probability) or (p_long >= long_probability)
    short_conviction_ok = (rel_short >= short_probability) or (p_short >= short_probability)

    long_trend_ok = (trend_15m > 0) and (trend_1h > 0 if require_1h_trend else True)
    short_trend_ok = (trend_15m < 0) and (trend_1h < 0 if require_1h_trend else True)

    stop_dist = max(atr_pct * 1.5, 0.008)

    if (
        long_conviction_ok
        and predicted_return >= min_expected_return
        and long_trend_ok
    ):
        return Signal(+1, predicted_return, rel_long, stop_dist, confidence=rel_long)

    if (
        short_conviction_ok
        and predicted_return <= -min_expected_return
        and short_trend_ok
    ):
        return Signal(-1, predicted_return, rel_short, stop_dist, confidence=rel_short)

    return Signal(0, predicted_return, best_conviction, 0.0, confidence=best_conviction)


def position_fraction(equity, stop_distance, risk_fraction=0.003,
                      max_fraction=0.20):
    """
    Risk-based sizing:
        position_fraction ≈ risk_budget / stop_distance

    Example:
        0.3% risk / 1% stop ≈ 30% notional,
        capped by max_fraction.
    """
    if stop_distance <= 0:
        return 0.0
    return min(risk_fraction / stop_distance, max_fraction)
