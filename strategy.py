from dataclasses import dataclass


@dataclass
class Signal:
    side: int          # +1 long, -1 short, 0 flat
    expected_return: float
    probability: float
    stop_distance: float


def make_signal(
    predicted_return: float,
    p_long: float,
    p_short: float,
    trend_15m: float,
    trend_1h: float,
    atr_pct: float,
    long_probability=0.58,
    short_probability=0.58,
    min_expected_return=0.0015,
):
    # A simple regime filter:
    # long requires positive higher-timeframe trend,
    # short requires negative higher-timeframe trend.
    if (
        p_long >= long_probability
        and predicted_return >= min_expected_return
        and trend_15m > 0
        and trend_1h > 0
    ):
        return Signal(+1, predicted_return, p_long, max(atr_pct * 1.5, 0.003))

    if (
        p_short >= short_probability
        and predicted_return <= -min_expected_return
        and trend_15m < 0
        and trend_1h < 0
    ):
        return Signal(-1, predicted_return, p_short, max(atr_pct * 1.5, 0.003))

    return Signal(0, predicted_return, max(p_long, p_short), 0.0)


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
