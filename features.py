import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_12", "ret_48",
    "log_volume_change",
    "range_pct", "body_pct",
    "ema_9_dist", "ema_21_dist", "ema_50_dist",
    "rsi_14", "atr_pct", "volatility_24",
    "volume_z",
    "trend_15m", "trend_1h",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    close = df["close"]
    volume = df["volume"]

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_12"] = close.pct_change(12)
    df["ret_48"] = close.pct_change(48)

    df["log_volume_change"] = np.log1p(volume).diff()
    df["range_pct"] = (df["high"] - df["low"]) / (close + 1e-12)
    df["body_pct"] = (df["close"] - df["open"]).abs() / (close + 1e-12)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    df["ema_9_dist"] = close / ema9 - 1
    df["ema_21_dist"] = close / ema21 - 1
    df["ema_50_dist"] = close / ema50 - 1

    df["rsi_14"] = rsi(close, 14) / 100.0
    df["atr_pct"] = atr(df, 14) / (close + 1e-12)
    df["volatility_24"] = df["ret_1"].rolling(24).std()

    vol_mean = volume.rolling(48).mean()
    vol_std = volume.rolling(48).std()
    df["volume_z"] = (volume - vol_mean) / (vol_std + 1e-12)

    # Higher-timeframe trend signals are calculated with resampled closes.
    x = df.set_index("timestamp")["close"]
    close_15m = x.resample("15min").last().ffill()
    close_1h = x.resample("1h").last().ffill()

    ema15_fast = close_15m.ewm(span=8, adjust=False).mean()
    ema15_slow = close_15m.ewm(span=21, adjust=False).mean()
    trend15 = (ema15_fast / ema15_slow - 1).rename("trend_15m")

    ema1h_fast = close_1h.ewm(span=8, adjust=False).mean()
    ema1h_slow = close_1h.ewm(span=21, adjust=False).mean()
    trend1h = (ema1h_fast / ema1h_slow - 1).rename("trend_1h")

    df = df.set_index("timestamp")
    df = df.join(trend15.resample("5min").ffill(), how="left")
    df = df.join(trend1h.resample("5min").ffill(), how="left")
    df["trend_15m"] = df["trend_15m"].ffill()
    df["trend_1h"] = df["trend_1h"].ffill()
    df = df.reset_index()

    return df


def make_labels(df: pd.DataFrame, horizon_bars: int = 12,
                fee_rate: float = 0.0005,
                slippage_rate: float = 0.0002) -> pd.DataFrame:
    df = df.copy()

    future_return = df["close"].shift(-horizon_bars) / df["close"] - 1.0
    round_trip_cost = 2 * (fee_rate + slippage_rate)

    # Target is cost-aware future return.
    df["target_return"] = future_return
    df["target_long"] = (future_return > round_trip_cost).astype(int)
    df["target_short"] = (future_return < -round_trip_cost).astype(int)

    return df


def prepare_data(df: pd.DataFrame, horizon_bars: int = 12,
                 fee_rate: float = 0.0005,
                 slippage_rate: float = 0.0002) -> pd.DataFrame:
    df = add_base_features(df)
    df = make_labels(df, horizon_bars, fee_rate, slippage_rate)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [
        "target_return", "target_long", "target_short"
    ]).reset_index(drop=True)
    return df
