import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 15000
ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

returns = 0.00002 + 0.0015 * rng.standard_normal(n)
close = 40000 * np.exp(np.cumsum(returns))
open_ = np.r_[close[0], close[:-1]]
noise = np.abs(rng.normal(0, 0.001, n))
high = np.maximum(open_, close) * (1 + noise)
low = np.minimum(open_, close) * (1 - noise)
volume = rng.lognormal(mean=10.5, sigma=0.4, size=n)

df = pd.DataFrame({
    "timestamp": ts,
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
})
Path("data").mkdir(exist_ok=True)
df.to_csv("data/example_5m.csv", index=False)
print("created data/example_5m.csv")
