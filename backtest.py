import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import yaml
import matplotlib.pyplot as plt

from features import prepare_data, FEATURE_COLUMNS
from dataset import SequenceDataset
from model import GRUTradingModel
from strategy import make_signal, position_fraction


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_models(config_path, model_path, scaler_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(model_path, map_location="cpu")
    model = GRUTradingModel(
        len(FEATURE_COLUMNS),
        cfg["model"]["hidden_size"],
        cfg["model"]["num_layers"],
        cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = joblib.load(scaler_path)
    return cfg, model, scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="artifacts/gru.pt")
    ap.add_argument("--scaler", default="artifacts/scaler.joblib")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg, model, scaler = load_models(args.config, args.model, args.scaler)

    raw = pd.read_csv(args.csv)
    df = prepare_data(
        raw,
        horizon_bars=cfg["data"]["horizon_bars"],
        fee_rate=cfg["data"]["fee_rate"],
        slippage_rate=cfg["data"]["slippage_rate"],
    )

    # Backtest ONLY on the last 15% to preserve the test set.
    start = int(len(df) * 0.85)
    test = df.iloc[start:].reset_index(drop=True)

    scaled = scaler.transform(test[FEATURE_COLUMNS].values)
    seq_len = cfg["data"]["sequence_length"]

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    returns = []
    trades = 0
    wins = 0

    fee = cfg["data"]["fee_rate"]
    slippage = cfg["data"]["slippage_rate"]

    model.eval()

    for i in range(seq_len - 1, len(test) - cfg["data"]["horizon_bars"]):
        x = torch.tensor(
            scaled[i-seq_len+1:i+1][None, ...],
            dtype=torch.float32
        )

        with torch.no_grad():
            pred_ret, long_logit, short_logit = model(x)

        pred_ret = float(pred_ret.item())
        p_long = sigmoid(float(long_logit.item()))
        p_short = sigmoid(float(short_logit.item()))

        row = test.iloc[i]
        signal = make_signal(
            pred_ret,
            p_long,
            p_short,
            row["trend_15m"],
            row["trend_1h"],
            row["atr_pct"],
            cfg["strategy"]["long_probability"],
            cfg["strategy"]["short_probability"],
            cfg["strategy"]["min_expected_return"],
        )

        if signal.side == 0:
            returns.append(0.0)
            continue

        equity_before = equity
        frac = position_fraction(
            equity,
            signal.stop_distance,
            cfg["strategy"]["risk_per_trade"],
            cfg["strategy"]["max_position_fraction"],
        )

        future_close = test.iloc[
            i + cfg["data"]["horizon_bars"]
        ]["close"]
        entry = row["close"]
        raw_move = signal.side * (future_close / entry - 1.0)

        # Cost-aware realized return.
        cost = 2 * (fee + slippage)
        trade_return = frac * (raw_move - cost)

        equity *= max(0.0, 1.0 + trade_return)
        returns.append(trade_return)

        trades += 1
        if trade_return > 0:
            wins += 1

        peak = max(peak, equity)
        dd = equity / peak - 1.0
        max_dd = min(max_dd, dd)

    rets = np.asarray(returns, dtype=float)

    if len(rets) and np.std(rets) > 0:
        sharpe = np.sqrt(252 * 24 * 12) * rets.mean() / rets.std()
    else:
        sharpe = 0.0

    print("\n=== OUT-OF-SAMPLE TEST ===")
    print(f"Final equity:      {equity:.4f}x")
    print(f"Total return:      {(equity - 1) * 100:.2f}%")
    print(f"Trades:            {trades}")
    print(f"Win rate:          {(wins / trades * 100) if trades else 0:.2f}%")
    print(f"Max drawdown:      {max_dd * 100:.2f}%")
    print(f"Approx Sharpe:     {sharpe:.2f}")

    Path("artifacts").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 4))
    curve = np.cumprod(1 + rets)
    plt.plot(curve)
    plt.title("Out-of-Sample Equity Curve")
    plt.xlabel("Decision")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig("artifacts/equity_curve.png", dpi=150)
    plt.close()

    print("Saved: artifacts/equity_curve.png")


if __name__ == "__main__":
    main()
