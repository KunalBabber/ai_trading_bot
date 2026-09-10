import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
import yaml

from features import prepare_data, FEATURE_COLUMNS
from dataset import SequenceDataset
from model import GRUTradingModel


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    losses = []

    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss()

    with torch.no_grad():
        for X, y_reg, y_long, y_short in loader:
            X = X.to(device)
            y_reg = y_reg.to(device)
            y_long = y_long.to(device)
            y_short = y_short.to(device)

            p_reg, p_long, p_short = model(X)
            loss = (
                huber(p_reg, y_reg)
                + 0.5 * bce(p_long, y_long)
                + 0.5 * bce(p_short, y_short)
            )
            losses.append(loss.item())

    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = pd.read_csv(args.csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = prepare_data(
        raw,
        horizon_bars=cfg["data"]["horizon_bars"],
        fee_rate=cfg["data"]["fee_rate"],
        slippage_rate=cfg["data"]["slippage_rate"],
    )

    # Strict chronological split.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_scaled = scaler.transform(val_df[FEATURE_COLUMNS])

    seq_len = cfg["data"]["sequence_length"]

    train_ds = SequenceDataset(
        train_scaled,
        train_df["target_return"].values,
        train_df["target_long"].values,
        train_df["target_short"].values,
        seq_len,
    )
    val_ds = SequenceDataset(
        val_scaled,
        val_df["target_return"].values,
        val_df["target_long"].values,
        val_df["target_short"].values,
        seq_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["model"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["model"]["batch_size"], shuffle=False
    )

    model = GRUTradingModel(
        len(FEATURE_COLUMNS),
        cfg["model"]["hidden_size"],
        cfg["model"]["num_layers"],
        cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["learning_rate"],
        weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss()

    best_val = float("inf")
    patience = 0
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    for epoch in range(cfg["model"]["epochs"]):
        model.train()

        for X, y_reg, y_long, y_short in train_loader:
            X = X.to(device)
            y_reg = y_reg.to(device)
            y_long = y_long.to(device)
            y_short = y_short.to(device)

            optimizer.zero_grad()
            p_reg, p_long, p_short = model(X)

            loss = (
                huber(p_reg, y_reg)
                + 0.5 * bce(p_long, y_long)
                + 0.5 * bce(p_short, y_short)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch+1:03d} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "features": FEATURE_COLUMNS,
                    "config": cfg,
                },
                artifacts / "gru.pt",
            )
        else:
            patience += 1
            if patience >= cfg["model"]["patience"]:
                print("Early stopping.")
                break

    joblib.dump(scaler, artifacts / "scaler.joblib")

    print("\nTraining complete.")
    print(f"Saved: {artifacts / 'gru.pt'}")
    print(f"Saved: {artifacts / 'scaler.joblib'}")
    print(f"Chronological splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Next: run the walk-forward backtest. Do not optimize on the test set.")


if __name__ == "__main__":
    main()
