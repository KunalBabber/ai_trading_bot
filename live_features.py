"""
Live Feature Pipeline for Delta Exchange
Transforms incoming live OHLCV candles into scaled PyTorch sequence tensors
compatible with the trained GRU model.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import joblib

from features import add_base_features, FEATURE_COLUMNS


class LiveFeatureEngine:
    def __init__(self, scaler_path: str = "artifacts/scaler.joblib", sequence_length: int = 96):
        self.scaler = joblib.load(scaler_path)
        self.sequence_length = sequence_length

    def process_candles(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Takes raw OHLCV DataFrame with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            tensor: torch.FloatTensor of shape (1, sequence_length, num_features)
            latest_metrics: Dict with latest bar context (close, atr_pct, trend_15m, trend_1h, timestamp)
        """
        if len(df) < self.sequence_length + 60:
            raise ValueError(
                f"Insufficient candles: got {len(df)}, need at least {self.sequence_length + 60} "
                "to compute rolling indicators and multi-timeframe trends."
            )

        # 1. Compute technical indicators & multi-timeframe EMAs
        feat_df = add_base_features(df)

        # 2. Clean infinite values and drop indicator warmup NaNs
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

        if len(feat_df) < self.sequence_length:
            raise ValueError(
                f"After indicator warmup, only {len(feat_df)} valid bars remain. "
                f"Need at least {self.sequence_length} bars."
            )

        # 3. Extract the most recent sequence_length bars
        seq_window = feat_df.iloc[-self.sequence_length:].reset_index(drop=True)
        latest_row = seq_window.iloc[-1]

        # 4. Scale features using the saved RobustScaler
        scaled_features = self.scaler.transform(seq_window[FEATURE_COLUMNS])

        # 5. Convert to tensor of shape (1, sequence_length, num_features)
        tensor = torch.tensor(
            scaled_features[np.newaxis, :, :],
            dtype=torch.float32,
        )

        metrics = {
            "timestamp": latest_row["timestamp"],
            "close": float(latest_row["close"]),
            "atr_pct": float(latest_row["atr_pct"]),
            "trend_15m": float(latest_row["trend_15m"]),
            "trend_1h": float(latest_row["trend_1h"]),
            "rsi_14": float(latest_row["rsi_14"]),
        }

        return tensor, metrics
