import torch
import torch.nn as nn


class GRUTradingModel(nn.Module):
    """
    Multi-task sequence model:
      - regression head: expected future return
      - long head: P(long opportunity)
      - short head: P(short opportunity)
    """

    def __init__(self, n_features, hidden_size=96, num_layers=2, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.return_head = nn.Linear(hidden_size, 1)
        self.long_head = nn.Linear(hidden_size, 1)
        self.short_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        h = self.shared(h)

        future_return = self.return_head(h).squeeze(-1)
        long_logit = self.long_head(h).squeeze(-1)
        short_logit = self.short_head(h).squeeze(-1)

        return future_return, long_logit, short_logit
