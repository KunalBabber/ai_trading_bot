"""
Exchange-agnostic paper-trading skeleton.

For live execution, connect the feed and order functions to your exchange
through CCXT or the exchange SDK. Keep this layer separate from the model.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class PaperPosition:
    side: int = 0
    size: float = 0.0
    entry: float = 0.0


class PaperBroker:
    def __init__(self, starting_equity=10000.0):
        self.equity = starting_equity
        self.position = PaperPosition()
        self.trades = []

    def mark_to_market(self, price):
        if self.position.side == 0:
            return self.equity
        pnl = (
            self.position.side
            * (price / self.position.entry - 1.0)
            * self.position.size
        )
        return self.equity + pnl

    def open(self, side, size, price):
        if self.position.side != 0:
            return False

        self.position = PaperPosition(
            side=side,
            size=size,
            entry=price,
        )
        return True

    def close(self, price, reason="signal"):
        if self.position.side == 0:
            return

        pnl = (
            self.position.side
            * (price / self.position.entry - 1.0)
            * self.position.size
        )
        self.equity += pnl

        self.trades.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "side": self.position.side,
                "size": self.position.size,
                "entry": self.position.entry,
                "exit": price,
                "pnl": pnl,
                "reason": reason,
                "equity": self.equity,
            }
        )

        self.position = PaperPosition()


if __name__ == "__main__":
    broker = PaperBroker(10000)
    broker.open(+1, 1000, 100.0)
    broker.close(101.5, "take_profit")
    print(broker.trades)
    print("Paper equity:", broker.equity)
