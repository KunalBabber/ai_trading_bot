"""
Generates a clear, visual explanatory infographic chart showing:
1. How the AI model analyzes 96 bars of sequential data
2. The exact moment and 4-step checklist to trigger a trade
3. The future trade execution with Stop Loss and Take Profit targets
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Seed for reproducibility
np.random.seed(42)

# Set high DPI and dark theme styling
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 9), facecolor="#0b0e14")

# Create grid: Panel 1 for Price Chart with Lookback & Future, Panel 2 for Flowchart/Checklist
gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# === PANEL 1: PRICE ACTION & FUTURE TRADE LIFECYCLE ===
# Simulate realistic price sequence: 96 past bars + 16 future bars = 112 bars total
n_past = 96
n_future = 16
n_total = n_past + n_future

# Generate random walk with slight upward trend in future
returns_past = np.random.normal(0.0001, 0.0012, n_past)
returns_future = np.random.normal(0.0010, 0.0010, n_future)
returns = np.concatenate([returns_past, returns_future])

base_price = 77500.0
price = base_price * np.exp(np.cumsum(returns))
x_bars = np.arange(n_total)

entry_bar = n_past - 1
entry_price = price[entry_bar]
stop_loss_price = entry_price * (1.0 - 0.008)  # -0.8%
take_profit_price = entry_price * (1.0 + 0.016)  # +1.6%

# Force price to hit take profit around bar 106 for visual clarity
price[entry_bar + 10:] = take_profit_price + np.random.normal(20, 10, len(price) - (entry_bar + 10))

# 1. Plot Past Lookback Area
ax1.axvspan(0, entry_bar, color="#1e293b", alpha=0.6, label="Past Lookback Context (8 Hours / 96 Candles)")
# 2. Plot Future Trade Area
ax1.axvspan(entry_bar, n_total - 1, color="#132f2e", alpha=0.5, label="Future Trade Window (1 Hour / 12 Candles)")

# Plot Price Curve
ax1.plot(x_bars[:n_past], price[:n_past], color="#38bdf8", linewidth=2.0, label="Past Price History")
ax1.plot(x_bars[n_past-1:], price[n_past-1:], color="#4ade80", linewidth=2.5, linestyle="--", label="Future Price Action (Trade Active)")

# Horizontal Target Lines
ax1.axhline(take_profit_price, color="#22c55e", linestyle="-.", linewidth=1.8, label=f"Take Profit Target: ${take_profit_price:,.0f} (+1.6%)")
ax1.axhline(entry_price, color="#94a3b8", linestyle=":", linewidth=1.2, label=f"Entry Price: ${entry_price:,.0f}")
ax1.axhline(stop_loss_price, color="#ef4444", linestyle="-.", linewidth=1.8, label=f"Stop Loss Target: ${stop_loss_price:,.0f} (-0.8%)")

# Vertical Decision Line
ax1.axvline(entry_bar, color="#fbbf24", linestyle="--", linewidth=2.2, label="Decision Point (T=0)")

# Markers
ax1.scatter([entry_bar], [entry_price], color="#22c55e", s=180, zorder=5, marker="^", edgecolors="white")
ax1.scatter([entry_bar + 10], [take_profit_price], color="#38bdf8", s=180, zorder=5, marker="*", edgecolors="white")

# Annotations on Price Chart
ax1.annotate(
    "1. WHAT THE AI READS:\n96 5m candles (RSI, ATR,\nEMAs, Volume, Momentum)",
    xy=(45, price[45]), xytext=(20, price.max() * 0.992),
    arrowprops=dict(facecolor="#38bdf8", shrink=0.08, width=1.5, headwidth=7),
    fontsize=10, color="#38bdf8", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9)
)

ax1.annotate(
    f"2. BUY SIGNAL TRIGGERED!\nAll 4 Conditions Met\nEntry: ${entry_price:,.0f}",
    xy=(entry_bar, entry_price), xytext=(entry_bar - 28, entry_price * 0.988),
    arrowprops=dict(facecolor="#22c55e", shrink=0.08, width=2, headwidth=8),
    fontsize=10, color="#22c55e", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#22c55e", alpha=0.9)
)

ax1.annotate(
    f"3. TAKE PROFIT HIT!\n+$160 Realized PnL\nAutomatically Closes",
    xy=(entry_bar + 10, take_profit_price), xytext=(entry_bar + 2, take_profit_price * 1.004),
    arrowprops=dict(facecolor="#4ade80", shrink=0.08, width=2, headwidth=8),
    fontsize=10, color="#4ade80", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#4ade80", alpha=0.9)
)

ax1.set_title("HOW THE AI BOT TAKES A TRADE (Lookback Context + Future Trade Lifecycle)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
ax1.set_xlabel("Time (5-Minute Bars)", fontsize=10, color="#94a3b8")
ax1.set_ylabel("Bitcoin Price (USD)", fontsize=10, color="#94a3b8")
ax1.legend(loc="lower left", fontsize=8.5, facecolor="#0f172a", edgecolor="#334155", framealpha=0.9)
ax1.grid(True, linestyle=":", alpha=0.25, color="#64748b")
ax1.set_facecolor("#0b0e14")


# === PANEL 2: THE 4-STEP DECISION CHECKLIST ===
ax2.set_facecolor("#0b0e14")
ax2.axis("off")

# Draw 4 visual condition cards
card_w = 0.21
card_h = 0.70
y_pos = 0.15

conditions = [
    ("1. AI Confidence", "P(Long) >= 58%", "Model estimates >58% odds\nof profitable upward move", "#38bdf8"),
    ("2. Expected Profit", "E(Return) >= +0.15%", "Predicted move covers\nDelta fees + slippage", "#818cf8"),
    ("3. 15m Trend", "Trend_15m > 0", "Fast EMA > Slow EMA\nShort-term trend is UP", "#34d399"),
    ("4. 1h Macro Trend", "Trend_1h > 0", "Fast EMA > Slow EMA\nMacro trend is UP", "#fbbf24"),
]

for i, (title, rule, desc, color) in enumerate(conditions):
    x_pos = 0.03 + i * 0.245
    # Card outline
    rect = patches.FancyBboxPatch(
        (x_pos, y_pos), card_w, card_h,
        boxstyle="round,pad=0.03",
        linewidth=1.8,
        edgecolor=color,
        facecolor="#151922"
    )
    ax2.add_patch(rect)
    
    # Text inside card
    ax2.text(x_pos + card_w/2, y_pos + 0.52, title, fontsize=11, fontweight="bold", color="#f8fafc", ha="center")
    ax2.text(x_pos + card_w/2, y_pos + 0.32, rule, fontsize=12, fontweight="heavy", color=color, ha="center")
    ax2.text(x_pos + card_w/2, y_pos + 0.12, desc, fontsize=8.5, color="#94a3b8", ha="center")

ax2.set_title("THE EXACT 4-STEP CHECKLIST: All 4 Must Be TRUE to Enter a Trade (Else Bot Stays FLAT/Wait)", fontsize=11, fontweight="bold", color="#fbbf24", pad=8)

# Save chart
out_dir = Path("artifacts")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "ai_trade_lifecycle_diagram.png"
plt.tight_layout()
plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)

print(f"Infographic successfully saved to: {out_path}")
