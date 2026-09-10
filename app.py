"""
Streamlit AI Trading Control Center for Delta Exchange
Real-time web dashboard for automated crypto perpetuals trading.
Provides dynamic coin selection, timeframe tuning, order sizing, leverage,
API credentials management, live Plotly candlestick charting, and AI metrics.
"""

import os
import time
from datetime import datetime, timezone
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml
from dotenv import load_dotenv

from bot_manager import bot_manager
from delta_client import DeltaClient
from live_features import LiveFeatureEngine
from train import GRUTradingModel
from timezone_utils import IST_TZ, TIMEZONE_MAP, get_now, format_now
from strategy import make_signal

@st.cache_resource
def get_preview_model():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load("artifacts/gru.pt", map_location="cpu")
    model = GRUTradingModel(16, cfg["model"]["hidden_size"], cfg["model"]["num_layers"], cfg["model"]["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    engine = LiveFeatureEngine(scaler_path="artifacts/scaler.joblib", sequence_length=96)
    return model, engine

# Load environment variables (.env locally, or Streamlit Cloud Secrets in production)
load_dotenv()
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="Delta AI Trading Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark UI aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0e14;
    }
    .stMetric {
        background: #151922;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #242b38;
    }
    .stMetric label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }
    .badge-buy {
        background-color: #1f6feb;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-sell {
        background-color: #da3633;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-flat {
        background-color: #30363d;
        color: #8b949e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .hud-card {
        background: #111726;
        border: 1px solid #1f293d;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .hud-card-pass {
        border-left: 4px solid #22c55e !important;
        background: #0d1e1c;
    }
    .hud-card-wait {
        border-left: 4px solid #475569 !important;
    }
    .hud-tag-pass {
        background: #065f46;
        color: #34d399;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .hud-tag-wait {
        background: #1e293b;
        color: #94a3b8;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .terminal-container {
        background-color: #050811;
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    }
    .terminal-header {
        background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
        padding: 9px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .terminal-dots {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .terminal-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-red { background-color: #ef4444; }
    .dot-yellow { background-color: #f59e0b; }
    .dot-green { background-color: #22c55e; }
    .terminal-body {
        padding: 12px 16px;
        height: 340px;
        overflow-y: auto;
        font-size: 0.77rem;
        line-height: 1.6;
        white-space: pre-wrap;
        background-color: #030712;
        color: #e2e8f0;
    }
    .terminal-body::-webkit-scrollbar {
        width: 6px;
    }
    .terminal-body::-webkit-scrollbar-track {
        background: #06090f;
    }
    .terminal-body::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }
    .terminal-body::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    .terminal-cursor {
        display: inline-block;
        width: 8px;
        height: 14px;
        background-color: #22c55e;
        vertical-align: -2px;
        animation: blink 1s step-end infinite;
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        box-shadow: 0 0 8px #22c55e;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
        vertical-align: middle;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def format_terminal_logs(logs, active_tz=IST_TZ):
    if not logs:
        now_str = datetime.now(active_tz).strftime("%I:%M:%S %p")
        lines = [
            '<span style="color: #38bdf8; font-weight: 700;">┌──(delta-ai-terminal)─[~/engine]</span>',
            '<span style="color: #38bdf8; font-weight: 700;">└─$</span> <span style="color: #f0f6fc;">python live_trader.py --daemon</span>',
            f'<span style="color: #64748b;">[{now_str}]</span> <span style="color: #38bdf8; font-weight: 600;">[SYSTEM]</span> Delta Exchange AI Terminal v2.4 initialized.',
            f'<span style="color: #64748b;">[{now_str}]</span> <span style="color: #a855f7; font-weight: 600;">[MODEL]</span> GRU Neural Network (96-step sequence, 16 features) loaded from artifacts/gru.pt',
            f'<span style="color: #64748b;">[{now_str}]</span> <span style="color: #22c55e; font-weight: 600;">[TRANSPORT]</span> REST/WebSocket client ready. Multi-timeframe pipeline (5m, 15m, 1h) active.',
            f'<span style="color: #64748b;">[{now_str}]</span> <span style="color: #fbbf24; font-weight: 600;">[STANDBY]</span> Engine ready. Click <b style="color: #4ade80;">[▶ Start Bot]</b> in left sidebar to begin live streaming scans.',
        ]
        return "<br>".join(lines)

    formatted = []
    for raw_line in logs:
        line = str(raw_line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ts_part = ""
        body_part = line
        if line.startswith("[") and "]" in line[:20]:
            idx = line.find("]") + 1
            ts_part = f'<span style="color: #64748b;">{line[:idx]}</span> '
            body_part = line[idx:].strip()

        # Syntax color patterns
        if "ORDER FILLED" in body_part or "EXECUTION" in body_part or "BRACKET" in body_part:
            styled = f'<span style="color: #fbbf24; font-weight: 700;">{body_part}</span>'
        elif "BUY" in body_part or "TAKE PROFIT" in body_part or "PROFIT" in body_part or "[OK]" in body_part or "LONG" in body_part:
            styled = f'<span style="color: #4ade80; font-weight: 600;">{body_part}</span>'
        elif "SELL" in body_part or "STOP LOSS" in body_part or "REVERSAL" in body_part or "PANIC" in body_part or "[ERROR]" in body_part or "SHORT" in body_part:
            styled = f'<span style="color: #f87171; font-weight: 600;">{body_part}</span>'
        elif "SCAN #" in body_part:
            styled = f'<span style="color: #38bdf8; font-weight: 600;">{body_part}</span>'
        elif "HEARTBEAT" in body_part or "[PULSE]" in body_part:
            styled = f'<span style="color: #818cf8;">{body_part}</span>'
        elif "[WARN]" in body_part or "Standing by" in body_part or "Waiting" in body_part or "Setup" in body_part:
            styled = f'<span style="color: #94a3b8;">{body_part}</span>'
        else:
            styled = f'<span style="color: #cbd5e1;">{body_part}</span>'

        formatted.append(f"{ts_part}{styled}")

    return "<br>".join(formatted)

def render_smart_terminal_html(symbol: str, resolution: str, is_running: bool, status_tag: str, body_content: str, cmd_prompt: str) -> str:
    """Renders authentic terminal with smart auto-scroll that pauses when user scrolls up."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        overflow: hidden;
      }}
      .terminal-container {{
        background-color: #050811;
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        display: flex;
        flex-direction: column;
        height: 375px;
        position: relative;
      }}
      .terminal-header {{
        background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
        padding: 9px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        font-size: 0.8rem;
        color: #94a3b8;
        flex-shrink: 0;
      }}
      .terminal-dots {{
        display: flex;
        gap: 6px;
        align-items: center;
      }}
      .terminal-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
      }}
      .dot-red {{ background-color: #ef4444; }}
      .dot-yellow {{ background-color: #f59e0b; }}
      .dot-green {{ background-color: #22c55e; }}
      .terminal-body {{
        flex: 1;
        padding: 12px 16px;
        overflow-y: auto;
        font-size: 0.77rem;
        line-height: 1.6;
        white-space: pre-wrap;
        background-color: #030712;
        color: #e2e8f0;
        scroll-behavior: smooth;
      }}
      .terminal-body::-webkit-scrollbar {{
        width: 6px;
      }}
      .terminal-body::-webkit-scrollbar-track {{
        background: #06090f;
      }}
      .terminal-body::-webkit-scrollbar-thumb {{
        background: #1e293b;
        border-radius: 3px;
      }}
      .terminal-body::-webkit-scrollbar-thumb:hover {{
        background: #334155;
      }}
      .terminal-cursor {{
        display: inline-block;
        width: 8px;
        height: 14px;
        background-color: #22c55e;
        vertical-align: -2px;
        animation: blink 1s step-end infinite;
      }}
      .scroll-pill {{
        position: absolute;
        bottom: 14px;
        right: 18px;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        cursor: pointer;
        display: none;
        align-items: center;
        gap: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.7), 0 0 10px rgba(2, 132, 199, 0.4);
        border: 1px solid #38bdf8;
        transition: all 0.2s ease;
        z-index: 10;
      }}
      .scroll-pill:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
      }}
      @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0; }}
      }}
    </style>
    </head>
    <body>
    <div class="terminal-container">
        <div class="terminal-header">
            <div class="terminal-dots">
                <span class="terminal-dot dot-red"></span>
                <span class="terminal-dot dot-yellow"></span>
                <span class="terminal-dot dot-green"></span>
                <span style="margin-left: 8px; font-weight: 600; color: #94a3b8; font-size: 0.78rem;">
                    delta-trader-daemon.sh (bash) &mdash; {symbol} [{resolution}]
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; font-size: 0.75rem;">
                <span id="scroll-status-badge" style="font-weight: 600; color: #4ade80;">🟢 AUTO-SCROLL</span>
                <span style="color: #475569;">|</span>
                {status_tag}
            </div>
        </div>
        <div class="terminal-body" id="delta-terminal-body">
            {body_content}
            {cmd_prompt}
        </div>
        <div class="scroll-pill" id="scroll-pill" onclick="jumpToBottom()">
            <span>⬇ Jump to Latest (Resume Auto-scroll)</span>
        </div>
    </div>
    <script>
      const term = document.getElementById('delta-terminal-body');
      const pill = document.getElementById('scroll-pill');
      const badge = document.getElementById('scroll-status-badge');

      function updateScrollState() {{
          const distance = term.scrollHeight - term.clientHeight - term.scrollTop;
          if (distance > 45) {{
              sessionStorage.setItem('delta_term_paused', 'true');
              sessionStorage.setItem('delta_term_scroll_top', term.scrollTop);
              if (pill) pill.style.display = 'flex';
              if (badge) {{
                  badge.innerHTML = '⏸ SCROLL PAUSED';
                  badge.style.color = '#fbbf24';
              }}
          }} else {{
              sessionStorage.setItem('delta_term_paused', 'false');
              sessionStorage.removeItem('delta_term_scroll_top');
              if (pill) pill.style.display = 'none';
              if (badge) {{
                  badge.innerHTML = '🟢 AUTO-SCROLL';
                  badge.style.color = '#4ade80';
              }}
          }}
      }}

      term.addEventListener('scroll', updateScrollState);

      function jumpToBottom() {{
          sessionStorage.setItem('delta_term_paused', 'false');
          sessionStorage.removeItem('delta_term_scroll_top');
          term.scrollTo({{ top: term.scrollHeight, behavior: 'smooth' }});
          if (pill) pill.style.display = 'none';
          if (badge) {{
              badge.innerHTML = '🟢 AUTO-SCROLL';
              badge.style.color = '#4ade80';
          }}
      }}

      window.addEventListener('DOMContentLoaded', () => {{
          const isPaused = sessionStorage.getItem('delta_term_paused') === 'true';
          const savedTop = sessionStorage.getItem('delta_term_scroll_top');

          if (isPaused && savedTop !== null) {{
              term.scrollTop = parseInt(savedTop, 10);
              if (pill) pill.style.display = 'flex';
              if (badge) {{
                  badge.innerHTML = '⏸ SCROLL PAUSED';
                  badge.style.color = '#fbbf24';
              }}
          }} else {{
              term.scrollTop = term.scrollHeight;
              if (pill) pill.style.display = 'none';
              if (badge) {{
                  badge.innerHTML = '🟢 AUTO-SCROLL';
                  badge.style.color = '#4ade80';
              }}
          }}
      }});
    </script>
    </body>
    </html>
    """

# Fetch available live perpetuals from Delta API
@st.cache_data(ttl=300)
def get_delta_symbols():
    try:
        c = DeltaClient(timeout=3)
        prods = c.get_products()
        live_perps = [
            p["symbol"] for p in prods
            if p.get("contract_type") == "perpetual_futures" and p.get("state") == "live"
        ]
        priority = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD"]
        ordered = [s for s in priority if s in live_perps] + [s for s in live_perps if s not in priority]
        return ordered if ordered else ["BTCUSD", "ETHUSD", "SOLUSD"]
    except Exception:
        return ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD"]

# Cached candle fetch for preview when the bot engine is not actively polling
@st.cache_data(ttl=15)
def get_preview_candles(symbol: str, resolution: str) -> pd.DataFrame:
    try:
        c = DeltaClient(timeout=4)
        return c.fetch_candles(symbol, resolution, limit=160)
    except Exception:
        return pd.DataFrame()


# Factory Default Settings & Persistence Configuration
SETTINGS_FILE = "user_settings.json"

DEFAULT_SETTINGS = {
    "cfg_mode": "Simulated Dry-Run (Safe)",
    "cfg_url": "https://cdn-ind.testnet.deltaex.org",
    "cfg_symbol": "BTCUSD",
    "cfg_timeframe": "5m",
    "cfg_timezone": "IST (India - UTC+5:30)",
    "cfg_leverage": 10,
    "cfg_sizing_mode": "Risk Budget %",
    "cfg_fixed_contracts": 1,
    "cfg_risk_pct": 0.3,
    "cfg_prob_threshold": 0.58,
    "cfg_min_return_hurdle": 0.15,
    "cfg_tp_target": 1.6,
}

def load_persisted_settings():
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                settings.update(saved)
        except Exception:
            pass
    return settings

def save_persisted_settings():
    current = {}
    for k in DEFAULT_SETTINGS.keys():
        if k in st.session_state:
            current[k] = st.session_state[k]
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass

# Initialize session state from persisted settings (runs once per browser reload)
if "_settings_initialized" not in st.session_state:
    persisted = load_persisted_settings()
    for _k, _v in persisted.items():
        st.session_state[_k] = _v
    st.session_state["_settings_initialized"] = True

def reset_settings_callback():
    for k, v in DEFAULT_SETTINGS.items():
        st.session_state[k] = v
    try:
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
    except Exception:
        pass
    st.session_state["_reset_toast"] = True


# === SIDEBAR CONTROLS ===
with st.sidebar:
    st.markdown("## ⚡ **DELTA AI BOT**")
    st.caption("GRU Sequence AI • Perpetual Futures")

    if st.session_state.get("_reset_toast", False):
        st.toast("Settings restored to factory defaults!", icon="🔄")
        st.session_state["_reset_toast"] = False

    st.divider()

    # 1. Trading Mode
    st.markdown("### 🛡️ **Execution Mode**")
    mode_choice = st.radio(
        "Mode Selection",
        ["Simulated Dry-Run (Safe)", "Live Real Money (⚠️)"],
        key="cfg_mode",
        help="Dry-run simulates order execution against live candles without risking capital.",
    )
    is_dry_run = "Dry-Run" in mode_choice

    # 2. Exchange Credentials
    with st.expander("🔑 **Delta API Credentials**", expanded=False):
        base_urls = [
            "https://cdn-ind.testnet.deltaex.org",
            "https://cdn-ind.deltaex.org",
            "https://testnet-api.delta.exchange",
            "https://api.delta.exchange",
        ]
        if st.session_state.get("cfg_url") not in base_urls:
            st.session_state["cfg_url"] = base_urls[0]

        selected_url = st.selectbox("API Base URL", base_urls, key="cfg_url")

        # Outbound IP detector (On-demand with fast fallback)
        def fetch_server_ip():
            for u in ["https://api4.ipify.org", "https://checkip.amazonaws.com", "https://ifconfig.co/ip"]:
                try:
                    r = requests.get(u, timeout=2.0)
                    if r.status_code == 200 and r.text.strip():
                        return r.text.strip()
                except Exception:
                    continue
            return "Unable to detect (Click [🔌 Test API] below)"

        if "detected_server_ip" not in st.session_state:
            st.session_state["detected_server_ip"] = None

        col_ip_lbl, col_ip_btn = st.columns([2.8, 1.2])
        with col_ip_lbl:
            cur_ip = st.session_state["detected_server_ip"]
            if cur_ip:
                st.info(f"🌐 **Server IP:** `{cur_ip}`\n\n*(Add to Delta API Key whitelist)*")
            else:
                st.caption("🌐 **Server Outbound IP:** Click 'Detect' or 'Test API'.")
        with col_ip_btn:
            if st.button("🔍 Detect", key="btn_detect_ip", use_container_width=True, help="Detect Streamlit Cloud server outbound IP"):
                with st.spinner("Checking IP..."):
                    st.session_state["detected_server_ip"] = fetch_server_ip()
                st.rerun()


        api_key_input = st.text_input(
            "API Key",
            value=os.getenv("DELTA_API_KEY", ""),
            type="password",
            help="Delta Exchange API Key",
        )
        api_secret_input = st.text_input(
            "API Secret",
            value=os.getenv("DELTA_API_SECRET", ""),
            type="password",
            help="Delta Exchange API Secret",
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("🔌 Test API", use_container_width=True):
                test_res = bot_manager.test_connection(api_key_input, api_secret_input, selected_url)
                if test_res.get("success"):
                    st.success(test_res["message"])
                else:
                    st.error(test_res.get("error"))
                    if test_res.get("ip_needed"):
                        st.session_state["detected_server_ip"] = test_res["ip_needed"]
                        st.caption("📋 **Copy this IP to your Delta whitelist:**")
                        st.code(test_res["ip_needed"], language="text")

        with col_c2:
            if st.button("💾 Save .env", use_container_width=True):
                bot_manager.save_env_credentials(api_key_input, api_secret_input, selected_url, is_dry_run)
                st.toast("Credentials saved to .env", icon="✅")

        with col_c3:
            if st.button("🔄 Balance", use_container_width=True):
                bal_res = bot_manager.update_balance(api_key_input, api_secret_input, selected_url)
                if bal_res.get("success"):
                    st.toast(f"Wallet Balance: ${bal_res.get('equity', 0.0):.2f}", icon="💰")
                else:
                    st.error(bal_res.get("error"))

    # 3. Pair & Timeframe Selection
    st.markdown("### 📊 **Market Selection**")
    available_symbols = get_delta_symbols()
    if st.session_state.get("cfg_symbol") not in available_symbols:
        st.session_state["cfg_symbol"] = available_symbols[0]

    symbol_selected = st.selectbox("Trading Pair", available_symbols, key="cfg_symbol")
    timeframe_selected = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], key="cfg_timeframe")

    # Timezone Selector (Defaults to Indian Standard Time IST for Delta Exchange)
    tz_names = list(TIMEZONE_MAP.keys())
    if st.session_state.get("cfg_timezone") not in tz_names:
        st.session_state["cfg_timezone"] = tz_names[0]

    selected_tz_name = st.selectbox(
        "Display Timezone",
        tz_names,
        key="cfg_timezone",
        help="Local clock timezone for charts, header, and terminal logs. Defaults to IST (+5:30).",
    )
    active_tz, active_tz_abbr = TIMEZONE_MAP[selected_tz_name]
    bot_manager.set_timezone(active_tz, active_tz_abbr)

    # 4. Sizing & Leverage
    st.markdown("### ⚙️ **Position & Leverage**")
    leverage_val = st.slider("Leverage", min_value=1, max_value=50, step=1, format="%dx", key="cfg_leverage")
    sizing_mode = st.radio("Order Sizing Mode", ["Risk Budget %", "Fixed Contracts"], key="cfg_sizing_mode")

    if sizing_mode == "Fixed Contracts":
        fixed_contracts = st.number_input("Contracts per Trade", min_value=1, max_value=1000, step=1, key="cfg_fixed_contracts")
        risk_per_trade = 0.003
    else:
        fixed_contracts = 1
        risk_pct = st.slider("Risk per Trade (% Equity)", min_value=0.1, max_value=2.0, step=0.1, key="cfg_risk_pct")
        risk_per_trade = risk_pct / 100.0

    # 5. AI Conviction & Targets
    with st.expander("🎯 **AI Conviction & Strategy**", expanded=False):
        prob_threshold = st.slider("Min Probability Cutoff", min_value=0.25, max_value=0.85, step=0.01, format="%.2f", key="cfg_prob_threshold")
        min_return_hurdle_pct = st.slider("Min Expected Return (%)", min_value=0.01, max_value=0.50, step=0.01, format="%.2f%%", key="cfg_min_return_hurdle")
        min_return_hurdle = min_return_hurdle_pct / 100.0
        take_profit_target_pct = st.slider("Take Profit (%)", min_value=0.3, max_value=5.0, step=0.1, format="%.1f%%", key="cfg_tp_target")
        take_profit_target = take_profit_target_pct / 100.0

    # Automatically persist settings to disk whenever altered
    save_persisted_settings()

    # Dynamically propagate updated strategy parameters to bot_manager if running
    bot_manager.update_config({
        "long_probability": prob_threshold,
        "short_probability": prob_threshold,
        "min_expected_return": min_return_hurdle,
        "take_profit": take_profit_target,
        "leverage": leverage_val,
        "sizing_mode": "fixed" if sizing_mode == "Fixed Contracts" else "risk_budget",
        "fixed_contracts": fixed_contracts,
        "risk_per_trade": risk_per_trade,
    })

    st.divider()

    # 6. Bot Action Buttons
    state_snapshot = bot_manager.get_state()
    is_running = state_snapshot["is_running"]

    if not is_running:
        if st.button("▶ Start Bot", type="primary", use_container_width=True):
            run_cfg = {
                "symbol": symbol_selected,
                "resolution": timeframe_selected,
                "leverage": leverage_val,
                "dry_run": is_dry_run,
                "api_key": api_key_input,
                "api_secret": api_secret_input,
                "base_url": selected_url,
                "sizing_mode": "fixed" if sizing_mode == "Fixed Contracts" else "risk_budget",
                "fixed_contracts": fixed_contracts,
                "risk_per_trade": risk_per_trade,
                "long_probability": prob_threshold,
                "short_probability": prob_threshold,
                "min_expected_return": min_return_hurdle,
                "take_profit": take_profit_target,
                "poll_interval": 8,
            }
            bot_manager.start_bot(run_cfg)
            st.rerun()
    else:
        if st.button("⏹ Stop Bot", type="secondary", use_container_width=True):
            bot_manager.stop_bot()
            st.rerun()

    if st.button("🚨 Emergency Panic Close All", use_container_width=True):
        panic_res = bot_manager.panic_close()
        st.toast(f"Panic close executed: {panic_res}", icon="⚠️")
        st.rerun()

    st.divider()

    # 7. Reset to Default Settings
    st.button(
        "🔄 Reset to Default Settings",
        on_click=reset_settings_callback,
        use_container_width=True,
        help="Instantly restores all trading pairs, timeframe, leverage, sizing, and AI strategy thresholds back to default values.",
    )


# === REAL-TIME DASHBOARD FRAGMENT ===
@st.fragment(run_every="3s")
def render_dashboard(
    symbol: str,
    resolution: str,
    active_tz=IST_TZ,
    active_tz_abbr: str = "IST",
    prob_threshold: float = 0.58,
    min_return_hurdle: float = 0.0015,
    take_profit_target: float = 0.016,
):
    # Ensure thresholds dynamically match current session state if modified in sidebar
    if "cfg_prob_threshold" in st.session_state:
        prob_threshold = float(st.session_state["cfg_prob_threshold"])
    if "cfg_min_return_hurdle" in st.session_state:
        min_return_hurdle = float(st.session_state["cfg_min_return_hurdle"]) / 100.0
    if "cfg_tp_target" in st.session_state:
        take_profit_target = float(st.session_state["cfg_tp_target"]) / 100.0

    state = bot_manager.get_state()
    is_running = state["is_running"]
    current_price = state["current_price"]
    candles = state["candle_df"]
    metrics = state["metrics"]
    predictions = state["predictions"]
    position = state["active_position"]
    trade_history = state["trade_history"]
    logs = state["recent_logs"]
    specs = state["product_specs"]

    # Fallback to fetch candles & evaluate live AI preview if bot not running yet
    if candles.empty:
        candles = get_preview_candles(symbol, resolution)
        if not candles.empty:
            current_price = float(candles["close"].iloc[-1])


    if (not is_running or not metrics) and len(candles) >= 96:
        try:
            model_prev, engine_prev = get_preview_model()
            tensor, comp_metrics = engine_prev.process_candles(candles)
            metrics = comp_metrics
            with torch.no_grad():
                ret_t, l_t, s_t = model_prev(tensor)
            p_long = float(torch.sigmoid(l_t).item())
            p_short = float(torch.sigmoid(s_t).item())
            exp_ret = float(ret_t.item())
            sig_prev = make_signal(
                predicted_return=exp_ret,
                p_long=p_long,
                p_short=p_short,
                trend_15m=metrics.get("trend_15m", 0.0),
                trend_1h=metrics.get("trend_1h", 0.0),
                atr_pct=metrics.get("atr_pct", 0.005),
                long_probability=prob_threshold,
                short_probability=prob_threshold,
                min_expected_return=min_return_hurdle,
            )
            predictions = {
                "p_long": p_long,
                "p_short": p_short,
                "expected_return": exp_ret,
                "signal_side": sig_prev.side,
            }
        except Exception:
            pass

    # Header Ribbon & Overall P&L Calculations
    total_realized_pnl = sum(float(t.get("pnl", 0.0)) for t in trade_history)
    unrealized_pnl = float(position.get("unrealized_pnl", 0.0))
    net_overall_pnl = total_realized_pnl + unrealized_pnl

    win_trades = [t for t in trade_history if float(t.get("pnl", 0.0)) > 0]
    loss_trades = [t for t in trade_history if float(t.get("pnl", 0.0)) < 0]
    total_trades = len(trade_history)
    win_rate = (len(win_trades) / total_trades * 100.0) if total_trades > 0 else 0.0

    local_time_str = datetime.now(active_tz).strftime("%I:%M:%S %p")

    col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])
    with col_h1:
        st.markdown(f"## **{symbol}** • {resolution} Timeframe")
        status_color = "🟢 RUNNING" if is_running else "⚪ STOPPED"
        mode_text = "SIMULATED DRY-RUN" if state["mode"] == "dry_run" else "LIVE REAL-MONEY TRADING"
        st.caption(f"Status: **{status_color}** • Engine: **{mode_text}** • System Time ({active_tz_abbr}): **{local_time_str}**")
    with col_h2:
        if current_price > 0:
            first_close = float(candles["close"].iloc[0]) if len(candles) else current_price
            price_change = ((current_price / first_close) - 1.0) * 100.0
            st.metric(
                label="Mark Price",
                value=f"${current_price:,.2f}",
                delta=f"{price_change:+.2f}%",
            )
    with col_h3:
        is_live_mode = state["mode"] == "live"
        wallet_eq = state.get("wallet_equity", 0.0)
        sim_eq = state.get("simulated_equity", 10000.0)

        if is_live_mode:
            bal_val = wallet_eq
            bal_label = "Live Wallet Balance"
        elif wallet_eq > 0:
            bal_val = wallet_eq
            bal_label = "Delta Wallet (Live)"
        else:
            bal_val = sim_eq
            bal_label = "Current Balance (Sim)"

        st.metric(
            label=f"💰 {bal_label}",
            value=f"${bal_val:,.2f}",
            delta=f"{position['unrealized_pnl']:+.2f} USD PnL" if position["side"] != 0 else None,
        )
    with col_h4:
        st.metric(
            label="📊 Total Net P&L",
            value=f"${net_overall_pnl:+,.2f}",
            delta=f"{len(win_trades)}W / {len(loss_trades)}L ({win_rate:.0f}% Win)" if total_trades > 0 else "0 Trades",
        )

    st.markdown("---")

    # Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    p_long = predictions.get("p_long", 0.5)
    p_short = predictions.get("p_short", 0.5)
    exp_ret = predictions.get("expected_return", 0.0)

    t15 = metrics.get("trend_15m", 0.0)
    t1h = metrics.get("trend_1h", 0.0)
    trend_15_str = "🟢 Bullish" if t15 > 0 else "🔴 Bearish"
    trend_1h_str = "🟢 Bullish" if t1h > 0 else "🔴 Bearish"

    # Evaluate dynamic signal using current thresholds
    dynamic_sig = make_signal(
        predicted_return=exp_ret,
        p_long=p_long,
        p_short=p_short,
        trend_15m=t15,
        trend_1h=t1h,
        atr_pct=metrics.get("atr_pct", 0.005),
        long_probability=prob_threshold,
        short_probability=prob_threshold,
        min_expected_return=min_return_hurdle,
    )
    sig_side = dynamic_sig.side
    sig_label = "BUY (+1)" if sig_side == 1 else ("SELL (-1)" if sig_side == -1 else "FLAT (0)")

    with m1:
        st.metric("AI Signal", sig_label)
    with m2:
        st.metric("P(Long)", f"{p_long * 100:.1f}%")
    with m3:
        st.metric("P(Short)", f"{p_short * 100:.1f}%")
    with m4:
        st.metric("Expected Return", f"{exp_ret * 100:+.2f}%")
    with m5:
        st.metric("15m Trend", trend_15_str)
    with m6:
        st.metric("1h Macro Trend", trend_1h_str)

    with st.expander("🎓 **Reference Infographic: Static Blueprint & Strategy Overview**", expanded=False):
        if os.path.exists("artifacts/ai_trade_lifecycle_diagram.png"):
            st.image("artifacts/ai_trade_lifecycle_diagram.png", use_container_width=True)
        st.markdown(
            """
            ### **How the Bot Works in 4 Simple Steps:**
            1. **Lookback Scan (Past 8 Hours):** Every cycle, the bot reads the last **96 candles** (5m bars) and computes 16 multi-resolution indicators.
            2. **GRU AI Prediction:** The neural network forecasts `P(Long)`, `P(Short)`, and `Expected Return` for the next 12 bars (1 hour).
            3. **4-Condition Checklist:** The bot checks the 4 rules below. If ANY rule fails, it stays FLAT/WAIT to protect capital.
            4. **Bracket Targets:** When triggered, automatic Stop Loss and Take Profit brackets are placed.
            """
        )

    # === LIVE AI DECISION ENGINE & 4-STEP CHECKLIST HUD ===
    c1_pass = (p_long >= prob_threshold) or (p_short >= prob_threshold)
    c2_pass = abs(exp_ret) >= min_return_hurdle
    c3_pass = (t15 > 0 if p_long >= p_short else t15 < 0)
    c4_pass = (t1h > 0 if p_long >= p_short else t1h < 0)
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    passed_count = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    best_odds = max(p_long, p_short) * 100

    st.markdown("### 🎓 **Live AI Decision Engine (Real-Time 4-Step Checklist)**")
    c_hud1, c_hud2, c_hud3, c_hud4 = st.columns(4)

    with c_hud1:
        cls1 = "hud-card-pass" if c1_pass else "hud-card-wait"
        tag1 = '<span class="hud-tag-pass">🟢 PASSED</span>' if c1_pass else '<span class="hud-tag-wait">❌ WAITING</span>'
        st.markdown(
            f"""
            <div class="hud-card {cls1}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #f0f6fc;">1. AI Confidence</span>
                    {tag1}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#38bdf8' if c1_pass else '#94a3b8'};">
                    {best_odds:.1f}% <span style="font-size: 0.75rem; color: #64748b;">(Need &ge; {prob_threshold * 100:.0f}%)</span>
                </div>
                <div style="font-size: 0.76rem; color: #8b949e; margin-top: 4px;">
                    P(L): {p_long*100:.1f}% | P(S): {p_short*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_hud2:
        cls2 = "hud-card-pass" if c2_pass else "hud-card-wait"
        tag2 = '<span class="hud-tag-pass">🟢 PASSED</span>' if c2_pass else '<span class="hud-tag-wait">❌ WAITING</span>'
        st.markdown(
            f"""
            <div class="hud-card {cls2}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #f0f6fc;">2. Profit Hurdle</span>
                    {tag2}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#38bdf8' if c2_pass else '#94a3b8'};">
                    {exp_ret*100:+.2f}% <span style="font-size: 0.75rem; color: #64748b;">(Need &ge; +{min_return_hurdle * 100:.2f}%)</span>
                </div>
                <div style="font-size: 0.76rem; color: #8b949e; margin-top: 4px;">
                    Predicted move covers fee + slippage
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_hud3:
        cls3 = "hud-card-pass" if c3_pass else "hud-card-wait"
        tag3 = '<span class="hud-tag-pass">🟢 ALIGNED</span>' if c3_pass else '<span class="hud-tag-wait">❌ DISAGREES</span>'
        t15_name = "Bullish (+1)" if t15 > 0 else ("Bearish (-1)" if t15 < 0 else "Neutral (0)")
        st.markdown(
            f"""
            <div class="hud-card {cls3}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #f0f6fc;">3. 15m Fast Trend</span>
                    {tag3}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#34d399' if t15 > 0 else ('#ef5350' if t15 < 0 else '#94a3b8')};">
                    {t15_name}
                </div>
                <div style="font-size: 0.76rem; color: #8b949e; margin-top: 4px;">
                    EMA 8 vs EMA 21 short momentum
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_hud4:
        cls4 = "hud-card-pass" if c4_pass else "hud-card-wait"
        tag4 = '<span class="hud-tag-pass">🟢 ALIGNED</span>' if c4_pass else '<span class="hud-tag-wait">❌ DISAGREES</span>'
        t1h_name = "Bullish (+1)" if t1h > 0 else ("Bearish (-1)" if t1h < 0 else "Neutral (0)")
        st.markdown(
            f"""
            <div class="hud-card {cls4}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #f0f6fc;">4. 1h Macro Trend</span>
                    {tag4}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#fbbf24' if t1h > 0 else ('#ef5350' if t1h < 0 else '#94a3b8')};">
                    {t1h_name}
                </div>
                <div style="font-size: 0.76rem; color: #8b949e; margin-top: 4px;">
                    Macro direction filter
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Decision Summary Banner
    if all_pass:
        hud_banner = """
        <div style="background-color: #064e3b; border: 1px solid #10b981; border-radius: 8px; padding: 9px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="pulse-dot"></span>
                <strong style="color: #4ade80;">READY TO TRADE</strong>
                <span style="color: #6ee7b7; font-size: 0.85rem;">All 4 AI & Regime conditions satisfied! Entry order active.</span>
            </div>
            <span style="color: #a7f3d0; font-weight: 700; font-size: 0.82rem;">4 / 4 CONDITIONS MET</span>
        </div>
        """
    else:
        reasons_needed = []
        if not c1_pass: reasons_needed.append(f"AI Odds ({best_odds:.1f}% < {prob_threshold * 100:.0f}%)")
        if not c2_pass: reasons_needed.append(f"Profit Hurdle ({exp_ret*100:+.2f}% < +{min_return_hurdle * 100:.2f}%)")
        if not c3_pass: reasons_needed.append("15m Trend Alignment")
        if not c4_pass: reasons_needed.append("1h Macro Trend Alignment")
        reason_txt = ", ".join(reasons_needed)
        hud_banner = f"""
        <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 9px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #fbbf24; font-size: 1rem;">⏳</span>
                <strong style="color: #f0f6fc;">STANDBY / WAITING</strong>
                <span style="color: #94a3b8; font-size: 0.85rem;">Waiting for: <span style="color: #38bdf8; font-weight: 600;">{reason_txt}</span>. Bot stays FLAT to protect your capital.</span>
            </div>
            <span style="color: #64748b; font-weight: 600; font-size: 0.82rem;">{passed_count} / 4 CONDITIONS MET</span>
        </div>
        """
    st.markdown(hud_banner, unsafe_allow_html=True)

    # === LIVE CANDLESTICK CHART WITH 96-BAR LOOKBACK & FUTURE PREDICTION CONE ===
    if not candles.empty and len(candles) > 5:
        candles_disp = candles.tail(120).copy()

        # Convert timestamps to user's selected timezone (default: IST)
        candles_disp["timestamp_local"] = pd.to_datetime(candles_disp["timestamp"]).dt.tz_convert(active_tz).dt.tz_localize(None)

        # Calculate fast & slow EMAs for chart overlay
        candles_disp["ema9"] = candles_disp["close"].ewm(span=9, adjust=False).mean()
        candles_disp["ema21"] = candles_disp["close"].ewm(span=21, adjust=False).mean()

        latest_time = candles_disp["timestamp_local"].iloc[-1]
        latest_close = float(candles_disp["close"].iloc[-1])

        # Dynamic timeframe step delta for future projection
        if resolution == "1m":
            step_delta = pd.Timedelta(minutes=1)
        elif resolution == "15m":
            step_delta = pd.Timedelta(minutes=15)
        elif resolution == "1h":
            step_delta = pd.Timedelta(hours=1)
        else:
            step_delta = pd.Timedelta(minutes=5)

        future_steps = 12
        future_times = [latest_time + step_delta * i for i in range(future_steps + 1)]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f"Price Action, AI Lookback & Future Target Horizons (System Local Time)", "Volume"),
            row_width=[0.2, 0.8],
        )

        # 1. Past AI Lookback Window Shading (96 Bars / Input Tensor)
        lookback_bars = min(96, len(candles_disp))
        lookback_start = candles_disp["timestamp_local"].iloc[-lookback_bars]

        fig.add_vrect(
            x0=lookback_start,
            x1=latest_time,
            fillcolor="#1e293b",
            opacity=0.30,
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"🧠 AI Lookback Window ({lookback_bars} Bars)",
            annotation_position="top left",
            annotation_font_color="#38bdf8",
            annotation_font_size=10,
            row=1,
            col=1,
        )

        # 2. Future Forecast Window Shading (12 Bars / 1 Hour)
        fig.add_vrect(
            x0=latest_time,
            x1=future_times[-1],
            fillcolor="#132f2e" if exp_ret >= 0 else "#2d1619",
            opacity=0.35,
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="#4ade80" if exp_ret >= 0 else "#f87171",
            annotation_text="🔮 AI Future Forecast Horizon (+12 Bars)",
            annotation_position="top left",
            annotation_font_color="#4ade80" if exp_ret >= 0 else "#f87171",
            annotation_font_size=10,
            row=1,
            col=1,
        )

        # 3. Decision Point Marker Line at T=0
        fig.add_vline(
            x=latest_time,
            line_width=2.2,
            line_dash="dash",
            line_color="#fbbf24",
            row=1,
            col=1,
        )

        # Candlestick Trace
        fig.add_trace(
            go.Candlestick(
                x=candles_disp["timestamp_local"],
                open=candles_disp["open"],
                high=candles_disp["high"],
                low=candles_disp["low"],
                close=candles_disp["close"],
                name="OHLCV",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        # EMA Lines
        fig.add_trace(
            go.Scatter(
                x=candles_disp["timestamp_local"],
                y=candles_disp["ema9"],
                line=dict(color="#f39c12", width=1.2),
                name="EMA 9 (Fast)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=candles_disp["timestamp_local"],
                y=candles_disp["ema21"],
                line=dict(color="#3498db", width=1.2),
                name="EMA 21 (Slow)",
            ),
            row=1,
            col=1,
        )

        # 4. AI Trajectory Path & Volatility Confidence Cone
        atr_pct = float(metrics.get("atr_pct", 0.005))
        upper_cone = [
            latest_close * (1.0 + exp_ret * (i / future_steps) + atr_pct * 1.5 * np.sqrt(i / future_steps))
            for i in range(future_steps + 1)
        ]
        lower_cone = [
            latest_close * (1.0 + exp_ret * (i / future_steps) - atr_pct * 1.5 * np.sqrt(i / future_steps))
            for i in range(future_steps + 1)
        ]
        traj_path = [
            latest_close * (1.0 + exp_ret * (i / future_steps))
            for i in range(future_steps + 1)
        ]

        # Upper bound of cone (invisible line)
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=upper_cone,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        # Lower bound with fill to upper bound
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=lower_cone,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.12)",
                name="AI Prediction Cone",
            ),
            row=1,
            col=1,
        )
        # Trajectory forecast line
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=traj_path,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5, dash="dot"),
                marker=dict(size=4, color="#38bdf8"),
                name=f"AI Forecast Path ({exp_ret*100:+.2f}%)",
            ),
            row=1,
            col=1,
        )

        # 5. Position Entry, Stop Loss & Take Profit Overlays
        if position["side"] != 0:
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[position["entry_price"], position["entry_price"]],
                    mode="lines",
                    line=dict(color="#3498db", width=1.8, dash="solid"),
                    name=f"Entry: ${position['entry_price']:.1f}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[position["take_profit"], position["take_profit"]],
                    mode="lines",
                    line=dict(color="#2ecc71", width=2.0, dash="dashdot"),
                    name=f"Take Profit: ${position['take_profit']:.1f}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[position["stop_loss"], position["stop_loss"]],
                    mode="lines",
                    line=dict(color="#e74c3c", width=2.0, dash="dashdot"),
                    name=f"Stop Loss: ${position['stop_loss']:.1f}",
                ),
                row=1,
                col=1,
            )
        else:
            tp_pending = latest_close * (1.0 + take_profit_target)
            sl_pending = latest_close * (1.0 - 0.008)
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[tp_pending, tp_pending],
                    mode="lines",
                    line=dict(color="#2ecc71", width=1.8, dash="dashdot"),
                    name=f"Pending TP: ${tp_pending:,.1f} (+{take_profit_target * 100:.1f}%)",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[latest_close, latest_close],
                    mode="lines",
                    line=dict(color="#94a3b8", width=1.2, dash="dot"),
                    name=f"Entry Level (Now): ${latest_close:,.1f}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[latest_time, future_times[-1]],
                    y=[sl_pending, sl_pending],
                    mode="lines",
                    line=dict(color="#ef4444", width=1.8, dash="dashdot"),
                    name=f"Pending SL: ${sl_pending:,.1f} (-0.8%)",
                ),
                row=1,
                col=1,
            )

        # Volume Subplot
        colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(candles_disp["open"], candles_disp["close"])]
        fig.add_trace(
            go.Bar(
                x=candles_disp["timestamp_local"],
                y=candles_disp["volume"],
                marker_color=colors,
                name="Volume",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=580,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(gridcolor="#1e2530")
        fig.update_xaxes(gridcolor="#1e2530")

        st.plotly_chart(fig, use_container_width=True)

    # Active Position & Activity Feed
    c_pos, c_logs = st.columns([1, 1])

    with c_pos:
        st.markdown("### 💼 **Active Position & Equity**")
        pos_side = position["side"]
        pos_str = "🟢 LONG" if pos_side == 1 else ("🔴 SHORT" if pos_side == -1 else "⚪ FLAT (No Open Position)")

        pnl_val = position["unrealized_pnl"]
        pnl_pct = position["unrealized_pnl_pct"]
        pnl_color = "#2ecc71" if pnl_val >= 0 else "#e74c3c"

        bals = state.get("balances_breakdown", [])
        active_bals = [
            f"{b.get('asset_symbol')}: {float(b.get('balance', 0)):.4f}"
            for b in bals if float(b.get("balance", 0)) > 0
        ]
        bals_text = " | ".join(active_bals) if active_bals else "Connect API to view breakdown"

        st.markdown(
            f"""
            <div style="background-color: #151922; padding: 18px; border-radius: 10px; border: 1px solid #242b38;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-weight: 700; font-size: 1.1rem;">{pos_str}</span>
                    <span style="color: {pnl_color}; font-size: 1.2rem; font-weight: 700;">
                        {pnl_val:+.2f} USD ({pnl_pct:+.2f}%)
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem; color: #8b949e;">
                    <div>Contracts: <strong style="color: #f0f6fc;">{position['size']}</strong></div>
                    <div>Contract Val: <strong style="color: #f0f6fc;">{specs.get('contract_value', '0.001')}</strong></div>
                    <div>Entry Price: <strong style="color: #f0f6fc;">${position['entry_price']:,.1f}</strong></div>
                    <div>Stop Loss: <strong style="color: #e74c3c;">${position['stop_loss']:,.1f}</strong></div>
                    <div>Take Profit: <strong style="color: #2ecc71;">${position['take_profit']:,.1f}</strong></div>
                    <div>Live Delta Balance: <strong style="color: #58a6ff;">${state.get('wallet_equity', 0.0):,.2f}</strong></div>
                    <div>Simulated Balance: <strong style="color: #f0f6fc;">${state['simulated_equity']:,.2f}</strong></div>
                    <div>Asset Wallets: <span style="color: #f0f6fc; font-size: 0.8rem;">{bals_text}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Overall Profit & Loss Card
        pnl_net_color = "#2ecc71" if net_overall_pnl >= 0 else "#e74c3c"
        st.markdown(
            f"""
            <div style="background-color: #151922; padding: 16px; border-radius: 10px; border: 1px solid #242b38; margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #f0f6fc;">🏆 Overall Trading Profit / Loss</span>
                    <span style="color: {pnl_net_color}; font-size: 1.15rem; font-weight: 700;">{net_overall_pnl:+.2f} USD</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem; color: #8b949e;">
                    <div>Realized P&L: <strong style="color: {'#2ecc71' if total_realized_pnl >= 0 else '#e74c3c'};">${total_realized_pnl:+.2f}</strong></div>
                    <div>Unrealized P&L: <strong style="color: {'#2ecc71' if unrealized_pnl >= 0 else '#e74c3c'};">${unrealized_pnl:+.2f}</strong></div>
                    <div>Total Trades: <strong style="color: #f0f6fc;">{total_trades}</strong></div>
                    <div>Win Rate: <strong style="color: #58a6ff;">{win_rate:.1f}% ({len(win_trades)}W / {len(loss_trades)}L)</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_logs:
        c_head_title, c_head_btn = st.columns([3.8, 1.2])
        with c_head_title:
            st.markdown("### 💻 **Live Trading Terminal**")
        with c_head_btn:
            if st.button("🧹 Clear", key="clear_term_btn", help="Clear terminal buffer", use_container_width=True):
                bot_manager.clear_logs()
                st.rerun()

        # Heartbeat & Activity Banner
        cycle_count = state.get("cycle_count", 0)
        last_hb = state.get("last_heartbeat", 0)
        sec_since_hb = max(0, int(time.time() - last_hb)) if last_hb > 0 else 0

        if is_running:
            hb_badge = f"""
            <div style="background-color: #0d1527; border: 1px solid #1e3a5f; border-radius: 8px; padding: 7px 14px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="pulse-dot"></span>
                    <strong style="color: #4ade80;">DAEMON ACTIVE & SCANNING</strong>
                    <span style="color: #475569;">|</span>
                    <span style="color: #38bdf8; font-weight: 600;">Scan #{cycle_count:03d}</span>
                    <span style="color: #475569;">|</span>
                    <span style="color: #94a3b8;">Last tick: {sec_since_hb}s ago</span>
                </div>
                <div style="color: #58a6ff; font-family: monospace; font-size: 0.76rem;">
                    ● POLLING DELTA ({state['symbol']} {state['resolution']})
                </div>
            </div>
            """
        else:
            hb_badge = f"""
            <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 7px 14px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #94a3b8; margin-right: 2px;"></span>
                    <strong style="color: #94a3b8;">ENGINE IN STANDBY</strong>
                    <span style="color: #475569;">|</span>
                    <span style="color: #64748b;">Ready to trade {state['symbol']}</span>
                </div>
                <div style="color: #38bdf8; font-weight: 600; font-size: 0.76rem;">
                    Click [▶ Start Bot] in sidebar
                </div>
            </div>
            """
        st.markdown(hb_badge, unsafe_allow_html=True)

        # Format Terminal Logs Content (Chronological: newest at bottom, last 80 entries for rich scrollback)
        display_logs = logs[-80:] if logs else []
        body_content = format_terminal_logs(display_logs, active_tz)

        status_tag = (
            '<span style="color: #4ade80; font-weight: 600;">● SCANNING (TICK OK)</span>'
            if is_running
            else '<span style="color: #94a3b8;">● IDLE</span>'
        )

        cursor_elem = '<span class="terminal-cursor"></span>' if is_running else ''
        cmd_prompt = (
            f'<div style="margin-top: 10px; padding-top: 6px; border-top: 1px dashed #1e293b; color: #38bdf8; font-family: monospace;">'
            f'<span style="color: #4ade80; font-weight: bold;">delta-bot@cloud</span>:<span style="color: #38bdf8;">~</span>$ '
            f'<span style="color: #94a3b8;">live_feed active &mdash; polling tick stream</span> {cursor_elem}</div>'
            if is_running
            else f'<div style="margin-top: 10px; padding-top: 6px; border-top: 1px dashed #1e293b; color: #64748b; font-family: monospace;">'
                 f'<span style="color: #94a3b8;">delta-bot@cloud</span>:<span style="color: #64748b;">~</span>$ standby {cursor_elem}</div>'
        )

        terminal_html = render_smart_terminal_html(
            symbol=state["symbol"],
            resolution=state["resolution"],
            is_running=is_running,
            status_tag=status_tag,
            body_content=body_content,
            cmd_prompt=cmd_prompt,
        )
        components.html(terminal_html, height=385)


    # Trade History Table
    st.markdown("### 📖 **Closed Trades History**")
    if trade_history:
        history_df = pd.DataFrame(trade_history)
        if "pnl" in history_df.columns:
            history_df["Cumulative PnL ($)"] = history_df["pnl"].cumsum().round(2)
            history_df["pnl"] = history_df["pnl"].apply(lambda x: f"${x:+.2f}")
            history_df["Cumulative PnL ($)"] = history_df["Cumulative PnL ($)"].apply(lambda x: f"${x:+.2f}")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No closed trades yet in this session.")


# Render main interactive dashboard
render_dashboard(
    symbol=symbol_selected,
    resolution=timeframe_selected,
    active_tz=active_tz,
    active_tz_abbr=active_tz_abbr,
    prob_threshold=prob_threshold,
    min_return_hurdle=min_return_hurdle,
    take_profit_target=take_profit_target,
)
