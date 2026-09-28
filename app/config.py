"""
Application configuration management using Pydantic Settings and Environment Variables.

Phase 1 Hard-Constraints:
- PAPER_TRADING is locked to True.
- LIVE_TRADING_ENABLED is locked to False.
- Secrets are encapsulated using SecretStr to prevent accidental exposure or logging.
- Asset symbol and timeframe are dynamically configurable via environment variables.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupportedTimeframe(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    D1 = "1d"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and/or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Project metadata
    APP_NAME: str = "XRP AI Trading Agent"
    APP_VERSION: str = "0.1.0"
    PHASE: str = "Phase 1 - Infrastructure Only"
    ENVIRONMENT: str = "development"

    # Delta Exchange REST and WebSocket URLs
    DELTA_BASE_URL: str = Field(
        default="https://api.delta.exchange",
        description="Base URL for Delta Exchange REST API (e.g. https://api.delta.exchange or testnet)"
    )
    DELTA_WS_URL: str = Field(
        default="wss://socket.delta.exchange",
        description="WebSocket URL for Delta Exchange streaming"
    )

    # Authentication (Optional in read-only / public market data mode)
    DELTA_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Delta Exchange API Key"
    )
    DELTA_API_SECRET: Optional[SecretStr] = Field(
        default=None,
        description="Delta Exchange API Secret Key"
    )

    # Trading asset & resolution configuration
    DELTA_SYMBOL: str = Field(
        default="XRPUSDT",
        description="Delta Exchange contract symbol for XRP (e.g. XRPUSDT, XRP_USDT, XRPUSD_PERP)"
    )
    TIMEFRAME: str = Field(
        default="5m",
        description="Default candle resolution / timeframe"
    )

    # Safety constraints - Strictly locked in Phase 1
    PAPER_TRADING: bool = Field(
        default=True,
        description="True for paper/read-only mode. Hard-locked to True in Phase 1."
    )
    LIVE_TRADING_ENABLED: bool = Field(
        default=False,
        description="Hard safety lock. Live order placement is forbidden in Phase 1."
    )

    # Database & Storage
    DB_PATH: str = Field(
        default="data/trading_agent.db",
        description="Path to SQLite database"
    )

    # Structured Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    LOG_FILE: Optional[str] = Field(
        default="logs/trading_agent.log",
        description="File path for logging output"
    )

    # API Server
    HOST: str = Field(default="127.0.0.1", description="FastAPI bind host")
    PORT: int = Field(default=8000, description="FastAPI bind port")

    @field_validator("DELTA_SYMBOL")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        symbol = v.strip().upper()
        if not symbol:
            raise ValueError("DELTA_SYMBOL cannot be empty.")
        return symbol

    @field_validator("TIMEFRAME")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        valid_resolutions = [tf.value for tf in SupportedTimeframe]
        cleaned = v.strip().lower()
        if cleaned not in valid_resolutions:
            raise ValueError(
                f"Invalid TIMEFRAME '{v}'. Supported timeframes are: {', '.join(valid_resolutions)}"
            )
        return cleaned

    @field_validator("DELTA_BASE_URL")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        url = v.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"DELTA_BASE_URL must begin with http:// or https://. Got: {v}")
        return url

    @model_validator(mode="after")
    def enforce_phase1_safety_guards(self) -> "Settings":
        """
        Enforce strict safety invariant: Phase 1 cannot run live trading.
        """
        if not self.PAPER_TRADING:
            raise ValueError(
                "SAFETY VIOLATION: PAPER_TRADING cannot be set to False in Phase 1. "
                "Phase 1 only supports read-only and paper-trading modes."
            )
        if self.LIVE_TRADING_ENABLED:
            raise ValueError(
                "SAFETY VIOLATION: LIVE_TRADING_ENABLED cannot be True in Phase 1. "
                "All live order execution is intentionally disabled in this phase."
            )
        return self

    @property
    def is_auth_configured(self) -> bool:
        """Check whether API credentials have been supplied without exposing them."""
        has_key = bool(self.DELTA_API_KEY and self.DELTA_API_KEY.get_secret_value().strip())
        has_secret = bool(self.DELTA_API_SECRET and self.DELTA_API_SECRET.get_secret_value().strip())
        return has_key and has_secret

    def safe_dict(self) -> dict:
        """
        Return configuration dictionary with all credentials strictly masked or omitted.
        Safe for logging, debugging, and /health responses.
        """
        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "phase": self.PHASE,
            "environment": self.ENVIRONMENT,
            "delta_base_url": self.DELTA_BASE_URL,
            "delta_symbol": self.DELTA_SYMBOL,
            "timeframe": self.TIMEFRAME,
            "paper_trading": self.PAPER_TRADING,
            "live_trading_enabled": self.LIVE_TRADING_ENABLED,
            "db_path": self.DB_PATH,
            "log_level": self.LOG_LEVEL,
            "auth_configured": self.is_auth_configured,
        }


# Global cached settings instance factory
_settings_instance: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Retrieve or initialize application settings."""
    global _settings_instance
    if _settings_instance is None or reload:
        _settings_instance = Settings()
    return _settings_instance
