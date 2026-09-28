"""
Unit tests for configuration validation, environment variable handling, and secret protection.
"""

import os
import pytest
from pydantic import ValidationError

from app.config import Settings, SupportedTimeframe, get_settings


def test_default_settings():
    """Verify standard default configuration parameters."""
    settings = Settings()
    assert settings.DELTA_SYMBOL == "XRPUSDT"
    assert settings.TIMEFRAME == "5m"
    assert settings.PAPER_TRADING is True
    assert settings.LIVE_TRADING_ENABLED is False
    assert settings.DELTA_BASE_URL == "https://api.delta.exchange"
    assert settings.is_auth_configured is False


def test_supported_timeframes():
    """Verify that all enum timeframes are accepted."""
    for tf in SupportedTimeframe:
        s = Settings(TIMEFRAME=tf.value)
        assert s.TIMEFRAME == tf.value


def test_invalid_timeframe_rejected():
    """Verify that unsupported timeframe strings raise a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(TIMEFRAME="10s")
    assert "Invalid TIMEFRAME" in str(exc_info.value)


def test_empty_symbol_rejected():
    """Verify that empty or whitespace symbol raises a ValidationError."""
    with pytest.raises(ValidationError):
        Settings(DELTA_SYMBOL="   ")


def test_custom_symbol_accepted():
    """Verify custom Delta contract symbol configuration."""
    s = Settings(DELTA_SYMBOL="xrp_usdt")
    assert s.DELTA_SYMBOL == "XRP_USDT"


def test_safety_guard_paper_trading_false_rejected():
    """Phase 1 invariant: PAPER_TRADING cannot be False."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(PAPER_TRADING=False)
    assert "SAFETY VIOLATION" in str(exc_info.value)


def test_safety_guard_live_trading_true_rejected():
    """Phase 1 invariant: LIVE_TRADING_ENABLED cannot be True."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(LIVE_TRADING_ENABLED=True)
    assert "SAFETY VIOLATION" in str(exc_info.value)


def test_secret_protection():
    """Verify secrets are masked in str representations and safe_dict."""
    secret_val = "SUPER_SECRET_KEY_12345"
    s = Settings(
        DELTA_API_KEY="my_api_key_999",
        DELTA_API_SECRET=secret_val
    )
    assert s.is_auth_configured is True

    # Pydantic SecretStr masks in str/repr
    assert secret_val not in str(s)
    assert secret_val not in repr(s)

    # safe_dict must omit raw secrets
    safe = s.safe_dict()
    assert "DELTA_API_SECRET" not in safe
    assert "delta_api_secret" not in safe
    assert secret_val not in str(safe)
    assert safe["auth_configured"] is True


def test_missing_env_variables_fallback_gracefully(monkeypatch):
    """Verify application runs with empty/missing optional environment variables."""
    monkeypatch.delenv("DELTA_API_KEY", raising=False)
    monkeypatch.delenv("DELTA_API_SECRET", raising=False)

    s = Settings()
    assert s.DELTA_API_KEY is None
    assert s.DELTA_API_SECRET is None
    assert s.is_auth_configured is False
