"""
Security and Safety Tests.

Verifies:
- All order execution and cancellation APIs are hard-disabled in Phase 1.
- Non-GET HTTP verbs trigger the safety shield.
- Structured logger sanitizer redacts sensitive patterns.
"""

import logging
import pytest
from app.config import Settings
from app.exchange.delta_client import DeltaExchangeClient, Phase1SafetyViolationError
from app.logging_config import SensitiveDataFilter, redact_sensitive_text


def test_order_creation_blocked():
    """Verify create_order unconditionally raises Phase1SafetyViolationError."""
    client = DeltaExchangeClient()
    with pytest.raises(Phase1SafetyViolationError) as exc_info:
        client.create_order(symbol="XRPUSDT", size=10, side="buy")
    assert "Order creation is completely disabled in Phase 1" in str(exc_info.value)


def test_order_cancellation_blocked():
    """Verify cancel_order unconditionally raises Phase1SafetyViolationError."""
    client = DeltaExchangeClient()
    with pytest.raises(Phase1SafetyViolationError) as exc_info:
        client.cancel_order(order_id="12345")
    assert "Order cancellation is disabled in Phase 1" in str(exc_info.value)


def test_cancel_all_orders_blocked():
    """Verify cancel_all_orders unconditionally raises Phase1SafetyViolationError."""
    client = DeltaExchangeClient()
    with pytest.raises(Phase1SafetyViolationError) as exc_info:
        client.cancel_all_orders()
    assert "Bulk order cancellation is disabled in Phase 1" in str(exc_info.value)


def test_http_mutations_blocked_by_safety_shield():
    """Verify _request blocks POST, PUT, DELETE even if invoked directly."""
    client = DeltaExchangeClient()
    for verb in ["POST", "PUT", "DELETE", "PATCH"]:
        with pytest.raises(Phase1SafetyViolationError) as exc_info:
            client._request(verb, "/v2/orders", json_data={"size": 1})
        assert "SAFETY SHIELD ACTIVATED" in str(exc_info.value)


def test_sensitive_text_redaction():
    """Verify sensitive credentials are scrubbed by the redaction logic."""
    raw_message = (
        'Sending request with api_secret="abc12345xyz67890" '
        'and signature="d2d0b6ca8b868e4de88d6c707567848f" '
        'and private_key="9988776655443322"'
    )
    sanitized = redact_sensitive_text(raw_message)

    assert "abc12345xyz67890" not in sanitized
    assert "d2d0b6ca8b868e4de88d6c707567848f" not in sanitized
    assert "9988776655443322" not in sanitized
    assert "[REDACTED]" in sanitized


def test_logging_filter_scrubs_log_records():
    """Verify SensitiveDataFilter redacts records passed to logger."""
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Leaked secret: api_secret=mysecretkey12345678",
        args=(),
        exc_info=None
    )
    filter_instance = SensitiveDataFilter()
    filter_instance.filter(record)
    assert "mysecretkey12345678" not in record.msg
    assert "[REDACTED]" in record.msg
