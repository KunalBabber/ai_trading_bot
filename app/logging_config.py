"""
Structured logging module for XRP Trading Agent.

Required log schema:
- timestamp (ISO 8601 UTC)
- module
- event
- symbol
- timeframe
- severity

Security Invariant:
- NEVER logs API secrets, private keys, or authentication headers.
- Sanitizes sensitive patterns proactively.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Regex patterns matching sensitive key names and headers
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?secret|secret[_-]?key|private[_-]?key)[\"':\s=]+([a-zA-Z0-9_\-\.]{8,})"),
    re.compile(r"(?i)(signature)[\"':\s=]+([a-fA-F0-9]{32,64})"),
    re.compile(r"(?i)(api[_-]?key)[\"':\s=]+([a-zA-Z0-9_\-\.]{8,})"),
    re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{16,})"),
]


def redact_sensitive_text(text: str) -> str:
    """Mask credentials and signature hashes from string content."""
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(r"\1: [REDACTED]", redacted)
    return redacted


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that sanitizes message strings and prevents secrets
    or authentication headers from leaking to logs.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact_sensitive_text(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_sensitive_text(str(a)) if isinstance(a, str) else a for a in record.args)
        return True


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON log formatter producing strictly structured events.
    Fields: timestamp, module, event, symbol, timeframe, severity, message.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Resolve UTC timestamp
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # Extract structured extra attributes or supply defaults
        event = getattr(record, "event", "GENERIC_EVENT")
        symbol = getattr(record, "symbol", "N/A")
        timeframe = getattr(record, "timeframe", "N/A")

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "severity": record.levelname,
            "module": record.name,
            "event": event,
            "symbol": symbol,
            "timeframe": timeframe,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class StructuredTextFormatter(logging.Formatter):
    """
    Human-readable structured console formatter preserving all required fields.
    """
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        event = getattr(record, "event", "LOG")
        symbol = getattr(record, "symbol", "-")
        timeframe = getattr(record, "timeframe", "-")
        severity = record.levelname

        formatted = (
            f"[{timestamp}] [{severity:<7}] [{record.name}] "
            f"event={event} symbol={symbol} tf={timeframe} :: {record.getMessage()}"
        )
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        return formatted


class AgentLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that automatically attaches context (symbol, timeframe, event) to log calls.
    """
    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = kwargs.setdefault("extra", {})
        # Merge contextual defaults from self.extra
        for key, value in self.extra.items():
            if key not in extra:
                extra[key] = value
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/trading_agent.log",
    json_format: bool = False
) -> None:
    """
    Configure application-wide structured logging.
    """
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Create Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = StructuredJsonFormatter() if json_format else StructuredTextFormatter()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(StructuredJsonFormatter())  # Always JSON in files for log ingestion
            file_handler.addFilter(SensitiveDataFilter())
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Could not initialize log file at {log_file}: {e}")


def get_logger(name: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> AgentLoggerAdapter:
    """
    Convenience factory to obtain an AgentLoggerAdapter with symbol and timeframe context.
    """
    base_logger = logging.getLogger(name)
    extra_context = {
        "symbol": symbol or "N/A",
        "timeframe": timeframe or "N/A",
        "event": "SYSTEM"
    }
    return AgentLoggerAdapter(base_logger, extra_context)
