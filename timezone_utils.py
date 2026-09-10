"""
Timezone helper utilities for Delta AI Trading Bot.
Defaults to Indian Standard Time (IST, UTC+5:30) used by Delta Exchange (India)
while supporting global timezones (UTC, EST, PST, GST, SGT).
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple

# Default Timezone: IST (UTC+5:30)
IST_TZ = timezone(timedelta(hours=5, minutes=30), name="IST")

TIMEZONE_MAP: Dict[str, Tuple[timezone, str]] = {
    "IST (India - UTC+5:30)": (timezone(timedelta(hours=5, minutes=30), name="IST"), "IST"),
    "UTC (Coordinated Universal Time)": (timezone.utc, "UTC"),
    "EST (US Eastern - UTC-5)": (timezone(timedelta(hours=-5), name="EST"), "EST"),
    "PST (US Pacific - UTC-8)": (timezone(timedelta(hours=-8), name="PST"), "PST"),
    "GST (Dubai - UTC+4)": (timezone(timedelta(hours=4), name="GST"), "GST"),
    "SGT (Singapore - UTC+8)": (timezone(timedelta(hours=8), name="SGT"), "SGT"),
}

def get_now(tz: timezone = IST_TZ) -> datetime:
    """Returns the current datetime in the specified timezone (default: IST)."""
    return datetime.now(tz)

def format_now(fmt: str = "%I:%M:%S %p", tz: timezone = IST_TZ) -> str:
    """Formats current datetime in the specified timezone."""
    return get_now(tz).strftime(fmt)
