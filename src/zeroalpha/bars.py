"""Small helpers for normalized OHLCV bar timestamps."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from zeroalpha.domain import Bar
from zeroalpha.timeutils import ensure_utc


def _parse_extra_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, str) and value:
        try:
            return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def bar_start_timestamp_utc(bar: Bar) -> datetime:
    """Return the opening timestamp for a close-stamped bar when metadata exists."""

    return _parse_extra_timestamp(bar.extra.get("bar_start_timestamp_utc")) or bar.timestamp_utc


def bar_close_timestamp_utc(bar: Bar) -> datetime:
    """Return the closing timestamp for a bar, falling back to its canonical timestamp."""

    return _parse_extra_timestamp(bar.extra.get("bar_close_timestamp_utc")) or bar.timestamp_utc
