"""Time helpers used across data, labels, and broker code."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_unix_timestamp(value: int | float) -> datetime:
    """Parse seconds, milliseconds, or microseconds since epoch."""
    numeric = float(value)
    if numeric > 10_000_000_000_000:
        numeric /= 1_000_000
    elif numeric > 10_000_000_000:
        numeric /= 1_000
    return datetime.fromtimestamp(numeric, tz=UTC)


def floor_to_interval(ts: datetime, interval: timedelta) -> datetime:
    ts = ensure_utc(ts)
    seconds = int(interval.total_seconds())
    if seconds <= 0:
        raise ValueError("interval must be positive")
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=UTC)
