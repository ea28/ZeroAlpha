"""Simple fill and missed-fill simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import blake2b

from zeroalpha.bars import bar_start_timestamp_utc
from zeroalpha.domain import Bar, Side
from zeroalpha.timeutils import ensure_utc


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    filled: bool
    price: float | None
    reason: str
    timestamp_utc: datetime | None = None
    fill_fraction: float = 1.0


def _deterministic_score(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def simulate_limit_fill(
    side: Side,
    limit_price: float,
    bars: list[Bar],
    *,
    activation_timestamp: datetime | None = None,
    latency_seconds: float = 0.0,
    require_trade_through_bps: float = 0.0,
    fill_probability: float = 1.0,
    fill_fraction: float = 1.0,
) -> SimulatedFill:
    if limit_price <= 0:
        raise ValueError("limit_price must be positive")
    if latency_seconds < 0:
        raise ValueError("latency_seconds must be nonnegative")
    if require_trade_through_bps < 0:
        raise ValueError("require_trade_through_bps must be nonnegative")
    if not 0 <= fill_probability <= 1:
        raise ValueError("fill_probability must be in [0, 1]")
    if not 0 < fill_fraction <= 1:
        raise ValueError("fill_fraction must be in (0, 1]")
    ordered = sorted(bars, key=lambda row: row.timestamp_utc)
    if not ordered:
        return SimulatedFill(False, None, "no_entry_bars")
    active_timestamp = (
        ensure_utc(activation_timestamp)
        if activation_timestamp is not None
        else ordered[0].timestamp_utc
    )
    active_after = active_timestamp.timestamp() + latency_seconds
    trade_through = require_trade_through_bps / 10_000
    for bar in ordered:
        bar_start = bar_start_timestamp_utc(bar)
        if bar.timestamp_utc.timestamp() < active_after:
            continue
        if bar_start.timestamp() < active_after:
            continue
        touched = (
            bar.low <= limit_price * (1 - trade_through)
            if side == Side.BUY
            else bar.high >= limit_price * (1 + trade_through)
        )
        if not touched:
            continue
        score = _deterministic_score(side.value, limit_price, bar.timestamp_utc.isoformat())
        if score > fill_probability:
            return SimulatedFill(False, None, "queue_not_filled", timestamp_utc=bar_start)
        reason = "limit_trade_through" if require_trade_through_bps > 0 else "limit_touched"
        if fill_fraction < 1:
            reason = f"partial_{reason}"
        return SimulatedFill(
            True,
            limit_price,
            reason,
            timestamp_utc=bar_start,
            fill_fraction=fill_fraction,
        )
    return SimulatedFill(False, None, "missed_fill")


def simulate_market_ioc(side: Side, bid: float, ask: float) -> SimulatedFill:
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError("invalid bid/ask")
    return SimulatedFill(True, ask if side == Side.BUY else bid, "market_ioc", fill_fraction=1.0)
