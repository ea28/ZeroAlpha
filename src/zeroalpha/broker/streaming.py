"""Streaming IBKR market-data helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any

from zeroalpha.domain import Bar, MarketQuote
from zeroalpha.timeutils import ensure_utc, floor_to_interval, utc_now


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _positive_float(value: Any) -> float | None:
    converted = _finite_float(value)
    return converted if converted is not None and converted > 0 else None


@dataclass(slots=True)
class _WorkingBar:
    start_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trade_count: int = 0
    quote_count: int = 0
    tick_count: int = 0
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None

    def add(
        self,
        price: float,
        *,
        size: float = 0.0,
        trade_count: int = 0,
        quote_count: int = 0,
        tick_count: int = 0,
        bid: float | None = None,
        ask: float | None = None,
        bid_size: float | None = None,
        ask_size: float | None = None,
    ) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += max(0.0, size)
        self.trade_count += max(0, trade_count)
        self.quote_count += max(0, quote_count)
        self.tick_count += max(0, tick_count)
        self.bid = bid if bid is not None else self.bid
        self.ask = ask if ask is not None else self.ask
        self.bid_size = bid_size if bid_size is not None else self.bid_size
        self.ask_size = ask_size if ask_size is not None else self.ask_size


@dataclass(slots=True)
class TickBarAggregator:
    """Aggregate streaming quotes or tick-by-tick rows into completed bars."""

    symbol: str
    bar_size_seconds: int = 1
    source: str = "IBKR:STREAM"
    _working: dict[datetime, _WorkingBar] = field(default_factory=dict)
    _ticker_offsets: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bar_size_seconds <= 0:
            raise ValueError("bar_size_seconds must be positive")

    @property
    def interval(self) -> timedelta:
        return timedelta(seconds=self.bar_size_seconds)

    @property
    def bar_size_label(self) -> str:
        if self.bar_size_seconds == 1:
            return "1 secs"
        if self.bar_size_seconds < 60:
            return f"{self.bar_size_seconds} secs"
        if self.bar_size_seconds % 3600 == 0:
            hours = self.bar_size_seconds // 3600
            return f"{hours} hour" if hours == 1 else f"{hours} hours"
        if self.bar_size_seconds % 60 == 0:
            minutes = self.bar_size_seconds // 60
            return f"{minutes} min" if minutes == 1 else f"{minutes} mins"
        return f"{self.bar_size_seconds} secs"

    def add_observation(
        self,
        *,
        timestamp_utc: datetime,
        price: float,
        size: float = 0.0,
        trade_count: int = 0,
        quote_count: int = 0,
        tick_count: int = 0,
        bid: float | None = None,
        ask: float | None = None,
        bid_size: float | None = None,
        ask_size: float | None = None,
    ) -> bool:
        price = _positive_float(price)
        if price is None:
            return False
        timestamp_utc = ensure_utc(timestamp_utc)
        bucket_start = floor_to_interval(timestamp_utc, self.interval)
        working = self._working.get(bucket_start)
        if working is None:
            working = _WorkingBar(
                start_utc=bucket_start,
                open=price,
                high=price,
                low=price,
                close=price,
            )
            self._working[bucket_start] = working
        working.add(
            price,
            size=size,
            trade_count=trade_count,
            quote_count=quote_count,
            tick_count=tick_count,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )
        return True

    def add_quote(self, quote: MarketQuote) -> bool:
        return self.add_observation(
            timestamp_utc=quote.received_timestamp_utc,
            price=quote.midpoint,
            quote_count=1,
            bid=quote.bid,
            ask=quote.ask,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
        )

    def process_ticker_ticks(self, ticker: Any) -> int:
        rows = getattr(ticker, "tickByTicks", []) or []
        key = id(ticker)
        start = self._ticker_offsets.get(key, 0)
        added = 0
        for tick in rows[start:]:
            if self._process_tick(tick):
                added += 1
        self._ticker_offsets[key] = len(rows)
        return added

    def _process_tick(self, tick: Any) -> bool:
        timestamp = getattr(tick, "time", None)
        timestamp_utc = ensure_utc(timestamp) if isinstance(timestamp, datetime) else utc_now()
        bid = _positive_float(getattr(tick, "bidPrice", None))
        ask = _positive_float(getattr(tick, "askPrice", None))
        if bid is not None and ask is not None and ask >= bid:
            return self.add_observation(
                timestamp_utc=timestamp_utc,
                price=(bid + ask) / 2,
                tick_count=1,
                bid=bid,
                ask=ask,
                bid_size=_finite_float(getattr(tick, "bidSize", None)),
                ask_size=_finite_float(getattr(tick, "askSize", None)),
            )
        price = _positive_float(getattr(tick, "price", None))
        if price is not None:
            return self.add_observation(
                timestamp_utc=timestamp_utc,
                price=price,
                size=max(0.0, _finite_float(getattr(tick, "size", None)) or 0.0),
                trade_count=1,
                tick_count=1,
            )
        midpoint = _positive_float(getattr(tick, "midPoint", None))
        if midpoint is not None:
            return self.add_observation(timestamp_utc=timestamp_utc, price=midpoint, tick_count=1)
        return False

    def completed_bars(self, now: datetime | None = None) -> list[Bar]:
        cutoff_start = floor_to_interval(ensure_utc(now or utc_now()), self.interval)
        ready = [start for start in self._working if start < cutoff_start]
        bars: list[Bar] = []
        for start in sorted(ready):
            working = self._working.pop(start)
            if working.quote_count > 0 and working.tick_count > 0:
                aggregated_from = "streaming_tick_by_tick_and_quote"
            elif working.tick_count > 0:
                aggregated_from = "streaming_tick_by_tick"
            else:
                aggregated_from = "streaming_quote_sample"
            extra: dict[str, float | str] = {
                "bar_start_timestamp_utc": start.isoformat(),
                "bar_close_timestamp_utc": (start + self.interval).isoformat(),
                "aggregated_from": aggregated_from,
                "quote_count": float(working.quote_count),
                "tick_count": float(working.tick_count),
            }
            for name, value in (
                ("bid", working.bid),
                ("ask", working.ask),
                ("bid_size", working.bid_size),
                ("ask_size", working.ask_size),
            ):
                if value is not None and math.isfinite(value):
                    extra[name] = value
            bars.append(
                Bar(
                    timestamp_utc=start + self.interval,
                    symbol=self.symbol,
                    bar_size=self.bar_size_label,
                    open=working.open,
                    high=working.high,
                    low=working.low,
                    close=working.close,
                    volume=working.volume,
                    trade_count=working.trade_count or None,
                    source=self.source,
                    extra=extra,
                )
            )
        return bars
