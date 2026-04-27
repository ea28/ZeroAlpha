"""Typed domain objects for ZeroAlpha."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from zeroalpha.timeutils import ensure_utc, utc_now


class RuntimeMode(StrEnum):
    RESEARCH = "research"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    LMT = "LMT"
    MKT = "MKT"


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    MINUTES = "Minutes"


class BotState(StrEnum):
    INITIALIZING = "INITIALIZING"
    IDLE = "IDLE"
    SIGNAL_DETECTED = "SIGNAL_DETECTED"
    SCORING = "SCORING"
    RISK_CHECK = "RISK_CHECK"
    ORDER_PENDING = "ORDER_PENDING"
    IN_POSITION = "IN_POSITION"
    EXIT_PENDING = "EXIT_PENDING"
    COOLDOWN = "COOLDOWN"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class MarketQuote:
    timestamp_utc: datetime
    received_timestamp_utc: datetime
    symbol: str
    bid: float
    ask: float
    source: str = "IBKR"
    bid_size: float | None = None
    ask_size: float | None = None
    market_data_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        object.__setattr__(self, "received_timestamp_utc", ensure_utc(self.received_timestamp_utc))
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("bid and ask must be positive")
        if self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return 10_000 * self.spread / self.midpoint

    def quote_age_ms(self, now: datetime | None = None) -> float:
        now = ensure_utc(now or utc_now())
        return max(0.0, (now - self.received_timestamp_utc).total_seconds() * 1000)


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp_utc: datetime
    symbol: str
    bar_size: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    quote_volume: float | None = None
    trade_count: int | None = None
    vwap: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is inconsistent with OHLC")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is inconsistent with OHLC")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    event_id: str
    timestamp_utc: datetime
    symbol: str
    candidate_type: str
    side: Side
    bar_size: str
    signal_strength: float
    reference_price: float
    max_holding_hours: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        if self.reference_price <= 0:
            raise ValueError("reference_price must be positive")

    @property
    def max_holding_period(self) -> timedelta:
        return timedelta(hours=self.max_holding_hours)

    @property
    def vertical_barrier_timestamp_utc(self) -> datetime:
        return self.timestamp_utc + self.max_holding_period


@dataclass(frozen=True, slots=True)
class TripleBarrierLabel:
    event_id: str
    entry_timestamp_utc: datetime
    entry_price: float
    upper_barrier_price: float
    lower_barrier_price: float
    vertical_barrier_timestamp_utc: datetime
    exit_timestamp_utc: datetime
    exit_price: float
    outcome_type: str
    gross_return: float
    net_return: float
    label: int
    t1: datetime

    def __post_init__(self) -> None:
        for name in (
            "entry_timestamp_utc",
            "vertical_barrier_timestamp_utc",
            "exit_timestamp_utc",
            "t1",
        ):
            object.__setattr__(self, name, ensure_utc(getattr(self, name)))
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")


@dataclass(frozen=True, slots=True)
class Prediction:
    event_id: str
    model_version: str
    calibrated_probability: float
    expected_value: float
    decision: str
    decision_reason: str

    def __post_init__(self) -> None:
        if not 0 <= self.calibrated_probability <= 1:
            raise ValueError("calibrated_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    internal_order_id: str
    event_id: str | None
    symbol: str
    side: Side
    order_type: OrderType
    time_in_force: TimeInForce
    quantity: float | None = None
    cash_qty: float | None = None
    limit_price: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LMT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.quantity is None and self.cash_qty is None:
            raise ValueError("order intent requires quantity or cash_qty")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.cash_qty is not None and self.cash_qty <= 0:
            raise ValueError("cash_qty must be positive")


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    internal_order_id: str
    timestamp_filled_utc: datetime
    symbol: str
    side: Side
    filled_quantity: float
    fill_price: float
    commission: float
    spread_at_fill_bps: float | None = None
    slippage_bps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_filled_utc", ensure_utc(self.timestamp_filled_utc))
        if self.filled_quantity <= 0 or self.fill_price <= 0 or self.commission < 0:
            raise ValueError("invalid fill values")
