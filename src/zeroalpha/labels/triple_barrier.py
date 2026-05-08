"""Triple-barrier, path-dependent labels."""

from __future__ import annotations

from datetime import datetime, timedelta
import math

from zeroalpha.bars import bar_start_timestamp_utc
from zeroalpha.domain import Bar, CandidateEvent, Side, TripleBarrierLabel
from zeroalpha.timeutils import ensure_utc


def _event_min_holding_seconds(event: CandidateEvent) -> float:
    try:
        min_holding_seconds = float(event.metadata.get("min_holding_seconds", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(min_holding_seconds) or min_holding_seconds <= 0:
        return 0.0
    return min_holding_seconds


def label_long_event(
    event: CandidateEvent,
    future_bars: list[Bar],
    *,
    entry_price: float,
    entry_timestamp_utc: datetime | None = None,
    round_trip_cost_bps: float,
    net_profit_target: float | None = None,
    net_stop_loss: float | None = None,
    profit_target: float | None = None,
    stop_distance: float | None = None,
    conservative_same_bar: bool = True,
) -> TripleBarrierLabel:
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be nonnegative")
    if net_profit_target is None:
        net_profit_target = profit_target
    if net_stop_loss is None:
        net_stop_loss = stop_distance
    if net_profit_target is None or net_stop_loss is None:
        raise ValueError("net_profit_target and net_stop_loss are required")
    if net_profit_target <= 0 or net_stop_loss <= 0:
        raise ValueError("net barrier outcomes must be positive")
    cost_fraction = round_trip_cost_bps / 10_000
    if net_stop_loss <= cost_fraction:
        raise ValueError("net_stop_loss must exceed estimated round-trip cost")

    upper = entry_price * (1 + net_profit_target + cost_fraction)
    lower = entry_price * (1 + cost_fraction - net_stop_loss)
    if lower <= 0:
        raise ValueError("net_stop_loss produces a nonpositive lower barrier")
    vertical = event.vertical_barrier_timestamp_utc
    ordered = [
        bar
        for bar in sorted(future_bars, key=lambda row: row.timestamp_utc)
        if event.timestamp_utc < bar.timestamp_utc <= vertical
    ]
    if not ordered:
        raise ValueError("future_bars must include at least one bar after the event")

    exit_bar = ordered[-1]
    outcome = "vertical"
    exit_price = exit_bar.close
    entry_timestamp = ensure_utc(entry_timestamp_utc) if entry_timestamp_utc else ordered[0].timestamp_utc
    min_exit_timestamp = entry_timestamp + timedelta(seconds=_event_min_holding_seconds(event))
    for bar in ordered:
        if bar_start_timestamp_utc(bar) < min_exit_timestamp:
            continue
        hit_upper = bar.high >= upper
        hit_lower = bar.low <= lower
        if hit_upper and hit_lower:
            exit_bar = bar
            if conservative_same_bar:
                outcome = "lower_same_bar"
                exit_price = lower
            else:
                outcome = "upper_same_bar"
                exit_price = upper
            break
        if hit_lower:
            exit_bar = bar
            outcome = "lower"
            exit_price = lower
            break
        if hit_upper:
            exit_bar = bar
            outcome = "upper"
            exit_price = upper
            break

    gross_return = exit_price / entry_price - 1
    net_return = gross_return - cost_fraction
    label = 1 if outcome.startswith("upper") and net_return >= net_profit_target - 1e-12 else 0
    return TripleBarrierLabel(
        event_id=event.event_id,
        entry_timestamp_utc=entry_timestamp,
        entry_price=entry_price,
        upper_barrier_price=upper,
        lower_barrier_price=lower,
        vertical_barrier_timestamp_utc=vertical,
        exit_timestamp_utc=exit_bar.timestamp_utc,
        exit_price=exit_price,
        outcome_type=outcome,
        gross_return=gross_return,
        net_return=net_return,
        label=label,
        t1=exit_bar.timestamp_utc,
    )


def label_short_event(
    event: CandidateEvent,
    future_bars: list[Bar],
    *,
    entry_price: float,
    entry_timestamp_utc: datetime | None = None,
    round_trip_cost_bps: float,
    net_profit_target: float,
    net_stop_loss: float,
    conservative_same_bar: bool = True,
) -> TripleBarrierLabel:
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be nonnegative")
    if net_profit_target <= 0 or net_stop_loss <= 0:
        raise ValueError("net barrier outcomes must be positive")
    cost_fraction = round_trip_cost_bps / 10_000
    if net_stop_loss <= cost_fraction:
        raise ValueError("net_stop_loss must exceed estimated round-trip cost")

    lower = entry_price * (1 - net_profit_target - cost_fraction)
    upper = entry_price * (1 + net_stop_loss - cost_fraction)
    if lower <= 0:
        raise ValueError("net_profit_target produces a nonpositive lower barrier")
    vertical = event.vertical_barrier_timestamp_utc
    ordered = [
        bar
        for bar in sorted(future_bars, key=lambda row: row.timestamp_utc)
        if event.timestamp_utc < bar.timestamp_utc <= vertical
    ]
    if not ordered:
        raise ValueError("future_bars must include at least one bar after the event")

    exit_bar = ordered[-1]
    outcome = "vertical"
    exit_price = exit_bar.close
    entry_timestamp = ensure_utc(entry_timestamp_utc) if entry_timestamp_utc else ordered[0].timestamp_utc
    min_exit_timestamp = entry_timestamp + timedelta(seconds=_event_min_holding_seconds(event))
    for bar in ordered:
        if bar_start_timestamp_utc(bar) < min_exit_timestamp:
            continue
        hit_lower = bar.low <= lower
        hit_upper = bar.high >= upper
        if hit_lower and hit_upper:
            exit_bar = bar
            if conservative_same_bar:
                outcome = "upper_same_bar"
                exit_price = upper
            else:
                outcome = "lower_same_bar"
                exit_price = lower
            break
        if hit_upper:
            exit_bar = bar
            outcome = "upper"
            exit_price = upper
            break
        if hit_lower:
            exit_bar = bar
            outcome = "lower"
            exit_price = lower
            break

    gross_return = (entry_price - exit_price) / entry_price
    net_return = gross_return - cost_fraction
    label = 1 if outcome.startswith("lower") and net_return >= net_profit_target - 1e-12 else 0
    return TripleBarrierLabel(
        event_id=event.event_id,
        entry_timestamp_utc=entry_timestamp,
        entry_price=entry_price,
        upper_barrier_price=upper,
        lower_barrier_price=lower,
        vertical_barrier_timestamp_utc=vertical,
        exit_timestamp_utc=exit_bar.timestamp_utc,
        exit_price=exit_price,
        outcome_type=outcome,
        gross_return=gross_return,
        net_return=net_return,
        label=label,
        t1=exit_bar.timestamp_utc,
    )


def label_event(
    event: CandidateEvent,
    future_bars: list[Bar],
    *,
    entry_price: float,
    entry_timestamp_utc: datetime | None = None,
    round_trip_cost_bps: float,
    net_profit_target: float,
    net_stop_loss: float,
    conservative_same_bar: bool = True,
) -> TripleBarrierLabel:
    if event.side == Side.SELL:
        return label_short_event(
            event,
            future_bars,
            entry_price=entry_price,
            entry_timestamp_utc=entry_timestamp_utc,
            round_trip_cost_bps=round_trip_cost_bps,
            net_profit_target=net_profit_target,
            net_stop_loss=net_stop_loss,
            conservative_same_bar=conservative_same_bar,
        )
    return label_long_event(
        event,
        future_bars,
        entry_price=entry_price,
        entry_timestamp_utc=entry_timestamp_utc,
        round_trip_cost_bps=round_trip_cost_bps,
        net_profit_target=net_profit_target,
        net_stop_loss=net_stop_loss,
        conservative_same_bar=conservative_same_bar,
    )
