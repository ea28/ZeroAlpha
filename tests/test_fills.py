from datetime import UTC, datetime, timedelta

from zeroalpha.backtest.fills import simulate_limit_fill
from zeroalpha.domain import Bar, Side


def _bar(ts: datetime, *, high: float, low: float) -> Bar:
    return Bar(
        timestamp_utc=ts,
        symbol="BTCUSDT",
        bar_size="1m",
        open=100,
        high=high,
        low=low,
        close=100,
        volume=1,
        source="TEST",
    )


def _close_stamped_bar(start: datetime, *, high: float, low: float) -> Bar:
    return Bar(
        timestamp_utc=start + timedelta(minutes=1),
        symbol="BTCUSDT",
        bar_size="1m",
        open=100,
        high=high,
        low=low,
        close=100,
        volume=1,
        source="TEST",
        extra={
            "bar_start_timestamp_utc": start.isoformat(),
            "bar_close_timestamp_utc": (start + timedelta(minutes=1)).isoformat(),
        },
    )


def test_limit_fill_requires_configured_trade_through() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(minutes=1), high=100.10, low=99.99)]

    missed = simulate_limit_fill(
        Side.BUY,
        100.0,
        bars,
        require_trade_through_bps=2.0,
    )
    filled = simulate_limit_fill(
        Side.BUY,
        100.0,
        [_bar(start + timedelta(minutes=2), high=100.10, low=99.90)],
        require_trade_through_bps=2.0,
    )

    assert not missed.filled
    assert missed.reason == "missed_fill"
    assert filled.filled
    assert filled.price == 100.0
    assert filled.timestamp_utc == start + timedelta(minutes=2)


def test_limit_fill_honors_latency_and_partial_fill_fraction() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        _bar(start + timedelta(seconds=10), high=100.10, low=99.0),
        _bar(start + timedelta(seconds=70), high=100.10, low=99.0),
    ]

    fill = simulate_limit_fill(
        Side.BUY,
        100.0,
        bars,
        latency_seconds=60,
        fill_fraction=0.25,
    )

    assert fill.filled
    assert fill.timestamp_utc == start + timedelta(seconds=70)
    assert fill.fill_fraction == 0.25
    assert fill.reason == "partial_limit_touched"


def test_limit_fill_does_not_use_ohlc_before_order_activation() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    activation = start + timedelta(seconds=30)

    fill = simulate_limit_fill(
        Side.BUY,
        100.0,
        [_close_stamped_bar(start, high=100.10, low=99.0)],
        activation_timestamp=activation,
    )

    assert not fill.filled
    assert fill.reason == "missed_fill"


def test_limit_fill_returns_bar_start_when_live_from_open() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)

    fill = simulate_limit_fill(
        Side.BUY,
        100.0,
        [_close_stamped_bar(start, high=100.10, low=99.0)],
        activation_timestamp=start,
    )

    assert fill.filled
    assert fill.timestamp_utc == start
