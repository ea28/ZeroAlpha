from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from zeroalpha.broker.streaming import TickBarAggregator
from zeroalpha.domain import MarketQuote


def test_tick_bar_aggregator_emits_completed_one_second_quote_bar() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    aggregator = TickBarAggregator("BTC/USD", bar_size_seconds=1)

    aggregator.add_quote(
        MarketQuote(
            timestamp_utc=start + timedelta(milliseconds=100),
            received_timestamp_utc=start + timedelta(milliseconds=100),
            symbol="BTC/USD",
            bid=99.0,
            ask=101.0,
            bid_size=1.5,
            ask_size=2.5,
        )
    )

    bars = aggregator.completed_bars(start + timedelta(seconds=1, milliseconds=1))

    assert len(bars) == 1
    assert bars[0].timestamp_utc == start + timedelta(seconds=1)
    assert bars[0].bar_size == "1 secs"
    assert bars[0].open == 100.0
    assert bars[0].close == 100.0
    assert bars[0].extra["bid"] == 99.0
    assert bars[0].extra["ask_size"] == 2.5


def test_tick_bar_aggregator_processes_tick_by_tick_rows_once() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    aggregator = TickBarAggregator("BTC/USD", bar_size_seconds=1)
    ticker = SimpleNamespace(
        tickByTicks=[
            SimpleNamespace(time=start + timedelta(milliseconds=100), price=100.0, size=2.0),
            SimpleNamespace(time=start + timedelta(milliseconds=600), price=101.0, size=3.0),
        ]
    )

    assert aggregator.process_ticker_ticks(ticker) == 2
    assert aggregator.process_ticker_ticks(ticker) == 0
    bars = aggregator.completed_bars(start + timedelta(seconds=2))

    assert len(bars) == 1
    assert bars[0].high == 101.0
    assert bars[0].low == 100.0
    assert bars[0].volume == 5.0
    assert bars[0].trade_count == 2
