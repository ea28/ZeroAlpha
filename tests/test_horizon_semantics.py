from datetime import UTC, datetime, timedelta

from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.labels.triple_barrier import label_long_event


def _event(start: datetime, *, hours: int = 72) -> CandidateEvent:
    return CandidateEvent(
        event_id="event",
        timestamp_utc=start,
        symbol="BTCUSDT",
        candidate_type="test",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1.0,
        reference_price=100.0,
        max_holding_hours=hours,
    )


def _bar(ts: datetime, *, bar_size: str, high: float = 100.5, low: float = 99.5, close: float = 100.0) -> Bar:
    return Bar(
        timestamp_utc=ts,
        symbol="BTCUSDT",
        bar_size=bar_size,
        open=100.0,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        source="TEST",
    )


def test_label_horizon_uses_elapsed_time_for_1m_bars_not_row_count() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(minutes=i), bar_size="1m") for i in range(1, 60 * 80)]
    bars[1000] = _bar(bars[1000].timestamp_utc, bar_size="1m", high=103.0)

    label = label_long_event(
        _event(start),
        bars,
        entry_price=100.0,
        round_trip_cost_bps=50.0,
        net_profit_target=0.01,
        net_stop_loss=0.02,
    )

    assert label.outcome_type == "upper"
    assert label.exit_timestamp_utc == bars[1000].timestamp_utc


def test_label_horizon_uses_elapsed_time_for_4h_bars() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=4 * i), bar_size="4h") for i in range(1, 25)]
    bars[-1] = _bar(bars[-1].timestamp_utc, bar_size="4h", high=103.0)

    label = label_long_event(
        _event(start),
        bars,
        entry_price=100.0,
        round_trip_cost_bps=50.0,
        net_profit_target=0.01,
        net_stop_loss=0.02,
    )

    assert label.outcome_type == "vertical"
    assert label.exit_timestamp_utc == start + timedelta(hours=72)
