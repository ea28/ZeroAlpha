from datetime import UTC, datetime, timedelta

from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.features.basic import build_event_features


def _bar(ts: datetime, close: float, *, bar_size: str = "1m") -> Bar:
    return Bar(
        timestamp_utc=ts,
        symbol="BTCUSDT",
        bar_size=bar_size,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        trade_count=10,
        source="TEST",
    )


def test_elapsed_features_keep_time_semantics_on_1m_bars() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(minutes=i), 100.0 + i) for i in range(60 * 30)]
    event = CandidateEvent(
        event_id="e",
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="1m",
        signal_strength=1.0,
        reference_price=bars[-1].close,
        max_holding_hours=24,
    )

    features = build_event_features(event, bars)

    raw_24_bar_return = bars[-1].close / bars[-25].close - 1
    elapsed_24h_return = bars[-1].close / bars[-1 - 60 * 24].close - 1
    assert features["return_24"] == raw_24_bar_return
    assert features["return_elapsed_24h"] == elapsed_24h_return
    assert features["return_elapsed_24h"] != features["return_24"]


def test_elapsed_features_match_bar_count_on_1h_bars() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(start + timedelta(hours=i), 100.0 + i, bar_size="1h") for i in range(200)]
    event = CandidateEvent(
        event_id="e",
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1.0,
        reference_price=bars[-1].close,
        max_holding_hours=24,
    )

    features = build_event_features(event, bars)

    assert features["return_elapsed_24h"] == features["return_24"]


def test_market_structure_features_are_causal() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(hours=i),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=100.0 + i,
            trade_count=10 + i,
            source="TEST",
        )
        for i in range(30)
    ]
    event = CandidateEvent(
        event_id="e",
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        candidate_type="volatility_breakout",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1.0,
        reference_price=bars[-1].close,
        max_holding_hours=24,
    )

    features = build_event_features(event, bars)

    assert features["candle_close_position"] == (bars[-1].close - bars[-1].low) / (bars[-1].high - bars[-1].low)
    assert features["breakout_close_through_24_high"] == bars[-1].close / max(bar.high for bar in bars[-24:-1]) - 1
    assert features["participation_ratio_24"] > 1.0
