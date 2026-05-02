from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.candidates.events import CandidateGenerationConfig, generate_candidate_events
from zeroalpha.candidates.rules import (
    active_breakout_continuation_candidate,
    intraday_volatility_breakout_candidate,
    liquidity_sweep_reclaim_candidate,
)
from zeroalpha.domain import Bar


def _bar(idx: int) -> Bar:
    return Bar(
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
        symbol="BTCUSDT",
        bar_size="1h",
        open=100 + idx * 0.01,
        high=101 + idx * 0.01,
        low=99 + idx * 0.01,
        close=100.5 + idx * 0.01,
        volume=1.0,
        source="TEST",
    )


def test_dense_research_candidates_can_emit_every_bar_after_warmup() -> None:
    bars = [_bar(idx) for idx in range(20)]
    events = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=5,
            dense_stride_bars=1,
            max_holding_hours=1,
        ),
    )

    assert len(events) == 16
    assert {event.candidate_type for event in events} == {"dense_research_bar"}
    assert events[0].timestamp_utc == bars[4].timestamp_utc
    assert all(isinstance(event.metadata.get("setup_family"), str) for event in events)
    assert all("dense_range_position_24" in event.metadata for event in events)


def test_spot_short_research_requires_explicit_override() -> None:
    bars = [_bar(idx) for idx in range(260)]

    with pytest.raises(ValueError, match="short candidate research"):
        generate_candidate_events(
            bars,
            config=CandidateGenerationConfig(min_history_bars=5, side_mode="long_short"),
        )

    events = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            min_history_bars=5,
            side_mode="long_short",
            allow_short_research=True,
        ),
    )

    assert isinstance(events, list)


def test_intraday_volatility_breakout_candidate_detects_fast_breakout() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100.0,
            high=100.2,
            low=99.9,
            close=100.05,
            volume=100.0,
            trade_count=100,
            source="TEST",
        )
        for idx in range(100)
    ]
    bars[-1] = Bar(
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        bar_size="1h",
        open=100.1,
        high=101.7,
        low=99.9,
        close=101.4,
        volume=130.0,
        trade_count=130,
        source="TEST",
    )

    event = intraday_volatility_breakout_candidate(bars)

    assert event is not None
    assert event.candidate_type == "intraday_volatility_breakout"
    assert event.max_holding_hours == 12


def test_liquidity_sweep_reclaim_candidate_detects_reclaim() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100.0,
            high=100.5,
            low=99.0,
            close=100.1,
            volume=100.0,
            source="TEST",
        )
        for idx in range(100)
    ]
    bars[-1] = Bar(
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        bar_size="1h",
        open=99.1,
        high=101.0,
        low=98.6,
        close=100.7,
        volume=120.0,
        source="TEST",
    )

    event = liquidity_sweep_reclaim_candidate(bars)

    assert event is not None
    assert event.candidate_type == "liquidity_sweep_reclaim"
    assert event.max_holding_hours == 12


def test_aggressive_candidate_mode_is_explicit() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100.0,
            high=100.2,
            low=99.9,
            close=100.05,
            volume=100.0,
            trade_count=100,
            source="TEST",
        )
        for idx in range(100)
    ]
    bars[-1] = Bar(
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        bar_size="1h",
        open=100.1,
        high=101.7,
        low=99.9,
        close=101.4,
        volume=130.0,
        trade_count=130,
        source="TEST",
    )

    standard = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(min_history_bars=5, mode="rules"),
    )
    aggressive = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(min_history_bars=5, mode="aggressive_rules"),
    )

    assert "intraday_volatility_breakout" not in {event.candidate_type for event in standard}
    assert "intraday_volatility_breakout" in {event.candidate_type for event in aggressive}


def test_active_candidate_mode_emits_lower_timeframe_setup_family() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * idx),
            symbol="BTCUSDT",
            bar_size="15m",
            open=100.0,
            high=100.2,
            low=99.9,
            close=100.05,
            volume=100.0,
            trade_count=100,
            source="TEST",
        )
        for idx in range(120)
    ]
    bars[-1] = Bar(
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        bar_size="15m",
        open=100.1,
        high=101.0,
        low=100.0,
        close=100.8,
        volume=130.0,
        trade_count=130,
        source="TEST",
    )

    event = active_breakout_continuation_candidate(bars, max_holding_hours=4)
    active = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            min_history_bars=5,
            mode="active_research",
            max_holding_hours=4,
        ),
    )

    assert event is not None
    assert event.metadata["setup_family"] == "breakout"
    assert "active_breakout_continuation" in {candidate.candidate_type for candidate in active}
