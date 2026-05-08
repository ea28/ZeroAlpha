from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.candidates.events import CandidateGenerationConfig, generate_candidate_events
from zeroalpha.candidates.rules import (
    active_breakout_continuation_candidate,
    intraday_volatility_breakout_candidate,
    liquidity_sweep_reclaim_candidate,
)
from zeroalpha.domain import Bar, Side


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


def _second_bar(idx: int, close: float) -> Bar:
    return Bar(
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=idx),
        symbol="BTCUSDT",
        bar_size="1s",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
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
    assert all(isinstance(event.metadata.get("specialist_setup_family"), str) for event in events)
    assert {event.metadata.get("setup_score_direction") for event in events} <= {"high", "none"}
    assert all("dense_range_position_24" in event.metadata for event in events)


def test_adaptive_horizon_shortens_when_recent_one_second_volatility_is_high() -> None:
    quiet = [_second_bar(idx, 100 + idx * 0.0001) for idx in range(400)]
    jumpy = [
        _second_bar(idx, 100 * (1 + (0.005 if idx % 2 else -0.005)))
        for idx in range(400)
    ]
    config = CandidateGenerationConfig(
        mode="dense_research",
        min_history_bars=300,
        max_holding_hours=4,
        adaptive_horizon=True,
        min_holding_seconds=1,
        adaptive_horizon_target_move_bps=50,
    )

    quiet_event = generate_candidate_events(quiet, config=config)[-1]
    jumpy_event = generate_candidate_events(jumpy, config=config)[-1]

    assert jumpy_event.max_holding_period >= timedelta(seconds=1)
    assert quiet_event.max_holding_period <= timedelta(hours=4)
    assert jumpy_event.max_holding_period < quiet_event.max_holding_period
    assert jumpy_event.metadata["adaptive_horizon_source"] == "volatility_scaled_setup_horizon"


def test_adaptive_horizon_can_round_to_label_bar_granularity_below_feature_interval() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=idx),
            symbol="BTCUSDT",
            bar_size="1m",
            open=100.0,
            high=100.0 * (1 + 0.008),
            low=100.0 * (1 - 0.008),
            close=100.0 * (1 + (0.006 if idx % 2 else -0.006)),
            volume=1.0,
            source="TEST",
        )
        for idx in range(400)
    ]

    event = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=300,
            max_holding_hours=4,
            adaptive_horizon=True,
            min_holding_seconds=1,
            adaptive_horizon_granularity_seconds=1,
            adaptive_horizon_target_move_bps=50,
        ),
    )[-1]

    assert event.max_holding_period < timedelta(seconds=60)
    assert event.metadata["adaptive_horizon_interval_seconds"] == 60.0
    assert event.metadata["adaptive_horizon_granularity_seconds"] == 1.0


def test_adaptive_horizon_does_not_treat_missing_volume_as_compression() -> None:
    bars = [
        Bar(
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=idx),
            symbol="BTCUSDT",
            bar_size="1s",
            open=100 + idx * 0.01,
            high=100 + idx * 0.01 + 0.02,
            low=100 + idx * 0.01 - 0.02,
            close=100 + idx * 0.01,
            volume=0.0,
            source="TEST",
        )
        for idx in range(400)
    ]

    event = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=300,
            max_holding_hours=4,
            adaptive_horizon=True,
            min_holding_seconds=1,
            adaptive_horizon_target_move_bps=50,
        ),
    )[-1]

    assert event.metadata["dense_volume_available"] is False
    assert event.metadata["dense_volume_ratio_24"] == 1.0
    assert event.metadata["adaptive_horizon_movement_bps"] > 0
    assert event.max_holding_period < timedelta(hours=4)


def test_dense_research_can_emit_long_and_short_candidates() -> None:
    bars = [_bar(idx) for idx in range(20)]
    events = generate_candidate_events(
        bars,
        config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=5,
            dense_stride_bars=1,
            max_holding_hours=1,
            side_mode="long_short",
            allow_short_research=True,
        ),
    )

    assert len(events) == 32
    assert {event.side for event in events} == {Side.BUY, Side.SELL}
    assert len({event.event_id for event in events}) == len(events)
    short = next(event for event in events if event.side == Side.SELL)
    assert short.metadata["dense_side"] == "SELL"
    assert short.signal_strength == pytest.approx(-short.metadata["dense_raw_signal_strength"])


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
