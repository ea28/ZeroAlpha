from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import AppConfig
from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.features.basic import build_event_features
from zeroalpha.models.dataset import _add_event_metadata, build_scoring_samples


def test_event_metadata_preserves_setup_family_for_specialist_routing() -> None:
    event = CandidateEvent(
        event_id="event-1",
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="active_liquidity_reversal",
        side=Side.BUY,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=100.0,
        max_holding_hours=4,
        metadata={"setup_family": "liquidation_reversal", "volume_ratio": 1.2},
    )
    features: dict[str, float | str] = {}

    _add_event_metadata(features, event)

    assert features["event_setup_family"] == "liquidation_reversal"
    assert features["event_volume_ratio"] == 1.2


def test_bar_features_include_volume_vwap_and_taker_flow() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start,
            symbol="BTCUSDT",
            bar_size="1m",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
            quote_volume=1000.0,
            trade_count=10,
            vwap=100.0,
            source="BINANCE",
            extra={"taker_buy_base_volume": 5.0},
        ),
        Bar(
            timestamp_utc=start.replace(minute=1),
            symbol="BTCUSDT",
            bar_size="1m",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=20.0,
            quote_volume=2020.0,
            trade_count=20,
            vwap=100.5,
            source="BINANCE",
            extra={
                "taker_buy_base_volume": 14.0,
                "bid": 100.9,
                "ask": 101.1,
                "bid_size": 3.0,
                "ask_size": 1.0,
                "bid_depth_5bps": 8.0,
                "ask_depth_5bps": 2.0,
            },
        ),
    ]
    event = CandidateEvent(
        event_id="event-1",
        timestamp_utc=bars[-1].timestamp_utc,
        symbol="BTCUSDT",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="1m",
        signal_strength=0.01,
        reference_price=101.0,
        max_holding_hours=1,
    )

    features = build_event_features(event, bars)

    assert features["has_quote_volume"] == 1.0
    assert features["dollar_volume"] == 2020.0
    assert features["volume_per_trade"] == 1.0
    assert features["vwap_distance_bps"] > 0
    assert features["taker_buy_volume_share"] == 0.7
    assert features["signed_taker_buy_imbalance"] == pytest.approx(0.4)
    assert features["side_taker_buy_imbalance"] == pytest.approx(0.4)
    assert features["dollar_volume_ratio_24"] > 1.0
    assert "rsi_14_centered" in features
    assert "macd_histogram_bps" in features
    assert features["top_book_spread_bps"] > 0
    assert features["side_microprice_distance_bps"] > 0
    assert features["side_l2_depth_imbalance"] == pytest.approx(0.6)


def test_scoring_samples_use_completed_bars_without_future_labels() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=idx),
            symbol="BTCUSDT",
            bar_size="1m",
            open=100.0 + idx * 0.01,
            high=100.5 + idx * 0.01,
            low=99.5 + idx * 0.01,
            close=100.1 + idx * 0.01,
            volume=10.0 + idx,
            source="BINANCE",
        )
        for idx in range(260)
    ]

    samples = build_scoring_samples(
        bars,
        config=AppConfig(),
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=240,
            max_holding_hours=1,
            dense_stride_bars=1,
        ),
        max_samples=3,
    )

    assert samples
    assert samples[-1].timestamp_utc <= bars[-1].timestamp_utc
    assert "bar_close" in samples[-1].features
