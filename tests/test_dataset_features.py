from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import AppConfig, CostConfig, LabelConfig, RiskConfig
from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.features.basic import build_event_features
from zeroalpha.models.dataset import _add_event_metadata, build_meta_label_samples, build_scoring_samples


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
            min_holding_seconds=5,
            dense_stride_bars=1,
        ),
        max_samples=3,
    )

    assert samples
    assert samples[-1].timestamp_utc == bars[-1].timestamp_utc
    assert "bar_close" in samples[-1].features
    assert samples[-1].features["feature_asof_is_entry"] == 0.0
    assert samples[-1].features["point_signal_delay_seconds"] == pytest.approx(0.0)
    assert samples[-1].features["min_holding_seconds"] == pytest.approx(5.0)
    assert samples[-1].features["event_min_holding_seconds"] == pytest.approx(5.0)


def test_scoring_samples_include_live_context_bars() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    primary = [
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
    eth = [
        Bar(
            timestamp_utc=start + timedelta(minutes=idx),
            symbol="ETHUSDT",
            bar_size="1m",
            open=50.0 + idx * 0.02,
            high=50.5 + idx * 0.02,
            low=49.5 + idx * 0.02,
            close=50.1 + idx * 0.02,
            volume=20.0 + idx,
            source="BINANCE",
        )
        for idx in range(260)
    ]

    samples = build_scoring_samples(
        primary,
        config=AppConfig(),
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=240,
            max_holding_hours=1,
            dense_stride_bars=1,
        ),
        context_bars={"ETH": eth},
        max_samples=1,
    )

    assert samples
    assert samples[-1].features["eth_available"] == 1.0
    assert "btc_relative_strength_vs_eth_24" in samples[-1].features


def test_scoring_samples_include_one_second_microstructure_features() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(seconds=idx),
            symbol="BTCUSDT",
            bar_size="1s",
            open=100.0 + idx * 0.01,
            high=100.03 + idx * 0.01,
            low=99.99 + idx * 0.01,
            close=100.02 + idx * 0.01 if idx % 3 else 100.0 + idx * 0.01,
            volume=0.1 + idx * 0.001,
            trade_count=idx % 5 + 1,
            source="IBKR:AGGTRADES",
            extra={"spread_bps": 0.15, "quote_age_seconds": 0.4},
        )
        for idx in range(320)
    ]

    samples = build_scoring_samples(
        bars,
        config=AppConfig(),
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=300,
            dense_stride_bars=1,
            max_holding_seconds=30,
            min_holding_seconds=1,
        ),
        max_samples=1,
    )

    assert samples
    features = samples[0].features
    assert features["micro_bar_interval_seconds"] == pytest.approx(1.0)
    assert features["micro_is_one_second_bar"] == 1.0
    assert features["micro_is_five_second_or_faster_bar"] == 1.0
    assert features["micro_tick_count"] > 0
    assert features["micro_tick_intensity_5s"] > 0
    assert "micro_signed_tick_imbalance_30s" in features
    assert "micro_realized_vol_burst_5s_vs_60s" in features
    assert features["micro_quote_stale_5s"] == 0.0


def test_meta_label_samples_can_use_faster_label_bars_for_second_level_exits() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    primary_bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=idx),
            symbol="BTCUSDT",
            bar_size="1m",
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.0,
            volume=10.0,
            source="IBKR:AGGTRADES",
        )
        for idx in range(320)
    ]
    label_bars = [
        Bar(
            timestamp_utc=start + timedelta(seconds=idx),
            symbol="BTCUSDT",
            bar_size="1 secs",
            open=100.0,
            high=100.8,
            low=99.9,
            close=100.2,
            volume=0.1,
            source="IBKR:AGGTRADES",
            extra={
                "bar_start_timestamp_utc": (start + timedelta(seconds=idx - 1)).isoformat(),
                "bar_close_timestamp_utc": (start + timedelta(seconds=idx)).isoformat(),
            },
        )
        for idx in range(1, 320 * 60 + 61)
    ]
    cfg = replace(
        AppConfig(),
        cost=CostConfig(
            tier_rate=0.0018,
            minimum_commission=0.0,
            base_slippage_bps=0.5,
            safety_margin_bps=1.0,
        ),
        risk=replace(RiskConfig(), paper_max_notional=10_000.0),
        labels=LabelConfig(
            max_holding_hours=1,
            max_holding_seconds=30,
            net_profit_target=0.001,
            net_stop_loss=0.01,
            minimum_gross_profit_bps=50,
            minimum_gross_stop_bps=120,
        ),
    )

    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=0.1,
        research_notional=10_000.0,
        label_bars=label_bars,
        feature_asof="entry",
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=300,
            dense_stride_bars=1,
            max_holding_seconds=30,
            max_holding_hours=1,
            min_holding_seconds=5,
        ),
    )

    assert samples
    sample = samples[0]
    assert sample.label_detail.entry_timestamp_utc == sample.timestamp_utc
    assert sample.label_detail.exit_timestamp_utc == sample.timestamp_utc + timedelta(seconds=6)
    assert sample.label_detail.vertical_barrier_timestamp_utc - sample.timestamp_utc == timedelta(seconds=30)
    assert sample.features["label_bar_interval_seconds"] == pytest.approx(1.0)
    assert sample.features["feature_bar_interval_seconds"] == pytest.approx(60.0)
    assert sample.features["point_signal_delay_seconds"] == pytest.approx(0.0)
    assert sample.features["min_holding_seconds"] == pytest.approx(5.0)
    assert sample.features["event_min_holding_seconds"] == pytest.approx(5.0)
    assert sample.label_detail.exit_timestamp_utc >= (
        sample.label_detail.entry_timestamp_utc + timedelta(seconds=5)
    )
    assert sample.max_adverse_excursion == pytest.approx(0.001)
    assert sample.max_favorable_excursion == pytest.approx(0.008)
    assert sample.time_to_exit_seconds == pytest.approx(6.0)
    assert sample.early_adverse_label == 0
