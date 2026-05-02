from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.data.external.prediction_markets import (
    PreparedPredictionMarketSnapshots,
    PredictionMarketSnapshot,
    _polymarket_slug,
)
from zeroalpha.domain import CandidateEvent, Side
from zeroalpha.models.dataset import _add_prediction_market_features


def _event(ts: datetime, *, side: Side = Side.BUY) -> CandidateEvent:
    return CandidateEvent(
        event_id="event-1",
        timestamp_utc=ts,
        symbol="BTCUSDT",
        candidate_type="dense_research_bar",
        side=side,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=100.0,
        max_holding_hours=4,
    )


def test_polymarket_short_form_slug_shapes() -> None:
    ts = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)

    assert _polymarket_slug("5m", ts) == "btc-updown-5m-1777752000"
    assert _polymarket_slug("15m", ts) == "btc-updown-15m-1777752000"
    assert _polymarket_slug("4h", ts) == "btc-updown-4h-1777752000"
    assert _polymarket_slug("1h", ts) == "bitcoin-up-or-down-may-2-2026-4pm-et"


def test_prediction_market_features_use_latest_active_causal_snapshot() -> None:
    event_time = datetime(2026, 5, 2, 20, 10, tzinfo=UTC)
    snapshots = [
        PredictionMarketSnapshot(
            provider="polymarket",
            duration="15m",
            timestamp_utc=event_time - timedelta(minutes=3),
            market_id="active",
            market_slug="btc-updown-15m-active",
            market_title="active",
            condition_id="active",
            window_start_utc=event_time - timedelta(minutes=10),
            window_end_utc=event_time + timedelta(minutes=5),
            up_mid=0.55,
            down_mid=0.45,
        ),
        PredictionMarketSnapshot(
            provider="polymarket",
            duration="15m",
            timestamp_utc=event_time - timedelta(minutes=2),
            market_id="old",
            market_slug="btc-updown-15m-old",
            market_title="old",
            condition_id="old",
            window_start_utc=event_time - timedelta(minutes=25),
            window_end_utc=event_time - timedelta(minutes=10),
            up_mid=0.99,
            down_mid=0.01,
        ),
        PredictionMarketSnapshot(
            provider="polymarket",
            duration="15m",
            timestamp_utc=event_time - timedelta(minutes=1),
            market_id="active",
            market_slug="btc-updown-15m-active",
            market_title="active",
            condition_id="active",
            window_start_utc=event_time - timedelta(minutes=10),
            window_end_utc=event_time + timedelta(minutes=5),
            up_bid=0.61,
            up_ask=0.63,
            up_mid=0.62,
            down_bid=0.37,
            down_ask=0.39,
            down_mid=0.38,
            up_bid_size=100.0,
            down_bid_size=50.0,
        ),
        PredictionMarketSnapshot(
            provider="polymarket",
            duration="15m",
            timestamp_utc=event_time + timedelta(seconds=1),
            market_id="future",
            market_slug="btc-updown-15m-future",
            market_title="future",
            condition_id="future",
            window_start_utc=event_time - timedelta(minutes=10),
            window_end_utc=event_time + timedelta(minutes=5),
            up_mid=0.10,
            down_mid=0.90,
        ),
    ]
    prepared = PreparedPredictionMarketSnapshots.from_snapshots(snapshots)
    features: dict[str, float | str] = {"return_elapsed_15m": 0.002}

    _add_prediction_market_features(features, event=_event(event_time), prediction_markets=prepared)

    assert features["pm_polymarket_15m_available"] == 1.0
    assert features["pm_polymarket_15m_up_mid"] == 0.62
    assert features["pm_polymarket_15m_down_mid"] == 0.38
    assert features["pm_polymarket_15m_direction_skew"] == pytest.approx(0.24)
    assert features["pm_polymarket_15m_side_aligned_skew"] == pytest.approx(0.24)
    assert features["pm_polymarket_15m_up_spread"] == pytest.approx(0.02)
    assert features["pm_polymarket_15m_bid_size_imbalance"] == pytest.approx(1 / 3)
    assert features["pm_polymarket_15m_up_mid_change"] == pytest.approx(0.07)
    assert features["pm_polymarket_15m_direction_skew_change"] == pytest.approx(0.14)
    assert features["pm_polymarket_15m_side_spot_agreement"] == 1.0


def test_prediction_market_features_flip_side_alignment_for_short_research() -> None:
    event_time = datetime(2026, 5, 2, 20, 10, tzinfo=UTC)
    snapshot = PredictionMarketSnapshot(
        provider="kalshi",
        duration="15m",
        timestamp_utc=event_time,
        market_id="kxbtc15m",
        market_slug="kxbtc15m",
        market_title="BTC price up in next 15 mins?",
        condition_id="event",
        window_start_utc=event_time - timedelta(minutes=5),
        window_end_utc=event_time + timedelta(minutes=10),
        up_mid=0.40,
        down_mid=0.60,
    )
    prepared = PreparedPredictionMarketSnapshots.from_snapshots([snapshot])
    features: dict[str, float | str] = {}

    _add_prediction_market_features(
        features,
        event=_event(event_time, side=Side.SELL),
        prediction_markets=prepared,
    )

    assert features["pm_kalshi_15m_direction_skew"] == pytest.approx(-0.20)
    assert features["pm_kalshi_15m_side_aligned_skew"] == pytest.approx(0.20)
    assert features["pm_kalshi_15m_side_aligned_mid"] == 0.60
