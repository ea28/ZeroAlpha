from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.data.external.prediction_markets import (
    KalshiPublicDataClient,
    PreparedPredictionMarketSnapshots,
    PredictionMarketSnapshot,
    PolymarketClobV2Client,
    _polymarket_slug,
)
from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import AppConfig
from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.models.dataset import _add_prediction_market_features, build_meta_label_samples


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


def test_polymarket_price_history_does_not_backfill_current_market_totals() -> None:
    class FakePolymarket(PolymarketClobV2Client):
        def order_books(self, token_ids):  # type: ignore[no-untyped-def]
            return {}

        def midpoints(self, token_ids):  # type: ignore[no-untyped-def]
            return {}

        def spreads(self, token_ids):  # type: ignore[no-untyped-def]
            return {}

        def batch_prices_history(self, token_ids, **kwargs):  # type: ignore[no-untyped-def]
            return {token_id: [{"t": 1777752300, "p": 0.55}] for token_id in token_ids}

    market = {
        "_zeroalpha_duration": "5m",
        "_zeroalpha_window_start_utc": "2026-05-02T20:00:00+00:00",
        "_zeroalpha_window_end_utc": "2026-05-02T20:05:00+00:00",
        "id": "m",
        "slug": "btc-updown-5m",
        "question": "BTC up or down",
        "conditionId": "c",
        "outcomes": '["Up","Down"]',
        "clobTokenIds": '["up","down"]',
        "lastTradePrice": "0.99",
        "volume": "100000",
        "volume24hr": "200000",
        "liquidity": "300000",
        "openInterest": "400000",
    }

    snapshots = FakePolymarket().snapshots_from_markets(
        [market],
        start=datetime(2026, 5, 2, 20, 0, tzinfo=UTC),
        end=datetime(2026, 5, 2, 20, 5, tzinfo=UTC),
        include_orderbooks=False,
    )

    assert snapshots
    assert snapshots[0].source == "clob_v2_prices_history"
    assert snapshots[0].last_price is None
    assert snapshots[0].volume is None
    assert snapshots[0].liquidity is None
    assert snapshots[0].open_interest is None


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


def test_prediction_market_features_measure_residual_leading_window() -> None:
    event_time = datetime(2026, 5, 2, 20, 10, tzinfo=UTC)
    window_start = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)
    history = [
        Bar(
            timestamp_utc=window_start + timedelta(minutes=minute),
            symbol="BTC/USD",
            bar_size="1m",
            open=100.0,
            high=101.5,
            low=99.5,
            close=100.0 + minute * 0.1,
            volume=1.0,
            source="IBKR:AGGTRADES",
        )
        for minute in range(11)
    ]
    snapshot = PredictionMarketSnapshot(
        provider="kalshi",
        duration="15m",
        timestamp_utc=event_time,
        market_id="kxbtc15m",
        market_slug="kxbtc15m",
        market_title="BTC price up in next 15 mins?",
        condition_id="event",
        window_start_utc=window_start,
        window_end_utc=window_start + timedelta(minutes=15),
        up_mid=0.62,
        down_mid=0.38,
        volume=100.0,
        volume_24h=250.0,
        liquidity=500.0,
    )
    prepared = PreparedPredictionMarketSnapshots.from_snapshots([snapshot])
    features: dict[str, float | str] = {}

    _add_prediction_market_features(
        features,
        event=_event(event_time),
        prediction_markets=prepared,
        history_bars=history,
    )

    assert features["pm_kalshi_15m_lead_seconds"] == 300.0
    assert features["pm_kalshi_15m_has_usable_lead"] == 1.0
    assert features["pm_kalshi_15m_contract_elapsed_return"] == pytest.approx(0.01)
    assert features["pm_kalshi_15m_side_contract_elapsed_bps"] == pytest.approx(100.0)
    assert features["pm_kalshi_15m_side_mid_minus_elapsed_direction"] == pytest.approx(-0.38)
    assert features["pm_kalshi_15m_residual_edge_per_remaining_hour"] == pytest.approx(-4.56)
    assert features["pm_kalshi_15m_volume_log"] > 0
    assert features["pm_kalshi_15m_volume_24h_to_liquidity"] == pytest.approx(0.5)
    assert features["pm_leading_available_count"] == 1.0
    assert features["pm_leading_side_aligned_mid_max"] == 0.62
    assert features["pm_leading_residual_edge_mean"] == pytest.approx(-0.38)
    assert features["pm_leading_side_aligned_mid_liquidity_weighted"] == pytest.approx(0.62)


def test_prediction_market_stable_profile_keeps_aggregates_without_raw_lead_features() -> None:
    event_time = datetime(2026, 5, 2, 20, 10, tzinfo=UTC)
    window_start = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)
    history = [
        Bar(
            timestamp_utc=window_start + timedelta(minutes=minute),
            symbol="BTC/USD",
            bar_size="1m",
            open=100.0,
            high=101.5,
            low=99.5,
            close=100.0 + minute * 0.1,
            volume=1.0,
            source="IBKR:AGGTRADES",
        )
        for minute in range(11)
    ]
    snapshot = PredictionMarketSnapshot(
        provider="kalshi",
        duration="15m",
        timestamp_utc=event_time,
        market_id="kxbtc15m",
        market_slug="kxbtc15m",
        market_title="BTC price up in next 15 mins?",
        condition_id="event",
        window_start_utc=window_start,
        window_end_utc=window_start + timedelta(minutes=15),
        up_mid=0.62,
        down_mid=0.38,
        volume=100.0,
        liquidity=500.0,
    )
    prepared = PreparedPredictionMarketSnapshots.from_snapshots([snapshot])
    features: dict[str, float | str] = {}

    _add_prediction_market_features(
        features,
        event=_event(event_time),
        prediction_markets=prepared,
        history_bars=history,
        feature_profile="stable",
    )

    assert "pm_kalshi_15m_lead_seconds" not in features
    assert "pm_kalshi_15m_contract_elapsed_return" not in features
    assert "pm_kalshi_15m_volume_log" not in features
    assert features["pm_leading_available_count"] == 1.0
    assert features["pm_leading_residual_edge_mean"] == pytest.approx(-0.38)


def test_dataset_prediction_market_asof_can_match_signal_or_entry_bar() -> None:
    start = datetime(2026, 5, 2, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=15 * idx),
            symbol="BTCUSDT",
            bar_size="15m",
            open=100.0 + idx * 0.01,
            high=101.0 + idx * 0.01,
            low=99.0 + idx * 0.01,
            close=100.5 + idx * 0.01,
            volume=10.0 + idx,
            source="TEST",
        )
        for idx in range(305)
    ]
    signal_time = bars[299].timestamp_utc
    entry_time = bars[300].timestamp_utc
    snapshots = [
        PredictionMarketSnapshot(
            provider="kalshi",
            duration="15m",
            timestamp_utc=signal_time,
            market_id="signal",
            market_slug="signal",
            market_title="signal",
            condition_id="signal",
            window_start_utc=signal_time,
            window_end_utc=signal_time + timedelta(minutes=30),
            up_mid=0.61,
            down_mid=0.39,
        ),
        PredictionMarketSnapshot(
            provider="kalshi",
            duration="15m",
            timestamp_utc=entry_time,
            market_id="entry",
            market_slug="entry",
            market_title="entry",
            condition_id="entry",
            window_start_utc=signal_time,
            window_end_utc=signal_time + timedelta(minutes=30),
            up_mid=0.22,
            down_mid=0.78,
        ),
    ]
    candidate_config = CandidateGenerationConfig(
        mode="dense_research",
        min_history_bars=300,
        max_holding_hours=1,
        dense_stride_bars=1,
    )

    signal_samples = build_meta_label_samples(
        bars,
        config=AppConfig(),
        assumed_spread_bps=1.0,
        research_notional=10_000,
        prediction_market_snapshots=snapshots,
        feature_asof="signal",
        candidate_config=candidate_config,
    )
    entry_samples = build_meta_label_samples(
        bars,
        config=AppConfig(),
        assumed_spread_bps=1.0,
        research_notional=10_000,
        prediction_market_snapshots=snapshots,
        feature_asof="entry",
        candidate_config=candidate_config,
    )

    assert signal_samples[0].timestamp_utc == signal_time
    assert signal_samples[0].features["pm_kalshi_15m_up_mid"] == 0.61
    assert entry_samples[0].timestamp_utc == entry_time
    assert entry_samples[0].features["pm_kalshi_15m_up_mid"] == 0.22


def test_kalshi_loader_attempts_requested_5m_series() -> None:
    class FakeKalshi(KalshiPublicDataClient):
        def __init__(self) -> None:
            super().__init__()
            self.seen_series: list[str] = []

        def get_markets(self, **kwargs):  # type: ignore[no-untyped-def]
            self.seen_series.append(kwargs["series_ticker"])
            if kwargs["series_ticker"] != "KXBTC5M" or kwargs.get("status") != "open":
                return []
            return [
                {
                    "ticker": "KXBTC5M-26MAY022355-55",
                    "event_ticker": "KXBTC5M-26MAY022355",
                    "title": "BTC price up in next 5 mins?",
                    "open_time": "2026-05-03T03:55:00Z",
                    "close_time": "2026-05-03T04:00:00Z",
                    "liquidity_dollars": "1000",
                }
            ]

        def market_candlesticks(self, **kwargs):  # type: ignore[no-untyped-def]
            return [
                {
                    "end_period_ts": 1777780800,
                    "price": {"close_dollars": "0.57"},
                    "yes_bid": {"close_dollars": "0.56"},
                    "yes_ask": {"close_dollars": "0.58"},
                    "volume": "12",
                }
            ]

        def multiple_market_orderbooks(self, tickers):  # type: ignore[no-untyped-def]
            return {}

    client = FakeKalshi()

    snapshots, coverage = client.snapshots_from_btc_series(
        start=datetime(2026, 5, 3, 3, 55, tzinfo=UTC),
        end=datetime(2026, 5, 3, 4, 0, tzinfo=UTC),
        max_markets=10,
        durations=("5m",),
    )

    assert "KXBTC5M" in client.seen_series
    assert coverage["series"]["KXBTC5M"]["markets"] == 1
    assert len(snapshots) == 1
    assert snapshots[0].duration == "5m"
    assert snapshots[0].up_mid == 0.57


def test_kalshi_unsupported_duration_coverage_includes_45m() -> None:
    class FakeKalshi(KalshiPublicDataClient):
        def get_markets(self, **kwargs):  # type: ignore[no-untyped-def]
            return []

        def multiple_market_orderbooks(self, tickers):  # type: ignore[no-untyped-def]
            return {}

    _, coverage = FakeKalshi().snapshots_from_btc_series(
        start=datetime(2026, 5, 3, 3, 55, tzinfo=UTC),
        end=datetime(2026, 5, 3, 4, 0, tzinfo=UTC),
        max_markets=10,
        durations=("5m", "45m", "2h"),
    )

    assert coverage["unsupported_durations"] == ["45m", "2h"]


def test_kalshi_ladder_markets_are_separate_feature_duration() -> None:
    class FakeKalshi(KalshiPublicDataClient):
        def get_markets(self, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["series_ticker"] != "KXBTC":
                return []
            return [
                {
                    "ticker": "KXBTC-26MAY03-T120000-B100000",
                    "event_ticker": "KXBTC-26MAY03-T120000",
                    "title": "BTC above a threshold",
                    "open_time": "2026-05-03T03:00:00Z",
                    "close_time": "2026-05-03T04:00:00Z",
                    "updated_time": "2026-05-03T03:30:00Z",
                    "yes_bid_dollars": "0.55",
                    "yes_ask_dollars": "0.57",
                }
            ]

        def multiple_market_orderbooks(self, tickers):  # type: ignore[no-untyped-def]
            return {}

    snapshots, coverage = FakeKalshi().snapshots_from_btc_series(
        start=datetime(2026, 5, 3, 3, 0, tzinfo=UTC),
        end=datetime(2026, 5, 3, 4, 0, tzinfo=UTC),
        max_markets=10,
        durations=("1h",),
    )

    assert coverage["unsupported_durations"] == []
    assert snapshots[0].duration == "ladder_1h"


def test_prediction_market_snapshot_features_include_orderbook_depth() -> None:
    event = _event(datetime(2026, 5, 3, 3, 58, tzinfo=UTC))
    snapshot = PredictionMarketSnapshot(
        provider="polymarket",
        duration="5m",
        timestamp_utc=datetime(2026, 5, 3, 3, 57, tzinfo=UTC),
        market_id="m",
        market_slug="btc-updown-5m",
        market_title="BTC up or down",
        condition_id="c",
        window_start_utc=datetime(2026, 5, 3, 3, 55, tzinfo=UTC),
        window_end_utc=datetime(2026, 5, 3, 4, 0, tzinfo=UTC),
        up_bid=0.55,
        up_ask=0.57,
        up_mid=0.56,
        down_mid=0.44,
        up_bid_depth_5c=100.0,
        up_ask_depth_5c=50.0,
        orderbook_imbalance_5c=1 / 3,
    )
    features: dict[str, float | str] = {}

    _add_prediction_market_features(
        features,
        event=event,
        prediction_markets=PreparedPredictionMarketSnapshots.from_snapshots([snapshot]),
        history_bars=[],
        feature_profile="full",
    )

    assert features["pm_polymarket_5m_up_bid_depth_5c"] == 100.0
    assert features["pm_polymarket_5m_orderbook_imbalance_5c"] == pytest.approx(1 / 3)
    assert features["pm_polymarket_5m_side_book_imbalance_5c"] == pytest.approx(1 / 3)
