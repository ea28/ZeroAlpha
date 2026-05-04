from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.models.dataset import _add_cross_asset_features, _prepare_context_bars


def _bar(
    ts: datetime,
    close: float,
    *,
    symbol: str = "BTC/USD",
    volume: float = 1.0,
    trade_count: int | None = None,
    taker_buy_base_volume: float | None = None,
) -> Bar:
    extra = {}
    if taker_buy_base_volume is not None:
        extra["taker_buy_base_volume"] = taker_buy_base_volume
    return Bar(
        timestamp_utc=ts,
        symbol=symbol,
        bar_size="15m",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=volume,
        quote_volume=close * volume,
        trade_count=trade_count,
        vwap=close * 0.999,
        source="test",
        extra=extra,
    )


def test_cross_asset_features_measure_futures_basis_and_impulse() -> None:
    start = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)
    history = [
        _bar(start, 100.0),
        _bar(start + timedelta(minutes=15), 100.0),
        _bar(start + timedelta(minutes=30), 101.0),
    ]
    futures = [
        _bar(start, 100.0, symbol="MBT/USD", volume=10.0, trade_count=5, taker_buy_base_volume=5.0),
        _bar(
            start + timedelta(minutes=15),
            101.0,
            symbol="MBT/USD",
            volume=20.0,
            trade_count=10,
            taker_buy_base_volume=11.0,
        ),
        _bar(
            start + timedelta(minutes=30),
            103.02,
            symbol="MBT/USD",
            volume=40.0,
            trade_count=20,
            taker_buy_base_volume=28.0,
        ),
    ]
    event = CandidateEvent(
        event_id="event-1",
        timestamp_utc=start + timedelta(minutes=30),
        symbol="BTC/USD",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=101.0,
        max_holding_hours=4,
    )
    features: dict[str, float | str] = {
        "bar_close": 101.0,
        "return_1": 0.01,
        "return_elapsed_15m": 0.01,
    }

    _add_cross_asset_features(
        features,
        event=event,
        context_bars=_prepare_context_bars({"IBKR_MBT": futures}),
        history_bars=history,
    )

    assert features["ibkrmbt_basis_bps"] == pytest.approx(200.0)
    assert features["ibkrmbt_basis_change_bps_15m"] == pytest.approx(100.0)
    assert features["ibkrmbt_side_basis_change_bps_15m"] == pytest.approx(100.0)
    assert features["ibkrmbt_return_spread_bps_1"] == pytest.approx(100.0)
    assert features["ibkrmbt_side_return_spread_bps_15m"] == pytest.approx(100.0)
    assert features["ibkrmbt_dollar_volume_ratio_1"] == pytest.approx(2.04, rel=0.01)
    assert features["ibkrmbt_trade_count_ratio_1"] == pytest.approx(2.0)
    assert features["ibkrmbt_taker_buy_share"] == pytest.approx(0.7)
    assert features["ibkrmbt_side_taker_buy_imbalance"] == pytest.approx(0.4)
    assert features["ibkrmbt_taker_buy_share_delta_1"] == pytest.approx(0.15)
    assert features["ibkrmbt_vwap_distance_bps"] > 0


def test_derivatives_metric_context_is_not_treated_as_price_basis() -> None:
    start = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)
    history = [_bar(start + timedelta(minutes=15 * idx), 100.0 + idx) for idx in range(3)]
    open_interest = [
        _bar(start, 1_000.0, symbol="BTCUSDT"),
        _bar(start + timedelta(minutes=15), 1_100.0, symbol="BTCUSDT"),
        _bar(start + timedelta(minutes=30), 1_200.0, symbol="BTCUSDT"),
    ]
    funding = [
        _bar(start, 1.0001, symbol="BTCUSDT"),
        _bar(start + timedelta(minutes=15), 1.0002, symbol="BTCUSDT"),
        _bar(start + timedelta(minutes=30), 1.0003, symbol="BTCUSDT"),
    ]
    event = CandidateEvent(
        event_id="event-1",
        timestamp_utc=start + timedelta(minutes=30),
        symbol="BTC/USD",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=102.0,
        max_holding_hours=4,
    )
    features: dict[str, float | str] = {"bar_close": 102.0}

    _add_cross_asset_features(
        features,
        event=event,
        context_bars=_prepare_context_bars(
            {
                "BINANCE_UM_OPEN_INTEREST_BTCUSDT": open_interest,
                "BINANCE_UM_FUNDING_BTCUSDT": funding,
            }
        ),
        history_bars=history,
    )

    assert "binanceumopeninterestbtcusdt_basis_bps" not in features
    assert "binanceumfundingbtcusdt_basis_bps" not in features
    assert features["binanceumopeninterestbtcusdt_value_change_1"] == pytest.approx(100.0)
    assert features["binanceumfundingbtcusdt_value"] == pytest.approx(0.0003)


def test_cross_asset_features_add_eth_futures_spot_pair_basis() -> None:
    start = datetime(2026, 5, 2, 20, 0, tzinfo=UTC)
    history = [_bar(start + timedelta(minutes=15 * idx), 100.0 + idx) for idx in range(3)]
    eth_spot = [
        _bar(start, 2_000.0, symbol="ETH/USD", volume=100.0, trade_count=50),
        _bar(start + timedelta(minutes=15), 2_010.0, symbol="ETH/USD", volume=110.0, trade_count=55),
        _bar(start + timedelta(minutes=30), 2_020.0, symbol="ETH/USD", volume=130.0, trade_count=65),
    ]
    met_futures = [
        _bar(start, 2_010.0, symbol="MET/USD", volume=20.0, trade_count=10),
        _bar(start + timedelta(minutes=15), 2_035.0, symbol="MET/USD", volume=40.0, trade_count=20),
        _bar(start + timedelta(minutes=30), 2_060.0, symbol="MET/USD", volume=80.0, trade_count=40),
    ]
    event = CandidateEvent(
        event_id="event-eth",
        timestamp_utc=start + timedelta(minutes=30),
        symbol="BTC/USD",
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=102.0,
        max_holding_hours=4,
    )
    features: dict[str, float | str] = {"bar_close": 102.0}

    _add_cross_asset_features(
        features,
        event=event,
        context_bars=_prepare_context_bars(
            {
                "ETH_IBKR_SPOT": eth_spot,
                "ETH_IBKR_FUTURES_MET": met_futures,
            }
        ),
        history_bars=history,
    )

    prefix = "eth_futures_spot_ethibkrfuturesmet_vs_ethibkrspot"
    assert features[f"{prefix}_basis_bps"] == pytest.approx((2_060.0 / 2_020.0 - 1) * 10_000)
    assert features[f"{prefix}_basis_change_bps_15m"] == pytest.approx(
        ((2_060.0 / 2_020.0) - (2_035.0 / 2_010.0)) * 10_000
    )
    assert features[f"{prefix}_return_spread_bps_15m"] == pytest.approx(
        ((2_060.0 / 2_035.0 - 1) - (2_020.0 / 2_010.0 - 1)) * 10_000
    )
    assert features[f"{prefix}_futures_dollar_volume_share"] > 0.0
