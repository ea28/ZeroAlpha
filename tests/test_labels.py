from datetime import datetime, timedelta, UTC
import pytest

from zeroalpha.domain import Bar, CandidateEvent, Side
from zeroalpha.labels.triple_barrier import label_long_event, label_short_event
from zeroalpha.config import AppConfig, CostConfig, LabelConfig
from zeroalpha.models.dataset import label_geometry_diagnostics


def _bar(i: int, high: float, low: float, close: float) -> Bar:
    return Bar(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i),
        symbol="BTCUSDT",
        bar_size="1h",
        open=100,
        high=high,
        low=low,
        close=close,
        volume=1,
        source="TEST",
    )


def test_triple_barrier_conservative_same_bar() -> None:
    event = CandidateEvent(
        event_id="e1",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="test",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1,
        reference_price=100,
        max_holding_hours=24,
    )
    label = label_long_event(
        event,
        [_bar(1, high=103, low=97, close=101)],
        entry_price=100,
        net_profit_target=0.01,
        net_stop_loss=0.01,
        round_trip_cost_bps=0,
        conservative_same_bar=True,
    )
    assert label.outcome_type == "lower_same_bar"
    assert label.label == 0


def test_triple_barrier_targets_explicit_net_outcomes() -> None:
    event = CandidateEvent(
        event_id="e1",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="test",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1,
        reference_price=100,
        max_holding_hours=24,
    )
    label = label_long_event(
        event,
        [_bar(1, high=103, low=99, close=102)],
        entry_price=100,
        net_profit_target=0.02,
        net_stop_loss=0.02,
        round_trip_cost_bps=86,
        conservative_same_bar=True,
    )
    assert round(label.upper_barrier_price, 2) == 102.86
    assert round(label.net_return, 4) == 0.02
    assert label.label == 1


def test_triple_barrier_rejects_stop_smaller_than_cost() -> None:
    event = CandidateEvent(
        event_id="e1",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="test",
        side=Side.BUY,
        bar_size="1h",
        signal_strength=1,
        reference_price=100,
        max_holding_hours=24,
    )
    with pytest.raises(ValueError, match="net_stop_loss"):
        label_long_event(
            event,
            [_bar(1, high=101, low=99, close=100)],
            entry_price=100,
            net_profit_target=0.02,
            net_stop_loss=0.005,
            round_trip_cost_bps=86,
        )


def test_label_geometry_warns_when_gross_stop_is_tiny_after_costs() -> None:
    diagnostics = label_geometry_diagnostics(
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.01, net_stop_loss=0.01),
            cost=CostConfig(tier_rate=0.0018, minimum_commission=1.75, base_slippage_bps=5.0),
        ),
        assumed_spread_bps=10.0,
        research_notional=10_000,
    )

    assert round(diagnostics.round_trip_cost_bps, 1) == 76.0
    assert round(diagnostics.round_trip_commission_bps, 1) == 36.0
    assert round(diagnostics.spread_bps, 1) == 10.0
    assert round(diagnostics.slippage_bps, 1) == 20.0
    assert round(diagnostics.safety_margin_bps, 1) == 10.0
    assert round(diagnostics.gross_profit_move, 4) == 0.0176
    assert round(diagnostics.gross_stop_distance, 4) == 0.0024
    assert diagnostics.warning == "gross_stop_distance_below_50_bps"


def test_label_geometry_supports_per_contract_futures_costs() -> None:
    diagnostics = label_geometry_diagnostics(
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.01, net_stop_loss=0.01),
            cost=CostConfig(
                tier_rate=0.0018,
                minimum_commission=1.75,
                base_slippage_bps=0.25,
                safety_margin_bps=1.0,
                futures_fee_per_contract=2.02,
                futures_contract_multiplier=0.1,
            ),
        ),
        assumed_spread_bps=0.5,
        research_notional=10_000,
        reference_price=100_000,
    )

    assert round(diagnostics.round_trip_commission_bps, 2) == 4.04
    assert round(diagnostics.round_trip_cost_bps, 2) == 6.54


def test_short_triple_barrier_targets_explicit_net_outcomes() -> None:
    event = CandidateEvent(
        event_id="e1",
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="short_test",
        side=Side.SELL,
        bar_size="1h",
        signal_strength=1,
        reference_price=100,
        max_holding_hours=24,
    )
    label = label_short_event(
        event,
        [_bar(1, high=101, low=97, close=98)],
        entry_price=100,
        net_profit_target=0.02,
        net_stop_loss=0.02,
        round_trip_cost_bps=86,
        conservative_same_bar=True,
    )

    assert round(label.lower_barrier_price, 2) == 97.14
    assert round(label.net_return, 4) == 0.02
    assert label.label == 1
