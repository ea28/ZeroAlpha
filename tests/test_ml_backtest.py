from datetime import UTC, datetime, timedelta
from dataclasses import replace

import pytest

from zeroalpha.backtest.ml import (
    _confidence_notional_scale,
    _replay_exit_from_fill,
    _trade_excursions,
    run_ml_backtest,
)
from zeroalpha.config import AppConfig, LabelConfig, ModelConfig, RiskConfig
from zeroalpha.domain import Bar, Side, TripleBarrierLabel
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.ensemble import FoldPrediction, MetaLabelWalkForwardReport


def _bar(ts: datetime, *, high: float = 103.0, low: float = 99.0, close: float = 101.0) -> Bar:
    return Bar(
        timestamp_utc=ts,
        symbol="BTCUSDT",
        bar_size="1h",
        open=100.0,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        source="TEST",
    )


def _sample(event_id: str, ts: datetime, *, exit_hours: int = 6) -> MetaLabelSample:
    detail = TripleBarrierLabel(
        event_id=event_id,
        entry_timestamp_utc=ts + timedelta(hours=1),
        entry_price=100.0,
        upper_barrier_price=103.0,
        lower_barrier_price=98.0,
        vertical_barrier_timestamp_utc=ts + timedelta(hours=72),
        exit_timestamp_utc=ts + timedelta(hours=exit_hours),
        exit_price=103.0,
        outcome_type="upper",
        gross_return=0.03,
        net_return=0.02,
        label=1,
        t1=ts + timedelta(hours=exit_hours),
    )
    return MetaLabelSample(
        event_id=event_id,
        timestamp_utc=ts,
        t1=detail.t1,
        candidate_type="volatility_breakout",
        side="BUY",
        net_profit_target=0.02,
        net_stop_loss=0.02,
        features={"candidate_type": "volatility_breakout", "signal_strength": 1.0},
        label=1,
        net_return=detail.net_return,
        notional=10_000,
        round_trip_cost_bps=86.0,
        outcome_type=detail.outcome_type,
        label_detail=detail,
    )


def _prediction(event_id: str, ts: datetime) -> FoldPrediction:
    return FoldPrediction(
        fold_id=0,
        event_id=event_id,
        timestamp_utc=ts.isoformat(),
        candidate_type="volatility_breakout",
        label=1,
        probability=0.90,
        expected_value=0.016,
        should_trade=True,
        decision_reason="approved",
        net_return=0.02,
        pnl=200.0,
    )


def test_replayed_exit_respects_minimum_holding_seconds_after_fill() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    detail = TripleBarrierLabel(
        event_id="a",
        entry_timestamp_utc=start,
        entry_price=100.0,
        upper_barrier_price=101.0,
        lower_barrier_price=99.0,
        vertical_barrier_timestamp_utc=start + timedelta(seconds=10),
        exit_timestamp_utc=start + timedelta(seconds=10),
        exit_price=100.0,
        outcome_type="vertical",
        gross_return=0.0,
        net_return=0.0,
        label=0,
        t1=start + timedelta(seconds=10),
    )
    sample = MetaLabelSample(
        event_id="a",
        timestamp_utc=start,
        t1=detail.t1,
        candidate_type="dense_research_bar",
        side="BUY",
        net_profit_target=0.01,
        net_stop_loss=0.01,
        features={"event_min_holding_seconds": 5.0},
        label=0,
        net_return=0.0,
        notional=10_000,
        round_trip_cost_bps=0.0,
        outcome_type="vertical",
        label_detail=detail,
    )
    exit_bars = [
        _bar(start + timedelta(seconds=1), high=102.0, low=99.5, close=101.5),
        _bar(start + timedelta(seconds=5), high=100.5, low=98.0, close=98.5),
        _bar(start + timedelta(seconds=10), high=100.5, low=99.5, close=100.0),
    ]

    replayed = _replay_exit_from_fill(
        sample=sample,
        side=Side.BUY,
        fill_price=100.0,
        fill_timestamp=start,
        exit_bars=exit_bars,
        round_trip_cost_bps=0.0,
        conservative_same_bar=True,
    )

    assert replayed is not None
    assert replayed.timestamp_utc == start + timedelta(seconds=5)
    assert replayed.outcome_type == "lower_replay"


def test_trade_excursions_track_adverse_and_favorable_moves() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        _bar(start + timedelta(seconds=1), high=101.0, low=99.0),
        _bar(start + timedelta(seconds=2), high=103.0, low=97.5),
        _bar(start + timedelta(seconds=3), high=110.0, low=90.0),
    ]

    adverse, favorable = _trade_excursions(
        side=Side.BUY,
        entry_price=100.0,
        exit_bars=bars,
        exit_timestamp=start + timedelta(seconds=2),
    )

    assert adverse == pytest.approx(0.025)
    assert favorable == pytest.approx(0.03)


def test_ml_backtest_reports_configured_window_day_rates() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"requested_window": {"start": start.isoformat(), "end": (start + timedelta(days=3)).isoformat(), "span_days": 3.0}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 12)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)

    summary, _, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 1
    assert summary.configured_span_days == pytest.approx(3.0)
    assert summary.trades_per_configured_day == pytest.approx(1 / 3)
    assert summary.pnl_per_configured_day == pytest.approx(summary.net_pnl / 3)
    assert summary.daily_sharpe == summary.sharpe
    assert summary.multiple_testing_trials == 1
    assert summary.deflated_sharpe == summary.sharpe


def test_ml_backtest_market_entry_model_matches_live_runner_fill() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 10}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=98.0) for i in range(1, 12)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)
    cfg = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        risk=RiskConfig(minimum_fee_efficient_notional=100.0),
    )

    limit_summary, _, limit_rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=cfg,
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        entry_limit_offset_bps=500.0,
    )
    market_summary, market_trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=cfg,
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        entry_limit_offset_bps=500.0,
        entry_order_model="market",
    )

    assert limit_summary.trades == 0
    assert limit_rejections[0].reason == "missed_entry_fill"
    assert market_summary.trades == 1
    assert market_trades[0].entry_fill_price == pytest.approx(100.0)
    assert market_trades[0].entry_fill_timestamp_utc == start + timedelta(hours=1)


def test_ml_backtest_rejects_overlapping_model_signals() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start), _sample("b", start + timedelta(hours=2))]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[_prediction("a", start), _prediction("b", start + timedelta(hours=2))],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 10}},
    )
    bars = [
        _bar(start + timedelta(hours=i), high=101.0, low=99.0)
        for i in range(1, 12)
    ]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 1
    assert summary.sharpe == 0.0
    assert trades[0].event_id == "a"
    assert summary.reject_reasons["position_overlap"] == 1
    assert any(rejection.event_id == "b" for rejection in rejections)


def test_ml_backtest_return_first_mode_bypasses_static_probability_gate() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    sample = _sample("a", start)
    prediction = replace(
        _prediction("a", start),
        probability=0.10,
        expected_value=-0.01,
        predicted_return=0.02,
        selection_score=0.02,
        decision_reason="target_frequency_quota",
    )
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["histgb"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 10}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 12)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=[sample],
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        selection_score_mode="return_first",
    )

    assert summary.trades == 1
    assert trades[0].event_id == "a"
    assert all(rejection.reason != "probability_below_threshold" for rejection in rejections)


def test_ml_backtest_rejects_signals_below_expected_gross_edge_floor() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    sample = _sample("a", start)
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["histgb"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 10}},
    )
    bars = [_bar(start + timedelta(hours=i), high=103.0, low=99.0) for i in range(1, 12)]

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=[sample],
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        min_expected_gross_edge_bps=300.0,
    )

    assert summary.trades == 0
    assert trades == []
    assert rejections[0].reason == "expected_gross_edge_below_cost_floor"


def test_ml_backtest_allows_configured_multiple_open_positions() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start), _sample("b", start + timedelta(hours=2))]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[_prediction("a", start), _prediction("b", start + timedelta(hours=2))],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 10}},
    )
    bars = [
        _bar(start + timedelta(hours=i), high=101.0, low=99.0)
        for i in range(1, 12)
    ]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)
    bars[7] = _bar(start + timedelta(hours=8), high=103.0, low=99.0)

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0, max_open_positions=2),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 2
    assert [trade.event_id for trade in trades] == ["a", "b"]
    assert "position_overlap" not in summary.reject_reasons
    assert all(rejection.reason != "position_overlap" for rejection in rejections)


def test_ml_backtest_sizes_overlapping_entries_before_unrealized_exit_pnl() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start, exit_hours=6), _sample("b", start + timedelta(hours=2), exit_hours=8)]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[_prediction("a", start), _prediction("b", start + timedelta(hours=2))],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 12}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 14)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)
    bars[10] = _bar(start + timedelta(hours=11), high=103.0, low=99.0)

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(max_open_positions=2, minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 2
    assert [trade.notional for trade in trades] == [1750.0, 1750.0]
    assert trades[0].equity_after == 10035.0
    assert trades[1].equity_after == 10070.0


def test_ml_backtest_rejects_overlapping_spot_entries_without_cash_capacity() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start, exit_hours=6), _sample("b", start + timedelta(hours=2), exit_hours=8)]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[_prediction("a", start), _prediction("b", start + timedelta(hours=2))],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 12}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 14)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(risk_per_trade=1.0, max_open_positions=2, minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 1
    assert trades[0].notional == 10_000
    assert any(rejection.event_id == "b" and rejection.reason == "insufficient_cash" for rejection in rejections)


def test_ml_backtest_caps_total_open_spot_notional_at_paper_limit() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start, exit_hours=6), _sample("b", start + timedelta(hours=2), exit_hours=8)]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[_prediction("a", start), _prediction("b", start + timedelta(hours=2))],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 12}},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 14)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(
                risk_per_trade=1.0,
                paper_max_notional=10_000.0,
                max_open_positions=2,
                minimum_fee_efficient_notional=100.0,
            ),
        ),
        starting_equity=30_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 1
    assert trades[0].notional == 10_000
    assert any(rejection.event_id == "b" and rejection.reason == "insufficient_cash" for rejection in rejections)


def test_ml_backtest_can_stop_experiment_after_absolute_realized_loss() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("loss", start, exit_hours=2), _sample("next", start + timedelta(hours=3))]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[
            _prediction("loss", start),
            _prediction("next", start + timedelta(hours=3)),
        ],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 8}},
    )
    bars = [
        _bar(start + timedelta(hours=1), high=100.5, low=99.0, close=100.0),
        _bar(start + timedelta(hours=2), high=100.5, low=98.0, close=98.5),
        _bar(start + timedelta(hours=3), high=101.0, low=99.0, close=100.5),
        _bar(start + timedelta(hours=4), high=103.0, low=99.0, close=102.0),
        _bar(start + timedelta(hours=5), high=103.0, low=99.0, close=102.0),
    ]

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(
                risk_per_trade=1.0,
                paper_max_notional=10_000.0,
                minimum_fee_efficient_notional=100.0,
            ),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        experiment_max_loss_usd=100.0,
    )

    assert summary.trades == 1
    assert trades[0].event_id == "loss"
    assert trades[0].pnl < -100.0
    assert summary.reject_reasons["experiment_max_loss_usd_stop"] == 1
    assert rejections[-1].event_id == "next"


def test_ml_backtest_zero_consecutive_loss_limit_disables_cooldown() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [
        _sample(f"loss-{idx}", start + timedelta(hours=idx * 3), exit_hours=2)
        for idx in range(4)
    ]
    report = MetaLabelWalkForwardReport(
        samples=4,
        folds=[],
        predictions=[
            _prediction(f"loss-{idx}", start + timedelta(hours=idx * 3))
            for idx in range(4)
        ],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={"primary": {"source": "TEST", "bars": 14}},
    )
    bars = [
        _bar(start + timedelta(hours=hour), high=100.5, low=99.2, close=100.0)
        for hour in range(1, 14)
    ]
    for idx in range(4):
        bars[idx * 3 + 1] = _bar(
            start + timedelta(hours=idx * 3 + 2),
            high=100.5,
            low=98.0,
            close=98.5,
        )

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(
                consecutive_loss_limit=0,
                cooldown_hours_after_stopouts=24,
                daily_loss_stop=1.0,
                weekly_loss_stop=1.0,
                rolling_drawdown_stop=1.0,
                minimum_fee_efficient_notional=100.0,
            ),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 4
    assert [trade.event_id for trade in trades] == [
        "loss-0",
        "loss-1",
        "loss-2",
        "loss-3",
    ]
    assert all(trade.pnl < 0 for trade in trades)
    assert "cooldown" not in summary.reject_reasons
    assert all(rejection.reason != "cooldown" for rejection in rejections)


def test_ml_backtest_rejects_negative_absolute_experiment_stop() -> None:
    report = MetaLabelWalkForwardReport(
        samples=0,
        folds=[],
        predictions=[],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )

    with pytest.raises(ValueError, match="experiment_max_loss_usd"):
        run_ml_backtest(
            report=report,
            samples=[],
            bars=[],
            config=AppConfig(),
            starting_equity=10_000,
            requested_notional=10_000,
            assumed_spread_bps=10.0,
            experiment_max_loss_usd=-1.0,
        )


def test_ml_backtest_replays_exit_from_actual_fill_instead_of_label_exit() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [
        _bar(start + timedelta(hours=1), high=100.5, low=99.0, close=100.0),
        _bar(start + timedelta(hours=2), high=100.5, low=98.0, close=98.5),
        _bar(start + timedelta(hours=3), high=103.0, low=98.5, close=102.0),
    ]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert summary.trades == 1
    assert trades[0].outcome_type == "lower_replay"
    assert trades[0].pnl < 0


def test_ml_backtest_reports_missed_entry_fills() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 4)]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        entry_limit_offset_bps=200.0,
    )

    assert trades == []
    assert summary.missed_fills == 1


def test_ml_backtest_rejects_wide_spread_before_sizing() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=100.0,
    )

    assert trades == []
    assert summary.trades == 0
    assert summary.reject_reasons["spread_too_wide"] == 1
    assert rejections[0].reason == "spread_too_wide"


def test_ml_backtest_rejects_spot_crypto_short_samples() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [replace(_sample("a", start), side="SELL")]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[_prediction("a", start)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )

    assert trades == []
    assert summary.trades == 0
    assert summary.reject_reasons["spot_short_not_executable"] == 1
    assert rejections[0].reason == "spot_short_not_executable"


def test_ml_backtest_allows_spot_short_only_for_research_backtest() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [replace(_sample("a", start), side="SELL")]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[replace(_prediction("a", start), side="SELL")],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    blocked_summary, blocked_trades, blocked_rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
    )
    research_summary, research_trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
        allow_research_short_backtest=True,
    )

    assert blocked_summary.trades == 0
    assert blocked_trades == []
    assert blocked_rejections[0].reason == "spot_short_not_executable"
    assert research_summary.trades == 1
    assert research_trades[0].side == "SELL"


def test_ml_backtest_research_gate_rejects_negative_ev_by_default() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = _prediction("a", start)
    prediction = replace(prediction, probability=0.20, expected_value=-0.01)
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    production_summary, _, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )
    research_summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
    )

    assert production_summary.trades == 0
    assert research_summary.trades == 0
    assert trades == []
    assert rejections[0].reason == "negative_ev_research_gate"


def test_ml_backtest_trusts_candidate_type_calibrated_research_signal() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(
        _prediction("a", start),
        expected_value=-0.01,
        decision_reason="candidate_type_calibration",
    )
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
    )

    assert summary.trades == 1
    assert trades[0].event_id == "a"


def test_ml_backtest_rejects_negative_utility_target_frequency_by_default() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(
        _prediction("a", start),
        expected_value=-0.01,
        decision_reason="target_frequency_rank",
        selection_score=-0.001,
    )
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
    )

    assert summary.trades == 0
    assert trades == []
    assert rejections[0].reason == "negative_ev_research_gate"


def test_ml_backtest_accepts_positive_utility_target_frequency_research_signal() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(
        _prediction("a", start),
        expected_value=0.001,
        decision_reason="target_frequency_rank",
        selection_score=0.001,
    )
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
    )

    assert summary.trades == 1
    assert trades[0].event_id == "a"


def test_ml_backtest_can_run_explicit_negative_ev_frequency_probe() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(_prediction("a", start), probability=0.20, expected_value=-0.01)
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    research_summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        enforce_production_gate=False,
        allow_negative_ev_research=True,
    )

    assert research_summary.trades == 1
    assert trades[0].event_id == "a"


def test_confidence_scaled_sizing_reduces_notional_without_exceeding_caps() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(_prediction("a", start), probability=0.61, expected_value=0.001)
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        confidence_scaled_sizing=True,
    )

    assert summary.trades == 1
    assert 0 < trades[0].notional < 10_000


def test_confidence_scaled_sizing_keeps_valid_signal_fee_efficient() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    prediction = replace(_prediction("a", start), probability=0.61, expected_value=0.001)
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[prediction],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i)) for i in range(1, 8)]

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        confidence_scaled_sizing=True,
    )

    assert summary.trades == 1
    assert trades[0].notional >= 1_000
    assert trades[0].notional < 10_000


def test_confidence_sizing_trusts_strong_candidate_type_calibration() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    prediction = replace(
        _prediction("a", start),
        probability=0.16,
        expected_value=-0.001,
        decision_reason="target_frequency_rank",
    )

    scale = _confidence_notional_scale(
        prediction,
        AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        type_threshold={
            "source": "candidate_type_calibration",
            "average_trade_return": 0.004,
            "utility_floor": 0.003,
            "hit_rate": 0.60,
            "traded_signals": 5,
        },
    )

    assert scale == 1.0


def test_confidence_sizing_uses_intraday_label_geometry_for_ev_floor() -> None:
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    sample = replace(_sample("a", ts), net_profit_target=0.001, net_stop_loss=0.001)
    prediction = replace(_prediction("a", ts), probability=0.80, expected_value=0.0004)

    scale = _confidence_notional_scale(
        prediction,
        AppConfig(
            labels=LabelConfig(net_profit_target=0.001, net_stop_loss=0.001),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        sample=sample,
    )

    assert scale > 0.60


def test_score_bucket_sizing_allocates_more_to_high_probability_signals() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("low", start), _sample("high", start + timedelta(hours=8))]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[
            replace(_prediction("low", start), probability=0.40),
            replace(_prediction("high", start + timedelta(hours=8)), probability=0.95),
        ],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 18)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)
    bars[13] = _bar(start + timedelta(hours=14), high=103.0, low=99.0)

    summary, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.0, minimum_expected_value=0.0),
            risk=RiskConfig(
                account_equity=30_000,
                risk_per_trade=1.0,
                paper_max_notional=15_000,
                max_open_positions=2,
                minimum_fee_efficient_notional=100.0,
            ),
        ),
        starting_equity=30_000,
        requested_notional=5_000,
        assumed_spread_bps=10.0,
        sizing_mode="score_bucket",
        sizing_score_field="probability",
        sizing_base_notional=5_000,
        sizing_mid_notional=10_000,
        sizing_high_notional=15_000,
        sizing_mid_score=0.50,
        sizing_high_score=0.90,
    )

    assert summary.trades == 2
    assert [trade.notional for trade in trades] == [5_000, 15_000]
    assert [trade.sizing_mode for trade in trades] == ["score_bucket", "score_bucket"]


def test_score_bucket_sizing_can_allocate_more_to_lower_scores() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("low", start), _sample("high", start + timedelta(hours=8))]
    report = MetaLabelWalkForwardReport(
        samples=2,
        folds=[],
        predictions=[
            replace(_prediction("low", start), expected_value=-0.012, selection_score=-0.012),
            replace(_prediction("high", start + timedelta(hours=8)), expected_value=-0.002, selection_score=-0.002),
        ],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [_bar(start + timedelta(hours=i), high=101.0, low=99.0) for i in range(1, 18)]
    bars[5] = _bar(start + timedelta(hours=6), high=103.0, low=99.0)
    bars[13] = _bar(start + timedelta(hours=14), high=103.0, low=99.0)

    _, trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.0, minimum_expected_value=-0.02),
            risk=RiskConfig(
                account_equity=30_000,
                risk_per_trade=1.0,
                paper_max_notional=15_000,
                max_open_positions=2,
                minimum_fee_efficient_notional=100.0,
            ),
        ),
        starting_equity=30_000,
        requested_notional=15_000,
        assumed_spread_bps=10.0,
        sizing_mode="score_bucket",
        sizing_score_field="expected_value",
        sizing_score_direction="low",
        sizing_base_notional=5_000,
        sizing_mid_notional=10_000,
        sizing_high_notional=15_000,
        sizing_mid_score=0.005,
        sizing_high_score=0.010,
    )

    assert [trade.notional for trade in trades] == [15_000, 5_000]


def test_dynamic_exit_overlay_can_exit_weak_trade_before_late_upper_hit() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[replace(_prediction("a", start), probability=0.80, predicted_return=-0.01)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [
        _bar(start + timedelta(hours=1), high=100.2, low=99.8, close=100.0),
        _bar(start + timedelta(hours=2), high=100.3, low=99.6, close=99.6),
        _bar(start + timedelta(hours=3), high=100.5, low=99.4, close=99.7),
        _bar(start + timedelta(hours=4), high=100.7, low=99.5, close=100.0),
        _bar(start + timedelta(hours=5), high=101.0, low=99.8, close=100.7),
        _bar(start + timedelta(hours=6), high=103.0, low=100.0, close=102.9),
    ]

    baseline_summary, baseline_trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
    )
    overlay_summary, overlay_trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        dynamic_exit_overlay=True,
        dynamic_exit_checkpoints_minutes=(60,),
        dynamic_exit_adverse_bps=10.0,
        dynamic_exit_weak_probability=0.95,
    )

    assert baseline_summary.trades == 1
    assert baseline_trades[0].outcome_type == "upper_replay"
    assert overlay_summary.trades == 1
    assert overlay_trades[0].outcome_type.startswith("dynamic_exit_adverse_60m")
    assert overlay_trades[0].exit_timestamp_utc < baseline_trades[0].exit_timestamp_utc


def test_dynamic_exit_overlay_can_use_second_level_checkpoints() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [_sample("a", start)]
    report = MetaLabelWalkForwardReport(
        samples=1,
        folds=[],
        predictions=[replace(_prediction("a", start), probability=0.80, predicted_return=-0.01)],
        feature_names=[],
        requested_models=["logistic"],
        calibration_method="sigmoid",
        stacker_mode="average",
        data_coverage={},
    )
    bars = [
        _bar(start + timedelta(hours=1), high=100.0, low=99.98, close=99.99),
        _bar(start + timedelta(hours=1, seconds=1), high=100.0, low=99.98, close=99.99),
        _bar(start + timedelta(hours=1, seconds=2), high=100.0, low=99.7, close=99.7),
        _bar(start + timedelta(hours=1, seconds=3), high=100.0, low=99.7, close=99.8),
        _bar(start + timedelta(hours=6), high=103.0, low=99.8, close=102.9),
    ]

    _, overlay_trades, _ = run_ml_backtest(
        report=report,
        samples=samples,
        bars=bars,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
            risk=RiskConfig(minimum_fee_efficient_notional=100.0),
        ),
        starting_equity=10_000,
        requested_notional=10_000,
        assumed_spread_bps=10.0,
        dynamic_exit_overlay=True,
        dynamic_exit_checkpoints_minutes=(),
        dynamic_exit_checkpoints_seconds=(2,),
        dynamic_exit_adverse_bps=10.0,
        dynamic_exit_weak_probability=0.95,
    )

    assert overlay_trades[0].outcome_type.startswith("dynamic_exit_adverse_2s")
    assert overlay_trades[0].exit_timestamp_utc == start + timedelta(hours=1, seconds=2)
