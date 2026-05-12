from argparse import Namespace
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import zeroalpha.cli as cli
from zeroalpha.cli import (
    _context_quality_or_raise,
    _effective_respect_open_positions,
    _filter_samples_from_args,
    _kill_switch_enabled,
    _load_research_bars,
    _quality_or_raise,
    _require_verified_one_second_data,
    _runner_exit_timing,
    _runner_min_holding_seconds,
    _strict_live_valid_1s_diagnostics,
    _trade_run_quote,
    _validate_paper_order_test_config,
    _validate_paper_test_config,
    _validate_research_gated_args,
    _validate_research_short_backtest_args,
    _validate_round_trip_test_config,
    _validate_trade_run_config,
    _warmup_live_one_second_stream,
)
from zeroalpha.config import AppConfig, BrokerConfig, ModelConfig, RiskConfig, RuntimeConfig
from zeroalpha.data.external.ibkr_bars import write_ibkr_bars
from zeroalpha.data.quality import validate_bars
from zeroalpha.domain import Bar, MarketQuote, RuntimeMode, TripleBarrierLabel
from zeroalpha.models.dataset import MetaLabelSample


def _meta_sample(name: str) -> MetaLabelSample:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    detail = TripleBarrierLabel(
        event_id=name,
        entry_timestamp_utc=timestamp + timedelta(minutes=15),
        entry_price=100,
        upper_barrier_price=101,
        lower_barrier_price=99,
        vertical_barrier_timestamp_utc=timestamp + timedelta(hours=2),
        exit_timestamp_utc=timestamp + timedelta(hours=1),
        exit_price=101,
        outcome_type="upper",
        gross_return=0.01,
        net_return=0.005,
        label=1,
        t1=timestamp + timedelta(hours=1),
    )
    return MetaLabelSample(
        event_id=name,
        timestamp_utc=timestamp,
        t1=detail.t1,
        candidate_type="dense_research_bar",
        side="BUY",
        net_profit_target=0.005,
        net_stop_loss=0.005,
        features={"event_setup_family": name},
        label=1,
        net_return=0.005,
        notional=1_000,
        round_trip_cost_bps=10,
        outcome_type="upper",
        label_detail=detail,
    )


def _paper_order_args(**updates) -> Namespace:
    values = {"confirm": "PAPER_ORDER_TEST", "notional": 100.0, "offset_bps": 100.0}
    values.update(updates)
    return Namespace(**values)


def _paper_test_args(**updates) -> Namespace:
    values = {
        "confirm": "IBKR_PAPER_TEST",
        "duration_seconds": 600.0,
        "interval_seconds": 30.0,
        "max_cash_usd": 10_000.0,
        "max_loss_usd": 1_000.0,
        "submit_order": False,
        "order_notional": 100.0,
        "order_offset_bps": 100.0,
    }
    values.update(updates)
    return Namespace(**values)


def _round_trip_args(**updates) -> Namespace:
    values = {
        "confirm": "IBKR_ROUND_TRIP_TEST",
        "notional": 100.0,
        "hold_seconds": 10.0,
        "synthetic_stop_loss_bps": 100.0,
        "monitor_interval_seconds": 1.0,
        "order_timeout_seconds": 30.0,
        "commission_wait_seconds": 2.0,
        "max_cash_usd": 10_000.0,
        "max_loss_usd": 1_000.0,
    }
    values.update(updates)
    return Namespace(**values)


def _trade_run_args(**updates) -> Namespace:
    values = {
        "model_artifact": "artifacts/models/prod.joblib",
        "capital_usd": 5_000.0,
        "max_loss_usd": 250.0,
        "max_order_notional_usd": 5_000.0,
        "duration_seconds": 600.0,
        "signal_interval": 60.0,
        "synthetic_stop_loss_bps": 100.0,
        "confirm": "IBKR_PAPER_TRADE_RUN",
    }
    values.update(updates)
    return Namespace(**values)


def _last_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _paper_cfg_with_account() -> AppConfig:
    return replace(AppConfig(), broker=replace(AppConfig().broker, account="DU123456"))


def test_paper_order_test_requires_explicit_confirmation() -> None:
    cfg = AppConfig()

    with pytest.raises(SystemExit, match="PAPER_ORDER_TEST"):
        _validate_paper_order_test_config(cfg, _paper_order_args(confirm=""))

    _validate_paper_order_test_config(_paper_cfg_with_account(), _paper_order_args())


def test_paper_order_test_enforces_config_notional_cap() -> None:
    cfg = replace(AppConfig(), risk=replace(AppConfig().risk, paper_max_notional=50.0))

    with pytest.raises(SystemExit, match="paper_max_notional"):
        _validate_paper_order_test_config(cfg, _paper_order_args(notional=100.0))


def test_broker_paper_test_parser_and_limits_are_available() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "broker",
            "paper-test",
            "--duration-seconds",
            "600",
            "--max-cash-usd",
            "10000",
            "--max-loss-usd",
            "1000",
            "--submit-order",
            "--order-notional",
            "100",
            "--confirm",
            "IBKR_PAPER_TEST",
        ]
    )

    assert args.broker_command == "paper-test"
    assert args.max_cash_usd == 10_000
    assert args.max_loss_usd == 1_000
    assert args.submit_order is True
    _validate_paper_test_config(_paper_cfg_with_account(), args)


def test_broker_paper_test_rejects_nonpaper_and_excess_cash() -> None:
    live_cfg = replace(
        AppConfig(),
        runtime=RuntimeConfig(
            mode=RuntimeMode.LIVE,
            enable_live_trading=True,
            live_confirmation="ZEROALPHA_LIVE",
        ),
        broker=BrokerConfig(port=4001),
    )
    capped_cfg = replace(AppConfig(), risk=replace(AppConfig().risk, paper_max_notional=5_000.0))

    with pytest.raises(SystemExit, match="paper-only"):
        _validate_paper_test_config(live_cfg, _paper_test_args())
    with pytest.raises(SystemExit, match="paper_max_notional"):
        _validate_paper_test_config(capped_cfg, _paper_test_args(max_cash_usd=10_000.0))


def test_paper_test_loss_delta_falls_back_to_net_liquidation() -> None:
    baseline = {
        "pnl": [{"daily_pnl": None}],
        "portfolio": [],
        "account_summary": [{"tag": "NetLiquidation", "value": "10000.00"}],
    }
    current = {
        "pnl": [{"daily_pnl": None}],
        "portfolio": [],
        "account_summary": [{"tag": "NetLiquidation", "value": "9875.25"}],
    }

    assert cli._snapshot_loss_delta(baseline=baseline, current=current) == pytest.approx(124.75)


def test_round_trip_session_state_captures_trades_fills_and_executions() -> None:
    execution = SimpleNamespace(
        execId="exec-1",
        time=datetime(2026, 1, 1, tzinfo=UTC),
        acctNumber="DU123",
        exchange="PAXOS",
        side="BOT",
        shares=0.01,
        price=50_000.0,
        permId=99,
        clientId=1,
        orderId=7,
        cumQty=0.01,
        avgPrice=50_000.0,
        lastLiquidity=1,
    )
    commission_report = SimpleNamespace(
        execId="exec-1",
        commission=1.0,
        currency="USD",
        realizedPNL=-1.0,
    )
    fill = SimpleNamespace(
        time=datetime(2026, 1, 1, tzinfo=UTC),
        execution=execution,
        commissionReport=commission_report,
    )
    older_execution = SimpleNamespace(**{**execution.__dict__, "execId": "exec-old", "permId": 100})
    older_fill = SimpleNamespace(
        time=datetime(2026, 1, 1, tzinfo=UTC),
        execution=older_execution,
        commissionReport=SimpleNamespace(
            execId="exec-old",
            commission=2.0,
            currency="USD",
            realizedPNL=-2.0,
        ),
    )
    trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=7,
            permId=99,
            action="BUY",
            orderType="MKT",
            lmtPrice=0.0,
            auxPrice=0.0,
            totalQuantity=0.01,
            cashQty=500.0,
        ),
        orderStatus=SimpleNamespace(
            status="Filled",
            filled=0.01,
            remaining=0.0,
            avgFillPrice=50_000.0,
        ),
        fills=[fill],
        log=[],
    )
    older_trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=7,
            permId=100,
            action="BUY",
            orderType="MKT",
            lmtPrice=0.0,
            auxPrice=0.0,
            totalQuantity=0.01,
            cashQty=500.0,
        ),
        orderStatus=SimpleNamespace(
            status="Filled",
            filled=0.01,
            remaining=0.0,
            avgFillPrice=50_000.0,
        ),
        fills=[older_fill],
        log=[],
    )
    ib = SimpleNamespace(
        trades=lambda: [older_trade, trade],
        openTrades=lambda: [],
        fills=lambda: [older_fill, fill],
        executions=lambda: [older_execution, execution],
        openOrders=lambda: [],
    )
    broker = SimpleNamespace(_ib=ib)

    payload = cli._broker_session_state_payload(
        broker,
        related_order_ids={7},
        related_perm_ids={99},
    )

    assert payload["trade_count"] == 2
    assert payload["open_trade_count"] == 0
    assert payload["fill_count"] == 2
    assert payload["execution_count"] == 2
    assert payload["related_order_ids"] == [7]
    assert payload["related_perm_ids"] == [99]
    assert payload["related_trade_count"] == 1
    assert payload["related_fill_count"] == 1
    assert payload["related_execution_count"] == 1
    assert payload["fills"][1]["commission_report"]["commission"] == 1.0
    assert payload["executions"][1]["exec_id"] == "exec-1"
    assert payload["related_fills"][0]["commission_report"]["commission"] == 1.0


def test_round_trip_fill_payload_marks_missing_tws_commission_reports() -> None:
    execution = SimpleNamespace(
        execId="exec-missing",
        time=datetime(2026, 1, 1, tzinfo=UTC),
        acctNumber="DU123",
        exchange="PAXOS",
        side="BOT",
        shares=0.01,
        price=50_000.0,
        permId=99,
        clientId=1,
        orderId=7,
        cumQty=0.01,
        avgPrice=50_000.0,
        lastLiquidity=1,
    )
    fill = SimpleNamespace(
        time=datetime(2026, 1, 1, tzinfo=UTC),
        execution=execution,
        commissionReport=SimpleNamespace(execId="exec-missing"),
    )
    trade = SimpleNamespace(fills=[fill])

    payload = cli._trade_fill_payload(trade)

    assert payload["commission_report_complete"] is False
    assert payload["realized_pnl_report_complete"] is False
    assert payload["missing_commission_exec_ids"] == ["exec-missing"]
    assert payload["missing_realized_pnl_exec_ids"] == ["exec-missing"]
    with pytest.raises(SystemExit, match="missing TWS commission"):
        cli._require_tws_fill_accounting("test order", {"fill": payload})


def test_broker_round_trip_parser_and_limits_are_available() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "broker",
            "round-trip-test",
            "--notional",
            "100",
            "--hold-seconds",
            "10",
            "--synthetic-stop-loss-bps",
            "100",
            "--max-cash-usd",
            "10000",
            "--max-loss-usd",
            "1000",
            "--confirm",
            "IBKR_ROUND_TRIP_TEST",
        ]
    )

    assert args.broker_command == "round-trip-test"
    assert args.notional == 100.0
    assert args.hold_seconds == 10.0
    _validate_round_trip_test_config(_paper_cfg_with_account(), args)


def test_broker_runtime_streaming_flags_are_available() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "broker",
            "round-trip-test",
            "--no-stream-events",
            "--stream-format",
            "json",
            "--event-log",
            "tmp/events.jsonl",
            "--confirm",
            "IBKR_ROUND_TRIP_TEST",
        ]
    )

    assert args.stream_events is False
    assert args.stream_format == "json"
    assert args.event_log == "tmp/events.jsonl"


def test_broker_trade_run_parser_and_paper_safety_gates() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "broker",
            "trade-run",
            "--model-artifact",
            "artifacts/models/prod.joblib",
            "--capital-usd",
            "5000",
            "--max-loss-usd",
            "250",
            "--max-order-notional-usd",
            "5000",
            "--max-open-positions",
            "10",
            "--duration-seconds",
            "600",
            "--stream-format",
            "json",
            "--confirm",
            "IBKR_PAPER_TRADE_RUN",
        ]
    )

    assert args.broker_command == "trade-run"
    assert args.capital_usd == 5_000
    assert args.max_loss_usd == 250
    assert args.max_open_positions == 10
    assert args.live_data_mode == "streaming"
    assert args.require_live_1s_data is True
    assert args.tick_by_tick_type == "Last"
    assert args.history_what_to_show == "AGGTRADES"
    assert args.history_bar_size == "1 secs"
    assert args.history_duration == "1800 S"
    assert args.live_1s_warmup_bars == 2
    assert args.live_1s_warmup_timeout_seconds == 8.0
    assert args.max_signal_bar_age_seconds == 2.5
    assert args.candidate_mode == "dense_research"
    assert args.dense_stride_bars == 1
    assert args.max_scoring_samples == 1
    assert args.context_bars_jsonl == ""
    assert args.decision_threshold == 0.0
    assert args.max_missing_model_feature_fraction == 0.0
    assert args.stream_format == "json"
    cfg = cli._override_config_from_args(_paper_cfg_with_account(), args)
    assert cfg.risk.max_open_positions == 10
    _validate_trade_run_config(cfg, args)


def test_easy_train_preset_expands_to_trade_runnable_artifact_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "easy",
            "train",
            "--artifact",
            "artifacts/models/prod.joblib",
            "--no-hpo",
            "--no-explain",
        ]
    )

    command = cli._build_easy_train_argv(args)

    assert command[:2] == ["model", "train-meta"]
    assert _last_value(command, "--config") == "configs/live.example.toml"
    assert _last_value(command, "--save-artifact") == "artifacts/models/prod.joblib"
    assert _last_value(command, "--starting-equity") == "10000.0"
    assert _last_value(command, "--entry-order-model") == "market"
    assert _last_value(command, "--sizing-mode") == "score_bucket"
    assert _last_value(command, "--sizing-score-field") == "selection_score"
    assert _last_value(command, "--target-frequency-mode") == "online"
    assert _last_value(command, "--capacity-release-mode") == "planned"
    assert "--adaptive-horizon" in command
    assert "--hpo" not in command
    assert "--permutation-importance" not in command


def test_easy_backtest_preset_keeps_short_command_surface() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "easy",
            "backtest",
            "--years",
            "2",
            "--capital-usd",
            "20000",
            "--high-notional-usd",
            "7500",
            "--no-shap",
        ]
    )

    command = cli._build_easy_backtest_argv(args)

    assert command[:2] == ["backtest", "ml"]
    assert _last_value(command, "--years") == "2"
    assert _last_value(command, "--starting-equity") == "20000.0"
    assert _last_value(command, "--notional") == "7500.0"
    assert _last_value(command, "--models") == cli.PRODUCTION_MODEL_SET
    assert "--hpo" in command
    assert "--permutation-importance" in command
    assert "--shap-importance" not in command


def test_easy_trade_preset_expands_to_broker_trade_run() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "easy",
            "trade",
            "--artifact",
            "artifacts/models/prod.joblib",
            "--account",
            "DU123",
            "--capital-usd",
            "5000",
            "--max-loss-usd",
            "250",
            "--max-order-notional-usd",
            "1000",
            "--confirm",
            "IBKR_PAPER_TRADE_RUN",
        ]
    )

    command = cli._build_easy_trade_argv(args)

    assert command[:2] == ["broker", "trade-run"]
    assert _last_value(command, "--config") == "configs/live.example.toml"
    assert _last_value(command, "--model-artifact") == "artifacts/models/prod.joblib"
    assert _last_value(command, "--account") == "DU123"
    assert _last_value(command, "--capital-usd") == "5000.0"
    assert _last_value(command, "--max-order-notional-usd") == "1000.0"
    assert _last_value(command, "--stream-format") == "json"
    assert _last_value(command, "--confirm") == "IBKR_PAPER_TRADE_RUN"


def test_trade_run_order_notional_uses_artifact_score_bucket_policy() -> None:
    args = _trade_run_args(max_order_notional_usd=10_000.0)
    training_config = {
        "sizing_policy": {
            "mode": "score_bucket",
            "score_field": "selection_score",
            "score_direction": "high",
            "base_notional": 1_250.0,
            "mid_notional": 2_500.0,
            "high_notional": 5_000.0,
            "mid_score": 0.30,
            "high_score": 0.75,
        }
    }
    sample = SimpleNamespace(features={})
    score = SimpleNamespace(probability=0.55, expected_value=0.0, selection_score=0.80)

    notional, source = cli._trade_run_order_notional_from_artifact_policy(
        args,
        training_config,
        sample,
        score,
        available_notional=8_000.0,
    )

    assert notional == pytest.approx(5_000.0)
    assert source == "artifact_score_bucket_high"


def test_trade_run_order_notional_can_use_low_score_bucket_policy() -> None:
    args = _trade_run_args(max_order_notional_usd=10_000.0)
    training_config = {
        "sizing_policy": {
            "mode": "score_bucket",
            "score_field": "expected_value",
            "score_direction": "low",
            "base_notional": 1_500.0,
            "mid_notional": 2_500.0,
            "high_notional": 3_333.0,
            "mid_score": 0.006,
            "high_score": 0.010,
        }
    }
    sample = SimpleNamespace(features={})
    score = SimpleNamespace(probability=0.25, expected_value=-0.011, selection_score=-0.011)

    notional, source = cli._trade_run_order_notional_from_artifact_policy(
        args,
        training_config,
        sample,
        score,
        available_notional=8_000.0,
    )

    assert notional == pytest.approx(3_333.0)
    assert source == "artifact_score_bucket_high"


def test_trade_run_order_notional_uses_artifact_confidence_policy() -> None:
    args = _trade_run_args(max_order_notional_usd=10_000.0)
    training_config = {
        "minimum_probability": 0.50,
        "minimum_expected_value": 0.0,
        "label_net_profit_target": 0.02,
        "label_net_stop_loss": 0.02,
        "sizing_policy": {"mode": "confidence"},
    }
    sample = SimpleNamespace(features={"net_profit_target": 0.02, "net_stop_loss": 0.02})
    score = SimpleNamespace(probability=0.75, expected_value=0.0025, selection_score=0.0025)

    notional, source = cli._trade_run_order_notional_from_artifact_policy(
        args,
        training_config,
        sample,
        score,
        available_notional=8_000.0,
    )

    assert source == "artifact_confidence"
    assert 2_000.0 < notional < 8_000.0


def test_trade_run_reads_artifact_execution_policy() -> None:
    assert cli._artifact_entry_order_model({"execution_policy": {"entry_order_model": "market"}}) == "market"
    assert cli._artifact_entry_order_model({"entry_order_model": "limit"}) == "limit"


def test_trade_run_live_artifact_contract_fails_closed_on_missing_metadata() -> None:
    args = _trade_run_args(history_bar_size="1 secs", tick_by_tick_type="Last")

    errors = cli._live_artifact_contract_errors(
        {"execution_policy": {"entry_order_model": "market"}},
        args,
    )

    assert "sizing_policy is missing" in errors
    assert "candidate_policy is missing" in errors
    assert "setup_family_policy is missing" in errors
    assert "horizon_policy is missing" in errors
    assert "data_contract is missing" in errors
    assert "feature_contract is missing" in errors


def test_trade_run_live_artifact_contract_requires_tick_backed_one_second_data() -> None:
    training_config = {
        "execution_policy": {"entry_order_model": "market"},
        "sizing_policy": {"mode": "score_bucket"},
        "candidate_policy": {"side_mode": "long", "allow_short_research": False},
        "setup_family_policy": {
            "setup_family_fields": ["event_specialist_setup_family", "event_setup_family"],
            "score_direction_field": "event_setup_score_direction",
        },
        "horizon_policy": {"min_holding_seconds": 1.0},
        "data_contract": {"requires_one_second_execution": True},
        "feature_contract": {"feature_count": 12},
    }

    ok_args = _trade_run_args(history_bar_size="1 secs", tick_by_tick_type="Last")
    missing_tick_args = _trade_run_args(history_bar_size="1 secs", tick_by_tick_type="none")

    assert cli._live_artifact_contract_errors(training_config, ok_args) == []
    assert "data_contract requires an IBKR tick-by-tick stream" in cli._live_artifact_contract_errors(
        training_config,
        missing_tick_args,
    )


def test_strict_live_valid_one_second_gate_rejects_short_or_quote_only_replay() -> None:
    args = Namespace(
        min_live_valid_1s_days=7.0,
        preferred_live_valid_1s_days=14.0,
        min_live_valid_folds=2,
    )
    data_coverage = {
        "requested_window": {
            "start": "2026-05-04T00:00:00+00:00",
            "end": "2026-05-04T06:00:00+00:00",
        }
    }
    label_coverage = {
        "enabled": True,
        "interval": "1s",
        "bars": 21_600,
        "provenance": {"tick_backed_bars": 20_000},
    }
    execution_coverage = {
        "enabled": True,
        "interval": "1s",
        "bars": 21_600,
        "provenance": {"tick_backed_bars": 21_600},
    }

    diagnostics = _strict_live_valid_1s_diagnostics(
        args=args,
        data_coverage=data_coverage,
        label_bar_coverage=label_coverage,
        execution_bar_coverage=execution_coverage,
        folds=1,
    )

    assert diagnostics["ok"] is False
    assert diagnostics["window_days"] == pytest.approx(0.25)
    assert set(diagnostics["errors"]) == {
        "label_bars_not_tick_backed",
        "window_too_short",
        "too_few_walk_forward_folds",
    }


def test_strict_live_valid_one_second_gate_accepts_multi_day_tick_backed_replay() -> None:
    args = Namespace(
        min_live_valid_1s_days=7.0,
        preferred_live_valid_1s_days=14.0,
        min_live_valid_folds=2,
    )
    data_coverage = {"requested_window": {"span_days": 8.0}}
    replay_coverage = {
        "enabled": True,
        "interval": "1s",
        "bars": 691_200,
        "provenance": {"tick_backed_bars": 691_200},
    }

    diagnostics = _strict_live_valid_1s_diagnostics(
        args=args,
        data_coverage=data_coverage,
        label_bar_coverage=replay_coverage,
        execution_bar_coverage=replay_coverage,
        folds=2,
    )

    assert diagnostics["ok"] is True
    assert diagnostics["errors"] == []
    assert diagnostics["label_tick_backed_ratio"] == pytest.approx(1.0)


def test_trade_run_model_exit_geometry_uses_gross_price_barriers() -> None:
    args = _trade_run_args(
        synthetic_stop_loss_bps=100.0,
        synthetic_profit_target_bps=0.0,
        use_model_exit_geometry=True,
    )
    sample = SimpleNamespace(
        features={
            "net_profit_target": 0.0005,
            "net_stop_loss": 0.0080,
            "gross_profit_move": 0.004308,
            "gross_stop_distance": 0.004192,
        }
    )

    stop_bps, profit_bps, source = cli._runner_exit_bps(args, sample)

    assert stop_bps == pytest.approx(41.92)
    assert profit_bps == pytest.approx(43.08)
    assert source == "model_gross_stop+model_gross_profit"


def test_trade_run_exit_timing_honors_minimum_holding_seconds() -> None:
    entered_utc = datetime(2026, 1, 1, tzinfo=UTC)
    sample = SimpleNamespace(features={"event_min_holding_seconds": 5.0})

    timing = _runner_exit_timing(
        entered_monotonic=100.0,
        entered_utc=entered_utc,
        hold_seconds=1.0,
        min_holding_seconds=_runner_min_holding_seconds(sample),
    )

    assert timing["min_exit_monotonic"] == pytest.approx(105.0)
    assert timing["exit_deadline_monotonic"] == pytest.approx(105.0)
    assert timing["min_exit_utc"] == (entered_utc + timedelta(seconds=5)).isoformat()
    assert timing["exit_deadline_utc"] == (entered_utc + timedelta(seconds=5)).isoformat()


def test_broker_trade_run_requires_account_and_live_confirmation() -> None:
    with pytest.raises(SystemExit, match="explicit broker.account"):
        _validate_trade_run_config(AppConfig(), _trade_run_args())

    live_cfg = replace(
        AppConfig(),
        runtime=RuntimeConfig(
            mode=RuntimeMode.LIVE,
            enable_live_trading=True,
            live_confirmation="ZEROALPHA_LIVE",
        ),
        broker=BrokerConfig(port=4001, account="DU123456"),
    )
    with pytest.raises(SystemExit, match="ZEROALPHA_LIVE_TRADE_RUN"):
        _validate_trade_run_config(live_cfg, _trade_run_args(confirm="IBKR_PAPER_TRADE_RUN"))
    with pytest.raises(SystemExit, match="decision-threshold"):
        _validate_trade_run_config(
            _paper_cfg_with_account(),
            _trade_run_args(decision_threshold=1.5),
        )

    _validate_trade_run_config(live_cfg, _trade_run_args(confirm="ZEROALPHA_LIVE_TRADE_RUN"))


def test_trade_run_kill_switch_checks_configured_file(tmp_path) -> None:
    kill_switch = tmp_path / "kill_switch.enabled"
    cfg = replace(AppConfig(), runtime=replace(AppConfig().runtime, kill_switch_file=kill_switch))

    assert _kill_switch_enabled(cfg) is False

    kill_switch.write_text("stop\n")

    assert _kill_switch_enabled(cfg) is True


def test_trade_run_verified_one_second_data_gate_requires_ibkr_one_second_bars() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(seconds=idx + 1),
            symbol="BTC/USD",
            bar_size="1 secs",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="IBKR:AGGTRADES",
        )
        for idx in range(60)
    ]
    args = Namespace(
        require_live_1s_data=True,
        live_data_mode="streaming",
        tick_by_tick_type="Last",
        history_what_to_show="AGGTRADES",
        history_bar_size="1 secs",
    )

    report = _require_verified_one_second_data(args, bars)

    assert report["ok"] is True
    assert report["bar_size_seconds"] == 1

    bad_tick_args = Namespace(
        require_live_1s_data=True,
        live_data_mode="streaming",
        tick_by_tick_type="BidAsk",
        history_what_to_show="AGGTRADES",
        history_bar_size="1 secs",
    )
    with pytest.raises(SystemExit, match="requires --tick-by-tick-type Last or AllLast"):
        _require_verified_one_second_data(bad_tick_args, bars)

    bad_bars = [replace(bars[0], bar_size="1 min")]
    with pytest.raises(SystemExit, match="verification failed"):
        _require_verified_one_second_data(args, bad_bars)


def test_live_one_second_warmup_uses_processed_tick_rows_for_gate() -> None:
    class FakeBroker:
        def __init__(self):
            self.waits = 0

        async def quote_from_ticker(self, contract, quote_ticker, *, max_wait_seconds):
            return MarketQuote(
                timestamp_utc=start,
                received_timestamp_utc=start,
                symbol="BTC/USD",
                bid=99.0,
                ask=101.0,
            )

        async def wait(self, seconds):
            self.waits += 1
            return None

    class FakeAggregator:
        def add_quote(self, quote):
            return True

        def process_ticker_ticks(self, ticker):
            return 1

        def completed_bars(self):
            return [
                Bar(
                    timestamp_utc=start + timedelta(seconds=1),
                    symbol="BTC/USD",
                    bar_size="1 secs",
                    open=100,
                    high=100,
                    low=100,
                    close=100,
                    volume=0,
                    source="IBKR:STREAM_BidAsk",
                    extra={
                        "aggregated_from": "streaming_tick_by_tick",
                        "tick_count": 1.0,
                    },
                )
            ]

    start = datetime(2026, 1, 1, tzinfo=UTC)
    args = Namespace(
        live_1s_warmup_bars=1,
        live_1s_warmup_timeout_seconds=1.0,
        require_live_1s_data=True,
        snapshot_timeout_seconds=1.0,
        signal_interval=1.0,
        history_max_bars=10,
    )

    _, report = asyncio.run(
        _warmup_live_one_second_stream(
            FakeBroker(),
            SimpleNamespace(),
            args,
            quote_ticker=SimpleNamespace(),
            tick_subscription=("BidAsk", SimpleNamespace(tickByTicks=[])),
            bar_aggregator=FakeAggregator(),
            history_bars=[],
        )
    )

    assert report["ok"] is True
    assert report["tick_rows_seen"] == 0
    assert report["processed_tick_rows"] == 1
    assert report["tick_completed_bars"] == 1


def test_live_one_second_warmup_waits_for_tick_backed_bar() -> None:
    class FakeBroker:
        def __init__(self):
            self.waits = 0

        async def quote_from_ticker(self, contract, quote_ticker, *, max_wait_seconds):
            return MarketQuote(
                timestamp_utc=start,
                received_timestamp_utc=start,
                symbol="BTC/USD",
                bid=99,
                ask=101,
                source="IBKR",
            )

        async def wait(self, seconds):
            self.waits += 1
            return None

    class FakeAggregator:
        def __init__(self):
            self.calls = 0

        def add_quote(self, quote):
            return True

        def process_ticker_ticks(self, ticker):
            return 1

        def completed_bars(self):
            self.calls += 1
            tick_count = 0.0 if self.calls == 1 else 1.0
            return [
                Bar(
                    timestamp_utc=start + timedelta(seconds=self.calls),
                    symbol="BTC/USD",
                    bar_size="1 secs",
                    open=100,
                    high=100,
                    low=100,
                    close=100,
                    volume=0,
                    source="IBKR:STREAM_Last",
                    extra={
                        "aggregated_from": (
                            "streaming_quote_sample"
                            if tick_count == 0.0
                            else "streaming_tick_by_tick"
                        ),
                        "tick_count": tick_count,
                    },
                )
            ]

    start = datetime(2026, 1, 1, tzinfo=UTC)
    aggregator = FakeAggregator()
    broker = FakeBroker()
    args = Namespace(
        live_1s_warmup_bars=1,
        live_1s_warmup_timeout_seconds=1.0,
        require_live_1s_data=True,
        snapshot_timeout_seconds=1.0,
        signal_interval=1.0,
        history_max_bars=10,
    )

    _, report = asyncio.run(
        _warmup_live_one_second_stream(
            broker,
            SimpleNamespace(),
            args,
            quote_ticker=SimpleNamespace(),
            tick_subscription=("Last", SimpleNamespace(tickByTicks=[])),
            bar_aggregator=aggregator,
            history_bars=[],
        )
    )

    assert aggregator.calls == 2
    assert broker.waits == 1
    assert report["ok"] is True
    assert report["completed_bars"] == 2
    assert report["tick_completed_bars"] == 1
    assert report["aggregated_from"] == {
        "streaming_quote_sample": 1,
        "streaming_tick_by_tick": 1,
    }


def test_trade_run_quote_uses_snapshot_timeout_for_subscribed_quote() -> None:
    class FakeBroker:
        async def quote_from_ticker(self, contract, quote_ticker, *, max_wait_seconds):
            self.max_wait_seconds = max_wait_seconds
            return "quote"

    broker = FakeBroker()
    args = Namespace(snapshot_timeout_seconds=8.0, signal_interval=1.0)

    quote = asyncio.run(
        _trade_run_quote(
            broker,
            SimpleNamespace(),
            args,
            quote_ticker=SimpleNamespace(),
        )
    )

    assert quote == "quote"
    assert broker.max_wait_seconds == pytest.approx(8.0)


def test_broker_round_trip_rejects_live_mode_and_oversized_notional() -> None:
    live_cfg = replace(
        AppConfig(),
        runtime=RuntimeConfig(
            mode=RuntimeMode.LIVE,
            enable_live_trading=True,
            live_confirmation="ZEROALPHA_LIVE",
        ),
        broker=BrokerConfig(port=4001),
    )

    with pytest.raises(SystemExit, match="paper-only"):
        _validate_round_trip_test_config(live_cfg, _round_trip_args())
    with pytest.raises(SystemExit, match="notional cannot exceed max cash"):
        _validate_round_trip_test_config(_paper_cfg_with_account(), _round_trip_args(notional=11_000.0))


def test_interpretability_flags_are_available_on_ml_commands() -> None:
    parser = cli.build_parser()

    backtest_args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--permutation-importance",
            "--permutation-repeats",
            "2",
            "--permutation-grouping",
            "both",
            "--shap-importance",
            "--shap-grouping",
            "both",
            "--importance-scoring",
            "brier,log_loss,net_pnl",
            "--feature-exclude-groups",
            "technical",
            "--feature-exclude-patterns",
            "rsi_*,macd_*",
        ]
    )
    train_args = parser.parse_args(
        ["model", "train-meta", "--permutation-importance", "--save-artifact", "artifacts/models/prod.joblib"]
    )
    audit_args = parser.parse_args(["model", "signal-audit", "--permutation-importance"])

    assert backtest_args.permutation_importance is True
    assert backtest_args.permutation_repeats == 2
    assert backtest_args.permutation_grouping == "both"
    assert backtest_args.shap_importance is True
    assert backtest_args.shap_grouping == "both"
    assert backtest_args.importance_scoring == "brier,log_loss,net_pnl"
    assert backtest_args.feature_exclude_groups == "technical"
    assert backtest_args.feature_exclude_patterns == "rsi_*,macd_*"
    assert train_args.permutation_importance is True
    assert train_args.save_artifact == "artifacts/models/prod.joblib"
    assert audit_args.permutation_importance is True


def test_train_meta_supports_negative_ev_frequency_probe_for_artifacts() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["model", "train-meta", "--allow-negative-ev-frequency-probe"])

    assert args.allow_negative_ev_frequency_probe is True


def test_ml_backtest_cli_can_disable_consecutive_loss_cooldown() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--consecutive-loss-limit",
            "0",
            "--cooldown-hours-after-stopouts",
            "0",
        ]
    )

    cfg = cli._override_config_from_args(AppConfig(), args)

    assert cfg.risk.consecutive_loss_limit == 0
    assert cfg.risk.cooldown_hours_after_stopouts == 0
    cfg.validate()


def test_ml_cli_can_explicitly_zero_minimum_probability_without_default_override() -> None:
    parser = cli.build_parser()
    base = AppConfig(model=ModelConfig(minimum_probability=0.72))

    default_args = parser.parse_args(["backtest", "ml"])
    default_cfg = cli._override_config_from_args(base, default_args)

    zero_args = parser.parse_args(["backtest", "ml", "--minimum-probability", "0"])
    zero_cfg = cli._override_config_from_args(base, zero_args)

    assert default_cfg.model.minimum_probability == 0.72
    assert zero_cfg.model.minimum_probability == 0.0


def test_ml_cli_can_explicitly_disable_loss_stops_without_default_override() -> None:
    parser = cli.build_parser()
    base = AppConfig(
        risk=RiskConfig(
            daily_loss_stop=0.01,
            weekly_loss_stop=0.03,
            rolling_drawdown_stop=0.08,
        )
    )

    default_args = parser.parse_args(["backtest", "ml"])
    default_cfg = cli._override_config_from_args(base, default_args)

    zero_args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--daily-loss-stop",
            "0",
            "--weekly-loss-stop",
            "0",
            "--rolling-drawdown-stop",
            "0",
        ]
    )
    zero_cfg = cli._override_config_from_args(base, zero_args)

    assert default_cfg.risk.daily_loss_stop == 0.01
    assert default_cfg.risk.weekly_loss_stop == 0.03
    assert default_cfg.risk.rolling_drawdown_stop == 0.08
    assert zero_cfg.risk.daily_loss_stop == 0.0
    assert zero_cfg.risk.weekly_loss_stop == 0.0
    assert zero_cfg.risk.rolling_drawdown_stop == 0.0


def test_ml_research_defaults_to_expected_utility_selection() -> None:
    parser = cli.build_parser()

    backtest_args = parser.parse_args(["backtest", "ml"])
    train_args = parser.parse_args(["model", "train-meta"])
    sweep_args = parser.parse_args(["model", "sweep-labels"])

    assert backtest_args.selection_score == "expected_utility"
    assert train_args.selection_score == "expected_utility"
    assert sweep_args.selection_score == "expected_utility"


def test_online_target_frequency_defaults_to_capacity_aware_selection() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--target-frequency-mode",
            "online",
            "--max-open-positions",
            "3",
        ]
    )
    cfg = cli._override_config_from_args(AppConfig(), args)

    assert args.respect_open_positions is None
    assert _effective_respect_open_positions(args, cfg) is True


def test_capacity_aware_selection_can_be_disabled_explicitly() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--target-frequency-mode",
            "online",
            "--max-open-positions",
            "3",
            "--no-respect-open-positions",
        ]
    )
    cfg = cli._override_config_from_args(AppConfig(), args)

    assert _effective_respect_open_positions(args, cfg) is False


def test_adaptive_horizon_default_is_cost_aware_for_high_commission_spot() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--adaptive-horizon",
            "--assumed-spread-bps",
            "0.04",
            "--base-slippage-bps",
            "0.5",
            "--safety-margin-bps",
            "1.0",
            "--tier-rate",
            "0.0018",
            "--minimum-commission",
            "1.75",
            "--notional",
            "10000",
            "--net-profit-target",
            "0.002",
            "--net-stop-loss",
            "0.008",
        ]
    )
    cfg = cli._override_config_from_args(AppConfig(), args)

    candidate_cfg = cli._candidate_config_from_args(args, cfg)

    assert 700 < candidate_cfg.adaptive_horizon_target_move_bps < 800


def test_adaptive_horizon_explicit_target_move_wins() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "ml",
            "--adaptive-horizon",
            "--adaptive-horizon-target-move-bps",
            "80",
        ]
    )

    candidate_cfg = cli._candidate_config_from_args(args, AppConfig())

    assert candidate_cfg.adaptive_horizon_target_move_bps == 80


def test_paper_order_test_rejects_live_mode_and_live_port() -> None:
    live_cfg = replace(
        AppConfig(),
        runtime=RuntimeConfig(
            mode=RuntimeMode.LIVE,
            enable_live_trading=True,
            live_confirmation="ZEROALPHA_LIVE",
        ),
        broker=BrokerConfig(port=4001),
    )
    custom_port_cfg = replace(AppConfig(), broker=BrokerConfig(port=4001))
    args = _paper_order_args()

    with pytest.raises(SystemExit, match="paper-only"):
        _validate_paper_order_test_config(live_cfg, args)
    with pytest.raises(SystemExit, match="paper port"):
        _validate_paper_order_test_config(custom_port_cfg, args)


def test_research_short_backtest_requires_research_gate() -> None:
    with pytest.raises(SystemExit, match="requires --research-gate"):
        _validate_research_short_backtest_args(
            Namespace(allow_research_short_backtest=True, research_gate=False)
        )

    _validate_research_short_backtest_args(
        Namespace(allow_research_short_backtest=True, research_gate=True)
    )


def test_actual_capacity_release_requires_research_gate() -> None:
    with pytest.raises(SystemExit, match="capacity-release-mode actual"):
        _validate_research_gated_args(
            Namespace(
                allow_spot_short_research=False,
                allow_research_short_backtest=False,
                capacity_release_mode="actual",
                research_gate=False,
            )
        )

    _validate_research_gated_args(
        Namespace(
            allow_spot_short_research=False,
            allow_research_short_backtest=False,
            capacity_release_mode="actual",
            research_gate=True,
        )
    )


def test_sample_filter_can_include_and_exclude_setup_families() -> None:
    samples = [_meta_sample("dense_baseline"), _meta_sample("dense_trend_continuation")]

    included = _filter_samples_from_args(
        samples,
        Namespace(candidate_types="", setup_families="dense_baseline", exclude_setup_families=""),
    )
    excluded = _filter_samples_from_args(
        samples,
        Namespace(
            candidate_types="",
            setup_families="",
            exclude_setup_families="dense_trend_continuation",
        ),
    )

    assert [sample.event_id for sample in included] == ["dense_baseline"]
    assert [sample.event_id for sample in excluded] == ["dense_baseline"]


def test_sample_filter_can_require_prediction_market_residual_edge() -> None:
    good = _meta_sample("good")
    weak = _meta_sample("weak")
    good.features.update(
        {
            "pm_leading_residual_edge_max": 0.12,
            "pm_leading_liquidity_weight_total": 7.0,
        }
    )
    weak.features.update(
        {
            "pm_leading_residual_edge_max": -0.05,
            "pm_leading_liquidity_weight_total": 7.0,
        }
    )

    filtered = _filter_samples_from_args(
        [good, weak],
        Namespace(
            candidate_types="",
            setup_families="",
            exclude_setup_families="",
            require_prediction_market_data=False,
            require_leading_prediction_market_data=False,
            prediction_market_min_available_count=0,
            prediction_market_min_side_mid=0.0,
            prediction_market_min_lead_seconds=0.0,
            prediction_market_min_leading_side_mid=0.0,
            prediction_market_min_leading_residual_edge=0.0,
            prediction_market_min_leading_liquidity_weight=5.0,
        ),
    )

    assert [sample.event_id for sample in filtered] == ["good"]


def test_allow_data_gaps_accepts_only_gap_issues() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start,
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
        Bar(
            timestamp_utc=start + timedelta(hours=2),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
    ]
    report = validate_bars(
        bars,
        expected_interval="1h",
        start=start,
        end=start + timedelta(hours=2),
        minimum_coverage_ratio=0.0,
    )

    accepted = _quality_or_raise(report, label="primary BTCUSDT", allow_data_gaps=True)

    assert accepted["accepted_with_issues"] is True
    assert accepted["accepted_issue_codes"] == ["bar_gap"]


def test_allow_data_gaps_still_rejects_insufficient_coverage() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start,
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
        Bar(
            timestamp_utc=start + timedelta(hours=3),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
    ]
    report = validate_bars(
        bars,
        expected_interval="1h",
        start=start,
        end=start + timedelta(hours=6),
        minimum_coverage_ratio=0.75,
    )

    with pytest.raises(ValueError, match="data quality gate failed"):
        _quality_or_raise(report, label="primary BTCUSDT", allow_data_gaps=True)


def test_allow_data_gaps_accepts_optional_context_short_coverage() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(hours=4),
            symbol="SOLUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
        Bar(
            timestamp_utc=start + timedelta(hours=6),
            symbol="SOLUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
    ]
    report = validate_bars(
        bars,
        expected_interval="1h",
        start=start,
        end=start + timedelta(hours=8),
        minimum_coverage_ratio=0.95,
    )

    accepted = _context_quality_or_raise(report, label="context SOLUSDT", allow_data_gaps=True)

    assert accepted["accepted_with_issues"] is True
    assert accepted["accepted_issue_codes"] == ["bar_gap", "insufficient_coverage"]


def test_primary_bars_jsonl_can_replace_binance_primary(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=minute),
            symbol="BTC/USD",
            bar_size="1 min",
            open=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100 + minute,
            volume=1,
            source="IBKR:MIDPOINT",
        )
        for minute in range(3)
    ]
    path = tmp_path / "ibkr_bars.jsonl"
    write_ibkr_bars(path, bars)

    primary, context, coverage = _load_research_bars(
        Namespace(
            cache_dir=str(tmp_path),
            primary_bars_jsonl=str(path),
            context_bars_jsonl="",
            symbol="BTCUSD",
            interval="1m",
            context_symbols="none",
            context_interval="",
            binance_um_futures_reference_symbols="none",
            coinbase_reference_products="none",
            minimum_data_coverage=0.9,
            max_bar_return_bps=0.0,
            allow_data_gaps=False,
            max_source_divergence_bps=500.0,
        ),
        start,
        start + timedelta(minutes=3),
    )

    assert primary == bars
    assert context == {}
    assert coverage["primary"]["source"] == "IBKR_JSONL"
    assert coverage["primary"]["bars"] == 3


def test_context_bars_jsonl_adds_named_context(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=minute),
            symbol="MBT/USD",
            bar_size="1 min",
            open=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100 + minute,
            volume=1,
            source="IBKR:TRADES",
        )
        for minute in range(3)
    ]
    primary_path = tmp_path / "spot.jsonl"
    context_path = tmp_path / "mbt.jsonl"
    write_ibkr_bars(primary_path, bars)
    write_ibkr_bars(context_path, bars)

    _, context, coverage = _load_research_bars(
        Namespace(
            cache_dir=str(tmp_path),
            primary_bars_jsonl=str(primary_path),
            context_bars_jsonl=f"IBKR_MBT={context_path}",
            symbol="BTCUSD",
            interval="1m",
            context_symbols="none",
            context_interval="",
            binance_um_futures_reference_symbols="none",
            coinbase_reference_products="none",
            minimum_data_coverage=0.9,
            max_bar_return_bps=0.0,
            allow_data_gaps=False,
            max_source_divergence_bps=500.0,
        ),
        start,
        start + timedelta(minutes=3),
    )

    assert list(context) == ["IBKR_MBT"]
    assert coverage["context"]["IBKR_MBT"]["source"] == "IBKR_JSONL"


def test_primary_market_can_use_binance_um_futures(monkeypatch, tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start + timedelta(minutes=minute),
            symbol="BTCUSDT",
            bar_size="1m",
            open=100 + minute,
            high=101 + minute,
            low=99 + minute,
            close=100 + minute,
            volume=1,
            source="BINANCE_UM_FUTURES",
        )
        for minute in range(3)
    ]

    def fake_fetch_futures(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["symbol"] == "BTCUSDT"
        assert kwargs["market_type"] == "um"
        return bars

    monkeypatch.setattr(cli, "fetch_futures_klines_archive_range", fake_fetch_futures)

    primary, _, coverage = _load_research_bars(
        Namespace(
            cache_dir=str(tmp_path),
            primary_market="binance_um_futures",
            primary_bars_jsonl="",
            context_bars_jsonl="",
            symbol="BTCUSDT",
            interval="1m",
            context_symbols="none",
            context_interval="",
            binance_um_futures_reference_symbols="none",
            coinbase_reference_products="none",
            minimum_data_coverage=0.9,
            max_bar_return_bps=0.0,
            allow_data_gaps=False,
            max_source_divergence_bps=500.0,
        ),
        start,
        start + timedelta(minutes=3),
    )

    assert primary == bars
    assert coverage["primary"]["source"] == "BINANCE_UM_FUTURES"
