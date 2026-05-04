from argparse import Namespace
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import zeroalpha.cli as cli
from zeroalpha.cli import (
    _context_quality_or_raise,
    _filter_samples_from_args,
    _load_research_bars,
    _quality_or_raise,
    _validate_paper_order_test_config,
    _validate_paper_test_config,
    _validate_research_gated_args,
    _validate_research_short_backtest_args,
    _validate_round_trip_test_config,
    _validate_trade_run_config,
)
from zeroalpha.config import AppConfig, BrokerConfig, RuntimeConfig
from zeroalpha.data.external.ibkr_bars import write_ibkr_bars
from zeroalpha.data.quality import validate_bars
from zeroalpha.domain import Bar, RuntimeMode, TripleBarrierLabel
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
    assert args.history_what_to_show == "AGGTRADES"
    assert args.max_signal_bar_age_seconds == 300.0
    assert args.stream_format == "json"
    _validate_trade_run_config(_paper_cfg_with_account(), args)


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

    _validate_trade_run_config(live_cfg, _trade_run_args(confirm="ZEROALPHA_LIVE_TRADE_RUN"))


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
