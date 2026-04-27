from argparse import Namespace
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from zeroalpha.cli import (
    _context_quality_or_raise,
    _quality_or_raise,
    _validate_paper_order_test_config,
    _validate_research_short_backtest_args,
)
from zeroalpha.config import AppConfig, BrokerConfig, RuntimeConfig
from zeroalpha.data.quality import validate_bars
from zeroalpha.domain import Bar, RuntimeMode


def test_paper_order_test_requires_explicit_confirmation() -> None:
    cfg = AppConfig()

    with pytest.raises(SystemExit, match="PAPER_ORDER_TEST"):
        _validate_paper_order_test_config(cfg, Namespace(confirm=""))

    _validate_paper_order_test_config(cfg, Namespace(confirm="PAPER_ORDER_TEST"))


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
    args = Namespace(confirm="PAPER_ORDER_TEST")

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
