from argparse import Namespace
from dataclasses import replace

import pytest

from zeroalpha.cli import _validate_paper_order_test_config, _validate_research_short_backtest_args
from zeroalpha.config import AppConfig, BrokerConfig, RuntimeConfig
from zeroalpha.domain import RuntimeMode


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
