from datetime import UTC, datetime, timedelta

from zeroalpha.backtest.simple import _estimate_trade_notional, _period_pnl


def test_backtest_notional_uses_risk_based_sizing() -> None:
    notional = _estimate_trade_notional(
        equity=10_000,
        risk_per_trade=0.0035,
        net_stop_loss=0.02,
        requested_notional=10_000,
        max_notional=10_000,
    )
    assert notional == 1_750


def test_backtest_notional_can_skip_equity_cap_for_futures_research() -> None:
    notional = _estimate_trade_notional(
        equity=10_000,
        risk_per_trade=0.006,
        net_stop_loss=0.001,
        requested_notional=12_000,
        max_notional=12_000,
        cap_by_equity=False,
    )
    assert notional == 12_000


def test_backtest_period_pnl_resets_by_day_and_week() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    realized = [
        (start, -100.0),
        (start + timedelta(hours=1), 25.0),
        (start + timedelta(days=2), -50.0),
    ]
    assert _period_pnl(realized, timestamp=start + timedelta(hours=2), weekly=False) == -75
    assert _period_pnl(realized, timestamp=start + timedelta(days=2, hours=1), weekly=False) == -50
    assert _period_pnl(realized, timestamp=start + timedelta(days=2, hours=1), weekly=True) == -125
