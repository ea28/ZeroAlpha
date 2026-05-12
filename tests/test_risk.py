from datetime import datetime, UTC, timedelta
from pathlib import Path

from zeroalpha.config import RiskConfig
from zeroalpha.domain import MarketQuote, Prediction, RuntimeMode
from zeroalpha.risk.engine import RiskEngine, RiskSnapshot


def test_risk_engine_approves_clean_trade(tmp_path: Path) -> None:
    engine = RiskEngine(RiskConfig(), mode=RuntimeMode.PAPER, kill_switch_file=tmp_path / "kill")
    quote = MarketQuote(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        received_timestamp_utc=datetime.now(tz=UTC) - timedelta(milliseconds=10),
        symbol="BTC/USD",
        bid=99_990,
        ask=100_010,
    )
    prediction = Prediction(
        event_id="e1",
        model_version="test",
        calibrated_probability=0.7,
        expected_value=0.02,
        decision="trade",
        decision_reason="test",
    )
    decision = engine.approve_entry(
        prediction=prediction,
        quote=quote,
        snapshot=RiskSnapshot(available_cash=100_000),
        stop_distance=0.02,
        minimum_probability=0.6,
        minimum_expected_value=0.01,
    )
    assert decision.approved
    assert decision.position_notional > 0


def test_risk_engine_zero_loss_stops_are_disabled(tmp_path: Path) -> None:
    engine = RiskEngine(
        RiskConfig(daily_loss_stop=0.0, weekly_loss_stop=0.0, rolling_drawdown_stop=0.0),
        mode=RuntimeMode.PAPER,
        kill_switch_file=tmp_path / "kill",
    )
    quote = MarketQuote(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        received_timestamp_utc=datetime.now(tz=UTC) - timedelta(milliseconds=10),
        symbol="BTC/USD",
        bid=99_990,
        ask=100_010,
    )
    prediction = Prediction("e1", "test", 0.7, 0.02, "trade", "test")

    decision = engine.approve_entry(
        prediction=prediction,
        quote=quote,
        snapshot=RiskSnapshot(
            account_equity=10_000,
            available_cash=10_000,
            daily_pnl=-1.0,
            weekly_pnl=-1.0,
            rolling_drawdown=0.01,
        ),
        stop_distance=0.02,
        minimum_probability=0.6,
        minimum_expected_value=0.01,
    )

    assert decision.approved


def test_risk_engine_caps_notional_by_account_equity(tmp_path: Path) -> None:
    engine = RiskEngine(
        RiskConfig(risk_per_trade=1.0, paper_max_notional=100_000),
        mode=RuntimeMode.PAPER,
        kill_switch_file=tmp_path / "kill",
    )

    notional = engine.position_notional(stop_distance=0.01, account_equity=5_000)

    assert notional == 5_000


def test_risk_engine_uses_equity_as_cash_cap_when_cash_missing(tmp_path: Path) -> None:
    engine = RiskEngine(
        RiskConfig(risk_per_trade=1.0, paper_max_notional=100_000),
        mode=RuntimeMode.PAPER,
        kill_switch_file=tmp_path / "kill",
    )
    quote = MarketQuote(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        received_timestamp_utc=datetime.now(tz=UTC) - timedelta(milliseconds=10),
        symbol="BTC/USD",
        bid=99_990,
        ask=100_010,
    )
    prediction = Prediction("e1", "test", 0.9, 0.05, "trade", "test")

    decision = engine.approve_entry(
        prediction=prediction,
        quote=quote,
        snapshot=RiskSnapshot(account_equity=5_000),
        stop_distance=0.01,
        minimum_probability=0.6,
        minimum_expected_value=0.01,
    )

    assert decision.approved
    assert decision.position_notional <= 5_000


def test_kill_switch_blocks_trade(tmp_path: Path) -> None:
    kill = tmp_path / "kill"
    kill.write_text("enabled")
    engine = RiskEngine(RiskConfig(), mode=RuntimeMode.PAPER, kill_switch_file=kill)
    quote = MarketQuote(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        received_timestamp_utc=datetime.now(tz=UTC),
        symbol="BTC/USD",
        bid=99_990,
        ask=100_010,
    )
    prediction = Prediction("e1", "test", 0.9, 0.05, "trade", "test")
    decision = engine.approve_entry(
        prediction=prediction,
        quote=quote,
        snapshot=RiskSnapshot(available_cash=100_000),
        stop_distance=0.02,
        minimum_probability=0.6,
        minimum_expected_value=0.01,
    )
    assert not decision.approved
    assert decision.reason == "kill_switch_enabled"
