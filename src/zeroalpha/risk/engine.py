"""Position sizing and entry approval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zeroalpha.config import RiskConfig
from zeroalpha.domain import MarketQuote, Prediction, RuntimeMode


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    open_positions: int = 0
    account_equity: float | None = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    rolling_drawdown: float = 0.0
    available_cash: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str
    position_notional: float = 0.0


class RiskEngine:
    def __init__(self, config: RiskConfig, *, mode: RuntimeMode, kill_switch_file: Path) -> None:
        self.config = config
        self.mode = mode
        self.kill_switch_file = kill_switch_file

    def kill_switch_enabled(self) -> bool:
        return self.kill_switch_file.exists()

    def position_notional(
        self,
        *,
        stop_distance: float,
        account_equity: float | None = None,
        forecast_volatility: float | None = None,
        confidence_scale: float = 1.0,
    ) -> float:
        if stop_distance <= 0:
            raise ValueError("stop_distance must be positive")
        equity = account_equity if account_equity is not None else self.config.account_equity
        risk_based = equity * self.config.risk_per_trade / stop_distance
        volatility_based = float("inf")
        if forecast_volatility and forecast_volatility > 0:
            volatility_based = equity * 0.10 / forecast_volatility
        max_notional = (
            self.config.live_max_notional if self.mode == RuntimeMode.LIVE else self.config.paper_max_notional
        )
        return max(
            0.0,
            min(risk_based, volatility_based, max_notional, equity) * max(0.0, min(1.0, confidence_scale)),
        )

    def approve_entry(
        self,
        *,
        prediction: Prediction,
        quote: MarketQuote,
        snapshot: RiskSnapshot,
        stop_distance: float,
        minimum_probability: float,
        minimum_expected_value: float,
    ) -> RiskDecision:
        if self.kill_switch_enabled():
            return RiskDecision(False, "kill_switch_enabled")
        if snapshot.open_positions >= self.config.max_open_positions:
            return RiskDecision(False, "max_open_positions")
        account_equity = snapshot.account_equity or self.config.account_equity
        if (
            self.config.daily_loss_stop > 0
            and snapshot.daily_pnl <= -account_equity * self.config.daily_loss_stop
        ):
            return RiskDecision(False, "daily_loss_stop")
        if (
            self.config.weekly_loss_stop > 0
            and snapshot.weekly_pnl <= -account_equity * self.config.weekly_loss_stop
        ):
            return RiskDecision(False, "weekly_loss_stop")
        if (
            self.config.rolling_drawdown_stop > 0
            and snapshot.rolling_drawdown >= self.config.rolling_drawdown_stop
        ):
            return RiskDecision(False, "rolling_drawdown_stop")
        if prediction.calibrated_probability < minimum_probability:
            return RiskDecision(False, "probability_below_threshold")
        if prediction.expected_value < minimum_expected_value:
            return RiskDecision(False, "expected_value_below_threshold")
        if quote.spread_bps > self.config.max_spread_bps:
            return RiskDecision(False, "spread_too_wide")
        if quote.quote_age_ms() > self.config.max_quote_age_ms:
            return RiskDecision(False, "quote_stale")

        confidence_scale = min(1.0, max(0.1, prediction.calibrated_probability))
        notional = self.position_notional(
            stop_distance=stop_distance,
            account_equity=account_equity,
            confidence_scale=confidence_scale,
        )
        if notional < self.config.minimum_fee_efficient_notional:
            return RiskDecision(False, "notional_below_fee_efficient_size", notional)
        available_cash = snapshot.available_cash if snapshot.available_cash > 0 else account_equity
        if notional > available_cash:
            return RiskDecision(False, "insufficient_cash", notional)
        return RiskDecision(True, "approved", notional)
