"""IBKR-aware commission, spread, and slippage cost modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommissionModel:
    tier_rate: float = 0.0018
    minimum_commission: float = 1.75
    maximum_commission_rate: float = 0.01

    def commission(self, trade_notional: float) -> float:
        if trade_notional <= 0:
            raise ValueError("trade_notional must be positive")
        raw = max(self.minimum_commission, self.tier_rate * trade_notional)
        return min(self.maximum_commission_rate * trade_notional, raw)

    def commission_bps(self, trade_notional: float) -> float:
        return 10_000 * self.commission(trade_notional) / trade_notional

    def round_trip_commission(self, trade_notional: float) -> float:
        return 2 * self.commission(trade_notional)

    def round_trip_commission_bps(self, trade_notional: float) -> float:
        return 2 * self.commission_bps(trade_notional)


@dataclass(frozen=True, slots=True)
class SlippageModel:
    base_slippage_bps: float = 5.0
    spread_multiplier: float = 0.5
    volatility_multiplier: float = 0.0
    urgency_bps: float = 0.0

    def estimate_bps(self, spread_bps: float, volatility_bps: float = 0.0) -> float:
        if spread_bps < 0 or volatility_bps < 0:
            raise ValueError("spread and volatility must be nonnegative")
        return (
            self.base_slippage_bps
            + self.spread_multiplier * spread_bps
            + self.volatility_multiplier * volatility_bps
            + self.urgency_bps
        )


@dataclass(frozen=True, slots=True)
class RoundTripCost:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    safety_margin_bps: float

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps + self.safety_margin_bps

    @property
    def total_return_fraction(self) -> float:
        return self.total_bps / 10_000


def estimate_round_trip_cost(
    trade_notional: float,
    spread_bps: float,
    commission_model: CommissionModel,
    slippage_model: SlippageModel,
    safety_margin_bps: float,
    volatility_bps: float = 0.0,
) -> RoundTripCost:
    if safety_margin_bps < 0:
        raise ValueError("safety_margin_bps must be nonnegative")
    return RoundTripCost(
        commission_bps=commission_model.round_trip_commission_bps(trade_notional),
        # ``spread_bps`` is measured as ask-bid over midpoint. Crossing into and
        # out of a position pays half the spread on each leg, or one full spread
        # round trip.
        spread_bps=spread_bps,
        slippage_bps=2 * slippage_model.estimate_bps(spread_bps, volatility_bps),
        safety_margin_bps=safety_margin_bps,
    )
