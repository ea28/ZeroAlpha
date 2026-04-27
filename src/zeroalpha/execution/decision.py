"""Expected-value gate for candidate trades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedValueInputs:
    calibrated_probability: float
    expected_win: float
    expected_loss: float
    total_cost: float


@dataclass(frozen=True, slots=True)
class TradeDecision:
    should_trade: bool
    expected_value: float
    reason: str


def expected_value(inputs: ExpectedValueInputs) -> float:
    if not 0 <= inputs.calibrated_probability <= 1:
        raise ValueError("calibrated_probability must be in [0, 1]")
    if inputs.expected_win < 0 or inputs.expected_loss < 0 or inputs.total_cost < 0:
        raise ValueError("win/loss/cost inputs must be nonnegative")
    p = inputs.calibrated_probability
    return p * inputs.expected_win - (1 - p) * inputs.expected_loss - inputs.total_cost


def gate_trade(
    inputs: ExpectedValueInputs,
    *,
    minimum_probability: float,
    minimum_expected_value: float,
) -> TradeDecision:
    ev = expected_value(inputs)
    if inputs.calibrated_probability < minimum_probability:
        return TradeDecision(False, ev, "probability_below_threshold")
    if ev < minimum_expected_value:
        return TradeDecision(False, ev, "expected_value_below_threshold")
    return TradeDecision(True, ev, "approved")
