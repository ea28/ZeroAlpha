"""Baseline research models that need no heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class BaselineResult:
    name: str
    net_return: float
    trade_count: int
    hit_rate: float


def no_trade_baseline() -> BaselineResult:
    return BaselineResult("no_trade", 0.0, 0, 0.0)


def event_rule_baseline(labels: list[int], net_returns: list[float], *, name: str) -> BaselineResult:
    if len(labels) != len(net_returns):
        raise ValueError("labels and returns must have the same length")
    if not labels:
        return BaselineResult(name, 0.0, 0, 0.0)
    return BaselineResult(
        name=name,
        net_return=sum(net_returns),
        trade_count=len(labels),
        hit_rate=mean(labels),
    )
