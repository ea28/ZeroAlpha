"""Causal market-regime features for research routing."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean, pstdev

from zeroalpha.domain import Bar


@dataclass(frozen=True, slots=True)
class MarketRegime:
    label: str
    trend_score: float
    efficiency_ratio: float
    volatility_percentile: float
    compression_ratio: float
    range_position: float
    shock_score: float

    def as_features(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["market_regime"] = self.label
        payload["regime_trend_score"] = self.trend_score
        payload["regime_efficiency_ratio"] = self.efficiency_ratio
        payload["regime_volatility_percentile"] = self.volatility_percentile
        payload["regime_compression_ratio"] = self.compression_ratio
        payload["regime_range_position"] = self.range_position
        payload["regime_shock_score"] = self.shock_score
        payload.pop("label", None)
        payload.pop("trend_score", None)
        payload.pop("efficiency_ratio", None)
        payload.pop("volatility_percentile", None)
        payload.pop("compression_ratio", None)
        payload.pop("range_position", None)
        payload.pop("shock_score", None)
        return payload


def _returns(bars: list[Bar]) -> list[float]:
    return [
        bars[idx].close / bars[idx - 1].close - 1
        for idx in range(1, len(bars))
        if bars[idx - 1].close > 0
    ]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _efficiency_ratio(bars: list[Bar]) -> float:
    if len(bars) < 3:
        return 0.0
    displacement = abs(bars[-1].close - bars[0].close)
    path = sum(abs(bars[idx].close - bars[idx - 1].close) for idx in range(1, len(bars)))
    return displacement / path if path > 0 else 0.0


def _volatility_percentile(bars: list[Bar], *, lookback: int = 24, history: int = 240) -> float:
    if len(bars) < lookback + 2:
        return 0.5
    window = bars[-min(len(bars), history + lookback + 1) :]
    realized: list[float] = []
    for idx in range(lookback + 1, len(window) + 1):
        returns = _returns(window[idx - lookback - 1 : idx])
        realized.append(pstdev(returns) if len(returns) > 1 else 0.0)
    if not realized:
        return 0.5
    current = realized[-1]
    return sum(value <= current for value in realized) / len(realized)


def classify_market_regime(bars: list[Bar]) -> MarketRegime:
    """Classify the current timestamp using only historical bars."""

    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    if len(ordered) < 24:
        return MarketRegime("unknown", 0.0, 0.0, 0.5, 1.0, 0.5, 0.0)

    recent = ordered[-min(len(ordered), 96) :]
    short = ordered[-min(len(ordered), 24) :]
    long = ordered[-min(len(ordered), 240) :]
    latest = ordered[-1]
    sma_24 = _safe_mean([bar.close for bar in short])
    sma_96 = _safe_mean([bar.close for bar in recent])
    trend_score = latest.close / sma_96 - 1 if sma_96 > 0 else 0.0
    efficiency = _efficiency_ratio(recent)
    vol_pct = _volatility_percentile(ordered)
    ranges = [(bar.high - bar.low) / bar.close for bar in long if bar.close > 0]
    recent_ranges = ranges[-min(12, len(ranges)) :]
    compression_ratio = _safe_mean(recent_ranges) / _safe_mean(ranges) if _safe_mean(ranges) > 0 else 1.0
    high = max(bar.high for bar in recent)
    low = min(bar.low for bar in recent)
    width = high - low
    range_position = (latest.close - low) / width if width > 0 else 0.5
    returns_24 = _returns(short)
    vol_24 = pstdev(returns_24) if len(returns_24) > 1 else 0.0
    latest_return = returns_24[-1] if returns_24 else 0.0
    shock_score = abs(latest_return) / vol_24 if vol_24 > 0 else 0.0

    if shock_score >= 2.5 and vol_pct >= 0.60:
        label = "liquidation_reversal"
    elif compression_ratio <= 0.70 and vol_pct <= 0.45:
        label = "squeeze_breakout"
    elif vol_pct >= 0.80 and efficiency < 0.25:
        label = "high_volatility_chop"
    elif abs(trend_score) >= 0.012 and efficiency >= 0.35 and abs(latest.close / sma_24 - 1) >= 0.002:
        label = "trend_day"
    else:
        label = "range_day"
    return MarketRegime(
        label=label,
        trend_score=trend_score,
        efficiency_ratio=efficiency,
        volatility_percentile=vol_pct,
        compression_ratio=compression_ratio,
        range_position=range_position,
        shock_score=shock_score,
    )
