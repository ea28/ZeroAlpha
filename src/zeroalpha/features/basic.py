"""Causal feature generation for event rows."""

from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
from math import cos, pi, sin, sqrt
from statistics import mean, pstdev

from zeroalpha.domain import Bar, CandidateEvent, MarketQuote
from zeroalpha.timeutils import ensure_utc


def _return(bars: list[Bar], periods: int) -> float | None:
    if len(bars) <= periods:
        return None
    return bars[-1].close / bars[-1 - periods].close - 1


def _return_since(causal_bars: list[Bar], lookback: timedelta) -> float | None:
    if len(causal_bars) < 2:
        return None
    timestamps = [bar.timestamp_utc for bar in causal_bars]
    target = causal_bars[-1].timestamp_utc - lookback
    idx = bisect_right(timestamps, ensure_utc(target)) - 1
    if idx < 0 or idx >= len(causal_bars) - 1:
        return None
    base = causal_bars[idx].close
    return causal_bars[-1].close / base - 1 if base > 0 else None


def _window_since(causal_bars: list[Bar], lookback: timedelta) -> list[Bar]:
    if not causal_bars:
        return []
    cutoff = causal_bars[-1].timestamp_utc - lookback
    idx = bisect_right([bar.timestamp_utc for bar in causal_bars], ensure_utc(cutoff)) - 1
    start = max(idx, 0)
    return causal_bars[start:]


def _returns(bars: list[Bar]) -> list[float]:
    return [bars[i].close / bars[i - 1].close - 1 for i in range(1, len(bars))]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _rolling_return_features(causal_bars: list[Bar]) -> dict[str, float]:
    periods = (1, 2, 3, 4, 6, 12, 24, 48, 72, 168)
    return {f"return_{period}": _return(causal_bars, period) or 0.0 for period in periods}


def _duration_return_features(causal_bars: list[Bar]) -> dict[str, float]:
    lookbacks = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "12h": timedelta(hours=12),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
        "168h": timedelta(hours=168),
    }
    return {
        f"return_elapsed_{name}": _return_since(causal_bars, lookback) or 0.0
        for name, lookback in lookbacks.items()
    }


def _volatility_features(causal_bars: list[Bar]) -> dict[str, float]:
    all_returns = _returns(causal_bars)
    features: dict[str, float] = {}
    for period in (6, 12, 24, 72, 168):
        recent = all_returns[-period:] if len(all_returns) >= period else all_returns
        vol = _safe_stdev(recent)
        features[f"realized_vol_{period}"] = vol
        features[f"realized_vol_annualized_proxy_{period}"] = vol * sqrt(period) if vol else 0.0
        if recent:
            downside = [value for value in recent if value < 0]
            upside = [value for value in recent if value > 0]
            features[f"downside_vol_{period}"] = _safe_stdev(downside)
            features[f"upside_vol_{period}"] = _safe_stdev(upside)
            features[f"momentum_consistency_{period}"] = sum(value > 0 for value in recent) / len(recent)
        else:
            features[f"downside_vol_{period}"] = 0.0
            features[f"upside_vol_{period}"] = 0.0
            features[f"momentum_consistency_{period}"] = 0.0
    if features["realized_vol_168"] > 0:
        features["volatility_ratio_24_to_168"] = features["realized_vol_24"] / features["realized_vol_168"]
    else:
        features["volatility_ratio_24_to_168"] = 0.0
    return features


def _duration_volatility_features(causal_bars: list[Bar]) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, lookback in {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "12h": timedelta(hours=12),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
        "168h": timedelta(hours=168),
    }.items():
        window = _window_since(causal_bars, lookback)
        returns = _returns(window)
        features[f"realized_vol_elapsed_{name}"] = _safe_stdev(returns)
        if returns:
            features[f"momentum_consistency_elapsed_{name}"] = sum(value > 0 for value in returns) / len(returns)
        else:
            features[f"momentum_consistency_elapsed_{name}"] = 0.0
    base = features["realized_vol_elapsed_168h"]
    features["volatility_ratio_elapsed_24h_to_168h"] = (
        features["realized_vol_elapsed_24h"] / base if base > 0 else 0.0
    )
    return features


def _range_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for period in (24, 72, 168):
        window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
        highs = [bar.high for bar in window]
        lows = [bar.low for bar in window]
        ranges = [(bar.high - bar.low) / bar.close for bar in window]
        high = max(highs)
        low = min(lows)
        width = high - low
        features[f"drawdown_from_{period}_high"] = latest.close / high - 1
        features[f"distance_from_{period}_low"] = latest.close / low - 1
        features[f"range_position_{period}"] = (latest.close - low) / width if width > 0 else 0.5
        features[f"range_width_{period}"] = width / latest.close
        features[f"range_compression_{period}"] = _safe_mean(ranges[-min(8, len(ranges)) :]) / _safe_mean(ranges) if _safe_mean(ranges) > 0 else 0.0
    return features


def _duration_range_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for name, lookback in {
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
        "168h": timedelta(hours=168),
    }.items():
        window = _window_since(causal_bars, lookback)
        highs = [bar.high for bar in window]
        lows = [bar.low for bar in window]
        ranges = [(bar.high - bar.low) / bar.close for bar in window if bar.close > 0]
        high = max(highs)
        low = min(lows)
        width = high - low
        features[f"drawdown_from_elapsed_{name}_high"] = latest.close / high - 1 if high > 0 else 0.0
        features[f"distance_from_elapsed_{name}_low"] = latest.close / low - 1 if low > 0 else 0.0
        features[f"range_position_elapsed_{name}"] = (latest.close - low) / width if width > 0 else 0.5
        features[f"range_width_elapsed_{name}"] = width / latest.close if latest.close > 0 else 0.0
        features[f"range_compression_elapsed_{name}"] = (
            _safe_mean(ranges[-min(8, len(ranges)) :]) / _safe_mean(ranges)
            if _safe_mean(ranges) > 0
            else 0.0
        )
    return features


def _trend_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for period in (8, 21, 72, 168):
        window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
        sma = _safe_mean([bar.close for bar in window])
        features[f"sma_distance_{period}"] = latest.close / sma - 1 if sma > 0 else 0.0
        if len(window) > 1:
            features[f"slope_{period}"] = window[-1].close / window[0].close - 1
        else:
            features[f"slope_{period}"] = 0.0
    return features


def _duration_trend_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for name, lookback in {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
        "168h": timedelta(hours=168),
    }.items():
        window = _window_since(causal_bars, lookback)
        sma = _safe_mean([bar.close for bar in window])
        features[f"sma_distance_elapsed_{name}"] = latest.close / sma - 1 if sma > 0 else 0.0
        features[f"slope_elapsed_{name}"] = window[-1].close / window[0].close - 1 if len(window) > 1 else 0.0
    return features


def _liquidity_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for period in (24, 72, 168):
        window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
        volume_mean = _safe_mean([bar.volume for bar in window])
        features[f"volume_ratio_{period}"] = latest.volume / volume_mean if volume_mean > 0 else 0.0
        if latest.trade_count is not None and any(bar.trade_count is not None for bar in window):
            count_mean = _safe_mean([float(bar.trade_count or 0) for bar in window])
            features[f"trade_count_ratio_{period}"] = float(latest.trade_count) / count_mean if count_mean > 0 else 0.0
        else:
            features[f"trade_count_ratio_{period}"] = 0.0
    return features


def _duration_liquidity_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for name, lookback in {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
    }.items():
        window = _window_since(causal_bars, lookback)
        volume_mean = _safe_mean([bar.volume for bar in window])
        features[f"volume_ratio_elapsed_{name}"] = latest.volume / volume_mean if volume_mean > 0 else 0.0
        if latest.trade_count is not None and any(bar.trade_count is not None for bar in window):
            count_mean = _safe_mean([float(bar.trade_count or 0) for bar in window])
            features[f"trade_count_ratio_elapsed_{name}"] = (
                float(latest.trade_count) / count_mean if count_mean > 0 else 0.0
            )
        else:
            features[f"trade_count_ratio_elapsed_{name}"] = 0.0
    return features


def _market_structure_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    latest_range = latest.high - latest.low
    body = latest.close - latest.open
    body_abs = abs(body)
    close_position = (latest.close - latest.low) / latest_range if latest_range > 0 else 0.5
    upper_wick = latest.high - max(latest.open, latest.close)
    lower_wick = min(latest.open, latest.close) - latest.low
    features: dict[str, float] = {
        "candle_body_to_range": body_abs / latest_range if latest_range > 0 else 0.0,
        "candle_signed_body_to_range": body / latest_range if latest_range > 0 else 0.0,
        "candle_close_position": close_position,
        "upper_wick_to_range": upper_wick / latest_range if latest_range > 0 else 0.0,
        "lower_wick_to_range": lower_wick / latest_range if latest_range > 0 else 0.0,
    }
    all_returns = _returns(causal_bars)
    for period in (12, 24, 72):
        window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
        previous = window[:-1]
        ranges = [(bar.high - bar.low) / bar.close for bar in window if bar.close > 0]
        previous_ranges = [(bar.high - bar.low) / bar.close for bar in previous if bar.close > 0]
        returns = all_returns[-period:] if len(all_returns) >= period else all_returns
        previous_high = max((bar.high for bar in previous), default=latest.high)
        previous_low = min((bar.low for bar in previous), default=latest.low)
        previous_volumes = [bar.volume for bar in previous]
        volume_mean = _safe_mean(previous_volumes)
        volume_stdev = _safe_stdev(previous_volumes)
        range_mean = _safe_mean(previous_ranges)
        downside = abs(_safe_mean([value for value in returns if value < 0]))
        upside = _safe_mean([value for value in returns if value > 0])
        features[f"breakout_close_through_{period}_high"] = (
            latest.close / previous_high - 1 if previous_high > 0 else 0.0
        )
        features[f"breakdown_close_through_{period}_low"] = (
            latest.close / previous_low - 1 if previous_low > 0 else 0.0
        )
        features[f"range_expansion_ratio_{period}"] = (
            ((latest.high - latest.low) / latest.close) / range_mean
            if latest.close > 0 and range_mean > 0
            else 0.0
        )
        features[f"volume_zscore_{period}"] = (
            (latest.volume - volume_mean) / volume_stdev
            if len(previous) > 1 and volume_stdev > 0
            else 0.0
        )
        features[f"participation_ratio_{period}"] = latest.volume / volume_mean if volume_mean > 0 else 0.0
        features[f"realized_skew_proxy_{period}"] = (
            upside / downside if downside > 0 else (upside / 1e-9 if upside > 0 else 0.0)
        )
        features[f"range_persistence_{period}"] = (
            _safe_mean(ranges[-min(4, len(ranges)) :]) / _safe_mean(ranges)
            if _safe_mean(ranges) > 0
            else 0.0
        )
    return features


def _calendar_features(causal_bars: list[Bar]) -> dict[str, float]:
    ts = causal_bars[-1].timestamp_utc
    hour = ts.hour
    weekday = ts.weekday()
    hour_angle = 2 * pi * hour / 24
    weekday_angle = 2 * pi * weekday / 7
    return {
        "utc_hour": float(hour),
        "day_of_week": float(weekday),
        "utc_hour_sin": sin(hour_angle),
        "utc_hour_cos": cos(hour_angle),
        "day_of_week_sin": sin(weekday_angle),
        "day_of_week_cos": cos(weekday_angle),
        "weekend": 1.0 if weekday >= 5 else 0.0,
        "us_equity_session": 1.0 if weekday < 5 and 14 <= hour < 21 else 0.0,
        "europe_session": 1.0 if 7 <= hour < 16 else 0.0,
        "asia_session": 1.0 if hour >= 23 or hour < 8 else 0.0,
    }


def _microstructure_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {
        "has_taker_flow": 0.0,
        "taker_buy_volume_share": 0.0,
        "order_book_imbalance": 0.0,
        "perp_basis_bps": 0.0,
        "funding_rate": 0.0,
        "open_interest_change": 0.0,
        "liquidation_pressure": 0.0,
        "quote_spread_bps": 0.0,
    }
    extra = latest.extra or {}
    taker_buy_base = extra.get("taker_buy_base_volume", extra.get("taker_buy_volume"))
    if isinstance(taker_buy_base, int | float) and latest.volume > 0:
        features["has_taker_flow"] = 1.0
        features["taker_buy_volume_share"] = max(0.0, min(1.0, float(taker_buy_base) / latest.volume))
    for source_key, feature_key in (
        ("order_book_imbalance", "order_book_imbalance"),
        ("perp_basis_bps", "perp_basis_bps"),
        ("funding_rate", "funding_rate"),
        ("open_interest_change", "open_interest_change"),
        ("liquidation_pressure", "liquidation_pressure"),
        ("quote_spread_bps", "quote_spread_bps"),
    ):
        value = extra.get(source_key)
        if isinstance(value, int | float):
            features[feature_key] = float(value)
    return features


def build_event_features(
    event: CandidateEvent,
    bars: list[Bar],
    quote: MarketQuote | None = None,
) -> dict[str, float | str]:
    causal_bars = sorted(
        [bar for bar in bars if bar.symbol == event.symbol and bar.timestamp_utc <= event.timestamp_utc],
        key=lambda bar: bar.timestamp_utc,
    )
    if len(causal_bars) < 2:
        raise ValueError("not enough causal bars for features")

    latest = causal_bars[-1]
    features: dict[str, float | str] = {
        "candidate_type": event.candidate_type,
        "side": event.side.value,
        "signal_strength": event.signal_strength,
        "bar_close": latest.close,
    }
    features.update(_rolling_return_features(causal_bars))
    features.update(_duration_return_features(causal_bars))
    features.update(_volatility_features(causal_bars))
    features.update(_duration_volatility_features(causal_bars))
    features.update(_range_features(causal_bars))
    features.update(_duration_range_features(causal_bars))
    features.update(_trend_features(causal_bars))
    features.update(_duration_trend_features(causal_bars))
    features.update(_liquidity_features(causal_bars))
    features.update(_duration_liquidity_features(causal_bars))
    features.update(_market_structure_features(causal_bars))
    features.update(_calendar_features(causal_bars))
    features.update(_microstructure_features(causal_bars))
    if quote is not None:
        features["ibkr_spread_bps"] = quote.spread_bps
        features["ibkr_quote_age_ms"] = quote.quote_age_ms()
    return features
