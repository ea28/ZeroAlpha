"""Causal feature generation for event rows."""

from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
from math import cos, log1p, pi, sin, sqrt
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


def _safe_last(values: list[float], default: float = 0.0) -> float:
    return values[-1] if values else default


def _bar_dollar_volume(bar: Bar) -> float:
    if bar.quote_volume is not None:
        return max(0.0, float(bar.quote_volume))
    price = bar.vwap or bar.close
    return max(0.0, float(price) * float(bar.volume))


def _bar_taker_buy_share(bar: Bar) -> float | None:
    extra = bar.extra or {}
    taker_buy_base = extra.get("taker_buy_base_volume", extra.get("taker_buy_volume"))
    if isinstance(taker_buy_base, int | float) and bar.volume > 0:
        return max(0.0, min(1.0, float(taker_buy_base) / bar.volume))
    return None


def _bar_signed_volume(bar: Bar) -> float:
    taker_share = _bar_taker_buy_share(bar)
    if taker_share is not None:
        return bar.volume * (2 * taker_share - 1)
    if bar.close > bar.open:
        return bar.volume
    if bar.close < bar.open:
        return -bar.volume
    return 0.0


def _bar_signed_dollar_volume(bar: Bar) -> float:
    volume = _bar_signed_volume(bar)
    price = bar.vwap or bar.close
    return volume * price


def _bar_trade_count(bar: Bar) -> float:
    return float(bar.trade_count or 0)


def _extra_float(extra: dict, *keys: str) -> float | None:
    for key in keys:
        value = extra.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


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


def _ema_series(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    ema = values[0]
    rows = [ema]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
        rows.append(ema)
    return rows


def _rsi(values: list[float], period: int) -> float:
    if len(values) <= period:
        return 50.0
    deltas = [values[idx] - values[idx - 1] for idx in range(1, len(values))]
    recent = deltas[-period:]
    gains = [max(delta, 0.0) for delta in recent]
    losses = [abs(min(delta, 0.0)) for delta in recent]
    avg_gain = _safe_mean(gains)
    avg_loss = _safe_mean(losses)
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _technical_indicator_features(causal_bars: list[Bar]) -> dict[str, float]:
    closes = [bar.close for bar in causal_bars]
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    for span in (8, 21, 55, 144):
        ema = _safe_last(_ema_series(closes, span), latest.close)
        features[f"ema_distance_{span}"] = latest.close / ema - 1 if ema > 0 else 0.0
    ema_fast = _safe_last(_ema_series(closes, 12), latest.close)
    ema_slow = _safe_last(_ema_series(closes, 26), latest.close)
    macd = ema_fast - ema_slow
    macd_signal = _safe_last(_ema_series([fast - slow for fast, slow in zip(_ema_series(closes, 12), _ema_series(closes, 26), strict=True)], 9))
    features["macd_line_bps"] = 10_000 * macd / latest.close if latest.close > 0 else 0.0
    features["macd_signal_bps"] = 10_000 * macd_signal / latest.close if latest.close > 0 else 0.0
    features["macd_histogram_bps"] = 10_000 * (macd - macd_signal) / latest.close if latest.close > 0 else 0.0
    for period in (14, 28):
        rsi = _rsi(closes, period)
        features[f"rsi_{period}"] = rsi
        features[f"rsi_{period}_centered"] = (rsi - 50.0) / 50.0
    for period in (20, 72):
        window = closes[-period:] if len(closes) >= period else closes
        center = _safe_mean(window)
        width = _safe_stdev(window)
        features[f"bollinger_z_{period}"] = (
            (latest.close - center) / width if width > 0 else 0.0
        )
        features[f"bollinger_width_bps_{period}"] = (
            10_000 * (2.0 * width) / center if center > 0 else 0.0
        )
    true_ranges: list[float] = []
    for idx, bar in enumerate(causal_bars):
        prev_close = causal_bars[idx - 1].close if idx > 0 else bar.close
        true_ranges.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
    for period in (14, 72):
        atr = _safe_mean(true_ranges[-period:])
        features[f"atr_bps_{period}"] = 10_000 * atr / latest.close if latest.close > 0 else 0.0
        prior_atr = _safe_mean(true_ranges[-2 * period : -period]) if len(true_ranges) >= 2 * period else 0.0
        features[f"atr_shock_{period}"] = atr / prior_atr if prior_atr > 0 else 0.0
    recent_returns = _returns(causal_bars)
    if recent_returns:
        latest_abs_return = abs(recent_returns[-1])
        for period in (24, 72):
            vol = _safe_stdev(recent_returns[-period:])
            features[f"realized_vol_shock_{period}"] = latest_abs_return / vol if vol > 0 else 0.0
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
    latest_dollar_volume = _bar_dollar_volume(latest)
    latest_trade_count = _bar_trade_count(latest)
    latest_signed_volume = _bar_signed_volume(latest)
    latest_signed_dollar_volume = _bar_signed_dollar_volume(latest)
    latest_volume_per_trade = latest.volume / latest_trade_count if latest_trade_count > 0 else 0.0
    latest_dollar_volume_per_trade = (
        latest_dollar_volume / latest_trade_count if latest_trade_count > 0 else 0.0
    )
    latest_taker_share = _bar_taker_buy_share(latest)
    for period in (24, 72, 168):
        window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
        volume_mean = _safe_mean([bar.volume for bar in window])
        volume_std = _safe_stdev([bar.volume for bar in window])
        features[f"volume_ratio_{period}"] = latest.volume / volume_mean if volume_mean > 0 else 0.0
        features[f"volume_surprise_z_{period}"] = (
            (latest.volume - volume_mean) / volume_std if volume_std > 0 else 0.0
        )
        dollar_volume_mean = _safe_mean([_bar_dollar_volume(bar) for bar in window])
        dollar_volume_std = _safe_stdev([_bar_dollar_volume(bar) for bar in window])
        features[f"dollar_volume_ratio_{period}"] = (
            latest_dollar_volume / dollar_volume_mean if dollar_volume_mean > 0 else 0.0
        )
        features[f"dollar_volume_surprise_z_{period}"] = (
            (latest_dollar_volume - dollar_volume_mean) / dollar_volume_std
            if dollar_volume_std > 0
            else 0.0
        )
        window_volume = sum(bar.volume for bar in window)
        window_dollar_volume = sum(_bar_dollar_volume(bar) for bar in window)
        signed_volume_sum = sum(_bar_signed_volume(bar) for bar in window)
        signed_dollar_volume_sum = sum(_bar_signed_dollar_volume(bar) for bar in window)
        features[f"signed_volume_imbalance_{period}"] = (
            signed_volume_sum / window_volume if window_volume > 0 else 0.0
        )
        features[f"signed_dollar_volume_imbalance_{period}"] = (
            signed_dollar_volume_sum / window_dollar_volume if window_dollar_volume > 0 else 0.0
        )
        features[f"signed_volume_impulse_{period}"] = (
            latest_signed_volume / window_volume if window_volume > 0 else 0.0
        )
        features[f"signed_dollar_volume_impulse_{period}"] = (
            latest_signed_dollar_volume / window_dollar_volume if window_dollar_volume > 0 else 0.0
        )
        if latest.trade_count is not None and any(bar.trade_count is not None for bar in window):
            trade_counts = [float(bar.trade_count or 0) for bar in window]
            count_mean = _safe_mean(trade_counts)
            count_std = _safe_stdev(trade_counts)
            features[f"trade_count_ratio_{period}"] = float(latest.trade_count) / count_mean if count_mean > 0 else 0.0
            features[f"trade_intensity_surprise_z_{period}"] = (
                (float(latest.trade_count) - count_mean) / count_std if count_std > 0 else 0.0
            )
        else:
            features[f"trade_count_ratio_{period}"] = 0.0
            features[f"trade_intensity_surprise_z_{period}"] = 0.0
        volume_per_trade_mean = _safe_mean(
            [
                bar.volume / _bar_trade_count(bar)
                for bar in window
                if _bar_trade_count(bar) > 0
            ]
        )
        dollar_volume_per_trade_mean = _safe_mean(
            [
                _bar_dollar_volume(bar) / _bar_trade_count(bar)
                for bar in window
                if _bar_trade_count(bar) > 0
            ]
        )
        features[f"volume_per_trade_ratio_{period}"] = (
            latest_volume_per_trade / volume_per_trade_mean if volume_per_trade_mean > 0 else 0.0
        )
        features[f"dollar_volume_per_trade_ratio_{period}"] = (
            latest_dollar_volume_per_trade / dollar_volume_per_trade_mean
            if dollar_volume_per_trade_mean > 0
            else 0.0
        )
        taker_shares = [share for bar in window if (share := _bar_taker_buy_share(bar)) is not None]
        taker_mean = _safe_mean(taker_shares)
        features[f"taker_buy_share_mean_{period}"] = taker_mean
        features[f"taker_buy_share_delta_{period}"] = (
            latest_taker_share - taker_mean if latest_taker_share is not None and taker_shares else 0.0
        )
    return features


def _duration_liquidity_features(causal_bars: list[Bar]) -> dict[str, float]:
    latest = causal_bars[-1]
    features: dict[str, float] = {}
    latest_dollar_volume = _bar_dollar_volume(latest)
    latest_trade_count = _bar_trade_count(latest)
    latest_signed_volume = _bar_signed_volume(latest)
    latest_signed_dollar_volume = _bar_signed_dollar_volume(latest)
    latest_taker_share = _bar_taker_buy_share(latest)
    for name, lookback in {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
    }.items():
        window = _window_since(causal_bars, lookback)
        volume_mean = _safe_mean([bar.volume for bar in window])
        volume_std = _safe_stdev([bar.volume for bar in window])
        features[f"volume_ratio_elapsed_{name}"] = latest.volume / volume_mean if volume_mean > 0 else 0.0
        features[f"volume_surprise_z_elapsed_{name}"] = (
            (latest.volume - volume_mean) / volume_std if volume_std > 0 else 0.0
        )
        dollar_volume_mean = _safe_mean([_bar_dollar_volume(bar) for bar in window])
        dollar_volume_std = _safe_stdev([_bar_dollar_volume(bar) for bar in window])
        features[f"dollar_volume_ratio_elapsed_{name}"] = (
            latest_dollar_volume / dollar_volume_mean if dollar_volume_mean > 0 else 0.0
        )
        features[f"dollar_volume_surprise_z_elapsed_{name}"] = (
            (latest_dollar_volume - dollar_volume_mean) / dollar_volume_std
            if dollar_volume_std > 0
            else 0.0
        )
        window_volume = sum(bar.volume for bar in window)
        window_dollar_volume = sum(_bar_dollar_volume(bar) for bar in window)
        signed_volume_sum = sum(_bar_signed_volume(bar) for bar in window)
        signed_dollar_volume_sum = sum(_bar_signed_dollar_volume(bar) for bar in window)
        features[f"signed_volume_imbalance_elapsed_{name}"] = (
            signed_volume_sum / window_volume if window_volume > 0 else 0.0
        )
        features[f"signed_dollar_volume_imbalance_elapsed_{name}"] = (
            signed_dollar_volume_sum / window_dollar_volume if window_dollar_volume > 0 else 0.0
        )
        features[f"signed_volume_impulse_elapsed_{name}"] = (
            latest_signed_volume / window_volume if window_volume > 0 else 0.0
        )
        features[f"signed_dollar_volume_impulse_elapsed_{name}"] = (
            latest_signed_dollar_volume / window_dollar_volume if window_dollar_volume > 0 else 0.0
        )
        if latest.trade_count is not None and any(bar.trade_count is not None for bar in window):
            trade_counts = [float(bar.trade_count or 0) for bar in window]
            count_mean = _safe_mean(trade_counts)
            count_std = _safe_stdev(trade_counts)
            features[f"trade_count_ratio_elapsed_{name}"] = (
                float(latest.trade_count) / count_mean if count_mean > 0 else 0.0
            )
            features[f"trade_intensity_surprise_z_elapsed_{name}"] = (
                (float(latest.trade_count) - count_mean) / count_std if count_std > 0 else 0.0
            )
        else:
            features[f"trade_count_ratio_elapsed_{name}"] = 0.0
            features[f"trade_intensity_surprise_z_elapsed_{name}"] = 0.0
        volume_per_trade_mean = _safe_mean(
            [
                bar.volume / _bar_trade_count(bar)
                for bar in window
                if _bar_trade_count(bar) > 0
            ]
        )
        latest_volume_per_trade = latest.volume / latest_trade_count if latest_trade_count > 0 else 0.0
        features[f"volume_per_trade_ratio_elapsed_{name}"] = (
            latest_volume_per_trade / volume_per_trade_mean if volume_per_trade_mean > 0 else 0.0
        )
        taker_shares = [share for bar in window if (share := _bar_taker_buy_share(bar)) is not None]
        taker_mean = _safe_mean(taker_shares)
        features[f"taker_buy_share_mean_elapsed_{name}"] = taker_mean
        features[f"taker_buy_share_delta_elapsed_{name}"] = (
            latest_taker_share - taker_mean if latest_taker_share is not None and taker_shares else 0.0
        )
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


def _bar_top_book_spread_bps(bar: Bar) -> float | None:
    extra = bar.extra or {}
    bid = _extra_float(extra, "bid", "best_bid", "ibkr_bid", "l1_bid")
    ask = _extra_float(extra, "ask", "best_ask", "ibkr_ask", "l1_ask")
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2.0
    return 10_000 * (ask - bid) / midpoint if midpoint > 0 else None


def _microstructure_features(causal_bars: list[Bar], event: CandidateEvent) -> dict[str, float]:
    latest = causal_bars[-1]
    dollar_volume = _bar_dollar_volume(latest)
    trade_count = _bar_trade_count(latest)
    taker_buy_share = _bar_taker_buy_share(latest)
    signed_taker_imbalance = 2 * taker_buy_share - 1 if taker_buy_share is not None else 0.0
    features: dict[str, float] = {
        "has_taker_flow": 0.0,
        "taker_buy_volume_share": 0.0,
        "signed_taker_buy_imbalance": 0.0,
        "side_taker_buy_imbalance": 0.0,
        "signed_volume": _bar_signed_volume(latest),
        "signed_dollar_volume": _bar_signed_dollar_volume(latest),
        "side_signed_volume": (
            _bar_signed_volume(latest) if event.side.value == "BUY" else -_bar_signed_volume(latest)
        ),
        "side_signed_dollar_volume": (
            _bar_signed_dollar_volume(latest)
            if event.side.value == "BUY"
            else -_bar_signed_dollar_volume(latest)
        ),
        "order_book_imbalance": 0.0,
        "perp_basis_bps": 0.0,
        "funding_rate": 0.0,
        "open_interest_change": 0.0,
        "liquidation_pressure": 0.0,
        "quote_spread_bps": 0.0,
        "has_quote_volume": 1.0 if latest.quote_volume is not None else 0.0,
        "dollar_volume": dollar_volume,
        "dollar_volume_log": log1p(dollar_volume),
        "volume_per_trade": latest.volume / trade_count if trade_count > 0 else 0.0,
        "dollar_volume_per_trade": dollar_volume / trade_count if trade_count > 0 else 0.0,
        "vwap_distance_bps": (
            10_000 * (latest.close / latest.vwap - 1) if latest.vwap and latest.vwap > 0 else 0.0
        ),
    }
    if taker_buy_share is not None:
        features["has_taker_flow"] = 1.0
        features["taker_buy_volume_share"] = taker_buy_share
        features["signed_taker_buy_imbalance"] = signed_taker_imbalance
        features["side_taker_buy_imbalance"] = (
            signed_taker_imbalance if event.side.value == "BUY" else -signed_taker_imbalance
        )
    extra = latest.extra or {}
    bid = _extra_float(extra, "bid", "best_bid", "ibkr_bid", "l1_bid")
    ask = _extra_float(extra, "ask", "best_ask", "ibkr_ask", "l1_ask")
    bid_size = _extra_float(extra, "bid_size", "best_bid_size", "ibkr_bid_size", "l1_bid_size")
    ask_size = _extra_float(extra, "ask_size", "best_ask_size", "ibkr_ask_size", "l1_ask_size")
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        midpoint = (bid + ask) / 2.0
        spread_bps = 10_000 * (ask - bid) / midpoint if midpoint > 0 else 0.0
        features["top_book_midpoint"] = midpoint
        features["top_book_spread_bps"] = spread_bps
        features["quote_spread_bps"] = spread_bps
        for period in (24, 72, 168):
            window = causal_bars[-period:] if len(causal_bars) >= period else causal_bars
            spread_values = [value for bar in window if (value := _bar_top_book_spread_bps(bar)) is not None]
            spread_mean = _safe_mean(spread_values)
            spread_std = _safe_stdev(spread_values)
            features[f"top_book_spread_z_{period}"] = (
                (spread_bps - spread_mean) / spread_std if spread_std > 0 else 0.0
            )
        features["top_book_midpoint_distance_bps"] = (
            10_000 * (latest.close / midpoint - 1) if midpoint > 0 else 0.0
        )
        if bid_size is not None and ask_size is not None:
            depth = bid_size + ask_size
            features["top_book_depth"] = depth
            features["top_book_depth_log"] = log1p(max(depth, 0.0))
            features["top_book_size_imbalance"] = (
                (bid_size - ask_size) / depth if depth > 0 else 0.0
            )
            microprice = (ask * bid_size + bid * ask_size) / depth if depth > 0 else midpoint
            features["microprice"] = microprice
            features["microprice_distance_bps"] = (
                10_000 * (microprice / midpoint - 1) if midpoint > 0 else 0.0
            )
            features["side_microprice_distance_bps"] = (
                features["microprice_distance_bps"]
                if event.side.value == "BUY"
                else -features["microprice_distance_bps"]
            )
            features["order_book_imbalance"] = features["top_book_size_imbalance"]
    bid_depth = _extra_float(extra, "bid_depth", "bid_depth_5bps", "l2_bid_depth")
    ask_depth = _extra_float(extra, "ask_depth", "ask_depth_5bps", "l2_ask_depth")
    if bid_depth is not None and ask_depth is not None:
        depth = bid_depth + ask_depth
        features["l2_depth_total"] = depth
        features["l2_depth_imbalance"] = (bid_depth - ask_depth) / depth if depth > 0 else 0.0
        features["side_l2_depth_imbalance"] = (
            features["l2_depth_imbalance"]
            if event.side.value == "BUY"
            else -features["l2_depth_imbalance"]
        )
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
    features.update(_technical_indicator_features(causal_bars))
    features.update(_duration_trend_features(causal_bars))
    features.update(_liquidity_features(causal_bars))
    features.update(_duration_liquidity_features(causal_bars))
    features.update(_market_structure_features(causal_bars))
    features.update(_calendar_features(causal_bars))
    features.update(_microstructure_features(causal_bars, event))
    if quote is not None:
        features["ibkr_spread_bps"] = quote.spread_bps
        features["ibkr_quote_age_ms"] = quote.quote_age_ms()
    return features
