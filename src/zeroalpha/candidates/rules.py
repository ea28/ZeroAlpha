"""Simple BTC long candidate rules for the first research pass."""

from __future__ import annotations

from statistics import mean, pstdev
from uuid import uuid5, NAMESPACE_URL

from zeroalpha.domain import Bar, CandidateEvent, Side


def _event_id(prefix: str, bar: Bar) -> str:
    return str(uuid5(NAMESPACE_URL, f"{prefix}:{bar.source}:{bar.symbol}:{bar.bar_size}:{bar.timestamp_utc}"))


def _close_return(a: Bar, b: Bar) -> float:
    return b.close / a.close - 1


def _returns(bars: list[Bar]) -> list[float]:
    return [_close_return(bars[i - 1], bars[i]) for i in range(1, len(bars))]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _volume_ratio(bars: list[Bar], lookback: int = 24) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    baseline = _safe_mean([bar.volume for bar in bars[-lookback - 1 : -1]])
    return bars[-1].volume / baseline if baseline > 0 else 0.0


def _trade_count_ratio(bars: list[Bar], lookback: int = 24) -> float:
    if len(bars) < lookback + 1 or bars[-1].trade_count is None:
        return 0.0
    counts = [bar.trade_count or 0 for bar in bars[-lookback - 1 : -1]]
    baseline = _safe_mean([float(value) for value in counts])
    return float(bars[-1].trade_count) / baseline if baseline > 0 else 0.0


def _efficiency_ratio(bars: list[Bar], lookback: int = 24) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    window = bars[-lookback - 1 :]
    displacement = abs(window[-1].close - window[0].close)
    path = sum(abs(window[i].close - window[i - 1].close) for i in range(1, len(window)))
    return displacement / path if path > 0 else 0.0


def _realized_volatility(bars: list[Bar], lookback: int = 24) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    recent = _returns(bars[-lookback - 1 :])
    return pstdev(recent) if len(recent) > 1 else 0.0


def _volatility_percentile(bars: list[Bar], lookback: int = 24, history: int = 240) -> float:
    if len(bars) < history + lookback + 1:
        return 0.5
    ranges = [(bar.high - bar.low) / bar.close for bar in bars[-history:]]
    current = _safe_mean(ranges[-lookback:])
    return sum(value <= current for value in ranges) / len(ranges)


def _higher_timeframe_trend_ok(bars: list[Bar]) -> bool:
    if len(bars) < 168:
        return True
    close = bars[-1].close
    sma_72 = _safe_mean([bar.close for bar in bars[-72:]])
    sma_168 = _safe_mean([bar.close for bar in bars[-168:]])
    return close > sma_72 and sma_72 >= sma_168 * 0.985


def _higher_timeframe_downtrend_ok(bars: list[Bar]) -> bool:
    if len(bars) < 168:
        return True
    close = bars[-1].close
    sma_72 = _safe_mean([bar.close for bar in bars[-72:]])
    sma_168 = _safe_mean([bar.close for bar in bars[-168:]])
    return close < sma_72 and sma_72 <= sma_168 * 1.015


def trend_pullback_reclaim_candidate(
    bars: list[Bar],
    *,
    max_holding_hours: int = 72,
) -> CandidateEvent | None:
    if len(bars) < 168:
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    previous = ordered[-2]
    sma_21 = _safe_mean([bar.close for bar in ordered[-21:]])
    sma_72 = _safe_mean([bar.close for bar in ordered[-72:]])
    sma_168 = _safe_mean([bar.close for bar in ordered[-168:]])
    recent_low = min(bar.low for bar in ordered[-8:])
    reclaimed_short_mean = previous.close <= sma_21 and latest.close > sma_21
    pulled_back_without_breaking_trend = recent_low <= sma_21 * 1.01 and recent_low >= sma_72 * 0.965
    trend_ok = latest.close > sma_72 and sma_72 > sma_168 * 1.005
    volume_ratio = _volume_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    efficiency = _efficiency_ratio(ordered)
    if (
        trend_ok
        and reclaimed_short_mean
        and pulled_back_without_breaking_trend
        and 0.12 <= vol_pct <= 0.75
        and volume_ratio >= 0.90
        and efficiency >= 0.16
    ):
        strength = (latest.close / sma_21 - 1) + (sma_72 / sma_168 - 1)
        return CandidateEvent(
            event_id=_event_id("trend_pullback_reclaim", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="trend_pullback_reclaim",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=strength,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "sma_distance_21": latest.close / sma_21 - 1,
                "sma_72_to_168": sma_72 / sma_168 - 1,
                "pullback_depth": recent_low / sma_21 - 1,
                "volume_ratio_24": volume_ratio,
                "volatility_percentile": vol_pct,
                "efficiency_ratio_24": efficiency,
            },
        )
    return None


def trend_continuation_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 72,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 72):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    range_high = max(bar.high for bar in prior)
    short_slope = ordered[-1].close - ordered[-8].close if len(ordered) >= 8 else 0
    medium_slope = ordered[-1].close - ordered[-21].close if len(ordered) >= 21 else 0
    volume_ratio = _volume_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    efficiency = _efficiency_ratio(ordered)
    breakout_extension = latest.close / range_high - 1 if range_high > 0 else 0.0
    if (
        latest.close > range_high
        and breakout_extension <= 0.018
        and short_slope > 0
        and medium_slope > 0
        and _higher_timeframe_trend_ok(ordered)
        and 0.18 <= vol_pct <= 0.80
        and volume_ratio >= 1.00
        and efficiency >= 0.25
    ):
        strength = (latest.close - range_high) / range_high
        return CandidateEvent(
            event_id=_event_id("trend_continuation", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="trend_continuation",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=strength,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "range_high": range_high,
                "breakout_extension": breakout_extension,
                "volume_ratio_24": volume_ratio,
                "volatility_percentile": vol_pct,
                "efficiency_ratio_24": efficiency,
            },
        )
    return None


def information_momentum_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 24,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 4, 168):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    four_bar_return = latest.close / ordered[-5].close - 1
    recent_returns = _returns(ordered[-lookback - 1 :])
    vol = pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    event_threshold = max(0.006, 1.25 * vol * 4**0.5)
    efficiency = _efficiency_ratio(ordered)
    volume_ratio = _volume_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    range_high = max(bar.high for bar in ordered[-lookback - 1 : -1])
    if (
        four_bar_return >= event_threshold
        and latest.close >= range_high * 0.995
        and _higher_timeframe_trend_ok(ordered)
        and efficiency >= 0.22
        and volume_ratio >= 0.75
        and 0.10 <= vol_pct <= 0.90
    ):
        return CandidateEvent(
            event_id=_event_id("information_momentum", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="information_momentum",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=four_bar_return / event_threshold,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "four_bar_return": four_bar_return,
                "event_threshold": event_threshold,
                "efficiency_ratio_24": efficiency,
                "volume_ratio_24": volume_ratio,
                "volatility_percentile": vol_pct,
            },
        )
    return None


def short_information_momentum_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 24,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 4, 168):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    four_bar_return = latest.close / ordered[-5].close - 1
    recent_returns = _returns(ordered[-lookback - 1 :])
    vol = pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    event_threshold = max(0.006, 1.25 * vol * 4**0.5)
    efficiency = _efficiency_ratio(ordered)
    volume_ratio = _volume_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    range_low = min(bar.low for bar in ordered[-lookback - 1 : -1])
    sma_72 = _safe_mean([bar.close for bar in ordered[-72:]])
    sma_168 = _safe_mean([bar.close for bar in ordered[-168:]])
    higher_timeframe_down = latest.close < sma_72 and sma_72 <= sma_168 * 1.015
    if (
        four_bar_return <= -event_threshold
        and latest.close <= range_low * 1.005
        and higher_timeframe_down
        and efficiency >= 0.22
        and volume_ratio >= 0.75
        and 0.10 <= vol_pct <= 0.95
    ):
        return CandidateEvent(
            event_id=_event_id("short_information_momentum", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="short_information_momentum",
            side=Side.SELL,
            bar_size=latest.bar_size,
            signal_strength=abs(four_bar_return) / event_threshold,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "four_bar_return": four_bar_return,
                "event_threshold": event_threshold,
                "efficiency_ratio_24": efficiency,
                "volume_ratio_24": volume_ratio,
                "volatility_percentile": vol_pct,
            },
        )
    return None


def volatility_breakout_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    compression_quantile: float = 0.35,
    max_holding_hours: int = 72,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    ranges = [(bar.high - bar.low) / bar.close for bar in prior]
    recent_compression = _safe_mean(ranges[-8:])
    sorted_ranges = sorted(ranges)
    threshold = sorted_ranges[int((len(sorted_ranges) - 1) * compression_quantile)]
    compression_high = max(bar.high for bar in prior[-8:])
    volume_ratio = _volume_ratio(ordered)
    trade_ratio = _trade_count_ratio(ordered)
    participation_ok = volume_ratio >= 1.10 or trade_ratio >= 1.10
    vol_pct = _volatility_percentile(ordered)
    if (
        recent_compression <= threshold
        and latest.close > compression_high
        and _higher_timeframe_trend_ok(ordered)
        and participation_ok
        and vol_pct <= 0.75
    ):
        return CandidateEvent(
            event_id=_event_id("volatility_breakout", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="volatility_breakout",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=(latest.close - compression_high) / compression_high,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "compression_high": compression_high,
                "recent_compression": recent_compression,
                "compression_threshold": threshold,
                "volume_ratio_24": volume_ratio,
                "trade_count_ratio_24": trade_ratio,
                "volatility_percentile": vol_pct,
            },
        )
    return None


def volatility_breakout_retest_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    compression_quantile: float = 0.40,
    max_holding_hours: int = 48,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 3, 120):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 3 : -3]
    if not prior:
        return None
    ranges = [(bar.high - bar.low) / bar.close for bar in prior if bar.close > 0]
    if not ranges:
        return None
    sorted_ranges = sorted(ranges)
    threshold = sorted_ranges[int((len(sorted_ranges) - 1) * compression_quantile)]
    recent_compression = _safe_mean(ranges[-8:])
    compression_high = max(bar.high for bar in prior[-8:])
    breakout_bar = ordered[-2]
    held_retest = latest.low >= compression_high * 0.992 and latest.close > compression_high
    volume_ratio = _volume_ratio(ordered)
    trade_ratio = _trade_count_ratio(ordered)
    efficiency = _efficiency_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    if (
        recent_compression <= threshold
        and breakout_bar.close > compression_high
        and held_retest
        and _higher_timeframe_trend_ok(ordered)
        and (volume_ratio >= 0.95 or trade_ratio >= 0.95)
        and efficiency >= 0.20
        and vol_pct <= 0.80
    ):
        return CandidateEvent(
            event_id=_event_id("volatility_breakout_retest", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="volatility_breakout_retest",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=(latest.close - compression_high) / compression_high,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "compression_high": compression_high,
                "recent_compression": recent_compression,
                "compression_threshold": threshold,
                "volume_ratio_24": volume_ratio,
                "trade_count_ratio_24": trade_ratio,
                "efficiency_ratio_24": efficiency,
                "volatility_percentile": vol_pct,
            },
        )
    return None


def intraday_volatility_breakout_candidate(
    bars: list[Bar],
    *,
    lookback: int = 12,
    max_holding_hours: int = 12,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    ranges = [(bar.high - bar.low) / bar.close for bar in prior if bar.close > 0]
    if not ranges:
        return None
    short_compression = _safe_mean(ranges[-4:])
    median_range = sorted(ranges)[len(ranges) // 2]
    range_high = max(bar.high for bar in prior[-lookback:])
    extension = latest.close / range_high - 1 if range_high > 0 else 0.0
    volume_ratio = _volume_ratio(ordered, lookback=12)
    trade_ratio = _trade_count_ratio(ordered, lookback=12)
    efficiency = _efficiency_ratio(ordered, lookback=12)
    latest_range = (latest.high - latest.low) / latest.close if latest.close > 0 else 0.0
    close_position = (latest.close - latest.low) / (latest.high - latest.low) if latest.high > latest.low else 0.5
    if (
        short_compression <= median_range
        and 0.001 <= extension <= 0.018
        and latest.close > range_high
        and close_position >= 0.68
        and latest_range >= median_range * 0.80
        and (volume_ratio >= 0.95 or trade_ratio >= 0.95)
        and efficiency >= 0.20
    ):
        return CandidateEvent(
            event_id=_event_id("intraday_volatility_breakout", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="intraday_volatility_breakout",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=extension + max(0.0, volume_ratio - 1.0) * 0.001,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "range_high": range_high,
                "breakout_extension": extension,
                "short_compression": short_compression,
                "median_range": median_range,
                "volume_ratio_12": volume_ratio,
                "trade_count_ratio_12": trade_ratio,
                "efficiency_ratio_12": efficiency,
                "close_position": close_position,
            },
        )
    return None


def liquidity_sweep_reclaim_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 12,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 2, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    prior_low = min(bar.low for bar in prior)
    sma_8 = _safe_mean([bar.close for bar in ordered[-8:]])
    sma_24 = _safe_mean([bar.close for bar in ordered[-24:]])
    latest_range = latest.high - latest.low
    lower_wick = min(latest.open, latest.close) - latest.low
    close_position = (latest.close - latest.low) / latest_range if latest_range > 0 else 0.5
    volume_ratio = _volume_ratio(ordered, lookback=24)
    swept_low = latest.low < prior_low * 0.998
    reclaimed = latest.close > prior_low and latest.close > sma_8
    trend_not_broken = sma_8 >= sma_24 * 0.985
    if (
        swept_low
        and reclaimed
        and trend_not_broken
        and lower_wick / latest.close >= 0.003
        and close_position >= 0.58
        and volume_ratio >= 0.85
    ):
        strength = (latest.close / prior_low - 1) + lower_wick / latest.close
        return CandidateEvent(
            event_id=_event_id("liquidity_sweep_reclaim", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="liquidity_sweep_reclaim",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=strength,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "prior_low": prior_low,
                "sma_8_to_24": sma_8 / sma_24 - 1 if sma_24 > 0 else 0.0,
                "lower_wick": lower_wick / latest.close,
                "close_position": close_position,
                "volume_ratio_24": volume_ratio,
            },
        )
    return None


def short_trend_rejection_candidate(
    bars: list[Bar],
    *,
    max_holding_hours: int = 48,
) -> CandidateEvent | None:
    if len(bars) < 168:
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    previous = ordered[-2]
    sma_21 = _safe_mean([bar.close for bar in ordered[-21:]])
    sma_72 = _safe_mean([bar.close for bar in ordered[-72:]])
    recent_high = max(bar.high for bar in ordered[-8:])
    rejected_short_mean = previous.close >= sma_21 and latest.close < sma_21
    failed_without_breaking_downtrend = recent_high >= sma_21 * 0.99 and recent_high <= sma_72 * 1.035
    volume_ratio = _volume_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    efficiency = _efficiency_ratio(ordered)
    sma_72_gap = max(0.0, sma_72 / latest.close - 1) if latest.close > 0 else 0.0
    if (
        _higher_timeframe_downtrend_ok(ordered)
        and rejected_short_mean
        and failed_without_breaking_downtrend
        and 0.12 <= vol_pct <= 0.85
        and volume_ratio >= 0.90
        and efficiency >= 0.16
    ):
        return CandidateEvent(
            event_id=_event_id("short_trend_rejection", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="short_trend_rejection",
            side=Side.SELL,
            bar_size=latest.bar_size,
            signal_strength=(sma_21 / latest.close - 1) + sma_72_gap if latest.close > 0 else 0.0,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "sma_distance_21": latest.close / sma_21 - 1 if sma_21 > 0 else 0.0,
                "sma_72_distance": latest.close / sma_72 - 1 if sma_72 > 0 else 0.0,
                "failed_retest_height": recent_high / sma_21 - 1 if sma_21 > 0 else 0.0,
                "sma_72_gap": sma_72_gap,
                "volume_ratio_24": volume_ratio,
                "volatility_percentile": vol_pct,
                "efficiency_ratio_24": efficiency,
            },
        )
    return None


def short_volatility_breakdown_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    compression_quantile: float = 0.40,
    max_holding_hours: int = 48,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 120):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    ranges = [(bar.high - bar.low) / bar.close for bar in prior if bar.close > 0]
    if not ranges:
        return None
    sorted_ranges = sorted(ranges)
    threshold = sorted_ranges[int((len(sorted_ranges) - 1) * compression_quantile)]
    recent_compression = _safe_mean(ranges[-8:])
    compression_low = min(bar.low for bar in prior[-8:])
    volume_ratio = _volume_ratio(ordered)
    trade_ratio = _trade_count_ratio(ordered)
    efficiency = _efficiency_ratio(ordered)
    vol_pct = _volatility_percentile(ordered)
    breakdown_extension = compression_low / latest.close - 1 if latest.close > 0 else 0.0
    if (
        recent_compression <= threshold
        and latest.close < compression_low
        and breakdown_extension <= 0.018
        and _higher_timeframe_downtrend_ok(ordered)
        and (volume_ratio >= 1.00 or trade_ratio >= 1.00)
        and efficiency >= 0.22
        and vol_pct <= 0.85
    ):
        return CandidateEvent(
            event_id=_event_id("short_volatility_breakdown", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="short_volatility_breakdown",
            side=Side.SELL,
            bar_size=latest.bar_size,
            signal_strength=(compression_low - latest.close) / compression_low,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "compression_low": compression_low,
                "recent_compression": recent_compression,
                "compression_threshold": threshold,
                "breakdown_extension": breakdown_extension,
                "volume_ratio_24": volume_ratio,
                "trade_count_ratio_24": trade_ratio,
                "efficiency_ratio_24": efficiency,
                "volatility_percentile": vol_pct,
            },
        )
    return None


def capitulation_mean_reversion_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    shock_threshold: float = -0.03,
    max_holding_hours: int = 48,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 2, 72):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    previous = ordered[-2]
    shock = (previous.close - ordered[-lookback].close) / ordered[-lookback].close
    recent_returns = _returns(ordered[-lookback - 1 :])
    vol = pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    shock_z = shock / (vol * lookback**0.5) if vol > 0 else 0.0
    reclaim = latest.close > sum(bar.close for bar in ordered[-5:]) / 5
    lower_wick = (min(latest.open, latest.close) - latest.low) / latest.close
    volume_ratio = _volume_ratio(ordered)
    if shock <= shock_threshold and shock_z <= -1.25 and reclaim and lower_wick >= 0.002 and volume_ratio >= 0.75:
        return CandidateEvent(
            event_id=_event_id("capitulation_mean_reversion", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="capitulation_mean_reversion",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=abs(shock),
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "shock": shock,
                "shock_z": shock_z,
                "lower_wick": lower_wick,
                "volume_ratio_24": volume_ratio,
            },
        )
    return None


def active_breakout_continuation_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 4,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 12, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    range_high = max(bar.high for bar in prior)
    extension = latest.close / range_high - 1 if range_high > 0 else 0.0
    volume_ratio = _volume_ratio(ordered, lookback=min(lookback, 24))
    efficiency = _efficiency_ratio(ordered, lookback=min(lookback, 24))
    close_position = (latest.close - latest.low) / (latest.high - latest.low) if latest.high > latest.low else 0.5
    if 0.0005 <= extension <= 0.012 and close_position >= 0.62 and volume_ratio >= 0.80 and efficiency >= 0.12:
        return CandidateEvent(
            event_id=_event_id("active_breakout_continuation", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_breakout_continuation",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=extension + max(0.0, volume_ratio - 1.0) * 0.001,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "range_high": range_high,
                "breakout_extension": extension,
                "volume_ratio": volume_ratio,
                "efficiency_ratio": efficiency,
                "close_position": close_position,
                "setup_family": "breakout",
            },
        )
    return None


def active_pullback_reclaim_candidate(
    bars: list[Bar],
    *,
    max_holding_hours: int = 4,
) -> CandidateEvent | None:
    if len(bars) < 96:
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    previous = ordered[-2]
    sma_8 = _safe_mean([bar.close for bar in ordered[-8:]])
    sma_24 = _safe_mean([bar.close for bar in ordered[-24:]])
    sma_96 = _safe_mean([bar.close for bar in ordered[-96:]])
    recent_low = min(bar.low for bar in ordered[-8:])
    reclaim = previous.close <= sma_8 and latest.close > sma_8
    trend_ok = sma_24 >= sma_96 * 0.995 and latest.close >= sma_24 * 0.985
    pullback_depth = recent_low / sma_24 - 1 if sma_24 > 0 else 0.0
    volume_ratio = _volume_ratio(ordered, lookback=24)
    if reclaim and trend_ok and -0.025 <= pullback_depth <= 0.004 and volume_ratio >= 0.70:
        return CandidateEvent(
            event_id=_event_id("active_pullback_reclaim", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_pullback_reclaim",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=(latest.close / sma_8 - 1) + max(0.0, sma_24 / sma_96 - 1),
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "pullback_depth": pullback_depth,
                "sma_8_distance": latest.close / sma_8 - 1 if sma_8 > 0 else 0.0,
                "sma_24_to_96": sma_24 / sma_96 - 1 if sma_96 > 0 else 0.0,
                "volume_ratio": volume_ratio,
                "setup_family": "pullback_reclaim",
            },
        )
    return None


def active_squeeze_breakout_candidate(
    bars: list[Bar],
    *,
    lookback: int = 32,
    max_holding_hours: int = 6,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 120):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    ranges = [(bar.high - bar.low) / bar.close for bar in prior if bar.close > 0]
    if len(ranges) < 8:
        return None
    recent_compression = _safe_mean(ranges[-8:])
    median_range = sorted(ranges)[len(ranges) // 2]
    compression_high = max(bar.high for bar in prior[-8:])
    compression_low = min(bar.low for bar in prior[-8:])
    width = compression_high - compression_low
    extension = latest.close / compression_high - 1 if compression_high > 0 else 0.0
    volume_ratio = _volume_ratio(ordered, lookback=24)
    if recent_compression <= median_range * 0.85 and width > 0 and latest.close > compression_high and volume_ratio >= 0.85:
        return CandidateEvent(
            event_id=_event_id("active_squeeze_breakout", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_squeeze_breakout",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=extension + width / latest.close,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "compression_high": compression_high,
                "compression_width": width / latest.close,
                "recent_compression": recent_compression,
                "median_range": median_range,
                "volume_ratio": volume_ratio,
                "setup_family": "breakout",
            },
        )
    return None


def active_liquidity_reversal_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 3,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 2, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    prior_low = min(bar.low for bar in prior)
    latest_range = latest.high - latest.low
    lower_wick = min(latest.open, latest.close) - latest.low
    close_position = (latest.close - latest.low) / latest_range if latest_range > 0 else 0.5
    volume_ratio = _volume_ratio(ordered, lookback=24)
    if latest.low < prior_low * 0.999 and latest.close > prior_low and close_position >= 0.55 and volume_ratio >= 0.75:
        return CandidateEvent(
            event_id=_event_id("active_liquidity_reversal", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_liquidity_reversal",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=(latest.close / prior_low - 1) + lower_wick / latest.close,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "prior_low": prior_low,
                "lower_wick": lower_wick / latest.close if latest.close > 0 else 0.0,
                "close_position": close_position,
                "volume_ratio": volume_ratio,
                "setup_family": "liquidation_reversal",
            },
        )
    return None


def active_range_mean_reversion_candidate(
    bars: list[Bar],
    *,
    lookback: int = 48,
    max_holding_hours: int = 3,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 1, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    window = ordered[-lookback:]
    high = max(bar.high for bar in window)
    low = min(bar.low for bar in window)
    width = high - low
    if width <= 0:
        return None
    position = (latest.close - low) / width
    efficiency = _efficiency_ratio(ordered, lookback=min(lookback, 48))
    lower_wick = min(latest.open, latest.close) - latest.low
    if position <= 0.22 and efficiency <= 0.35 and lower_wick / latest.close >= 0.001:
        return CandidateEvent(
            event_id=_event_id("active_range_mean_reversion", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_range_mean_reversion",
            side=Side.BUY,
            bar_size=latest.bar_size,
            signal_strength=(0.5 - position) + lower_wick / latest.close,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "range_position": position,
                "range_width": width / latest.close,
                "efficiency_ratio": efficiency,
                "lower_wick": lower_wick / latest.close,
                "setup_family": "mean_reversion",
            },
        )
    return None


def active_short_breakdown_candidate(
    bars: list[Bar],
    *,
    lookback: int = 24,
    max_holding_hours: int = 4,
) -> CandidateEvent | None:
    if len(bars) < max(lookback + 12, 96):
        return None
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    prior = ordered[-lookback - 1 : -1]
    range_low = min(bar.low for bar in prior)
    extension = range_low / latest.close - 1 if latest.close > 0 else 0.0
    volume_ratio = _volume_ratio(ordered, lookback=min(lookback, 24))
    efficiency = _efficiency_ratio(ordered, lookback=min(lookback, 24))
    close_position = (latest.close - latest.low) / (latest.high - latest.low) if latest.high > latest.low else 0.5
    if 0.0005 <= extension <= 0.012 and close_position <= 0.38 and volume_ratio >= 0.80 and efficiency >= 0.12:
        return CandidateEvent(
            event_id=_event_id("active_short_breakdown", latest),
            timestamp_utc=latest.timestamp_utc,
            symbol=latest.symbol,
            candidate_type="active_short_breakdown",
            side=Side.SELL,
            bar_size=latest.bar_size,
            signal_strength=extension + max(0.0, volume_ratio - 1.0) * 0.001,
            reference_price=latest.close,
            max_holding_hours=max_holding_hours,
            metadata={
                "range_low": range_low,
                "breakdown_extension": extension,
                "volume_ratio": volume_ratio,
                "efficiency_ratio": efficiency,
                "close_position": close_position,
                "setup_family": "breakout",
            },
        )
    return None
