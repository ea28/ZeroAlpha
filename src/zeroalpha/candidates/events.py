"""Candidate event table generation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from zeroalpha.candidates.rules import (
    active_breakout_continuation_candidate,
    active_liquidity_reversal_candidate,
    active_pullback_reclaim_candidate,
    active_range_mean_reversion_candidate,
    active_short_breakdown_candidate,
    active_squeeze_breakout_candidate,
    capitulation_mean_reversion_candidate,
    information_momentum_candidate,
    intraday_volatility_breakout_candidate,
    liquidity_sweep_reclaim_candidate,
    short_information_momentum_candidate,
    short_trend_rejection_candidate,
    short_volatility_breakdown_candidate,
    trend_pullback_reclaim_candidate,
    trend_continuation_candidate,
    volatility_breakout_candidate,
    volatility_breakout_retest_candidate,
)
from zeroalpha.domain import Bar, CandidateEvent, Side


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    lookback: int = 24
    max_holding_hours: int = 72
    min_history_bars: int = 240
    rolling_window_bars: int = 500
    mode: str = "rules"
    dense_stride_bars: int = 1
    side_mode: str = "long"
    allow_short_research: bool = False


def _bar_returns(bars: list[Bar]) -> list[float]:
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    return [
        ordered[idx].close / ordered[idx - 1].close - 1
        for idx in range(1, len(ordered))
        if ordered[idx - 1].close > 0
    ]


def _safe_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dense_setup_metadata(bars: list[Bar]) -> dict[str, float | str | bool]:
    """Describe every dense bar as a tradable setup family for specialist models."""

    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    latest = ordered[-1]
    previous = ordered[:-1]
    recent_24 = ordered[-25:]
    previous_24 = recent_24[:-1] or previous[-24:]
    returns = _bar_returns(ordered)
    return_4 = sum(returns[-4:]) if len(returns) >= 4 else sum(returns)
    return_16 = sum(returns[-16:]) if len(returns) >= 16 else sum(returns)
    prior_return = returns[-1] if returns else 0.0
    previous_high = max((bar.high for bar in previous_24), default=latest.high)
    previous_low = min((bar.low for bar in previous_24), default=latest.low)
    range_width = previous_high - previous_low
    range_position = (latest.close - previous_low) / range_width if range_width > 0 else 0.5
    previous_volume = _safe_average([bar.volume for bar in previous_24])
    volume_ratio = latest.volume / previous_volume if previous_volume > 0 else 0.0
    current_range = (latest.high - latest.low) / latest.close if latest.close > 0 else 0.0
    previous_ranges = [
        (bar.high - bar.low) / bar.close
        for bar in previous_24
        if bar.close > 0
    ]
    average_range = _safe_average(previous_ranges)
    range_expansion = current_range / average_range if average_range > 0 else 0.0
    compression_window = previous_ranges[-8:] if len(previous_ranges) >= 8 else previous_ranges
    range_compression = _safe_average(compression_window) / average_range if average_range > 0 else 0.0
    lower_wick = min(latest.open, latest.close) - latest.low
    upper_wick = latest.high - max(latest.open, latest.close)
    candle_range = latest.high - latest.low
    lower_wick_share = lower_wick / candle_range if candle_range > 0 else 0.0
    upper_wick_share = upper_wick / candle_range if candle_range > 0 else 0.0

    if latest.close > previous_high and volume_ratio >= 1.05:
        family = "dense_breakout_momentum"
    elif return_4 > 0 and return_16 > 0 and range_position >= 0.55:
        family = "dense_trend_continuation"
    elif range_position <= 0.25 and lower_wick_share > upper_wick_share:
        family = "dense_support_reclaim"
    elif range_position >= 0.75 and prior_return < 0:
        family = "dense_pullback_reclaim"
    elif range_compression <= 0.75 and range_expansion >= 1.15:
        family = "dense_volatility_expansion"
    else:
        family = "dense_baseline"

    return {
        "dense_research": True,
        "setup_family": family,
        "dense_setup_family": family,
        "prior_bar_return": prior_return,
        "dense_return_4": return_4,
        "dense_return_16": return_16,
        "dense_range_position_24": range_position,
        "dense_volume_ratio_24": volume_ratio,
        "dense_range_expansion_24": range_expansion,
        "dense_range_compression_24": range_compression,
        "dense_lower_wick_share": lower_wick_share,
        "dense_upper_wick_share": upper_wick_share,
    }


def dense_research_candidate(
    bars: list[Bar],
    *,
    max_holding_hours: int,
    side: Side = Side.BUY,
) -> CandidateEvent | None:
    if len(bars) < 2:
        return None
    latest = sorted(bars, key=lambda bar: bar.timestamp_utc)[-1]
    previous = sorted(bars, key=lambda bar: bar.timestamp_utc)[-2]
    raw_signal = latest.close / previous.close - 1
    signal_strength = raw_signal if side == Side.BUY else -raw_signal
    metadata = {
        **_dense_setup_metadata(bars),
        "dense_side": side.value,
        "dense_raw_signal_strength": raw_signal,
        "dense_side_aligned_signal_strength": signal_strength,
    }
    event_id = str(
        uuid5(
            NAMESPACE_URL,
            f"dense_research_bar:{side.value}:{latest.source}:{latest.symbol}:{latest.bar_size}:{latest.timestamp_utc}",
        )
    )
    return CandidateEvent(
        event_id=event_id,
        timestamp_utc=latest.timestamp_utc,
        symbol=latest.symbol,
        candidate_type="dense_research_bar",
        side=side,
        bar_size=latest.bar_size,
        signal_strength=signal_strength,
        reference_price=latest.close,
        max_holding_hours=max_holding_hours,
        metadata=metadata,
    )


def candidates_for_history(
    bars: list[Bar],
    *,
    config: CandidateGenerationConfig,
) -> list[CandidateEvent]:
    if len(bars) < config.min_history_bars:
        return []
    if config.mode == "dense_research":
        dense_candidates: list[CandidateEvent] = []
        if config.side_mode in {"long", "long_short"}:
            candidate = dense_research_candidate(
                bars,
                max_holding_hours=config.max_holding_hours,
                side=Side.BUY,
            )
            if candidate is not None:
                dense_candidates.append(candidate)
        if config.side_mode in {"short", "long_short"}:
            candidate = dense_research_candidate(
                bars,
                max_holding_hours=config.max_holding_hours,
                side=Side.SELL,
            )
            if candidate is not None:
                dense_candidates.append(candidate)
        return dense_candidates
    candidates: list[CandidateEvent] = []
    if config.mode == "active_research":
        active_horizon = min(config.max_holding_hours, 12)
        if config.side_mode in {"long", "long_short"}:
            for candidate in (
                active_breakout_continuation_candidate(
                    bars,
                    lookback=max(8, min(config.lookback, 32)),
                    max_holding_hours=min(active_horizon, 4),
                ),
                active_pullback_reclaim_candidate(
                    bars,
                    max_holding_hours=min(active_horizon, 4),
                ),
                active_squeeze_breakout_candidate(
                    bars,
                    lookback=max(16, min(config.lookback * 2, 48)),
                    max_holding_hours=min(active_horizon, 6),
                ),
                active_liquidity_reversal_candidate(
                    bars,
                    lookback=max(12, min(config.lookback, 32)),
                    max_holding_hours=min(active_horizon, 3),
                ),
                active_range_mean_reversion_candidate(
                    bars,
                    lookback=max(24, min(config.lookback * 2, 64)),
                    max_holding_hours=min(active_horizon, 3),
                ),
            ):
                if candidate is not None:
                    candidates.append(candidate)
        if config.side_mode in {"short", "long_short"}:
            candidate = active_short_breakdown_candidate(
                bars,
                lookback=max(8, min(config.lookback, 32)),
                max_holding_hours=min(active_horizon, 4),
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates
    if config.side_mode in {"long", "long_short"}:
        long_candidates = [
            information_momentum_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=min(24, config.max_holding_hours),
            ),
            trend_pullback_reclaim_candidate(
                bars,
                max_holding_hours=config.max_holding_hours,
            ),
            trend_continuation_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=config.max_holding_hours,
            ),
            volatility_breakout_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=config.max_holding_hours,
            ),
            volatility_breakout_retest_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=min(48, config.max_holding_hours),
            ),
            capitulation_mean_reversion_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=min(48, config.max_holding_hours),
            ),
        ]
        if config.mode == "aggressive_rules":
            long_candidates.extend(
                [
                    intraday_volatility_breakout_candidate(
                        bars,
                        lookback=max(8, min(config.lookback, 16)),
                        max_holding_hours=min(12, config.max_holding_hours),
                    ),
                    liquidity_sweep_reclaim_candidate(
                        bars,
                        lookback=config.lookback,
                        max_holding_hours=min(12, config.max_holding_hours),
                    ),
                ]
            )
        for candidate in long_candidates:
            if candidate is not None:
                candidates.append(candidate)
    if config.side_mode in {"short", "long_short"}:
        for candidate in (
            short_information_momentum_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=min(24, config.max_holding_hours),
            ),
            short_trend_rejection_candidate(
                bars,
                max_holding_hours=min(48, config.max_holding_hours),
            ),
            short_volatility_breakdown_candidate(
                bars,
                lookback=config.lookback,
                max_holding_hours=min(48, config.max_holding_hours),
            ),
        ):
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def candidate_for_history(
    bars: list[Bar],
    *,
    config: CandidateGenerationConfig,
) -> CandidateEvent | None:
    candidates = candidates_for_history(bars, config=config)
    return candidates[0] if candidates else None


def candidate_for_index(
    bars: list[Bar],
    idx: int,
    *,
    config: CandidateGenerationConfig,
) -> CandidateEvent | None:
    if idx < 0 or idx >= len(bars):
        raise IndexError("candidate index out of range")
    if idx + 1 < config.min_history_bars:
        return None
    window_size = max(config.rolling_window_bars, config.min_history_bars, config.lookback + 240)
    start = max(0, idx + 1 - window_size)
    return candidate_for_history(bars[start : idx + 1], config=config)


def candidates_for_index(
    bars: list[Bar],
    idx: int,
    *,
    config: CandidateGenerationConfig,
) -> list[CandidateEvent]:
    if idx < 0 or idx >= len(bars):
        raise IndexError("candidate index out of range")
    if idx + 1 < config.min_history_bars:
        return []
    window_size = max(config.rolling_window_bars, config.min_history_bars, config.lookback + 240)
    start = max(0, idx + 1 - window_size)
    return candidates_for_history(bars[start : idx + 1], config=config)


def generate_candidate_events(
    bars: list[Bar],
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[CandidateEvent]:
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    cfg = config or CandidateGenerationConfig()
    if cfg.mode not in {"rules", "aggressive_rules", "dense_research", "active_research"}:
        raise ValueError(
            "candidate generation mode must be rules, aggressive_rules, dense_research, or active_research"
        )
    if cfg.side_mode not in {"long", "short", "long_short"}:
        raise ValueError("candidate side_mode must be long, short, or long_short")
    if cfg.side_mode in {"short", "long_short"} and not cfg.allow_short_research:
        raise ValueError(
            "short candidate research requires a futures instrument model or explicit "
            "allow_short_research; spot crypto execution remains long/flat"
        )
    if cfg.dense_stride_bars <= 0:
        raise ValueError("dense_stride_bars must be positive")
    events: list[CandidateEvent] = []
    first_idx = max(cfg.min_history_bars - 1, 0)
    seen_event_ids: set[str] = set()
    for idx in range(first_idx, len(ordered)):
        if (idx - first_idx) % cfg.dense_stride_bars:
            continue
        for event in candidates_for_index(ordered, idx, config=cfg):
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            events.append(event)
    return events
