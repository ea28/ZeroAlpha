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


def dense_research_candidate(
    bars: list[Bar],
    *,
    max_holding_hours: int,
) -> CandidateEvent | None:
    if len(bars) < 2:
        return None
    latest = sorted(bars, key=lambda bar: bar.timestamp_utc)[-1]
    previous = sorted(bars, key=lambda bar: bar.timestamp_utc)[-2]
    signal_strength = latest.close / previous.close - 1
    event_id = str(
        uuid5(
            NAMESPACE_URL,
            f"dense_research_bar:{latest.source}:{latest.symbol}:{latest.bar_size}:{latest.timestamp_utc}",
        )
    )
    return CandidateEvent(
        event_id=event_id,
        timestamp_utc=latest.timestamp_utc,
        symbol=latest.symbol,
        candidate_type="dense_research_bar",
        side=Side.BUY,
        bar_size=latest.bar_size,
        signal_strength=signal_strength,
        reference_price=latest.close,
        max_holding_hours=max_holding_hours,
        metadata={"dense_research": True, "prior_bar_return": signal_strength},
    )


def candidates_for_history(
    bars: list[Bar],
    *,
    config: CandidateGenerationConfig,
) -> list[CandidateEvent]:
    if len(bars) < config.min_history_bars:
        return []
    if config.mode == "dense_research":
        candidate = dense_research_candidate(bars, max_holding_hours=config.max_holding_hours)
        return [candidate] if candidate is not None else []
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
