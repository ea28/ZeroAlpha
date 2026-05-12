"""Model-driven strategy backtest using walk-forward meta-label predictions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import floor, isfinite, log, sqrt
from pathlib import Path
from statistics import mean, pstdev
import json

from zeroalpha.bars import bar_start_timestamp_utc
from zeroalpha.backtest.fills import SimulatedFill, simulate_limit_fill
from zeroalpha.backtest.simple import (
    _estimate_trade_notional,
    _median,
    _period_pnl,
    _trade_cost_components,
)
from zeroalpha.config import AppConfig
from zeroalpha.costs import CommissionModel, SlippageModel, estimate_round_trip_cost
from zeroalpha.domain import Bar, Side
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.ensemble import (
    FoldPrediction,
    MetaLabelWalkForwardReport,
    _passes_selection_and_ev_gate,
    report_feature_importance_summary,
    report_model_family_summary,
    report_native_importance_summary,
    report_shap_importance_summary,
)
from zeroalpha.timeutils import ensure_utc


@dataclass(frozen=True, slots=True)
class MLBacktestTrade:
    fold_id: int
    event_id: str
    timestamp_utc: datetime
    candidate_type: str
    probability: float
    expected_value: float
    predicted_return: float
    predicted_downside: float
    selection_score: float
    notional: float
    requested_notional: float
    notional_scale: float
    sizing_mode: str
    gross_return: float
    net_return: float
    gross_pnl: float
    pnl: float
    equity_after: float
    entry_fill_price: float
    entry_fill_timestamp_utc: datetime
    entry_fill_fraction: float
    exit_timestamp_utc: datetime
    outcome_type: str
    round_trip_cost_bps: float
    commission_estimate: float
    spread_cost_estimate: float
    slippage_cost_estimate: float
    safety_margin_estimate: float
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    side: str = ""
    exit_overlay_reason: str = ""


@dataclass(frozen=True, slots=True)
class MLBacktestRejection:
    timestamp_utc: datetime
    event_id: str
    candidate_type: str
    reason: str
    equity: float
    probability: float
    expected_value: float
    proposed_notional: float = 0.0


@dataclass(frozen=True, slots=True)
class MLBacktestSummary:
    start_equity: float
    end_equity: float
    total_return: float
    max_drawdown: float
    candidate_predictions: int
    model_approved_signals: int
    trades: int
    configured_span_days: float
    prediction_span_days: float
    active_trade_span_days: float
    max_simultaneous_open_positions: int
    max_open_notional: float
    max_trade_notional: float
    trades_per_configured_day: float
    trades_per_prediction_day: float
    trades_per_active_day: float
    pnl_per_configured_day: float
    pnl_per_prediction_day: float
    hit_rate: float
    average_net_return: float
    median_net_return: float
    gross_pnl: float
    net_pnl: float
    sharpe: float
    daily_sharpe: float
    trade_level_sharpe: float
    deflated_sharpe: float
    multiple_testing_trials: int
    multiple_testing_haircut: float
    profit_factor: float
    max_intratrade_drawdown: float
    average_max_adverse_excursion: float
    total_commission_estimate: float
    total_spread_cost_estimate: float
    total_slippage_cost_estimate: float
    total_safety_margin_estimate: float
    missed_fills: int
    rejected_signals: int
    reject_reasons: dict[str, int]
    by_candidate_type: dict[str, dict[str, float]]
    by_side: dict[str, dict[str, float]]
    calibration_metrics: dict[str, float | None]
    data_coverage: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReplayedExit:
    timestamp_utc: datetime
    price: float
    outcome_type: str
    gross_return: float
    net_return: float
    overlay_reason: str = ""


@dataclass(frozen=True, slots=True)
class PendingMLBacktestTrade:
    fold_id: int
    event_id: str
    timestamp_utc: datetime
    candidate_type: str
    probability: float
    expected_value: float
    predicted_return: float
    predicted_downside: float
    selection_score: float
    notional: float
    requested_notional: float
    notional_scale: float
    sizing_mode: str
    gross_return: float
    net_return: float
    gross_pnl: float
    pnl: float
    entry_fill_price: float
    entry_fill_timestamp_utc: datetime
    entry_fill_fraction: float
    exit_timestamp_utc: datetime
    outcome_type: str
    round_trip_cost_bps: float
    commission_estimate: float
    spread_cost_estimate: float
    slippage_cost_estimate: float
    safety_margin_estimate: float
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    side: str = ""
    exit_overlay_reason: str = ""


def _prediction_timestamp(prediction: FoldPrediction) -> datetime:
    return ensure_utc(datetime.fromisoformat(prediction.timestamp_utc.replace("Z", "+00:00")))


def _bars_for_window(
    bars: list[Bar],
    timestamps: list[datetime],
    *,
    start: datetime,
    end: datetime,
) -> list[Bar]:
    left = bisect_left(timestamps, ensure_utc(start))
    right = bisect_right(timestamps, ensure_utc(end))
    return bars[left:right]


def _record_rejection(
    rejections: list[MLBacktestRejection],
    prediction: FoldPrediction,
    *,
    reason: str,
    equity: float,
    proposed_notional: float = 0.0,
) -> None:
    rejections.append(
        MLBacktestRejection(
            timestamp_utc=_prediction_timestamp(prediction),
            event_id=prediction.event_id,
            candidate_type=prediction.candidate_type,
            reason=reason,
            equity=equity,
            probability=prediction.probability,
            expected_value=prediction.expected_value,
            proposed_notional=proposed_notional,
        )
    )


def _sample_min_exit_timestamp(sample: MetaLabelSample, fill_timestamp: datetime) -> datetime:
    value = sample.features.get(
        "event_min_holding_seconds",
        sample.features.get("min_holding_seconds", 0.0),
    )
    try:
        min_holding_seconds = float(value)
    except (TypeError, ValueError):
        min_holding_seconds = 0.0
    if not isfinite(min_holding_seconds) or min_holding_seconds <= 0:
        return fill_timestamp
    return fill_timestamp + timedelta(seconds=min_holding_seconds)


def _execution_entry_price(entry_bars: list[Bar], entry_timestamp: datetime, fallback: float) -> float:
    for bar in sorted(entry_bars, key=lambda row: row.timestamp_utc):
        if bar_start_timestamp_utc(bar) >= entry_timestamp and bar.open > 0:
            return bar.open
    return fallback


def _simulate_market_entry_fill(
    *,
    entry_bars: list[Bar],
    entry_timestamp: datetime,
    fallback_price: float,
) -> SimulatedFill:
    for bar in sorted(entry_bars, key=lambda row: row.timestamp_utc):
        bar_start = bar_start_timestamp_utc(bar)
        if bar_start >= entry_timestamp and bar.open > 0:
            return SimulatedFill(
                True,
                bar.open,
                "market_entry_open",
                timestamp_utc=bar_start,
            )
    if fallback_price > 0:
        return SimulatedFill(
            True,
            fallback_price,
            "market_entry_fallback",
            timestamp_utc=entry_timestamp,
        )
    return SimulatedFill(False, None, "no_entry_bars")


def _replay_exit_from_fill(
    *,
    sample: MetaLabelSample,
    side: Side,
    fill_price: float,
    fill_timestamp: datetime,
    exit_bars: list[Bar],
    round_trip_cost_bps: float,
    conservative_same_bar: bool,
) -> ReplayedExit | None:
    ordered = [
        bar
        for bar in sorted(exit_bars, key=lambda row: row.timestamp_utc)
        if fill_timestamp <= bar.timestamp_utc <= sample.label_detail.vertical_barrier_timestamp_utc
    ]
    if not ordered:
        return None
    cost_fraction = round_trip_cost_bps / 10_000
    exit_bar = ordered[-1]
    outcome = "vertical_replay"
    exit_price = exit_bar.close
    min_exit_timestamp = _sample_min_exit_timestamp(sample, fill_timestamp)

    if side == Side.SELL:
        lower = fill_price * (1 - sample.net_profit_target - cost_fraction)
        upper = fill_price * (1 + sample.net_stop_loss - cost_fraction)
        for bar in ordered:
            if bar_start_timestamp_utc(bar) <= fill_timestamp:
                continue
            if bar_start_timestamp_utc(bar) < min_exit_timestamp:
                continue
            hit_profit = bar.low <= lower
            hit_stop = bar.high >= upper
            if hit_profit and hit_stop:
                exit_bar = bar
                if conservative_same_bar:
                    outcome = "upper_same_bar_replay"
                    exit_price = upper
                else:
                    outcome = "lower_same_bar_replay"
                    exit_price = lower
                break
            if hit_stop:
                exit_bar = bar
                outcome = "upper_replay"
                exit_price = upper
                break
            if hit_profit:
                exit_bar = bar
                outcome = "lower_replay"
                exit_price = lower
                break
        gross_return = (fill_price - exit_price) / fill_price
    else:
        upper = fill_price * (1 + sample.net_profit_target + cost_fraction)
        lower = fill_price * (1 + cost_fraction - sample.net_stop_loss)
        for bar in ordered:
            if bar_start_timestamp_utc(bar) <= fill_timestamp:
                continue
            if bar_start_timestamp_utc(bar) < min_exit_timestamp:
                continue
            hit_profit = bar.high >= upper
            hit_stop = bar.low <= lower
            if hit_profit and hit_stop:
                exit_bar = bar
                if conservative_same_bar:
                    outcome = "lower_same_bar_replay"
                    exit_price = lower
                else:
                    outcome = "upper_same_bar_replay"
                    exit_price = upper
                break
            if hit_stop:
                exit_bar = bar
                outcome = "lower_replay"
                exit_price = lower
                break
            if hit_profit:
                exit_bar = bar
                outcome = "upper_replay"
                exit_price = upper
                break
        gross_return = exit_price / fill_price - 1
    return ReplayedExit(
        timestamp_utc=exit_bar.timestamp_utc,
        price=exit_price,
        outcome_type=outcome,
        gross_return=gross_return,
        net_return=gross_return - cost_fraction,
    )


def _side_gross_return(side: Side, *, entry_price: float, exit_price: float) -> float:
    if side == Side.SELL:
        return (entry_price - exit_price) / entry_price
    return exit_price / entry_price - 1


def _bar_side_extreme_return(side: Side, *, entry_price: float, bar: Bar) -> tuple[float, float]:
    if side == Side.SELL:
        favorable = (entry_price - bar.low) / entry_price
        adverse = (entry_price - bar.high) / entry_price
    else:
        favorable = bar.high / entry_price - 1
        adverse = bar.low / entry_price - 1
    return favorable, adverse


def _trade_excursions(
    *,
    side: Side,
    entry_price: float,
    exit_bars: list[Bar],
    exit_timestamp: datetime,
) -> tuple[float, float]:
    if entry_price <= 0:
        return 0.0, 0.0
    adverse_excursion = 0.0
    favorable_excursion = 0.0
    for bar in sorted(exit_bars, key=lambda row: row.timestamp_utc):
        if bar.timestamp_utc > exit_timestamp:
            break
        favorable, adverse = _bar_side_extreme_return(
            side,
            entry_price=entry_price,
            bar=bar,
        )
        favorable_excursion = max(favorable_excursion, favorable)
        adverse_excursion = max(adverse_excursion, -adverse)
    return adverse_excursion, favorable_excursion


def _checkpoint_bar(
    ordered: list[Bar],
    *,
    fill_timestamp: datetime,
    replayed_exit_timestamp: datetime,
    checkpoint_seconds: int,
) -> tuple[int, Bar] | None:
    target = fill_timestamp + timedelta(seconds=checkpoint_seconds)
    if target >= replayed_exit_timestamp:
        return None
    for idx, bar in enumerate(ordered):
        if bar.timestamp_utc >= target:
            if bar.timestamp_utc >= replayed_exit_timestamp:
                return None
            return idx, bar
    return None


def _checkpoint_label(checkpoint_seconds: int) -> str:
    if checkpoint_seconds >= 60 and checkpoint_seconds % 60 == 0:
        return f"{checkpoint_seconds // 60}m"
    return f"{checkpoint_seconds}s"


def _checkpoint_seconds(
    *,
    checkpoints_minutes: tuple[int, ...],
    checkpoints_seconds: tuple[int, ...],
) -> tuple[int, ...]:
    seconds = {value * 60 for value in checkpoints_minutes if value > 0}
    seconds.update(value for value in checkpoints_seconds if value > 0)
    return tuple(sorted(seconds))


def _apply_dynamic_exit_overlay(
    *,
    sample: MetaLabelSample,
    prediction: FoldPrediction,
    side: Side,
    fill_price: float,
    fill_timestamp: datetime,
    exit_bars: list[Bar],
    replayed_exit: ReplayedExit,
    round_trip_cost_bps: float,
    checkpoints_minutes: tuple[int, ...],
    checkpoints_seconds: tuple[int, ...],
    adverse_bps: float,
    giveback_bps: float,
    min_profit_bps: float,
    weak_probability: float,
    weak_expected_value_bps: float,
) -> ReplayedExit:
    checkpoint_values = _checkpoint_seconds(
        checkpoints_minutes=checkpoints_minutes,
        checkpoints_seconds=checkpoints_seconds,
    )
    if not checkpoint_values:
        return replayed_exit
    min_exit_timestamp = _sample_min_exit_timestamp(sample, fill_timestamp)
    ordered = [
        bar
        for bar in sorted(exit_bars, key=lambda row: row.timestamp_utc)
        if min_exit_timestamp <= bar.timestamp_utc <= replayed_exit.timestamp_utc
    ]
    if not ordered:
        return replayed_exit

    max_favorable_bps = 0.0
    weak_signal = (
        prediction.probability <= weak_probability
        or prediction.expected_value * 10_000 <= weak_expected_value_bps
        or prediction.predicted_return < 0
    )
    for checkpoint in checkpoint_values:
        label = _checkpoint_label(checkpoint)
        row = _checkpoint_bar(
            ordered,
            fill_timestamp=fill_timestamp,
            replayed_exit_timestamp=replayed_exit.timestamp_utc,
            checkpoint_seconds=checkpoint,
        )
        if row is None:
            continue
        idx, bar = row
        for prior in ordered[: idx + 1]:
            favorable, _ = _bar_side_extreme_return(side, entry_price=fill_price, bar=prior)
            max_favorable_bps = max(max_favorable_bps, favorable * 10_000)
        current_gross_return = _side_gross_return(side, entry_price=fill_price, exit_price=bar.close)
        current_bps = current_gross_return * 10_000
        giveback = max_favorable_bps - current_bps
        bar_return_bps = _side_gross_return(side, entry_price=bar.open, exit_price=bar.close) * 10_000
        bar_range_bps = 10_000 * max(bar.high - bar.low, 0.0) / bar.close if bar.close > 0 else 0.0
        reason = ""
        if current_bps <= -adverse_bps and weak_signal:
            reason = f"adverse_{label}"
        elif bar_return_bps <= -max(adverse_bps * 0.50, 5.0) and weak_signal:
            reason = f"micro_reversal_{label}"
        elif bar_range_bps >= max(adverse_bps * 2.0, giveback_bps) and weak_signal:
            reason = f"volatility_shock_{label}"
        elif giveback >= giveback_bps and current_bps >= min_profit_bps:
            reason = f"giveback_{label}"
        elif checkpoint >= 120 * 60 and weak_signal and current_bps <= 0:
            reason = f"weak_no_progress_{label}"
        if not reason:
            continue
        net_return = current_gross_return - round_trip_cost_bps / 10_000
        return ReplayedExit(
            timestamp_utc=bar.timestamp_utc,
            price=bar.close,
            outcome_type=f"dynamic_exit_{reason}",
            gross_return=current_gross_return,
            net_return=net_return,
            overlay_reason=reason,
        )
    return replayed_exit


def _annualized_daily_sharpe(
    *,
    report: MetaLabelWalkForwardReport,
    trades: list[MLBacktestTrade],
    start_equity: float,
) -> float:
    prediction_dates = [_prediction_timestamp(prediction).date() for prediction in report.predictions]
    trade_dates = [trade.exit_timestamp_utc.date() for trade in trades]
    all_dates = prediction_dates + trade_dates
    if not all_dates:
        return 0.0

    pnl_by_date: dict[object, float] = {}
    for trade in trades:
        key = trade.exit_timestamp_utc.date()
        pnl_by_date[key] = pnl_by_date.get(key, 0.0) + trade.pnl

    current = min(all_dates)
    end = max(all_dates)
    equity = start_equity
    returns: list[float] = []
    while current <= end:
        pnl = pnl_by_date.get(current, 0.0)
        returns.append(pnl / equity if equity > 0 else 0.0)
        equity += pnl
        current += timedelta(days=1)

    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    return mean(returns) / volatility * sqrt(365)


def _annualized_trade_level_sharpe(
    *,
    trades: list[MLBacktestTrade],
    span_days: float,
) -> float:
    returns = [trade.net_return for trade in trades]
    if len(returns) < 2 or span_days <= 0:
        return 0.0
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    trades_per_year = len(returns) / span_days * 365
    return mean(returns) / volatility * sqrt(max(trades_per_year, 0.0))


def _multiple_testing_trials(data_coverage: dict[str, object]) -> int:
    value = data_coverage.get("multiple_testing_trials")
    if not isinstance(value, int | float):
        diagnostics = data_coverage.get("multiple_testing")
        if isinstance(diagnostics, dict):
            value = diagnostics.get("trials")
    if not isinstance(value, int | float) or not isfinite(float(value)):
        return 1
    return max(1, int(value))


def _deflated_sharpe_approximation(sharpe: float, trials: int) -> tuple[float, float]:
    if trials <= 1:
        return sharpe, 0.0
    haircut = sqrt(2.0 * log(float(trials)))
    return sharpe - haircut, haircut


def _candidate_type_notional_scale(type_threshold: dict[str, object] | None) -> float | None:
    if not type_threshold or type_threshold.get("source") != "candidate_type_calibration":
        return None
    average_return = float(type_threshold.get("average_trade_return", 0.0))
    utility_floor = float(type_threshold.get("utility_floor", 0.0))
    hit_rate = float(type_threshold.get("hit_rate", 0.0))
    traded_signals = int(type_threshold.get("traded_signals", 0))
    if utility_floor <= 0 or traded_signals <= 0:
        return None
    if average_return >= utility_floor and hit_rate >= 0.50:
        return 1.0
    if average_return > 0:
        return max(0.25, min(1.0, average_return / utility_floor))
    return None


def _confidence_notional_scale(
    prediction: FoldPrediction,
    config: AppConfig,
    *,
    sample: MetaLabelSample | None = None,
    type_threshold: dict[str, object] | None = None,
) -> float:
    type_scale = _candidate_type_notional_scale(type_threshold)
    if type_scale is not None:
        return type_scale
    if prediction.expected_value <= 0:
        return 0.25
    probability_threshold = max(config.model.minimum_probability, 1e-9)
    probability_room = max(1.0 - probability_threshold, 1e-9)
    probability_scale = max(0.0, (prediction.probability - probability_threshold) / probability_room)
    label_target = sample.net_profit_target if sample is not None else config.labels.net_profit_target
    label_stop = sample.net_stop_loss if sample is not None else config.labels.net_stop_loss
    geometry_floor = max(min(label_target, label_stop) * 0.25, 1e-6)
    ev_floor = max(config.model.minimum_expected_value, geometry_floor)
    ev_scale = max(0.0, prediction.expected_value / ev_floor)
    confidence = min(1.0, probability_scale, ev_scale)
    return max(0.25, min(1.0, 0.25 + 0.75 * confidence))


def _prediction_score_value(prediction: FoldPrediction, field: str) -> float:
    if field == "probability":
        return prediction.probability
    if field == "expected_value":
        return prediction.expected_value
    if field == "predicted_return":
        return prediction.predicted_return
    if field == "selection_score":
        return prediction.selection_score
    if field == "trade_score":
        return prediction.trade_score
    raise ValueError(
        "sizing_score_field must be probability, expected_value, predicted_return, "
        "selection_score, or trade_score"
    )


def _sample_spread_bps(sample: MetaLabelSample) -> float | None:
    candidates = [
        float(value)
        for key, value in sample.features.items()
        if key.endswith("spread_bps") and isinstance(value, int | float) and float(value) > 0
    ]
    return min(candidates) if candidates else None


def _sample_liquidity_score(sample: MetaLabelSample) -> float:
    values: list[float] = []
    for key in (
        "pm_leading_liquidity_weight_total",
        "pm_leading_liquidity_weight_mean",
        "ibkr_top_of_book_size_log",
        "ibkr_futures_top_of_book_size_log",
        "dollar_volume_log",
    ):
        value = sample.features.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return max(values) if values else 0.0


def _bucket_sized_notional(
    *,
    prediction: FoldPrediction,
    sample: MetaLabelSample,
    cap_notional: float,
    sizing_mode: str,
    score_field: str,
    score_direction: str,
    base_notional: float,
    mid_notional: float,
    high_notional: float,
    mid_score: float,
    high_score: float,
    max_spread_bps: float,
    min_liquidity_score: float,
) -> tuple[float, float, str]:
    if cap_notional <= 0:
        return 0.0, 0.0, sizing_mode
    base = base_notional if base_notional > 0 else min(5_000.0, cap_notional)
    mid = mid_notional if mid_notional > 0 else min(max(base, (base + cap_notional) / 2), cap_notional)
    high = high_notional if high_notional > 0 else cap_notional
    base = min(base, cap_notional)
    mid = min(max(mid, base), cap_notional)
    high = min(max(high, mid), cap_notional)

    if score_direction not in {"high", "low"}:
        raise ValueError("sizing_score_direction must be high or low")
    raw_score = _prediction_score_value(prediction, score_field)
    score = raw_score if score_direction == "high" else -raw_score
    spread = _sample_spread_bps(sample)
    spread_ok = spread is None or max_spread_bps <= 0 or spread <= max_spread_bps
    liquidity = _sample_liquidity_score(sample)
    liquidity_ok = min_liquidity_score <= 0 or liquidity >= min_liquidity_score
    high_quality = score >= high_score and spread_ok and liquidity_ok
    mid_quality = score >= mid_score
    if high_quality:
        selected = high
    elif mid_quality:
        selected = mid
    else:
        selected = base
    scale = selected / cap_notional if cap_notional > 0 else 0.0
    return selected, scale, sizing_mode


def _apply_notional_sizing(
    *,
    prediction: FoldPrediction,
    sample: MetaLabelSample,
    config: AppConfig,
    cap_notional: float,
    sizing_mode: str,
    score_field: str,
    score_direction: str,
    base_notional: float,
    mid_notional: float,
    high_notional: float,
    mid_score: float,
    high_score: float,
    max_spread_bps: float,
    min_liquidity_score: float,
    type_threshold: dict[str, object] | None,
) -> tuple[float, float, str]:
    if sizing_mode == "fixed":
        return cap_notional, 1.0, "fixed"
    if sizing_mode == "confidence":
        scale = _confidence_notional_scale(
            prediction,
            config,
            sample=sample,
            type_threshold=type_threshold,
        )
        return cap_notional * scale, scale, "confidence"
    if sizing_mode in {"score_bucket", "liquidity_score_bucket"}:
        return _bucket_sized_notional(
            prediction=prediction,
            sample=sample,
            cap_notional=cap_notional,
            sizing_mode=sizing_mode,
            score_field=score_field,
            score_direction=score_direction,
            base_notional=base_notional,
            mid_notional=mid_notional,
            high_notional=high_notional,
            mid_score=mid_score,
            high_score=high_score,
            max_spread_bps=max_spread_bps if sizing_mode == "liquidity_score_bucket" else 0.0,
            min_liquidity_score=min_liquidity_score if sizing_mode == "liquidity_score_bucket" else 0.0,
        )
    raise ValueError("sizing_mode must be fixed, confidence, score_bucket, or liquidity_score_bucket")


def _requested_notional_for_sizing_cap(
    *,
    requested_notional: float,
    sizing_mode: str,
    base_notional: float,
    mid_notional: float,
    high_notional: float,
) -> float:
    if sizing_mode not in {"score_bucket", "liquidity_score_bucket"}:
        return requested_notional
    bucket_cap = max(base_notional, mid_notional, high_notional, 0.0)
    return max(requested_notional, bucket_cap)


def _research_selection_allows_negative_ev(prediction: FoldPrediction) -> bool:
    if prediction.decision_reason == "candidate_type_calibration":
        return True
    return False


def _open_spot_notional(pending_trades: list[PendingMLBacktestTrade]) -> float:
    return sum(trade.notional for trade in pending_trades if trade.side == Side.BUY.value)


def _open_notional(pending_trades: list[PendingMLBacktestTrade]) -> float:
    return sum(trade.notional for trade in pending_trades)


def _futures_contract_cost_enabled(config: AppConfig) -> bool:
    return (
        config.contract.instrument_model == "futures"
        and config.cost.futures_fee_per_contract > 0
        and config.cost.futures_contract_multiplier > 0
    )


def _round_futures_notional(
    *,
    requested_notional: float,
    fill_price: float,
    fill_fraction: float,
    contract_multiplier: float,
) -> float:
    contract_notional = fill_price * contract_multiplier
    if contract_notional <= 0:
        raise ValueError("futures contract notional must be positive")
    contracts = max(1, floor(requested_notional / contract_notional))
    return contracts * contract_notional * fill_fraction


def _round_trip_commission_from_cost(notional: float, cost) -> float:
    return notional * cost.commission_bps / 10_000


def _coverage_span_days(coverage: object) -> float:
    if not isinstance(coverage, dict):
        return 0.0
    span = coverage.get("span_days")
    if isinstance(span, (int, float)) and isfinite(float(span)) and float(span) > 0:
        return float(span)
    quality_start = coverage.get("start")
    quality_end = coverage.get("end")
    if not isinstance(quality_start, str) or not isinstance(quality_end, str):
        return 0.0
    try:
        start = datetime.fromisoformat(quality_start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(quality_end.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max((end - start).total_seconds() / 86_400, 0.0)


def _summarize(
    *,
    report: MetaLabelWalkForwardReport,
    trades: list[MLBacktestTrade],
    rejections: list[MLBacktestRejection],
    start_equity: float,
    max_simultaneous_open_positions: int,
    max_open_notional: float,
) -> MLBacktestSummary:
    equity_curve = [start_equity]
    equity_curve.extend(trade.equity_after for trade in trades)
    peak = start_equity
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    returns = [trade.net_return for trade in trades]
    wins = [trade.pnl for trade in trades if trade.pnl > 0]
    losses = [-trade.pnl for trade in trades if trade.pnl < 0]
    adverse_excursions = [trade.max_adverse_excursion for trade in trades]
    intratrade_drawdowns = [
        (trade.max_adverse_excursion * trade.notional) / start_equity
        for trade in trades
        if start_equity > 0
    ]
    reject_reasons: dict[str, int] = {}
    for rejection in rejections:
        reject_reasons[rejection.reason] = reject_reasons.get(rejection.reason, 0) + 1

    by_type: dict[str, dict[str, float]] = {}
    by_side: dict[str, dict[str, float]] = {}
    for trade in trades:
        bucket = by_type.setdefault(
            trade.candidate_type,
            {"trades": 0.0, "wins": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0, "average_net_return": 0.0},
        )
        bucket["trades"] += 1
        bucket["wins"] += 1 if trade.pnl > 0 else 0
        bucket["gross_pnl"] += trade.gross_pnl
        bucket["net_pnl"] += trade.pnl
        bucket["average_net_return"] += trade.net_return
        side_bucket = by_side.setdefault(
            trade.side or "UNKNOWN",
            {"trades": 0.0, "wins": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0, "average_net_return": 0.0},
        )
        side_bucket["trades"] += 1
        side_bucket["wins"] += 1 if trade.pnl > 0 else 0
        side_bucket["gross_pnl"] += trade.gross_pnl
        side_bucket["net_pnl"] += trade.pnl
        side_bucket["average_net_return"] += trade.net_return
    for bucket in by_type.values():
        if bucket["trades"]:
            bucket["hit_rate"] = bucket["wins"] / bucket["trades"]
            bucket["average_net_return"] /= bucket["trades"]
    for bucket in by_side.values():
        if bucket["trades"]:
            bucket["hit_rate"] = bucket["wins"] / bucket["trades"]
            bucket["average_net_return"] /= bucket["trades"]

    end_equity = equity_curve[-1]
    fold_briers = [fold.brier_score for fold in report.folds]
    fold_log_losses = [fold.log_loss for fold in report.folds if fold.log_loss is not None]
    prediction_times = [_prediction_timestamp(prediction) for prediction in report.predictions]
    configured_span_days = _coverage_span_days(report.data_coverage.get("requested_window"))
    if configured_span_days <= 0:
        configured_span_days = _coverage_span_days(report.data_coverage.get("primary"))
    if configured_span_days <= 0 and prediction_times:
        configured_span_days = max(
            (max(prediction_times) - min(prediction_times)).total_seconds() / 86_400,
            0.0,
        )
    if len(prediction_times) >= 2:
        prediction_span_days = max(
            (max(prediction_times) - min(prediction_times)).total_seconds() / 86_400,
            1 / 24,
        )
    else:
        prediction_span_days = 0.0
    if trades:
        active_start = min(trade.timestamp_utc for trade in trades)
        active_end = max(trade.exit_timestamp_utc for trade in trades)
        active_trade_span_days = max((active_end - active_start).total_seconds() / 86_400, 1 / 24)
    else:
        active_trade_span_days = 0.0
    net_pnl = sum(trade.pnl for trade in trades)
    daily_sharpe = _annualized_daily_sharpe(report=report, trades=trades, start_equity=start_equity)
    trade_level_sharpe = _annualized_trade_level_sharpe(
        trades=trades,
        span_days=prediction_span_days,
    )
    multiple_testing_trials = _multiple_testing_trials(report.data_coverage)
    deflated_sharpe, multiple_testing_haircut = _deflated_sharpe_approximation(
        daily_sharpe,
        multiple_testing_trials,
    )
    return MLBacktestSummary(
        start_equity=start_equity,
        end_equity=end_equity,
        total_return=(end_equity / start_equity - 1) if start_equity else 0.0,
        max_drawdown=max_drawdown,
        candidate_predictions=len(report.predictions),
        model_approved_signals=sum(1 for prediction in report.predictions if prediction.should_trade),
        trades=len(trades),
        configured_span_days=configured_span_days,
        prediction_span_days=prediction_span_days,
        active_trade_span_days=active_trade_span_days,
        max_simultaneous_open_positions=max_simultaneous_open_positions,
        max_open_notional=max_open_notional,
        max_trade_notional=max((trade.notional for trade in trades), default=0.0),
        trades_per_configured_day=(len(trades) / configured_span_days if configured_span_days else 0.0),
        trades_per_prediction_day=(len(trades) / prediction_span_days if prediction_span_days else 0.0),
        trades_per_active_day=(len(trades) / active_trade_span_days if active_trade_span_days else 0.0),
        pnl_per_configured_day=(net_pnl / configured_span_days if configured_span_days else 0.0),
        pnl_per_prediction_day=(net_pnl / prediction_span_days if prediction_span_days else 0.0),
        hit_rate=mean([1 if trade.pnl > 0 else 0 for trade in trades]) if trades else 0.0,
        average_net_return=mean(returns) if returns else 0.0,
        median_net_return=_median(returns),
        gross_pnl=sum(trade.gross_pnl for trade in trades),
        net_pnl=net_pnl,
        sharpe=daily_sharpe,
        daily_sharpe=daily_sharpe,
        trade_level_sharpe=trade_level_sharpe,
        deflated_sharpe=deflated_sharpe,
        multiple_testing_trials=multiple_testing_trials,
        multiple_testing_haircut=multiple_testing_haircut,
        profit_factor=(sum(wins) / sum(losses)) if losses else float("inf") if wins else 0.0,
        max_intratrade_drawdown=max(intratrade_drawdowns, default=0.0),
        average_max_adverse_excursion=mean(adverse_excursions) if adverse_excursions else 0.0,
        total_commission_estimate=sum(trade.commission_estimate for trade in trades),
        total_spread_cost_estimate=sum(trade.spread_cost_estimate for trade in trades),
        total_slippage_cost_estimate=sum(trade.slippage_cost_estimate for trade in trades),
        total_safety_margin_estimate=sum(trade.safety_margin_estimate for trade in trades),
        missed_fills=reject_reasons.get("missed_entry_fill", 0),
        rejected_signals=len(rejections),
        reject_reasons=reject_reasons,
        by_candidate_type=by_type,
        by_side=by_side,
        calibration_metrics={
            "average_brier_score": sum(fold_briers) / len(fold_briers) if fold_briers else 0.0,
            "average_log_loss": sum(fold_log_losses) / len(fold_log_losses) if fold_log_losses else None,
        },
        data_coverage=report.data_coverage,
    )


def run_ml_backtest(
    *,
    report: MetaLabelWalkForwardReport,
    samples: list[MetaLabelSample],
    bars: list[Bar],
    config: AppConfig,
    starting_equity: float,
    requested_notional: float,
    assumed_spread_bps: float,
    entry_limit_offset_bps: float = 0.0,
    enforce_production_gate: bool = True,
    allow_negative_ev_research: bool = False,
    allow_research_short_backtest: bool = False,
    confidence_scaled_sizing: bool = False,
    sizing_mode: str = "fixed",
    sizing_score_field: str = "probability",
    sizing_score_direction: str = "high",
    sizing_base_notional: float = 0.0,
    sizing_mid_notional: float = 0.0,
    sizing_high_notional: float = 0.0,
    sizing_mid_score: float = 0.45,
    sizing_high_score: float = 0.90,
    sizing_max_spread_bps: float = 1.0,
    sizing_min_liquidity_score: float = 0.0,
    dynamic_exit_overlay: bool = False,
    dynamic_exit_checkpoints_minutes: tuple[int, ...] = (15, 30, 60, 120, 240),
    dynamic_exit_checkpoints_seconds: tuple[int, ...] = (),
    dynamic_exit_adverse_bps: float = 25.0,
    dynamic_exit_giveback_bps: float = 35.0,
    dynamic_exit_min_profit_bps: float = 8.0,
    dynamic_exit_weak_probability: float = 0.55,
    dynamic_exit_weak_expected_value_bps: float = 0.0,
    experiment_max_loss_usd: float = 0.0,
    min_expected_gross_edge_bps: float = 0.0,
    selection_score_mode: str = "expected_utility",
    entry_order_model: str = "limit",
) -> tuple[MLBacktestSummary, list[MLBacktestTrade], list[MLBacktestRejection]]:
    if experiment_max_loss_usd < 0:
        raise ValueError("experiment_max_loss_usd must be nonnegative")
    if entry_order_model not in {"limit", "market"}:
        raise ValueError("entry_order_model must be limit or market")
    ordered_bars = sorted(bars, key=lambda bar: bar.timestamp_utc)
    timestamps = [bar.timestamp_utc for bar in ordered_bars]
    sample_by_event = {sample.event_id: sample for sample in samples}
    commission_model = CommissionModel(
        tier_rate=config.cost.tier_rate,
        minimum_commission=config.cost.minimum_commission,
        maximum_commission_rate=config.cost.maximum_commission_rate,
    )
    slippage_model = SlippageModel(base_slippage_bps=config.cost.base_slippage_bps)
    effective_sizing_mode = "confidence" if confidence_scaled_sizing and sizing_mode == "fixed" else sizing_mode

    equity = starting_equity
    max_equity = starting_equity
    trades: list[MLBacktestTrade] = []
    pending_trades: list[PendingMLBacktestTrade] = []
    rejections: list[MLBacktestRejection] = []
    realized_pnls: list[tuple[datetime, float]] = []
    consecutive_losses = 0
    cooldown_until: datetime | None = None
    max_simultaneous_open_positions = 0
    max_open_notional = 0.0
    type_threshold_by_fold = {
        (fold.fold_id, candidate_type): threshold
        for fold in report.folds
        for candidate_type, threshold in fold.candidate_type_thresholds.items()
    }

    def settle_trades_through(timestamp: datetime | None = None) -> None:
        nonlocal equity, max_equity, consecutive_losses, cooldown_until, pending_trades
        due = [
            trade
            for trade in pending_trades
            if timestamp is None or trade.exit_timestamp_utc <= timestamp
        ]
        if not due:
            return
        pending_trades = [
            trade
            for trade in pending_trades
            if timestamp is not None and trade.exit_timestamp_utc > timestamp
        ]
        for pending in sorted(due, key=lambda trade: trade.exit_timestamp_utc):
            equity += pending.pnl
            max_equity = max(max_equity, equity)
            realized_pnls.append((pending.exit_timestamp_utc, pending.pnl))
            if pending.pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            if (
                config.risk.consecutive_loss_limit > 0
                and consecutive_losses >= config.risk.consecutive_loss_limit
            ):
                cooldown_until = pending.exit_timestamp_utc + timedelta(
                    hours=config.risk.cooldown_hours_after_stopouts
                )
                consecutive_losses = 0
            trades.append(
                MLBacktestTrade(
                    fold_id=pending.fold_id,
                    event_id=pending.event_id,
                    timestamp_utc=pending.timestamp_utc,
                    candidate_type=pending.candidate_type,
                    probability=pending.probability,
                    expected_value=pending.expected_value,
                    predicted_return=pending.predicted_return,
                    predicted_downside=pending.predicted_downside,
                    selection_score=pending.selection_score,
                    notional=pending.notional,
                    requested_notional=pending.requested_notional,
                    notional_scale=pending.notional_scale,
                    sizing_mode=pending.sizing_mode,
                    gross_return=pending.gross_return,
                    net_return=pending.net_return,
                    gross_pnl=pending.gross_pnl,
                    pnl=pending.pnl,
                    equity_after=equity,
                    entry_fill_price=pending.entry_fill_price,
                    entry_fill_timestamp_utc=pending.entry_fill_timestamp_utc,
                    entry_fill_fraction=pending.entry_fill_fraction,
                    exit_timestamp_utc=pending.exit_timestamp_utc,
                    outcome_type=pending.outcome_type,
                    round_trip_cost_bps=pending.round_trip_cost_bps,
                    commission_estimate=pending.commission_estimate,
                    spread_cost_estimate=pending.spread_cost_estimate,
                    slippage_cost_estimate=pending.slippage_cost_estimate,
                    safety_margin_estimate=pending.safety_margin_estimate,
                    max_adverse_excursion=pending.max_adverse_excursion,
                    max_favorable_excursion=pending.max_favorable_excursion,
                    side=pending.side,
                    exit_overlay_reason=pending.exit_overlay_reason,
                )
            )

    for prediction in sorted(report.predictions, key=_prediction_timestamp):
        timestamp = _prediction_timestamp(prediction)
        settle_trades_through(timestamp)
        if experiment_max_loss_usd > 0 and (starting_equity - equity) >= experiment_max_loss_usd:
            _record_rejection(
                rejections,
                prediction,
                reason="experiment_max_loss_usd_stop",
                equity=equity,
            )
            break
        sample = sample_by_event.get(prediction.event_id)
        if sample is None:
            _record_rejection(rejections, prediction, reason="missing_sample", equity=equity)
            continue
        if not prediction.should_trade:
            _record_rejection(rejections, prediction, reason=prediction.decision_reason, equity=equity)
            continue
        probability_gate_required = selection_score_mode != "return_first"
        if (
            enforce_production_gate
            and probability_gate_required
            and prediction.probability < config.model.minimum_probability
        ):
            _record_rejection(rejections, prediction, reason="probability_below_threshold", equity=equity)
            continue
        production_ev_gate_ok = (
            _passes_selection_and_ev_gate(
                selection_score_mode=selection_score_mode,
                selection_score=prediction.selection_score,
                expected_value=prediction.expected_value,
                minimum_expected_value=config.model.minimum_expected_value,
            )
            if selection_score_mode == "return_first"
            else prediction.expected_value >= config.model.minimum_expected_value
        )
        if enforce_production_gate and not production_ev_gate_ok:
            _record_rejection(rejections, prediction, reason="expected_value_below_threshold", equity=equity)
            continue
        if (
            not enforce_production_gate
            and not allow_negative_ev_research
            and prediction.expected_value < 0
            and not _research_selection_allows_negative_ev(prediction)
        ):
            _record_rejection(rejections, prediction, reason="negative_ev_research_gate", equity=equity)
            continue
        research_short_allowed = allow_research_short_backtest and not enforce_production_gate
        if (
            config.contract.instrument_model == "spot_crypto"
            and sample.side == Side.SELL.value
            and not research_short_allowed
        ):
            _record_rejection(rejections, prediction, reason="spot_short_not_executable", equity=equity)
            continue
        if min_expected_gross_edge_bps > 0:
            predicted_net_edge_bps = max(prediction.expected_value, prediction.predicted_return) * 10_000
            predicted_gross_edge_bps = predicted_net_edge_bps + sample.round_trip_cost_bps
            if predicted_gross_edge_bps < min_expected_gross_edge_bps:
                _record_rejection(
                    rejections,
                    prediction,
                    reason="expected_gross_edge_below_cost_floor",
                    equity=equity,
                )
                continue
        if assumed_spread_bps > config.risk.max_spread_bps:
            _record_rejection(rejections, prediction, reason="spread_too_wide", equity=equity)
            continue
        if len(pending_trades) >= config.risk.max_open_positions:
            _record_rejection(rejections, prediction, reason="position_overlap", equity=equity)
            continue
        if cooldown_until and timestamp < cooldown_until:
            _record_rejection(rejections, prediction, reason="cooldown", equity=equity)
            continue

        daily_pnl = _period_pnl(realized_pnls, timestamp=timestamp, weekly=False)
        weekly_pnl = _period_pnl(realized_pnls, timestamp=timestamp, weekly=True)
        if (
            config.risk.daily_loss_stop > 0
            and daily_pnl <= -starting_equity * config.risk.daily_loss_stop
        ):
            _record_rejection(rejections, prediction, reason="daily_loss_stop", equity=equity)
            continue
        if (
            config.risk.weekly_loss_stop > 0
            and weekly_pnl <= -starting_equity * config.risk.weekly_loss_stop
        ):
            _record_rejection(rejections, prediction, reason="weekly_loss_stop", equity=equity)
            continue
        rolling_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        if (
            config.risk.rolling_drawdown_stop > 0
            and rolling_drawdown >= config.risk.rolling_drawdown_stop
        ):
            _record_rejection(rejections, prediction, reason="rolling_drawdown_stop", equity=equity)
            break

        cap_requested_notional = _requested_notional_for_sizing_cap(
            requested_notional=requested_notional,
            sizing_mode=effective_sizing_mode,
            base_notional=sizing_base_notional,
            mid_notional=sizing_mid_notional,
            high_notional=sizing_high_notional,
        )
        cap_notional = _estimate_trade_notional(
            equity=equity,
            risk_per_trade=config.risk.risk_per_trade,
            net_stop_loss=sample.net_stop_loss,
            requested_notional=cap_requested_notional,
            max_notional=config.risk.paper_max_notional,
            cap_by_equity=config.contract.instrument_model != "futures",
        )
        trade_notional, notional_scale, applied_sizing_mode = _apply_notional_sizing(
            prediction=prediction,
            sample=sample,
            config=config,
            cap_notional=cap_notional,
            sizing_mode=effective_sizing_mode,
            score_field=sizing_score_field,
            score_direction=sizing_score_direction,
            base_notional=sizing_base_notional,
            mid_notional=sizing_mid_notional,
            high_notional=sizing_high_notional,
            mid_score=sizing_mid_score,
            high_score=sizing_high_score,
            max_spread_bps=sizing_max_spread_bps,
            min_liquidity_score=sizing_min_liquidity_score,
            type_threshold=type_threshold_by_fold.get((prediction.fold_id, prediction.candidate_type)),
        )
        if applied_sizing_mode == "confidence":
            if cap_notional >= config.risk.minimum_fee_efficient_notional:
                trade_notional = max(config.risk.minimum_fee_efficient_notional, trade_notional)
                notional_scale = trade_notional / max(cap_notional, 1e-9)
        elif confidence_scaled_sizing:
            # Backward-compatible guard: an explicit non-confidence sizing mode wins.
            pass
        if trade_notional < 0:
            trade_notional = 0.0
        if config.contract.instrument_model == "spot_crypto" and sample.side == Side.BUY.value:
            spot_exposure_cap = min(equity, config.risk.paper_max_notional)
            available_spot_notional = max(
                spot_exposure_cap - _open_spot_notional(pending_trades),
                0.0,
            )
            if available_spot_notional <= 0:
                _record_rejection(rejections, prediction, reason="insufficient_cash", equity=equity)
                continue
            if trade_notional > available_spot_notional:
                trade_notional = available_spot_notional
                notional_scale = trade_notional / max(cap_notional, 1e-9)
        if trade_notional < config.risk.minimum_fee_efficient_notional:
            _record_rejection(
                rejections,
                prediction,
                reason="notional_below_fee_efficient_size",
                equity=equity,
                proposed_notional=trade_notional,
            )
            continue

        entry_deadline = sample.label_detail.entry_timestamp_utc + timedelta(
            seconds=config.execution.entry_timeout_seconds
        )
        entry_bars = _bars_for_window(
            ordered_bars,
            timestamps,
            start=sample.label_detail.entry_timestamp_utc,
            end=entry_deadline,
        )
        entry_side = Side.SELL if sample.side == Side.SELL.value else Side.BUY
        entry_reference_price = _execution_entry_price(
            entry_bars,
            sample.label_detail.entry_timestamp_utc,
            sample.label_detail.entry_price,
        )
        if entry_order_model == "market":
            fill = _simulate_market_entry_fill(
                entry_bars=entry_bars,
                entry_timestamp=sample.label_detail.entry_timestamp_utc,
                fallback_price=entry_reference_price,
            )
        else:
            limit_adjustment = entry_limit_offset_bps / 10_000
            limit_price = (
                entry_reference_price * (1 + limit_adjustment)
                if entry_side == Side.SELL
                else entry_reference_price * (1 - limit_adjustment)
            )
            fill = simulate_limit_fill(
                entry_side,
                limit_price,
                entry_bars,
                activation_timestamp=sample.label_detail.entry_timestamp_utc,
                latency_seconds=config.execution.simulated_latency_seconds,
                require_trade_through_bps=config.execution.limit_trade_through_bps,
                fill_probability=config.execution.limit_fill_probability,
                fill_fraction=config.execution.partial_fill_fraction,
            )
        if not fill.filled or fill.price is None:
            _record_rejection(rejections, prediction, reason="missed_entry_fill", equity=equity)
            continue
        if fill.timestamp_utc is None:
            _record_rejection(rejections, prediction, reason="missing_fill_timestamp", equity=equity)
            continue

        if _futures_contract_cost_enabled(config):
            filled_notional = _round_futures_notional(
                requested_notional=trade_notional,
                fill_price=fill.price,
                fill_fraction=fill.fill_fraction,
                contract_multiplier=config.cost.futures_contract_multiplier,
            )
        else:
            filled_notional = trade_notional * fill.fill_fraction
        cost = estimate_round_trip_cost(
            filled_notional,
            spread_bps=assumed_spread_bps,
            commission_model=commission_model,
            slippage_model=slippage_model,
            safety_margin_bps=config.cost.safety_margin_bps,
            futures_fee_per_contract=config.cost.futures_fee_per_contract,
            futures_contract_multiplier=config.cost.futures_contract_multiplier,
            reference_price=fill.price,
        )
        exit_bars = _bars_for_window(
            ordered_bars,
            timestamps,
            start=fill.timestamp_utc,
            end=sample.label_detail.vertical_barrier_timestamp_utc,
        )
        replayed_exit = _replay_exit_from_fill(
            sample=sample,
            side=entry_side,
            fill_price=fill.price,
            fill_timestamp=fill.timestamp_utc,
            exit_bars=exit_bars,
            round_trip_cost_bps=cost.total_bps,
            conservative_same_bar=config.labels.conservative_same_bar,
        )
        if replayed_exit is None:
            _record_rejection(rejections, prediction, reason="no_exit_bars_after_fill", equity=equity)
            continue
        if dynamic_exit_overlay:
            replayed_exit = _apply_dynamic_exit_overlay(
                sample=sample,
                prediction=prediction,
                side=entry_side,
                fill_price=fill.price,
                fill_timestamp=fill.timestamp_utc,
                exit_bars=exit_bars,
                replayed_exit=replayed_exit,
                round_trip_cost_bps=cost.total_bps,
                checkpoints_minutes=dynamic_exit_checkpoints_minutes,
                checkpoints_seconds=dynamic_exit_checkpoints_seconds,
                adverse_bps=dynamic_exit_adverse_bps,
                giveback_bps=dynamic_exit_giveback_bps,
                min_profit_bps=dynamic_exit_min_profit_bps,
                weak_probability=dynamic_exit_weak_probability,
                weak_expected_value_bps=dynamic_exit_weak_expected_value_bps,
            )
        gross_return = replayed_exit.gross_return
        net_return = replayed_exit.net_return
        gross_pnl = filled_notional * gross_return
        pnl = filled_notional * net_return
        cost_components = _trade_cost_components(
            filled_notional,
            cost,
            _round_trip_commission_from_cost(filled_notional, cost),
        )
        max_adverse_excursion, max_favorable_excursion = _trade_excursions(
            side=entry_side,
            entry_price=fill.price,
            exit_bars=exit_bars,
            exit_timestamp=replayed_exit.timestamp_utc,
        )
        pending_trades.append(
            PendingMLBacktestTrade(
                fold_id=prediction.fold_id,
                event_id=prediction.event_id,
                timestamp_utc=timestamp,
                candidate_type=prediction.candidate_type,
                probability=prediction.probability,
                expected_value=prediction.expected_value,
                predicted_return=prediction.predicted_return,
                predicted_downside=prediction.predicted_downside,
                selection_score=prediction.selection_score,
                notional=filled_notional,
                requested_notional=trade_notional,
                notional_scale=notional_scale,
                sizing_mode=applied_sizing_mode,
                gross_return=gross_return,
                net_return=net_return,
                gross_pnl=gross_pnl,
                pnl=pnl,
                entry_fill_price=fill.price,
                entry_fill_timestamp_utc=fill.timestamp_utc,
                entry_fill_fraction=fill.fill_fraction,
                exit_timestamp_utc=replayed_exit.timestamp_utc,
                outcome_type=replayed_exit.outcome_type,
                round_trip_cost_bps=cost.total_bps,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                side=sample.side,
                exit_overlay_reason=replayed_exit.overlay_reason,
                **cost_components,
            )
        )
        max_simultaneous_open_positions = max(max_simultaneous_open_positions, len(pending_trades))
        max_open_notional = max(max_open_notional, _open_notional(pending_trades))

    settle_trades_through(None)
    summary = _summarize(
        report=report,
        trades=trades,
        rejections=rejections,
        start_equity=starting_equity,
        max_simultaneous_open_positions=max_simultaneous_open_positions,
        max_open_notional=max_open_notional,
    )
    return summary, trades, rejections


def write_ml_backtest_artifact(
    path: Path,
    summary: MLBacktestSummary,
    trades: list[MLBacktestTrade],
    rejections: list[MLBacktestRejection],
    report: MetaLabelWalkForwardReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "trades": [asdict(trade) for trade in trades],
        "rejections": [asdict(rejection) for rejection in rejections],
        "model_report_summary": {
            "samples": report.samples,
            "folds": len(report.folds),
            "traded_signals": report.traded_signals,
            "net_pnl": report.net_pnl,
            "requested_models": report.requested_models,
            "calibration_method": report.calibration_method,
            "stacker_mode": report.stacker_mode,
            "data_coverage": report.data_coverage,
            "feature_importance_enabled": any(fold.permutation_importance for fold in report.folds),
        },
        "model_report": {
            "folds": [asdict(fold) for fold in report.folds],
            "predictions": [asdict(prediction) for prediction in report.predictions],
            "feature_names": report.feature_names,
            "feature_importance_summary": report_feature_importance_summary(report),
            "shap_importance_summary": report_shap_importance_summary(report),
            "native_importance_summary": report_native_importance_summary(report),
            "model_family_summary": report_model_family_summary(report),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
