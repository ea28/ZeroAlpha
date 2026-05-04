"""Model-driven strategy backtest using walk-forward meta-label predictions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import floor, sqrt
from pathlib import Path
from statistics import mean, pstdev
import json

from zeroalpha.backtest.fills import simulate_limit_fill
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
    prediction_span_days: float
    active_trade_span_days: float
    trades_per_prediction_day: float
    trades_per_active_day: float
    hit_rate: float
    average_net_return: float
    median_net_return: float
    gross_pnl: float
    net_pnl: float
    sharpe: float
    profit_factor: float
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

    if side == Side.SELL:
        lower = fill_price * (1 - sample.net_profit_target - cost_fraction)
        upper = fill_price * (1 + sample.net_stop_loss - cost_fraction)
        for bar in ordered:
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


def _checkpoint_bar(
    ordered: list[Bar],
    *,
    fill_timestamp: datetime,
    replayed_exit_timestamp: datetime,
    checkpoint_minutes: int,
) -> tuple[int, Bar] | None:
    target = fill_timestamp + timedelta(minutes=checkpoint_minutes)
    if target >= replayed_exit_timestamp:
        return None
    for idx, bar in enumerate(ordered):
        if bar.timestamp_utc >= target:
            if bar.timestamp_utc >= replayed_exit_timestamp:
                return None
            return idx, bar
    return None


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
    adverse_bps: float,
    giveback_bps: float,
    min_profit_bps: float,
    weak_probability: float,
    weak_expected_value_bps: float,
) -> ReplayedExit:
    if not checkpoints_minutes:
        return replayed_exit
    ordered = [
        bar
        for bar in sorted(exit_bars, key=lambda row: row.timestamp_utc)
        if fill_timestamp <= bar.timestamp_utc <= replayed_exit.timestamp_utc
    ]
    if not ordered:
        return replayed_exit

    max_favorable_bps = 0.0
    weak_signal = (
        prediction.probability <= weak_probability
        or prediction.expected_value * 10_000 <= weak_expected_value_bps
        or prediction.predicted_return < 0
    )
    for checkpoint in sorted({value for value in checkpoints_minutes if value > 0}):
        row = _checkpoint_bar(
            ordered,
            fill_timestamp=fill_timestamp,
            replayed_exit_timestamp=replayed_exit.timestamp_utc,
            checkpoint_minutes=checkpoint,
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
        reason = ""
        if current_bps <= -adverse_bps and weak_signal:
            reason = f"adverse_{checkpoint}m"
        elif giveback >= giveback_bps and current_bps >= min_profit_bps:
            reason = f"giveback_{checkpoint}m"
        elif checkpoint >= 120 and weak_signal and current_bps <= 0:
            reason = f"weak_no_progress_{checkpoint}m"
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
    raise ValueError("sizing_score_field must be probability, expected_value, predicted_return, or selection_score")


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

    score = _prediction_score_value(prediction, score_field)
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
            base_notional=base_notional,
            mid_notional=mid_notional,
            high_notional=high_notional,
            mid_score=mid_score,
            high_score=high_score,
            max_spread_bps=max_spread_bps if sizing_mode == "liquidity_score_bucket" else 0.0,
            min_liquidity_score=min_liquidity_score if sizing_mode == "liquidity_score_bucket" else 0.0,
        )
    raise ValueError("sizing_mode must be fixed, confidence, score_bucket, or liquidity_score_bucket")


def _research_selection_allows_negative_ev(prediction: FoldPrediction) -> bool:
    if prediction.decision_reason == "candidate_type_calibration":
        return True
    return False


def _open_spot_notional(pending_trades: list[PendingMLBacktestTrade]) -> float:
    return sum(trade.notional for trade in pending_trades if trade.side == Side.BUY.value)


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


def _summarize(
    *,
    report: MetaLabelWalkForwardReport,
    trades: list[MLBacktestTrade],
    rejections: list[MLBacktestRejection],
    start_equity: float,
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
    return MLBacktestSummary(
        start_equity=start_equity,
        end_equity=end_equity,
        total_return=(end_equity / start_equity - 1) if start_equity else 0.0,
        max_drawdown=max_drawdown,
        candidate_predictions=len(report.predictions),
        model_approved_signals=sum(1 for prediction in report.predictions if prediction.should_trade),
        trades=len(trades),
        prediction_span_days=prediction_span_days,
        active_trade_span_days=active_trade_span_days,
        trades_per_prediction_day=(len(trades) / prediction_span_days if prediction_span_days else 0.0),
        trades_per_active_day=(len(trades) / active_trade_span_days if active_trade_span_days else 0.0),
        hit_rate=mean([1 if trade.pnl > 0 else 0 for trade in trades]) if trades else 0.0,
        average_net_return=mean(returns) if returns else 0.0,
        median_net_return=_median(returns),
        gross_pnl=sum(trade.gross_pnl for trade in trades),
        net_pnl=sum(trade.pnl for trade in trades),
        sharpe=_annualized_daily_sharpe(report=report, trades=trades, start_equity=start_equity),
        profit_factor=(sum(wins) / sum(losses)) if losses else float("inf") if wins else 0.0,
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
    sizing_base_notional: float = 0.0,
    sizing_mid_notional: float = 0.0,
    sizing_high_notional: float = 0.0,
    sizing_mid_score: float = 0.45,
    sizing_high_score: float = 0.90,
    sizing_max_spread_bps: float = 1.0,
    sizing_min_liquidity_score: float = 0.0,
    dynamic_exit_overlay: bool = False,
    dynamic_exit_checkpoints_minutes: tuple[int, ...] = (15, 30, 60, 120, 240),
    dynamic_exit_adverse_bps: float = 25.0,
    dynamic_exit_giveback_bps: float = 35.0,
    dynamic_exit_min_profit_bps: float = 8.0,
    dynamic_exit_weak_probability: float = 0.55,
    dynamic_exit_weak_expected_value_bps: float = 0.0,
    experiment_max_loss_usd: float = 0.0,
) -> tuple[MLBacktestSummary, list[MLBacktestTrade], list[MLBacktestRejection]]:
    if experiment_max_loss_usd < 0:
        raise ValueError("experiment_max_loss_usd must be nonnegative")
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
            if consecutive_losses >= config.risk.consecutive_loss_limit:
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
        if enforce_production_gate and prediction.probability < config.model.minimum_probability:
            _record_rejection(rejections, prediction, reason="probability_below_threshold", equity=equity)
            continue
        if enforce_production_gate and prediction.expected_value < config.model.minimum_expected_value:
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
        if daily_pnl <= -starting_equity * config.risk.daily_loss_stop:
            _record_rejection(rejections, prediction, reason="daily_loss_stop", equity=equity)
            continue
        if weekly_pnl <= -starting_equity * config.risk.weekly_loss_stop:
            _record_rejection(rejections, prediction, reason="weekly_loss_stop", equity=equity)
            continue
        rolling_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        if rolling_drawdown >= config.risk.rolling_drawdown_stop:
            _record_rejection(rejections, prediction, reason="rolling_drawdown_stop", equity=equity)
            break

        cap_notional = _estimate_trade_notional(
            equity=equity,
            risk_per_trade=config.risk.risk_per_trade,
            net_stop_loss=sample.net_stop_loss,
            requested_notional=requested_notional,
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
            available_spot_notional = max(equity - _open_spot_notional(pending_trades), 0.0)
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
        limit_adjustment = entry_limit_offset_bps / 10_000
        limit_price = (
            sample.label_detail.entry_price * (1 + limit_adjustment)
            if entry_side == Side.SELL
            else sample.label_detail.entry_price * (1 - limit_adjustment)
        )
        fill = simulate_limit_fill(
            entry_side,
            limit_price,
            entry_bars,
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
                side=sample.side,
                exit_overlay_reason=replayed_exit.overlay_reason,
                **cost_components,
            )
        )

    settle_trades_through(None)
    summary = _summarize(report=report, trades=trades, rejections=rejections, start_equity=starting_equity)
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
