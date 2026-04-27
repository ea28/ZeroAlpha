"""Runnable candidate-rule backtest for BTCUSDT archive data."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
import json
from bisect import bisect_right

from zeroalpha.candidates.events import CandidateGenerationConfig, candidate_for_index
from zeroalpha.config import AppConfig
from zeroalpha.costs import CommissionModel, RoundTripCost, SlippageModel, estimate_round_trip_cost
from zeroalpha.data.external.binance import fetch_klines_archive_range
from zeroalpha.domain import Bar, CandidateEvent, TripleBarrierLabel
from zeroalpha.labels.triple_barrier import label_long_event
from zeroalpha.timeutils import ensure_utc


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    event: CandidateEvent
    label: TripleBarrierLabel
    notional: float
    pnl: float
    equity_after: float
    round_trip_cost_bps: float
    commission_estimate: float
    spread_cost_estimate: float
    slippage_cost_estimate: float
    safety_margin_estimate: float


@dataclass(frozen=True, slots=True)
class BacktestRejection:
    timestamp_utc: datetime
    reason: str
    candidate_type: str
    equity: float
    proposed_notional: float = 0.0


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    symbol: str
    interval: str
    start: str
    end: str
    bars: int
    start_equity: float
    end_equity: float
    total_return: float
    max_drawdown: float
    trades: int
    hit_rate: float
    average_net_return: float
    median_net_return: float
    profit_factor: float
    total_commission_estimate: float
    total_spread_cost_estimate: float
    total_slippage_cost_estimate: float
    total_safety_margin_estimate: float
    average_notional: float
    rejected_candidates: int
    reject_reasons: dict[str, int]
    by_candidate_type: dict[str, dict[str, float]]


def _estimate_trade_notional(
    *,
    equity: float,
    risk_per_trade: float,
    net_stop_loss: float,
    requested_notional: float,
    max_notional: float,
) -> float:
    if equity <= 0:
        return 0.0
    risk_based = equity * risk_per_trade / net_stop_loss
    return max(0.0, min(risk_based, requested_notional, max_notional, equity))


def _dollar_component(notional: float, bps: float) -> float:
    return notional * bps / 10_000


def _trade_cost_components(notional: float, cost: RoundTripCost, commission: float) -> dict[str, float]:
    return {
        "commission_estimate": commission,
        "spread_cost_estimate": _dollar_component(notional, cost.spread_bps),
        "slippage_cost_estimate": _dollar_component(notional, cost.slippage_bps),
        "safety_margin_estimate": _dollar_component(notional, cost.safety_margin_bps),
    }


def _period_pnl(realized: list[tuple[datetime, float]], *, timestamp: datetime, weekly: bool) -> float:
    timestamp = ensure_utc(timestamp)
    if weekly:
        year, week, _ = timestamp.isocalendar()
        return sum(pnl for ts, pnl in realized if ts.isocalendar()[:2] == (year, week))
    day = timestamp.date()
    return sum(pnl for ts, pnl in realized if ensure_utc(ts).date() == day)


def _record_rejection(
    rejections: list[BacktestRejection],
    event: CandidateEvent,
    *,
    reason: str,
    equity: float,
    proposed_notional: float = 0.0,
) -> None:
    rejections.append(
        BacktestRejection(
            timestamp_utc=event.timestamp_utc,
            reason=reason,
            candidate_type=event.candidate_type,
            equity=equity,
            proposed_notional=proposed_notional,
        )
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarize_trades(
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    bars: list[Bar],
    trades: list[BacktestTrade],
    rejections: list[BacktestRejection],
    start_equity: float,
) -> BacktestSummary:
    equity_curve = [start_equity]
    equity_curve.extend(trade.equity_after for trade in trades)
    peak = start_equity
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    returns = [trade.label.net_return for trade in trades]
    wins = [trade.pnl for trade in trades if trade.pnl > 0]
    losses = [-trade.pnl for trade in trades if trade.pnl < 0]
    by_type: dict[str, dict[str, float]] = {}
    for trade in trades:
        bucket = by_type.setdefault(
            trade.event.candidate_type,
            {"trades": 0.0, "wins": 0.0, "net_pnl": 0.0, "average_net_return": 0.0},
        )
        bucket["trades"] += 1
        bucket["wins"] += trade.label.label
        bucket["net_pnl"] += trade.pnl
        bucket["average_net_return"] += trade.label.net_return
    for bucket in by_type.values():
        if bucket["trades"]:
            bucket["hit_rate"] = bucket["wins"] / bucket["trades"]
            bucket["average_net_return"] /= bucket["trades"]
    end_equity = equity_curve[-1]
    reject_reasons: dict[str, int] = {}
    for rejection in rejections:
        reject_reasons[rejection.reason] = reject_reasons.get(rejection.reason, 0) + 1
    return BacktestSummary(
        symbol=symbol,
        interval=interval,
        start=ensure_utc(start).isoformat(),
        end=ensure_utc(end).isoformat(),
        bars=len(bars),
        start_equity=start_equity,
        end_equity=end_equity,
        total_return=(end_equity / start_equity - 1) if start_equity else 0.0,
        max_drawdown=max_drawdown,
        trades=len(trades),
        hit_rate=mean([trade.label.label for trade in trades]) if trades else 0.0,
        average_net_return=mean(returns) if returns else 0.0,
        median_net_return=_median(returns),
        profit_factor=(sum(wins) / sum(losses)) if losses else float("inf") if wins else 0.0,
        total_commission_estimate=sum(trade.commission_estimate for trade in trades),
        total_spread_cost_estimate=sum(trade.spread_cost_estimate for trade in trades),
        total_slippage_cost_estimate=sum(trade.slippage_cost_estimate for trade in trades),
        total_safety_margin_estimate=sum(trade.safety_margin_estimate for trade in trades),
        average_notional=mean([trade.notional for trade in trades]) if trades else 0.0,
        rejected_candidates=len(rejections),
        reject_reasons=reject_reasons,
        by_candidate_type=by_type,
    )


def run_candidate_backtest(
    *,
    config: AppConfig,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    starting_equity: float,
    notional: float,
    assumed_spread_bps: float,
    cache_dir: Path,
) -> tuple[BacktestSummary, list[BacktestTrade], list[BacktestRejection]]:
    bars = fetch_klines_archive_range(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        cache_dir=cache_dir,
    )
    if len(bars) < 100:
        raise ValueError("not enough bars fetched for backtest")

    commission_model = CommissionModel(
        tier_rate=config.cost.tier_rate,
        minimum_commission=config.cost.minimum_commission,
        maximum_commission_rate=config.cost.maximum_commission_rate,
    )
    slippage_model = SlippageModel(base_slippage_bps=config.cost.base_slippage_bps)

    equity = starting_equity
    max_equity = starting_equity
    trades: list[BacktestTrade] = []
    rejections: list[BacktestRejection] = []
    realized_pnls: list[tuple[datetime, float]] = []
    consecutive_losses = 0
    cooldown_until: datetime | None = None
    generation_config = CandidateGenerationConfig(
        max_holding_hours=config.labels.max_holding_hours,
        min_history_bars=240,
    )
    next_allowed_idx = generation_config.min_history_bars
    idx = generation_config.min_history_bars
    timestamps = [bar.timestamp_utc for bar in bars]
    while idx < len(bars) - 2:
        if idx < next_allowed_idx:
            idx += 1
            continue
        event = candidate_for_index(bars, idx, config=generation_config)
        if event is None:
            idx += 1
            continue

        if assumed_spread_bps > config.risk.max_spread_bps:
            _record_rejection(rejections, event, reason="spread_too_wide", equity=equity)
            idx += 1
            continue

        if cooldown_until and event.timestamp_utc < cooldown_until:
            _record_rejection(rejections, event, reason="cooldown", equity=equity)
            idx += 1
            continue

        daily_pnl = _period_pnl(realized_pnls, timestamp=event.timestamp_utc, weekly=False)
        weekly_pnl = _period_pnl(realized_pnls, timestamp=event.timestamp_utc, weekly=True)
        if daily_pnl <= -starting_equity * config.risk.daily_loss_stop:
            _record_rejection(rejections, event, reason="daily_loss_stop", equity=equity)
            idx += 1
            continue
        if weekly_pnl <= -starting_equity * config.risk.weekly_loss_stop:
            _record_rejection(rejections, event, reason="weekly_loss_stop", equity=equity)
            idx += 1
            continue
        rolling_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        if rolling_drawdown >= config.risk.rolling_drawdown_stop:
            _record_rejection(rejections, event, reason="rolling_drawdown_stop", equity=equity)
            break

        trade_notional = _estimate_trade_notional(
            equity=equity,
            risk_per_trade=config.risk.risk_per_trade,
            net_stop_loss=config.labels.net_stop_loss,
            requested_notional=notional,
            max_notional=config.risk.paper_max_notional,
        )
        if trade_notional < config.risk.minimum_fee_efficient_notional:
            _record_rejection(
                rejections,
                event,
                reason="notional_below_fee_efficient_size",
                equity=equity,
                proposed_notional=trade_notional,
            )
            idx += 1
            continue

        cost = estimate_round_trip_cost(
            trade_notional,
            spread_bps=assumed_spread_bps,
            commission_model=commission_model,
            slippage_model=slippage_model,
            safety_margin_bps=config.cost.safety_margin_bps,
        )
        entry_bar = bars[idx + 1]
        horizon_end = bisect_right(timestamps, event.vertical_barrier_timestamp_utc)
        future = bars[idx + 1 : horizon_end]
        if not future:
            _record_rejection(rejections, event, reason="no_future_bars_before_vertical_barrier", equity=equity)
            idx += 1
            continue
        label = label_long_event(
            event,
            future,
            entry_price=entry_bar.open,
            net_profit_target=config.labels.net_profit_target,
            net_stop_loss=config.labels.net_stop_loss,
            round_trip_cost_bps=cost.total_bps,
            conservative_same_bar=config.labels.conservative_same_bar,
        )
        pnl = trade_notional * label.net_return
        equity += pnl
        max_equity = max(max_equity, equity)
        realized_pnls.append((label.exit_timestamp_utc, pnl))
        if pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        if consecutive_losses >= config.risk.consecutive_loss_limit:
            cooldown_until = label.exit_timestamp_utc + timedelta(
                hours=config.risk.cooldown_hours_after_stopouts
            )
            consecutive_losses = 0

        cost_components = _trade_cost_components(
            trade_notional,
            cost,
            commission_model.round_trip_commission(trade_notional),
        )
        trades.append(
            BacktestTrade(
                event=event,
                label=label,
                notional=trade_notional,
                pnl=pnl,
                equity_after=equity,
                round_trip_cost_bps=cost.total_bps,
                **cost_components,
            )
        )
        exit_idx = next(
            (j for j in range(idx + 1, len(bars)) if bars[j].timestamp_utc >= label.exit_timestamp_utc),
            min(horizon_end, len(bars) - 1),
        )
        next_allowed_idx = exit_idx + 1
        idx = next_allowed_idx

    summary = summarize_trades(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        bars=bars,
        trades=trades,
        rejections=rejections,
        start_equity=starting_equity,
    )
    return summary, trades, rejections


def write_backtest_artifact(
    path: Path,
    summary: BacktestSummary,
    trades: list[BacktestTrade],
    rejections: list[BacktestRejection] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "trades": [
            {
                "event": asdict(trade.event),
                "label": asdict(trade.label),
                "notional": trade.notional,
                "pnl": trade.pnl,
                "equity_after": trade.equity_after,
            }
            for trade in trades
        ],
        "rejections": [asdict(rejection) for rejection in rejections or []],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
