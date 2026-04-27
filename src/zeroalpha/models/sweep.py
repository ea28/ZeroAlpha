"""Research sweeps for label geometry and horizon settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

from zeroalpha.backtest.ml import MLBacktestSummary, run_ml_backtest
from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import AppConfig
from zeroalpha.domain import Bar
from zeroalpha.models.dataset import build_meta_label_samples
from zeroalpha.models.ensemble import (
    MetaLabelWalkForwardReport,
    report_candidate_type_summary,
    run_meta_label_walk_forward,
)


@dataclass(frozen=True, slots=True)
class LabelSweepResult:
    net_profit_target: float
    net_stop_loss: float
    max_holding_hours: int
    samples: int
    folds: int
    traded_signals: int
    net_pnl: float
    average_brier_score: float
    average_log_loss: float | None
    max_probability: float
    fitted_models: list[str]
    candidate_type_summary: dict[str, dict[str, float]]
    backtest_trades: int
    backtest_net_pnl: float
    backtest_total_return: float
    backtest_sharpe: float
    backtest_max_drawdown: float
    backtest_profit_factor: float
    backtest_turnover_per_prediction_day: float
    backtest_missed_fill_rate: float
    side_summary: dict[str, dict[str, float]]
    score: float


def _report_to_result(
    *,
    report: MetaLabelWalkForwardReport,
    backtest_summary: MLBacktestSummary,
    net_profit_target: float,
    net_stop_loss: float,
    max_holding_hours: int,
    optimize_metric: str,
) -> LabelSweepResult:
    briers = [fold.brier_score for fold in report.folds]
    log_losses = [fold.log_loss for fold in report.folds if fold.log_loss is not None]
    fitted_models = sorted({name for fold in report.folds for name in fold.fitted_models})
    missed_fill_rate = (
        backtest_summary.missed_fills / backtest_summary.model_approved_signals
        if backtest_summary.model_approved_signals
        else 0.0
    )
    return LabelSweepResult(
        net_profit_target=net_profit_target,
        net_stop_loss=net_stop_loss,
        max_holding_hours=max_holding_hours,
        samples=report.samples,
        folds=len(report.folds),
        traded_signals=report.traded_signals,
        net_pnl=report.net_pnl,
        average_brier_score=sum(briers) / len(briers) if briers else 0.0,
        average_log_loss=sum(log_losses) / len(log_losses) if log_losses else None,
        max_probability=max((fold.probability_max for fold in report.folds), default=0.0),
        fitted_models=fitted_models,
        candidate_type_summary=report_candidate_type_summary(report),
        backtest_trades=backtest_summary.trades,
        backtest_net_pnl=backtest_summary.net_pnl,
        backtest_total_return=backtest_summary.total_return,
        backtest_sharpe=backtest_summary.sharpe,
        backtest_max_drawdown=backtest_summary.max_drawdown,
        backtest_profit_factor=backtest_summary.profit_factor,
        backtest_turnover_per_prediction_day=backtest_summary.trades_per_prediction_day,
        backtest_missed_fill_rate=missed_fill_rate,
        side_summary=backtest_summary.by_side,
        score=_optimization_score(backtest_summary, optimize_metric),
    )


def _optimization_score(summary: MLBacktestSummary, optimize_metric: str) -> float:
    if optimize_metric == "net_pnl":
        return summary.net_pnl
    if optimize_metric == "calmar":
        return summary.total_return / max(summary.max_drawdown, 1e-9)
    if optimize_metric != "sharpe":
        raise ValueError("optimize_metric must be sharpe, net_pnl, or calmar")
    return summary.sharpe


def run_label_geometry_sweep(
    bars: list[Bar],
    *,
    config: AppConfig,
    assumed_spread_bps: float,
    research_notional: float,
    context_bars: Mapping[str, list[Bar]] | None,
    net_profit_targets: list[float],
    net_stop_losses: list[float],
    max_holding_hours_values: list[int],
    model_names: list[str],
    starting_equity: float = 10_000.0,
    adaptive_threshold: bool = True,
    stacker_mode: str = "average",
    adaptive_minimum_threshold: float = 0.0,
    optimize_metric: str = "sharpe",
    target_trades_per_day: float | None = None,
    allow_negative_ev_target_frequency: bool = False,
    candidate_type_thresholds: bool = False,
    empirical_payoff_ev: bool = False,
    confidence_scaled_sizing: bool = False,
    enforce_production_gate: bool = True,
    allow_research_short_backtest: bool = False,
    candidate_side_mode: str = "long",
    allow_short_research: bool = False,
    candidate_mode: str = "rules",
    selection_score_mode: str = "expected_value",
    specialist_models: bool = False,
    require_calibrated_selection: bool = False,
    min_signal_spacing_hours: float = 0.0,
    max_signals_per_group_per_day: int = 0,
    max_signals_per_timestamp: int = 0,
) -> list[LabelSweepResult]:
    results: list[LabelSweepResult] = []
    for max_holding_hours in max_holding_hours_values:
        for net_profit_target in net_profit_targets:
            for net_stop_loss in net_stop_losses:
                sweep_config = replace(
                    config,
                    labels=replace(
                        config.labels,
                        max_holding_hours=max_holding_hours,
                        net_profit_target=net_profit_target,
                        net_stop_loss=net_stop_loss,
                    ),
                )
                samples = build_meta_label_samples(
                    bars,
                    config=sweep_config,
                    assumed_spread_bps=assumed_spread_bps,
                    research_notional=research_notional,
                    context_bars=context_bars,
                    candidate_config=CandidateGenerationConfig(
                        max_holding_hours=max_holding_hours,
                        side_mode=candidate_side_mode,
                        allow_short_research=allow_short_research,
                        mode=candidate_mode,
                    ),
                )
                if len(samples) < 60:
                    continue
                report = run_meta_label_walk_forward(
                    samples,
                    config=sweep_config,
                    model_names=model_names,
                    adaptive_threshold=adaptive_threshold,
                    min_calibration_trades=5,
                    stacker_mode=stacker_mode,
                    adaptive_minimum_threshold=adaptive_minimum_threshold,
                    target_trades_per_day=target_trades_per_day,
                    allow_negative_ev_target_frequency=allow_negative_ev_target_frequency,
                    candidate_type_thresholds=candidate_type_thresholds,
                    empirical_payoff_ev=empirical_payoff_ev,
                    selection_score_mode=selection_score_mode,
                    specialist_models=specialist_models,
                    require_calibrated_selection=require_calibrated_selection,
                    min_signal_spacing_hours=min_signal_spacing_hours,
                    max_signals_per_group_per_day=max_signals_per_group_per_day,
                    max_signals_per_timestamp=max_signals_per_timestamp,
                )
                backtest_summary, _, _ = run_ml_backtest(
                    report=report,
                    samples=samples,
                    bars=bars,
                    config=sweep_config,
                    starting_equity=starting_equity,
                    requested_notional=research_notional,
                    assumed_spread_bps=assumed_spread_bps,
                    enforce_production_gate=enforce_production_gate,
                    allow_negative_ev_research=allow_negative_ev_target_frequency,
                    allow_research_short_backtest=allow_research_short_backtest,
                    confidence_scaled_sizing=confidence_scaled_sizing,
                )
                results.append(
                    _report_to_result(
                        report=report,
                        backtest_summary=backtest_summary,
                        net_profit_target=net_profit_target,
                        net_stop_loss=net_stop_loss,
                        max_holding_hours=max_holding_hours,
                        optimize_metric=optimize_metric,
                    )
                )
    return sorted(results, key=lambda row: (row.score, row.backtest_net_pnl, row.traded_signals), reverse=True)


def sweep_results_asdict(results: list[LabelSweepResult]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]
