"""Local IBKR 1-minute/1-second-replay experiment suite.

The experiments in this module are deliberately command-line wrappers around
``zeroalpha backtest ml``.  That keeps local research aligned with the same
walk-forward model, feature, sizing, and execution replay code used by the
paper runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, Mapping


DEFAULT_OUTPUT_DIR = Path("artifacts/backtests/ibkr_1s_experiments_20260504/next_experiment_suite")
DEFAULT_PRIMARY_BARS = Path("data/raw/ibkr/historical_btcusd_paxos_aggtrades_1m_20260503_latest.jsonl")
DEFAULT_ETH_BARS = Path("data/raw/ibkr/historical_ethusd_paxos_aggtrades_1m_20260503_latest.jsonl")
DEFAULT_MBT_BARS = Path("data/raw/ibkr/historical_mbt_202605_cme_trades_1m_20260503_latest.jsonl")
DEFAULT_BTC_BIDASK_BARS = Path(
    "data/raw/ibkr/historical_btcusd_paxos_bidask_1m_20260503_latest.jsonl"
)
DEFAULT_ETH_BIDASK_BARS = Path(
    "data/raw/ibkr/historical_ethusd_paxos_bidask_1m_20260503_latest.jsonl"
)
DEFAULT_MBT_BIDASK_BARS = Path(
    "data/raw/ibkr/historical_mbt_202605_cme_bidask_1m_20260503_latest.jsonl"
)
DEFAULT_MET_BARS = Path("data/raw/ibkr/historical_met_202605_cme_trades_1m_20260503_latest.jsonl")
DEFAULT_MET_BIDASK_BARS = Path(
    "data/raw/ibkr/historical_met_202605_cme_bidask_1m_20260503_latest.jsonl"
)
DEFAULT_1S_BARS = Path("data/raw/ibkr/historical_btcusd_paxos_aggtrades_1s_20260504_6h_merged.jsonl")
DEFAULT_1M_FROM_1S_BARS = Path(
    "data/raw/ibkr/historical_btcusd_paxos_aggtrades_1m_from_1s_20260504_6h_merged.jsonl"
)

BASE_CONTEXT_BARS = f"ETH={DEFAULT_ETH_BARS},MBT={DEFAULT_MBT_BARS}"
PHASE17_CONTEXT_BARS = (
    f"IBKR_BTC_BIDASK={DEFAULT_BTC_BIDASK_BARS},IBKR_MBT={DEFAULT_MBT_BARS},"
    f"IBKR_MBT_BIDASK={DEFAULT_MBT_BIDASK_BARS}"
)
FULL_CONTEXT_BARS = (
    f"ETH={DEFAULT_ETH_BARS},MBT={DEFAULT_MBT_BARS},"
    f"BTC_BIDASK={DEFAULT_BTC_BIDASK_BARS},ETH_BIDASK={DEFAULT_ETH_BIDASK_BARS},"
    f"MBT_BIDASK={DEFAULT_MBT_BIDASK_BARS},MET={DEFAULT_MET_BARS},"
    f"MET_BIDASK={DEFAULT_MET_BIDASK_BARS}"
)

ArgValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class IbkrExperiment:
    name: str
    category: str
    description: str
    overrides: tuple[tuple[str, ArgValue], ...] = ()


@dataclass(frozen=True, slots=True)
class IbkrExperimentResult:
    name: str
    category: str
    description: str
    artifact: str
    status: str
    trades: int = 0
    configured_span_days: float = 0.0
    trades_per_prediction_day: float = 0.0
    trades_per_configured_day: float = 0.0
    trades_per_active_day: float = 0.0
    pnl_per_prediction_day: float = 0.0
    pnl_per_configured_day: float = 0.0
    candidate_predictions: int = 0
    model_approved_signals: int = 0
    rejected_signals: int = 0
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    daily_sharpe: float = 0.0
    trade_level_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    multiple_testing_trials: int = 1
    multiple_testing_haircut: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_intratrade_drawdown: float = 0.0
    average_max_adverse_excursion: float = 0.0
    hit_rate: float = 0.0
    total_commission_estimate: float = 0.0
    total_slippage_cost_estimate: float = 0.0
    total_spread_cost_estimate: float = 0.0
    primary_interval: str = ""
    label_interval: str = ""
    execution_interval: str = ""
    uses_one_second_execution: bool = False
    elapsed_seconds: float = 0.0
    log_path: str = ""
    error: str = ""

    @property
    def rank_key(self) -> tuple[float, float, float, int]:
        return (self.net_pnl, self.sharpe, -self.max_drawdown, self.trades)


def _items(values: Mapping[str, ArgValue]) -> tuple[tuple[str, ArgValue], ...]:
    return tuple(values.items())


def _append_arg(args: list[str], option: str, value: ArgValue) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(option)
        return
    args.extend([option, str(value)])


def _base_command(*, python_executable: str, output_path: Path) -> list[str]:
    return [
        python_executable,
        "-m",
        "zeroalpha.cli",
        "backtest",
        "ml",
        "--config",
        "configs/paper.example.toml",
        "--symbol",
        "BTCUSDT",
        "--interval",
        "1m",
        "--context-symbols",
        "",
        "--primary-bars-jsonl",
        str(DEFAULT_PRIMARY_BARS),
        "--context-bars-jsonl",
        BASE_CONTEXT_BARS,
        "--start",
        "2026-04-26T07:00:00+00:00",
        "--end",
        "2026-05-01T20:00:00+00:00",
        "--allow-data-gaps",
        "--minimum-data-coverage",
        "0.85",
        "--starting-equity",
        "10000",
        "--paper-max-notional",
        "10000",
        "--notional",
        "1250",
        "--instrument-model",
        "spot_crypto",
        "--tier-rate",
        "0.0018",
        "--minimum-commission",
        "1.75",
        "--assumed-spread-bps",
        "0.1",
        "--base-slippage-bps",
        "0.5",
        "--safety-margin-bps",
        "1.0",
        "--candidate-mode",
        "dense",
        "--side-mode",
        "long",
        "--dense-stride-bars",
        "10",
        "--min-history-bars",
        "240",
        "--max-holding-hours",
        "24",
        "--net-profit-target",
        "0.001",
        "--net-stop-loss",
        "0.001",
        "--minimum-gross-profit-bps",
        "200",
        "--minimum-gross-stop-bps",
        "100",
        "--minimum-probability",
        "0",
        "--minimum-expected-value",
        "-0.02",
        "--target-frequency-mode",
        "quota",
        "--target-trades-per-day",
        "12",
        "--selection-score",
        "return_first",
        "--optimize-metric",
        "net_pnl",
        "--models",
        "extratrees",
        "--stacker",
        "average",
        "--specialist-models",
        "--train-size",
        "180",
        "--calibration-size",
        "60",
        "--test-size",
        "90",
        "--embargo-hours",
        "0",
        "--max-open-positions",
        "8",
        "--max-signals-per-timestamp",
        "1",
        "--consecutive-loss-limit",
        "0",
        "--output",
        str(output_path),
    ]


def experiment_artifact_path(output_dir: Path, experiment: IbkrExperiment) -> Path:
    return output_dir / experiment.category / f"{experiment.name}.json"


def experiment_log_path(output_dir: Path, experiment: IbkrExperiment) -> Path:
    return output_dir / "logs" / experiment.category / f"{experiment.name}.log"


def build_backtest_command(
    experiment: IbkrExperiment,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python_executable: str = sys.executable,
) -> list[str]:
    output_path = experiment_artifact_path(output_dir, experiment)
    args = _base_command(python_executable=python_executable, output_path=output_path)
    for option, value in experiment.overrides:
        _append_arg(args, option, value)
    return args


def _with_research_gate(overrides: Mapping[str, ArgValue] | None = None) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--research-gate": True,
        "--allow-negative-ev-frequency-probe": True,
    }
    if overrides:
        values.update(overrides)
    return _items(values)


def _sizing_name(notional: int, slots: int) -> str:
    return f"notional{notional}_slots{slots}"


def _phase17_overrides(overrides: Mapping[str, ArgValue] | None = None) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--context-symbols": "ETHUSDT,SOLUSDT,ETHBTC",
        "--context-bars-jsonl": PHASE17_CONTEXT_BARS,
        "--start": "2026-04-26T22:01:00+00:00",
        "--end": "2026-05-01T20:00:00+00:00",
        "--starting-equity": 30_000,
        "--paper-max-notional": 30_000,
        "--notional": 15_000,
        "--dense-stride-bars": 15,
        "--min-history-bars": 1_438,
        "--max-holding-hours": 6,
        "--net-profit-target": 0.001,
        "--net-stop-loss": 0.001,
        "--minimum-gross-profit-bps": 100,
        "--minimum-gross-stop-bps": 80,
        "--target-frequency-mode": "online",
        "--target-trades-per-day": 16,
        "--respect-open-positions": True,
        "--capacity-release-mode": "planned",
        "--adaptive-selection-score-floor": True,
        "--selection-score": "expected_value",
        "--models": "histgb,lightgbm,xgboost,catboost,extratrees",
        "--stacker": "weighted",
        "--calibration-method": "sigmoid",
        "--empirical-payoff-ev": True,
        "--hpo": True,
        "--hpo-profile": "capacity",
        "--hpo-trials": 16,
        "--train-size": 0,
        "--calibration-size": 0,
        "--test-size": 0,
        "--max-open-positions": 6,
        "--sizing-mode": "score_bucket",
        "--sizing-score-field": "probability",
        "--sizing-base-notional": 5_000,
        "--sizing-mid-notional": 10_000,
        "--sizing-high-notional": 15_000,
        "--sizing-mid-score": 0.45,
        "--sizing-high-score": 0.9,
    }
    if overrides:
        values.update(overrides)
    return _with_research_gate(values)


def _phase17_light_overrides(overrides: Mapping[str, ArgValue] | None = None) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--context-symbols": "ETHUSDT,SOLUSDT,ETHBTC",
        "--context-bars-jsonl": PHASE17_CONTEXT_BARS,
        "--start": "2026-04-26T22:01:00+00:00",
        "--end": "2026-05-01T20:00:00+00:00",
        "--starting-equity": 10_000,
        "--paper-max-notional": 10_000,
        "--notional": 1_250,
        "--dense-stride-bars": 15,
        "--min-history-bars": 1_438,
        "--max-holding-hours": 6,
        "--net-profit-target": 0.001,
        "--net-stop-loss": 0.001,
        "--minimum-gross-profit-bps": 100,
        "--minimum-gross-stop-bps": 80,
        "--target-frequency-mode": "online",
        "--target-trades-per-day": 16,
        "--respect-open-positions": True,
        "--capacity-release-mode": "actual",
        "--adaptive-selection-score-floor": True,
        "--selection-score": "expected_value",
        "--models": "histgb,extratrees,lightgbm",
        "--stacker": "weighted",
        "--calibration-method": "sigmoid",
        "--empirical-payoff-ev": True,
        "--train-size": 0,
        "--calibration-size": 0,
        "--test-size": 0,
        "--max-open-positions": 8,
        "--sizing-mode": "fixed",
    }
    if overrides:
        values.update(overrides)
    return _with_research_gate(values)


def _phase17_90k_overrides(overrides: Mapping[str, ArgValue] | None = None) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--starting-equity": 90_000,
        "--paper-max-notional": 90_000,
        "--notional": 45_000,
        "--sizing-base-notional": 15_000,
        "--sizing-mid-notional": 30_000,
        "--sizing-high-notional": 45_000,
    }
    if overrides:
        values.update(overrides)
    return _phase17_overrides(values)


def _phase17_90k_no_breakout_overrides(
    overrides: Mapping[str, ArgValue] | None = None,
) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--exclude-setup-families": "dense_breakout_momentum",
    }
    if overrides:
        values.update(overrides)
    return _phase17_90k_overrides(values)


def _bucket_slug(base: int, mid: int, high: int) -> str:
    return f"b{base // 1000}k_m{mid // 1000}k_h{high // 1000}k"


def _live_valid_strict_10k_overrides(
    overrides: Mapping[str, ArgValue] | None = None,
) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--entry-order-model": "market",
        "--label-bars-jsonl": str(DEFAULT_1S_BARS),
        "--execution-bars-jsonl": str(DEFAULT_1S_BARS),
        "--strict-live-valid-1s": True,
        "--min-live-valid-1s-days": 7,
        "--preferred-live-valid-1s-days": 14,
        "--min-live-valid-folds": 2,
        "--starting-equity": 10_000,
        "--paper-max-notional": 10_000,
        "--instrument-model": "spot_crypto",
        "--side-mode": "long",
        "--target-frequency-mode": "online",
        "--respect-open-positions": True,
        "--capacity-release-mode": "planned",
        "--adaptive-horizon": True,
        "--min-holding-seconds": 1,
        "--adaptive-horizon-granularity-seconds": 1,
        "--dynamic-exit-overlay": True,
        "--dynamic-exit-checkpoints-seconds": "5,30",
        "--dynamic-exit-checkpoints-minutes": "1,5,15,30,60",
        "--dynamic-exit-adverse-bps": 25,
        "--dynamic-exit-giveback-bps": 35,
        "--dynamic-exit-min-profit-bps": 8,
        "--min-expected-gross-edge-bps": 38.2,
        "--models": "histgb,extratrees,lightgbm,xgboost,catboost",
        "--stacker": "weighted",
        "--calibration-method": "sigmoid",
        "--hpo": True,
        "--hpo-profile": "capacity",
        "--hpo-trials": 24,
        "--notional": 4_000,
        "--selection-score": "capital_efficiency",
        "--sizing-mode": "score_bucket",
        "--sizing-score-field": "expected_value",
        "--sizing-score-direction": "low",
        "--sizing-base-notional": 1_000,
        "--sizing-mid-notional": 3_000,
        "--sizing-high-notional": 4_000,
        "--sizing-mid-score": 0.0072,
        "--sizing-high-score": 0.0108,
        "--target-trades-per-day": 20,
        "--max-open-positions": 5,
    }
    if overrides:
        values.update(overrides)
    return _strict_1s_label_replay_overrides(values)


def build_live_valid_strict_10k_experiment_suite() -> list[IbkrExperiment]:
    experiments: list[IbkrExperiment] = []
    model_sets = {
        "alltrees": "histgb,extratrees,lightgbm,xgboost,catboost",
        "forest": "randomforest,extratrees,histgb",
        "boost": "histgb,lightgbm,xgboost,catboost",
        "rf": "randomforest",
    }
    buckets = [
        (1_000, 2_500, 4_000),
        (1_250, 2_500, 5_000),
        (1_000, 3_000, 5_000),
        (2_000, 4_000, 8_000),
    ]
    score_fields = ("probability", "expected_value", "selection_score", "predicted_return", "trade_score")
    score_directions = ("high", "low")
    target_trades_per_day_values = (6, 8, 10, 12, 16, 20)
    max_open_positions_values = (1, 2, 3, 4, 5)

    for target_trades_per_day in target_trades_per_day_values:
        for max_open_positions in max_open_positions_values:
            for base, mid, high in buckets:
                for score_field in score_fields:
                    for score_direction in score_directions:
                        experiments.append(
                            IbkrExperiment(
                                name=(
                                    "lv10k_"
                                    f"tpd{target_trades_per_day}_pos{max_open_positions}_"
                                    f"{_bucket_slug(base, mid, high)}_{score_field}_{score_direction}"
                                ),
                                category="live_valid_strict_10k",
                                description=(
                                    "Live-valid strict $10K BTC spot matrix member: market "
                                    "entries, IBKR 1s label/execution replay, online cash-aware "
                                    "selection, dynamic horizons, and dynamic exit checkpoints."
                                ),
                                overrides=_live_valid_strict_10k_overrides(
                                    {
                                        "--target-trades-per-day": target_trades_per_day,
                                        "--max-open-positions": max_open_positions,
                                        "--notional": high,
                                        "--sizing-base-notional": base,
                                        "--sizing-mid-notional": mid,
                                        "--sizing-high-notional": high,
                                        "--sizing-score-field": score_field,
                                        "--sizing-score-direction": score_direction,
                                    }
                                ),
                            )
                        )

    for model_slug, models in model_sets.items():
        for calibration_method in ("sigmoid", "isotonic"):
            experiments.append(
                IbkrExperiment(
                    name=f"lv10k_model_{model_slug}_{calibration_method}",
                    category="live_valid_strict_10k_models",
                    description=(
                        "Live-valid strict $10K model/calibration ablation around the "
                        "current best inverse-score bucket recipe."
                    ),
                    overrides=_live_valid_strict_10k_overrides(
                        {
                            "--models": models,
                            "--calibration-method": calibration_method,
                        }
                    ),
                )
            )

    for latency_seconds in (0, 5, 30, 60):
        experiments.append(
            IbkrExperiment(
                name=f"lv10k_latency_{latency_seconds}s",
                category="live_valid_strict_10k_features",
                description=(
                    "Causal external-feature latency stress for the live-valid strict "
                    "$10K BTC spot recipe."
                ),
                overrides=_live_valid_strict_10k_overrides(
                    {"--external-feature-latency-seconds": latency_seconds}
                ),
            )
        )

    for max_holding_seconds in (30, 300, 900, 3_600, 10_800, 21_600, 43_200, 86_400):
        experiments.append(
            IbkrExperiment(
                name=f"lv10k_horizon_max{max_holding_seconds}s",
                category="live_valid_strict_10k_horizons",
                description=(
                    "Dynamic-horizon ablation spanning 30 seconds through 24 hours "
                    "while preserving one-second label/execution replay."
                ),
                overrides=_live_valid_strict_10k_overrides(
                    {
                        "--max-holding-seconds": max_holding_seconds,
                        "--adaptive-horizon-max-seconds": max_holding_seconds,
                    }
                ),
            )
        )

    for families, suffix in [
        ("mean_reversion_exhaustion", "mean_reversion_exhaustion"),
        ("momentum_continuation", "momentum_continuation"),
        ("liquidity_vacuum_breakout", "liquidity_vacuum_breakout"),
        ("chop_no_trade", "chop_no_trade"),
        ("dense_pullback_reclaim", "legacy_pullback_only"),
        ("dense_breakout_momentum", "legacy_breakout_only"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"lv10k_specialist_{suffix}",
                category="live_valid_strict_10k_specialists",
                description=(
                    "Explicit setup-family specialist pass to test whether the negative-score "
                    "cohort is a real family-specific edge instead of a global score inversion."
                ),
                overrides=_live_valid_strict_10k_overrides(
                    {"--selection-setup-families": families}
                ),
            )
        )

    for slug, overrides in {
        "1m_features": {},
        "1s_micro_features": {
            "--primary-bars-jsonl": str(DEFAULT_1S_BARS),
            "--interval": "1s",
            "--min-history-bars": 3_600,
            "--candidate-rolling-window-bars": 3_600,
            "--dense-stride-bars": 60,
        },
        "binance_derivatives_context": {
            "--binance-um-derivatives-metrics": True,
            "--binance-um-taker-flow": True,
            "--binance-um-basis": True,
            "--external-feature-latency-seconds": 30,
        },
    }.items():
        experiments.append(
            IbkrExperiment(
                name=f"lv10k_data_{slug}",
                category="live_valid_strict_10k_data",
                description=(
                    "Strict live-valid data variant for 1m features, 1s microstructure "
                    "features, and causal Binance derivatives context."
                ),
                overrides=_live_valid_strict_10k_overrides(overrides),
            )
        )

    return experiments


def _strict_anchor_g210_overrides(
    overrides: Mapping[str, ArgValue] | None = None,
) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--notional": 1_275,
        "--max-open-positions": 8,
        "--target-trades-per-day": 20,
        "--dense-stride-bars": 10,
        "--minimum-gross-profit-bps": 210,
        "--minimum-gross-stop-bps": 100,
        "--train-size": 180,
        "--calibration-size": 60,
        "--test-size": 90,
    }
    if overrides:
        values.update(overrides)
    return _with_research_gate(values)


def _fast_hold_overrides(overrides: Mapping[str, ArgValue] | None = None) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--context-symbols": "",
        "--context-bars-jsonl": "",
        "--primary-bars-jsonl": str(DEFAULT_1S_BARS),
        "--label-bars-jsonl": str(DEFAULT_1S_BARS),
        "--execution-bars-jsonl": str(DEFAULT_1S_BARS),
        "--interval": "1s",
        "--start": "2026-05-03T19:20:15+00:00",
        "--end": "2026-05-04T01:20:14+00:00",
        "--minimum-data-coverage": 0.90,
        "--starting-equity": 10_000,
        "--paper-max-notional": 10_000,
        "--notional": 10_000,
        "--candidate-mode": "dense",
        "--side-mode": "long",
        "--dense-stride-bars": 60,
        "--min-history-bars": 3_600,
        "--candidate-rolling-window-bars": 3_600,
        "--max-holding-seconds": 21_600,
        "--adaptive-horizon": True,
        "--min-holding-seconds": 1,
        "--adaptive-horizon-granularity-seconds": 1,
        "--adaptive-horizon-max-seconds": 21_600,
        "--adaptive-horizon-target-move-bps": 0,
        "--net-profit-target": 0.001,
        "--net-stop-loss": 0.001,
        "--minimum-gross-profit-bps": 100,
        "--minimum-gross-stop-bps": 80,
        "--minimum-probability": 0,
        "--minimum-expected-value": -0.02,
        "--target-frequency-mode": "online",
        "--target-trades-per-day": 24,
        "--respect-open-positions": True,
        "--capacity-release-mode": "actual",
        "--adaptive-selection-score-floor": True,
        "--selection-score": "capital_efficiency",
        "--optimize-metric": "net_pnl",
        "--models": "histgb,extratrees",
        "--stacker": "weighted",
        "--calibration-method": "sigmoid",
        "--empirical-payoff-ev": True,
        "--hpo": False,
        "--train-size": 96,
        "--calibration-size": 48,
        "--test-size": 48,
        "--embargo-hours": 0,
        "--max-open-positions": 4,
        "--max-signals-per-timestamp": 1,
        "--consecutive-loss-limit": 0,
        "--sizing-mode": "score_bucket",
        "--sizing-score-field": "probability",
        "--sizing-base-notional": 2_500,
        "--sizing-mid-notional": 5_000,
        "--sizing-high-notional": 10_000,
        "--sizing-mid-score": 0.45,
        "--sizing-high-score": 0.85,
    }
    if overrides:
        values.update(overrides)
    return _with_research_gate(values)


def _strict_1s_label_replay_overrides(
    overrides: Mapping[str, ArgValue] | None = None,
) -> tuple[tuple[str, ArgValue], ...]:
    values: dict[str, ArgValue] = {
        "--context-symbols": "",
        "--context-bars-jsonl": "",
        "--primary-bars-jsonl": str(DEFAULT_1M_FROM_1S_BARS),
        "--label-bars-jsonl": str(DEFAULT_1S_BARS),
        "--execution-bars-jsonl": str(DEFAULT_1S_BARS),
        "--interval": "1m",
        "--start": "2026-05-03T19:21:00+00:00",
        "--end": "2026-05-04T01:20:14+00:00",
        "--minimum-data-coverage": 0.90,
        "--starting-equity": 10_000,
        "--paper-max-notional": 10_000,
        "--notional": 2_000,
        "--candidate-mode": "dense",
        "--side-mode": "long",
        "--dense-stride-bars": 1,
        "--min-history-bars": 120,
        "--candidate-rolling-window-bars": 180,
        "--max-holding-seconds": 21_600,
        "--adaptive-horizon": True,
        "--min-holding-seconds": 1,
        "--adaptive-horizon-granularity-seconds": 1,
        "--adaptive-horizon-max-seconds": 21_600,
        "--adaptive-horizon-target-move-bps": 80,
        "--net-profit-target": 0.001,
        "--net-stop-loss": 0.001,
        "--minimum-gross-profit-bps": 100,
        "--minimum-gross-stop-bps": 80,
        "--minimum-probability": 0,
        "--minimum-expected-value": -0.02,
        "--target-frequency-mode": "online",
        "--target-trades-per-day": 18,
        "--respect-open-positions": True,
        "--capacity-release-mode": "planned",
        "--adaptive-selection-score-floor": True,
        "--selection-score": "expected_value",
        "--optimize-metric": "net_pnl",
        "--models": "histgb,gboost,extratrees,lightgbm",
        "--stacker": "weighted",
        "--calibration-method": "sigmoid",
        "--empirical-payoff-ev": True,
        "--hpo": True,
        "--hpo-profile": "capacity",
        "--hpo-trials": 8,
        "--train-size": 120,
        "--calibration-size": 60,
        "--test-size": 60,
        "--embargo-hours": 0,
        "--max-open-positions": 5,
        "--max-signals-per-timestamp": 1,
        "--consecutive-loss-limit": 0,
        "--sizing-mode": "fixed",
        "--entry-order-model": "market",
    }
    if overrides:
        values.update(overrides)
    return _with_research_gate(values)


def build_ibkr_experiment_suite() -> list[IbkrExperiment]:
    experiments: list[IbkrExperiment] = [
        IbkrExperiment(
            name="champion_repro_extratrees_return_first_1250x8",
            category="anchor",
            description=(
                "Reproduce the current best local IBKR spot setup: ExtraTrees, return-first "
                "ranking, $1,250 slots, 8 max open positions, and $10K aggregate cap."
            ),
            overrides=_with_research_gate(),
        ),
        IbkrExperiment(
            name="phase17_no_pm_bucket_repro",
            category="phase17_revival",
            description=(
                "Reproduce the stronger no-PM phase17 long-only BTC spot recipe: online "
                "selection, adaptive score floor, 6h 100/80 geometry, five-model weighted "
                "HPO ensemble, and 5/10/15K probability bucket sizing."
            ),
            overrides=_phase17_overrides(),
        ),
        IbkrExperiment(
            name="phase17_no_pm_bucket_mid020_repro",
            category="phase17_revival",
            description=(
                "Re-run phase17 with the probability bucket mid threshold recalibrated "
                "for the compressed current probability distribution."
            ),
            overrides=_phase17_overrides({"--sizing-mid-score": 0.20}),
        ),
        IbkrExperiment(
            name="phase17_fixed_10k_slots3_quality",
            category="phase17_revival",
            description=(
                "Use the phase17 selector with fixed $10K notional and three concurrent "
                "spot slots to favor quality and Sharpe over raw turnover."
            ),
            overrides=_phase17_overrides(
                {
                    "--sizing-mode": "fixed",
                    "--notional": 10_000,
                    "--paper-max-notional": 30_000,
                    "--max-open-positions": 3,
                }
            ),
        ),
        IbkrExperiment(
            name="phase17_scaled_10k_bucket_2500_5000_10000",
            category="phase17_revival",
            description=(
                "Scale the phase17 bucket strategy down to a strict $10K aggregate spend cap."
            ),
            overrides=_phase17_overrides(
                {
                    "--starting-equity": 10_000,
                    "--paper-max-notional": 10_000,
                    "--notional": 10_000,
                    "--sizing-base-notional": 2_500,
                    "--sizing-mid-notional": 5_000,
                    "--sizing-high-notional": 10_000,
                }
            ),
        ),
        IbkrExperiment(
            name="phase17_adaptive_horizon_1s_to_6h",
            category="phase17_revival",
            description=(
                "Keep phase17 selection and bucket sizing, but let each setup choose a "
                "volatility-scaled horizon from 1 second up to 6 hours."
            ),
            overrides=_phase17_overrides(
                {
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--adaptive-horizon-target-move-bps": 50,
                }
            ),
        ),
        IbkrExperiment(
            name="phase17_label_replay_1s_micro_window",
            category="phase17_revival",
            description=(
                "Exercise the new separate label/execution stream: 1-minute features can "
                "train and replay exits on verified IBKR 1-second BTC bars."
            ),
            overrides=_phase17_overrides(
                {
                    "--context-symbols": "",
                    "--context-bars-jsonl": "",
                    "--primary-bars-jsonl": str(DEFAULT_1S_BARS),
                    "--label-bars-jsonl": str(DEFAULT_1S_BARS),
                    "--execution-bars-jsonl": str(DEFAULT_1S_BARS),
                    "--interval": "1s",
                    "--start": "2026-05-03T19:20:15+00:00",
                    "--end": "2026-05-04T01:20:14+00:00",
                    "--minimum-data-coverage": 0.90,
                    "--starting-equity": 10_000,
                    "--paper-max-notional": 10_000,
                    "--notional": 10_000,
                    "--dense-stride-bars": 300,
                    "--min-history-bars": 3_600,
                    "--max-holding-seconds": 21_600,
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--train-size": 24,
                    "--calibration-size": 12,
                    "--test-size": 12,
                    "--hpo": False,
                    "--models": "histgb,extratrees",
                    "--sizing-base-notional": 2_500,
                    "--sizing-mid-notional": 5_000,
                    "--sizing-high-notional": 10_000,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_phase17_market_entry_bucket",
            category="live_aligned",
            description=(
                "Replay the phase17 no-PM bucket strategy with market-entry fills so "
                "the backtest entry model matches broker trade-run market buy cash orders."
            ),
            overrides=_phase17_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,lightgbm,xgboost,catboost,extratrees",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_gboost_hpo",
            category="live_aligned",
            description=(
                "Strict $10K-cap live-aligned market-entry HPO pass with the new sklearn "
                "gradient boosting candidate included in the weighted ensemble."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--sizing-mode": "score_bucket",
                    "--sizing-base-notional": 1_250,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 5_000,
                    "--sizing-mid-score": 0.30,
                    "--sizing-high-score": 0.75,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_fast_hold_1s_market_entry_hpo",
            category="live_aligned",
            description=(
                "Verified 1-second BTC bars with market-entry replay, 1-second minimum "
                "hold, adaptive exits, and capacity-aware HPO."
            ),
            overrides=_fast_hold_overrides(
                {
                    "--entry-order-model": "market",
                    "--min-holding-seconds": 1,
                    "--models": "histgb,gboost,extratrees,lightgbm",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 10,
                    "--target-trades-per-day": 36,
                    "--optimize-metric": "net_pnl",
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_tpd24_buckets",
            category="live_aligned",
            description=(
                "Push the strict $10K market-entry selector toward higher turnover with "
                "smaller buckets, while keeping capacity-aware HPO."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 10,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--sizing-mode": "score_bucket",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.25,
                    "--sizing-high-score": 0.70,
                    "--max-open-positions": 10,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_fixed2500_slots4",
            category="live_aligned",
            description=(
                "Compare simple fixed $2.5K sizing with four slots against the probability "
                "bucket strategy under live-aligned market entries."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 10,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_no_pullback_breakout",
            category="live_aligned",
            description=(
                "Filter weak setup families from the strict $10K live-aligned run to see "
                "whether PnL/day improves without sacrificing the 10+ trade/day objective."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 10,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--exclude-setup-families": "dense_pullback_reclaim,dense_breakout_momentum",
                    "--sizing-mode": "score_bucket",
                    "--sizing-base-notional": 1_250,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 5_000,
                    "--sizing-mid-score": 0.30,
                    "--sizing-high-score": 0.75,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_return_first",
            category="live_aligned",
            description=(
                "Use return-first ranking with the HPO objective now aligned to predicted "
                "returns/downside instead of silently falling back to raw probability."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 10,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--selection-score": "return_first",
                    "--sizing-mode": "score_bucket",
                    "--sizing-base-notional": 1_250,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 5_000,
                    "--sizing-mid-score": 0.30,
                    "--sizing-high-score": 0.75,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_fixed2500_forest_mix",
            category="live_aligned",
            description=(
                "Start from the best strict-$10K fixed $2.5K slot policy, but replace "
                "the booster-heavy ensemble with random forest, ExtraTrees, HistGB, and "
                "sklearn gradient boosting to test a lower-latency tree bagging mix."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,randomforest,extratrees",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_fixed2000_slots5",
            category="live_aligned",
            description=(
                "Use five $2K concurrent spot slots to push turnover above 10 trades/day "
                "without dropping to tiny fee-inefficient tickets."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_fixed2000_slots5_tpd18",
            category="live_aligned_dynamic",
            description=(
                "Retune the strict $10K $2K x 5 champion toward slightly fewer, higher-quality "
                "signals while preserving market-entry live alignment."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 18,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_fixed2000_slots5_tpd30",
            category="live_aligned_dynamic",
            description=(
                "Push the strict $10K $2K x 5 policy to a higher turnover target to test "
                "whether marginal signals still clear the 0.18% per-side fee hurdle."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 30,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_fixed1800_slots5_tpd30",
            category="live_aligned_dynamic",
            description=(
                "Use five smaller $1.8K slots to increase capital recycling while staying "
                "above the configured fee-efficient minimum notional."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 30,
                    "--sizing-mode": "fixed",
                    "--notional": 1_800,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_fixed2250_slots4_tpd24",
            category="live_aligned_dynamic",
            description=(
                "Try a four-slot $2.25K middle ground between the $2K turnover winner and "
                "the lower-frequency $2.5K/$3.33K quality variants."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--sizing-mode": "fixed",
                    "--notional": 2_250,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_adaptive_1m_move50_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "Enable volatility-scaled horizons on the full strict-$10K 1-minute window. "
                "This is dynamic-hold research, but true one-second exits require the separate "
                "1-second label replay experiments."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--adaptive-horizon-target-move-bps": 50,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_adaptive_1m_move90_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "Use a wider adaptive-horizon target move on the full 1-minute strict-$10K "
                "window so low-volatility signals can hold longer while fast signals can exit sooner."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--adaptive-horizon-target-move-bps": 90,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_move60_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "Use 1-minute candidates but true IBKR 1-second labels/execution replay, "
                "allowing dynamic holds from one second up to six hours."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--adaptive-horizon-target-move-bps": 60,
                    "--target-trades-per-day": 18,
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_move100_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "A wider true 1-second replay variant that lets quiet signals hold longer "
                "before the vertical barrier while still allowing one-second exits."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--adaptive-horizon-target-move-bps": 100,
                    "--target-trades-per-day": 14,
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_fixed1666_slots6_tpd36",
            category="live_aligned_dynamic",
            description=(
                "Use six smaller $1.67K slots to test whether open-position capacity, not "
                "target frequency, is the binding constraint on trade count."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 36,
                    "--sizing-mode": "fixed",
                    "--notional": 1_666,
                    "--max-open-positions": 6,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_adaptive_1m_move150_g200_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "Retry adaptive horizons on the full 1-minute window, but require wider "
                "gross target/stop geometry so short holds are not pure commission churn."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 18,
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--adaptive-horizon-target-move-bps": 150,
                    "--minimum-gross-profit-bps": 200,
                    "--minimum-gross-stop-bps": 120,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_adaptive_1m_move220_g300_fixed2000",
            category="live_aligned_dynamic",
            description=(
                "A wider adaptive-horizon/noise-filter test on the full window, requiring "
                "roughly 3x the round-trip fee hurdle before labels count as wins."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 14,
                    "--adaptive-horizon": True,
                    "--min-holding-seconds": 1,
                    "--adaptive-horizon-granularity-seconds": 1,
                    "--adaptive-horizon-max-seconds": 21_600,
                    "--adaptive-horizon-target-move-bps": 220,
                    "--minimum-gross-profit-bps": 300,
                    "--minimum-gross-stop-bps": 160,
                    "--sizing-mode": "fixed",
                    "--notional": 2_000,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_move180_g220_fixed3333",
            category="live_aligned_dynamic",
            description=(
                "True 1-second label/execution replay with wider fee-aware geometry and "
                "larger $3.33K slots to reduce minimum-commission drag."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--adaptive-horizon-target-move-bps": 180,
                    "--minimum-gross-profit-bps": 220,
                    "--minimum-gross-stop-bps": 140,
                    "--target-trades-per-day": 8,
                    "--notional": 3_333,
                    "--max-open-positions": 3,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_move260_g320_fixed3333",
            category="live_aligned_dynamic",
            description=(
                "True 1-second replay with a much wider target move, testing whether only "
                "larger intraday BTC moves can survive spot crypto fees."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--adaptive-horizon-target-move-bps": 260,
                    "--minimum-gross-profit-bps": 320,
                    "--minimum-gross-stop-bps": 180,
                    "--target-trades-per-day": 6,
                    "--notional": 3_333,
                    "--max-open-positions": 3,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_fixed3333_slots3_quality",
            category="live_aligned",
            description=(
                "Use three larger $3.33K slots and a lower turnover target to see whether "
                "quality beats the higher-frequency $2K/$2.5K variants after market-entry fees."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 14,
                    "--sizing-mode": "fixed",
                    "--notional": 3_333,
                    "--max-open-positions": 3,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2000_3333",
            category="live_aligned",
            description=(
                "Use dynamic score-bucket sizing under the strict $10K cap, but invert the "
                "expected-value sizing score because the live-aligned ledger showed realized "
                "edge concentrated in the lower-score BTC spot cohort."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 14,
                    "--notional": 3_333,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_000,
                    "--sizing-high-notional": 3_333,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1500_2500_3333",
            category="live_aligned",
            description=(
                "A less aggressive inverse-EV sizing schedule that still avoids fixed "
                "tickets, using a larger base bucket to keep every trade safely above "
                "the $1.75 minimum commission break-even notional."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 14,
                    "--notional": 3_333,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_500,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 3_333,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2000_5000",
            category="live_aligned",
            description=(
                "Concentrate capital more aggressively into the inverted low-score cohort "
                "by allowing two $5K BTC spot slots while using smaller tickets for weaker "
                "realized buckets."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 10,
                    "--notional": 5_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_000,
                    "--sizing-high-notional": 5_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2500_4000_tpd20",
            category="live_aligned",
            description=(
                "Higher-turnover inverse-EV sizing that can still recycle capital into "
                "more than three trades/day without pinning every signal to the same "
                "notional."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2500_4000_tpd14",
            category="live_aligned",
            description=(
                "Lower-frequency variant of the best inverse-EV bucket schedule to test "
                "whether reducing marginal middle-score entries improves strict-$10K PnL."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 14,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2500_4000_tpd24",
            category="live_aligned",
            description=(
                "Higher-frequency variant of the best inverse-EV bucket schedule, checking "
                "whether the corrected bucket cap can support more turnover without fee churn."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_3000_4000_tpd20",
            category="live_aligned",
            description=(
                "Raise only the inverse-EV middle bucket while leaving the base/high "
                "schedule intact, testing whether the mid cohort has enough edge for "
                "larger dynamic tickets."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2500_4000_mid735",
            category="live_aligned",
            description=(
                "Tighten the inverse-EV middle bucket threshold to remove weaker mid-score "
                "signals while preserving the low-score high bucket."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.00735,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_bucket_1000_2500_4000_mid745",
            category="live_aligned",
            description=(
                "Use an even tighter middle-bucket threshold to see if the mid cohort's "
                "largest loss can be filtered without losing the stronger low-score entries."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 2_500,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.00745,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_adaptive_direction",
            category="live_aligned_deep",
            description=(
                "Start from the best strict-$10K inverse-EV bucket schedule, but let each "
                "walk-forward fold causally flip the selection ranking if low scores had "
                "better calibration-slice returns."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--adaptive-selection-score-direction": True,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_actual_release",
            category="live_aligned_deep",
            description=(
                "Use actual target/stop exit times for capacity release instead of reserving "
                "slots until the full planned barrier, matching a live bracket monitor more closely."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "actual",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_hpo28",
            category="live_aligned_deep",
            description=(
                "Increase capacity-HPO depth on the best strict-$10K inverse-EV schedule "
                "to test whether the model hyperparameters, not strategy geometry, are now binding."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 28,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_ceiling0",
            category="live_aligned_deep",
            description=(
                "Filter the overconfident positive-EV selection cohort by capping raw expected-value "
                "selection at zero while keeping inverse-EV bucket sizing."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--selection-score-ceiling": 0.0,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_ceiling0_tpd24",
            category="live_aligned_deep",
            description=(
                "Use the zero EV ceiling but ask for more daily candidates, testing whether filtered "
                "replacement signals can lift PnL without falling below the trade-frequency goal."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--selection-score-ceiling": 0.0,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_ceiling_neg0072",
            category="live_aligned_deep",
            description=(
                "Require only the stronger inverse-EV cohort by capping selection score at the "
                "middle bucket boundary."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--selection-score-ceiling": -0.0072,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_ceiling_neg00735",
            category="live_aligned_deep",
            description=(
                "Retest the stronger inverse-EV cohort with a slightly tighter ceiling found near "
                "the previous mid-bucket boundary."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--selection-score-ceiling": -0.00735,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_catboost",
            category="live_aligned_deep",
            description=(
                "Reintroduce CatBoost into the best strict-$10K inverse-EV ensemble, testing "
                "whether a richer tree stack improves ranking enough to justify extra latency."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost,catboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 20,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_forest_fast",
            category="live_aligned_deep",
            description=(
                "Use a lower-latency forest/HGB ensemble on the best inverse-EV strategy "
                "to see if simpler models retain the edge while scoring faster live."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,randomforest,extratrees",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 20,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_best_stacker",
            category="live_aligned_deep",
            description=(
                "Use best-base-model selection instead of weighted blending on the current "
                "best inverse-EV strategy."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "best",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 20,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_spacing30",
            category="live_aligned_deep",
            description=(
                "Add a 30-minute same-group spacing constraint to reduce repeated correlated "
                "entries while retaining the inverse-EV bucket sizing."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--min-signal-spacing-hours": 0.5,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_spacing15",
            category="live_aligned_deep",
            description=(
                "Use a gentler 15-minute same-group spacing constraint to reduce cash pileups "
                "while preserving more of the incumbent inverse-EV entries."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--min-signal-spacing-hours": 0.25,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_spacing45",
            category="live_aligned_deep",
            description=(
                "Increase same-group spacing to 45 minutes to test whether capital freed from "
                "overlapping clusters is more valuable than the extra skipped entries."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--min-signal-spacing-hours": 0.75,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_stride10",
            category="live_aligned_deep",
            description=(
                "Generate denser 10-minute BTC spot candidates with the same inverse-EV sizing, "
                "testing whether more entry timestamps improve the online selector."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--dense-stride-bars": 10,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_stride10_spacing15",
            category="live_aligned_deep",
            description=(
                "Combine denser 10-minute candidates with a 15-minute group spacing constraint "
                "to trade more often without concentrating all cash into one signal cluster."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 24,
                    "--dense-stride-bars": 10,
                    "--min-signal-spacing-hours": 0.25,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_3333_tpd20",
            category="live_aligned_deep",
            description=(
                "Cap the high inverse-EV bucket at roughly one third of the strict $10K cash limit, "
                "keeping the stronger $3K middle bucket while reducing insufficient-cash skips."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 3_333,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 3_333,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_3500_tpd20",
            category="live_aligned_deep",
            description=(
                "Use a slightly higher $3.5K high bucket to balance the incumbent's PnL scaling "
                "against the strict cash-cap rejection rate."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 3_500,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 3_500,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1250_3000_3500_tpd20",
            category="live_aligned_deep",
            description=(
                "Raise the base bucket modestly while keeping the $3K/$3.5K inverse-EV tiers, "
                "testing whether the smaller positive-EV cohort can pay its fixed costs."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 3_500,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_250,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 3_500,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_inverse_ev_1000_3000_4000_maxpos3",
            category="live_aligned_deep",
            description=(
                "Keep incumbent bucket sizes but restrict the online selector to three planned "
                "open positions so selected signals better match the strict $10K cash limit."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 18,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 3,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_inverse_ev_1000_3000_4000",
            category="live_aligned_deep",
            description=(
                "Replay the inverse-EV dynamic bucket strategy on the verified IBKR one-second "
                "sample with one-second minimum holding and second-level execution bars."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--models": "histgb,gboost,extratrees,lightgbm",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 12,
                    "--target-trades-per-day": 20,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_1s_replay_inverse_ev_1000_3000_4000_5s",
            category="live_aligned_deep",
            description=(
                "Same true one-second execution replay, but enforce a five-second minimum hold "
                "to test whether a tiny anti-noise delay improves spot-fee survival."
            ),
            overrides=_strict_1s_label_replay_overrides(
                {
                    "--models": "histgb,gboost,extratrees,lightgbm",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 12,
                    "--target-trades-per-day": 20,
                    "--min-holding-seconds": 5,
                    "--notional": 4_000,
                    "--sizing-mode": "score_bucket",
                    "--sizing-score-field": "expected_value",
                    "--sizing-score-direction": "low",
                    "--sizing-base-notional": 1_000,
                    "--sizing-mid-notional": 3_000,
                    "--sizing-high-notional": 4_000,
                    "--sizing-mid-score": 0.0072,
                    "--sizing-high-score": 0.0108,
                    "--max-open-positions": 5,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_best_stacker_fixed2500",
            category="live_aligned",
            description=(
                "Let the walk-forward calibration slice choose one best base model instead "
                "of blending models, using the fixed $2.5K live-aligned slot policy."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,randomforest,extratrees,lightgbm,xgboost",
                    "--stacker": "best",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_logistic_stacker_fixed2500",
            category="live_aligned",
            description=(
                "Use a logistic stacker over the same tuned base models to test whether "
                "a learned meta-combination improves fixed-slot live-aligned PnL."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,randomforest,extratrees,lightgbm,xgboost",
                    "--stacker": "logistic",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_candidate_thresholds_fixed2500",
            category="live_aligned",
            description=(
                "Add causal candidate-type calibration to the fixed $2.5K policy so weak "
                "setup families can abstain instead of consuming a slot."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--candidate-type-thresholds": True,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_adaptive_direction_fixed2500",
            category="live_aligned",
            description=(
                "Allow each fold to flip score direction when calibration shows lower "
                "scores had better realized returns, guarding against inverted model ranks."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--adaptive-selection-score-direction": True,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
        IbkrExperiment(
            name="live_aligned_strict_10k_market_entry_dynamic_exit_fixed2500",
            category="live_aligned",
            description=(
                "Overlay causal 15/30/60/120/240 minute early exits on the fixed $2.5K "
                "policy to cut weak trades before the six-hour vertical barrier."
            ),
            overrides=_phase17_light_overrides(
                {
                    "--entry-order-model": "market",
                    "--capacity-release-mode": "planned",
                    "--models": "histgb,gboost,extratrees,lightgbm,xgboost",
                    "--stacker": "weighted",
                    "--hpo": True,
                    "--hpo-profile": "capacity",
                    "--hpo-trials": 14,
                    "--optimize-metric": "net_pnl",
                    "--target-trades-per-day": 20,
                    "--dynamic-exit-overlay": True,
                    "--dynamic-exit-adverse-bps": 35,
                    "--dynamic-exit-giveback-bps": 45,
                    "--dynamic-exit-min-profit-bps": 12,
                    "--dynamic-exit-weak-probability": 0.62,
                    "--sizing-mode": "fixed",
                    "--notional": 2_500,
                    "--max-open-positions": 4,
                }
            ),
        ),
    ]

    for min_hold in [1, 5]:
        for stride, target_trades_per_day, target_move_bps in [
            (60, 24, 80),
            (30, 48, 60),
            (15, 72, 50),
        ]:
            experiments.append(
                IbkrExperiment(
                    name=(
                        f"fast_hold_{min_hold}s_stride{stride}_tpd"
                        f"{target_trades_per_day}_move{target_move_bps}"
                    ),
                    category="fast_hold_sweep",
                    description=(
                        "Use verified IBKR 1-second spot bars for labels and execution, "
                        f"enforce a {min_hold}s minimum hold after fill, and let adaptive "
                        "barriers choose exits from seconds up to six hours."
                    ),
                    overrides=_fast_hold_overrides(
                        {
                            "--min-holding-seconds": min_hold,
                            "--dense-stride-bars": stride,
                            "--target-trades-per-day": target_trades_per_day,
                            "--adaptive-horizon-target-move-bps": target_move_bps,
                            "--minimum-gross-profit-bps": max(60, target_move_bps),
                            "--minimum-gross-stop-bps": max(60, int(target_move_bps * 0.9)),
                        }
                    ),
                )
            )

    for min_hold, name, gross, stop, move, max_seconds in [
        (1, "phase17_geometry", 100, 80, 80, 21_600),
        (5, "phase17_geometry", 100, 80, 80, 21_600),
        (1, "tight_fee_edge", 70, 80, 55, 7_200),
        (5, "tight_fee_edge", 70, 80, 55, 7_200),
        (1, "wide_noise_filter", 140, 100, 110, 21_600),
        (5, "wide_noise_filter", 140, 100, 110, 21_600),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_{name}",
                category="fast_hold_sweep",
                description=(
                    "Compare target/stop geometry under the corrected 0.18% per-side "
                    "commission and minimum hold enforcement."
                ),
                overrides=_fast_hold_overrides(
                    {
                        "--min-holding-seconds": min_hold,
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                        "--adaptive-horizon-target-move-bps": move,
                        "--adaptive-horizon-max-seconds": max_seconds,
                        "--target-trades-per-day": 36,
                    }
                ),
            )
        )

    for min_hold, models, stacker, suffix in [
        (1, "extratrees", "average", "extratrees"),
        (5, "extratrees", "average", "extratrees"),
        (1, "lightgbm,extratrees", "weighted", "lgbm_extratrees"),
        (5, "lightgbm,extratrees", "weighted", "lgbm_extratrees"),
        (1, "histgb,lightgbm,xgboost,extratrees", "weighted", "tree_ensemble"),
        (5, "histgb,lightgbm,xgboost,extratrees", "weighted", "tree_ensemble"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_model_{suffix}",
                category="fast_hold_sweep",
                description=(
                    "Hold the 1-second data and adaptive exit logic constant while swapping "
                    "model families for faster, more aggressive signal selection."
                ),
                overrides=_fast_hold_overrides(
                    {
                        "--min-holding-seconds": min_hold,
                        "--models": models,
                        "--stacker": stacker,
                        "--target-trades-per-day": 36,
                    }
                ),
            )
        )

    for min_hold, adverse, giveback, min_profit in [
        (1, 10, 16, 4),
        (5, 10, 16, 4),
        (1, 18, 28, 8),
        (5, 18, 28, 8),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_dynamic_exit_a{adverse}_g{giveback}",
                category="fast_hold_sweep",
                description=(
                    "Add a causal second-level dynamic exit overlay; the overlay is also "
                    "blocked from exiting before the configured minimum hold."
                ),
                overrides=_fast_hold_overrides(
                    {
                        "--min-holding-seconds": min_hold,
                        "--dynamic-exit-overlay": True,
                        "--dynamic-exit-checkpoints-minutes": "",
                        "--dynamic-exit-checkpoints-seconds": "1,5,15,30,60,120,300,900",
                        "--dynamic-exit-adverse-bps": adverse,
                        "--dynamic-exit-giveback-bps": giveback,
                        "--dynamic-exit-min-profit-bps": min_profit,
                        "--target-trades-per-day": 36,
                    }
                ),
            )
        )

    for min_hold, mode, production_gate in [
        (1, "adaptive_cost_aware", False),
        (5, "adaptive_cost_aware", False),
        (1, "adaptive_cost_aware_production_gate", True),
        (5, "adaptive_cost_aware_production_gate", True),
        (1, "fixed_2h", False),
        (5, "fixed_2h", False),
        (1, "fixed_2h_production_gate", True),
        (5, "fixed_2h_production_gate", True),
    ]:
        adaptive = "adaptive" in mode
        values: dict[str, ArgValue] = {
            "--min-holding-seconds": min_hold,
            "--adaptive-horizon": adaptive,
            "--adaptive-horizon-target-move-bps": 0,
            "--target-trades-per-day": 24,
            "--dense-stride-bars": 60,
            "--minimum-gross-profit-bps": 100,
            "--minimum-gross-stop-bps": 80,
            "--models": "histgb,extratrees,lightgbm",
            "--stacker": "weighted",
        }
        if not adaptive:
            values.update(
                {
                    "--adaptive-horizon-max-seconds": 0,
                    "--max-holding-seconds": 7_200,
                    "--train-size": 24,
                    "--calibration-size": 12,
                    "--test-size": 12,
                }
            )
        if production_gate:
            values.update(
                {
                    "--research-gate": False,
                    "--allow-negative-ev-frequency-probe": False,
                    "--minimum-probability": 0.52,
                    "--minimum-expected-value": 0.0,
                    "--adaptive-selection-score-floor": False,
                    "--capacity-release-mode": "planned",
                }
            )
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_{mode}",
                category="fast_hold_sweep",
                description=(
                    "Guard against fee churn by using the CLI's cost-aware adaptive target "
                    "move or a full 6h vertical barrier, with optional production gates."
                ),
                overrides=_fast_hold_overrides(values),
            )
        )

    for min_hold, gross, stop, production_gate in [
        (1, 40, 50, False),
        (5, 40, 50, False),
        (1, 50, 60, False),
        (5, 50, 60, False),
        (1, 40, 50, True),
        (5, 40, 50, True),
        (1, 50, 60, True),
        (5, 50, 60, True),
    ]:
        suffix = "production_gate" if production_gate else "research"
        values = {
            "--min-holding-seconds": min_hold,
            "--adaptive-horizon": False,
            "--adaptive-horizon-max-seconds": 0,
            "--max-holding-seconds": 7_200,
            "--net-profit-target": 0.0001,
            "--minimum-gross-profit-bps": gross,
            "--minimum-gross-stop-bps": stop,
            "--target-trades-per-day": 24,
            "--dense-stride-bars": 60,
            "--models": "histgb,extratrees,lightgbm",
            "--stacker": "weighted",
            "--train-size": 96,
            "--calibration-size": 48,
            "--test-size": 48,
        }
        if production_gate:
            values.update(
                {
                    "--research-gate": False,
                    "--allow-negative-ev-frequency-probe": False,
                    "--minimum-probability": 0.52,
                    "--minimum-expected-value": 0.0,
                    "--adaptive-selection-score-floor": False,
                    "--capacity-release-mode": "planned",
                }
            )
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_fixed_2h_g{gross}_s{stop}_{suffix}",
                category="fast_hold_sweep",
                description=(
                    "Fee-edge 1-second execution experiment with lower gross targets that "
                    "actually produce positive labels on the verified IBKR sample."
                ),
                overrides=_fast_hold_overrides(values),
            )
        )

    for min_hold in [1, 5]:
        experiments.append(
            IbkrExperiment(
                name=f"fast_hold_{min_hold}s_hpo_capacity_t6",
                category="fast_hold_sweep",
                description=(
                    "Small HPO pass on the 1-second fast-hold setup, optimized for net PnL "
                    "after spot fees rather than raw signal count."
                ),
                overrides=_fast_hold_overrides(
                    {
                        "--min-holding-seconds": min_hold,
                        "--models": "histgb,extratrees,lightgbm,xgboost",
                        "--stacker": "weighted",
                        "--hpo": True,
                        "--hpo-profile": "capacity",
                        "--hpo-trials": 6,
                        "--target-trades-per-day": 36,
                    }
                ),
            )
        )

    for notional, slots in [
        (1000, 10),
        (1050, 9),
        (1100, 9),
        (1150, 8),
        (1200, 8),
        (1225, 8),
        (1250, 8),
        (1275, 8),
        (1300, 7),
        (1350, 7),
        (1500, 6),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"capacity_{_sizing_name(notional, slots)}",
                category="capacity_sizing",
                description=(
                    "Re-test slot size against IBKR's 0.18% per-side commission and $1.75 "
                    "minimum; this is where too-small orders can look busy but lose edge."
                ),
                overrides=_with_research_gate(
                    {"--notional": notional, "--max-open-positions": slots}
                ),
            )
        )

    for target_trades_per_day, stride, train, calibration, test in [
        (14, 10, 180, 60, 90),
        (16, 10, 180, 60, 90),
        (20, 10, 180, 60, 90),
        (16, 8, 225, 75, 112),
        (20, 8, 225, 75, 112),
        (16, 6, 300, 100, 150),
        (20, 6, 300, 100, 150),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_anchor_stride{stride}_tpd{target_trades_per_day}",
                category="stretch_goal",
                description=(
                    "Try to push the best $10K-cap anchor above 10 trades/day while "
                    "keeping the high net edge per trade."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--dense-stride-bars": stride,
                        "--target-trades-per-day": target_trades_per_day,
                        "--train-size": train,
                        "--calibration-size": calibration,
                        "--test-size": test,
                    }
                ),
            )
        )

    for target_trades_per_day, stride in [
        (20, 10),
        (24, 10),
        (20, 8),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_anchor_stride{stride}_tpd{target_trades_per_day}_score_direction",
                category="stretch_goal",
                description=(
                    "Calibrate score direction per fold before ranking target-frequency "
                    "signals, catching regimes where the model score is inverted but still "
                    "rank-informative."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--dense-stride-bars": stride,
                        "--target-trades-per-day": target_trades_per_day,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                        "--adaptive-selection-score-direction": True,
                    }
                ),
            )
        )

    for mode, release, target_trades_per_day in [
        ("online", "planned", 14),
        ("online", "actual", 14),
        ("online", "planned", 16),
        ("online", "actual", 16),
        ("online", "actual", 20),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_anchor_{mode}_{release}_tpd{target_trades_per_day}",
                category="stretch_goal",
                description=(
                    "Use online target-frequency selection and explicit open-position "
                    "accounting to test whether capacity, not model edge, is limiting turnover."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-frequency-mode": mode,
                        "--capacity-release-mode": release,
                        "--respect-open-positions": True,
                        "--target-trades-per-day": target_trades_per_day,
                    }
                ),
            )
        )

    for families in [
        "dense_baseline,dense_support_reclaim,dense_trend_continuation",
        "dense_baseline,dense_support_reclaim,dense_trend_continuation,dense_volatility_expansion",
    ]:
        suffix = families.replace(",", "_").replace("dense_", "")
        experiments.append(
            IbkrExperiment(
                name=f"stretch_anchor_family_{suffix}",
                category="stretch_goal",
                description=(
                    "Filter selection to setup families that carried most historical PnL "
                    "instead of increasing turnover indiscriminately."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-trades-per-day": 16,
                        "--selection-setup-families": families,
                    }
                ),
            )
        )

    for target_trades_per_day, notional, slots in [
        (14, 1_250, 8),
        (16, 1_250, 8),
        (20, 1_250, 8),
        (16, 1_000, 10),
        (20, 1_000, 10),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_phase17_fixed{notional}_slots{slots}_tpd{target_trades_per_day}",
                category="stretch_goal",
                description=(
                    "Port phase17's 10+ trades/day online selector to fixed small spot "
                    "slots under a strict $10K aggregate exposure cap."
                ),
                overrides=_phase17_light_overrides(
                    {
                        "--notional": notional,
                        "--max-open-positions": slots,
                        "--target-trades-per-day": target_trades_per_day,
                    }
                ),
            )
        )

    for target_trades_per_day, excluded in [
        (16, "dense_pullback_reclaim"),
        (16, "dense_pullback_reclaim,dense_breakout_momentum"),
        (20, "dense_pullback_reclaim"),
    ]:
        suffix = excluded.replace(",", "_").replace("dense_", "")
        experiments.append(
            IbkrExperiment(
                name=f"stretch_phase17_exclude_{suffix}_tpd{target_trades_per_day}",
                category="stretch_goal",
                description=(
                    "Remove setup families that were net negative in phase17 while keeping "
                    "the online selector above the desired turnover target."
                ),
                overrides=_phase17_light_overrides(
                    {
                        "--exclude-setup-families": excluded,
                        "--target-trades-per-day": target_trades_per_day,
                    }
                ),
            )
        )

    for target_trades_per_day, base, mid, high in [
        (16, 1_250, 2_500, 5_000),
        (20, 1_250, 2_500, 5_000),
        (20, 1_000, 2_000, 4_000),
        (24, 1_000, 2_000, 4_000),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_phase17_bucket_{base}_{mid}_{high}_tpd{target_trades_per_day}",
                category="stretch_goal",
                description=(
                    "Keep phase17's high-Sharpe selector but use smaller probability "
                    "buckets so a strict $10K cap can still take 10+ trades/day."
                ),
                overrides=_phase17_light_overrides(
                    {
                        "--target-trades-per-day": target_trades_per_day,
                        "--sizing-mode": "score_bucket",
                        "--sizing-score-field": "probability",
                        "--sizing-base-notional": base,
                        "--sizing-mid-notional": mid,
                        "--sizing-high-notional": high,
                        "--sizing-mid-score": 0.45,
                        "--sizing-high-score": 0.85,
                        "--max-open-positions": 8,
                    }
                ),
            )
        )

    for cap, base, mid, high in [
        (60_000, 10_000, 20_000, 30_000),
        (90_000, 15_000, 30_000, 45_000),
        (120_000, 20_000, 40_000, 60_000),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_phase17_scale_{cap // 1000}k_bucket",
                category="stretch_goal",
                description=(
                    "Scale the high-Sharpe phase17 bucket strategy to estimate the spot "
                    "capital needed for $500/day while preserving 10+ trades/day."
                ),
                overrides=_phase17_overrides(
                    {
                        "--starting-equity": cap,
                        "--paper-max-notional": cap,
                        "--notional": high,
                        "--sizing-base-notional": base,
                        "--sizing-mid-notional": mid,
                        "--sizing-high-notional": high,
                    }
                ),
            )
        )

    for cap, base, mid, high in [
        (10_000, 1_250, 2_500, 5_000),
        (30_000, 5_000, 10_000, 15_000),
        (90_000, 15_000, 30_000, 45_000),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_phase17_direction_{cap // 1000}k_bucket",
                category="stretch_goal",
                description=(
                    "Re-run the phase17 no-PM bucket selector with the calibrated score "
                    "direction fix and the adaptive floor preserved on low-probability folds."
                ),
                overrides=_phase17_overrides(
                    {
                        "--starting-equity": cap,
                        "--paper-max-notional": cap,
                        "--notional": high,
                        "--sizing-base-notional": base,
                        "--sizing-mid-notional": mid,
                        "--sizing-high-notional": high,
                        "--adaptive-selection-score-direction": True,
                    }
                ),
            )
        )

    for cap, notional, slots in [
        (20_000, 2_550, 8),
        (30_000, 3_825, 8),
        (40_000, 5_100, 8),
        (50_000, 6_375, 8),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"stretch_cap_scale_{cap // 1000}k_anchor",
                category="stretch_goal",
                description=(
                    "Scale the current best anchor to estimate the capital required for "
                    "$500/day while keeping spot execution and the same signal logic."
                ),
                overrides=_with_research_gate(
                    {
                        "--starting-equity": cap,
                        "--paper-max-notional": cap,
                        "--notional": notional,
                        "--max-open-positions": slots,
                    }
                ),
            )
        )

    core_families = "dense_baseline,dense_support_reclaim,dense_trend_continuation,dense_volatility_expansion"
    for stride, target_trades_per_day, hold_hours in [
        (10, 20, 24),
        (10, 24, 24),
        (8, 20, 24),
        (8, 24, 24),
        (6, 20, 24),
        (10, 20, 18),
        (10, 20, 36),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"deep_anchor_core_stride{stride}_tpd{target_trades_per_day}_hold{hold_hours}h",
                category="deep_tuning",
                description=(
                    "Push turnover only through historically productive setup families "
                    "instead of letting pullback/breakout candidates consume the $10K cap."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--dense-stride-bars": stride,
                        "--target-trades-per-day": target_trades_per_day,
                        "--max-holding-hours": hold_hours,
                        "--selection-setup-families": core_families,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for models, stacker, hpo_profile, hpo_trials, suffix in [
        ("extratrees", "average", "quota", 18, "extratrees_quota18"),
        ("histgb,extratrees", "weighted", "quota", 18, "histgb_extratrees_quota18"),
        ("histgb,lightgbm,extratrees", "weighted", "quota", 18, "histgb_lgbm_extratrees_quota18"),
        ("histgb,lightgbm,xgboost,extratrees", "weighted", "quota", 12, "tree_ensemble_quota12"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"deep_anchor_hpo_{suffix}",
                category="deep_tuning",
                description=(
                    "HPO around the strict $10K anchor, optimizing the model for quota "
                    "turnover and net PnL rather than generic classification score."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--dense-stride-bars": 10,
                        "--target-trades-per-day": 20,
                        "--selection-setup-families": core_families,
                        "--models": models,
                        "--stacker": stacker,
                        "--hpo": True,
                        "--hpo-profile": hpo_profile,
                        "--hpo-trials": hpo_trials,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for target_trades_per_day, excluded, suffix in [
        (20, "dense_pullback_reclaim,dense_breakout_momentum", "no_pullback_breakout"),
        (20, "dense_pullback_reclaim", "no_pullback"),
        (24, "dense_pullback_reclaim,dense_breakout_momentum", "tpd24_no_pullback_breakout"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"deep_phase17_90k_{suffix}",
                category="deep_tuning",
                description=(
                    "Tune the capital-scaled phase17 bucket strategy by removing weak "
                    "setup families while preserving the >10/day high-Sharpe objective."
                ),
                overrides=_phase17_90k_overrides(
                    {
                        "--target-trades-per-day": target_trades_per_day,
                        "--exclude-setup-families": excluded,
                    }
                ),
            )
        )

    for max_open_positions, capacity_release_mode, suffix in [
        (8, "planned", "slots8"),
        (8, "actual", "slots8_actual"),
        (10, "actual", "slots10_actual"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"deep_phase17_90k_no_pullback_breakout_{suffix}",
                category="deep_tuning",
                description=(
                    "Capacity pass on the strongest phase17 family-filtered variant, "
                    "testing whether blocked core-family signals can be added safely."
                ),
                overrides=_phase17_90k_overrides(
                    {
                        "--target-trades-per-day": 24,
                        "--exclude-setup-families": "dense_pullback_reclaim,dense_breakout_momentum",
                        "--max-open-positions": max_open_positions,
                        "--capacity-release-mode": capacity_release_mode,
                    }
                ),
            )
        )

    for gross, stop, suffix in [
        (180, 100, "g180_s100"),
        (220, 120, "g220_s120"),
        (160, 90, "g160_s90"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"deep_anchor_geometry_{suffix}",
                category="deep_tuning",
                description=(
                    "Retune target/stop geometry for the strict $10K anchor after the "
                    "0.18% per-side fee correction."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-trades-per-day": 20,
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                        "--selection-setup-families": core_families,
                    }
                ),
            )
        )

    for notional, slots in [
        (1_100, 9),
        (1_150, 8),
        (1_200, 8),
        (1_250, 8),
        (1_275, 8),
        (1_325, 7),
        (1_400, 7),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"param_anchor_n{notional}_slots{slots}_tpd20",
                category="parameter_sweep",
                description=(
                    "Parameter sweep around strict $10K anchor sizing; test whether "
                    "smaller slots add enough good signals or only increase fee drag."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": notional,
                        "--max-open-positions": slots,
                        "--target-trades-per-day": 20,
                        "--dense-stride-bars": 10,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for hold_hours in [14, 16, 18, 20, 22, 26, 30]:
        experiments.append(
            IbkrExperiment(
                name=f"param_anchor_hold{hold_hours}h_tpd20",
                category="parameter_sweep",
                description=(
                    "Vary strict $10K anchor vertical barrier to see whether faster "
                    "capital recycling improves turnover-adjusted PnL."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-trades-per-day": 20,
                        "--max-holding-hours": hold_hours,
                        "--dense-stride-bars": 10,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for target_trades_per_day in [18, 19, 21, 22]:
        experiments.append(
            IbkrExperiment(
                name=f"param_anchor_tpd{target_trades_per_day}",
                category="parameter_sweep",
                description=(
                    "Fine-grained strict $10K target-frequency sweep around the current "
                    "10+ trades/day anchor."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-trades-per-day": target_trades_per_day,
                        "--dense-stride-bars": 10,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for gross, stop in [
        (180, 90),
        (190, 100),
        (200, 90),
        (200, 110),
        (210, 100),
        (220, 110),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"param_anchor_geom_g{gross}_s{stop}",
                category="parameter_sweep",
                description=(
                    "Strict $10K target/stop geometry sweep after 0.18% per-side "
                    "commission correction."
                ),
                overrides=_with_research_gate(
                    {
                        "--notional": 1_275,
                        "--max-open-positions": 8,
                        "--target-trades-per-day": 20,
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                        "--dense-stride-bars": 10,
                        "--train-size": 180,
                        "--calibration-size": 60,
                        "--test-size": 90,
                    }
                ),
            )
        )

    for excluded, suffix in [
        ("dense_breakout_momentum", "no_breakout"),
        ("dense_volatility_expansion,dense_breakout_momentum", "no_vol_breakout"),
        ("dense_pullback_reclaim,dense_breakout_momentum", "no_pullback_breakout_tpd18"),
    ]:
        overrides: dict[str, ArgValue] = {
            "--exclude-setup-families": excluded,
        }
        if suffix.endswith("tpd18"):
            overrides["--target-trades-per-day"] = 18
        experiments.append(
            IbkrExperiment(
                name=f"param_phase17_90k_{suffix}",
                category="parameter_sweep",
                description=(
                    "Phase17 90k setup-family parameter sweep to test whether the "
                    "Pnl/day gain can keep Sharpe above 25."
                ),
                overrides=_phase17_90k_overrides(overrides),
            )
        )

    for gross, stop, suffix in [
        (100, 90, "g100_s90"),
        (110, 80, "g110_s80"),
        (120, 90, "g120_s90"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"param_phase17_90k_geom_{suffix}",
                category="parameter_sweep",
                description=(
                    "Phase17 90k target/stop geometry sweep around the current 100/80 "
                    "high-Sharpe setting."
                ),
                overrides=_phase17_90k_overrides(
                    {
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                    }
                ),
            )
        )

    for suffix, models, stacker, hpo_profile, hpo_trials, calibration, optimize_metric in [
        (
            "all_weighted_capacity_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_weighted_capacity_sharpe",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "sharpe",
        ),
        (
            "all_weighted_wide_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "wide",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_weighted_deep_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "deep",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "no_cat_capacity_pnl",
            "histgb,lightgbm,xgboost,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "boosted_extra_capacity_pnl",
            "lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "hist_lgbm_extra_capacity_pnl",
            "histgb,lightgbm,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "hist_extra_capacity_pnl",
            "histgb,extratrees",
            "weighted",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_average_capacity_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "average",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_best_capacity_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "best",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_logistic_capacity_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "logistic",
            "capacity",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_weighted_isotonic_capacity_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "capacity",
            18,
            "isotonic",
            "net_pnl",
        ),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"large_phase17_90k_no_breakout_{suffix}",
                category="large_hpo",
                description=(
                    "Large HPO/model-training sweep around the strongest 90K scaled "
                    "phase17 no-breakout BTC spot recipe."
                ),
                overrides=_phase17_90k_no_breakout_overrides(
                    {
                        "--models": models,
                        "--stacker": stacker,
                        "--hpo": True,
                        "--hpo-profile": hpo_profile,
                        "--hpo-trials": hpo_trials,
                        "--calibration-method": calibration,
                        "--optimize-metric": optimize_metric,
                    }
                ),
            )
        )

    for target_trades_per_day, optimize_metric in [
        (14, "net_pnl"),
        (18, "net_pnl"),
        (20, "net_pnl"),
        (22, "net_pnl"),
        (18, "sharpe"),
        (20, "sharpe"),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"large_phase17_90k_no_breakout_tpd{target_trades_per_day}_{optimize_metric}",
                category="large_hpo",
                description=(
                    "Fine sweep target frequency on the 90K no-breakout setup while "
                    "keeping capacity-aware HPO and bucket sizing."
                ),
                overrides=_phase17_90k_no_breakout_overrides(
                    {
                        "--target-trades-per-day": target_trades_per_day,
                        "--optimize-metric": optimize_metric,
                        "--hpo": True,
                        "--hpo-profile": "capacity",
                        "--hpo-trials": 24,
                    }
                ),
            )
        )

    for suffix, models, stacker, hpo_profile, hpo_trials, calibration, optimize_metric in [
        ("extratrees_quota_pnl", "extratrees", "average", "quota", 24, "sigmoid", "net_pnl"),
        ("hist_extra_quota_pnl", "histgb,extratrees", "weighted", "quota", 24, "sigmoid", "net_pnl"),
        (
            "hist_lgbm_extra_quota_pnl",
            "histgb,lightgbm,extratrees",
            "weighted",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "tree_ensemble_quota_pnl",
            "histgb,lightgbm,xgboost,extratrees",
            "weighted",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_trees_quota_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_trees_quota_sharpe",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "weighted",
            "quota",
            24,
            "sigmoid",
            "sharpe",
        ),
        (
            "randomforest_extra_quota_pnl",
            "randomforest,extratrees",
            "weighted",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "hist_extra_isotonic_quota_pnl",
            "histgb,extratrees",
            "weighted",
            "quota",
            18,
            "isotonic",
            "net_pnl",
        ),
        (
            "all_trees_average_quota_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "average",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
        (
            "all_trees_best_quota_pnl",
            "histgb,lightgbm,xgboost,catboost,extratrees",
            "best",
            "quota",
            24,
            "sigmoid",
            "net_pnl",
        ),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"large_anchor_g210_{suffix}",
                category="large_hpo",
                description=(
                    "Large HPO/model-training sweep around the best strict $10K anchor "
                    "geometry found so far."
                ),
                overrides=_strict_anchor_g210_overrides(
                    {
                        "--models": models,
                        "--stacker": stacker,
                        "--hpo": True,
                        "--hpo-profile": hpo_profile,
                        "--hpo-trials": hpo_trials,
                        "--calibration-method": calibration,
                        "--optimize-metric": optimize_metric,
                    }
                ),
            )
        )

    for gross, stop, hold_hours, target_trades_per_day in [
        (200, 90, 18, 20),
        (210, 100, 18, 20),
        (210, 100, 22, 22),
        (220, 100, 18, 20),
        (220, 110, 22, 22),
        (230, 110, 24, 24),
    ]:
        experiments.append(
            IbkrExperiment(
                name=(
                    f"large_anchor_geom_g{gross}_s{stop}_hold{hold_hours}h_"
                    f"tpd{target_trades_per_day}"
                ),
                category="large_hpo",
                description=(
                    "Joint strict $10K geometry, holding-window, and quota-HPO sweep "
                    "to improve trade count without throwing away edge."
                ),
                overrides=_strict_anchor_g210_overrides(
                    {
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                        "--max-holding-hours": hold_hours,
                        "--target-trades-per-day": target_trades_per_day,
                        "--models": "histgb,extratrees",
                        "--stacker": "weighted",
                        "--hpo": True,
                        "--hpo-profile": "quota",
                        "--hpo-trials": 24,
                    }
                ),
            )
        )

    for target_trades_per_day in [8, 10, 12, 14, 16, 20, 24]:
        experiments.append(
            IbkrExperiment(
                name=f"turnover_tpd{target_trades_per_day}_1250x8",
                category="turnover",
                description=(
                    "Probe how far the quota can be pushed before incremental trades become "
                    "fee drag or start consuming capacity from better signals."
                ),
                overrides=_with_research_gate({"--target-trades-per-day": target_trades_per_day}),
            )
        )

    for name, floor in [
        ("none", None),
        ("adaptive", "adaptive"),
        ("neg160bps", -0.016),
        ("neg150bps", -0.015),
        ("neg145bps", -0.0145),
        ("neg140bps", -0.014),
        ("neg135bps", -0.0135),
        ("neg130bps", -0.013),
    ]:
        overrides: dict[str, ArgValue] = {}
        if floor == "adaptive":
            overrides["--adaptive-selection-score-floor"] = True
        elif floor is not None:
            overrides["--selection-score-floor"] = floor
        experiments.append(
            IbkrExperiment(
                name=f"score_floor_{name}_return_first",
                category="selection_floor",
                description=(
                    "Find whether a weak-signal floor improves turnover quality without "
                    "reintroducing the too-few-trades failure mode."
                ),
                overrides=_with_research_gate(overrides),
            )
        )

    model_variants = [
        ("extratrees_avg", "extratrees", "average"),
        ("extratrees_weighted", "extratrees", "weighted"),
        ("histgb_avg", "histgb", "average"),
        ("lightgbm_avg", "lightgbm", "average"),
        ("xgboost_avg", "xgboost", "average"),
        ("catboost_avg", "catboost", "average"),
        ("extratrees_histgb_avg", "extratrees,histgb", "average"),
        ("extratrees_histgb_weighted", "extratrees,histgb", "weighted"),
        ("extratrees_lightgbm_avg", "extratrees,lightgbm", "average"),
        ("extratrees_xgboost_avg", "extratrees,xgboost", "average"),
        ("extratrees_lgbm_xgb_weighted", "extratrees,lightgbm,xgboost", "weighted"),
        ("trees_best_selector", "histgb,extratrees,lightgbm,xgboost", "best"),
    ]
    for name, models, stacker in model_variants:
        experiments.append(
            IbkrExperiment(
                name=f"model_{name}",
                category="model_stack",
                description=(
                    "Compare model families and stackers while holding data, costs, and "
                    "strategy geometry constant."
                ),
                overrides=_with_research_gate({"--models": models, "--stacker": stacker}),
            )
        )

    expanded_model_variants = [
        ("randomforest_avg", "randomforest", "average", {}),
        ("randomforest_weighted", "randomforest", "weighted", {}),
        ("randomforest_extratrees_weighted", "randomforest,extratrees", "weighted", {}),
        ("randomforest_histgb_extratrees_weighted", "randomforest,histgb,extratrees", "weighted", {}),
        ("catboost_extratrees_weighted", "catboost,extratrees", "weighted", {}),
        ("catboost_lightgbm_extratrees_weighted", "catboost,lightgbm,extratrees", "weighted", {}),
        ("boosted_forest_weighted", "lightgbm,xgboost,catboost,randomforest,extratrees", "weighted", {}),
        ("boosted_forest_best", "lightgbm,xgboost,catboost,randomforest,extratrees", "best", {}),
        ("boosted_forest_logistic", "lightgbm,xgboost,catboost,randomforest,extratrees", "logistic", {}),
        (
            "foundation_tabicl_extratrees",
            "tabicl,extratrees",
            "weighted",
            {"--foundation-max-samples": 384},
        ),
        (
            "foundation_tabpfn_extratrees",
            "tabpfn,extratrees",
            "weighted",
            {"--foundation-max-samples": 384},
        ),
        (
            "foundation_tabicl_tabpfn_extratrees",
            "tabicl,tabpfn,extratrees",
            "weighted",
            {"--foundation-max-samples": 384},
        ),
    ]
    for name, models, stacker, extra_overrides in expanded_model_variants:
        experiments.append(
            IbkrExperiment(
                name=f"model_expand_{name}",
                category="model_family_expansion",
                description=(
                    "Expand the strict $10K anchor model-family search to random forests, "
                    "CatBoost stacks, stacker variants, and bounded foundation-model probes."
                ),
                overrides=_strict_anchor_g210_overrides(
                    {
                        "--models": models,
                        "--stacker": stacker,
                        **extra_overrides,
                    }
                ),
            )
        )

    expanded_hpo_variants = [
        ("randomforest_quota_t12", "randomforest", "average", "quota", 12),
        ("randomforest_extra_quota_t18", "randomforest,extratrees", "weighted", "quota", 18),
        ("catboost_extra_quota_t12", "catboost,extratrees", "weighted", "quota", 12),
        ("boosted_forest_quota_t12", "lightgbm,xgboost,catboost,randomforest,extratrees", "weighted", "quota", 12),
        ("boosted_forest_capacity_t12", "lightgbm,xgboost,catboost,randomforest,extratrees", "weighted", "capacity", 12),
    ]
    for name, models, stacker, profile, trials in expanded_hpo_variants:
        experiments.append(
            IbkrExperiment(
                name=f"hpo_expand_{name}",
                category="model_family_expansion",
                description=(
                    "Run bounded HPO for the expanded model stacks, optimizing for net PnL "
                    "under the strict $10K BTC spot cap and high IBKR spot fees."
                ),
                overrides=_strict_anchor_g210_overrides(
                    {
                        "--models": models,
                        "--stacker": stacker,
                        "--hpo": True,
                        "--hpo-profile": profile,
                        "--hpo-trials": trials,
                        "--calibration-method": "sigmoid",
                        "--optimize-metric": "net_pnl",
                    }
                ),
            )
        )

    for name, models, profile, trials in [
        ("extratrees_quota_t6", "extratrees", "quota", 6),
        ("extratrees_capacity_t8", "extratrees", "capacity", 8),
        ("histgb_capacity_t6", "histgb", "capacity", 6),
        ("lightgbm_quota_t6", "lightgbm", "quota", 6),
        ("xgboost_quota_t6", "xgboost", "quota", 6),
        ("trees_capacity_t8", "histgb,extratrees,lightgbm,xgboost", "capacity", 8),
        ("trees_wide_t10", "histgb,extratrees,lightgbm,xgboost", "wide", 10),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"hpo_{name}",
                category="hpo",
                description=(
                    "Nested fold-level hyperparameter search, optimized for net PnL under "
                    "the current high-fee spot crypto assumptions."
                ),
                overrides=_with_research_gate(
                    {
                        "--models": models,
                        "--stacker": "weighted" if "," in models else "average",
                        "--hpo": True,
                        "--hpo-profile": profile,
                        "--hpo-trials": trials,
                    }
                ),
            )
        )

    for stride, train, calibration, test in [
        (5, 360, 120, 180),
        (8, 225, 75, 112),
        (10, 180, 60, 90),
        (12, 150, 50, 75),
        (15, 120, 40, 60),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"candidate_stride{stride}_fold{train}_{calibration}_{test}",
                category="candidate_density",
                description=(
                    "Stress candidate cadence and fold sizing to see whether more frequent "
                    "decision points add real edge or just more correlated labels."
                ),
                overrides=_with_research_gate(
                    {
                        "--dense-stride-bars": stride,
                        "--train-size": train,
                        "--calibration-size": calibration,
                        "--test-size": test,
                    }
                ),
            )
        )

    for hours, gross, stop in [
        (12, 180, 90),
        (18, 180, 90),
        (24, 200, 100),
        (30, 220, 110),
        (36, 250, 120),
        (48, 300, 150),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"exit_h{hours}_g{gross}_s{stop}",
                category="exit_geometry",
                description=(
                    "Vary vertical barrier and gross stop/take-profit geometry; exits may "
                    "last seconds or hours, but the model should not use a strict horizon."
                ),
                overrides=_with_research_gate(
                    {
                        "--max-holding-hours": hours,
                        "--minimum-gross-profit-bps": gross,
                        "--minimum-gross-stop-bps": stop,
                    }
                ),
            )
        )

    for move_bps, max_seconds in [(35, 21_600), (50, 43_200), (75, 86_400)]:
        experiments.append(
            IbkrExperiment(
                name=f"adaptive_horizon_move{move_bps}_max{max_seconds}s",
                category="adaptive_exit",
                description=(
                    "Enable volatility-scaled horizons with a 1-second minimum and a cap "
                    "that can still hold winners for hours."
                ),
                overrides=_with_research_gate(
                    {
                        "--adaptive-horizon": True,
                        "--min-holding-seconds": 1,
                        "--adaptive-horizon-target-move-bps": move_bps,
                        "--adaptive-horizon-max-seconds": max_seconds,
                    }
                ),
            )
        )

    for name, adverse, giveback, min_profit in [
        ("fast", 12, 18, 5),
        ("balanced", 18, 28, 8),
        ("loose", 30, 45, 12),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"dynamic_exit_{name}_seconds",
                category="adaptive_exit",
                description=(
                    "Replay causal second-level dynamic exits before the vertical barrier "
                    "to protect weak trades without imposing a fixed hold time."
                ),
                overrides=_with_research_gate(
                    {
                        "--dynamic-exit-overlay": True,
                        "--dynamic-exit-checkpoints-seconds": "5,15,30,60,120,300,900,1800",
                        "--dynamic-exit-adverse-bps": adverse,
                        "--dynamic-exit-giveback-bps": giveback,
                        "--dynamic-exit-min-profit-bps": min_profit,
                    }
                ),
            )
        )

    for name, context_bars in [
        ("base_eth_mbt", BASE_CONTEXT_BARS),
        ("full_ibkr_bidask_futures", FULL_CONTEXT_BARS),
        (
            "spot_bidask_only",
            f"ETH={DEFAULT_ETH_BARS},BTC_BIDASK={DEFAULT_BTC_BIDASK_BARS},"
            f"ETH_BIDASK={DEFAULT_ETH_BIDASK_BARS}",
        ),
        ("futures_mbt_met_trades", f"ETH={DEFAULT_ETH_BARS},MBT={DEFAULT_MBT_BARS},MET={DEFAULT_MET_BARS}"),
        (
            "futures_bidask_only",
            f"ETH={DEFAULT_ETH_BARS},MBT_BIDASK={DEFAULT_MBT_BIDASK_BARS},"
            f"MET_BIDASK={DEFAULT_MET_BIDASK_BARS}",
        ),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"data_context_{name}",
                category="data_features",
                description=(
                    "Compare spot-only, futures, and bid/ask context feature sets while "
                    "still executing spot crypto only."
                ),
                overrides=_with_research_gate({"--context-bars-jsonl": context_bars}),
            )
        )

    for mode, capacity_mode, research_gate in [
        ("online", "planned", False),
        ("online", "planned", True),
        ("online", "actual", True),
        ("quota", "planned", True),
    ]:
        suffix = "research" if research_gate else "production_gate"
        experiments.append(
            IbkrExperiment(
                name=f"selection_{mode}_{capacity_mode}_{suffix}",
                category="production_like",
                description=(
                    "Check production-like online selection against quota research and "
                    "research-only actual capacity release."
                ),
                overrides=_items(
                    {
                        "--target-frequency-mode": mode,
                        "--capacity-release-mode": capacity_mode,
                        "--respect-open-positions": True,
                        "--research-gate": research_gate,
                    }
                ),
            )
        )

    for name, commission, slippage, spread in [
        ("base_cost", 0.0018, 0.5, 0.1),
        ("wider_spread", 0.0018, 0.5, 0.5),
        ("high_slippage", 0.0018, 1.5, 0.5),
        ("fee_20bps", 0.0020, 0.5, 0.1),
        ("fee_25bps_stress", 0.0025, 1.0, 1.0),
    ]:
        experiments.append(
            IbkrExperiment(
                name=f"cost_{name}",
                category="cost_stress",
                description=(
                    "Confirm that a candidate improvement survives realistic fee/slippage "
                    "stress instead of relying on optimistic execution assumptions."
                ),
                overrides=_with_research_gate(
                    {
                        "--tier-rate": commission,
                        "--base-slippage-bps": slippage,
                        "--assumed-spread-bps": spread,
                    }
                ),
            )
        )

    experiments.extend(build_live_valid_strict_10k_experiment_suite())

    return experiments


def write_manifest(
    experiments: Iterable[IbkrExperiment],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    python_executable: str = sys.executable,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment in experiments:
        command = build_backtest_command(
            experiment,
            output_dir=output_dir,
            python_executable=python_executable,
        )
        rows.append(
            {
                **asdict(experiment),
                "artifact": str(experiment_artifact_path(output_dir, experiment)),
                "log_path": str(experiment_log_path(output_dir, experiment)),
                "command": command,
                "shell_command": shlex.join(command),
            }
        )
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "output_dir": str(output_dir),
        "experiments": rows,
        "summary": {
            "total": len(rows),
            "by_category": {
                category: sum(1 for row in rows if row["category"] == category)
                for category in sorted({str(row["category"]) for row in rows})
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_backtest_artifact(
    artifact: Path,
    *,
    experiment: IbkrExperiment | None = None,
    elapsed_seconds: float = 0.0,
    log_path: Path | None = None,
    error: str = "",
) -> IbkrExperimentResult:
    name = experiment.name if experiment else artifact.stem
    category = experiment.category if experiment else artifact.parent.name
    description = experiment.description if experiment else ""
    if not artifact.exists():
        return IbkrExperimentResult(
            name=name,
            category=category,
            description=description,
            artifact=str(artifact),
            status="missing",
            elapsed_seconds=elapsed_seconds,
            log_path=str(log_path or ""),
            error=error,
        )
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        return IbkrExperimentResult(
            name=name,
            category=category,
            description=description,
            artifact=str(artifact),
            status="invalid_json",
            elapsed_seconds=elapsed_seconds,
            log_path=str(log_path or ""),
            error=error or f"{type(exc).__name__}: {exc}",
        )
    summary = payload.get("summary") or payload.get("backtest_summary") or {}
    coverage = summary.get("data_coverage") if isinstance(summary, dict) else {}
    coverage = coverage if isinstance(coverage, dict) else {}
    primary_coverage = coverage.get("primary") if isinstance(coverage.get("primary"), dict) else {}
    label_coverage = coverage.get("label_bars") if isinstance(coverage.get("label_bars"), dict) else {}
    execution_coverage = (
        coverage.get("execution_bars") if isinstance(coverage.get("execution_bars"), dict) else {}
    )
    primary_interval = str(primary_coverage.get("interval") or "")
    label_interval = str(label_coverage.get("interval") or "")
    execution_interval = str(execution_coverage.get("interval") or "")
    uses_one_second_execution = (
        bool(label_coverage.get("enabled"))
        and bool(execution_coverage.get("enabled"))
        and label_interval == "1s"
        and execution_interval == "1s"
    )
    return IbkrExperimentResult(
        name=name,
        category=category,
        description=description,
        artifact=str(artifact),
        status="ok" if isinstance(summary, dict) and summary else "non_result",
        trades=_safe_int(summary.get("trades")),
        configured_span_days=_safe_float(summary.get("configured_span_days")),
        trades_per_prediction_day=_safe_float(summary.get("trades_per_prediction_day")),
        trades_per_configured_day=_safe_float(summary.get("trades_per_configured_day")),
        trades_per_active_day=_safe_float(summary.get("trades_per_active_day")),
        pnl_per_prediction_day=_safe_float(summary.get("pnl_per_prediction_day")),
        pnl_per_configured_day=_safe_float(summary.get("pnl_per_configured_day")),
        candidate_predictions=_safe_int(summary.get("candidate_predictions")),
        model_approved_signals=_safe_int(summary.get("model_approved_signals")),
        rejected_signals=_safe_int(summary.get("rejected_signals")),
        net_pnl=_safe_float(summary.get("net_pnl")),
        gross_pnl=_safe_float(summary.get("gross_pnl")),
        total_return=_safe_float(summary.get("total_return")),
        sharpe=_safe_float(summary.get("sharpe")),
        daily_sharpe=_safe_float(summary.get("daily_sharpe", summary.get("sharpe"))),
        trade_level_sharpe=_safe_float(summary.get("trade_level_sharpe")),
        deflated_sharpe=_safe_float(summary.get("deflated_sharpe", summary.get("sharpe"))),
        multiple_testing_trials=_safe_int(summary.get("multiple_testing_trials") or 1),
        multiple_testing_haircut=_safe_float(summary.get("multiple_testing_haircut")),
        profit_factor=_safe_float(summary.get("profit_factor")),
        max_drawdown=_safe_float(summary.get("max_drawdown")),
        max_intratrade_drawdown=_safe_float(summary.get("max_intratrade_drawdown")),
        average_max_adverse_excursion=_safe_float(summary.get("average_max_adverse_excursion")),
        hit_rate=_safe_float(summary.get("hit_rate")),
        total_commission_estimate=_safe_float(summary.get("total_commission_estimate")),
        total_slippage_cost_estimate=_safe_float(summary.get("total_slippage_cost_estimate")),
        total_spread_cost_estimate=_safe_float(summary.get("total_spread_cost_estimate")),
        primary_interval=primary_interval,
        label_interval=label_interval,
        execution_interval=execution_interval,
        uses_one_second_execution=uses_one_second_execution,
        elapsed_seconds=elapsed_seconds,
        log_path=str(log_path or ""),
        error=error,
    )


def summarize_results(results: Iterable[IbkrExperimentResult]) -> list[IbkrExperimentResult]:
    rows = list(results)
    ok_trials = max(1, sum(1 for result in rows if result.status == "ok"))
    haircut = math.sqrt(2.0 * math.log(float(ok_trials))) if ok_trials > 1 else 0.0
    adjusted = [
        replace(
            result,
            multiple_testing_trials=ok_trials,
            multiple_testing_haircut=haircut,
            deflated_sharpe=result.daily_sharpe - haircut,
        )
        if result.status == "ok"
        else result
        for result in rows
    ]
    return sorted(adjusted, key=lambda result: result.rank_key, reverse=True)


def write_results(
    results: Iterable[IbkrExperimentResult],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    rows = [asdict(result) for result in summarize_results(results)]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(IbkrExperimentResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def collect_existing_results(
    experiments: Iterable[IbkrExperiment],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[IbkrExperimentResult]:
    results = []
    for experiment in experiments:
        results.append(
            parse_backtest_artifact(
                experiment_artifact_path(output_dir, experiment),
                experiment=experiment,
                log_path=experiment_log_path(output_dir, experiment),
            )
        )
    return [result for result in results if result.status != "missing"]


def filter_experiments(
    experiments: Iterable[IbkrExperiment],
    *,
    categories: Iterable[str] = (),
    names: Iterable[str] = (),
    limit: int = 0,
) -> list[IbkrExperiment]:
    category_set = {category for category in categories if category}
    name_set = {name for name in names if name}
    selected = [
        experiment
        for experiment in experiments
        if (not category_set or experiment.category in category_set)
        and (not name_set or experiment.name in name_set)
    ]
    if limit > 0:
        selected = selected[:limit]
    return selected


def run_experiments(
    experiments: Iterable[IbkrExperiment],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python_executable: str = sys.executable,
    force: bool = False,
) -> list[IbkrExperimentResult]:
    results: list[IbkrExperimentResult] = []
    for experiment in experiments:
        artifact = experiment_artifact_path(output_dir, experiment)
        log_path = experiment_log_path(output_dir, experiment)
        if artifact.exists() and not force:
            results.append(parse_backtest_artifact(artifact, experiment=experiment, log_path=log_path))
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_backtest_command(
            experiment,
            output_dir=output_dir,
            python_executable=python_executable,
        )
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("$ " + shlex.join(command) + "\n\n")
            handle.flush()
            proc = subprocess.run(
                command,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = round(time.time() - started, 3)
        error = ""
        if proc.returncode != 0:
            error = f"command exited {proc.returncode}; see {log_path}"
        results.append(
            parse_backtest_artifact(
                artifact,
                experiment=experiment,
                elapsed_seconds=elapsed,
                log_path=log_path,
                error=error,
            )
        )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local IBKR 1s research experiment suite.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    experiments = build_ibkr_experiment_suite()
    selected = filter_experiments(
        experiments,
        categories=args.category,
        names=args.name,
        limit=args.limit,
    )
    manifest = write_manifest(experiments, args.output_dir, python_executable=args.python)
    if args.dry_run or not args.run:
        payload = {
            "manifest": str(manifest),
            "selected": len(selected),
            "total": len(experiments),
            "commands": [
                shlex.join(
                    build_backtest_command(
                        experiment,
                        output_dir=args.output_dir,
                        python_executable=args.python,
                    )
                )
                for experiment in selected
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    results = run_experiments(
        selected,
        output_dir=args.output_dir,
        python_executable=args.python,
        force=args.force,
    )
    all_results_by_name = {
        result.name: result
        for result in collect_existing_results(experiments, args.output_dir)
    }
    for result in results:
        all_results_by_name[result.name] = result
    json_path, csv_path = write_results(all_results_by_name.values(), args.output_dir)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "summary_json": str(json_path),
                "summary_csv": str(csv_path),
                "ran": len(results),
                "ok": sum(1 for result in results if result.status == "ok" and not result.error),
                "failed": [
                    asdict(result)
                    for result in results
                    if result.status != "ok" or result.error
                ],
                "top": [asdict(result) for result in summarize_results(all_results_by_name.values())[:10]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
