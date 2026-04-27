"""Colab-oriented research runner helpers.

The functions in this module intentionally wrap the existing CLI instead of
reimplementing training.  A Colab notebook can then run large, resumable
experiments while still exercising the same code paths as local research.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import csv
import json
import os
import subprocess
import sys
import time


DEFAULT_MODELS = "logistic,histgb,extratrees,lightgbm,xgboost,catboost"
FAST_MODELS = "logistic,histgb,extratrees,lightgbm,xgboost"
H100_MODELS = DEFAULT_MODELS


@dataclass(frozen=True, slots=True)
class ResearchExperiment:
    name: str
    interval: str
    years: int
    context_interval: str
    candidate_mode: str
    candidate_types: str
    max_holding_hours: int
    net_profit_target: float
    net_stop_loss: float
    minimum_gross_profit_bps: float
    minimum_gross_stop_bps: float
    selection_score: str
    calibration_method: str
    stacker: str
    models: str = DEFAULT_MODELS
    side_mode: str = "long"
    dense_stride_bars: int = 1
    candidate_lookback_bars: int = 24
    candidate_rolling_window_bars: int = 1000
    target_trades_per_day: float = 4.0
    min_calibration_trades: int = 8
    hpo: bool = True
    specialist_models: bool = True
    candidate_type_thresholds: bool = False
    empirical_payoff_ev: bool = True
    confidence_scaled_sizing: bool = True
    allow_negative_ev_frequency_probe: bool = False
    allow_spot_short_research: bool = False
    allow_research_short_backtest: bool = False
    kronos_features: bool = False
    kronos_mode: str = "proxy"
    kronos_lookback_bars: int = 0
    kronos_embedding_dims: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    name: str
    artifact: str
    status: str
    trades: int = 0
    trades_per_day: float = 0.0
    raw_candidates_per_day: float = 0.0
    net_pnl: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    samples: int = 0
    error: str = ""
    log_path: str = ""

    @property
    def champion_score(self) -> tuple[float, float, float, int]:
        return (self.sharpe, self.net_pnl, -self.max_drawdown, self.trades)


def _flag(args: list[str], enabled: bool, value: str) -> None:
    if enabled:
        args.append(value)


def experiment_artifact_path(artifact_dir: Path, experiment: ResearchExperiment) -> Path:
    return artifact_dir / f"{experiment.name}.json"


def build_signal_audit_command(
    experiment: ResearchExperiment,
    *,
    artifact_dir: Path,
    cache_dir: Path,
    python_executable: str = sys.executable,
    config: str = "configs/paper.example.toml",
    symbol: str = "BTCUSDT",
    context_symbols: str = "ETHUSDT,SOLUSDT,ETHBTC",
) -> list[str]:
    output = experiment_artifact_path(artifact_dir, experiment)
    args = [
        python_executable,
        "-m",
        "zeroalpha.cli",
        "model",
        "signal-audit",
        "--config",
        config,
        "--symbol",
        symbol,
        "--interval",
        experiment.interval,
        "--context-symbols",
        context_symbols,
        "--context-interval",
        experiment.context_interval,
        "--years",
        str(experiment.years),
        "--cache-dir",
        str(cache_dir),
        "--candidate-mode",
        experiment.candidate_mode,
        "--side-mode",
        experiment.side_mode,
        "--dense-stride-bars",
        str(experiment.dense_stride_bars),
        "--candidate-lookback-bars",
        str(experiment.candidate_lookback_bars),
        "--candidate-rolling-window-bars",
        str(experiment.candidate_rolling_window_bars),
        "--max-holding-hours",
        str(experiment.max_holding_hours),
        "--net-profit-target",
        str(experiment.net_profit_target),
        "--net-stop-loss",
        str(experiment.net_stop_loss),
        "--minimum-gross-profit-bps",
        str(experiment.minimum_gross_profit_bps),
        "--minimum-gross-stop-bps",
        str(experiment.minimum_gross_stop_bps),
        "--assumed-spread-bps",
        "4",
        "--tier-rate",
        "0.0004",
        "--base-slippage-bps",
        "1",
        "--safety-margin-bps",
        "2",
        "--minimum-probability",
        "0",
        "--minimum-expected-value",
        "0",
        "--target-trades-per-day",
        str(experiment.target_trades_per_day),
        "--research-gate",
        "--selection-score",
        experiment.selection_score,
        "--calibration-method",
        experiment.calibration_method,
        "--stacker",
        experiment.stacker,
        "--models",
        experiment.models,
        "--min-calibration-trades",
        str(experiment.min_calibration_trades),
        "--max-open-positions",
        "4",
        "--max-signals-per-timestamp",
        "1",
        "--output",
        str(output),
    ]
    if experiment.candidate_types:
        args.extend(["--candidate-types", experiment.candidate_types])
    _flag(args, experiment.hpo, "--hpo")
    _flag(args, experiment.specialist_models, "--specialist-models")
    _flag(args, experiment.candidate_type_thresholds, "--candidate-type-thresholds")
    _flag(args, experiment.empirical_payoff_ev, "--empirical-payoff-ev")
    _flag(args, experiment.confidence_scaled_sizing, "--confidence-scaled-sizing")
    _flag(args, experiment.allow_negative_ev_frequency_probe, "--allow-negative-ev-frequency-probe")
    _flag(args, experiment.allow_spot_short_research, "--allow-spot-short-research")
    _flag(args, experiment.allow_research_short_backtest, "--allow-research-short-backtest")
    _flag(args, experiment.kronos_features, "--kronos-features")
    if experiment.kronos_features:
        args.extend(["--kronos-mode", experiment.kronos_mode])
        if experiment.kronos_lookback_bars:
            args.extend(["--kronos-lookback-bars", str(experiment.kronos_lookback_bars)])
        if experiment.kronos_embedding_dims:
            args.extend(["--kronos-embedding-dims", str(experiment.kronos_embedding_dims)])
    return args


def _active_experiments(interval: str, years: int, *, models: str) -> list[ResearchExperiment]:
    context_interval = "1h" if interval in {"1m", "3m", "5m", "15m"} else interval
    base = {
        "interval": interval,
        "years": years,
        "context_interval": context_interval,
        "candidate_mode": "active",
        "models": models,
        "dense_stride_bars": 1 if interval != "1m" else 5,
        "candidate_rolling_window_bars": 1000 if interval != "1m" else 1500,
    }
    geometries = [
        ("g25_20_h2", 0.0025, 0.0020, 2, 36, 18),
        ("g30_22_h3", 0.0030, 0.0022, 3, 40, 20),
        ("g35_30_h4", 0.0035, 0.0030, 4, 45, 22),
        ("g45_30_h6", 0.0045, 0.0030, 6, 55, 24),
        ("g60_40_h8", 0.0060, 0.0040, 8, 70, 30),
    ]
    candidate_sets = [
        ("all", ""),
        (
            "champion_types",
            "active_liquidity_reversal,active_pullback_reclaim,active_breakout_continuation",
        ),
        ("trend_breakout", "active_pullback_reclaim,active_breakout_continuation,active_squeeze_breakout"),
        ("reversion", "active_liquidity_reversal,active_range_mean_reversion"),
    ]
    ranking_modes = ["expected_utility", "expected_value"]
    calibrators = ["isotonic", "sigmoid"]
    experiments: list[ResearchExperiment] = []
    for geom_name, target, stop, horizon, gross_target, gross_stop in geometries:
        for set_name, candidate_types in candidate_sets:
            for ranking in ranking_modes:
                for calibrator in calibrators:
                    threshold_variants = [False, True] if set_name in {"all", "champion_types"} else [False]
                    for candidate_type_thresholds in threshold_variants:
                        threshold_suffix = "_ctype" if candidate_type_thresholds else ""
                        experiments.append(
                            ResearchExperiment(
                                name=(
                                    f"{interval}_{years}y_{geom_name}_{set_name}_"
                                    f"{ranking}_{calibrator}{threshold_suffix}"
                                ),
                                max_holding_hours=horizon,
                                net_profit_target=target,
                                net_stop_loss=stop,
                                minimum_gross_profit_bps=gross_target,
                                minimum_gross_stop_bps=gross_stop,
                                selection_score=ranking,
                                calibration_method=calibrator,
                                stacker="weighted",
                                candidate_types=candidate_types,
                                min_calibration_trades=6 if set_name != "all" else 10,
                                candidate_type_thresholds=candidate_type_thresholds,
                                **base,
                            )
                        )
    return experiments


def build_experiment_matrix(
    *,
    include_15m: bool = True,
    include_5m: bool = True,
    include_1m: bool = True,
    models: str = FAST_MODELS,
    years_15m: int = 6,
    years_5m: int = 4,
    years_1m: int = 2,
) -> list[ResearchExperiment]:
    experiments: list[ResearchExperiment] = []
    if include_15m:
        experiments.extend(_active_experiments("15m", years_15m, models=models))
    if include_5m:
        experiments.extend(_active_experiments("5m", years_5m, models=models))
    if include_1m:
        experiments.extend(_active_experiments("1m", years_1m, models=models))
    experiments.extend(
        [
            ResearchExperiment(
                name=f"15m_{years_15m}y_long_short_research_g35_30",
                interval="15m",
                years=years_15m,
                context_interval="1h",
                candidate_mode="active",
                candidate_types="",
                max_holding_hours=4,
                net_profit_target=0.0035,
                net_stop_loss=0.0030,
                minimum_gross_profit_bps=45,
                minimum_gross_stop_bps=22,
                selection_score="expected_utility",
                calibration_method="isotonic",
                stacker="weighted",
                models=models,
                side_mode="long_short",
                allow_spot_short_research=True,
                allow_research_short_backtest=True,
            ),
            ResearchExperiment(
                name=f"15m_{years_15m}y_forced_frequency_probe",
                interval="15m",
                years=years_15m,
                context_interval="1h",
                candidate_mode="active",
                candidate_types="",
                max_holding_hours=4,
                net_profit_target=0.0035,
                net_stop_loss=0.0030,
                minimum_gross_profit_bps=45,
                minimum_gross_stop_bps=22,
                selection_score="expected_utility",
                calibration_method="isotonic",
                stacker="weighted",
                models=models,
                allow_negative_ev_frequency_probe=True,
            ),
        ]
    )
    return experiments


def parse_signal_audit_artifact(path: Path) -> ExperimentResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ExperimentResult(name=path.stem, artifact=str(path), status="invalid_json", error=str(exc))
    summary = payload.get("backtest_summary") or {}
    return ExperimentResult(
        name=path.stem,
        artifact=str(path),
        status="ok",
        trades=int(summary.get("trades") or 0),
        trades_per_day=float(summary.get("trades_per_prediction_day") or 0.0),
        raw_candidates_per_day=float(payload.get("raw_candidates_per_day") or 0.0),
        net_pnl=float(summary.get("net_pnl") or 0.0),
        total_return=float(summary.get("total_return") or 0.0),
        sharpe=float(summary.get("sharpe") or 0.0),
        profit_factor=float(summary.get("profit_factor") or 0.0),
        max_drawdown=float(summary.get("max_drawdown") or 0.0),
        hit_rate=float(summary.get("hit_rate") or 0.0),
        samples=int(payload.get("samples") or 0),
    )


def summarize_artifacts(artifact_dir: Path) -> list[ExperimentResult]:
    results = [parse_signal_audit_artifact(path) for path in sorted(artifact_dir.glob("*.json"))]
    return sorted(results, key=lambda row: row.champion_score, reverse=True)


def write_summary_csv(results: Iterable[ExperimentResult], output_path: Path) -> None:
    rows = [asdict(result) for result in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExperimentResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_manifest(experiments: Iterable[ResearchExperiment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(experiment) for experiment in experiments]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_research_report(results: Iterable[ExperimentResult], output_path: Path) -> None:
    rows = list(results)
    ok = [row for row in rows if row.status == "ok"]
    failures = [row for row in rows if row.status != "ok"]

    def top(filtered: list[ExperimentResult], *, limit: int = 25) -> list[dict[str, object]]:
        ranked = sorted(filtered, key=lambda row: row.champion_score, reverse=True)[:limit]
        return [asdict(row) for row in ranked]

    positive = [row for row in ok if row.sharpe > 0 and row.net_pnl > 0]
    payload = {
        "total_results": len(rows),
        "ok_results": len(ok),
        "failed_results": len(failures),
        "failures": [asdict(row) for row in failures[:50]],
        "top_overall": top(ok),
        "top_positive": top(positive),
        "top_trade_frequency_positive": {
            "tpd_ge_0_5": top([row for row in positive if row.trades_per_day >= 0.5]),
            "tpd_ge_1": top([row for row in positive if row.trades_per_day >= 1.0]),
            "tpd_ge_2": top([row for row in positive if row.trades_per_day >= 2.0]),
            "tpd_ge_4": top([row for row in positive if row.trades_per_day >= 4.0]),
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "ZEROALPHA_USE_GPU",
                "ZEROALPHA_MODEL_N_JOBS",
                "ZEROALPHA_XGBOOST_DEVICE",
                "ZEROALPHA_GPU_DEVICES",
            )
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_experiments(
    experiments: Iterable[ResearchExperiment],
    *,
    artifact_dir: Path,
    cache_dir: Path,
    python_executable: str = sys.executable,
    resume: bool = True,
    timeout_seconds: int | None = None,
    stop_after: int | None = None,
    stream_output: bool = False,
) -> list[ExperimentResult]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir = artifact_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    for index, experiment in enumerate(experiments, start=1):
        if stop_after is not None and index > stop_after:
            break
        artifact = experiment_artifact_path(artifact_dir, experiment)
        log_path = log_dir / f"{experiment.name}.log"
        if resume and artifact.exists():
            result = parse_signal_audit_artifact(artifact)
            results.append(ExperimentResult(**{**asdict(result), "log_path": str(log_path)}))
            continue
        command = build_signal_audit_command(
            experiment,
            artifact_dir=artifact_dir,
            cache_dir=cache_dir,
            python_executable=python_executable,
        )
        started = time.time()
        print(f"\n[{index}] running {experiment.name}")
        print(f"artifact={artifact}")
        print(f"log={log_path}")
        try:
            if stream_output:
                with log_path.open("w", encoding="utf-8") as handle:
                    handle.write(" ".join(command) + "\n\n")
                    handle.flush()
                    completed = subprocess.run(
                        command,
                        check=False,
                        timeout=timeout_seconds,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
            else:
                completed = subprocess.run(
                    command,
                    check=False,
                    timeout=timeout_seconds,
                    capture_output=True,
                    text=True,
                )
                log_path.write_text(
                    " ".join(command) + "\n\n" + completed.stdout + completed.stderr,
                    encoding="utf-8",
                )
        except subprocess.TimeoutExpired as exc:
            results.append(
                ExperimentResult(
                    name=experiment.name,
                    artifact=str(artifact),
                    status="timeout",
                    error=str(exc),
                    log_path=str(log_path),
                )
            )
            continue
        elapsed = time.time() - started
        if completed.returncode != 0:
            tail = ""
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            if tail:
                print(tail)
            results.append(
                ExperimentResult(
                    name=experiment.name,
                    artifact=str(artifact),
                    status=f"failed:{completed.returncode}",
                    error=f"elapsed_seconds={elapsed:.1f}",
                    log_path=str(log_path),
                )
            )
            continue
        result = parse_signal_audit_artifact(artifact)
        result = ExperimentResult(**{**asdict(result), "log_path": str(log_path)})
        results.append(result)
        print(
            f"{experiment.name}: trades={result.trades} tpd={result.trades_per_day:.3f} "
            f"pnl={result.net_pnl:.2f} sharpe={result.sharpe:.3f} "
            f"pf={result.profit_factor:.3f} elapsed={elapsed:.1f}s"
        )
    return sorted(results, key=lambda row: row.champion_score, reverse=True)
