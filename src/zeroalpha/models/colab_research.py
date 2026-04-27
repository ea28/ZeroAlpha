"""Colab-oriented research runner helpers.

The functions in this module intentionally wrap the existing CLI instead of
reimplementing training.  A Colab notebook can then run large, resumable
experiments while still exercising the same code paths as local research.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

from zeroalpha.data.external.binance import fetch_klines_archive_range


DEFAULT_MODELS = "logistic,histgb,extratrees,lightgbm,xgboost,catboost"
FAST_MODELS = "logistic,histgb,extratrees,lightgbm,xgboost"
H100_MODELS = DEFAULT_MODELS
FOUNDATION_MODELS = "lightgbm,xgboost,catboost,tabicl,tabpfn"
DEFAULT_CONTEXT_SYMBOLS = "ETHUSDT,SOLUSDT,ETHBTC,BNBUSDT,XRPUSDT"
DEFAULT_KRONOS_SWEEP = ((64, 8), (128, 16), (256, 16))
DEFAULT_FOUNDATION_SAMPLE_WINDOWS = (512, 1024, 2048)


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
    foundation_max_samples: int = 0
    hpo_profile: str = "standard"
    assumed_spread_bps: float = 4.0
    tier_rate: float = 0.0004
    base_slippage_bps: float = 1.0
    safety_margin_bps: float = 2.0


@dataclass(frozen=True, slots=True)
class PrefetchRequest:
    symbol: str
    interval: str
    years: int
    role: str
    required: bool


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
    elapsed_seconds: float = 0.0
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
    context_symbols: str = DEFAULT_CONTEXT_SYMBOLS,
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
        "--allow-data-gaps",
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
        str(experiment.assumed_spread_bps),
        "--tier-rate",
        str(experiment.tier_rate),
        "--base-slippage-bps",
        str(experiment.base_slippage_bps),
        "--safety-margin-bps",
        str(experiment.safety_margin_bps),
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
    if experiment.foundation_max_samples:
        args.extend(["--foundation-max-samples", str(experiment.foundation_max_samples)])
    if experiment.hpo_profile != "standard":
        args.extend(["--hpo-profile", experiment.hpo_profile])
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


def prefetch_requests_for_experiments(
    experiments: Iterable[ResearchExperiment],
    *,
    primary_symbol: str = "BTCUSDT",
    context_symbols: str = DEFAULT_CONTEXT_SYMBOLS,
) -> list[PrefetchRequest]:
    by_key: dict[tuple[str, str, str, bool], PrefetchRequest] = {}
    for experiment in experiments:
        primary_key = (primary_symbol.upper(), experiment.interval, "primary", True)
        current = by_key.get(primary_key)
        if current is None or experiment.years > current.years:
            by_key[primary_key] = PrefetchRequest(
                symbol=primary_symbol.upper(),
                interval=experiment.interval,
                years=experiment.years,
                role="primary",
                required=True,
            )
        for raw_symbol in context_symbols.split(","):
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            context_key = (symbol, experiment.context_interval, "context", False)
            current = by_key.get(context_key)
            if current is None or experiment.years > current.years:
                by_key[context_key] = PrefetchRequest(
                    symbol=symbol,
                    interval=experiment.context_interval,
                    years=experiment.years,
                    role="context",
                    required=False,
                )
    return sorted(by_key.values(), key=lambda item: (item.role, item.symbol, item.interval))


def prefetch_binance_cache(
    experiments: Iterable[ResearchExperiment],
    *,
    cache_dir: Path,
    manifest_path: Path,
    primary_symbol: str = "BTCUSDT",
    context_symbols: str = DEFAULT_CONTEXT_SYMBOLS,
    end: datetime | None = None,
) -> dict[str, object]:
    end = end or datetime.now(tz=UTC)
    requests = prefetch_requests_for_experiments(
        experiments,
        primary_symbol=primary_symbol,
        context_symbols=context_symbols,
    )
    rows: list[dict[str, object]] = []
    started = time.time()
    for request in requests:
        request_start = time.time()
        start = end - timedelta(days=365 * request.years)
        try:
            bars = fetch_klines_archive_range(
                symbol=request.symbol,
                interval=request.interval,
                start=start,
                end=end,
                cache_dir=cache_dir / request.symbol,
            )
            if request.required and not bars:
                raise RuntimeError(
                    f"required {request.symbol} {request.interval} prefetch returned no bars"
                )
            rows.append(
                {
                    **asdict(request),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "bars": len(bars),
                    "status": "ok" if bars else "missing_optional",
                    "elapsed_seconds": round(time.time() - request_start, 3),
                    "cache_path": str(cache_dir / request.symbol),
                }
            )
        except Exception as exc:
            if request.required:
                raise
            rows.append(
                {
                    **asdict(request),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "bars": 0,
                    "status": "failed_optional",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - request_start, 3),
                    "cache_path": str(cache_dir / request.symbol),
                }
            )
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": round(time.time() - started, 3),
        "cache_dir": str(cache_dir),
        "requests": rows,
        "summary": {
            "total": len(rows),
            "ok": sum(1 for row in rows if row["status"] == "ok"),
            "missing_optional": sum(1 for row in rows if row["status"] == "missing_optional"),
            "failed_optional": sum(1 for row in rows if row["status"] == "failed_optional"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def foundation_model_stack_from_smoke(
    smoke_stdout: str,
    *,
    include_catboost: bool = True,
) -> str:
    models = ["lightgbm", "xgboost"]
    if include_catboost:
        models.append("catboost")
    try:
        payload = json.loads(smoke_stdout)
    except json.JSONDecodeError:
        payload = []
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict) or not row.get("ok"):
                continue
            model_name = str(row.get("model_name") or "").lower()
            if model_name in {"tabicl", "tabiclv2", "tabpfn"} and model_name not in models:
                models.append(model_name)
    return ",".join(models)


def make_foundation_kronos_experiments(
    results: Iterable[ExperimentResult],
    experiments: Iterable[ResearchExperiment],
    *,
    top_n: int = 12,
    models: str = FOUNDATION_MODELS,
    sample_windows: tuple[int, ...] = DEFAULT_FOUNDATION_SAMPLE_WINDOWS,
    kronos_configs: tuple[tuple[int, int], ...] = DEFAULT_KRONOS_SWEEP,
    kronos_mode: str = "proxy",
) -> list[ResearchExperiment]:
    by_name = _experiment_lookup(experiments)
    candidates = [
        row
        for row in sorted(results, key=lambda item: item.champion_score, reverse=True)
        if row.status == "ok"
        and row.net_pnl > 0
        and row.sharpe > 0
        and row.profit_factor >= 1.0
        and row.trades > 0
    ][:top_n]
    foundation: list[ResearchExperiment] = []
    seen: set[str] = set()
    for row in candidates:
        source = by_name.get(row.name)
        if source is None:
            continue
        for lookback, dims in kronos_configs:
            for window in sample_windows:
                name = f"{source.name}_foundation_kronos_l{lookback}_d{dims}_w{window}"
                if name in seen:
                    continue
                seen.add(name)
                foundation.append(
                    replace(
                        source,
                        name=name,
                        models=models,
                        kronos_features=True,
                        kronos_mode=kronos_mode,
                        kronos_lookback_bars=lookback,
                        kronos_embedding_dims=dims,
                        foundation_max_samples=window,
                        hpo_profile="deep",
                    )
                )
    return foundation


def parse_signal_audit_artifact(path: Path) -> ExperimentResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ExperimentResult(name=path.stem, artifact=str(path), status="invalid_json", error=str(exc))
    if not isinstance(payload, dict) or "backtest_summary" not in payload:
        return ExperimentResult(name=path.stem, artifact=str(path), status="non_result")
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
    results = [result for result in results if result.status != "non_result"]
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _top_counter(counter: Counter[str], *, limit: int = 25) -> list[dict[str, object]]:
    return [{"value": key, "count": count} for key, count in counter.most_common(limit)]


def _artifact_diagnostics(result: ExperimentResult) -> dict[str, object]:
    path = Path(result.artifact)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": result.name, "artifact": result.artifact, "error": f"{type(exc).__name__}: {exc}"}

    folds = payload.get("folds") or []
    trades = payload.get("trades") or []
    selected_params: Counter[str] = Counter()
    skipped_models: Counter[str] = Counter()
    fitted_models: Counter[str] = Counter()
    model_weights: dict[str, list[float]] = defaultdict(list)
    model_utilities: dict[str, list[float]] = defaultdict(list)
    model_briers: dict[str, list[float]] = defaultdict(list)
    fold_rows: list[dict[str, object]] = []

    for fold in folds:
        for model in fold.get("fitted_models") or []:
            fitted_models[str(model)] += 1
        for model, reason in (fold.get("skipped_models") or {}).items():
            skipped_models[f"{model}: {reason}"] += 1
        for model, params in (fold.get("selected_model_params") or {}).items():
            selected_params[f"{model}:{_json_key(params)}"] += 1
        for model, diagnostics in (fold.get("model_diagnostics") or {}).items():
            if not isinstance(diagnostics, dict):
                continue
            model_weights[model].append(_safe_float(diagnostics.get("validation_weight")))
            model_utilities[model].append(_safe_float(diagnostics.get("validation_utility")))
            model_briers[model].append(_safe_float(diagnostics.get("validation_brier")))
        fold_rows.append(
            {
                "fold_id": fold.get("fold_id"),
                "train_samples": fold.get("train_samples"),
                "calibration_samples": fold.get("calibration_samples"),
                "test_samples": fold.get("test_samples"),
                "traded_signals": fold.get("traded_signals"),
                "net_pnl": fold.get("net_pnl"),
                "trade_hit_rate": fold.get("trade_hit_rate"),
                "average_trade_return": fold.get("average_trade_return"),
                "brier_score": fold.get("brier_score"),
                "log_loss": fold.get("log_loss"),
                "selected_threshold": fold.get("selected_threshold"),
                "selected_threshold_source": fold.get("selected_threshold_source"),
            }
        )

    trade_by_candidate: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "pnl": 0.0})
    trade_by_side: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "pnl": 0.0})
    trade_by_outcome: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "pnl": 0.0})
    fold_pnl = [_safe_float(fold.get("net_pnl")) for fold in folds]
    fold_trades = [_safe_int(fold.get("traded_signals")) for fold in folds]
    for trade in trades:
        pnl = _safe_float(trade.get("pnl"))
        for key, grouped in (
            (str(trade.get("candidate_type") or "unknown"), trade_by_candidate),
            (str(trade.get("side") or "unknown"), trade_by_side),
            (str(trade.get("outcome_type") or "unknown"), trade_by_outcome),
        ):
            grouped[key]["trades"] += 1.0
            grouped[key]["pnl"] += pnl

    def summarize_metric(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": float(len(values)),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    model_summary = {
        model: {
            "folds": float(len(model_weights[model])),
            "validation_weight": summarize_metric(model_weights[model]),
            "validation_utility": summarize_metric(model_utilities[model]),
            "validation_brier": summarize_metric(model_briers[model]),
        }
        for model in sorted(model_weights)
    }
    total_pnl = abs(_safe_float((payload.get("backtest_summary") or {}).get("net_pnl")))
    largest_candidate_pnl = max((abs(row["pnl"]) for row in trade_by_candidate.values()), default=0.0)
    return {
        "name": result.name,
        "artifact": result.artifact,
        "summary": asdict(result),
        "data_coverage": payload.get("data_coverage") or {},
        "candidate_type_summary": payload.get("candidate_type_summary") or {},
        "regime_summary": payload.get("regime_summary") or {},
        "rejection_reasons": payload.get("rejection_reasons") or {},
        "probability_distribution": payload.get("probability_distribution") or {},
        "expected_value_distribution": payload.get("expected_value_distribution") or {},
        "selection_score_distribution": payload.get("selection_score_distribution") or {},
        "fold_stability": {
            "folds": len(folds),
            "positive_pnl_folds": sum(1 for value in fold_pnl if value > 0),
            "negative_pnl_folds": sum(1 for value in fold_pnl if value < 0),
            "folds_with_trades": sum(1 for value in fold_trades if value > 0),
            "min_fold_pnl": min(fold_pnl, default=0.0),
            "max_fold_pnl": max(fold_pnl, default=0.0),
            "min_fold_trades": min(fold_trades, default=0),
            "max_fold_trades": max(fold_trades, default=0),
        },
        "folds": fold_rows,
        "fitted_model_counts": dict(fitted_models),
        "skipped_model_counts": dict(skipped_models),
        "selected_param_counts": _top_counter(selected_params, limit=50),
        "model_validation_summary": model_summary,
        "trade_by_candidate_type": dict(sorted(trade_by_candidate.items())),
        "trade_by_side": dict(sorted(trade_by_side.items())),
        "trade_by_outcome": dict(sorted(trade_by_outcome.items())),
        "concentration": {
            "largest_abs_candidate_type_pnl_share": largest_candidate_pnl / max(total_pnl, 1e-9),
        },
    }


def write_research_report(results: Iterable[ExperimentResult], output_path: Path) -> None:
    rows = list(results)
    ok = [row for row in rows if row.status == "ok"]
    failures = [row for row in rows if row.status != "ok"]

    def top(filtered: list[ExperimentResult], *, limit: int = 25) -> list[dict[str, object]]:
        ranked = sorted(filtered, key=lambda row: row.champion_score, reverse=True)[:limit]
        return [asdict(row) for row in ranked]

    positive = [row for row in ok if row.sharpe > 0 and row.net_pnl > 0]
    foundation = [row for row in ok if "foundation_kronos" in row.name]
    foundation_positive = [row for row in foundation if row.sharpe > 0 and row.net_pnl > 0]
    diagnostic_rows = sorted(ok, key=lambda row: row.champion_score, reverse=True)[:10]
    top_diagnostics = [_artifact_diagnostics(row) for row in diagnostic_rows]
    payload = {
        "total_results": len(rows),
        "ok_results": len(ok),
        "failed_results": len(failures),
        "failures": [asdict(row) for row in failures[:50]],
        "top_overall": top(ok),
        "top_positive": top(positive),
        "top_foundation_kronos": top(foundation),
        "top_foundation_kronos_positive": top(foundation_positive),
        "top_trade_frequency_positive": {
            "tpd_ge_0_5": top([row for row in positive if row.trades_per_day >= 0.5]),
            "tpd_ge_1": top([row for row in positive if row.trades_per_day >= 1.0]),
            "tpd_ge_2": top([row for row in positive if row.trades_per_day >= 2.0]),
            "tpd_ge_4": top([row for row in positive if row.trades_per_day >= 4.0]),
        },
        "top_diagnostics": top_diagnostics,
        "selection_guidance": {
            "minimum_research_bar": {
                "net_pnl": "> 0",
                "sharpe": "> 0",
                "profit_factor": "> 1",
                "max_drawdown": "inspect relative to return and baseline",
            },
            "strong_research_bar": {
                "trades_per_day": ">= 2 preferred, >= 4 target",
                "sharpe": ">= 0.75",
                "profit_factor": ">= 1.15",
                "fold_stability": "no single fold/month/candidate type should explain most profit",
                "cost_stress": "must remain positive under stress experiments before production consideration",
            },
            "final_model_bar": {
                "foundation_kronos": "preferred only when it beats the tree-only champion after costs",
                "profit_factor": ">= 1.15",
                "drawdown": "must be no worse than the tree-only champion",
            },
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


def _experiment_lookup(experiments: Iterable[ResearchExperiment]) -> dict[str, ResearchExperiment]:
    return {experiment.name: experiment for experiment in experiments}


def make_cost_stress_experiments(
    results: Iterable[ExperimentResult],
    experiments: Iterable[ResearchExperiment],
    *,
    top_n: int = 12,
) -> list[ResearchExperiment]:
    by_name = _experiment_lookup(experiments)
    candidates = [
        row
        for row in sorted(results, key=lambda item: item.champion_score, reverse=True)
        if row.status == "ok" and row.net_pnl > 0 and row.sharpe > 0
    ][:top_n]
    stressed: list[ResearchExperiment] = []
    for row in candidates:
        source = by_name.get(row.name)
        if source is None:
            continue
        stressed.append(
            replace(
                source,
                name=f"{source.name}_stress_cost",
                assumed_spread_bps=max(source.assumed_spread_bps, 8.0),
                base_slippage_bps=max(source.base_slippage_bps, 3.0),
                safety_margin_bps=max(source.safety_margin_bps, 4.0),
            )
        )
        stressed.append(
            replace(
                source,
                name=f"{source.name}_high_fee_stress",
                tier_rate=max(source.tier_rate, 0.0006),
                assumed_spread_bps=max(source.assumed_spread_bps, 6.0),
                base_slippage_bps=max(source.base_slippage_bps, 2.0),
                safety_margin_bps=max(source.safety_margin_bps, 3.0),
            )
        )
    return stressed


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
                    elapsed_seconds=round(time.time() - started, 3),
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
                    elapsed_seconds=round(elapsed, 3),
                    error=f"elapsed_seconds={elapsed:.1f}",
                    log_path=str(log_path),
                )
            )
            continue
        result = parse_signal_audit_artifact(artifact)
        result = ExperimentResult(
            **{**asdict(result), "elapsed_seconds": round(elapsed, 3), "log_path": str(log_path)}
        )
        results.append(result)
        print(
            f"{experiment.name}: trades={result.trades} tpd={result.trades_per_day:.3f} "
            f"pnl={result.net_pnl:.2f} sharpe={result.sharpe:.3f} "
            f"pf={result.profit_factor:.3f} elapsed={elapsed:.1f}s"
        )
    return sorted(results, key=lambda row: row.champion_score, reverse=True)
