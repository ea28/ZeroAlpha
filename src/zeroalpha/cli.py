"""ZeroAlpha command line interface."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, UTC
import json
import math
from pathlib import Path
import time

from zeroalpha.backtest.ml import run_ml_backtest, write_ml_backtest_artifact
from zeroalpha.backtest.simple import run_candidate_backtest, write_backtest_artifact
from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import load_config
from zeroalpha.data.external.binance import (
    BinanceFuturesMetricsClient,
    BinancePublicDataClient,
    fetch_futures_klines_archive_range,
    fetch_klines_archive_range,
)
from zeroalpha.data.external.coinbase import CoinbaseExchangeClient
from zeroalpha.data.external.ibkr_bars import read_ibkr_bars, write_ibkr_bars
from zeroalpha.data.external.ibkr_quotes import read_ibkr_quote_records
from zeroalpha.data.external.prediction_markets import (
    BTC_PREDICTION_MARKET_DURATIONS,
    load_prediction_market_snapshots,
)
from zeroalpha.data.health import health_checks_as_dict, run_external_data_health_checks
from zeroalpha.data.quality import interval_to_timedelta, validate_bars, validate_source_divergence
from zeroalpha.db.schema import initialize_sqlite
from zeroalpha.domain import RuntimeMode
from zeroalpha.models.dataset import (
    build_meta_label_samples,
    build_scoring_samples,
    label_geometry_diagnostics,
)
from zeroalpha.models.artifact import (
    fit_production_model_artifact,
    load_production_model_artifact,
    save_production_model_artifact,
    score_production_artifact,
)
from zeroalpha.models.ensemble import (
    SelectionExecutionPolicy,
    report_candidate_type_summary,
    report_feature_importance_summary,
    report_model_family_summary,
    report_native_importance_summary,
    report_shap_importance_summary,
    report_side_summary,
    run_meta_label_walk_forward,
    smoke_test_model_stack,
    write_meta_label_report,
)
from zeroalpha.models.sweep import run_label_geometry_sweep, sweep_results_asdict
from zeroalpha.monitoring.events import RuntimeEventStream


def _cmd_config_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"ok: mode={cfg.runtime.mode.value} broker={cfg.broker.host}:{cfg.broker.port}")
    return 0


def _cmd_binance_url(args: argparse.Namespace) -> int:
    client = BinancePublicDataClient()
    if args.month:
        print(client.monthly_klines_url(args.symbol, args.interval, args.month))
    else:
        raise SystemExit("--month is required for now")
    return 0


def _cmd_data_health_check(args: argparse.Namespace) -> int:
    checks = run_external_data_health_checks(cache_dir=Path(args.cache_dir))
    print(json.dumps(health_checks_as_dict(checks), indent=2, sort_keys=True))
    mandatory = {"binance_archive", "coinbase_candles", "kraken_ohlc"}
    return 0 if all(check.ok for check in checks if check.source in mandatory) else 1


def _date_range_from_args(args: argparse.Namespace) -> tuple[datetime, datetime]:
    end = (
        datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if args.end
        else datetime.now(tz=UTC)
    )
    start = (
        datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        if args.start
        else end - timedelta(days=365 * args.years)
    )
    return start, end


def _coinbase_reference_products(raw: str, interval: str) -> list[str]:
    value = raw.strip()
    if value.lower() in {"", "none", "off", "false"}:
        return []
    if value.lower() == "auto":
        try:
            _interval_to_coinbase_granularity(interval)
        except ValueError:
            return []
        return ["BTC-USD"]
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _csv_values(raw: str) -> list[str]:
    value = raw.strip()
    if value.lower() in {"", "none", "off", "false"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _importance_scoring_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    raw = getattr(args, "importance_scoring", "brier,log_loss") or "brier,log_loss"
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _named_paths(raw: str) -> dict[str, Path]:
    named: dict[str, Path] = {}
    for idx, item in enumerate(_csv_values(raw), start=1):
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = f"CONTEXT_{idx}", item
        name = name.strip().upper()
        if not name:
            raise ValueError(f"empty context name in {item!r}")
        named[name] = Path(path.strip())
    return named


def _quality_or_raise(report, *, label: str, allow_data_gaps: bool = False) -> dict[str, object]:
    payload = report.as_dict()
    if not report.ok:
        issue_codes = {issue.code for issue in report.issues}
        if allow_data_gaps and issue_codes <= {"bar_gap"}:
            payload["accepted_with_issues"] = True
            payload["accepted_issue_codes"] = sorted(issue_codes)
            return payload
        issue_preview = "; ".join(
            f"{issue.code}@{issue.timestamp_utc.isoformat() if issue.timestamp_utc else 'n/a'}"
            for issue in report.issues[:5]
        )
        raise ValueError(f"{label} data quality gate failed: {issue_preview}")
    return payload


def _context_quality_or_raise(report, *, label: str, allow_data_gaps: bool = False) -> dict[str, object]:
    payload = report.as_dict()
    if not report.ok:
        issue_codes = {issue.code for issue in report.issues}
        if allow_data_gaps and issue_codes <= {"bar_gap", "insufficient_coverage"}:
            payload["accepted_with_issues"] = True
            payload["accepted_issue_codes"] = sorted(issue_codes)
            return payload
        return _quality_or_raise(report, label=label, allow_data_gaps=allow_data_gaps)
    return payload


def _quality_payload(report) -> dict[str, object]:
    return report.as_dict()


def _load_research_bars(args: argparse.Namespace, start: datetime, end: datetime) -> tuple[list, dict, dict]:
    cache_dir = Path(args.cache_dir)
    primary_bars_path = getattr(args, "primary_bars_jsonl", "") or ""
    if primary_bars_path:
        primary_bars = [
            bar
            for bar in read_ibkr_bars(Path(primary_bars_path))
            if start <= bar.timestamp_utc < end
        ]
        primary_source = "IBKR_JSONL"
    elif getattr(args, "primary_market", "binance_spot") == "binance_um_futures":
        primary_bars = fetch_futures_klines_archive_range(
            symbol=args.symbol,
            interval=args.interval,
            start=start,
            end=end,
            cache_dir=cache_dir / "futures" / "um" / args.symbol.upper(),
            market_type="um",
        )
        primary_source = "BINANCE_UM_FUTURES"
    else:
        primary_bars = fetch_klines_archive_range(
            symbol=args.symbol,
            interval=args.interval,
            start=start,
            end=end,
            cache_dir=cache_dir / args.symbol.upper(),
        )
        primary_source = "BINANCE"
    context_bars = {}
    coverage = {
        "primary": {
            "source": primary_source,
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "bars": len(primary_bars),
            "path": primary_bars_path,
        },
        "context": {},
    }
    primary_quality = validate_bars(
        primary_bars,
        expected_interval=args.interval,
        start=start,
        end=end,
        minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
        max_return_bps=getattr(args, "max_bar_return_bps", 0.0) or None,
    )
    allow_data_gaps = bool(getattr(args, "allow_data_gaps", False))
    coverage["primary"]["quality"] = _quality_or_raise(
        primary_quality,
        label=f"primary {args.symbol}",
        allow_data_gaps=allow_data_gaps,
    )
    context_interval = getattr(args, "context_interval", "") or args.interval
    for context_symbol in _csv_values(args.context_symbols):
        bars = fetch_klines_archive_range(
            symbol=context_symbol,
            interval=context_interval,
            start=start,
            end=end,
            cache_dir=cache_dir / context_symbol.upper(),
        )
        key = context_symbol.upper()
        context_bars[key] = bars
        context_quality = validate_bars(
            bars,
            expected_interval=context_interval,
            start=start,
            end=end,
            minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
            max_return_bps=getattr(args, "max_bar_return_bps", 0.0) or None,
        )
        coverage["context"][key] = {
            "source": "BINANCE",
            "interval": context_interval,
            "bars": len(bars),
            "quality": _context_quality_or_raise(
                context_quality,
                label=f"context {key}",
                allow_data_gaps=allow_data_gaps,
            ),
        }
    for futures_symbol in _csv_values(getattr(args, "binance_um_futures_reference_symbols", "")):
        bars = fetch_futures_klines_archive_range(
            symbol=futures_symbol,
            interval=context_interval,
            start=start,
            end=end,
            cache_dir=cache_dir / "futures" / "um" / futures_symbol.upper(),
            market_type="um",
        )
        key = f"BINANCE_UM_{futures_symbol.upper()}"
        if bars:
            context_bars[key] = bars
        context_quality = validate_bars(
            bars,
            expected_interval=context_interval,
            start=start,
            end=end,
            minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
            max_return_bps=getattr(args, "max_bar_return_bps", 0.0) or None,
        )
        coverage["context"][key] = {
            "source": "BINANCE_UM_FUTURES",
            "interval": context_interval,
            "bars": len(bars),
            "quality": _context_quality_or_raise(
                context_quality,
                label=f"context {key}",
                allow_data_gaps=allow_data_gaps,
            ),
        }
    if getattr(args, "binance_um_derivatives_metrics", False):
        metrics_symbol = (getattr(args, "binance_um_metrics_symbol", "") or "").strip().upper()
        if not metrics_symbol:
            metrics_symbol = args.symbol.upper() if args.symbol.upper().endswith("USDT") else "BTCUSDT"
        metrics_client = BinanceFuturesMetricsClient()
        oi_period = getattr(args, "binance_um_open_interest_period", "5m")
        metrics_period = getattr(args, "binance_um_metrics_period", "5m")
        metric_loaders = [
            (
                f"BINANCE_UM_OPEN_INTEREST_{metrics_symbol}",
                lambda: metrics_client.fetch_open_interest_history(
                    symbol=metrics_symbol,
                    period=oi_period,
                    start=start,
                    end=end,
                ),
                oi_period,
            ),
            (
                f"BINANCE_UM_FUNDING_{metrics_symbol}",
                lambda: metrics_client.fetch_funding_rates(
                    symbol=metrics_symbol,
                    start=start,
                    end=end,
                ),
                "8h",
            ),
        ]
        if getattr(args, "binance_um_taker_flow", False):
            metric_loaders.append(
                (
                    f"BINANCE_UM_TAKERFLOW_{metrics_symbol}",
                    lambda: metrics_client.fetch_taker_buy_sell_volume(
                        symbol=metrics_symbol,
                        period=metrics_period,
                        start=start,
                        end=end,
                    ),
                    metrics_period,
                )
            )
        if getattr(args, "binance_um_basis", False):
            metric_loaders.append(
                (
                    f"BINANCE_UM_BASIS_{metrics_symbol}",
                    lambda: metrics_client.fetch_basis_history(
                        pair=metrics_symbol,
                        period=metrics_period,
                        start=start,
                        end=end,
                    ),
                    metrics_period,
                )
            )
        for key, loader, expected_interval in metric_loaders:
            try:
                bars = loader()
            except Exception as exc:
                coverage["context"][key] = {
                    "source": key.rsplit("_", 2)[0],
                    "bars": 0,
                    "used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            if bars:
                context_bars[key] = bars
            context_quality = validate_bars(
                bars,
                expected_interval=expected_interval,
                start=start,
                end=end,
                minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
                max_return_bps=None,
            )
            coverage["context"][key] = {
                "source": bars[0].source if bars else key,
                "interval": expected_interval,
                "bars": len(bars),
                "quality": _context_quality_or_raise(
                    context_quality,
                    label=f"context {key}",
                    allow_data_gaps=allow_data_gaps,
                ),
            }
    for key, path in _named_paths(getattr(args, "context_bars_jsonl", "")).items():
        bars = [
            bar
            for bar in read_ibkr_bars(path)
            if start <= bar.timestamp_utc < end
        ]
        if bars:
            context_bars[key] = bars
        context_quality = validate_bars(
            bars,
            expected_interval=context_interval,
            start=start,
            end=end,
            minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
            max_return_bps=getattr(args, "max_bar_return_bps", 0.0) or None,
        )
        coverage["context"][key] = {
            "source": "IBKR_JSONL",
            "interval": context_interval,
            "bars": len(bars),
            "path": str(path),
            "quality": _context_quality_or_raise(
                context_quality,
                label=f"context {key}",
                allow_data_gaps=allow_data_gaps,
            ),
        }
    coinbase_products = _coinbase_reference_products(args.coinbase_reference_products, context_interval)
    if coinbase_products:
        coinbase = CoinbaseExchangeClient()
        granularity = _interval_to_coinbase_granularity(context_interval)
        for product_id in coinbase_products:
            bars = coinbase.fetch_candles_range(product_id, granularity, start, end)
            key = f"COINBASE_{product_id}"
            cb_interval = f"{granularity}s"
            context_quality = validate_bars(
                bars,
                expected_interval=cb_interval,
                start=start,
                end=end,
                minimum_coverage_ratio=getattr(args, "minimum_data_coverage", 0.0),
                max_return_bps=getattr(args, "max_bar_return_bps", 0.0) or None,
            )
            divergence = validate_source_divergence(
                primary_bars,
                bars,
                max_divergence_bps=getattr(args, "max_source_divergence_bps", 500.0),
                expected_interval=args.interval,
            )
            used = context_quality.ok and divergence.ok
            if used:
                context_bars[key] = bars
            coverage["context"][key] = {
                "source": "COINBASE",
                "interval": cb_interval,
                "bars": len(bars),
                "used": used,
                "quality": _quality_payload(context_quality),
                "divergence": _quality_payload(divergence),
            }
    return primary_bars, context_bars, coverage


def _ibkr_bar_size_interval(bar_size: str, fallback: str) -> str:
    parts = bar_size.strip().lower().split()
    if len(parts) >= 2 and parts[0].isdigit():
        value = parts[0]
        unit = parts[1]
        if unit.startswith("sec"):
            return f"{value}s"
        if unit.startswith("min"):
            return f"{value}m"
        if unit.startswith("hour"):
            return f"{value}h"
        if unit.startswith("day"):
            return f"{value}d"
    return fallback


def _load_optional_replay_bars(
    path_raw: str,
    *,
    start: datetime,
    end: datetime,
    fallback_interval: str,
    allow_data_gaps: bool,
    minimum_data_coverage: float,
    label: str,
) -> tuple[list, dict[str, object]]:
    path_raw = path_raw.strip()
    if not path_raw:
        return [], {"enabled": False}
    path = Path(path_raw)
    bars = [
        bar
        for bar in read_ibkr_bars(path)
        if start <= bar.timestamp_utc < end
    ]
    expected_interval = (
        _ibkr_bar_size_interval(bars[0].bar_size, fallback_interval)
        if bars
        else fallback_interval
    )
    quality = validate_bars(
        bars,
        expected_interval=expected_interval,
        start=start,
        end=end,
        minimum_coverage_ratio=minimum_data_coverage,
        max_return_bps=None,
    )
    source_counts = Counter(str(bar.source) for bar in bars)
    what_to_show_counts = Counter(str(bar.extra.get("what_to_show", "")) for bar in bars)
    aggregated_from_counts = Counter(str(bar.extra.get("aggregated_from", "")) for bar in bars)
    tick_backed_bars = sum(
        1
        for bar in bars
        if (
            str(bar.source).upper().startswith("IBKR")
            and (
                str(bar.extra.get("what_to_show", "")).upper() in {"AGGTRADES", "TRADES"}
                or str(bar.extra.get("aggregated_from", "")).startswith("streaming_tick_by_tick")
                or float(bar.extra.get("tick_count", 0.0) or 0.0) > 0
                or (bar.trade_count is not None and bar.trade_count > 0)
            )
        )
    )
    quote_only_bars = sum(
        1
        for bar in bars
        if "quote" in str(bar.extra.get("aggregated_from", "")).lower()
        and not str(bar.extra.get("aggregated_from", "")).startswith("streaming_tick_by_tick")
    )
    return bars, {
        "enabled": True,
        "source": "IBKR_JSONL",
        "path": str(path),
        "bars": len(bars),
        "interval": expected_interval,
        "provenance": {
            "source_counts": dict(sorted(source_counts.items())),
            "what_to_show_counts": dict(sorted(what_to_show_counts.items())),
            "aggregated_from_counts": dict(sorted(aggregated_from_counts.items())),
            "tick_backed_bars": tick_backed_bars,
            "quote_only_bars": quote_only_bars,
        },
        "quality": _context_quality_or_raise(
            quality,
            label=label,
            allow_data_gaps=allow_data_gaps,
        ),
    }


def _requested_window_coverage(start: datetime, end: datetime) -> dict[str, object]:
    span_days = max((end - start).total_seconds() / 86_400, 0.0)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "span_days": span_days,
    }


def _load_label_execution_bars(
    args: argparse.Namespace,
    *,
    start: datetime,
    end: datetime,
) -> tuple[list, list, dict[str, object], dict[str, object]]:
    allow_data_gaps = bool(getattr(args, "allow_data_gaps", False))
    label_bars, label_bar_coverage = _load_optional_replay_bars(
        getattr(args, "label_bars_jsonl", ""),
        start=start,
        end=end,
        fallback_interval=args.interval,
        allow_data_gaps=allow_data_gaps,
        minimum_data_coverage=getattr(args, "minimum_data_coverage", 0.0),
        label="label bars",
    )
    execution_bars, execution_bar_coverage = _load_optional_replay_bars(
        getattr(args, "execution_bars_jsonl", ""),
        start=start,
        end=end,
        fallback_interval=args.interval,
        allow_data_gaps=allow_data_gaps,
        minimum_data_coverage=getattr(args, "minimum_data_coverage", 0.0),
        label="execution bars",
    )
    if not execution_bars and label_bars:
        execution_bars = label_bars
        execution_bar_coverage = {
            **label_bar_coverage,
            "source_alias": "label_bars_jsonl",
        }
    return label_bars, execution_bars, label_bar_coverage, execution_bar_coverage


def _interval_seconds_or_none(interval: object) -> float | None:
    if not interval:
        return None
    try:
        return interval_to_timedelta(str(interval)).total_seconds()
    except ValueError:
        return None


def _prediction_market_durations_from_args(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "prediction_market_durations", "") or ",".join(BTC_PREDICTION_MARKET_DURATIONS)
    return [value.strip().lower() for value in raw.split(",") if value.strip()]


def _load_prediction_market_signals(args: argparse.Namespace, start: datetime, end: datetime):
    if not getattr(args, "prediction_market_signals", False):
        return [], {"enabled": False}
    lookback_days = getattr(args, "prediction_market_lookback_days", 0)
    fetch_start = max(start, end - timedelta(days=lookback_days)) if lookback_days else start
    result = load_prediction_market_snapshots(
        start=fetch_start,
        end=end,
        durations=_prediction_market_durations_from_args(args),
        cache_dir=Path(getattr(args, "prediction_market_cache_dir", "data/raw/prediction_markets")),
        max_markets=getattr(args, "prediction_market_max_markets", 500),
        fidelity_minutes=getattr(args, "prediction_market_fidelity_minutes", 1),
        refresh=getattr(args, "refresh_prediction_market_cache", False),
    )
    return result.snapshots, {
        **result.coverage,
        "enabled": True,
        "feature_profile": getattr(args, "prediction_market_feature_profile", "full"),
        "feature_asof": getattr(args, "feature_asof", "signal"),
        "requested_start": start.isoformat(),
        "fetch_start": fetch_start.isoformat(),
        "end": end.isoformat(),
    }


def _load_ibkr_quote_records(
    args: argparse.Namespace,
    start: datetime,
    end: datetime,
    *,
    attr: str = "ibkr_quote_records",
    symbol_contains: str | None = "BTC",
):
    raw_path = getattr(args, attr, "") or ""
    if not raw_path:
        return [], {"enabled": False}
    path = Path(raw_path)
    quotes = read_ibkr_quote_records(path)
    filtered = [
        quote
        for quote in quotes
        if start <= quote.timestamp_utc <= end
        and (symbol_contains is None or symbol_contains.upper() in quote.symbol.upper())
    ]
    average_spread_bps = (
        sum(quote.spread_bps for quote in filtered) / len(filtered) if filtered else 0.0
    )
    return filtered, {
        "enabled": True,
        "path": str(path),
        "records": len(quotes),
        "used_records": len(filtered),
        "average_spread_bps": average_spread_bps,
        "start": filtered[0].timestamp_utc.isoformat() if filtered else "",
        "end": filtered[-1].timestamp_utc.isoformat() if filtered else "",
    }


def _override_config_from_args(cfg, args: argparse.Namespace):
    if getattr(args, "instrument_model", ""):
        cfg = replace(cfg, contract=replace(cfg.contract, instrument_model=args.instrument_model))
    label_updates = {}
    if getattr(args, "max_holding_hours", 0):
        label_updates["max_holding_hours"] = args.max_holding_hours
    if getattr(args, "max_holding_seconds", 0.0):
        label_updates["max_holding_seconds"] = args.max_holding_seconds
    if getattr(args, "net_profit_target", 0.0):
        label_updates["net_profit_target"] = args.net_profit_target
    if getattr(args, "net_stop_loss", 0.0):
        label_updates["net_stop_loss"] = args.net_stop_loss
    for attr in (
        "volatility_lookback_bars",
        "profit_volatility_multiplier",
        "stop_volatility_multiplier",
        "minimum_gross_profit_bps",
        "minimum_gross_stop_bps",
    ):
        value = getattr(args, attr, 0)
        if value:
            label_updates[attr] = value
    model_updates = {}
    if hasattr(args, "minimum_probability") and args.minimum_probability is not None:
        model_updates["minimum_probability"] = args.minimum_probability
    if getattr(args, "minimum_expected_value", None) is not None:
        model_updates["minimum_expected_value"] = args.minimum_expected_value
    if getattr(args, "calibration_method", ""):
        model_updates["calibration_method"] = args.calibration_method
    risk_updates = {}
    for attr in (
        "risk_per_trade",
        "daily_loss_stop",
        "weekly_loss_stop",
        "rolling_drawdown_stop",
        "paper_max_notional",
        "minimum_fee_efficient_notional",
        "max_open_positions",
    ):
        value = getattr(args, attr, 0.0)
        if value:
            risk_updates[attr] = value
    if getattr(args, "consecutive_loss_limit", None) is not None:
        risk_updates["consecutive_loss_limit"] = args.consecutive_loss_limit
    if getattr(args, "cooldown_hours_after_stopouts", None) is not None:
        risk_updates["cooldown_hours_after_stopouts"] = args.cooldown_hours_after_stopouts
    if label_updates:
        cfg = replace(cfg, labels=replace(cfg.labels, **label_updates))
    if model_updates:
        cfg = replace(cfg, model=replace(cfg.model, **model_updates))
    if risk_updates:
        cfg = replace(cfg, risk=replace(cfg.risk, **risk_updates))
    cost_updates = {}
    for attr in (
        "tier_rate",
        "minimum_commission",
        "maximum_commission_rate",
        "base_slippage_bps",
        "safety_margin_bps",
        "futures_fee_per_contract",
        "futures_contract_multiplier",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            cost_updates[attr] = value
    if cost_updates:
        cfg = replace(cfg, cost=replace(cfg.cost, **cost_updates))
    kronos_updates = {}
    if getattr(args, "kronos_features", False):
        kronos_updates["enabled"] = True
    if getattr(args, "kronos_mode", ""):
        kronos_updates["mode"] = args.kronos_mode
    if getattr(args, "kronos_lookback_bars", 0):
        kronos_updates["lookback_bars"] = args.kronos_lookback_bars
    if getattr(args, "kronos_embedding_dims", 0):
        kronos_updates["embedding_dims"] = args.kronos_embedding_dims
    if getattr(args, "kronos_device", ""):
        kronos_updates["device"] = args.kronos_device
    if kronos_updates:
        cfg = replace(cfg, kronos=replace(cfg.kronos, **kronos_updates))
    cfg.validate()
    return cfg


def _candidate_config_from_args(args: argparse.Namespace, cfg) -> CandidateGenerationConfig:
    mode = getattr(args, "candidate_mode", "rules")
    candidate_mode = {
        "dense": "dense_research",
        "aggressive": "aggressive_rules",
        "active": "active_research",
    }.get(mode, "rules")
    allow_short_research = (
        cfg.contract.instrument_model == "futures"
        or bool(getattr(args, "allow_spot_short_research", False))
    )
    return CandidateGenerationConfig(
        max_holding_hours=cfg.labels.max_holding_hours,
        max_holding_seconds=cfg.labels.max_holding_seconds,
        min_history_bars=getattr(args, "min_history_bars", 240),
        lookback=getattr(args, "candidate_lookback_bars", 24),
        rolling_window_bars=getattr(args, "candidate_rolling_window_bars", 500),
        mode=candidate_mode,
        dense_stride_bars=getattr(args, "dense_stride_bars", 1),
        side_mode=getattr(args, "side_mode", "long"),
        allow_short_research=allow_short_research,
        adaptive_horizon=bool(getattr(args, "adaptive_horizon", False)),
        min_holding_seconds=getattr(args, "min_holding_seconds", 1.0),
        adaptive_horizon_granularity_seconds=(
            getattr(args, "adaptive_horizon_granularity_seconds", 0.0) or None
        ),
        adaptive_horizon_max_seconds=(
            getattr(args, "adaptive_horizon_max_seconds", 0.0) or None
        ),
        adaptive_horizon_target_move_bps=_adaptive_horizon_target_move_bps(args, cfg),
    )


def _adaptive_horizon_target_move_bps(
    args: argparse.Namespace,
    cfg,
    *,
    assumed_spread_bps: float | None = None,
    research_notional: float | None = None,
    reference_price: float | None = None,
) -> float:
    explicit = float(getattr(args, "adaptive_horizon_target_move_bps", 0.0) or 0.0)
    if explicit > 0:
        return explicit
    spread_bps = (
        float(assumed_spread_bps)
        if assumed_spread_bps is not None
        else float(getattr(args, "assumed_spread_bps", 0.0) or 0.0)
    )
    notional = (
        research_notional
        if research_notional is not None
        else (
            getattr(args, "notional", None)
            or getattr(args, "max_order_notional_usd", None)
            or cfg.risk.paper_max_notional
        )
    )
    diagnostics = label_geometry_diagnostics(
        config=cfg,
        assumed_spread_bps=spread_bps,
        research_notional=float(notional),
        reference_price=reference_price,
    )
    gross_profit_bps = diagnostics.gross_profit_move * 10_000
    return max(
        50.0,
        gross_profit_bps * 4.0,
        diagnostics.round_trip_cost_bps * 20.0,
    )


def _filter_samples_from_args(samples, args: argparse.Namespace):
    raw = getattr(args, "candidate_types", "").strip()
    filtered = samples
    if raw:
        allowed = {value.strip() for value in raw.split(",") if value.strip()}
        filtered = [sample for sample in filtered if sample.candidate_type in allowed]
        if samples and not filtered:
            available = ", ".join(sorted({sample.candidate_type for sample in samples}))
            requested = ", ".join(sorted(allowed))
            raise SystemExit(
                f"--candidate-types matched no samples. Requested: {requested}. Available: {available}"
            )
    setup_raw = getattr(args, "setup_families", "").strip()
    exclude_setup_raw = getattr(args, "exclude_setup_families", "").strip()
    if setup_raw:
        allowed_setups = {value.strip() for value in setup_raw.split(",") if value.strip()}
        filtered = [
            sample
            for sample in filtered
            if _sample_matches_setup_family_filter(sample, allowed=allowed_setups, excluded=set())
        ]
        if samples and not filtered:
            available = ", ".join(sorted({_sample_setup_family(sample) for sample in samples}))
            requested = ", ".join(sorted(allowed_setups))
            raise SystemExit(
                f"--setup-families matched no samples. Requested: {requested}. Available: {available}"
            )
    if exclude_setup_raw:
        excluded_setups = {value.strip() for value in exclude_setup_raw.split(",") if value.strip()}
        filtered = [
            sample
            for sample in filtered
            if _sample_matches_setup_family_filter(sample, allowed=set(), excluded=excluded_setups)
        ]
        if samples and not filtered:
            requested = ", ".join(sorted(excluded_setups))
            raise SystemExit(f"--exclude-setup-families removed every sample. Excluded: {requested}")
    if getattr(args, "require_prediction_market_data", False):
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_available_count", 0.0)) > 0
        ]
    if getattr(args, "require_leading_prediction_market_data", False):
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_leading_available_count", 0.0)) > 0
        ]
    min_pm_count = getattr(args, "prediction_market_min_available_count", 0)
    if min_pm_count:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_available_count", 0.0)) >= min_pm_count
        ]
    min_side_mid = getattr(args, "prediction_market_min_side_mid", 0.0)
    if min_side_mid:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_side_aligned_mid_max", 0.0)) >= min_side_mid
        ]
    min_lead_seconds = getattr(args, "prediction_market_min_lead_seconds", 0.0)
    if min_lead_seconds:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_seconds_to_close_max", 0.0)) >= min_lead_seconds
        ]
    min_leading_side_mid = getattr(args, "prediction_market_min_leading_side_mid", 0.0)
    if min_leading_side_mid:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_leading_side_aligned_mid_max", 0.0)) >= min_leading_side_mid
        ]
    min_residual_edge = getattr(args, "prediction_market_min_leading_residual_edge", None)
    if min_residual_edge is not None:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_leading_residual_edge_max", -1.0)) >= min_residual_edge
        ]
    min_liquidity_weight = getattr(args, "prediction_market_min_leading_liquidity_weight", 0.0)
    if min_liquidity_weight:
        filtered = [
            sample
            for sample in filtered
            if float(sample.features.get("pm_leading_liquidity_weight_total", 0.0)) >= min_liquidity_weight
        ]
    if samples and not filtered:
        raise SystemExit("prediction-market/sample filters removed every sample")
    return filtered


def _sample_setup_family(sample) -> str:
    for key in ("event_setup_family", "event_dense_setup_family", "setup_family", "dense_setup_family"):
        value = sample.features.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sample_specialist_setup_family(sample) -> str:
    for key in (
        "event_specialist_setup_family",
        "specialist_setup_family",
        "event_setup_specialist_family",
        "setup_specialist_family",
    ):
        value = sample.features.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sample_setup_family_values(sample) -> set[str]:
    return {
        value
        for value in (
            _sample_setup_family(sample),
            _sample_specialist_setup_family(sample),
        )
        if value and value != "unknown"
    }


def _sample_matches_setup_family_filter(sample, *, allowed: set[str], excluded: set[str]) -> bool:
    values = _sample_setup_family_values(sample) or {_sample_setup_family(sample)}
    return (not allowed or bool(values & allowed)) and not bool(values & excluded)


def _sample_span_days(samples) -> float:
    if len(samples) < 2:
        return 1.0
    ordered = sorted(sample.timestamp_utc for sample in samples)
    return max((ordered[-1] - ordered[0]).total_seconds() / 86_400, 1 / 24)


def _coverage_span_days(coverage: dict[str, object]) -> float:
    span = coverage.get("span_days")
    if isinstance(span, int | float) and math.isfinite(float(span)):
        return max(float(span), 0.0)
    start = coverage.get("start")
    end = coverage.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return 0.0
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max((end_dt - start_dt).total_seconds() / 86_400, 0.0)


def _replay_coverage_tick_backed_ratio(coverage: dict[str, object]) -> float:
    bars = _finite_number(coverage.get("bars")) or 0.0
    provenance = coverage.get("provenance")
    tick_backed = 0.0
    if isinstance(provenance, dict):
        tick_backed = _finite_number(provenance.get("tick_backed_bars")) or 0.0
    return tick_backed / bars if bars > 0 else 0.0


def _strict_live_valid_1s_diagnostics(
    *,
    args: argparse.Namespace,
    data_coverage: dict[str, object],
    label_bar_coverage: dict[str, object],
    execution_bar_coverage: dict[str, object],
    folds: int,
) -> dict[str, object]:
    requested_window = data_coverage.get("requested_window")
    window_days = _coverage_span_days(requested_window if isinstance(requested_window, dict) else {})
    min_days = float(getattr(args, "min_live_valid_1s_days", 0.0) or 0.0)
    preferred_days = float(getattr(args, "preferred_live_valid_1s_days", 0.0) or 0.0)
    min_folds = int(getattr(args, "min_live_valid_folds", 0) or 0)
    label_interval = str(label_bar_coverage.get("interval") or "")
    execution_interval = str(execution_bar_coverage.get("interval") or "")
    label_tick_ratio = _replay_coverage_tick_backed_ratio(label_bar_coverage)
    execution_tick_ratio = _replay_coverage_tick_backed_ratio(execution_bar_coverage)
    errors: list[str] = []
    if label_interval != "1s":
        errors.append("label_bars_not_1s")
    if execution_interval != "1s":
        errors.append("execution_bars_not_1s")
    if not bool(label_bar_coverage.get("enabled")):
        errors.append("label_bars_missing")
    if not bool(execution_bar_coverage.get("enabled")):
        errors.append("execution_bars_missing")
    if label_tick_ratio < 0.95:
        errors.append("label_bars_not_tick_backed")
    if execution_tick_ratio < 0.95:
        errors.append("execution_bars_not_tick_backed")
    if min_days > 0 and window_days < min_days:
        errors.append("window_too_short")
    if min_folds > 0 and folds < min_folds:
        errors.append("too_few_walk_forward_folds")
    return {
        "ok": not errors,
        "errors": errors,
        "window_days": window_days,
        "minimum_days": min_days,
        "preferred_days": preferred_days,
        "folds": folds,
        "minimum_folds": min_folds,
        "label_interval": label_interval,
        "execution_interval": execution_interval,
        "label_tick_backed_ratio": label_tick_ratio,
        "execution_tick_backed_ratio": execution_tick_ratio,
        "label_provenance": label_bar_coverage.get("provenance", {}),
        "execution_provenance": execution_bar_coverage.get("provenance", {}),
    }


def _enforce_strict_live_valid_1s(
    *,
    args: argparse.Namespace,
    data_coverage: dict[str, object],
    label_bar_coverage: dict[str, object],
    execution_bar_coverage: dict[str, object],
    folds: int,
) -> None:
    diagnostics = _strict_live_valid_1s_diagnostics(
        args=args,
        data_coverage=data_coverage,
        label_bar_coverage=label_bar_coverage,
        execution_bar_coverage=execution_bar_coverage,
        folds=folds,
    )
    data_coverage["strict_live_valid_1s"] = diagnostics
    if getattr(args, "strict_live_valid_1s", False) and not diagnostics["ok"]:
        raise SystemExit(
            "strict live-valid 1s promotion gate failed: "
            + ",".join(str(value) for value in diagnostics["errors"])
        )


def _validate_research_short_backtest_args(args: argparse.Namespace) -> None:
    if getattr(args, "allow_spot_short_research", False) and not getattr(args, "research_gate", False):
        raise SystemExit("--allow-spot-short-research requires --research-gate")
    if getattr(args, "allow_research_short_backtest", False) and not getattr(args, "research_gate", False):
        raise SystemExit("--allow-research-short-backtest requires --research-gate")


def _validate_research_gated_args(args: argparse.Namespace) -> None:
    _validate_research_short_backtest_args(args)
    if (
        getattr(args, "capacity_release_mode", "planned") == "actual"
        and not getattr(args, "research_gate", False)
    ):
        raise SystemExit("--capacity-release-mode actual requires --research-gate")


def _effective_respect_open_positions(args: argparse.Namespace, cfg) -> bool:
    explicit = getattr(args, "respect_open_positions", None)
    if explicit is not None:
        return bool(explicit)
    return (
        getattr(args, "target_frequency_mode", "online") == "online"
        and getattr(cfg.risk, "max_open_positions", 0) > 0
    )


def _cmd_backtest_candidate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    end = (
        datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if args.end
        else datetime.now(tz=UTC)
    )
    start = (
        datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        if args.start
        else end - timedelta(days=365 * args.years)
    )
    summary, trades, rejections = run_candidate_backtest(
        config=cfg,
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        starting_equity=args.starting_equity,
        notional=args.notional,
        assumed_spread_bps=args.assumed_spread_bps,
        cache_dir=Path(args.cache_dir),
    )
    if args.output:
        write_backtest_artifact(Path(args.output), summary, trades, rejections)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _cmd_backtest_ml(args: argparse.Namespace) -> int:
    cfg = _override_config_from_args(load_config(args.config), args)
    _validate_research_gated_args(args)
    respect_open_positions = _effective_respect_open_positions(args, cfg)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    label_bars, execution_bars, label_bar_coverage, execution_bar_coverage = _load_label_execution_bars(
        args,
        start=start,
        end=end,
    )
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    futures_market_quotes, ibkr_futures_quote_coverage = _load_ibkr_quote_records(
        args,
        start,
        end,
        attr="ibkr_futures_quote_records",
        symbol_contains=None,
    )
    data_coverage["label_geometry"] = asdict(
        label_geometry_diagnostics(
            config=cfg,
            assumed_spread_bps=args.assumed_spread_bps,
            research_notional=args.notional,
            reference_price=primary_bars[0].close if primary_bars else None,
        )
    )
    data_coverage["requested_window"] = _requested_window_coverage(start, end)
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["label_bars"] = label_bar_coverage
    data_coverage["execution_bars"] = execution_bar_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    data_coverage["ibkr_futures_quotes"] = ibkr_futures_quote_coverage
    data_coverage["execution_research"] = _ml_execution_kwargs(args)
    candidate_config = _candidate_config_from_args(args, cfg)
    if (
        label_bars
        and candidate_config.adaptive_horizon
        and candidate_config.adaptive_horizon_granularity_seconds is None
    ):
        label_interval_seconds = _interval_seconds_or_none(label_bar_coverage.get("interval"))
        if label_interval_seconds is not None:
            candidate_config = replace(
                candidate_config,
                adaptive_horizon_granularity_seconds=label_interval_seconds,
            )
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        label_bars=label_bars or None,
        context_bars=context_bars,
        market_quotes=market_quotes,
        futures_market_quotes=futures_market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        prediction_market_feature_profile=args.prediction_market_feature_profile,
        feature_asof=args.feature_asof,
        external_feature_latency_seconds=args.external_feature_latency_seconds,
        candidate_config=candidate_config,
    )
    samples = _filter_samples_from_args(samples, args)
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    report = run_meta_label_walk_forward(
        samples,
        config=cfg,
        model_names=model_names,
        train_size=args.train_size or None,
        calibration_size=args.calibration_size or None,
        test_size=args.test_size or None,
        embargo_hours=args.embargo_hours or None,
        adaptive_threshold=args.adaptive_threshold,
        min_calibration_trades=args.min_calibration_trades,
        stacker_mode=args.stacker,
        adaptive_minimum_threshold=args.adaptive_minimum_threshold,
        tune_hyperparameters=args.hpo,
        hpo_profile=args.hpo_profile,
        hpo_trials=args.hpo_trials,
        foundation_max_samples=args.foundation_max_samples or 1024,
        data_coverage=data_coverage,
        target_trades_per_day=args.target_trades_per_day,
        allow_negative_ev_target_frequency=args.allow_negative_ev_frequency_probe,
        candidate_type_thresholds=args.candidate_type_thresholds,
        empirical_payoff_ev=args.empirical_payoff_ev,
        selection_score_mode=args.selection_score,
        target_frequency_mode=args.target_frequency_mode,
        selection_score_floor=args.selection_score_floor,
        selection_score_ceiling=args.selection_score_ceiling,
        adaptive_selection_score_floor=args.adaptive_selection_score_floor,
        adaptive_selection_score_direction=args.adaptive_selection_score_direction,
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
        respect_open_positions=respect_open_positions,
        capacity_release_mode=args.capacity_release_mode,
        selection_execution_policy=_selection_execution_policy_from_args(args),
        optimize_metric=args.optimize_metric,
        permutation_importance=args.permutation_importance,
        permutation_repeats=args.permutation_repeats,
        permutation_max_features=args.permutation_max_features,
        permutation_sample_limit=args.permutation_sample_limit,
        interpretability_top_n=args.interpretability_top_n,
        importance_scoring=_importance_scoring_from_args(args),
        permutation_grouping=args.permutation_grouping,
        shap_importance=args.shap_importance,
        shap_sample_limit=args.shap_sample_limit,
        shap_background_limit=args.shap_background_limit,
        shap_top_n=args.shap_top_n,
        shap_grouping=args.shap_grouping,
        feature_include_families=tuple(_csv_values(args.feature_include_groups)),
        feature_exclude_families=tuple(_csv_values(args.feature_exclude_groups)),
        feature_exclude_patterns=tuple(_csv_values(args.feature_exclude_patterns)),
    )
    _enforce_strict_live_valid_1s(
        args=args,
        data_coverage=data_coverage,
        label_bar_coverage=label_bar_coverage,
        execution_bar_coverage=execution_bar_coverage,
        folds=len(report.folds),
    )
    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=execution_bars or primary_bars,
        config=cfg,
        starting_equity=args.starting_equity,
        requested_notional=args.notional,
        assumed_spread_bps=args.assumed_spread_bps,
        entry_limit_offset_bps=args.entry_limit_offset_bps,
        enforce_production_gate=not args.research_gate,
        allow_negative_ev_research=args.allow_negative_ev_frequency_probe,
        allow_research_short_backtest=args.allow_research_short_backtest,
        confidence_scaled_sizing=args.confidence_scaled_sizing,
        selection_score_mode=args.selection_score,
        **_ml_execution_kwargs(args),
    )
    if args.output:
        write_ml_backtest_artifact(Path(args.output), summary, trades, rejections, report)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _cmd_model_train_meta(args: argparse.Namespace) -> int:
    cfg = _override_config_from_args(load_config(args.config), args)
    _validate_research_gated_args(args)
    respect_open_positions = _effective_respect_open_positions(args, cfg)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    label_bars, _execution_bars, label_bar_coverage, execution_bar_coverage = _load_label_execution_bars(
        args,
        start=start,
        end=end,
    )
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    futures_market_quotes, ibkr_futures_quote_coverage = _load_ibkr_quote_records(
        args,
        start,
        end,
        attr="ibkr_futures_quote_records",
        symbol_contains=None,
    )
    data_coverage["label_geometry"] = asdict(
        label_geometry_diagnostics(
            config=cfg,
            assumed_spread_bps=args.assumed_spread_bps,
            research_notional=args.notional,
            reference_price=primary_bars[0].close if primary_bars else None,
        )
    )
    data_coverage["requested_window"] = _requested_window_coverage(start, end)
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["label_bars"] = label_bar_coverage
    data_coverage["execution_bars"] = execution_bar_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    data_coverage["ibkr_futures_quotes"] = ibkr_futures_quote_coverage
    data_coverage["execution_research"] = _ml_execution_kwargs(args)
    candidate_config = _candidate_config_from_args(args, cfg)
    if (
        label_bars
        and candidate_config.adaptive_horizon
        and candidate_config.adaptive_horizon_granularity_seconds is None
    ):
        label_interval_seconds = _interval_seconds_or_none(label_bar_coverage.get("interval"))
        if label_interval_seconds is not None:
            candidate_config = replace(
                candidate_config,
                adaptive_horizon_granularity_seconds=label_interval_seconds,
            )
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        label_bars=label_bars or None,
        context_bars=context_bars,
        market_quotes=market_quotes,
        futures_market_quotes=futures_market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        prediction_market_feature_profile=args.prediction_market_feature_profile,
        feature_asof=args.feature_asof,
        external_feature_latency_seconds=args.external_feature_latency_seconds,
        candidate_config=candidate_config,
    )
    samples = _filter_samples_from_args(samples, args)
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    report = run_meta_label_walk_forward(
        samples,
        config=cfg,
        model_names=model_names,
        train_size=args.train_size or None,
        calibration_size=args.calibration_size or None,
        test_size=args.test_size or None,
        embargo_hours=args.embargo_hours or None,
        adaptive_threshold=args.adaptive_threshold,
        min_calibration_trades=args.min_calibration_trades,
        stacker_mode=args.stacker,
        adaptive_minimum_threshold=args.adaptive_minimum_threshold,
        tune_hyperparameters=args.hpo,
        hpo_profile=args.hpo_profile,
        hpo_trials=args.hpo_trials,
        foundation_max_samples=args.foundation_max_samples or 1024,
        data_coverage=data_coverage,
        target_trades_per_day=args.target_trades_per_day,
        allow_negative_ev_target_frequency=args.allow_negative_ev_frequency_probe,
        candidate_type_thresholds=args.candidate_type_thresholds,
        empirical_payoff_ev=args.empirical_payoff_ev,
        selection_score_mode=args.selection_score,
        target_frequency_mode=args.target_frequency_mode,
        selection_score_floor=args.selection_score_floor,
        selection_score_ceiling=args.selection_score_ceiling,
        adaptive_selection_score_floor=args.adaptive_selection_score_floor,
        adaptive_selection_score_direction=args.adaptive_selection_score_direction,
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
        respect_open_positions=respect_open_positions,
        capacity_release_mode=args.capacity_release_mode,
        selection_execution_policy=_selection_execution_policy_from_args(args),
        optimize_metric=args.optimize_metric,
        permutation_importance=args.permutation_importance,
        permutation_repeats=args.permutation_repeats,
        permutation_max_features=args.permutation_max_features,
        permutation_sample_limit=args.permutation_sample_limit,
        interpretability_top_n=args.interpretability_top_n,
        importance_scoring=_importance_scoring_from_args(args),
        permutation_grouping=args.permutation_grouping,
        shap_importance=args.shap_importance,
        shap_sample_limit=args.shap_sample_limit,
        shap_background_limit=args.shap_background_limit,
        shap_top_n=args.shap_top_n,
        shap_grouping=args.shap_grouping,
        feature_include_families=tuple(_csv_values(args.feature_include_groups)),
        feature_exclude_families=tuple(_csv_values(args.feature_exclude_groups)),
        feature_exclude_patterns=tuple(_csv_values(args.feature_exclude_patterns)),
    )
    _enforce_strict_live_valid_1s(
        args=args,
        data_coverage=data_coverage,
        label_bar_coverage=label_bar_coverage,
        execution_bar_coverage=execution_bar_coverage,
        folds=len(report.folds),
    )
    artifact_payload: dict[str, object] | None = None
    if getattr(args, "save_artifact", ""):
        artifact = fit_production_model_artifact(
            samples,
            config=cfg,
            report=report,
            model_names=model_names,
            stacker_mode=args.stacker,
            tune_hyperparameters=args.hpo,
            hpo_profile=args.hpo_profile,
            hpo_trials=args.hpo_trials,
            foundation_max_samples=args.foundation_max_samples or 1024,
            target_trades_per_day=args.target_trades_per_day or None,
            target_frequency_mode=args.target_frequency_mode,
            allow_negative_ev_target_frequency=args.allow_negative_ev_frequency_probe,
            selection_score_mode=args.selection_score,
            selection_score_floor=args.selection_score_floor,
            selection_score_ceiling=args.selection_score_ceiling,
            adaptive_selection_score_floor=args.adaptive_selection_score_floor,
            adaptive_selection_score_direction=args.adaptive_selection_score_direction,
            min_signal_spacing_hours=args.min_signal_spacing_hours,
            max_signals_per_group_per_day=args.max_signals_per_group_per_day,
            max_signals_per_timestamp=args.max_signals_per_timestamp,
            respect_open_positions=respect_open_positions,
            capacity_release_mode=args.capacity_release_mode,
            optimize_metric=args.optimize_metric,
            candidate_type_thresholds=args.candidate_type_thresholds,
            empirical_payoff_ev=args.empirical_payoff_ev,
            specialist_models=args.specialist_models,
            require_calibrated_selection=args.require_calibrated_selection,
            selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
            selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
            feature_asof=args.feature_asof,
            external_feature_latency_seconds=args.external_feature_latency_seconds,
            entry_order_model=getattr(args, "entry_order_model", "limit"),
            sizing_mode=getattr(args, "sizing_mode", "fixed"),
            sizing_score_field=getattr(args, "sizing_score_field", "probability"),
            sizing_score_direction=getattr(args, "sizing_score_direction", "high"),
            sizing_base_notional=getattr(args, "sizing_base_notional", 0.0),
            sizing_mid_notional=getattr(args, "sizing_mid_notional", 0.0),
            sizing_high_notional=getattr(args, "sizing_high_notional", 0.0),
            sizing_mid_score=getattr(args, "sizing_mid_score", 0.45),
            sizing_high_score=getattr(args, "sizing_high_score", 0.90),
            sizing_max_spread_bps=getattr(args, "sizing_max_spread_bps", 1.0),
            sizing_min_liquidity_score=getattr(args, "sizing_min_liquidity_score", 0.0),
            candidate_policy=_artifact_candidate_policy_from_args(args, candidate_config),
            horizon_policy=_artifact_horizon_policy_from_args(args, candidate_config),
            data_contract=_artifact_data_contract_from_args(
                args,
                data_coverage,
                label_bar_coverage=label_bar_coverage,
                execution_bar_coverage=execution_bar_coverage,
            ),
            feature_contract={"max_missing_feature_fraction": 0.0},
        )
        checksum = save_production_model_artifact(Path(args.save_artifact), artifact)
        artifact_payload = {
            "path": args.save_artifact,
            "sha256": checksum,
            "manifest": str(Path(args.save_artifact).with_suffix(Path(args.save_artifact).suffix + ".manifest.json")),
        }
    if args.output:
        write_meta_label_report(Path(args.output), report)
    payload = {
        "samples": report.samples,
        "folds": len(report.folds),
        "traded_signals": report.traded_signals,
        "net_pnl": report.net_pnl,
        "requested_models": report.requested_models,
        "calibration_method": report.calibration_method,
        "stacker_mode": report.stacker_mode,
        "data_coverage": report.data_coverage,
        "optimize_metric": args.optimize_metric,
        "candidate_type_summary": report_candidate_type_summary(report),
        "feature_importance_summary": report_feature_importance_summary(report),
        "shap_importance_summary": report_shap_importance_summary(report),
        "native_importance_summary": report_native_importance_summary(report),
        "model_family_summary": report_model_family_summary(report),
        "side_summary": report_side_summary(report),
        "folds_detail": [asdict(fold) for fold in report.folds],
        "saved_artifact": artifact_payload,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_model_smoke(args: argparse.Namespace) -> int:
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    results = smoke_test_model_stack(model_names, isolated=True, timeout_seconds=args.timeout_seconds)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    return 0 if all(result.ok for result in results) else 1


def _override_broker_connection_args(cfg, args: argparse.Namespace):
    client_id = getattr(args, "client_id", 0)
    broker_updates = {}
    if client_id:
        broker_updates["client_id"] = client_id
    account = (getattr(args, "account", "") or "").strip()
    if account:
        broker_updates["account"] = account
    if broker_updates:
        cfg = replace(cfg, broker=replace(cfg.broker, **broker_updates))
    return cfg


def _override_broker_client_id(cfg, args: argparse.Namespace):
    return _override_broker_connection_args(cfg, args)


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "count": float(len(values)),
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "mean": sum(values) / len(values),
    }


def _cmd_model_signal_audit(args: argparse.Namespace) -> int:
    cfg = _override_config_from_args(load_config(args.config), args)
    _validate_research_gated_args(args)
    respect_open_positions = _effective_respect_open_positions(args, cfg)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    futures_market_quotes, ibkr_futures_quote_coverage = _load_ibkr_quote_records(
        args,
        start,
        end,
        attr="ibkr_futures_quote_records",
        symbol_contains=None,
    )
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    data_coverage["ibkr_futures_quotes"] = ibkr_futures_quote_coverage
    data_coverage["execution_research"] = _ml_execution_kwargs(args)
    candidate_config = _candidate_config_from_args(args, cfg)
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        context_bars=context_bars,
        market_quotes=market_quotes,
        futures_market_quotes=futures_market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        prediction_market_feature_profile=args.prediction_market_feature_profile,
        feature_asof=args.feature_asof,
        external_feature_latency_seconds=args.external_feature_latency_seconds,
        candidate_config=candidate_config,
    )
    samples = _filter_samples_from_args(samples, args)
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    report = run_meta_label_walk_forward(
        samples,
        config=cfg,
        model_names=model_names,
        train_size=args.train_size or None,
        calibration_size=args.calibration_size or None,
        test_size=args.test_size or None,
        embargo_hours=args.embargo_hours or None,
        adaptive_threshold=args.adaptive_threshold,
        min_calibration_trades=args.min_calibration_trades,
        stacker_mode=args.stacker,
        adaptive_minimum_threshold=args.adaptive_minimum_threshold,
        tune_hyperparameters=args.hpo,
        hpo_profile=args.hpo_profile,
        hpo_trials=args.hpo_trials,
        foundation_max_samples=args.foundation_max_samples or 1024,
        data_coverage=data_coverage,
        target_trades_per_day=args.target_trades_per_day or None,
        allow_negative_ev_target_frequency=args.allow_negative_ev_frequency_probe,
        candidate_type_thresholds=args.candidate_type_thresholds,
        empirical_payoff_ev=args.empirical_payoff_ev,
        selection_score_mode=args.selection_score,
        target_frequency_mode=args.target_frequency_mode,
        selection_score_floor=args.selection_score_floor,
        selection_score_ceiling=args.selection_score_ceiling,
        adaptive_selection_score_floor=args.adaptive_selection_score_floor,
        adaptive_selection_score_direction=args.adaptive_selection_score_direction,
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
        respect_open_positions=respect_open_positions,
        capacity_release_mode=args.capacity_release_mode,
        selection_execution_policy=_selection_execution_policy_from_args(args),
        optimize_metric=args.optimize_metric,
        permutation_importance=args.permutation_importance,
        permutation_repeats=args.permutation_repeats,
        permutation_max_features=args.permutation_max_features,
        permutation_sample_limit=args.permutation_sample_limit,
        interpretability_top_n=args.interpretability_top_n,
        importance_scoring=_importance_scoring_from_args(args),
        permutation_grouping=args.permutation_grouping,
        shap_importance=args.shap_importance,
        shap_sample_limit=args.shap_sample_limit,
        shap_background_limit=args.shap_background_limit,
        shap_top_n=args.shap_top_n,
        shap_grouping=args.shap_grouping,
        feature_include_families=tuple(_csv_values(args.feature_include_groups)),
        feature_exclude_families=tuple(_csv_values(args.feature_exclude_groups)),
        feature_exclude_patterns=tuple(_csv_values(args.feature_exclude_patterns)),
    )
    summary, trades, rejections = run_ml_backtest(
        report=report,
        samples=samples,
        bars=primary_bars,
        config=cfg,
        starting_equity=args.starting_equity,
        requested_notional=args.notional,
        assumed_spread_bps=args.assumed_spread_bps,
        entry_limit_offset_bps=args.entry_limit_offset_bps,
        enforce_production_gate=not args.research_gate,
        allow_negative_ev_research=args.allow_negative_ev_frequency_probe,
        allow_research_short_backtest=args.allow_research_short_backtest,
        confidence_scaled_sizing=args.confidence_scaled_sizing,
        selection_score_mode=args.selection_score,
        **_ml_execution_kwargs(args),
    )
    prediction_span_days = summary.prediction_span_days or 1.0
    sample_span_days = _sample_span_days(samples)
    sample_by_type: dict[str, list] = {}
    sample_by_regime: dict[str, list] = {}
    for sample in samples:
        sample_by_type.setdefault(sample.candidate_type, []).append(sample)
        regime = sample.features.get("market_regime")
        sample_by_regime.setdefault(regime if isinstance(regime, str) else "unknown", []).append(sample)
    rejection_counts: dict[str, int] = {}
    for rejection in rejections:
        rejection_counts[rejection.reason] = rejection_counts.get(rejection.reason, 0) + 1
    payload = {
        "data_coverage": data_coverage,
        "candidate_mode": candidate_config.mode,
        "side_mode": candidate_config.side_mode,
        "samples": len(samples),
        "raw_candidates_per_day": len(samples) / sample_span_days,
        "model_approved_per_day": summary.model_approved_signals / prediction_span_days,
        "executed_trades_per_day": summary.trades_per_prediction_day,
        "backtest_summary": asdict(summary),
        "rejection_reasons": rejection_counts,
        "probability_distribution": _summary_stats([prediction.probability for prediction in report.predictions]),
        "expected_value_distribution": _summary_stats([prediction.expected_value for prediction in report.predictions]),
        "predicted_return_distribution": _summary_stats([prediction.predicted_return for prediction in report.predictions]),
        "selection_score_distribution": _summary_stats([prediction.selection_score for prediction in report.predictions]),
        "feature_importance_summary": report_feature_importance_summary(report),
        "shap_importance_summary": report_shap_importance_summary(report),
        "native_importance_summary": report_native_importance_summary(report),
        "model_family_summary": report_model_family_summary(report),
        "candidate_type_summary": {
            key: {
                "samples": len(rows),
                "samples_per_day": len(rows) / sample_span_days,
                "label_rate": sum(sample.label for sample in rows) / len(rows),
                "average_net_return": sum(sample.net_return for sample in rows) / len(rows),
            }
            for key, rows in sorted(sample_by_type.items())
        },
        "regime_summary": {
            key: {
                "samples": len(rows),
                "samples_per_day": len(rows) / sample_span_days,
                "label_rate": sum(sample.label for sample in rows) / len(rows),
                "average_net_return": sum(sample.net_return for sample in rows) / len(rows),
            }
            for key, rows in sorted(sample_by_regime.items())
        },
        "folds": [asdict(fold) for fold in report.folds],
        "trades": [asdict(trade) for trade in trades],
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_model_kronos_status(args: argparse.Namespace) -> int:
    from zeroalpha.features.kronos import kronos_import_status

    status = kronos_import_status()
    print(
        json.dumps(
            {
                "proxy": {
                    "available": True,
                    "detail": "built-in causal K-line proxy features available",
                    "provider": "proxy",
                },
                "official": asdict(status),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _interval_to_coinbase_granularity(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86_400,
    }
    if interval not in mapping:
        raise ValueError(f"unsupported Coinbase interval {interval}")
    return mapping[interval]


def _cmd_model_sweep_labels(args: argparse.Namespace) -> int:
    cfg = _override_config_from_args(load_config(args.config), args)
    _validate_research_gated_args(args)
    respect_open_positions = _effective_respect_open_positions(args, cfg)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    futures_market_quotes, ibkr_futures_quote_coverage = _load_ibkr_quote_records(
        args,
        start,
        end,
        attr="ibkr_futures_quote_records",
        symbol_contains=None,
    )
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    data_coverage["ibkr_futures_quotes"] = ibkr_futures_quote_coverage
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    results = run_label_geometry_sweep(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        starting_equity=args.starting_equity,
        context_bars=context_bars,
        market_quotes=market_quotes,
        futures_market_quotes=futures_market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        prediction_market_feature_profile=args.prediction_market_feature_profile,
        feature_asof=args.feature_asof,
        external_feature_latency_seconds=args.external_feature_latency_seconds,
        net_profit_targets=_parse_float_list(args.net_profit_targets),
        net_stop_losses=_parse_float_list(args.net_stop_losses),
        max_holding_hours_values=_parse_int_list(args.max_holding_hours_values),
        model_names=model_names,
        adaptive_threshold=args.adaptive_threshold,
        stacker_mode=args.stacker,
        adaptive_minimum_threshold=args.adaptive_minimum_threshold,
        optimize_metric=args.optimize_metric,
        target_trades_per_day=args.target_trades_per_day or None,
        allow_negative_ev_target_frequency=args.allow_negative_ev_frequency_probe,
        candidate_type_thresholds=args.candidate_type_thresholds,
        empirical_payoff_ev=args.empirical_payoff_ev,
        confidence_scaled_sizing=args.confidence_scaled_sizing,
        enforce_production_gate=not args.research_gate,
        allow_research_short_backtest=args.allow_research_short_backtest,
        candidate_side_mode=args.side_mode,
        allow_short_research=(
            cfg.contract.instrument_model == "futures" or bool(args.allow_spot_short_research)
        ),
        candidate_mode={
            "dense": "dense_research",
            "aggressive": "aggressive_rules",
            "active": "active_research",
        }.get(args.candidate_mode, "rules"),
        selection_score_mode=args.selection_score,
        target_frequency_mode=args.target_frequency_mode,
        selection_score_floor=args.selection_score_floor,
        selection_score_ceiling=args.selection_score_ceiling,
        adaptive_selection_score_floor=args.adaptive_selection_score_floor,
        adaptive_selection_score_direction=args.adaptive_selection_score_direction,
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        tune_hyperparameters=args.hpo,
        hpo_profile=args.hpo_profile,
        hpo_trials=args.hpo_trials,
        respect_open_positions=respect_open_positions,
        capacity_release_mode=args.capacity_release_mode,
    )
    payload = {
        "data_coverage": data_coverage,
        "results": sweep_results_asdict(results),
        "top": sweep_results_asdict(results[: min(args.top, len(results))]),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["top"], indent=2, sort_keys=True))
    return 0


async def _broker_smoke_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker

    cfg = _override_broker_client_id(
        _override_config_from_args(load_config(args.config), args),
        args,
    )
    broker = IBKRBroker(cfg)
    await broker.connect(read_only=args.read_only)
    contract = await broker.qualify_crypto_contract()
    print(
        "ok: qualified "
        f"{contract.symbol}/{contract.currency} exchange={contract.exchange} con_id={contract.con_id}"
    )
    await broker.disconnect()
    return 0


def _cmd_broker_smoke(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_smoke_async(args))


def _floor_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        raise ValueError("increment must be positive")
    return math.floor(value / increment) * increment


async def _broker_order_test_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker
    from zeroalpha.execution.orders import CryptoOrderFactory

    cfg = _override_broker_client_id(
        _override_config_from_args(load_config(args.config), args),
        args,
    )
    _validate_paper_order_test_config(cfg, args)
    broker = IBKRBroker(cfg)
    with _runtime_event_stream_from_args(args, "broker.order_test") as events:
        connected = False
        try:
            events.emit(
                "broker.order_test.start",
                "starting paper order submit/cancel check",
                config=args.config,
                client_id=cfg.broker.client_id,
                notional_usd=args.notional,
            )
            await broker.connect(read_only=False)
            connected = True
            events.emit(
                "broker.connected",
                "connected to IBKR Gateway/TWS",
                host=cfg.broker.host,
                port=cfg.broker.port,
                read_only=False,
            )
            contract = await broker.qualify_crypto_contract()
            events.emit(
                "broker.contract_qualified",
                "qualified crypto contract",
                symbol=f"{contract.symbol}/{contract.currency}",
                exchange=contract.exchange,
                con_id=contract.con_id,
            )
            quote = await broker.snapshot_quote(contract)
            events.emit(
                "market.quote",
                "received TWS top-of-book quote",
                bid=quote.bid,
                ask=quote.ask,
                spread_bps=quote.spread_bps,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                market_data_type=quote.market_data_type,
            )
            raw_limit = quote.bid * (1 - args.offset_bps / 10_000)
            limit_price = round(_floor_to_increment(raw_limit, args.price_increment), 8)
            quantity = round(args.notional / limit_price, 8)
            intent = CryptoOrderFactory.limit_entry(
                event_id="paper_order_test",
                symbol=f"{contract.symbol}/{contract.currency}",
                quantity=quantity,
                limit_price=limit_price,
            )
            trade = broker.place_order_intent(contract, intent)
            events.emit(
                "order.submitted",
                "submitted non-marketable paper limit order",
                order_id=getattr(getattr(trade, "order", None), "orderId", None),
                perm_id=getattr(getattr(trade, "order", None), "permId", None),
                side=intent.side.value,
                order_type=intent.order_type.value,
                quantity=quantity,
                limit_price=limit_price,
            )
            await broker.wait(args.wait_seconds)
            broker.cancel_trade(trade)
            events.emit(
                "order.cancel_requested",
                "requested order cancel",
                order_id=getattr(getattr(trade, "order", None), "orderId", None),
                status=getattr(getattr(trade, "orderStatus", None), "status", "unknown"),
            )
            await broker.wait(args.cancel_wait_seconds)
            status = getattr(getattr(trade, "orderStatus", None), "status", "unknown")
            order_id = getattr(getattr(trade, "order", None), "orderId", None)
            events.emit(
                "order.finished",
                "paper order test finished",
                order_id=order_id,
                status=status,
                filled=_finite_number(getattr(getattr(trade, "orderStatus", None), "filled", None)),
                remaining=_finite_number(getattr(getattr(trade, "orderStatus", None), "remaining", None)),
            )
            print(
                json.dumps(
                    {
                        "submitted": True,
                        "cancel_requested": True,
                        "symbol": intent.symbol,
                        "exchange": contract.exchange,
                        "con_id": contract.con_id,
                        "order_id": order_id,
                        "status": status,
                        "bid": quote.bid,
                        "ask": quote.ask,
                        "limit_price": limit_price,
                        "price_increment": args.price_increment,
                        "raw_limit_price": raw_limit,
                        "quantity": quantity,
                        "notional": args.notional,
                        "offset_bps": args.offset_bps,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        finally:
            if connected:
                await broker.disconnect()
                events.emit("broker.disconnected", "disconnected from IBKR Gateway/TWS")
    return 0


def _cmd_broker_order_test(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_order_test_async(args))


def _finite_number(value: object) -> float | None:
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted) or abs(converted) > 1e100:
        return None
    return converted


def _trade_status_payload(trade) -> dict[str, object]:
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    return {
        "order_id": getattr(order, "orderId", None),
        "perm_id": getattr(order, "permId", None),
        "action": getattr(order, "action", ""),
        "order_type": getattr(order, "orderType", ""),
        "limit_price": _finite_number(getattr(order, "lmtPrice", None)),
        "aux_price": _finite_number(getattr(order, "auxPrice", None)),
        "total_quantity": _finite_number(getattr(order, "totalQuantity", None)),
        "cash_qty": _finite_number(getattr(order, "cashQty", None)),
        "status": getattr(status, "status", "unknown"),
        "filled": _finite_number(getattr(status, "filled", None)),
        "remaining": _finite_number(getattr(status, "remaining", None)),
        "average_fill_price": _finite_number(getattr(status, "avgFillPrice", None)),
    }


def _execution_payload(execution) -> dict[str, object]:
    timestamp = getattr(execution, "time", None)
    return {
        "exec_id": getattr(execution, "execId", ""),
        "time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or ""),
        "account": getattr(execution, "acctNumber", ""),
        "exchange": getattr(execution, "exchange", ""),
        "side": getattr(execution, "side", ""),
        "quantity": _finite_number(getattr(execution, "shares", None)),
        "price": _finite_number(getattr(execution, "price", None)),
        "perm_id": getattr(execution, "permId", None),
        "client_id": getattr(execution, "clientId", None),
        "order_id": getattr(execution, "orderId", None),
        "cum_quantity": _finite_number(getattr(execution, "cumQty", None)),
        "average_price": _finite_number(getattr(execution, "avgPrice", None)),
        "last_liquidity": getattr(execution, "lastLiquidity", None),
    }


def _commission_report_payload(commission_report) -> dict[str, object]:
    commission = _finite_number(getattr(commission_report, "commission", None))
    realized_pnl = _finite_number(getattr(commission_report, "realizedPNL", None))
    return {
        "exec_id": getattr(commission_report, "execId", ""),
        "commission": commission,
        "currency": getattr(commission_report, "currency", ""),
        "realized_pnl": realized_pnl,
        "commission_available": commission is not None,
        "realized_pnl_available": realized_pnl is not None,
    }


def _fill_payload(fill) -> dict[str, object]:
    timestamp = getattr(fill, "time", None)
    return {
        "time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or ""),
        "execution": _execution_payload(getattr(fill, "execution", None)),
        "commission_report": _commission_report_payload(getattr(fill, "commissionReport", None)),
    }


def _trade_log_payload(trade) -> list[dict[str, object]]:
    rows = []
    for row in getattr(trade, "log", []) or []:
        timestamp = getattr(row, "time", None)
        rows.append(
            {
                "time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or ""),
                "status": getattr(row, "status", ""),
                "message": getattr(row, "message", ""),
                "error_code": getattr(row, "errorCode", 0),
            }
        )
    return rows


def _trade_fill_payload(trade) -> dict[str, object]:
    fills = []
    filled_quantity = 0.0
    gross_value = 0.0
    commission = 0.0
    realized_pnl = 0.0
    missing_commission_exec_ids: list[str] = []
    missing_realized_pnl_exec_ids: list[str] = []
    for fill in getattr(trade, "fills", []) or []:
        execution = getattr(fill, "execution", None)
        commission_report = getattr(fill, "commissionReport", None)
        quantity = _finite_number(getattr(execution, "shares", None)) or 0.0
        price = _finite_number(getattr(execution, "price", None)) or 0.0
        exec_id = getattr(execution, "execId", "")
        raw_commission = _finite_number(getattr(commission_report, "commission", None))
        raw_realized_pnl = _finite_number(getattr(commission_report, "realizedPNL", None))
        if raw_commission is None:
            missing_commission_exec_ids.append(str(exec_id))
        if raw_realized_pnl is None:
            missing_realized_pnl_exec_ids.append(str(exec_id))
        fill_commission = raw_commission or 0.0
        fill_realized_pnl = raw_realized_pnl or 0.0
        filled_quantity += quantity
        gross_value += quantity * price
        commission += fill_commission
        realized_pnl += fill_realized_pnl
        timestamp = getattr(fill, "time", None)
        fills.append(
            {
                "time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or ""),
                "exec_id": exec_id,
                "side": getattr(execution, "side", ""),
                "exchange": getattr(execution, "exchange", ""),
                "quantity": quantity,
                "price": price,
                "average_price": _finite_number(getattr(execution, "avgPrice", None)),
                "commission": raw_commission,
                "commission_currency": getattr(commission_report, "currency", ""),
                "realized_pnl": raw_realized_pnl,
                "commission_report_available": raw_commission is not None,
                "realized_pnl_available": raw_realized_pnl is not None,
            }
        )
    fill_count = len(fills)
    return {
        "filled_quantity": filled_quantity,
        "average_price": gross_value / filled_quantity if filled_quantity > 0 else None,
        "gross_value": gross_value,
        "commission": commission,
        "realized_pnl_reported": realized_pnl,
        "fill_count": fill_count,
        "commission_report_complete": fill_count > 0 and not missing_commission_exec_ids,
        "realized_pnl_report_complete": fill_count > 0 and not missing_realized_pnl_exec_ids,
        "missing_commission_exec_ids": missing_commission_exec_ids,
        "missing_realized_pnl_exec_ids": missing_realized_pnl_exec_ids,
        "fills": fills,
    }


def _broker_session_state_payload(
    broker,
    *,
    related_order_ids: set[int] | None = None,
    related_perm_ids: set[int] | None = None,
) -> dict[str, object]:
    ib = getattr(broker, "_ib", None)
    if ib is None:
        return {"available": False}
    trades = ib.trades()
    open_trades = ib.openTrades()
    fills = ib.fills()
    executions = ib.executions()
    related_order_ids = related_order_ids or set()
    related_perm_ids = related_perm_ids or set()

    def is_related(order_id: object, perm_id: object) -> bool:
        if related_perm_ids:
            return perm_id in related_perm_ids
        return order_id in related_order_ids

    related_trades = [
        trade
        for trade in trades
        if is_related(
            getattr(getattr(trade, "order", None), "orderId", None),
            getattr(getattr(trade, "order", None), "permId", None),
        )
    ]
    related_fills = [
        fill
        for fill in fills
        if is_related(
            getattr(getattr(fill, "execution", None), "orderId", None),
            getattr(getattr(fill, "execution", None), "permId", None),
        )
    ]
    related_executions = [
        execution
        for execution in executions
        if is_related(getattr(execution, "orderId", None), getattr(execution, "permId", None))
    ]
    return {
        "available": True,
        "related_order_ids": sorted(related_order_ids),
        "related_perm_ids": sorted(related_perm_ids),
        "trade_count": len(trades),
        "open_trade_count": len(open_trades),
        "fill_count": len(fills),
        "execution_count": len(executions),
        "open_order_count": len(ib.openOrders()),
        "trades": [
            {
                "status": _trade_status_payload(trade),
                "fill": _trade_fill_payload(trade),
                "log": _trade_log_payload(trade),
            }
            for trade in trades
        ],
        "open_trades": [_trade_status_payload(trade) for trade in open_trades],
        "fills": [_fill_payload(fill) for fill in fills],
        "executions": [_execution_payload(execution) for execution in executions],
        "related_trade_count": len(related_trades),
        "related_fill_count": len(related_fills),
        "related_execution_count": len(related_executions),
        "related_trades": [
            {
                "status": _trade_status_payload(trade),
                "fill": _trade_fill_payload(trade),
                "log": _trade_log_payload(trade),
            }
            for trade in related_trades
        ],
        "related_fills": [_fill_payload(fill) for fill in related_fills],
        "related_executions": [_execution_payload(execution) for execution in related_executions],
    }


def _trade_done(trade) -> bool:
    is_done = getattr(trade, "isDone", None)
    if callable(is_done):
        return bool(is_done())
    status = getattr(getattr(trade, "orderStatus", None), "status", "")
    return status in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def _require_tws_fill_accounting(order_name: str, result: dict[str, object]) -> None:
    fill = result.get("fill")
    if not isinstance(fill, dict):
        raise SystemExit(f"{order_name} has no TWS fill payload")
    if not fill.get("commission_report_complete"):
        missing = fill.get("missing_commission_exec_ids")
        raise SystemExit(
            f"{order_name} is missing TWS commission reports for executions {missing}; "
            "increase --commission-wait-seconds and rerun"
        )
    if not fill.get("realized_pnl_report_complete"):
        missing = fill.get("missing_realized_pnl_exec_ids")
        raise SystemExit(
            f"{order_name} is missing TWS realized PnL reports for executions {missing}; "
            "increase --commission-wait-seconds and rerun"
        )


async def _wait_for_trade_done(
    broker,
    trade,
    *,
    timeout_seconds: float,
    commission_wait_seconds: float = 0.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _trade_done(trade):
            break
        await broker.wait(0.25)
    if commission_wait_seconds > 0:
        await broker.wait(commission_wait_seconds)
    return {
        "status": _trade_status_payload(trade),
        "fill": _trade_fill_payload(trade),
        "log": _trade_log_payload(trade),
        "done": _trade_done(trade),
    }


def _total_portfolio_pnl(portfolio: list[dict[str, object]]) -> float | None:
    total = 0.0
    seen = False
    for item in portfolio:
        for key in ("unrealized_pnl", "realized_pnl"):
            value = _finite_number(item.get(key))
            if value is not None:
                total += value
                seen = True
    return total if seen else None


def _daily_pnl(pnl: list[dict[str, object]]) -> float | None:
    for row in pnl:
        value = _finite_number(row.get("daily_pnl"))
        if value is not None:
            return value
    return None


def _account_summary_value(snapshot: dict[str, object], tag: str) -> float | None:
    rows = snapshot.get("account_summary", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("tag") != tag:
            continue
        value = _finite_number(row.get("value"))
        if value is not None:
            return value
    return None


def _snapshot_loss_delta(
    *,
    baseline: dict[str, object],
    current: dict[str, object],
) -> float | None:
    baseline_daily = _daily_pnl(baseline.get("pnl", []))  # type: ignore[arg-type]
    current_daily = _daily_pnl(current.get("pnl", []))  # type: ignore[arg-type]
    if baseline_daily is not None and current_daily is not None:
        return max(0.0, baseline_daily - current_daily)
    baseline_portfolio = _total_portfolio_pnl(baseline.get("portfolio", []))  # type: ignore[arg-type]
    current_portfolio = _total_portfolio_pnl(current.get("portfolio", []))  # type: ignore[arg-type]
    if baseline_portfolio is not None and current_portfolio is not None:
        return max(0.0, baseline_portfolio - current_portfolio)
    for tag in ("NetLiquidation", "TotalCashValue"):
        baseline_value = _account_summary_value(baseline, tag)
        current_value = _account_summary_value(current, tag)
        if baseline_value is not None and current_value is not None:
            return max(0.0, baseline_value - current_value)
    return None


async def _paper_account_snapshot(
    broker,
    contract,
    args: argparse.Namespace,
    *,
    quote=None,
) -> dict[str, object]:
    quote = quote or await broker.snapshot_quote(
        contract,
        max_wait_seconds=args.snapshot_timeout_seconds,
    )
    account = broker.resolved_account()
    summary = await broker.account_summary(
        account=account,
        timeout_seconds=args.account_refresh_timeout_seconds,
    )
    portfolio = await broker.portfolio_items(
        account=account,
        refresh=True,
        timeout_seconds=args.account_refresh_timeout_seconds,
    )
    positions = await broker.positions(
        account=account,
        timeout_seconds=args.account_refresh_timeout_seconds,
    )
    pnl = await broker.pnl_snapshot(account=account, wait_seconds=args.pnl_wait_seconds)
    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "account": account,
        "quote": asdict(quote),
        "account_summary": summary,
        "portfolio": portfolio,
        "positions": positions,
        "pnl": pnl,
    }


def _position_quantity_from_snapshot(snapshot: dict[str, object], contract) -> float:
    rows = snapshot.get("positions", [])
    if not isinstance(rows, list):
        return 0.0
    con_id = getattr(contract, "con_id", None)
    symbol = str(getattr(contract, "symbol", "")).upper()
    security_type = str(getattr(getattr(contract, "raw", None), "secType", "") or "").upper()
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_position = _finite_number(row.get("position"))
        if row_position is None:
            continue
        row_con_id = row.get("con_id")
        row_symbol = str(row.get("symbol") or "").upper()
        row_security_type = str(row.get("security_type") or "").upper()
        if con_id is not None and row_con_id == con_id:
            total += row_position
            continue
        if row_symbol == symbol and (not security_type or row_security_type == security_type):
            total += row_position
    return max(0.0, total)


def _write_jsonl_record(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def _runtime_event_stream_from_args(args: argparse.Namespace, run_name: str) -> RuntimeEventStream:
    event_log = getattr(args, "event_log", "") or ""
    return RuntimeEventStream(
        run_name=run_name,
        console=bool(getattr(args, "stream_events", True)),
        console_format=getattr(args, "stream_format", "text"),
        output_path=Path(event_log) if event_log else None,
    )


def _account_tags(snapshot: dict[str, object], tags: tuple[str, ...]) -> dict[str, float]:
    return {
        tag: value
        for tag in tags
        if (value := _account_summary_value(snapshot, tag)) is not None
    }


def _position_summary(snapshot: dict[str, object]) -> list[dict[str, object]]:
    rows = snapshot.get("positions", [])
    if not isinstance(rows, list):
        return []
    summary: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        position = _finite_number(row.get("position"))
        if position is None or abs(position) <= 1e-12:
            continue
        summary.append(
            {
                "symbol": row.get("symbol", ""),
                "security_type": row.get("security_type", ""),
                "local_symbol": row.get("local_symbol", ""),
                "position": position,
                "average_cost": row.get("average_cost"),
            }
        )
    return summary


async def _current_contract_position_quantity(
    broker,
    contract,
    args: argparse.Namespace,
) -> float:
    account = broker.resolved_account(require_explicit=True)
    rows = await broker.positions(
        account=account,
        timeout_seconds=args.account_refresh_timeout_seconds,
    )
    con_id = getattr(contract, "con_id", None)
    symbol = str(getattr(contract, "symbol", "")).upper()
    security_type = str(getattr(getattr(contract, "raw", None), "secType", "") or "").upper()
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_position = _finite_number(row.get("position"))
        if row_position is None:
            continue
        row_con_id = row.get("con_id")
        row_symbol = str(row.get("symbol") or "").upper()
        row_security_type = str(row.get("security_type") or "").upper()
        if con_id is not None and row_con_id == con_id:
            total += row_position
            continue
        if row_symbol == symbol and (not security_type or row_security_type == security_type):
            total += row_position
    return max(0.0, total)


async def _submit_and_cancel_paper_limit_order(broker, contract, args: argparse.Namespace) -> dict[str, object]:
    from zeroalpha.execution.orders import CryptoOrderFactory

    quote = await broker.snapshot_quote(contract, max_wait_seconds=args.snapshot_timeout_seconds)
    raw_limit = quote.bid * (1 - args.order_offset_bps / 10_000)
    limit_price = round(_floor_to_increment(raw_limit, args.price_increment), 8)
    if limit_price <= 0:
        raise SystemExit("paper-test computed a nonpositive limit price; refusing order")
    quantity = round(args.order_notional / limit_price, 8)
    intent = CryptoOrderFactory.limit_entry(
        event_id="paper_trading_test",
        symbol=f"{contract.symbol}/{contract.currency}",
        quantity=quantity,
        limit_price=limit_price,
    )
    trade = broker.place_order_intent(contract, intent)
    await broker.wait(args.order_wait_seconds)
    submitted_status = _trade_status_payload(trade)
    broker.cancel_trade(trade)
    await broker.wait(args.cancel_wait_seconds)
    canceled_status = _trade_status_payload(trade)
    return {
        "submitted": True,
        "cancel_requested": True,
        "symbol": intent.symbol,
        "exchange": contract.exchange,
        "con_id": contract.con_id,
        "bid": quote.bid,
        "ask": quote.ask,
        "spread_bps": quote.spread_bps,
        "limit_price": limit_price,
        "quantity": quantity,
        "notional": args.order_notional,
        "offset_bps": args.order_offset_bps,
        "submitted_status": submitted_status,
        "canceled_status": canceled_status,
    }


async def _broker_paper_test_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker

    cfg = _override_broker_client_id(
        _override_config_from_args(load_config(args.config), args),
        args,
    )
    _validate_paper_test_config(cfg, args)
    broker = IBKRBroker(cfg)
    output = Path(args.output) if args.output else None
    order_payload: dict[str, object] | None = None
    stop_reason = "duration_complete"
    snapshots = 0
    max_observed_loss = 0.0
    with _runtime_event_stream_from_args(args, "broker.paper_test") as events:
        events.emit(
            "broker.paper_test.start",
            "starting bounded paper health run",
            config=args.config,
            client_id=cfg.broker.client_id,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            max_cash_usd=args.max_cash_usd,
            max_loss_usd=args.max_loss_usd,
            submit_order=bool(args.submit_order),
            output=str(output) if output else "",
        )
        await broker.connect(read_only=not args.submit_order)
        try:
            events.emit(
                "broker.connected",
                "connected to IBKR Gateway/TWS",
                host=cfg.broker.host,
                port=cfg.broker.port,
                read_only=not args.submit_order,
            )
            contract = await broker.qualify_crypto_contract()
            events.emit(
                "broker.contract_qualified",
                "qualified crypto contract",
                symbol=f"{contract.symbol}/{contract.currency}",
                exchange=contract.exchange,
                con_id=contract.con_id,
            )
            baseline = await _paper_account_snapshot(broker, contract, args)
            if output:
                _write_jsonl_record(output, {"type": "baseline", **baseline})
            quote = baseline["quote"] if isinstance(baseline.get("quote"), dict) else {}
            events.emit(
                "account.baseline",
                "captured baseline account snapshot",
                account=baseline.get("account"),
                bid=quote.get("bid"),
                ask=quote.get("ask"),
                spread_bps=quote.get("spread_bps"),
                positions=_position_summary(baseline),
                **_account_tags(baseline, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
            )
            if args.submit_order:
                events.emit(
                    "order.health_check.start",
                    "submitting paper health-check limit order",
                    notional_usd=args.order_notional,
                    offset_bps=args.order_offset_bps,
                )
                order_payload = await _submit_and_cancel_paper_limit_order(broker, contract, args)
                events.emit(
                    "order.health_check.finished",
                    "paper health-check order finished",
                    submitted_status=order_payload.get("submitted_status"),
                    canceled_status=order_payload.get("canceled_status"),
                    limit_price=order_payload.get("limit_price"),
                    quantity=order_payload.get("quantity"),
                )
                if output:
                    _write_jsonl_record(output, {"type": "order_test", **order_payload})

            deadline = time.monotonic() + args.duration_seconds
            while True:
                current = await _paper_account_snapshot(broker, contract, args)
                loss_delta = _snapshot_loss_delta(baseline=baseline, current=current)
                if loss_delta is not None:
                    max_observed_loss = max(max_observed_loss, loss_delta)
                record = {
                    "type": "snapshot",
                    "loss_delta_usd": loss_delta,
                    **current,
                }
                if output:
                    _write_jsonl_record(output, record)
                snapshots += 1
                quote = current["quote"] if isinstance(current.get("quote"), dict) else {}
                events.emit(
                    "account.snapshot",
                    "captured paper account snapshot",
                    snapshot_index=snapshots,
                    account=current.get("account"),
                    bid=quote.get("bid"),
                    ask=quote.get("ask"),
                    spread_bps=quote.get("spread_bps"),
                    loss_delta_usd=loss_delta,
                    max_observed_loss_usd=max_observed_loss,
                    positions=_position_summary(current),
                    **_account_tags(current, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
                )
                if args.max_loss_usd > 0 and loss_delta is not None and loss_delta >= args.max_loss_usd:
                    stop_reason = "max_loss_usd"
                    events.emit(
                        "risk.max_loss_triggered",
                        "paper run max-loss guard triggered",
                        loss_delta_usd=loss_delta,
                        max_loss_usd=args.max_loss_usd,
                    )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await broker.wait(min(args.interval_seconds, remaining))
        finally:
            await broker.disconnect()
            events.emit(
                "broker.paper_test.finished",
                "paper health run finished",
                snapshots=snapshots,
                max_observed_loss_usd=max_observed_loss,
                stop_reason=stop_reason,
            )
            events.emit("broker.disconnected", "disconnected from IBKR Gateway/TWS")

    print(
        json.dumps(
            {
                "ok": True,
                "paper_mode": True,
                "submitted_order": bool(args.submit_order),
                "max_cash_usd": args.max_cash_usd,
                "max_loss_usd": args.max_loss_usd,
                "snapshots": snapshots,
                "max_observed_loss_usd": max_observed_loss,
                "stop_reason": stop_reason,
                "output": str(output) if output else "",
                "order": order_payload,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def _cmd_broker_paper_test(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_paper_test_async(args))


async def _monitor_synthetic_stop(
    broker,
    contract,
    *,
    entry_price: float,
    notional: float,
    hold_seconds: float,
    stop_loss_bps: float,
    max_loss_usd: float,
    monitor_interval_seconds: float,
    output: Path | None,
    events: RuntimeEventStream | None = None,
) -> dict[str, object]:
    stop_price = entry_price * (1 - stop_loss_bps / 10_000)
    deadline = time.monotonic() + hold_seconds
    last_quote: dict[str, object] | None = None
    while True:
        quote = await broker.snapshot_quote(contract, max_wait_seconds=10.0)
        loss_bps = max(0.0, 10_000 * (entry_price / quote.bid - 1)) if quote.bid > 0 else 0.0
        last_quote = {**asdict(quote), "loss_bps_from_entry": loss_bps, "stop_price": stop_price}
        if output:
            _write_jsonl_record(output, {"type": "hold_quote", **last_quote})
        if events is not None:
            events.emit(
                "position.monitor",
                "synthetic stop monitor quote",
                bid=quote.bid,
                ask=quote.ask,
                spread_bps=quote.spread_bps,
                loss_bps_from_entry=loss_bps,
                stop_price=stop_price,
                remaining_seconds=max(0.0, deadline - time.monotonic()),
            )
        if quote.bid <= stop_price:
            if events is not None:
                events.emit(
                    "position.exit_triggered",
                    "synthetic stop-loss trigger hit",
                    exit_reason="synthetic_stop_loss",
                    bid=quote.bid,
                    stop_price=stop_price,
                    loss_bps_from_entry=loss_bps,
                )
            return {
                "exit_reason": "synthetic_stop_loss",
                "stop_price": stop_price,
                "last_quote": last_quote,
            }
        if max_loss_usd > 0 and notional * loss_bps / 10_000 >= max_loss_usd:
            if events is not None:
                events.emit(
                    "position.exit_triggered",
                    "position max-loss trigger hit",
                    exit_reason="max_loss_usd",
                    bid=quote.bid,
                    stop_price=stop_price,
                    loss_bps_from_entry=loss_bps,
                    max_loss_usd=max_loss_usd,
                )
            return {
                "exit_reason": "max_loss_usd",
                "stop_price": stop_price,
                "last_quote": last_quote,
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if events is not None:
                events.emit(
                    "position.exit_triggered",
                    "timed exit trigger hit",
                    exit_reason="timed_exit",
                    bid=quote.bid,
                    stop_price=stop_price,
                    loss_bps_from_entry=loss_bps,
                )
            return {
                "exit_reason": "timed_exit",
                "stop_price": stop_price,
                "last_quote": last_quote,
            }
        await broker.wait(min(monitor_interval_seconds, remaining))


async def _attempt_emergency_cleanup(
    broker,
    contract,
    args: argparse.Namespace,
    *,
    reason: str,
    output: Path | None,
    events: RuntimeEventStream,
) -> dict[str, object]:
    from zeroalpha.execution.orders import CryptoOrderFactory

    payload: dict[str, object] = {
        "reason": reason,
        "cancel_attempts": 0,
        "liquidation_attempted": False,
        "liquidation_result": None,
        "final_snapshot": None,
        "error": "",
    }
    events.emit(
        "risk.emergency_cleanup.start",
        "round-trip cleanup started after entry-side exposure",
        priority="critical",
        reason=reason,
    )
    try:
        ib = getattr(broker, "_ib", None)
        for trade in (ib.openTrades() if ib is not None else []):
            try:
                broker.cancel_trade(trade)
                payload["cancel_attempts"] = int(payload["cancel_attempts"]) + 1
            except Exception as exc:  # pragma: no cover - gateway dependent
                events.emit(
                    "risk.emergency_cleanup.cancel_failed",
                    "failed to cancel an open order during emergency cleanup",
                    priority="critical",
                    error=f"{type(exc).__name__}: {exc}",
                )
        await broker.wait(1.0)
        position_quantity = await _current_contract_position_quantity(broker, contract, args)
        payload["position_quantity_before_liquidation"] = position_quantity
        if position_quantity > 1e-8:
            quote = await broker.snapshot_quote(contract, max_wait_seconds=args.snapshot_timeout_seconds)
            max_notional = (
                broker.config.risk.live_max_notional
                if broker.config.runtime.mode == RuntimeMode.LIVE
                else broker.config.risk.paper_max_notional
            )
            bounded_quantity = min(position_quantity, max_notional / quote.bid)
            bounded_quantity = round(max(0.0, bounded_quantity), 8)
            if bounded_quantity > 0:
                intent = CryptoOrderFactory.urgent_market_exit(
                    symbol=f"{contract.symbol}/{contract.currency}",
                    quantity=bounded_quantity,
                    reason="emergency_cleanup",
                )
                trade = broker.place_order_intent(
                    contract,
                    intent,
                    reference_price=quote.bid,
                    current_position_quantity=position_quantity,
                )
                payload["liquidation_attempted"] = True
                payload["liquidation_quantity"] = bounded_quantity
                payload["liquidation_reference_bid"] = quote.bid
                result = await _wait_for_trade_done(
                    broker,
                    trade,
                    timeout_seconds=args.order_timeout_seconds,
                    commission_wait_seconds=args.commission_wait_seconds,
                )
                payload["liquidation_result"] = result
        final_snapshot = await _paper_account_snapshot(broker, contract, args)
        payload["final_snapshot"] = final_snapshot
        if output:
            _write_jsonl_record(output, {"type": "emergency_cleanup", **payload})
        events.emit(
            "risk.emergency_cleanup.finished",
            "round-trip emergency cleanup finished",
            priority="critical",
            cancel_attempts=payload["cancel_attempts"],
            liquidation_attempted=payload["liquidation_attempted"],
            position_quantity_before_liquidation=payload.get("position_quantity_before_liquidation"),
            positions=_position_summary(final_snapshot),
        )
    except Exception as exc:  # pragma: no cover - gateway dependent
        payload["error"] = f"{type(exc).__name__}: {exc}"
        if output:
            _write_jsonl_record(output, {"type": "emergency_cleanup_failed", **payload})
        events.emit(
            "risk.emergency_cleanup.failed",
            "round-trip emergency cleanup failed",
            priority="critical",
            error=payload["error"],
        )
    return payload


async def _broker_round_trip_test_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker
    from zeroalpha.execution.orders import CryptoOrderFactory

    cfg = _override_broker_client_id(load_config(args.config), args)
    _validate_round_trip_test_config(cfg, args)
    broker = IBKRBroker(cfg)
    output = Path(args.output) if args.output else None
    events = _runtime_event_stream_from_args(args, "broker.round_trip_test")
    events.emit(
        "broker.round_trip.start",
        "starting controlled paper round trip",
        config=args.config,
        client_id=cfg.broker.client_id,
        notional_usd=args.notional,
        hold_seconds=args.hold_seconds,
        synthetic_stop_loss_bps=args.synthetic_stop_loss_bps,
        max_cash_usd=args.max_cash_usd,
        max_loss_usd=args.max_loss_usd,
        output=str(output) if output else "",
    )
    await broker.connect(read_only=False)
    events.emit(
        "broker.connected",
        "connected to IBKR Gateway/TWS",
        host=cfg.broker.host,
        port=cfg.broker.port,
        read_only=False,
    )
    buy_result: dict[str, object] | None = None
    sell_result: dict[str, object] | None = None
    monitor_result: dict[str, object] | None = None
    baseline: dict[str, object] | None = None
    final_snapshot: dict[str, object] | None = None
    contract = None
    entry_exposed = False
    try:
        contract = await broker.qualify_crypto_contract()
        symbol = f"{contract.symbol}/{contract.currency}"
        events.emit(
            "broker.contract_qualified",
            "qualified crypto contract",
            symbol=symbol,
            exchange=contract.exchange,
            con_id=contract.con_id,
        )
        baseline = await _paper_account_snapshot(broker, contract, args)
        if output:
            _write_jsonl_record(output, {"type": "baseline", **baseline})
        quote = baseline["quote"] if isinstance(baseline.get("quote"), dict) else {}
        events.emit(
            "account.baseline",
            "captured baseline account snapshot",
            account=baseline.get("account"),
            bid=quote.get("bid"),
            ask=quote.get("ask"),
            spread_bps=quote.get("spread_bps"),
            positions=_position_summary(baseline),
            **_account_tags(baseline, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
        )

        buy_intent = CryptoOrderFactory.market_buy_cash(
            event_id="manual_round_trip_buy",
            symbol=symbol,
            cash_qty=args.notional,
        )
        buy_trade = broker.place_order_intent(contract, buy_intent)
        events.emit(
            "order.submitted",
            "submitted round-trip buy",
            order_id=getattr(getattr(buy_trade, "order", None), "orderId", None),
            perm_id=getattr(getattr(buy_trade, "order", None), "permId", None),
            side=buy_intent.side.value,
            order_type=buy_intent.order_type.value,
            cash_qty=buy_intent.cash_qty,
        )
        buy_result = await _wait_for_trade_done(
            broker,
            buy_trade,
            timeout_seconds=args.order_timeout_seconds,
            commission_wait_seconds=args.commission_wait_seconds,
        )
        if output:
            _write_jsonl_record(output, {"type": "buy_order", **buy_result})
        events.emit(
            "order.filled",
            "round-trip buy completed",
            **buy_result["status"],  # type: ignore[arg-type]
            fill=buy_result["fill"],
        )
        buy_fill = buy_result["fill"]
        buy_quantity = float(buy_fill["filled_quantity"])  # type: ignore[index]
        buy_average_price = _finite_number(buy_fill["average_price"])  # type: ignore[index]
        if buy_quantity <= 0 or buy_average_price is None:
            raise SystemExit("round-trip buy did not fill; refusing to submit sell")
        entry_exposed = True

        monitor_result = await _monitor_synthetic_stop(
            broker,
            contract,
            entry_price=buy_average_price,
            notional=args.notional,
            hold_seconds=args.hold_seconds,
            stop_loss_bps=args.synthetic_stop_loss_bps,
            max_loss_usd=args.max_loss_usd,
            monitor_interval_seconds=args.monitor_interval_seconds,
            output=output,
            events=events,
        )
        if output:
            _write_jsonl_record(output, {"type": "synthetic_stop_monitor", **monitor_result})

        sell_intent = CryptoOrderFactory.urgent_market_exit(
            symbol=symbol,
            quantity=round(buy_quantity, 8),
            reason=str(monitor_result["exit_reason"]),
        )
        exit_quote = await broker.snapshot_quote(contract, max_wait_seconds=args.snapshot_timeout_seconds)
        current_position_quantity = await _current_contract_position_quantity(broker, contract, args)
        sell_trade = broker.place_order_intent(
            contract,
            sell_intent,
            reference_price=exit_quote.bid,
            current_position_quantity=current_position_quantity,
        )
        events.emit(
            "order.submitted",
            "submitted round-trip sell",
            order_id=getattr(getattr(sell_trade, "order", None), "orderId", None),
            perm_id=getattr(getattr(sell_trade, "order", None), "permId", None),
            side=sell_intent.side.value,
            order_type=sell_intent.order_type.value,
            quantity=sell_intent.quantity,
            reason=sell_intent.reason,
            reference_bid=exit_quote.bid,
            verified_position_quantity=current_position_quantity,
        )
        sell_result = await _wait_for_trade_done(
            broker,
            sell_trade,
            timeout_seconds=args.order_timeout_seconds,
            commission_wait_seconds=args.commission_wait_seconds,
        )
        if output:
            _write_jsonl_record(output, {"type": "sell_order", **sell_result})
        events.emit(
            "order.filled",
            "round-trip sell completed",
            **sell_result["status"],  # type: ignore[arg-type]
            fill=sell_result["fill"],
        )
        sell_fill = sell_result["fill"]
        sell_quantity = float(sell_fill["filled_quantity"])  # type: ignore[index]
        if sell_quantity <= 0:
            raise SystemExit("round-trip sell did not fill")
        if sell_quantity + 1e-8 < buy_quantity:
            raise SystemExit(
                "round-trip sell filled less than the buy quantity; check TWS positions before rerunning"
            )
        entry_exposed = False

        order_ids = {
            value
            for value in (
                buy_result["status"].get("order_id"),  # type: ignore[union-attr]
                sell_result["status"].get("order_id"),  # type: ignore[union-attr]
            )
            if isinstance(value, int)
        }
        perm_ids = {
            value
            for value in (
                buy_result["status"].get("perm_id"),  # type: ignore[union-attr]
                sell_result["status"].get("perm_id"),  # type: ignore[union-attr]
            )
            if isinstance(value, int)
        }
        session_state = _broker_session_state_payload(
            broker,
            related_order_ids=order_ids,
            related_perm_ids=perm_ids,
        )
        if output:
            _write_jsonl_record(output, {"type": "session_state", **session_state})
        events.emit(
            "broker.session_state",
            "captured related TWS trades/fills/executions",
            related_trade_count=session_state.get("related_trade_count"),
            related_fill_count=session_state.get("related_fill_count"),
            related_execution_count=session_state.get("related_execution_count"),
            related_order_ids=session_state.get("related_order_ids"),
            related_perm_ids=session_state.get("related_perm_ids"),
        )

        final_snapshot = await _paper_account_snapshot(broker, contract, args)
        if output:
            loss_delta = _snapshot_loss_delta(baseline=baseline, current=final_snapshot)
            _write_jsonl_record(
                output,
                {"type": "final_snapshot", "loss_delta_usd": loss_delta, **final_snapshot},
            )
        quote = final_snapshot["quote"] if isinstance(final_snapshot.get("quote"), dict) else {}
        events.emit(
            "account.final",
            "captured final account snapshot",
            account=final_snapshot.get("account"),
            bid=quote.get("bid"),
            ask=quote.get("ask"),
            spread_bps=quote.get("spread_bps"),
            loss_delta_usd=_snapshot_loss_delta(baseline=baseline, current=final_snapshot),
            positions=_position_summary(final_snapshot),
            **_account_tags(final_snapshot, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
        )
        _require_tws_fill_accounting("round-trip buy", buy_result)
        _require_tws_fill_accounting("round-trip sell", sell_result)
    except BaseException as exc:
        if contract is not None and entry_exposed:
            await _attempt_emergency_cleanup(
                broker,
                contract,
                args,
                reason=f"{type(exc).__name__}: {exc}",
                output=output,
                events=events,
            )
        raise
    finally:
        await broker.disconnect()
        events.emit("broker.disconnected", "disconnected from IBKR Gateway/TWS")
        if buy_result is None or sell_result is None:
            events.close()

    if buy_result is None or sell_result is None or monitor_result is None or baseline is None:
        raise SystemExit("round-trip test did not complete")
    buy_fill = buy_result["fill"]
    sell_fill = sell_result["fill"]
    buy_quantity = float(buy_fill["filled_quantity"])  # type: ignore[index]
    sell_quantity = float(sell_fill["filled_quantity"])  # type: ignore[index]
    buy_average_price = _finite_number(buy_fill["average_price"]) or 0.0  # type: ignore[index]
    sell_average_price = _finite_number(sell_fill["average_price"]) or 0.0  # type: ignore[index]
    matched_quantity = min(buy_quantity, sell_quantity)
    fill_gross_pnl = matched_quantity * (sell_average_price - buy_average_price)
    ibkr_commission = float(buy_fill["commission"]) + float(sell_fill["commission"])  # type: ignore[index]
    fill_net_pnl = fill_gross_pnl - ibkr_commission
    ibkr_realized_pnl = float(buy_fill["realized_pnl_reported"]) + float(  # type: ignore[index]
        sell_fill["realized_pnl_reported"]  # type: ignore[index]
    )
    account_loss_delta = (
        _snapshot_loss_delta(baseline=baseline, current=final_snapshot)
        if final_snapshot is not None
        else None
    )
    account_pnl_delta = -account_loss_delta if account_loss_delta is not None else None
    fill_pnl_reconciliation_delta = fill_net_pnl - ibkr_realized_pnl
    account_pnl_reconciliation_delta = (
        fill_net_pnl - account_pnl_delta if account_pnl_delta is not None else None
    )
    payload = {
        "ok": True,
        "paper_mode": True,
        "pnl_source": "IBKR TWS executions, commission reports, realized PnL reports, and account snapshots",
        "symbol": f"{contract.symbol}/{contract.currency}",
        "exchange": contract.exchange,
        "con_id": contract.con_id,
        "notional": args.notional,
        "hold_seconds": args.hold_seconds,
        "synthetic_stop_loss_bps": args.synthetic_stop_loss_bps,
        "exit_reason": monitor_result["exit_reason"],
        "buy_status": buy_result["status"],
        "sell_status": sell_result["status"],
        "buy_fill": buy_fill,
        "sell_fill": sell_fill,
        "matched_quantity": matched_quantity,
        "fill_gross_pnl_from_tws_executions": fill_gross_pnl,
        "ibkr_commission_from_tws_reports": ibkr_commission,
        "fill_net_pnl_from_tws_reports": fill_net_pnl,
        "ibkr_realized_pnl": ibkr_realized_pnl,
        "fill_pnl_reconciliation_delta": fill_pnl_reconciliation_delta,
        "fill_pnl_reconciliation_passed": abs(fill_pnl_reconciliation_delta) <= 0.01,
        "account_loss_delta_usd": account_loss_delta,
        "account_pnl_delta_usd": account_pnl_delta,
        "account_pnl_reconciliation_delta": account_pnl_reconciliation_delta,
        "account_pnl_reconciliation_passed": (
            abs(account_pnl_reconciliation_delta) <= 0.10
            if account_pnl_reconciliation_delta is not None
            else False
        ),
        "output": str(output) if output else "",
    }
    events.emit(
        "pnl.reconciled",
        "round-trip TWS accounting reconciled",
        fill_net_pnl_from_tws_reports=fill_net_pnl,
        ibkr_commission_from_tws_reports=ibkr_commission,
        ibkr_realized_pnl=ibkr_realized_pnl,
        account_pnl_delta_usd=account_pnl_delta,
        fill_pnl_reconciliation_delta=fill_pnl_reconciliation_delta,
        account_pnl_reconciliation_delta=account_pnl_reconciliation_delta,
        fill_pnl_reconciliation_passed=payload["fill_pnl_reconciliation_passed"],
        account_pnl_reconciliation_passed=payload["account_pnl_reconciliation_passed"],
    )
    events.emit(
        "broker.round_trip.finished",
        "controlled paper round trip finished",
        exit_reason=monitor_result["exit_reason"],
        output=str(output) if output else "",
    )
    events.close()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_broker_round_trip_test(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_round_trip_test_async(args))


def _validate_trade_run_config(cfg, args: argparse.Namespace) -> None:
    if cfg.runtime.mode not in {RuntimeMode.PAPER, RuntimeMode.LIVE}:
        raise SystemExit("broker trade-run requires paper or live runtime mode")
    _require_explicit_broker_account(cfg, "broker trade-run")
    if args.capital_usd <= 0:
        raise SystemExit("broker trade-run requires positive --capital-usd")
    if args.max_loss_usd <= 0:
        raise SystemExit("broker trade-run requires positive --max-loss-usd")
    if args.max_loss_usd > args.capital_usd:
        raise SystemExit("broker trade-run max loss cannot exceed capital")
    if args.max_order_notional_usd <= 0:
        raise SystemExit("broker trade-run requires positive --max-order-notional-usd")
    if args.max_order_notional_usd > args.capital_usd:
        raise SystemExit("broker trade-run max order notional cannot exceed capital")
    configured_cap = (
        cfg.risk.live_max_notional
        if cfg.runtime.mode == RuntimeMode.LIVE
        else cfg.risk.paper_max_notional
    )
    if args.max_order_notional_usd > configured_cap:
        raise SystemExit(
            f"broker trade-run max order notional {args.max_order_notional_usd:.2f} exceeds "
            f"configured {cfg.runtime.mode.value} max notional {configured_cap:.2f}"
        )
    if args.duration_seconds <= 0 or args.signal_interval <= 0:
        raise SystemExit("broker trade-run requires positive duration and signal interval")
    if getattr(args, "account_refresh_interval_seconds", 30.0) <= 0:
        raise SystemExit("broker trade-run requires positive --account-refresh-interval-seconds")
    if getattr(args, "max_scoring_samples", 20) <= 0:
        raise SystemExit("broker trade-run requires positive --max-scoring-samples")
    if getattr(args, "dense_stride_bars", 1) <= 0:
        raise SystemExit("broker trade-run requires positive --dense-stride-bars")
    if getattr(args, "history_max_bars", 12_000) <= 0:
        raise SystemExit("broker trade-run requires positive --history-max-bars")
    if getattr(args, "live_1s_warmup_bars", 2) <= 0:
        raise SystemExit("broker trade-run requires positive --live-1s-warmup-bars")
    if getattr(args, "live_1s_warmup_timeout_seconds", 8.0) <= 0:
        raise SystemExit("broker trade-run requires positive --live-1s-warmup-timeout-seconds")
    if getattr(args, "max_position_hold_seconds", 0.0) < 0:
        raise SystemExit("broker trade-run requires nonnegative --max-position-hold-seconds")
    if getattr(args, "synthetic_profit_target_bps", 0.0) < 0:
        raise SystemExit("broker trade-run requires nonnegative --synthetic-profit-target-bps")
    if args.synthetic_stop_loss_bps <= 0:
        raise SystemExit("broker trade-run requires positive --synthetic-stop-loss-bps")
    if getattr(args, "decision_threshold", 0.0) < 0 or getattr(args, "decision_threshold", 0.0) > 1:
        raise SystemExit("broker trade-run requires --decision-threshold between 0 and 1")
    if not 0 <= getattr(args, "max_missing_model_feature_fraction", 0.0) <= 1:
        raise SystemExit("broker trade-run requires --max-missing-model-feature-fraction between 0 and 1")
    if not args.model_artifact:
        raise SystemExit("broker trade-run requires --model-artifact")
    if cfg.runtime.mode == RuntimeMode.PAPER:
        if cfg.broker.port not in {4002, 7497}:
            raise SystemExit("broker trade-run paper mode requires a standard IBKR paper port (4002 or 7497)")
        if args.confirm != "IBKR_PAPER_TRADE_RUN":
            raise SystemExit('broker trade-run paper mode requires --confirm IBKR_PAPER_TRADE_RUN')
    else:
        if cfg.broker.port not in {4001, 7496}:
            raise SystemExit("broker trade-run live mode requires a standard IBKR live port (4001 or 7496)")
        if not cfg.runtime.enable_live_trading or cfg.runtime.live_confirmation != "ZEROALPHA_LIVE":
            raise SystemExit("broker trade-run live mode requires enable_live_trading and ZEROALPHA_LIVE config")
        if args.confirm != "ZEROALPHA_LIVE_TRADE_RUN":
            raise SystemExit('broker trade-run live mode requires --confirm ZEROALPHA_LIVE_TRADE_RUN')


async def _submit_market_exit(
    broker,
    contract,
    args: argparse.Namespace,
    *,
    reason: str,
    events: RuntimeEventStream,
    output: Path | None = None,
    quote=None,
    position_quantity: float | None = None,
    position_quantity_source: str = "broker_positions",
) -> dict[str, object]:
    from zeroalpha.execution.orders import CryptoOrderFactory

    quote = quote or await broker.snapshot_quote(contract, max_wait_seconds=args.snapshot_timeout_seconds)
    current_position_quantity = (
        _finite_number(position_quantity)
        if position_quantity is not None
        else await _current_contract_position_quantity(broker, contract, args)
    )
    if current_position_quantity is None:
        current_position_quantity = 0.0
    if current_position_quantity <= 1e-8:
        return {
            "submitted": False,
            "reason": "flat_position",
            "position_quantity": current_position_quantity,
            "position_quantity_source": position_quantity_source,
        }
    intent = CryptoOrderFactory.urgent_market_exit(
        symbol=f"{contract.symbol}/{contract.currency}",
        quantity=round(current_position_quantity, 8),
        reason=reason,
    )
    trade = broker.place_order_intent(
        contract,
        intent,
        reference_price=quote.bid,
        current_position_quantity=current_position_quantity,
    )
    events.emit(
        "order.submitted",
        "submitted market exit",
        side=intent.side.value,
        order_type=intent.order_type.value,
        quantity=intent.quantity,
        reason=reason,
        reference_bid=quote.bid,
        verified_position_quantity=current_position_quantity,
        position_quantity_source=position_quantity_source,
        order_id=getattr(getattr(trade, "order", None), "orderId", None),
        perm_id=getattr(getattr(trade, "order", None), "permId", None),
    )
    result = await _wait_for_trade_done(
        broker,
        trade,
        timeout_seconds=args.order_timeout_seconds,
        commission_wait_seconds=args.commission_wait_seconds,
    )
    payload = {
        "submitted": True,
        "reason": reason,
        "quote": asdict(quote),
        "position_quantity_source": position_quantity_source,
        **result,
    }
    if output:
        _write_jsonl_record(output, {"type": "market_exit", **payload})
    return payload


def _history_bar_seconds(bar_size: str) -> int:
    from zeroalpha.broker.ibkr import _bar_size_delta

    seconds = int(_bar_size_delta(bar_size).total_seconds())
    return max(1, seconds)


def _one_second_history_report(bars: list) -> dict[str, object]:
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    diffs = [
        (ordered[idx].timestamp_utc - ordered[idx - 1].timestamp_utc).total_seconds()
        for idx in range(1, len(ordered))
    ]
    recent = [value for value in diffs[-120:] if value > 0]
    if recent:
        ordered_gaps = sorted(recent)
        mid = len(ordered_gaps) // 2
        median_gap = (
            ordered_gaps[mid]
            if len(ordered_gaps) % 2
            else (ordered_gaps[mid - 1] + ordered_gaps[mid]) / 2
        )
    else:
        median_gap = 0.0
    max_gap = max(recent) if recent else 0.0
    ibkr_sources = [bar.source for bar in ordered if str(bar.source).upper().startswith("IBKR")]
    return {
        "bars": len(ordered),
        "ibkr_bars": len(ibkr_sources),
        "start": ordered[0].timestamp_utc.isoformat() if ordered else "",
        "end": ordered[-1].timestamp_utc.isoformat() if ordered else "",
        "bar_size_seconds": _history_bar_seconds(ordered[-1].bar_size) if ordered else 0,
        "median_recent_gap_seconds": median_gap,
        "max_recent_gap_seconds": max_gap,
        "source_examples": sorted(set(ibkr_sources))[:5],
        "ok": (
            len(ordered) >= 30
            and len(ibkr_sources) == len(ordered)
            and ordered[-1].bar_size.strip().lower() in {"1 secs", "1 sec", "1 second", "1 seconds"}
            and 0.9 <= median_gap <= 1.1
            and max_gap <= 2.0
        ),
    }


def _require_verified_one_second_data(args: argparse.Namespace, bars: list) -> dict[str, object]:
    report = _one_second_history_report(bars)
    if not getattr(args, "require_live_1s_data", True):
        return report
    if getattr(args, "live_data_mode", "streaming") != "streaming":
        raise SystemExit("--require-live-1s-data requires --live-data-mode streaming")
    tick_by_tick_type = getattr(args, "tick_by_tick_type", "Last")
    if tick_by_tick_type == "none":
        raise SystemExit("--require-live-1s-data requires an IBKR tick-by-tick subscription")
    what_to_show = str(getattr(args, "history_what_to_show", "") or "").strip().upper()
    if what_to_show in {"AGGTRADES", "TRADES"} and tick_by_tick_type not in {"Last", "AllLast"}:
        raise SystemExit(
            "--require-live-1s-data with AGGTRADES/TRADES requires "
            "--tick-by-tick-type Last or AllLast"
        )
    if _history_bar_seconds(args.history_bar_size) != 1:
        raise SystemExit("--require-live-1s-data requires --history-bar-size '1 secs'")
    if not report["ok"]:
        raise SystemExit(
            "IBKR 1-second data verification failed: "
            f"bars={report['bars']} ibkr_bars={report['ibkr_bars']} "
            f"bar_size_seconds={report['bar_size_seconds']} "
            f"median_gap={report['median_recent_gap_seconds']} "
            f"max_gap={report['max_recent_gap_seconds']} "
            f"sources={report['source_examples']}"
        )
    return report


async def _warmup_live_one_second_stream(
    broker,
    contract,
    args: argparse.Namespace,
    *,
    quote_ticker,
    tick_subscription: tuple[str, object] | None,
    bar_aggregator,
    history_bars: list,
) -> tuple[list, dict[str, object]]:
    required_bars = max(1, int(getattr(args, "live_1s_warmup_bars", 2)))
    timeout_seconds = max(1.0, float(getattr(args, "live_1s_warmup_timeout_seconds", 8.0)))
    deadline = time.monotonic() + timeout_seconds
    completed: list = []
    quote_samples = 0
    tick_rows_seen = 0
    processed_tick_rows = 0
    requires_tick_by_tick = tick_subscription is not None and getattr(args, "require_live_1s_data", True)
    tick_completed_bars = 0
    while (
        time.monotonic() < deadline
        and (
            len(completed) < required_bars
            or (requires_tick_by_tick and tick_completed_bars < required_bars)
        )
    ):
        quote = await _trade_run_quote(
            broker,
            contract,
            args,
            quote_ticker=quote_ticker,
        )
        quote_samples += 1
        bar_aggregator.add_quote(quote)
        if tick_subscription is not None:
            ticker = tick_subscription[1]
            tick_rows_seen = max(tick_rows_seen, len(getattr(ticker, "tickByTicks", []) or []))
            processed_tick_rows += bar_aggregator.process_ticker_ticks(ticker)
        appended = bar_aggregator.completed_bars()
        if appended:
            completed.extend(appended)
            tick_completed_bars += sum(
                1
                for bar in appended
                if (_finite_number(bar.extra.get("tick_count")) or 0.0) > 0
            )
            history_bars = _trim_history_bars(
                [*history_bars, *appended],
                max_bars=args.history_max_bars,
            )
        remaining = deadline - time.monotonic()
        warmup_ready = len(completed) >= required_bars and (
            not requires_tick_by_tick or tick_completed_bars >= required_bars
        )
        if remaining > 0 and not warmup_ready:
            await broker.wait(min(0.20, remaining))
    latest = completed[-1] if completed else None
    aggregated_from: dict[str, int] = {}
    tick_completed_bars = 0
    for bar in completed:
        source = str(bar.extra.get("aggregated_from", "unknown"))
        aggregated_from[source] = aggregated_from.get(source, 0) + 1
        tick_count = _finite_number(bar.extra.get("tick_count"))
        if tick_count is not None and tick_count > 0:
            tick_completed_bars += 1
    ok = len(completed) >= required_bars
    if requires_tick_by_tick:
        ok = ok and processed_tick_rows > 0 and tick_completed_bars >= required_bars
    report = {
        "ok": ok,
        "required_bars": required_bars,
        "completed_bars": len(completed),
        "quote_samples": quote_samples,
        "tick_by_tick_type": tick_subscription[0] if tick_subscription is not None else "none",
        "tick_rows_seen": tick_rows_seen,
        "processed_tick_rows": processed_tick_rows,
        "tick_completed_bars": tick_completed_bars,
        "aggregated_from": aggregated_from,
        "latest_timestamp_utc": latest.timestamp_utc.isoformat() if latest else "",
        "latest_bar_size_seconds": _history_bar_seconds(latest.bar_size) if latest else 0,
        "latest_source": latest.source if latest else "",
    }
    if not report["ok"]:
        raise SystemExit(
            "IBKR live 1-second streaming verification failed: "
            f"completed_bars={report['completed_bars']} required_bars={required_bars} "
            f"quote_samples={quote_samples} tick_rows_seen={tick_rows_seen} "
            f"processed_tick_rows={processed_tick_rows} "
            f"tick_completed_bars={tick_completed_bars} "
            f"timeout_seconds={timeout_seconds}"
        )
    return history_bars, report


def _trim_history_bars(bars: list, *, max_bars: int) -> list:
    deduped = {bar.timestamp_utc: bar for bar in bars}
    ordered = sorted(deduped.values(), key=lambda bar: bar.timestamp_utc)
    if max_bars <= 0:
        return ordered
    return ordered[-max_bars:]


def _kill_switch_enabled(cfg) -> bool:
    try:
        return Path(cfg.runtime.kill_switch_file).expanduser().exists()
    except OSError:
        return False


async def _trade_run_quote(
    broker,
    contract,
    args: argparse.Namespace,
    *,
    quote_ticker=None,
):
    if quote_ticker is None:
        return await broker.snapshot_quote(contract, max_wait_seconds=args.snapshot_timeout_seconds)
    return await broker.quote_from_ticker(
        contract,
        quote_ticker,
        max_wait_seconds=args.snapshot_timeout_seconds,
    )


def _runner_hold_seconds(cfg, args: argparse.Namespace, sample=None) -> float:
    override = getattr(args, "max_position_hold_seconds", 0.0) or 0.0
    if override > 0:
        return override
    if sample is not None:
        value = sample.features.get("max_holding_seconds")
        if isinstance(value, int | float) and float(value) > 0:
            return float(value)
    if cfg.labels.max_holding_seconds is not None:
        return float(cfg.labels.max_holding_seconds)
    return float(cfg.labels.max_holding_hours) * 3600


def _runner_min_holding_seconds(sample=None) -> float:
    if sample is None:
        return 0.0
    value = sample.features.get(
        "event_min_holding_seconds",
        sample.features.get("min_holding_seconds", 0.0),
    )
    minimum = _finite_number(value)
    return minimum if minimum is not None and minimum > 0 else 0.0


def _runner_exit_timing(
    *,
    entered_monotonic: float,
    entered_utc: datetime,
    hold_seconds: float,
    min_holding_seconds: float,
) -> dict[str, object]:
    min_holding_seconds = max(0.0, min_holding_seconds)
    hold_seconds = max(0.0, hold_seconds)
    exit_delay_seconds = max(hold_seconds, min_holding_seconds)
    return {
        "min_exit_monotonic": entered_monotonic + min_holding_seconds,
        "min_exit_utc": (entered_utc + timedelta(seconds=min_holding_seconds)).isoformat(),
        "exit_deadline_monotonic": entered_monotonic + exit_delay_seconds,
        "exit_deadline_utc": (entered_utc + timedelta(seconds=exit_delay_seconds)).isoformat(),
    }


def _runner_exit_bps(args: argparse.Namespace, sample=None) -> tuple[float, float, str]:
    stop_bps = float(args.synthetic_stop_loss_bps)
    profit_bps = float(getattr(args, "synthetic_profit_target_bps", 0.0) or 0.0)
    if not getattr(args, "use_model_exit_geometry", True) or sample is None:
        return stop_bps, profit_bps, "cli_synthetic"
    sample_stop = _finite_number(sample.features.get("gross_stop_distance"))
    sample_profit = _finite_number(sample.features.get("gross_profit_move"))
    fallback_stop = _finite_number(sample.features.get("net_stop_loss"))
    fallback_profit = _finite_number(sample.features.get("net_profit_target"))
    source_parts: list[str] = []
    if sample_stop is not None and sample_stop > 0:
        stop_bps = max(sample_stop * 10_000, 1.0)
        source_parts.append("model_gross_stop")
    elif fallback_stop is not None and fallback_stop > 0:
        stop_bps = max(fallback_stop * 10_000, 1.0)
        source_parts.append("model_net_stop")
    if sample_profit is not None and sample_profit > 0:
        profit_bps = max(sample_profit * 10_000, 1.0)
        source_parts.append("model_gross_profit")
    elif fallback_profit is not None and fallback_profit > 0:
        profit_bps = max(fallback_profit * 10_000, 1.0)
        source_parts.append("model_net_profit")
    return stop_bps, profit_bps, "+".join(source_parts) if source_parts else "cli_synthetic"


def _decision_threshold_override(args: argparse.Namespace) -> float | None:
    value = float(getattr(args, "decision_threshold", 0.0) or 0.0)
    return value if value > 0.0 else None


def _artifact_policy_dict(training_config: dict[str, object], key: str) -> dict[str, object]:
    policy = training_config.get(key)
    return policy if isinstance(policy, dict) else {}


def _artifact_entry_order_model(training_config: dict[str, object]) -> str:
    execution_policy = _artifact_policy_dict(training_config, "execution_policy")
    value = execution_policy.get("entry_order_model", training_config.get("entry_order_model", ""))
    return str(value or "")


def _artifact_sizing_mode(training_config: dict[str, object]) -> str:
    sizing_policy = _artifact_policy_dict(training_config, "sizing_policy")
    value = sizing_policy.get("mode", training_config.get("sizing_mode", "fixed"))
    return str(value or "fixed")


def _artifact_policy_float(
    training_config: dict[str, object],
    policy: dict[str, object],
    *,
    policy_key: str,
    legacy_key: str,
    default: float,
) -> float:
    value = _finite_number(policy.get(policy_key, training_config.get(legacy_key, default)))
    return float(default) if value is None else value


def _artifact_score_field(score, sample, field: str) -> float:
    if field == "probability":
        return float(score.probability)
    if field == "expected_value":
        return float(score.expected_value)
    if field == "selection_score":
        return float(score.selection_score)
    if field == "trade_score":
        return float(getattr(score, "trade_score", score.selection_score))
    if field == "predicted_return":
        value = _finite_number(getattr(sample, "features", {}).get("predicted_return"))
        return float(score.expected_value) if value is None else value
    return float(score.probability)


def _directed_sizing_score(value: float, direction: str) -> float:
    if direction == "low":
        return -value
    return value


def _sample_spread_bps_for_sizing(sample) -> float | None:
    spreads = [
        float(value)
        for key, value in getattr(sample, "features", {}).items()
        if key.endswith("spread_bps")
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0
    ]
    return min(spreads) if spreads else None


def _sample_liquidity_score_for_sizing(sample) -> float:
    features = getattr(sample, "features", {})
    values: list[float] = []
    for key in (
        "pm_leading_liquidity_weight_total",
        "pm_leading_liquidity_weight_mean",
        "ibkr_top_of_book_size_log",
        "ibkr_futures_top_of_book_size_log",
        "dollar_volume_log",
    ):
        value = features.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return max(values) if values else 0.0


def _artifact_confidence_notional_scale(training_config: dict[str, object], sample, score) -> float:
    expected_value = float(getattr(score, "expected_value", 0.0) or 0.0)
    if expected_value <= 0:
        return 0.25
    probability = float(getattr(score, "probability", 0.0) or 0.0)
    probability_threshold = max(float(training_config.get("minimum_probability", 0.0) or 0.0), 1e-9)
    probability_room = max(1.0 - probability_threshold, 1e-9)
    probability_scale = max(0.0, (probability - probability_threshold) / probability_room)
    features = getattr(sample, "features", {})
    label_target = _finite_number(features.get("net_profit_target"))
    if label_target is None:
        label_target = float(training_config.get("label_net_profit_target", 0.0) or 0.0)
    label_stop = _finite_number(features.get("net_stop_loss"))
    if label_stop is None:
        label_stop = float(training_config.get("label_net_stop_loss", 0.0) or 0.0)
    geometry_floor = max(min(label_target, label_stop) * 0.25, 1e-6)
    ev_floor = max(float(training_config.get("minimum_expected_value", 0.0) or 0.0), geometry_floor)
    ev_scale = max(0.0, expected_value / ev_floor)
    confidence = min(1.0, probability_scale, ev_scale)
    return max(0.25, min(1.0, 0.25 + 0.75 * confidence))


def _trade_run_order_notional_from_artifact_policy(
    args: argparse.Namespace,
    training_config: dict[str, object],
    sample,
    score,
    *,
    available_notional: float,
) -> tuple[float, str]:
    cap = max(min(float(getattr(args, "max_order_notional_usd", 0.0)), available_notional), 0.0)
    if cap <= 0:
        return 0.0, "no_available_notional"

    sizing_policy = _artifact_policy_dict(training_config, "sizing_policy")
    mode = _artifact_sizing_mode(training_config)
    if mode == "confidence":
        return cap * _artifact_confidence_notional_scale(training_config, sample, score), "artifact_confidence"
    if mode not in {"score_bucket", "liquidity_score_bucket"}:
        return cap, f"artifact_{mode}"

    score_field = str(
        sizing_policy.get("score_field", training_config.get("sizing_score_field", "probability"))
        or "probability"
    )
    score_direction = str(
        sizing_policy.get(
            "score_direction",
            training_config.get("sizing_score_direction", "high"),
        )
        or "high"
    )
    base = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="base_notional",
        legacy_key="sizing_base_notional",
        default=0.0,
    )
    mid = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="mid_notional",
        legacy_key="sizing_mid_notional",
        default=0.0,
    )
    high = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="high_notional",
        legacy_key="sizing_high_notional",
        default=0.0,
    )
    mid_score = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="mid_score",
        legacy_key="sizing_mid_score",
        default=0.45,
    )
    high_score = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="high_score",
        legacy_key="sizing_high_score",
        default=0.90,
    )
    max_spread_bps = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="max_spread_bps",
        legacy_key="sizing_max_spread_bps",
        default=1.0,
    )
    min_liquidity_score = _artifact_policy_float(
        training_config,
        sizing_policy,
        policy_key="min_liquidity_score",
        legacy_key="sizing_min_liquidity_score",
        default=0.0,
    )

    base = min(base if base > 0 else min(5_000.0, cap), cap)
    mid = min(max(mid if mid > 0 else max(base, (base + cap) / 2), base), cap)
    high = min(max(high if high > 0 else cap, mid), cap)
    sizing_score = _directed_sizing_score(
        _artifact_score_field(score, sample, score_field),
        score_direction,
    )
    spread = _sample_spread_bps_for_sizing(sample)
    spread_ok = spread is None or max_spread_bps <= 0 or spread <= max_spread_bps
    liquidity_ok = (
        min_liquidity_score <= 0
        or _sample_liquidity_score_for_sizing(sample) >= min_liquidity_score
    )
    if sizing_score >= high_score and (mode == "score_bucket" or (spread_ok and liquidity_ok)):
        return high, f"artifact_{mode}_high"
    if sizing_score >= mid_score:
        return mid, f"artifact_{mode}_mid"
    return base, f"artifact_{mode}_base"


async def _broker_trade_run_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker
    from zeroalpha.broker.streaming import TickBarAggregator
    from zeroalpha.execution.orders import CryptoOrderFactory

    cfg = _override_broker_client_id(
        _override_config_from_args(load_config(args.config), args),
        args,
    )
    _validate_trade_run_config(cfg, args)
    artifact = load_production_model_artifact(Path(args.model_artifact))
    artifact_training_config = getattr(artifact, "training_config", {}) or {}
    artifact_feature_asof = str(artifact_training_config.get("feature_asof", "signal") or "signal")
    artifact_external_latency_seconds = float(
        artifact_training_config.get("external_feature_latency_seconds", 0.0) or 0.0
    )
    artifact_target_frequency_mode = str(
        artifact_training_config.get("target_frequency_mode", "") or ""
    )
    artifact_order_model = _artifact_entry_order_model(artifact_training_config)
    artifact_sizing_mode = _artifact_sizing_mode(artifact_training_config)
    threshold_override = _decision_threshold_override(args)
    broker = IBKRBroker(cfg)
    output = Path(args.state_log) if args.state_log else None
    stop_reason = "duration_complete"
    snapshots = 0
    signals = 0
    entries = 0
    exits = 0
    max_observed_loss = 0.0
    open_entries: list[dict[str, object]] = []
    entry_exposed = False
    last_loss_delta: float | None = None
    last_position_quantity = 0.0
    history_bars = []
    context_bars = {
        key: _trim_history_bars(read_ibkr_bars(path), max_bars=getattr(args, "history_max_bars", 12_000))
        for key, path in _named_paths(getattr(args, "context_bars_jsonl", "")).items()
    }
    traded_event_ids: set[str] = set()
    quote_ticker = None
    tick_subscription: tuple[str, object] | None = None
    contract = None
    with _runtime_event_stream_from_args(args, "broker.trade_run") as events:
        events.emit(
            "broker.trade_run.start",
            "starting autonomous IBKR paper/live-gated runner",
            mode=cfg.runtime.mode.value,
            account=cfg.broker.account,
            client_id=cfg.broker.client_id,
            model_artifact=args.model_artifact,
            capital_usd=args.capital_usd,
            max_loss_usd=args.max_loss_usd,
            max_order_notional_usd=args.max_order_notional_usd,
            duration_seconds=args.duration_seconds,
            signal_interval=args.signal_interval,
            live_data_mode=getattr(args, "live_data_mode", "streaming"),
            history_bar_size=args.history_bar_size,
            tick_by_tick_type=getattr(args, "tick_by_tick_type", "Last"),
            adaptive_horizon=getattr(args, "adaptive_horizon", False),
            use_model_exit_geometry=getattr(args, "use_model_exit_geometry", True),
            artifact_selected_threshold=artifact.selected_threshold,
            artifact_feature_asof=artifact_feature_asof,
            artifact_target_frequency_mode=artifact_target_frequency_mode,
            artifact_entry_order_model=artifact_order_model,
            artifact_sizing_mode=artifact_sizing_mode,
            decision_threshold_override=threshold_override,
            candidate_mode=args.candidate_mode,
            dense_stride_bars=getattr(args, "dense_stride_bars", 1),
            max_scoring_samples=args.max_scoring_samples,
            context_sources={key: len(value) for key, value in context_bars.items()},
        )
        if artifact_order_model != "market":
            events.emit(
                "model.execution_policy_mismatch",
                "refusing trade-run because artifact was not explicitly trained for market-entry live execution",
                artifact_entry_order_model=artifact_order_model,
                runner_entry_order_model="market",
                priority="critical",
            )
            return 2
        contract_errors = _live_artifact_contract_errors(artifact_training_config, args)
        if contract_errors:
            events.emit(
                "model.artifact_contract_mismatch",
                "refusing trade-run because artifact is missing live-valid strategy contract metadata",
                errors=contract_errors,
                priority="critical",
            )
            return 2
        await broker.connect(read_only=False)
        try:
            contract = await broker.qualify_crypto_contract()
            live_data_mode = getattr(args, "live_data_mode", "streaming")
            if live_data_mode == "streaming":
                quote_ticker = broker.subscribe_quote(contract)
                tick_type = getattr(args, "tick_by_tick_type", "Last")
                if tick_type != "none":
                    tick_subscription = (
                        tick_type,
                        broker.subscribe_tick_by_tick(
                            contract,
                            tick_type=tick_type,
                            ignore_size=False,
                        ),
                    )
            bar_aggregator = (
                TickBarAggregator(
                    symbol=f"{contract.symbol}/{contract.currency}",
                    bar_size_seconds=_history_bar_seconds(args.history_bar_size),
                    source=f"IBKR:STREAM_{getattr(args, 'tick_by_tick_type', 'quote')}",
                )
                if live_data_mode == "streaming"
                else None
            )
            history_bars = await broker.historical_bars(
                contract,
                end=datetime.now(tz=UTC),
                duration=args.history_duration,
                bar_size=args.history_bar_size,
                what_to_show=args.history_what_to_show,
            )
            history_bars = _trim_history_bars(
                history_bars,
                max_bars=getattr(args, "history_max_bars", 12_000),
            )
            one_second_report = _require_verified_one_second_data(args, history_bars)
            events.emit(
                "market.one_second_data_verified",
                "verified IBKR one-second bars for model input",
                **one_second_report,
            )
            if getattr(args, "require_live_1s_data", True) and bar_aggregator is not None:
                history_bars, live_one_second_report = await _warmup_live_one_second_stream(
                    broker,
                    contract,
                    args,
                    quote_ticker=quote_ticker,
                    tick_subscription=tick_subscription,
                    bar_aggregator=bar_aggregator,
                    history_bars=history_bars,
                )
                events.emit(
                    "market.live_one_second_stream_verified",
                    "verified completed live one-second streaming bars for model input",
                    **live_one_second_report,
                )
            baseline_quote = await _trade_run_quote(
                broker,
                contract,
                args,
                quote_ticker=quote_ticker,
            )
            baseline = await _paper_account_snapshot(broker, contract, args, quote=baseline_quote)
            last_position_quantity = _position_quantity_from_snapshot(baseline, contract)
            last_loss_delta = 0.0
            max_runner_open_positions = max(1, int(getattr(cfg.risk, "max_open_positions", 1)))
            if output:
                _write_jsonl_record(output, {"type": "baseline", **baseline})
            events.emit(
                "account.baseline",
                "captured trade-run baseline account snapshot",
                account=baseline.get("account"),
                positions=_position_summary(baseline),
                position_quantity=last_position_quantity,
                history_bars=len(history_bars),
                max_open_positions=max_runner_open_positions,
                history_start=history_bars[0].timestamp_utc.isoformat() if history_bars else "",
                history_end=history_bars[-1].timestamp_utc.isoformat() if history_bars else "",
                **_account_tags(baseline, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
            )
            if context_bars:
                events.emit(
                    "market.context_loaded",
                    "loaded static context bar files for live model features",
                    context_sources={key: len(value) for key, value in context_bars.items()},
                    context_start={
                        key: value[0].timestamp_utc.isoformat()
                        for key, value in context_bars.items()
                        if value
                    },
                    context_end={
                        key: value[-1].timestamp_utc.isoformat()
                        for key, value in context_bars.items()
                        if value
                    },
                )
            deadline = time.monotonic() + args.duration_seconds
            last_account_refresh = time.monotonic()
            while True:
                loop_started = time.monotonic()
                now_remaining = deadline - loop_started
                if now_remaining <= 0:
                    break
                if _kill_switch_enabled(cfg):
                    stop_reason = "kill_switch"
                    events.emit(
                        "risk.kill_switch_triggered",
                        "trade-run kill switch file detected",
                        kill_switch_file=str(cfg.runtime.kill_switch_file),
                        priority="critical",
                    )
                    break
                quote = await _trade_run_quote(
                    broker,
                    contract,
                    args,
                    quote_ticker=quote_ticker,
                )
                if bar_aggregator is not None:
                    bar_aggregator.add_quote(quote)
                    if tick_subscription is not None:
                        bar_aggregator.process_ticker_ticks(tick_subscription[1])
                    completed = bar_aggregator.completed_bars()
                    if completed:
                        history_bars = _trim_history_bars(
                            [*history_bars, *completed],
                            max_bars=args.history_max_bars,
                        )
                        events.emit(
                            "market.bars",
                            "appended completed streaming bars",
                            appended=len(completed),
                            latest_timestamp_utc=history_bars[-1].timestamp_utc.isoformat(),
                            history_bars=len(history_bars),
                            bar_size=args.history_bar_size,
                        )
                elif live_data_mode == "historical_poll":
                    history_bars = await broker.historical_bars(
                        contract,
                        end=datetime.now(tz=UTC),
                        duration=args.history_duration,
                        bar_size=args.history_bar_size,
                        what_to_show=args.history_what_to_show,
                    )
                    history_bars = _trim_history_bars(
                        history_bars,
                        max_bars=args.history_max_bars,
                    )

                bid = quote.bid
                if open_entries and bid is not None:
                    remaining_entries: list[dict[str, object]] = []
                    for open_entry in open_entries:
                        stop_price = float(open_entry["stop_price"])
                        profit_price = _finite_number(open_entry.get("profit_price"))
                        can_exit_barrier = time.monotonic() >= float(
                            open_entry.get("min_exit_monotonic", 0.0)
                        )
                        exit_reason = ""
                        if can_exit_barrier and bid <= stop_price:
                            exit_reason = "synthetic_stop_loss"
                        elif can_exit_barrier and profit_price is not None and bid >= profit_price:
                            exit_reason = "synthetic_profit_target"
                        elif time.monotonic() >= float(open_entry["exit_deadline_monotonic"]):
                            exit_reason = "timed_exit"
                        if not exit_reason:
                            remaining_entries.append(open_entry)
                            continue
                        quantity = float(open_entry["quantity"])
                        exit_payload = await _submit_market_exit(
                            broker,
                            contract,
                            args,
                            reason=exit_reason,
                            events=events,
                            output=output,
                            quote=quote,
                            position_quantity=quantity,
                            position_quantity_source="entry_fill",
                        )
                        exit_fill = exit_payload.get("fill")
                        exit_filled_quantity = (
                            _finite_number(exit_fill.get("filled_quantity"))
                            if isinstance(exit_fill, dict)
                            else None
                        )
                        exit_filled = (
                            bool(exit_payload.get("submitted"))
                            and exit_filled_quantity is not None
                            and exit_filled_quantity > 0
                        )
                        if exit_filled:
                            exits += 1
                            closed_quantity = min(quantity, exit_filled_quantity)
                            last_position_quantity = max(0.0, last_position_quantity - closed_quantity)
                            if closed_quantity + 1e-8 < quantity:
                                updated_entry = dict(open_entry)
                                updated_entry["quantity"] = quantity - closed_quantity
                                updated_entry["cash_qty"] = float(open_entry.get("cash_qty", 0.0)) * (
                                    (quantity - closed_quantity) / quantity
                                )
                                remaining_entries.append(updated_entry)
                        else:
                            remaining_entries.append(open_entry)
                    open_entries = remaining_entries
                    entry_exposed = bool(open_entries)

                should_refresh_account = (
                    loop_started - last_account_refresh >= args.account_refresh_interval_seconds
                )
                if should_refresh_account:
                    snapshot = await _paper_account_snapshot(broker, contract, args, quote=quote)
                    last_account_refresh = time.monotonic()
                    snapshots += 1
                    last_loss_delta = _snapshot_loss_delta(baseline=baseline, current=snapshot)
                    if last_loss_delta is not None:
                        max_observed_loss = max(max_observed_loss, last_loss_delta)
                    last_position_quantity = _position_quantity_from_snapshot(snapshot, contract)
                    if output:
                        _write_jsonl_record(
                            output,
                            {
                                "type": "runner_snapshot",
                                "loss_delta_usd": last_loss_delta,
                                "position_quantity": last_position_quantity,
                                **snapshot,
                            },
                        )
                    events.emit(
                        "account.snapshot",
                        "captured trade-run account snapshot",
                        snapshot_index=snapshots,
                        bid=quote.bid,
                        ask=quote.ask,
                        spread_bps=quote.spread_bps,
                        loss_delta_usd=last_loss_delta,
                        max_observed_loss_usd=max_observed_loss,
                        position_quantity=last_position_quantity,
                        positions=_position_summary(snapshot),
                        **_account_tags(snapshot, ("NetLiquidation", "TotalCashValue", "GrossPositionValue")),
                    )
                elif output:
                    _write_jsonl_record(
                        output,
                        {
                            "type": "runner_tick",
                            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                            "quote": asdict(quote),
                            "position_quantity": last_position_quantity,
                            "open_entries": open_entries,
                            "history_bars": len(history_bars),
                        },
                    )
                tracked_position_quantity = sum(float(entry.get("quantity", 0.0)) for entry in open_entries)
                tracked_open_notional = sum(float(entry.get("cash_qty", 0.0)) for entry in open_entries)
                position_quantity = max(last_position_quantity, tracked_position_quantity)
                if last_loss_delta is not None and last_loss_delta >= args.max_loss_usd:
                    stop_reason = "max_loss_usd"
                    events.emit(
                        "risk.max_loss_triggered",
                        "trade-run max-loss guard triggered",
                        loss_delta_usd=last_loss_delta,
                        max_loss_usd=args.max_loss_usd,
                        priority="critical",
                    )
                    break
                unmanaged_position_open = position_quantity > tracked_position_quantity + 1e-8
                available_entry_slots = max_runner_open_positions - len(open_entries)
                available_notional = max(args.capital_usd - tracked_open_notional, 0.0)
                if deadline - time.monotonic() <= 0:
                    break
                if available_entry_slots > 0 and available_notional >= cfg.risk.minimum_fee_efficient_notional and not unmanaged_position_open:
                    scoring_samples = build_scoring_samples(
                        history_bars,
                        config=cfg,
                        candidate_config=CandidateGenerationConfig(
                            mode=args.candidate_mode,
                            min_history_bars=args.min_history_bars,
                            max_holding_hours=cfg.labels.max_holding_hours,
                            max_holding_seconds=cfg.labels.max_holding_seconds,
                            lookback=args.candidate_lookback_bars,
                            rolling_window_bars=args.candidate_rolling_window_bars,
                            dense_stride_bars=getattr(args, "dense_stride_bars", 1),
                            side_mode="long",
                            allow_short_research=False,
                            adaptive_horizon=bool(getattr(args, "adaptive_horizon", False)),
                            min_holding_seconds=getattr(args, "min_holding_seconds", 1.0),
                            adaptive_horizon_max_seconds=(
                                getattr(args, "adaptive_horizon_max_seconds", 0.0) or None
                            ),
                            adaptive_horizon_target_move_bps=_adaptive_horizon_target_move_bps(
                                args,
                                cfg,
                                assumed_spread_bps=quote.spread_bps,
                                research_notional=args.max_order_notional_usd,
                                reference_price=quote.midpoint,
                            ),
                        ),
                        context_bars=context_bars,
                        quote=quote,
                        max_samples=args.max_scoring_samples,
                        assumed_spread_bps=quote.spread_bps,
                        research_notional=min(args.max_order_notional_usd, args.capital_usd),
                        feature_asof=artifact_feature_asof,
                        external_feature_latency_seconds=artifact_external_latency_seconds,
                    )
                    if not scoring_samples:
                        events.emit("signal.none", "no completed-bar candidate available")
                    else:
                        scored = []
                        stale_count = 0
                        duplicate_count = 0
                        for sample in scoring_samples:
                            signal_age_seconds = (
                                datetime.now(tz=UTC) - sample.timestamp_utc.astimezone(UTC)
                            ).total_seconds()
                            if signal_age_seconds > args.max_signal_bar_age_seconds:
                                stale_count += 1
                                continue
                            if sample.event_id in traded_event_ids:
                                duplicate_count += 1
                                continue
                            score = score_production_artifact(
                                artifact,
                                sample.features,
                                threshold_override=threshold_override,
                            )
                            feature_contract_ok = (
                                score.missing_feature_fraction
                                <= args.max_missing_model_feature_fraction
                            )
                            signals += 1
                            scored.append((sample, score, signal_age_seconds))
                            events.emit(
                                "signal.scored",
                                "scored fresh completed-bar signal",
                                event_id=sample.event_id,
                                timestamp_utc=sample.timestamp_utc.isoformat(),
                                candidate_type=sample.candidate_type,
                                probability=score.probability,
                                selected_threshold=score.selected_threshold,
                                artifact_selected_threshold=artifact.selected_threshold,
                                threshold_edge=score.probability - score.selected_threshold,
                                expected_value=score.expected_value,
                                selection_score=score.selection_score,
                                should_trade=score.should_trade and feature_contract_ok,
                                decision_reason=(
                                    score.decision_reason
                                    if feature_contract_ok
                                    else "missing_model_features"
                                ),
                                model_count=score.model_count,
                                feature_count=score.feature_count,
                                missing_feature_count=score.missing_feature_count,
                                missing_feature_fraction=score.missing_feature_fraction,
                                max_missing_feature_fraction=args.max_missing_model_feature_fraction,
                                signal_age_seconds=signal_age_seconds,
                                model_input_bar_size_seconds=(
                                    _history_bar_seconds(history_bars[-1].bar_size)
                                    if history_bars
                                    else 0
                                ),
                                model_input_latest_bar_utc=(
                                    history_bars[-1].timestamp_utc.isoformat()
                                    if history_bars
                                    else ""
                                ),
                            )
                        if not scored:
                            events.emit(
                                "signal.stale",
                                "no fresh untraded completed-bar signal available",
                                stale_count=stale_count,
                                duplicate_count=duplicate_count,
                                max_signal_bar_age_seconds=args.max_signal_bar_age_seconds,
                                priority="warning",
                            )
                        approved = [
                            row
                            for row in scored
                            if row[1].should_trade
                            and row[1].missing_feature_fraction <= args.max_missing_model_feature_fraction
                        ]
                        if approved:
                            if deadline - time.monotonic() <= 0:
                                break
                            if _kill_switch_enabled(cfg):
                                stop_reason = "kill_switch"
                                events.emit(
                                    "risk.kill_switch_triggered",
                                    "trade-run kill switch file detected before order submission",
                                    kill_switch_file=str(cfg.runtime.kill_switch_file),
                                    priority="critical",
                                )
                                break
                            if artifact_target_frequency_mode == "online":
                                sample, score, _ = min(
                                    approved,
                                    key=lambda row: (
                                        row[0].timestamp_utc,
                                        -row[1].selection_score,
                                        -row[1].expected_value,
                                        -(row[1].probability - row[1].selected_threshold),
                                        -row[1].probability,
                                    ),
                                )
                            else:
                                sample, score, _ = max(
                                    approved,
                                    key=lambda row: (
                                        row[1].selection_score,
                                        row[1].expected_value,
                                        row[1].probability - row[1].selected_threshold,
                                        row[1].probability,
                                    ),
                                )
                            notional, notional_source = _trade_run_order_notional_from_artifact_policy(
                                args,
                                artifact_training_config,
                                sample,
                                score,
                                available_notional=available_notional,
                            )
                            if notional < cfg.risk.minimum_fee_efficient_notional:
                                events.emit(
                                    "risk.notional_below_minimum",
                                    "skipping approved signal because remaining capital is below fee-efficient size",
                                    available_notional=available_notional,
                                    minimum_fee_efficient_notional=cfg.risk.minimum_fee_efficient_notional,
                                )
                                await broker.wait(min(args.signal_interval, max(deadline - time.monotonic(), 0.0)))
                                continue
                            intent = CryptoOrderFactory.market_buy_cash(
                                event_id=sample.event_id,
                                symbol=f"{contract.symbol}/{contract.currency}",
                                cash_qty=notional,
                            )
                            trade = broker.place_order_intent(contract, intent)
                            events.emit(
                                "order.submitted",
                                "submitted trade-run market buy",
                                event_id=sample.event_id,
                                cash_qty=notional,
                                probability=score.probability,
                                selected_threshold=score.selected_threshold,
                                artifact_selected_threshold=artifact.selected_threshold,
                                expected_value=score.expected_value,
                                selection_score=score.selection_score,
                                decision_reason=score.decision_reason,
                                sizing_source=notional_source,
                                artifact_sizing_mode=artifact_sizing_mode,
                                order_id=getattr(getattr(trade, "order", None), "orderId", None),
                                perm_id=getattr(getattr(trade, "order", None), "permId", None),
                            )
                            result = await _wait_for_trade_done(
                                broker,
                                trade,
                                timeout_seconds=args.order_timeout_seconds,
                                commission_wait_seconds=args.commission_wait_seconds,
                            )
                            if output:
                                _write_jsonl_record(output, {"type": "entry_order", **result})
                            fill = result.get("fill", {})
                            filled_quantity = (
                                _finite_number(fill.get("filled_quantity")) if isinstance(fill, dict) else None
                            )
                            average_price = (
                                _finite_number(fill.get("average_price")) if isinstance(fill, dict) else None
                            )
                            if filled_quantity and average_price:
                                entries += 1
                                entry_exposed = True
                                traded_event_ids.add(sample.event_id)
                                hold_seconds = _runner_hold_seconds(cfg, args, sample)
                                min_holding_seconds = _runner_min_holding_seconds(sample)
                                entered_monotonic = time.monotonic()
                                entered_utc = datetime.now(tz=UTC)
                                exit_timing = _runner_exit_timing(
                                    entered_monotonic=entered_monotonic,
                                    entered_utc=entered_utc,
                                    hold_seconds=hold_seconds,
                                    min_holding_seconds=min_holding_seconds,
                                )
                                stop_bps, profit_bps, exit_geometry_source = _runner_exit_bps(
                                    args,
                                    sample,
                                )
                                profit_price = (
                                    average_price * (1 + profit_bps / 10_000)
                                    if profit_bps > 0
                                    else None
                                )
                                last_position_quantity = max(last_position_quantity, tracked_position_quantity) + filled_quantity
                                open_entry = {
                                    "event_id": sample.event_id,
                                    "entry_price": average_price,
                                    "quantity": filled_quantity,
                                    "cash_qty": notional,
                                    "stop_price": average_price * (1 - stop_bps / 10_000),
                                    "profit_price": profit_price,
                                    "stop_bps": stop_bps,
                                    "profit_bps": profit_bps,
                                    "exit_geometry_source": exit_geometry_source,
                                    "hold_seconds": hold_seconds,
                                    "min_holding_seconds": min_holding_seconds,
                                    **exit_timing,
                                    "entered_at_utc": entered_utc.isoformat(),
                                }
                                open_entries.append(open_entry)
                                events.emit(
                                    "position.opened",
                                    "trade-run position opened with synthetic exits",
                                    **open_entry,
                                    open_positions=len(open_entries),
                                    open_notional=sum(float(entry.get("cash_qty", 0.0)) for entry in open_entries),
                                )
                            else:
                                events.emit(
                                    "order.entry_unfilled",
                                    "trade-run market buy did not report a filled quantity",
                                    result=result,
                                    priority="critical",
                                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await broker.wait(min(args.signal_interval, remaining))
            final_position = await _current_contract_position_quantity(broker, contract, args)
            if args.flatten_on_exit and final_position > 1e-8:
                exit_quote = await _trade_run_quote(
                    broker,
                    contract,
                    args,
                    quote_ticker=quote_ticker,
                )
                exit_payload = await _submit_market_exit(
                    broker,
                    contract,
                    args,
                    reason="runner_duration_complete",
                    events=events,
                    output=output,
                    quote=exit_quote,
                    position_quantity=final_position,
                    position_quantity_source="broker_positions",
                )
                exits += 1 if exit_payload.get("submitted") else 0
                open_entries = []
                entry_exposed = False
            final_quote = await _trade_run_quote(
                broker,
                contract,
                args,
                quote_ticker=quote_ticker,
            )
            final_snapshot = await _paper_account_snapshot(broker, contract, args, quote=final_quote)
            if output:
                _write_jsonl_record(
                    output,
                    {
                        "type": "final_snapshot",
                        "loss_delta_usd": _snapshot_loss_delta(baseline=baseline, current=final_snapshot),
                        **final_snapshot,
                    },
                )
            events.emit(
                "broker.trade_run.finished",
                "autonomous trade run finished",
                stop_reason=stop_reason,
                snapshots=snapshots,
                signals=signals,
                entries=entries,
                exits=exits,
                max_observed_loss_usd=max_observed_loss,
                final_positions=_position_summary(final_snapshot),
                open_entries=open_entries,
            )
        except BaseException as exc:
            if contract is not None and entry_exposed:
                await _attempt_emergency_cleanup(
                    broker,
                    contract,
                    args,
                    reason=f"trade_run_exception:{type(exc).__name__}: {exc}",
                    output=output,
                    events=events,
                )
            raise
        finally:
            if contract is not None:
                if tick_subscription is not None:
                    broker.cancel_tick_by_tick(contract, tick_type=tick_subscription[0])
                if quote_ticker is not None:
                    broker.cancel_quote(contract)
            await broker.disconnect()
            events.emit("broker.disconnected", "disconnected from IBKR Gateway/TWS")
    print(
        json.dumps(
            {
                "ok": True,
                "mode": cfg.runtime.mode.value,
                "account": cfg.broker.account,
                "snapshots": snapshots,
                "signals": signals,
                "entries": entries,
                "exits": exits,
                "artifact_selected_threshold": artifact.selected_threshold,
                "decision_threshold_override": threshold_override,
                "max_observed_loss_usd": max_observed_loss,
                "stop_reason": stop_reason,
                "state_log": str(output) if output else "",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_broker_trade_run(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_trade_run_async(args))


def _validate_paper_order_test_config(cfg, args: argparse.Namespace) -> None:
    if cfg.runtime.mode != RuntimeMode.PAPER:
        raise SystemExit("broker order-test is paper-only; refusing non-paper config")
    if cfg.broker.port not in {4002, 7497}:
        raise SystemExit("broker order-test requires a standard IBKR paper port (4002 or 7497)")
    if args.notional <= 0:
        raise SystemExit("broker order-test requires a positive --notional")
    if args.notional > cfg.risk.paper_max_notional:
        raise SystemExit(
            f"broker order-test notional {args.notional:.2f} exceeds "
            f"paper_max_notional {cfg.risk.paper_max_notional:.2f}"
        )
    if args.offset_bps <= 0:
        raise SystemExit("broker order-test requires positive --offset-bps")
    if args.confirm != "PAPER_ORDER_TEST":
        raise SystemExit('broker order-test requires --confirm PAPER_ORDER_TEST')
    _require_explicit_broker_account(cfg, "broker order-test")


def _validate_paper_test_config(cfg, args: argparse.Namespace) -> None:
    if cfg.runtime.mode != RuntimeMode.PAPER:
        raise SystemExit("broker paper-test is paper-only; refusing non-paper config")
    if cfg.broker.port not in {4002, 7497}:
        raise SystemExit("broker paper-test requires a standard IBKR paper port (4002 or 7497)")
    if args.confirm != "IBKR_PAPER_TEST":
        raise SystemExit('broker paper-test requires --confirm IBKR_PAPER_TEST')
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        raise SystemExit("broker paper-test requires positive duration and interval")
    if args.max_cash_usd <= 0:
        raise SystemExit("broker paper-test requires positive --max-cash-usd")
    if args.max_loss_usd <= 0:
        raise SystemExit("broker paper-test requires positive --max-loss-usd")
    if args.max_loss_usd > args.max_cash_usd:
        raise SystemExit("broker paper-test max loss cannot exceed max cash")
    if args.max_cash_usd > cfg.risk.paper_max_notional:
        raise SystemExit(
            f"broker paper-test max cash {args.max_cash_usd:.2f} exceeds "
            f"paper_max_notional {cfg.risk.paper_max_notional:.2f}"
        )
    if args.submit_order:
        _require_explicit_broker_account(cfg, "broker paper-test --submit-order")
        if args.order_notional <= 0:
            raise SystemExit("broker paper-test requires positive --order-notional")
        if args.order_notional > args.max_cash_usd:
            raise SystemExit("broker paper-test order notional cannot exceed max cash")
        if args.order_notional > cfg.risk.paper_max_notional:
            raise SystemExit("broker paper-test order notional exceeds paper_max_notional")
        if args.order_offset_bps <= 0:
            raise SystemExit("broker paper-test requires positive --order-offset-bps")


def _validate_round_trip_test_config(cfg, args: argparse.Namespace) -> None:
    if cfg.runtime.mode != RuntimeMode.PAPER:
        raise SystemExit("broker round-trip-test is paper-only; refusing non-paper config")
    if cfg.broker.port not in {4002, 7497}:
        raise SystemExit("broker round-trip-test requires a standard IBKR paper port (4002 or 7497)")
    _require_explicit_broker_account(cfg, "broker round-trip-test")
    if args.confirm != "IBKR_ROUND_TRIP_TEST":
        raise SystemExit('broker round-trip-test requires --confirm IBKR_ROUND_TRIP_TEST')
    if args.notional <= 0:
        raise SystemExit("broker round-trip-test requires positive --notional")
    if args.max_cash_usd <= 0 or args.max_loss_usd <= 0:
        raise SystemExit("broker round-trip-test requires positive max cash and max loss")
    if args.notional > args.max_cash_usd:
        raise SystemExit("broker round-trip-test notional cannot exceed max cash")
    if args.max_cash_usd > cfg.risk.paper_max_notional:
        raise SystemExit("broker round-trip-test max cash exceeds paper_max_notional")
    if args.max_loss_usd > args.max_cash_usd:
        raise SystemExit("broker round-trip-test max loss cannot exceed max cash")
    if args.hold_seconds <= 0 or args.monitor_interval_seconds <= 0:
        raise SystemExit("broker round-trip-test requires positive hold and monitor intervals")
    if args.synthetic_stop_loss_bps <= 0:
        raise SystemExit("broker round-trip-test requires positive synthetic stop loss bps")
    if args.order_timeout_seconds <= 0 or args.commission_wait_seconds < 0:
        raise SystemExit("broker round-trip-test requires valid order timeout settings")


def _require_explicit_broker_account(cfg, command_name: str) -> None:
    if not cfg.broker.account.strip():
        raise SystemExit(f"{command_name} requires explicit broker.account in config or --account")


async def _broker_record_quotes_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.quote_recorder import IBKRQuoteRecorder

    cfg = _override_broker_client_id(load_config(args.config), args)
    with _runtime_event_stream_from_args(args, "broker.record_quotes") as events:
        recorder = IBKRQuoteRecorder(
            cfg,
            output_path=Path(args.output),
            interval_seconds=args.interval_seconds,
            snapshot_timeout_seconds=args.snapshot_timeout_seconds,
            events=events,
        )
        count = await recorder.run(
            duration_seconds=args.duration_seconds,
            symbol=args.symbol or None,
            security_type=args.security_type or None,
            currency=args.currency or None,
            exchange=args.exchange or None,
            last_trade_date_or_contract_month=args.last_trade_date_or_contract_month,
            local_symbol=args.local_symbol,
        )
    print(f"ok: wrote {count} quote records to {args.output}")
    return 0


def _cmd_broker_record_quotes(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_record_quotes_async(args))


async def _broker_historical_bars_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.ibkr import IBKRBroker

    cfg = _override_broker_client_id(load_config(args.config), args)
    broker = IBKRBroker(cfg)
    end = (
        datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if args.end
        else datetime.now(tz=UTC)
    )
    await broker.connect(read_only=True)
    try:
        if (
            args.symbol
            or args.security_type
            or args.currency
            or args.exchange
            or args.last_trade_date_or_contract_month
            or args.local_symbol
        ):
            contract = await broker.qualify_contract(
                symbol=args.symbol or cfg.contract.symbol,
                security_type=args.security_type or cfg.contract.security_type,
                currency=args.currency or cfg.contract.currency,
                exchange=args.exchange or cfg.broker.crypto_exchanges[0],
                last_trade_date_or_contract_month=args.last_trade_date_or_contract_month,
                local_symbol=args.local_symbol,
            )
        else:
            contract = await broker.qualify_crypto_contract()
        bars = await broker.historical_bars(
            contract,
            end=end,
            duration=args.duration,
            bar_size=args.bar_size,
            what_to_show=args.what_to_show,
        )
    finally:
        await broker.disconnect()
    count = write_ibkr_bars(Path(args.output), bars)
    print(
        json.dumps(
            {
                "bars": count,
                "output": args.output,
                "symbol": f"{contract.symbol}/{contract.currency}",
                "exchange": contract.exchange,
                "con_id": contract.con_id,
                "duration": args.duration,
                "bar_size": args.bar_size,
                "what_to_show": args.what_to_show,
                "start": bars[0].timestamp_utc.isoformat() if bars else "",
                "end": bars[-1].timestamp_utc.isoformat() if bars else "",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_broker_historical_bars(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_historical_bars_async(args))


def _cmd_kill_switch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = Path(cfg.runtime.kill_switch_file)
    if args.action == "enable":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enabled\n", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
    print(f"ok: kill switch {args.action}d at {path}")
    return 0


def _cmd_db_init(args: argparse.Namespace) -> int:
    initialize_sqlite(args.path)
    print(f"ok: initialized sqlite schema at {args.path}")
    return 0


def _add_prediction_market_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prediction-market-signals",
        action="store_true",
        help="Fetch Polymarket CLOB v2 and Kalshi BTC prediction-market signals for model features.",
    )
    parser.add_argument(
        "--prediction-market-durations",
        default=",".join(BTC_PREDICTION_MARKET_DURATIONS),
        help="Comma-separated BTC Up/Down durations to attempt, e.g. 5m,15m,30m,1h,2h,4h,24h.",
    )
    parser.add_argument("--prediction-market-lookback-days", type=int, default=14)
    parser.add_argument("--prediction-market-max-markets", type=int, default=500)
    parser.add_argument("--prediction-market-fidelity-minutes", type=int, default=1)
    parser.add_argument("--prediction-market-cache-dir", default="data/raw/prediction_markets")
    parser.add_argument("--refresh-prediction-market-cache", action="store_true")
    parser.add_argument(
        "--prediction-market-feature-profile",
        choices=["stable", "full"],
        default="full",
        help=(
            "full preserves the expanded per-provider/orderbook feature set used by the "
            "futures research branch; stable keeps compact causal aggregates for ablations."
        ),
    )
    parser.add_argument(
        "--feature-asof",
        choices=["signal", "entry"],
        default="signal",
        help=(
            "Timestamp used for externally observed point-in-time features such as "
            "Polymarket/Kalshi snapshots and IBKR quotes. signal is the conservative "
            "default; entry is only valid when those feeds are available before the "
            "simulated execution timestamp."
        ),
    )
    parser.add_argument(
        "--external-feature-latency-seconds",
        type=float,
        default=0.0,
        help=(
            "Subtract a causal latency buffer from externally observed features "
            "such as Polymarket/Kalshi snapshots and IBKR quotes."
        ),
    )
    parser.add_argument(
        "--require-prediction-market-data",
        action="store_true",
        help="Drop candidate samples that have no causal Polymarket/Kalshi snapshot features.",
    )
    parser.add_argument(
        "--require-leading-prediction-market-data",
        action="store_true",
        help="Drop samples unless at least one prediction-market snapshot still has usable time to close.",
    )
    parser.add_argument(
        "--prediction-market-min-available-count",
        type=int,
        default=0,
        help="Drop samples unless at least this many provider-duration prediction-market snapshots are present.",
    )
    parser.add_argument(
        "--prediction-market-min-side-mid",
        type=float,
        default=0.0,
        help="Drop samples unless the best side-aligned prediction-market midpoint is at least this value.",
    )
    parser.add_argument(
        "--prediction-market-min-lead-seconds",
        type=float,
        default=0.0,
        help="Drop samples unless a prediction-market contract has at least this many seconds to close.",
    )
    parser.add_argument(
        "--prediction-market-min-leading-side-mid",
        type=float,
        default=0.0,
        help="Drop samples unless the best still-leading side-aligned midpoint is at least this value.",
    )
    parser.add_argument(
        "--prediction-market-min-leading-residual-edge",
        type=float,
        default=None,
        help=(
            "Drop samples unless a still-leading prediction-market contract has at least this "
            "side-aligned residual edge after accounting for the contract's elapsed spot move."
        ),
    )
    parser.add_argument(
        "--prediction-market-min-leading-liquidity-weight",
        type=float,
        default=0.0,
        help="Drop samples unless leading prediction-market liquidity/volume log weight is at least this value.",
    )


def _add_interpretability_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--permutation-max-features", type=int, default=80)
    parser.add_argument("--permutation-sample-limit", type=int, default=500)
    parser.add_argument(
        "--permutation-grouping",
        choices=["feature", "family", "both"],
        default="feature",
        help="Permutation grouping: original encoded feature, whole feature family, or both.",
    )
    parser.add_argument("--shap-importance", action="store_true")
    parser.add_argument("--shap-sample-limit", type=int, default=200)
    parser.add_argument("--shap-background-limit", type=int, default=200)
    parser.add_argument("--shap-top-n", type=int, default=30)
    parser.add_argument(
        "--shap-grouping",
        choices=["feature", "family", "both"],
        default="feature",
        help="SHAP grouping: original encoded feature, whole feature family, or both.",
    )
    parser.add_argument("--interpretability-top-n", type=int, default=30)
    parser.add_argument(
        "--feature-include-groups",
        default="",
        help=(
            "Optional comma-separated feature-family allow-list for importance-pruned experiments "
            "(for example: futures_context,ibkr_spot,cross_asset,volume_order_flow)."
        ),
    )
    parser.add_argument(
        "--feature-exclude-groups",
        default="",
        help=(
            "Optional comma-separated feature families to remove before walk-forward training "
            "(for example: technical,prediction_market)."
        ),
    )
    parser.add_argument(
        "--feature-exclude-patterns",
        default="",
        help=(
            "Optional comma-separated fnmatch patterns for exact feature pruning "
            "(for example: rsi_*,macd_*,bollinger_*)."
        ),
    )
    parser.add_argument(
        "--importance-scoring",
        default="brier,log_loss",
        help="Comma-separated permutation importance metrics: brier,log_loss,net_pnl.",
    )


def _add_runtime_event_args(parser: argparse.ArgumentParser, *, default_event_log: str = "") -> None:
    parser.add_argument(
        "--stream-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream compact runtime events to stderr while the broker command is running.",
    )
    parser.add_argument(
        "--stream-format",
        choices=["text", "json"],
        default="text",
        help="Console event stream format.",
    )
    parser.add_argument(
        "--event-log",
        default=default_event_log,
        help="Optional JSONL runtime event log path for replay/monitoring.",
    )


def _add_ibkr_quote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--primary-market",
        choices=["binance_spot", "binance_um_futures"],
        default="binance_spot",
        help="Primary downloaded bar source when --primary-bars-jsonl is not set.",
    )
    parser.add_argument(
        "--primary-bars-jsonl",
        default="",
        help=(
            "Optional IBKR historical-bars JSONL to use as the primary research bar series "
            "instead of downloading Binance spot archives."
        ),
    )
    parser.add_argument(
        "--label-bars-jsonl",
        default="",
        help=(
            "Optional faster IBKR historical-bars JSONL used for triple-barrier labels. "
            "This lets 1-minute feature candidates train against 1-second target/stop hits."
        ),
    )
    parser.add_argument(
        "--execution-bars-jsonl",
        default="",
        help=(
            "Optional faster IBKR historical-bars JSONL used for backtest fill/exit replay. "
            "Defaults to --label-bars-jsonl when label bars are provided."
        ),
    )
    parser.add_argument(
        "--context-bars-jsonl",
        default="",
        help=(
            "Optional comma-separated KEY=path JSONL bar files to add as context features, "
            "for example IBKR_MBT=data/raw/ibkr/historical_mbt.jsonl."
        ),
    )
    parser.add_argument(
        "--ibkr-quote-records",
        default="",
        help="Path to broker record-quotes JSONL for IBKR bid/ask, spread, and top-of-book size features.",
    )
    parser.add_argument(
        "--ibkr-futures-quote-records",
        default="",
        help=(
            "Path to IBKR futures quote-recorder JSONL for futures basis, spread, "
            "and top-of-book features."
        ),
    )
    parser.add_argument(
        "--binance-um-derivatives-metrics",
        action="store_true",
        help="Add Binance USD-M futures open-interest and funding-rate context streams.",
    )
    parser.add_argument("--binance-um-metrics-symbol", default="BTCUSDT")
    parser.add_argument(
        "--binance-um-open-interest-period",
        default="5m",
        choices=["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
    )
    parser.add_argument(
        "--binance-um-taker-flow",
        action="store_true",
        help="With --binance-um-derivatives-metrics, add USD-M taker buy/sell volume context.",
    )
    parser.add_argument(
        "--binance-um-basis",
        action="store_true",
        help="With --binance-um-derivatives-metrics, add USD-M futures basis context.",
    )
    parser.add_argument(
        "--binance-um-metrics-period",
        default="5m",
        choices=["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        help="Period for Binance USD-M taker-flow and basis metric streams.",
    )


def _add_strict_live_valid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strict-live-valid-1s",
        action="store_true",
        help=(
            "Fail closed unless label/execution replay uses tick-backed 1s IBKR bars "
            "over the configured minimum window and fold count."
        ),
    )
    parser.add_argument("--min-live-valid-1s-days", type=float, default=0.0)
    parser.add_argument("--preferred-live-valid-1s-days", type=float, default=0.0)
    parser.add_argument("--min-live-valid-folds", type=int, default=0)


def _add_target_frequency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-frequency-mode",
        choices=["strict", "quota", "online"],
        default="online",
        help=(
            "strict keeps the old probability/EV pre-gates before ranking; quota ranks "
            "all non-vetoed candidates to deliberately hit the requested daily turnover; "
            "online walks forward chronologically and is the production-like default."
        ),
    )
    parser.add_argument(
        "--selection-score-floor",
        type=float,
        default=None,
        help="Optional minimum selection score for quota mode. Omit to always choose the daily best-ranked setups.",
    )
    parser.add_argument(
        "--selection-score-ceiling",
        type=float,
        default=None,
        help=(
            "Optional maximum selection score for target-frequency modes. "
            "Useful when calibration shows an overconfident high-score cohort is weak."
        ),
    )
    parser.add_argument(
        "--adaptive-selection-score-floor",
        action="store_true",
        help=(
            "Calibrate a causal selection-score floor on each fold's threshold slice "
            "when --selection-score-floor is omitted."
        ),
    )
    parser.add_argument(
        "--adaptive-selection-score-direction",
        action="store_true",
        help=(
            "Flip target-frequency ranking per fold when the calibration slice shows "
            "lower model scores had better realized returns than higher scores."
        ),
    )
    parser.add_argument(
        "--respect-open-positions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When selecting online target-frequency trades, track selected trades as open "
            "until their vertical barrier and avoid choosing signals that would exceed "
            "max-open-positions. Defaults on for online selection when max-open-positions "
            "is positive."
        ),
    )
    parser.add_argument(
        "--capacity-release-mode",
        choices=["planned", "actual"],
        default="planned",
        help=(
            "When --respect-open-positions is enabled, planned reserves a slot until "
            "the vertical barrier; actual frees it at the backtested stop/take-profit/"
            "vertical exit time, matching what a live bracket-order monitor would know."
        ),
    )


def _add_adaptive_horizon_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adaptive-horizon",
        action="store_true",
        help=(
            "Choose a per-signal maximum holding period from recent volatility and setup "
            "metadata instead of assigning every candidate the same vertical barrier."
        ),
    )
    parser.add_argument(
        "--min-holding-seconds",
        type=float,
        default=1.0,
        help="Minimum adaptive vertical barrier. With 1-second bars this allows one-second exits.",
    )
    parser.add_argument(
        "--adaptive-horizon-max-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional cap for adaptive horizons. Omit/0 to use --max-holding-seconds "
            "when set, otherwise --max-holding-hours."
        ),
    )
    parser.add_argument(
        "--adaptive-horizon-granularity-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional rounding granularity for adaptive vertical barriers. "
            "When omitted and --label-bars-jsonl is supplied, backtests use the label bar interval."
        ),
    )
    parser.add_argument(
        "--adaptive-horizon-target-move-bps",
        type=float,
        default=0.0,
        help=(
            "Move size used to scale adaptive horizons from recent per-bar volatility. "
            "Omit/0 for a cost-aware value based on the configured round-trip cost."
        ),
    )


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in _csv_values(raw))


def _ml_execution_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "entry_order_model": getattr(args, "entry_order_model", "limit"),
        "sizing_mode": getattr(args, "sizing_mode", "fixed"),
        "sizing_score_field": getattr(args, "sizing_score_field", "probability"),
        "sizing_score_direction": getattr(args, "sizing_score_direction", "high"),
        "sizing_base_notional": getattr(args, "sizing_base_notional", 0.0),
        "sizing_mid_notional": getattr(args, "sizing_mid_notional", 0.0),
        "sizing_high_notional": getattr(args, "sizing_high_notional", 0.0),
        "sizing_mid_score": getattr(args, "sizing_mid_score", 0.45),
        "sizing_high_score": getattr(args, "sizing_high_score", 0.90),
        "sizing_max_spread_bps": getattr(args, "sizing_max_spread_bps", 1.0),
        "sizing_min_liquidity_score": getattr(args, "sizing_min_liquidity_score", 0.0),
        "dynamic_exit_overlay": getattr(args, "dynamic_exit_overlay", False),
        "dynamic_exit_checkpoints_minutes": _parse_int_tuple(
            getattr(args, "dynamic_exit_checkpoints_minutes", "15,30,60,120,240")
        ),
        "dynamic_exit_checkpoints_seconds": _parse_int_tuple(
            getattr(args, "dynamic_exit_checkpoints_seconds", "")
        ),
        "dynamic_exit_adverse_bps": getattr(args, "dynamic_exit_adverse_bps", 25.0),
        "dynamic_exit_giveback_bps": getattr(args, "dynamic_exit_giveback_bps", 35.0),
        "dynamic_exit_min_profit_bps": getattr(args, "dynamic_exit_min_profit_bps", 8.0),
        "dynamic_exit_weak_probability": getattr(args, "dynamic_exit_weak_probability", 0.55),
        "dynamic_exit_weak_expected_value_bps": getattr(
            args,
            "dynamic_exit_weak_expected_value_bps",
            0.0,
        ),
        "experiment_max_loss_usd": getattr(args, "experiment_max_loss_usd", 0.0),
        "min_expected_gross_edge_bps": getattr(args, "min_expected_gross_edge_bps", 0.0),
    }


def _selection_execution_policy_from_args(args: argparse.Namespace) -> SelectionExecutionPolicy:
    sizing_mode = getattr(args, "sizing_mode", "fixed")
    if getattr(args, "confidence_scaled_sizing", False) and sizing_mode == "fixed":
        sizing_mode = "confidence"
    return SelectionExecutionPolicy(
        starting_equity=float(getattr(args, "starting_equity", 0.0) or 0.0),
        requested_notional=float(getattr(args, "notional", 0.0) or 0.0),
        sizing_mode=sizing_mode,
        sizing_score_field=getattr(args, "sizing_score_field", "probability"),
        sizing_score_direction=getattr(args, "sizing_score_direction", "high"),
        sizing_base_notional=float(getattr(args, "sizing_base_notional", 0.0) or 0.0),
        sizing_mid_notional=float(getattr(args, "sizing_mid_notional", 0.0) or 0.0),
        sizing_high_notional=float(getattr(args, "sizing_high_notional", 0.0) or 0.0),
        sizing_mid_score=float(getattr(args, "sizing_mid_score", 0.45) or 0.45),
        sizing_high_score=float(getattr(args, "sizing_high_score", 0.90) or 0.90),
        sizing_max_spread_bps=float(getattr(args, "sizing_max_spread_bps", 1.0) or 0.0),
        sizing_min_liquidity_score=float(getattr(args, "sizing_min_liquidity_score", 0.0) or 0.0),
    )


def _artifact_candidate_policy_from_args(
    args: argparse.Namespace,
    candidate_config: CandidateGenerationConfig,
) -> dict[str, object]:
    return {
        "mode": candidate_config.mode,
        "side_mode": candidate_config.side_mode,
        "allow_short_research": bool(candidate_config.allow_short_research),
        "min_history_bars": int(candidate_config.min_history_bars),
        "dense_stride_bars": int(candidate_config.dense_stride_bars),
        "rolling_window_bars": int(candidate_config.rolling_window_bars),
        "lookback": int(candidate_config.lookback),
        "adaptive_horizon": bool(candidate_config.adaptive_horizon),
        "min_holding_seconds": float(candidate_config.min_holding_seconds),
        "max_holding_hours": float(candidate_config.max_holding_hours),
        "max_holding_seconds": (
            float(candidate_config.max_holding_seconds)
            if candidate_config.max_holding_seconds is not None
            else None
        ),
        "adaptive_horizon_max_seconds": (
            float(candidate_config.adaptive_horizon_max_seconds)
            if candidate_config.adaptive_horizon_max_seconds is not None
            else None
        ),
        "adaptive_horizon_granularity_seconds": (
            float(candidate_config.adaptive_horizon_granularity_seconds)
            if candidate_config.adaptive_horizon_granularity_seconds is not None
            else None
        ),
        "adaptive_horizon_target_move_bps": float(candidate_config.adaptive_horizon_target_move_bps),
        "cli_interval": str(getattr(args, "interval", "")),
    }


def _artifact_horizon_policy_from_args(
    args: argparse.Namespace,
    candidate_config: CandidateGenerationConfig,
) -> dict[str, object]:
    return {
        "label_net_profit_target": float(getattr(args, "net_profit_target", 0.0) or 0.0),
        "label_net_stop_loss": float(getattr(args, "net_stop_loss", 0.0) or 0.0),
        "minimum_gross_profit_bps": float(getattr(args, "minimum_gross_profit_bps", 0.0) or 0.0),
        "minimum_gross_stop_bps": float(getattr(args, "minimum_gross_stop_bps", 0.0) or 0.0),
        "min_holding_seconds": float(candidate_config.min_holding_seconds),
        "max_holding_hours": float(candidate_config.max_holding_hours),
        "max_holding_seconds": (
            float(candidate_config.max_holding_seconds)
            if candidate_config.max_holding_seconds is not None
            else None
        ),
        "adaptive_horizon": bool(candidate_config.adaptive_horizon),
        "dynamic_exit": _ml_execution_kwargs(args),
    }


def _artifact_data_contract_from_args(
    args: argparse.Namespace,
    data_coverage: dict[str, object],
    *,
    label_bar_coverage: dict[str, object],
    execution_bar_coverage: dict[str, object],
) -> dict[str, object]:
    primary = data_coverage.get("primary") if isinstance(data_coverage.get("primary"), dict) else {}
    context = data_coverage.get("context") if isinstance(data_coverage.get("context"), dict) else {}
    label_interval = str(label_bar_coverage.get("interval") or "")
    execution_interval = str(execution_bar_coverage.get("interval") or "")
    requires_one_second_execution = (
        bool(label_bar_coverage.get("enabled"))
        and bool(execution_bar_coverage.get("enabled"))
        and label_interval == "1s"
        and execution_interval == "1s"
    )
    return {
        "primary_source": str(primary.get("source") or ""),
        "primary_interval": str(primary.get("interval") or getattr(args, "interval", "")),
        "label_interval": label_interval,
        "execution_interval": execution_interval,
        "requires_one_second_execution": requires_one_second_execution,
        "requires_tick_backed_live_one_second": requires_one_second_execution,
        "context_sources": sorted(str(key) for key in context),
        "feature_asof": str(getattr(args, "feature_asof", "signal")),
        "external_feature_latency_seconds": float(
            getattr(args, "external_feature_latency_seconds", 0.0) or 0.0
        ),
        "strict_live_valid_1s": data_coverage.get("strict_live_valid_1s", {}),
        "min_live_valid_1s_days": float(getattr(args, "min_live_valid_1s_days", 0.0) or 0.0),
        "preferred_live_valid_1s_days": float(
            getattr(args, "preferred_live_valid_1s_days", 0.0) or 0.0
        ),
        "min_live_valid_folds": int(getattr(args, "min_live_valid_folds", 0) or 0),
        "prediction_market_required": bool(getattr(args, "require_prediction_market_data", False)),
        "leading_prediction_market_required": bool(
            getattr(args, "require_leading_prediction_market_data", False)
        ),
    }


def _live_artifact_contract_errors(
    training_config: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    errors: list[str] = []
    if _artifact_entry_order_model(training_config) != "market":
        errors.append("execution_policy.entry_order_model must be market")
    if not _artifact_policy_dict(training_config, "sizing_policy"):
        errors.append("sizing_policy is missing")
    candidate_policy = training_config.get("candidate_policy")
    if not isinstance(candidate_policy, dict) or not candidate_policy:
        errors.append("candidate_policy is missing")
    else:
        if candidate_policy.get("side_mode") != "long":
            errors.append("candidate_policy.side_mode must be long for spot crypto live trading")
        if bool(candidate_policy.get("allow_short_research")):
            errors.append("candidate_policy.allow_short_research must be false")
    setup_family_policy = training_config.get("setup_family_policy")
    if not isinstance(setup_family_policy, dict) or not setup_family_policy:
        errors.append("setup_family_policy is missing")
    else:
        if not setup_family_policy.get("setup_family_fields"):
            errors.append("setup_family_policy.setup_family_fields is missing")
        if not setup_family_policy.get("score_direction_field"):
            errors.append("setup_family_policy.score_direction_field is missing")
    horizon_policy = training_config.get("horizon_policy")
    if not isinstance(horizon_policy, dict) or not horizon_policy:
        errors.append("horizon_policy is missing")
    data_contract = training_config.get("data_contract")
    if not isinstance(data_contract, dict) or not data_contract:
        errors.append("data_contract is missing")
    else:
        if data_contract.get("requires_one_second_execution"):
            if _history_bar_seconds(getattr(args, "history_bar_size", "")) != 1:
                errors.append("data_contract requires --history-bar-size '1 secs'")
            if getattr(args, "tick_by_tick_type", "Last") == "none":
                errors.append("data_contract requires an IBKR tick-by-tick stream")
            strict_diagnostics = data_contract.get("strict_live_valid_1s")
            if isinstance(strict_diagnostics, dict) and not strict_diagnostics.get("ok", False):
                errors.append("data_contract strict_live_valid_1s gate was not passed")
    feature_contract = training_config.get("feature_contract")
    if not isinstance(feature_contract, dict) or not feature_contract:
        errors.append("feature_contract is missing")
    return errors


def _add_ml_execution_research_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--entry-order-model",
        choices=["limit", "market"],
        default="limit",
        help=(
            "Backtest entry execution model. Use market to align research with broker "
            "trade-run market buy cash entries."
        ),
    )
    parser.add_argument(
        "--sizing-mode",
        choices=["fixed", "confidence", "score_bucket", "liquidity_score_bucket"],
        default="fixed",
        help="Backtest sizing policy. score_bucket uses base/mid/high notionals by model score.",
    )
    parser.add_argument(
        "--sizing-score-field",
        choices=["probability", "expected_value", "predicted_return", "selection_score", "trade_score"],
        default="probability",
    )
    parser.add_argument(
        "--sizing-score-direction",
        choices=["high", "low"],
        default="high",
        help=(
            "Use high for ordinary bucket sizing where larger scores receive more capital. "
            "Use low when calibration shows lower scores are the stronger cohort."
        ),
    )
    parser.add_argument("--sizing-base-notional", type=float, default=0.0)
    parser.add_argument("--sizing-mid-notional", type=float, default=0.0)
    parser.add_argument("--sizing-high-notional", type=float, default=0.0)
    parser.add_argument("--sizing-mid-score", type=float, default=0.45)
    parser.add_argument("--sizing-high-score", type=float, default=0.90)
    parser.add_argument("--sizing-max-spread-bps", type=float, default=1.0)
    parser.add_argument("--sizing-min-liquidity-score", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-exit-overlay",
        action="store_true",
        help="Replay a causal early-exit overlay at configured checkpoints before the vertical barrier.",
    )
    parser.add_argument("--dynamic-exit-checkpoints-minutes", default="15,30,60,120,240")
    parser.add_argument(
        "--dynamic-exit-checkpoints-seconds",
        default="",
        help=(
            "Optional comma-separated second-level dynamic-exit checkpoints for tick/1-second "
            "research, for example 5,15,30,60."
        ),
    )
    parser.add_argument("--dynamic-exit-adverse-bps", type=float, default=25.0)
    parser.add_argument("--dynamic-exit-giveback-bps", type=float, default=35.0)
    parser.add_argument("--dynamic-exit-min-profit-bps", type=float, default=8.0)
    parser.add_argument("--dynamic-exit-weak-probability", type=float, default=0.55)
    parser.add_argument("--dynamic-exit-weak-expected-value-bps", type=float, default=0.0)
    parser.add_argument(
        "--experiment-max-loss-usd",
        type=float,
        default=0.0,
        help=(
            "Optional absolute research stop. When positive, the ML backtest stops "
            "accepting new entries after realized equity loss reaches this dollar amount."
        ),
    )
    parser.add_argument(
        "--min-expected-gross-edge-bps",
        type=float,
        default=0.0,
        help=(
            "Reject entries whose predicted gross edge is below this cost/safety floor. "
            "Use for strict live-valid runs to avoid fee-only churn."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeroalpha")
    sub = parser.add_subparsers(dest="command", required=True)

    config = sub.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    check = config_sub.add_parser("check")
    check.add_argument("--config", default="configs/paper.example.toml")
    check.set_defaults(func=_cmd_config_check)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    binance = data_sub.add_parser("binance-url")
    binance.add_argument("--symbol", required=True)
    binance.add_argument("--interval", required=True)
    binance.add_argument("--month", required=True)
    binance.set_defaults(func=_cmd_binance_url)
    health = data_sub.add_parser("health-check")
    health.add_argument("--cache-dir", default="data/raw/source_health")
    health.set_defaults(func=_cmd_data_health_check)

    backtest = sub.add_parser("backtest")
    backtest_sub = backtest.add_subparsers(dest="backtest_command", required=True)
    candidate = backtest_sub.add_parser("candidate")
    candidate.add_argument("--config", default="configs/paper.example.toml")
    candidate.add_argument("--symbol", default="BTCUSDT")
    candidate.add_argument("--interval", default="1h")
    candidate.add_argument("--years", type=int, default=3)
    candidate.add_argument("--start", default="")
    candidate.add_argument("--end", default="")
    candidate.add_argument("--starting-equity", type=float, default=10_000.0)
    candidate.add_argument("--notional", type=float, default=10_000.0)
    candidate.add_argument("--assumed-spread-bps", type=float, default=10.0)
    candidate.add_argument("--cache-dir", default="data/raw/binance")
    candidate.add_argument("--output", default="artifacts/backtests/candidate_btcusdt_1h.json")
    candidate.set_defaults(func=_cmd_backtest_candidate)
    ml_backtest = backtest_sub.add_parser("ml")
    ml_backtest.add_argument("--config", default="configs/paper.example.toml")
    ml_backtest.add_argument("--symbol", default="BTCUSDT")
    ml_backtest.add_argument("--context-symbols", default="ETHUSDT,SOLUSDT,ETHBTC")
    ml_backtest.add_argument("--context-interval", default="")
    ml_backtest.add_argument("--coinbase-reference-products", default="none")
    ml_backtest.add_argument("--binance-um-futures-reference-symbols", default="none")
    ml_backtest.add_argument("--interval", default="1h")
    ml_backtest.add_argument("--years", type=int, default=3)
    ml_backtest.add_argument("--start", default="")
    ml_backtest.add_argument("--end", default="")
    ml_backtest.add_argument("--starting-equity", type=float, default=10_000.0)
    ml_backtest.add_argument("--notional", type=float, default=10_000.0)
    ml_backtest.add_argument("--instrument-model", choices=["spot_crypto", "futures"], default="")
    ml_backtest.add_argument("--assumed-spread-bps", type=float, default=10.0)
    ml_backtest.add_argument("--minimum-data-coverage", type=float, default=0.95)
    ml_backtest.add_argument("--allow-data-gaps", action="store_true")
    ml_backtest.add_argument("--max-source-divergence-bps", type=float, default=500.0)
    ml_backtest.add_argument("--max-bar-return-bps", type=float, default=0.0)
    ml_backtest.add_argument("--entry-limit-offset-bps", type=float, default=0.0)
    ml_backtest.add_argument("--cache-dir", default="data/raw/binance")
    ml_backtest.add_argument("--candidate-mode", choices=["rules", "aggressive", "dense", "active"], default="rules")
    ml_backtest.add_argument("--candidate-types", default="")
    ml_backtest.add_argument("--setup-families", default="")
    ml_backtest.add_argument("--exclude-setup-families", default="")
    ml_backtest.add_argument("--side-mode", choices=["long", "short", "long_short"], default="long")
    ml_backtest.add_argument(
        "--allow-spot-short-research",
        action="store_true",
        help="Research-only override. Spot crypto execution remains long/flat unless futures are modeled.",
    )
    ml_backtest.add_argument("--dense-stride-bars", type=int, default=1)
    ml_backtest.add_argument("--min-history-bars", type=int, default=240)
    ml_backtest.add_argument("--max-holding-hours", type=float, default=0)
    ml_backtest.add_argument(
        "--max-holding-seconds",
        type=float,
        default=0.0,
        help="Optional holding horizon in seconds. Use only with data granular enough to replay it.",
    )
    _add_adaptive_horizon_args(ml_backtest)
    ml_backtest.add_argument("--net-profit-target", type=float, default=0.0)
    ml_backtest.add_argument("--net-stop-loss", type=float, default=0.0)
    ml_backtest.add_argument("--volatility-lookback-bars", type=int, default=0)
    ml_backtest.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    ml_backtest.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-gross-profit-bps", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-gross-stop-bps", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-probability", type=float, default=None)
    ml_backtest.add_argument("--minimum-expected-value", type=float, default=None)
    ml_backtest.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="")
    ml_backtest.add_argument("--target-trades-per-day", type=float, default=0.0)
    _add_target_frequency_args(ml_backtest)
    ml_backtest.add_argument("--research-gate", action="store_true")
    ml_backtest.add_argument("--allow-negative-ev-frequency-probe", action="store_true")
    ml_backtest.add_argument("--allow-research-short-backtest", action="store_true")
    ml_backtest.add_argument(
        "--optimize-metric",
        choices=["sharpe", "net_pnl", "calmar"],
        default="sharpe",
    )
    ml_backtest.add_argument("--candidate-type-thresholds", action="store_true")
    ml_backtest.add_argument("--empirical-payoff-ev", action="store_true")
    ml_backtest.add_argument("--confidence-scaled-sizing", action="store_true")
    _add_ml_execution_research_args(ml_backtest)
    ml_backtest.add_argument(
        "--selection-score",
        choices=[
            "probability",
            "expected_value",
            "predicted_return",
            "expected_utility",
            "risk_adjusted_return",
            "return_first",
            "blended_rank",
            "capital_efficiency",
        ],
        default="expected_utility",
    )
    ml_backtest.add_argument("--specialist-models", action="store_true")
    ml_backtest.add_argument("--require-calibrated-selection", action="store_true")
    ml_backtest.add_argument("--min-signal-spacing-hours", type=float, default=0.0)
    ml_backtest.add_argument("--max-signals-per-group-per-day", type=int, default=0)
    ml_backtest.add_argument("--max-signals-per-timestamp", type=int, default=0)
    ml_backtest.add_argument("--selection-setup-families", default="")
    ml_backtest.add_argument("--selection-exclude-setup-families", default="")
    ml_backtest.add_argument("--risk-per-trade", type=float, default=0.0)
    ml_backtest.add_argument("--daily-loss-stop", type=float, default=0.0)
    ml_backtest.add_argument("--weekly-loss-stop", type=float, default=0.0)
    ml_backtest.add_argument("--rolling-drawdown-stop", type=float, default=0.0)
    ml_backtest.add_argument("--paper-max-notional", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-fee-efficient-notional", type=float, default=0.0)
    ml_backtest.add_argument("--tier-rate", type=float, default=None)
    ml_backtest.add_argument("--minimum-commission", type=float, default=None)
    ml_backtest.add_argument("--maximum-commission-rate", type=float, default=None)
    ml_backtest.add_argument("--base-slippage-bps", type=float, default=None)
    ml_backtest.add_argument("--safety-margin-bps", type=float, default=None)
    ml_backtest.add_argument(
        "--futures-fee-per-contract",
        type=float,
        default=None,
        help="One-way fixed futures fee per contract, including IBKR execution plus exchange/regulatory fees.",
    )
    ml_backtest.add_argument(
        "--futures-contract-multiplier",
        type=float,
        default=None,
        help="Futures contract BTC multiplier, e.g. 0.1 for CME MBT.",
    )
    ml_backtest.add_argument("--max-open-positions", type=int, default=0)
    ml_backtest.add_argument("--consecutive-loss-limit", type=int, default=None)
    ml_backtest.add_argument("--cooldown-hours-after-stopouts", type=int, default=None)
    ml_backtest.add_argument(
        "--models",
        default="logistic,histgb,randomforest,extratrees,lightgbm,catboost,xgboost,tabicl,tabpfn",
        help="Comma-separated base models. TabICL/TabPFN are skipped when packages are unavailable.",
    )
    ml_backtest.add_argument("--train-size", type=int, default=0)
    ml_backtest.add_argument("--candidate-lookback-bars", type=int, default=24)
    ml_backtest.add_argument("--candidate-rolling-window-bars", type=int, default=500)
    ml_backtest.add_argument("--calibration-size", type=int, default=0)
    ml_backtest.add_argument("--test-size", type=int, default=0)
    ml_backtest.add_argument("--embargo-hours", type=int, default=0)
    ml_backtest.add_argument("--adaptive-threshold", action="store_true")
    ml_backtest.add_argument("--min-calibration-trades", type=int, default=5)
    ml_backtest.add_argument("--adaptive-minimum-threshold", type=float, default=0.0)
    ml_backtest.add_argument("--stacker", choices=["average", "logistic", "best", "weighted"], default="average")
    ml_backtest.add_argument("--hpo", action="store_true")
    ml_backtest.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota", "capacity"], default="standard")
    ml_backtest.add_argument("--hpo-trials", type=int, default=0)
    ml_backtest.add_argument("--foundation-max-samples", type=int, default=0)
    ml_backtest.add_argument("--kronos-features", action="store_true")
    ml_backtest.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    ml_backtest.add_argument("--kronos-lookback-bars", type=int, default=0)
    ml_backtest.add_argument("--kronos-embedding-dims", type=int, default=0)
    ml_backtest.add_argument("--kronos-device", default="")
    _add_prediction_market_args(ml_backtest)
    _add_ibkr_quote_args(ml_backtest)
    _add_strict_live_valid_args(ml_backtest)
    _add_interpretability_args(ml_backtest)
    ml_backtest.add_argument("--output", default="artifacts/backtests/ml_btcusdt_1h.json")
    ml_backtest.set_defaults(func=_cmd_backtest_ml)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    train_meta = model_sub.add_parser("train-meta")
    train_meta.add_argument("--config", default="configs/paper.example.toml")
    train_meta.add_argument("--symbol", default="BTCUSDT")
    train_meta.add_argument("--context-symbols", default="ETHUSDT,SOLUSDT,ETHBTC")
    train_meta.add_argument("--context-interval", default="")
    train_meta.add_argument("--coinbase-reference-products", default="none")
    train_meta.add_argument("--binance-um-futures-reference-symbols", default="none")
    train_meta.add_argument("--interval", default="1h")
    train_meta.add_argument("--years", type=int, default=3)
    train_meta.add_argument("--start", default="")
    train_meta.add_argument("--end", default="")
    train_meta.add_argument("--notional", type=float, default=10_000.0)
    train_meta.add_argument("--instrument-model", choices=["spot_crypto", "futures"], default="")
    train_meta.add_argument("--assumed-spread-bps", type=float, default=10.0)
    train_meta.add_argument("--minimum-data-coverage", type=float, default=0.95)
    train_meta.add_argument("--allow-data-gaps", action="store_true")
    train_meta.add_argument("--max-source-divergence-bps", type=float, default=500.0)
    train_meta.add_argument("--max-bar-return-bps", type=float, default=0.0)
    train_meta.add_argument("--cache-dir", default="data/raw/binance")
    train_meta.add_argument("--candidate-mode", choices=["rules", "aggressive", "dense", "active"], default="rules")
    train_meta.add_argument("--candidate-types", default="")
    train_meta.add_argument("--setup-families", default="")
    train_meta.add_argument("--exclude-setup-families", default="")
    train_meta.add_argument("--side-mode", choices=["long", "short", "long_short"], default="long")
    train_meta.add_argument(
        "--allow-spot-short-research",
        action="store_true",
        help="Research-only override. Spot crypto execution remains long/flat unless futures are modeled.",
    )
    train_meta.add_argument("--dense-stride-bars", type=int, default=1)
    train_meta.add_argument("--min-history-bars", type=int, default=240)
    train_meta.add_argument("--max-holding-hours", type=float, default=0)
    train_meta.add_argument(
        "--max-holding-seconds",
        type=float,
        default=0.0,
        help="Optional holding horizon in seconds. Use only with data granular enough to replay it.",
    )
    _add_adaptive_horizon_args(train_meta)
    train_meta.add_argument("--net-profit-target", type=float, default=0.0)
    train_meta.add_argument("--net-stop-loss", type=float, default=0.0)
    train_meta.add_argument("--volatility-lookback-bars", type=int, default=0)
    train_meta.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    train_meta.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    train_meta.add_argument("--minimum-gross-profit-bps", type=float, default=0.0)
    train_meta.add_argument("--minimum-gross-stop-bps", type=float, default=0.0)
    train_meta.add_argument("--minimum-probability", type=float, default=None)
    train_meta.add_argument("--minimum-expected-value", type=float, default=None)
    train_meta.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="")
    train_meta.add_argument("--target-trades-per-day", type=float, default=0.0)
    _add_target_frequency_args(train_meta)
    train_meta.add_argument("--research-gate", action="store_true")
    train_meta.add_argument("--allow-negative-ev-frequency-probe", action="store_true")
    train_meta.add_argument("--allow-research-short-backtest", action="store_true")
    train_meta.add_argument(
        "--optimize-metric",
        choices=["sharpe", "net_pnl", "calmar"],
        default="sharpe",
    )
    train_meta.add_argument("--candidate-type-thresholds", action="store_true")
    train_meta.add_argument("--empirical-payoff-ev", action="store_true")
    train_meta.add_argument("--confidence-scaled-sizing", action="store_true")
    train_meta.add_argument(
        "--selection-score",
        choices=[
            "probability",
            "expected_value",
            "predicted_return",
            "expected_utility",
            "risk_adjusted_return",
            "return_first",
            "blended_rank",
            "capital_efficiency",
        ],
        default="expected_utility",
    )
    train_meta.add_argument("--specialist-models", action="store_true")
    train_meta.add_argument("--require-calibrated-selection", action="store_true")
    train_meta.add_argument("--min-signal-spacing-hours", type=float, default=0.0)
    train_meta.add_argument("--max-signals-per-group-per-day", type=int, default=0)
    train_meta.add_argument("--max-signals-per-timestamp", type=int, default=0)
    train_meta.add_argument("--selection-setup-families", default="")
    train_meta.add_argument("--selection-exclude-setup-families", default="")
    train_meta.add_argument("--max-open-positions", type=int, default=0)
    train_meta.add_argument("--tier-rate", type=float, default=None)
    train_meta.add_argument("--minimum-commission", type=float, default=None)
    train_meta.add_argument("--maximum-commission-rate", type=float, default=None)
    train_meta.add_argument("--base-slippage-bps", type=float, default=None)
    train_meta.add_argument("--safety-margin-bps", type=float, default=None)
    train_meta.add_argument(
        "--futures-fee-per-contract",
        type=float,
        default=None,
        help="One-way fixed futures fee per contract, including IBKR execution plus exchange/regulatory fees.",
    )
    train_meta.add_argument(
        "--futures-contract-multiplier",
        type=float,
        default=None,
        help="Futures contract BTC multiplier, e.g. 0.1 for CME MBT.",
    )
    train_meta.add_argument(
        "--models",
        default="logistic,histgb,randomforest,extratrees,lightgbm,catboost,xgboost,tabicl,tabpfn",
        help="Comma-separated base models. TabICL/TabPFN are skipped when packages are unavailable.",
    )
    train_meta.add_argument("--train-size", type=int, default=0)
    train_meta.add_argument("--candidate-lookback-bars", type=int, default=24)
    train_meta.add_argument("--candidate-rolling-window-bars", type=int, default=500)
    train_meta.add_argument("--calibration-size", type=int, default=0)
    train_meta.add_argument("--test-size", type=int, default=0)
    train_meta.add_argument("--embargo-hours", type=int, default=0)
    train_meta.add_argument("--adaptive-threshold", action="store_true")
    train_meta.add_argument("--min-calibration-trades", type=int, default=5)
    train_meta.add_argument("--adaptive-minimum-threshold", type=float, default=0.0)
    train_meta.add_argument("--stacker", choices=["average", "logistic", "best", "weighted"], default="average")
    train_meta.add_argument("--hpo", action="store_true")
    train_meta.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota", "capacity"], default="standard")
    train_meta.add_argument("--hpo-trials", type=int, default=0)
    train_meta.add_argument("--foundation-max-samples", type=int, default=0)
    train_meta.add_argument("--kronos-features", action="store_true")
    train_meta.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    train_meta.add_argument("--kronos-lookback-bars", type=int, default=0)
    train_meta.add_argument("--kronos-embedding-dims", type=int, default=0)
    train_meta.add_argument("--kronos-device", default="")
    _add_prediction_market_args(train_meta)
    _add_ibkr_quote_args(train_meta)
    _add_strict_live_valid_args(train_meta)
    _add_interpretability_args(train_meta)
    train_meta.add_argument(
        "--output",
        default="artifacts/models/meta_label_walk_forward_btcusdt_1h.json",
    )
    train_meta.add_argument(
        "--save-artifact",
        default="",
        help=(
            "Optional path for a joblib production scoring artifact containing the final "
            "encoder, fitted model stack, calibrator, selected threshold, and manifest checksum."
        ),
    )
    train_meta.set_defaults(func=_cmd_model_train_meta)
    smoke_model = model_sub.add_parser("smoke")
    smoke_model.add_argument(
        "--models",
        default="logistic,histgb,randomforest,extratrees,lightgbm,catboost,xgboost,tabicl,tabpfn",
        help="Comma-separated models to instantiate, fit, and predict on synthetic data.",
    )
    smoke_model.add_argument("--timeout-seconds", type=int, default=90)
    smoke_model.set_defaults(func=_cmd_model_smoke)
    kronos_status = model_sub.add_parser("kronos-status")
    kronos_status.set_defaults(func=_cmd_model_kronos_status)
    signal_audit = model_sub.add_parser("signal-audit")
    signal_audit.add_argument("--config", default="configs/paper.example.toml")
    signal_audit.add_argument("--symbol", default="BTCUSDT")
    signal_audit.add_argument("--context-symbols", default="ETHUSDT,SOLUSDT,ETHBTC")
    signal_audit.add_argument("--context-interval", default="")
    signal_audit.add_argument("--coinbase-reference-products", default="none")
    signal_audit.add_argument("--binance-um-futures-reference-symbols", default="none")
    signal_audit.add_argument("--interval", default="15m")
    signal_audit.add_argument("--years", type=int, default=1)
    signal_audit.add_argument("--start", default="")
    signal_audit.add_argument("--end", default="")
    signal_audit.add_argument("--starting-equity", type=float, default=10_000.0)
    signal_audit.add_argument("--notional", type=float, default=10_000.0)
    signal_audit.add_argument("--instrument-model", choices=["spot_crypto", "futures"], default="")
    signal_audit.add_argument("--assumed-spread-bps", type=float, default=4.0)
    signal_audit.add_argument("--minimum-data-coverage", type=float, default=0.95)
    signal_audit.add_argument("--allow-data-gaps", action="store_true")
    signal_audit.add_argument("--max-source-divergence-bps", type=float, default=500.0)
    signal_audit.add_argument("--max-bar-return-bps", type=float, default=0.0)
    signal_audit.add_argument("--entry-limit-offset-bps", type=float, default=0.0)
    signal_audit.add_argument("--cache-dir", default="data/raw/binance")
    signal_audit.add_argument("--candidate-mode", choices=["rules", "aggressive", "dense", "active"], default="active")
    signal_audit.add_argument("--candidate-types", default="")
    signal_audit.add_argument("--setup-families", default="")
    signal_audit.add_argument("--exclude-setup-families", default="")
    signal_audit.add_argument("--side-mode", choices=["long", "short", "long_short"], default="long")
    signal_audit.add_argument("--allow-spot-short-research", action="store_true")
    signal_audit.add_argument("--dense-stride-bars", type=int, default=1)
    signal_audit.add_argument("--min-history-bars", type=int, default=240)
    signal_audit.add_argument("--max-holding-hours", type=float, default=4)
    signal_audit.add_argument(
        "--max-holding-seconds",
        type=float,
        default=0.0,
        help="Optional holding horizon in seconds. Use only with data granular enough to replay it.",
    )
    _add_adaptive_horizon_args(signal_audit)
    signal_audit.add_argument("--net-profit-target", type=float, default=0.001)
    signal_audit.add_argument("--net-stop-loss", type=float, default=0.001)
    signal_audit.add_argument("--volatility-lookback-bars", type=int, default=96)
    signal_audit.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    signal_audit.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    signal_audit.add_argument("--minimum-gross-profit-bps", type=float, default=100.0)
    signal_audit.add_argument("--minimum-gross-stop-bps", type=float, default=80.0)
    signal_audit.add_argument("--minimum-probability", type=float, default=None)
    signal_audit.add_argument("--minimum-expected-value", type=float, default=None)
    signal_audit.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="")
    signal_audit.add_argument("--target-trades-per-day", type=float, default=4.0)
    _add_target_frequency_args(signal_audit)
    signal_audit.add_argument("--research-gate", action="store_true")
    signal_audit.add_argument("--allow-negative-ev-frequency-probe", action="store_true")
    signal_audit.add_argument("--allow-research-short-backtest", action="store_true")
    signal_audit.add_argument("--candidate-type-thresholds", action="store_true")
    signal_audit.add_argument("--empirical-payoff-ev", action="store_true")
    signal_audit.add_argument("--confidence-scaled-sizing", action="store_true")
    _add_ml_execution_research_args(signal_audit)
    signal_audit.add_argument("--risk-per-trade", type=float, default=0.0)
    signal_audit.add_argument("--daily-loss-stop", type=float, default=0.0)
    signal_audit.add_argument("--weekly-loss-stop", type=float, default=0.0)
    signal_audit.add_argument("--rolling-drawdown-stop", type=float, default=0.0)
    signal_audit.add_argument("--paper-max-notional", type=float, default=0.0)
    signal_audit.add_argument("--minimum-fee-efficient-notional", type=float, default=0.0)
    signal_audit.add_argument("--tier-rate", type=float, default=None)
    signal_audit.add_argument("--minimum-commission", type=float, default=None)
    signal_audit.add_argument("--maximum-commission-rate", type=float, default=None)
    signal_audit.add_argument("--base-slippage-bps", type=float, default=1.0)
    signal_audit.add_argument("--safety-margin-bps", type=float, default=2.0)
    signal_audit.add_argument("--max-open-positions", type=int, default=4)
    signal_audit.add_argument("--consecutive-loss-limit", type=int, default=None)
    signal_audit.add_argument("--cooldown-hours-after-stopouts", type=int, default=None)
    signal_audit.add_argument(
        "--models",
        default="logistic,histgb,randomforest,extratrees,lightgbm,catboost,xgboost",
    )
    signal_audit.add_argument("--train-size", type=int, default=0)
    signal_audit.add_argument("--candidate-lookback-bars", type=int, default=24)
    signal_audit.add_argument("--candidate-rolling-window-bars", type=int, default=1000)
    signal_audit.add_argument("--calibration-size", type=int, default=0)
    signal_audit.add_argument("--test-size", type=int, default=0)
    signal_audit.add_argument("--embargo-hours", type=int, default=0)
    signal_audit.add_argument("--adaptive-threshold", action="store_true")
    signal_audit.add_argument("--min-calibration-trades", type=int, default=10)
    signal_audit.add_argument("--adaptive-minimum-threshold", type=float, default=0.0)
    signal_audit.add_argument("--stacker", choices=["average", "logistic", "best", "weighted"], default="weighted")
    signal_audit.add_argument("--hpo", action="store_true")
    signal_audit.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota", "capacity"], default="standard")
    signal_audit.add_argument("--hpo-trials", type=int, default=0)
    signal_audit.add_argument("--foundation-max-samples", type=int, default=0)
    signal_audit.add_argument(
        "--selection-score",
        choices=[
            "probability",
            "expected_value",
            "predicted_return",
            "expected_utility",
            "risk_adjusted_return",
            "return_first",
            "blended_rank",
            "capital_efficiency",
        ],
        default="expected_utility",
    )
    signal_audit.add_argument("--specialist-models", action="store_true")
    signal_audit.add_argument("--require-calibrated-selection", action="store_true")
    signal_audit.add_argument("--min-signal-spacing-hours", type=float, default=0.0)
    signal_audit.add_argument("--max-signals-per-group-per-day", type=int, default=0)
    signal_audit.add_argument("--max-signals-per-timestamp", type=int, default=1)
    signal_audit.add_argument("--selection-setup-families", default="")
    signal_audit.add_argument("--selection-exclude-setup-families", default="")
    signal_audit.add_argument("--kronos-features", action="store_true")
    signal_audit.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    signal_audit.add_argument("--kronos-lookback-bars", type=int, default=0)
    signal_audit.add_argument("--kronos-embedding-dims", type=int, default=0)
    signal_audit.add_argument("--kronos-device", default="")
    _add_prediction_market_args(signal_audit)
    _add_ibkr_quote_args(signal_audit)
    _add_interpretability_args(signal_audit)
    signal_audit.add_argument("--output", default="artifacts/models/signal_audit_btcusdt_15m.json")
    signal_audit.set_defaults(func=_cmd_model_signal_audit)
    sweep_labels = model_sub.add_parser("sweep-labels")
    sweep_labels.add_argument("--config", default="configs/paper.example.toml")
    sweep_labels.add_argument("--symbol", default="BTCUSDT")
    sweep_labels.add_argument("--context-symbols", default="ETHUSDT,SOLUSDT,ETHBTC")
    sweep_labels.add_argument("--coinbase-reference-products", default="none")
    sweep_labels.add_argument("--binance-um-futures-reference-symbols", default="none")
    sweep_labels.add_argument("--interval", default="1h")
    sweep_labels.add_argument("--years", type=int, default=3)
    sweep_labels.add_argument("--start", default="")
    sweep_labels.add_argument("--end", default="")
    sweep_labels.add_argument("--starting-equity", type=float, default=10_000.0)
    sweep_labels.add_argument("--notional", type=float, default=10_000.0)
    sweep_labels.add_argument("--instrument-model", choices=["spot_crypto", "futures"], default="")
    sweep_labels.add_argument("--assumed-spread-bps", type=float, default=10.0)
    sweep_labels.add_argument("--max-source-divergence-bps", type=float, default=500.0)
    sweep_labels.add_argument("--max-bar-return-bps", type=float, default=0.0)
    sweep_labels.add_argument("--cache-dir", default="data/raw/binance")
    sweep_labels.add_argument("--net-profit-targets", default="0.015,0.02,0.03")
    sweep_labels.add_argument("--net-stop-losses", default="0.015,0.02,0.03")
    sweep_labels.add_argument("--max-holding-hours-values", default="48,72,96")
    sweep_labels.add_argument("--models", default="logistic,histgb,extratrees,lightgbm")
    sweep_labels.add_argument("--candidate-mode", choices=["rules", "aggressive", "dense", "active"], default="rules")
    sweep_labels.add_argument("--side-mode", choices=["long", "short", "long_short"], default="long")
    sweep_labels.add_argument("--allow-spot-short-research", action="store_true")
    sweep_labels.add_argument("--target-trades-per-day", type=float, default=0.0)
    _add_target_frequency_args(sweep_labels)
    sweep_labels.add_argument("--research-gate", action="store_true")
    sweep_labels.add_argument("--allow-negative-ev-frequency-probe", action="store_true")
    sweep_labels.add_argument("--allow-research-short-backtest", action="store_true")
    sweep_labels.add_argument(
        "--optimize-metric",
        choices=["sharpe", "net_pnl", "calmar"],
        default="sharpe",
    )
    sweep_labels.add_argument("--candidate-type-thresholds", action="store_true")
    sweep_labels.add_argument("--empirical-payoff-ev", action="store_true")
    sweep_labels.add_argument("--confidence-scaled-sizing", action="store_true")
    sweep_labels.add_argument(
        "--selection-score",
        choices=[
            "probability",
            "expected_value",
            "predicted_return",
            "expected_utility",
            "risk_adjusted_return",
            "return_first",
            "blended_rank",
            "capital_efficiency",
        ],
        default="expected_utility",
    )
    sweep_labels.add_argument("--specialist-models", action="store_true")
    sweep_labels.add_argument("--require-calibrated-selection", action="store_true")
    sweep_labels.add_argument("--min-signal-spacing-hours", type=float, default=0.0)
    sweep_labels.add_argument("--max-signals-per-group-per-day", type=int, default=0)
    sweep_labels.add_argument("--max-signals-per-timestamp", type=int, default=0)
    sweep_labels.add_argument("--max-open-positions", type=int, default=0)
    sweep_labels.add_argument("--stacker", choices=["average", "logistic"], default="average")
    sweep_labels.add_argument("--adaptive-threshold", action="store_true", default=True)
    sweep_labels.add_argument("--fixed-threshold", dest="adaptive_threshold", action="store_false")
    sweep_labels.add_argument("--adaptive-minimum-threshold", type=float, default=0.0)
    sweep_labels.add_argument("--hpo", action="store_true")
    sweep_labels.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota", "capacity"], default="standard")
    sweep_labels.add_argument("--hpo-trials", type=int, default=0)
    sweep_labels.add_argument("--kronos-features", action="store_true")
    sweep_labels.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    sweep_labels.add_argument("--kronos-lookback-bars", type=int, default=0)
    sweep_labels.add_argument("--kronos-embedding-dims", type=int, default=0)
    sweep_labels.add_argument("--kronos-device", default="")
    _add_prediction_market_args(sweep_labels)
    _add_ibkr_quote_args(sweep_labels)
    sweep_labels.add_argument("--top", type=int, default=5)
    sweep_labels.add_argument("--output", default="artifacts/models/label_geometry_sweep.json")
    sweep_labels.set_defaults(func=_cmd_model_sweep_labels)

    db = sub.add_parser("db")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_init = db_sub.add_parser("init")
    db_init.add_argument("--path", default=".zeroalpha/zeroalpha.sqlite")
    db_init.set_defaults(func=_cmd_db_init)

    broker = sub.add_parser("broker")
    broker_sub = broker.add_subparsers(dest="broker_command", required=True)
    smoke = broker_sub.add_parser("smoke")
    smoke.add_argument("--config", default="configs/paper.example.toml")
    smoke.add_argument("--client-id", type=int, default=0)
    smoke.add_argument("--account", default="")
    smoke.add_argument("--read-only", action="store_true", default=True)
    smoke.set_defaults(func=_cmd_broker_smoke)
    order_test = broker_sub.add_parser("order-test")
    order_test.add_argument("--config", default="configs/paper.example.toml")
    order_test.add_argument("--client-id", type=int, default=0)
    order_test.add_argument("--account", default="")
    order_test.add_argument("--notional", type=float, default=100.0)
    order_test.add_argument("--offset-bps", type=float, default=20.0)
    order_test.add_argument("--price-increment", type=float, default=0.25)
    order_test.add_argument("--wait-seconds", type=float, default=3.0)
    order_test.add_argument("--cancel-wait-seconds", type=float, default=2.0)
    order_test.add_argument("--confirm", default="")
    _add_runtime_event_args(order_test, default_event_log="data/raw/ibkr/order_test_events.jsonl")
    order_test.set_defaults(func=_cmd_broker_order_test)
    paper_test = broker_sub.add_parser("paper-test")
    paper_test.add_argument("--config", default="configs/paper.example.toml")
    paper_test.add_argument("--client-id", type=int, default=0)
    paper_test.add_argument("--account", default="")
    paper_test.add_argument("--duration-seconds", type=float, default=600.0)
    paper_test.add_argument("--interval-seconds", type=float, default=30.0)
    paper_test.add_argument("--snapshot-timeout-seconds", type=float, default=10.0)
    paper_test.add_argument("--account-refresh-timeout-seconds", type=float, default=5.0)
    paper_test.add_argument("--pnl-wait-seconds", type=float, default=1.0)
    paper_test.add_argument("--max-cash-usd", type=float, default=10_000.0)
    paper_test.add_argument("--max-loss-usd", type=float, default=1_000.0)
    paper_test.add_argument("--submit-order", action="store_true")
    paper_test.add_argument("--order-notional", type=float, default=100.0)
    paper_test.add_argument("--order-offset-bps", type=float, default=20.0)
    paper_test.add_argument("--price-increment", type=float, default=0.25)
    paper_test.add_argument("--order-wait-seconds", type=float, default=3.0)
    paper_test.add_argument("--cancel-wait-seconds", type=float, default=3.0)
    paper_test.add_argument("--output", default="data/raw/ibkr/paper_test_snapshots.jsonl")
    paper_test.add_argument("--confirm", default="")
    _add_runtime_event_args(paper_test, default_event_log="data/raw/ibkr/paper_test_events.jsonl")
    paper_test.set_defaults(func=_cmd_broker_paper_test)
    round_trip_test = broker_sub.add_parser("round-trip-test")
    round_trip_test.add_argument("--config", default="configs/paper.example.toml")
    round_trip_test.add_argument("--client-id", type=int, default=0)
    round_trip_test.add_argument("--account", default="")
    round_trip_test.add_argument("--notional", type=float, default=100.0)
    round_trip_test.add_argument("--hold-seconds", type=float, default=10.0)
    round_trip_test.add_argument("--synthetic-stop-loss-bps", type=float, default=100.0)
    round_trip_test.add_argument("--monitor-interval-seconds", type=float, default=1.0)
    round_trip_test.add_argument("--order-timeout-seconds", type=float, default=30.0)
    round_trip_test.add_argument("--commission-wait-seconds", type=float, default=2.0)
    round_trip_test.add_argument("--snapshot-timeout-seconds", type=float, default=10.0)
    round_trip_test.add_argument("--account-refresh-timeout-seconds", type=float, default=5.0)
    round_trip_test.add_argument("--pnl-wait-seconds", type=float, default=1.0)
    round_trip_test.add_argument("--max-cash-usd", type=float, default=10_000.0)
    round_trip_test.add_argument("--max-loss-usd", type=float, default=1_000.0)
    round_trip_test.add_argument("--output", default="data/raw/ibkr/round_trip_test.jsonl")
    round_trip_test.add_argument("--confirm", default="")
    _add_runtime_event_args(round_trip_test, default_event_log="data/raw/ibkr/round_trip_events.jsonl")
    round_trip_test.set_defaults(func=_cmd_broker_round_trip_test)
    trade_run = broker_sub.add_parser("trade-run")
    trade_run.add_argument("--config", default="configs/paper.example.toml")
    trade_run.add_argument("--client-id", type=int, default=0)
    trade_run.add_argument("--account", default="")
    trade_run.add_argument("--model-artifact", required=True)
    trade_run.add_argument("--capital-usd", type=float, required=True)
    trade_run.add_argument("--max-loss-usd", type=float, required=True)
    trade_run.add_argument("--max-order-notional-usd", type=float, required=True)
    trade_run.add_argument("--max-open-positions", type=int, default=0)
    trade_run.add_argument("--duration-seconds", type=float, required=True)
    trade_run.add_argument("--signal-interval", type=float, default=1.0)
    trade_run.add_argument(
        "--live-data-mode",
        choices=["streaming", "historical_poll"],
        default="streaming",
        help="streaming keeps IBKR market data subscribed and aggregates completed bars; historical_poll is diagnostic only.",
    )
    trade_run.add_argument(
        "--require-live-1s-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require verified IBKR 1-second historical bars plus a live tick-by-tick subscription before scoring.",
    )
    trade_run.add_argument(
        "--tick-by-tick-type",
        choices=["none", "BidAsk", "Last", "AllLast", "MidPoint"],
        default="Last",
        help=(
            "IBKR tick-by-tick stream used to maintain live completed bars after the "
            "1-second bootstrap. Last/AllLast are required for AGGTRADES/TRADES parity."
        ),
    )
    trade_run.add_argument("--account-refresh-interval-seconds", type=float, default=30.0)
    trade_run.add_argument("--snapshot-timeout-seconds", type=float, default=10.0)
    trade_run.add_argument("--account-refresh-timeout-seconds", type=float, default=5.0)
    trade_run.add_argument("--pnl-wait-seconds", type=float, default=1.0)
    trade_run.add_argument("--order-timeout-seconds", type=float, default=30.0)
    trade_run.add_argument("--commission-wait-seconds", type=float, default=2.0)
    trade_run.add_argument("--synthetic-stop-loss-bps", type=float, default=100.0)
    trade_run.add_argument("--synthetic-profit-target-bps", type=float, default=0.0)
    trade_run.add_argument(
        "--use-model-exit-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the scored sample's model label geometry for synthetic stop/profit exits; "
            "disable to use only --synthetic-stop-loss-bps/--synthetic-profit-target-bps."
        ),
    )
    trade_run.add_argument(
        "--decision-threshold",
        type=float,
        default=0.0,
        help=(
            "Override the artifact's saved probability threshold for controlled paper/live-gated "
            "experiments. Leave 0 to use the artifact threshold."
        ),
    )
    trade_run.add_argument(
        "--max-missing-model-feature-fraction",
        type=float,
        default=0.0,
        help=(
            "Reject live approvals when the scoring sample is missing more than this fraction "
            "of the artifact's trained feature keys."
        ),
    )
    trade_run.add_argument(
        "--max-position-hold-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional hard timed-exit override for runner positions. Leave 0 to use "
            "the signal/model horizon, including adaptive horizons."
        ),
    )
    trade_run.add_argument("--history-duration", default="1800 S")
    trade_run.add_argument("--history-bar-size", default="1 secs")
    trade_run.add_argument("--history-what-to-show", default="AGGTRADES")
    trade_run.add_argument("--history-max-bars", type=int, default=12_000)
    trade_run.add_argument(
        "--live-1s-warmup-bars",
        type=int,
        default=2,
        help="Require this many completed live 1-second streaming bars before the first model score.",
    )
    trade_run.add_argument(
        "--live-1s-warmup-timeout-seconds",
        type=float,
        default=8.0,
        help="Maximum time to wait for completed live 1-second streaming bars during runner startup.",
    )
    trade_run.add_argument(
        "--max-signal-bar-age-seconds",
        type=float,
        default=2.5,
        help="Skip completed-bar signals older than this freshness window.",
    )
    trade_run.add_argument(
        "--candidate-mode",
        choices=["rules", "aggressive_rules", "dense_research", "active_research"],
        default="dense_research",
    )
    trade_run.add_argument("--min-history-bars", type=int, default=240)
    trade_run.add_argument("--candidate-lookback-bars", type=int, default=24)
    trade_run.add_argument("--candidate-rolling-window-bars", type=int, default=1000)
    trade_run.add_argument("--dense-stride-bars", type=int, default=1)
    trade_run.add_argument(
        "--context-bars-jsonl",
        default="",
        help=(
            "Optional comma-separated KEY=path context bar JSONL files to add to live scoring, "
            "matching research context feature names such as ETH or MBT."
        ),
    )
    _add_adaptive_horizon_args(trade_run)
    trade_run.add_argument("--max-scoring-samples", type=int, default=1)
    trade_run.add_argument(
        "--flatten-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flatten any runner-opened BTC spot position when duration/max-loss exits the run.",
    )
    trade_run.add_argument("--state-log", default="data/raw/ibkr/trade_run_state.jsonl")
    trade_run.add_argument("--confirm", default="")
    _add_runtime_event_args(trade_run, default_event_log="data/raw/ibkr/trade_run_events.jsonl")
    trade_run.set_defaults(func=_cmd_broker_trade_run)
    record_quotes = broker_sub.add_parser("record-quotes")
    record_quotes.add_argument("--config", default="configs/paper.example.toml")
    record_quotes.add_argument("--client-id", type=int, default=0)
    record_quotes.add_argument("--account", default="")
    record_quotes.add_argument("--interval-seconds", type=float, default=5.0)
    record_quotes.add_argument("--duration-seconds", type=float, default=60.0)
    record_quotes.add_argument("--snapshot-timeout-seconds", type=float, default=10.0)
    record_quotes.add_argument("--symbol", default="")
    record_quotes.add_argument("--security-type", default="")
    record_quotes.add_argument("--currency", default="")
    record_quotes.add_argument("--exchange", default="")
    record_quotes.add_argument("--last-trade-date-or-contract-month", default="")
    record_quotes.add_argument("--local-symbol", default="")
    record_quotes.add_argument("--output", default="data/raw/ibkr/quotes_btcusd.jsonl")
    _add_runtime_event_args(record_quotes, default_event_log="data/raw/ibkr/quote_recorder_events.jsonl")
    record_quotes.set_defaults(func=_cmd_broker_record_quotes)
    historical_bars = broker_sub.add_parser("historical-bars")
    historical_bars.add_argument("--config", default="configs/paper.example.toml")
    historical_bars.add_argument("--client-id", type=int, default=0)
    historical_bars.add_argument("--account", default="")
    historical_bars.add_argument("--duration", default="2 D")
    historical_bars.add_argument("--bar-size", default="1 min")
    historical_bars.add_argument("--what-to-show", default="MIDPOINT")
    historical_bars.add_argument("--end", default="")
    historical_bars.add_argument("--symbol", default="")
    historical_bars.add_argument("--security-type", default="")
    historical_bars.add_argument("--currency", default="")
    historical_bars.add_argument("--exchange", default="")
    historical_bars.add_argument("--last-trade-date-or-contract-month", default="")
    historical_bars.add_argument("--local-symbol", default="")
    historical_bars.add_argument("--output", default="data/raw/ibkr/historical_btcusd_1m.jsonl")
    historical_bars.set_defaults(func=_cmd_broker_historical_bars)

    kill = sub.add_parser("kill-switch")
    kill.add_argument("action", choices=["enable", "disable"])
    kill.add_argument("--config", default="configs/paper.example.toml")
    kill.set_defaults(func=_cmd_kill_switch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
