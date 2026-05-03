"""ZeroAlpha command line interface."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timedelta, UTC
import json
import math
from pathlib import Path

from zeroalpha.backtest.ml import run_ml_backtest, write_ml_backtest_artifact
from zeroalpha.backtest.simple import run_candidate_backtest, write_backtest_artifact
from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import load_config
from zeroalpha.data.external.binance import (
    BinancePublicDataClient,
    fetch_futures_klines_archive_range,
    fetch_klines_archive_range,
)
from zeroalpha.data.external.coinbase import CoinbaseExchangeClient
from zeroalpha.data.external.ibkr_quotes import read_ibkr_quote_records
from zeroalpha.data.external.prediction_markets import (
    BTC_PREDICTION_MARKET_DURATIONS,
    load_prediction_market_snapshots,
)
from zeroalpha.data.health import health_checks_as_dict, run_external_data_health_checks
from zeroalpha.data.quality import validate_bars, validate_source_divergence
from zeroalpha.db.schema import initialize_sqlite
from zeroalpha.domain import RuntimeMode
from zeroalpha.models.dataset import build_meta_label_samples, label_geometry_diagnostics
from zeroalpha.models.ensemble import (
    report_candidate_type_summary,
    report_side_summary,
    run_meta_label_walk_forward,
    smoke_test_model_stack,
    write_meta_label_report,
)
from zeroalpha.models.sweep import run_label_geometry_sweep, sweep_results_asdict


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
    primary_bars = fetch_klines_archive_range(
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        cache_dir=cache_dir / args.symbol.upper(),
    )
    context_bars = {}
    coverage = {
        "primary": {
            "source": "BINANCE",
            "symbol": args.symbol.upper(),
            "interval": args.interval,
            "bars": len(primary_bars),
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
    for context_symbol in [value.strip() for value in args.context_symbols.split(",") if value.strip()]:
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
        "requested_start": start.isoformat(),
        "fetch_start": fetch_start.isoformat(),
        "end": end.isoformat(),
    }


def _load_ibkr_quote_records(args: argparse.Namespace, start: datetime, end: datetime):
    raw_path = getattr(args, "ibkr_quote_records", "") or ""
    if not raw_path:
        return [], {"enabled": False}
    path = Path(raw_path)
    quotes = read_ibkr_quote_records(path)
    filtered = [
        quote
        for quote in quotes
        if start <= quote.timestamp_utc <= end
        and (not getattr(args, "symbol", "") or "BTC" in quote.symbol.upper())
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
    if getattr(args, "minimum_probability", 0.0):
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
    if getattr(args, "consecutive_loss_limit", 0):
        risk_updates["consecutive_loss_limit"] = args.consecutive_loss_limit
    if getattr(args, "cooldown_hours_after_stopouts", -1) >= 0:
        risk_updates["cooldown_hours_after_stopouts"] = args.cooldown_hours_after_stopouts
    if label_updates:
        cfg = replace(cfg, labels=replace(cfg.labels, **label_updates))
    if model_updates:
        cfg = replace(cfg, model=replace(cfg.model, **model_updates))
    if risk_updates:
        cfg = replace(cfg, risk=replace(cfg.risk, **risk_updates))
    cost_updates = {}
    for attr in ("tier_rate", "base_slippage_bps", "safety_margin_bps"):
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
        min_history_bars=getattr(args, "min_history_bars", 240),
        lookback=getattr(args, "candidate_lookback_bars", 24),
        rolling_window_bars=getattr(args, "candidate_rolling_window_bars", 500),
        mode=candidate_mode,
        dense_stride_bars=getattr(args, "dense_stride_bars", 1),
        side_mode=getattr(args, "side_mode", "long"),
        allow_short_research=allow_short_research,
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
            if _sample_setup_family(sample) in allowed_setups
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
            if _sample_setup_family(sample) not in excluded_setups
        ]
        if samples and not filtered:
            requested = ", ".join(sorted(excluded_setups))
            raise SystemExit(f"--exclude-setup-families removed every sample. Excluded: {requested}")
    return filtered


def _sample_setup_family(sample) -> str:
    for key in ("event_setup_family", "event_dense_setup_family", "setup_family", "dense_setup_family"):
        value = sample.features.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _sample_span_days(samples) -> float:
    if len(samples) < 2:
        return 1.0
    ordered = sorted(sample.timestamp_utc for sample in samples)
    return max((ordered[-1] - ordered[0]).total_seconds() / 86_400, 1 / 24)


def _validate_research_short_backtest_args(args: argparse.Namespace) -> None:
    if getattr(args, "allow_spot_short_research", False) and not getattr(args, "research_gate", False):
        raise SystemExit("--allow-spot-short-research requires --research-gate")
    if getattr(args, "allow_research_short_backtest", False) and not getattr(args, "research_gate", False):
        raise SystemExit("--allow-research-short-backtest requires --research-gate")


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
    _validate_research_short_backtest_args(args)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    data_coverage["label_geometry"] = asdict(
        label_geometry_diagnostics(
            config=cfg,
            assumed_spread_bps=args.assumed_spread_bps,
            research_notional=args.notional,
        )
    )
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        context_bars=context_bars,
        market_quotes=market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        candidate_config=_candidate_config_from_args(args, cfg),
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
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
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
    )
    if args.output:
        write_ml_backtest_artifact(Path(args.output), summary, trades, rejections, report)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _cmd_model_train_meta(args: argparse.Namespace) -> int:
    cfg = _override_config_from_args(load_config(args.config), args)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    data_coverage["label_geometry"] = asdict(
        label_geometry_diagnostics(
            config=cfg,
            assumed_spread_bps=args.assumed_spread_bps,
            research_notional=args.notional,
        )
    )
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        context_bars=context_bars,
        market_quotes=market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
        candidate_config=_candidate_config_from_args(args, cfg),
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
        candidate_type_thresholds=args.candidate_type_thresholds,
        empirical_payoff_ev=args.empirical_payoff_ev,
        selection_score_mode=args.selection_score,
        target_frequency_mode=args.target_frequency_mode,
        selection_score_floor=args.selection_score_floor,
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
    )
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
        "side_summary": report_side_summary(report),
        "folds_detail": [asdict(fold) for fold in report.folds],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_model_smoke(args: argparse.Namespace) -> int:
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    results = smoke_test_model_stack(model_names, isolated=True, timeout_seconds=args.timeout_seconds)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    return 0 if all(result.ok for result in results) else 1


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
    _validate_research_short_backtest_args(args)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    candidate_config = _candidate_config_from_args(args, cfg)
    samples = build_meta_label_samples(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        context_bars=context_bars,
        market_quotes=market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
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
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        selection_setup_families=tuple(_csv_values(args.selection_setup_families)),
        selection_exclude_setup_families=tuple(_csv_values(args.selection_exclude_setup_families)),
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
    _validate_research_short_backtest_args(args)
    start, end = _date_range_from_args(args)
    primary_bars, context_bars, data_coverage = _load_research_bars(args, start, end)
    prediction_market_snapshots, prediction_market_coverage = _load_prediction_market_signals(args, start, end)
    market_quotes, ibkr_quote_coverage = _load_ibkr_quote_records(args, start, end)
    data_coverage["kronos"] = asdict(cfg.kronos)
    data_coverage["prediction_markets"] = prediction_market_coverage
    data_coverage["ibkr_quotes"] = ibkr_quote_coverage
    model_names = [value.strip().lower() for value in args.models.split(",") if value.strip()]
    results = run_label_geometry_sweep(
        primary_bars,
        config=cfg,
        assumed_spread_bps=args.assumed_spread_bps,
        research_notional=args.notional,
        starting_equity=args.starting_equity,
        context_bars=context_bars,
        market_quotes=market_quotes,
        prediction_market_snapshots=prediction_market_snapshots,
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
        specialist_models=args.specialist_models,
        require_calibrated_selection=args.require_calibrated_selection,
        min_signal_spacing_hours=args.min_signal_spacing_hours,
        max_signals_per_group_per_day=args.max_signals_per_group_per_day,
        max_signals_per_timestamp=args.max_signals_per_timestamp,
        tune_hyperparameters=args.hpo,
        hpo_profile=args.hpo_profile,
        hpo_trials=args.hpo_trials,
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

    cfg = load_config(args.config)
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

    cfg = load_config(args.config)
    _validate_paper_order_test_config(cfg, args)
    broker = IBKRBroker(cfg)
    await broker.connect(read_only=False)
    try:
        contract = await broker.qualify_crypto_contract()
        quote = await broker.snapshot_quote(contract)
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
        await broker.wait(args.wait_seconds)
        broker.cancel_trade(trade)
        await broker.wait(args.cancel_wait_seconds)
        status = getattr(getattr(trade, "orderStatus", None), "status", "unknown")
        order_id = getattr(getattr(trade, "order", None), "orderId", None)
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
        await broker.disconnect()
    return 0


def _cmd_broker_order_test(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_order_test_async(args))


def _validate_paper_order_test_config(cfg, args: argparse.Namespace) -> None:
    if cfg.runtime.mode != RuntimeMode.PAPER:
        raise SystemExit("broker order-test is paper-only; refusing non-paper config")
    if cfg.broker.port not in {4002, 7497}:
        raise SystemExit("broker order-test requires a standard IBKR paper port (4002 or 7497)")
    if args.confirm != "PAPER_ORDER_TEST":
        raise SystemExit('broker order-test requires --confirm PAPER_ORDER_TEST')


async def _broker_record_quotes_async(args: argparse.Namespace) -> int:
    from zeroalpha.broker.quote_recorder import IBKRQuoteRecorder

    cfg = load_config(args.config)
    recorder = IBKRQuoteRecorder(
        cfg,
        output_path=Path(args.output),
        interval_seconds=args.interval_seconds,
    )
    count = await recorder.run(duration_seconds=args.duration_seconds)
    print(f"ok: wrote {count} quote records to {args.output}")
    return 0


def _cmd_broker_record_quotes(args: argparse.Namespace) -> int:
    return asyncio.run(_broker_record_quotes_async(args))


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


def _add_ibkr_quote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ibkr-quote-records",
        default="",
        help="Path to broker record-quotes JSONL for IBKR bid/ask, spread, and top-of-book size features.",
    )


def _add_target_frequency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-frequency-mode",
        choices=["strict", "quota"],
        default="strict",
        help=(
            "strict keeps the old probability/EV pre-gates before ranking; quota ranks "
            "all non-vetoed candidates to deliberately hit the requested daily turnover."
        ),
    )
    parser.add_argument(
        "--selection-score-floor",
        type=float,
        default=None,
        help="Optional minimum selection score for quota mode. Omit to always choose the daily best-ranked setups.",
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
    ml_backtest.add_argument("--max-holding-hours", type=int, default=0)
    ml_backtest.add_argument("--net-profit-target", type=float, default=0.0)
    ml_backtest.add_argument("--net-stop-loss", type=float, default=0.0)
    ml_backtest.add_argument("--volatility-lookback-bars", type=int, default=0)
    ml_backtest.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    ml_backtest.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-gross-profit-bps", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-gross-stop-bps", type=float, default=0.0)
    ml_backtest.add_argument("--minimum-probability", type=float, default=0.0)
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
    ml_backtest.add_argument(
        "--selection-score",
        choices=["probability", "expected_value", "predicted_return", "expected_utility"],
        default="expected_value",
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
    ml_backtest.add_argument("--base-slippage-bps", type=float, default=None)
    ml_backtest.add_argument("--safety-margin-bps", type=float, default=None)
    ml_backtest.add_argument("--max-open-positions", type=int, default=0)
    ml_backtest.add_argument("--consecutive-loss-limit", type=int, default=0)
    ml_backtest.add_argument("--cooldown-hours-after-stopouts", type=int, default=-1)
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
    ml_backtest.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota"], default="standard")
    ml_backtest.add_argument("--hpo-trials", type=int, default=0)
    ml_backtest.add_argument("--foundation-max-samples", type=int, default=0)
    ml_backtest.add_argument("--kronos-features", action="store_true")
    ml_backtest.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    ml_backtest.add_argument("--kronos-lookback-bars", type=int, default=0)
    ml_backtest.add_argument("--kronos-embedding-dims", type=int, default=0)
    ml_backtest.add_argument("--kronos-device", default="")
    _add_prediction_market_args(ml_backtest)
    _add_ibkr_quote_args(ml_backtest)
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
    train_meta.add_argument("--max-holding-hours", type=int, default=0)
    train_meta.add_argument("--net-profit-target", type=float, default=0.0)
    train_meta.add_argument("--net-stop-loss", type=float, default=0.0)
    train_meta.add_argument("--volatility-lookback-bars", type=int, default=0)
    train_meta.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    train_meta.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    train_meta.add_argument("--minimum-gross-profit-bps", type=float, default=0.0)
    train_meta.add_argument("--minimum-gross-stop-bps", type=float, default=0.0)
    train_meta.add_argument("--minimum-probability", type=float, default=0.0)
    train_meta.add_argument("--minimum-expected-value", type=float, default=None)
    train_meta.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="")
    train_meta.add_argument("--target-trades-per-day", type=float, default=0.0)
    _add_target_frequency_args(train_meta)
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
        choices=["probability", "expected_value", "predicted_return", "expected_utility"],
        default="expected_value",
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
    train_meta.add_argument("--base-slippage-bps", type=float, default=None)
    train_meta.add_argument("--safety-margin-bps", type=float, default=None)
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
    train_meta.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota"], default="standard")
    train_meta.add_argument("--hpo-trials", type=int, default=0)
    train_meta.add_argument("--foundation-max-samples", type=int, default=0)
    train_meta.add_argument("--kronos-features", action="store_true")
    train_meta.add_argument("--kronos-mode", choices=["proxy", "auto", "official"], default="")
    train_meta.add_argument("--kronos-lookback-bars", type=int, default=0)
    train_meta.add_argument("--kronos-embedding-dims", type=int, default=0)
    train_meta.add_argument("--kronos-device", default="")
    _add_prediction_market_args(train_meta)
    _add_ibkr_quote_args(train_meta)
    train_meta.add_argument(
        "--output",
        default="artifacts/models/meta_label_walk_forward_btcusdt_1h.json",
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
    signal_audit.add_argument("--max-holding-hours", type=int, default=4)
    signal_audit.add_argument("--net-profit-target", type=float, default=0.001)
    signal_audit.add_argument("--net-stop-loss", type=float, default=0.001)
    signal_audit.add_argument("--volatility-lookback-bars", type=int, default=96)
    signal_audit.add_argument("--profit-volatility-multiplier", type=float, default=0.0)
    signal_audit.add_argument("--stop-volatility-multiplier", type=float, default=0.0)
    signal_audit.add_argument("--minimum-gross-profit-bps", type=float, default=100.0)
    signal_audit.add_argument("--minimum-gross-stop-bps", type=float, default=80.0)
    signal_audit.add_argument("--minimum-probability", type=float, default=0.0)
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
    signal_audit.add_argument("--risk-per-trade", type=float, default=0.0)
    signal_audit.add_argument("--daily-loss-stop", type=float, default=0.0)
    signal_audit.add_argument("--weekly-loss-stop", type=float, default=0.0)
    signal_audit.add_argument("--rolling-drawdown-stop", type=float, default=0.0)
    signal_audit.add_argument("--paper-max-notional", type=float, default=0.0)
    signal_audit.add_argument("--minimum-fee-efficient-notional", type=float, default=0.0)
    signal_audit.add_argument("--tier-rate", type=float, default=0.0004)
    signal_audit.add_argument("--base-slippage-bps", type=float, default=1.0)
    signal_audit.add_argument("--safety-margin-bps", type=float, default=2.0)
    signal_audit.add_argument("--max-open-positions", type=int, default=4)
    signal_audit.add_argument("--consecutive-loss-limit", type=int, default=0)
    signal_audit.add_argument("--cooldown-hours-after-stopouts", type=int, default=-1)
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
    signal_audit.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota"], default="standard")
    signal_audit.add_argument("--hpo-trials", type=int, default=0)
    signal_audit.add_argument("--foundation-max-samples", type=int, default=0)
    signal_audit.add_argument("--selection-score", choices=["probability", "expected_value", "predicted_return", "expected_utility"], default="expected_utility")
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
    sweep_labels.add_argument("--selection-score", choices=["probability", "expected_value", "predicted_return", "expected_utility"], default="expected_value")
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
    sweep_labels.add_argument("--hpo-profile", choices=["standard", "deep", "wide", "quota"], default="standard")
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
    smoke.add_argument("--read-only", action="store_true", default=True)
    smoke.set_defaults(func=_cmd_broker_smoke)
    order_test = broker_sub.add_parser("order-test")
    order_test.add_argument("--config", default="configs/paper.example.toml")
    order_test.add_argument("--notional", type=float, default=100.0)
    order_test.add_argument("--offset-bps", type=float, default=100.0)
    order_test.add_argument("--price-increment", type=float, default=0.25)
    order_test.add_argument("--wait-seconds", type=float, default=3.0)
    order_test.add_argument("--cancel-wait-seconds", type=float, default=2.0)
    order_test.add_argument("--confirm", default="")
    order_test.set_defaults(func=_cmd_broker_order_test)
    record_quotes = broker_sub.add_parser("record-quotes")
    record_quotes.add_argument("--config", default="configs/paper.example.toml")
    record_quotes.add_argument("--interval-seconds", type=float, default=5.0)
    record_quotes.add_argument("--duration-seconds", type=float, default=60.0)
    record_quotes.add_argument("--output", default="data/raw/ibkr/quotes_btcusd.jsonl")
    record_quotes.set_defaults(func=_cmd_broker_record_quotes)

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
