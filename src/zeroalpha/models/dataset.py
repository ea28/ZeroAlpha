"""Meta-label event dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from bisect import bisect_right
from math import sqrt
from statistics import median, pstdev
from typing import Mapping, Sequence

from zeroalpha.candidates.events import CandidateGenerationConfig, generate_candidate_events
from zeroalpha.config import AppConfig
from zeroalpha.costs import CommissionModel, SlippageModel, estimate_round_trip_cost
from zeroalpha.data.external.prediction_markets import (
    PreparedPredictionMarketSnapshots,
    PredictionMarketSnapshot,
    prediction_market_duration_seconds,
)
from zeroalpha.domain import Bar, CandidateEvent, TripleBarrierLabel
from zeroalpha.features.basic import build_event_features
from zeroalpha.features.kronos import build_kronos_features
from zeroalpha.features.regime import classify_market_regime
from zeroalpha.labels.triple_barrier import label_event
from zeroalpha.timeutils import ensure_utc


@dataclass(frozen=True, slots=True)
class MetaLabelSample:
    event_id: str
    timestamp_utc: datetime
    t1: datetime
    candidate_type: str
    side: str
    net_profit_target: float
    net_stop_loss: float
    features: dict[str, float | str]
    label: int
    net_return: float
    notional: float
    round_trip_cost_bps: float
    outcome_type: str
    label_detail: TripleBarrierLabel

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        object.__setattr__(self, "t1", ensure_utc(self.t1))


@dataclass(frozen=True, slots=True)
class LabelGeometryDiagnostics:
    notional: float
    round_trip_cost_bps: float
    round_trip_cost_fraction: float
    net_profit_target: float
    net_stop_loss: float
    gross_profit_move: float
    gross_stop_distance: float
    stop_to_cost_ratio: float
    warning: str


@dataclass(frozen=True, slots=True)
class _PreparedContext:
    timestamps: tuple[datetime, ...]
    closes: tuple[float, ...]
    volumes: tuple[float, ...]


def _prepare_context_bars(context_bars: Mapping[str, list[Bar]] | None) -> dict[str, _PreparedContext]:
    prepared: dict[str, _PreparedContext] = {}
    if not context_bars:
        return prepared
    for raw_name, bars in context_bars.items():
        name = raw_name.lower().replace("/", "").replace("-", "").replace("_", "")
        ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
        prepared[name] = _PreparedContext(
            timestamps=tuple(bar.timestamp_utc for bar in ordered),
            closes=tuple(bar.close for bar in ordered),
            volumes=tuple(bar.volume for bar in ordered),
        )
    return prepared


def _latest_close_before(context: _PreparedContext, timestamp: datetime, periods: int) -> float | None:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx < periods:
        return None
    return context.closes[idx] / context.closes[idx - periods] - 1


def _latest_close_before_duration(
    context: _PreparedContext,
    timestamp: datetime,
    lookback: timedelta,
) -> float | None:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx < 0:
        return None
    target_idx = bisect_right(context.timestamps, ensure_utc(timestamp) - lookback) - 1
    if target_idx < 0 or target_idx >= idx:
        return None
    base = context.closes[target_idx]
    return context.closes[idx] / base - 1 if base > 0 else None


def _latest_close_value_before(context: _PreparedContext, timestamp: datetime) -> float | None:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx < 0:
        return None
    return context.closes[idx]


def _context_returns_before(context: _PreparedContext, timestamp: datetime, periods: int) -> list[float]:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx <= 0:
        return []
    start = max(1, idx - periods + 1)
    return [
        context.closes[position] / context.closes[position - 1] - 1
        for position in range(start, idx + 1)
        if context.closes[position - 1] > 0
    ]


def _context_volume_ratio_before(context: _PreparedContext, timestamp: datetime, periods: int) -> float:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx <= 0:
        return 0.0
    start = max(0, idx - periods)
    baseline = context.volumes[start:idx]
    average_volume = sum(baseline) / len(baseline) if baseline else 0.0
    return context.volumes[idx] / average_volume if average_volume > 0 else 0.0


def _add_cross_asset_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    context_bars: Mapping[str, _PreparedContext] | None,
) -> None:
    if not context_bars:
        return
    btc_return_24 = float(features.get("return_elapsed_24h", features.get("return_24", 0.0)) or 0.0)
    btc_vol_24 = float(features.get("realized_vol_elapsed_24h", features.get("realized_vol_24", 0.0)) or 0.0)
    btc_vol_72 = float(features.get("realized_vol_elapsed_72h", features.get("realized_vol_72", 0.0)) or 0.0)
    btc_close = float(features.get("bar_close", 0.0) or 0.0)
    for name, context in context_bars.items():
        context_idx = bisect_right(context.timestamps, ensure_utc(event.timestamp_utc)) - 1
        if context_idx < 0:
            features[f"{name}_available"] = 0.0
            continue
        context_age = event.timestamp_utc - context.timestamps[context_idx]
        features[f"{name}_available"] = 1.0
        features[f"{name}_age_seconds"] = max(0.0, context_age.total_seconds())
        context_close = context.closes[context_idx]
        if btc_close > 0 and context_close > 0:
            features[f"{name}_close_to_btc"] = context_close / btc_close - 1
        for periods in (1, 4, 24, 72):
            value = _latest_close_before(context, event.timestamp_utc, periods)
            if value is not None:
                features[f"{name}_return_{periods}"] = value
        for label, lookback in {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "12h": timedelta(hours=12),
            "24h": timedelta(hours=24),
            "72h": timedelta(hours=72),
            "168h": timedelta(hours=168),
        }.items():
            value = _latest_close_before_duration(context, event.timestamp_utc, lookback)
            if value is not None:
                features[f"{name}_return_elapsed_{label}"] = value
        alt_return_24 = features.get(f"{name}_return_24")
        alt_return_24 = features.get(f"{name}_return_elapsed_24h", alt_return_24)
        if isinstance(alt_return_24, float):
            features[f"btc_relative_strength_vs_{name}_24"] = btc_return_24 - alt_return_24
            features[f"{name}_return_spread_24"] = alt_return_24 - btc_return_24
        context_return_1h = features.get(f"{name}_return_elapsed_1h")
        btc_return_1h = features.get("return_elapsed_1h")
        if isinstance(context_return_1h, float) and isinstance(btc_return_1h, float):
            features[f"{name}_return_spread_1h"] = context_return_1h - btc_return_1h
        context_return_4h = features.get(f"{name}_return_elapsed_4h")
        btc_return_4h = features.get("return_elapsed_4h")
        if isinstance(context_return_4h, float) and isinstance(btc_return_4h, float):
            features[f"{name}_return_spread_4h"] = context_return_4h - btc_return_4h
        for periods, btc_vol in ((24, btc_vol_24), (72, btc_vol_72)):
            returns = _context_returns_before(context, event.timestamp_utc, periods)
            vol = pstdev(returns) if len(returns) > 1 else 0.0
            features[f"{name}_realized_vol_{periods}"] = vol
            features[f"{name}_momentum_consistency_{periods}"] = (
                sum(value > 0 for value in returns) / len(returns) if returns else 0.0
            )
            features[f"{name}_volatility_ratio_vs_btc_{periods}"] = vol / btc_vol if btc_vol > 0 else 0.0
            features[f"{name}_volume_ratio_{periods}"] = _context_volume_ratio_before(
                context,
                event.timestamp_utc,
                periods,
            )


def _add_event_metadata(features: dict[str, float | str], event: CandidateEvent) -> None:
    for key, value in event.metadata.items():
        if isinstance(value, bool):
            features[f"event_{key}"] = float(value)
        elif isinstance(value, int | float):
            features[f"event_{key}"] = float(value)
        elif isinstance(value, str):
            features[f"event_{key}"] = value


def _prediction_market_max_age(duration: str) -> timedelta:
    try:
        seconds = prediction_market_duration_seconds(duration)
    except ValueError:
        seconds = 3600
    return timedelta(seconds=max(seconds, 900))


def _set_float_feature(
    features: dict[str, float | str],
    key: str,
    value: float | None,
) -> None:
    if value is not None:
        features[key] = float(value)


def _add_prediction_market_snapshot_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    snapshot: PredictionMarketSnapshot,
) -> None:
    prefix = f"pm_{snapshot.provider}_{snapshot.duration}"
    event_time = ensure_utc(event.timestamp_utc)
    features[f"{prefix}_available"] = 1.0
    features[f"{prefix}_age_seconds"] = max(
        0.0,
        (event_time - snapshot.timestamp_utc).total_seconds(),
    )
    try:
        features[f"{prefix}_duration_seconds"] = float(prediction_market_duration_seconds(snapshot.duration))
    except ValueError:
        pass
    if snapshot.window_start_utc is not None and snapshot.window_end_utc is not None:
        total_seconds = (snapshot.window_end_utc - snapshot.window_start_utc).total_seconds()
        elapsed_seconds = (event_time - snapshot.window_start_utc).total_seconds()
        features[f"{prefix}_seconds_to_close"] = (
            (snapshot.window_end_utc - event_time).total_seconds()
        )
        features[f"{prefix}_progress"] = (
            min(max(elapsed_seconds / total_seconds, 0.0), 1.0) if total_seconds > 0 else 0.0
        )
    for name, value in (
        ("up_bid", snapshot.up_bid),
        ("up_ask", snapshot.up_ask),
        ("up_mid", snapshot.up_mid),
        ("down_bid", snapshot.down_bid),
        ("down_ask", snapshot.down_ask),
        ("down_mid", snapshot.down_mid),
        ("up_bid_size", snapshot.up_bid_size),
        ("up_ask_size", snapshot.up_ask_size),
        ("down_bid_size", snapshot.down_bid_size),
        ("down_ask_size", snapshot.down_ask_size),
        ("last_price", snapshot.last_price),
        ("volume", snapshot.volume),
        ("volume_24h", snapshot.volume_24h),
        ("liquidity", snapshot.liquidity),
        ("open_interest", snapshot.open_interest),
    ):
        _set_float_feature(features, f"{prefix}_{name}", value)
    if snapshot.up_bid is not None and snapshot.up_ask is not None:
        features[f"{prefix}_up_spread"] = snapshot.up_ask - snapshot.up_bid
    if snapshot.down_bid is not None and snapshot.down_ask is not None:
        features[f"{prefix}_down_spread"] = snapshot.down_ask - snapshot.down_bid
    if snapshot.up_mid is not None and snapshot.down_mid is not None:
        direction_skew = snapshot.up_mid - snapshot.down_mid
        features[f"{prefix}_direction_skew"] = direction_skew
        features[f"{prefix}_absolute_skew"] = abs(direction_skew)
        features[f"{prefix}_binary_mid_overround"] = snapshot.up_mid + snapshot.down_mid - 1.0
        features[f"{prefix}_side_aligned_skew"] = (
            direction_skew if event.side.value == "BUY" else -direction_skew
        )
        features[f"{prefix}_side_aligned_mid"] = (
            snapshot.up_mid if event.side.value == "BUY" else snapshot.down_mid
        )
    if snapshot.up_bid_size is not None and snapshot.down_bid_size is not None:
        denominator = snapshot.up_bid_size + snapshot.down_bid_size
        features[f"{prefix}_bid_size_imbalance"] = (
            (snapshot.up_bid_size - snapshot.down_bid_size) / denominator if denominator > 0 else 0.0
        )
    if snapshot.up_ask_size is not None and snapshot.down_ask_size is not None:
        denominator = snapshot.up_ask_size + snapshot.down_ask_size
        features[f"{prefix}_ask_size_imbalance"] = (
            (snapshot.down_ask_size - snapshot.up_ask_size) / denominator if denominator > 0 else 0.0
        )


def _add_prediction_market_change_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    snapshot: PredictionMarketSnapshot,
    prior: PredictionMarketSnapshot | None,
) -> None:
    if prior is None:
        return
    prefix = f"pm_{snapshot.provider}_{snapshot.duration}"
    gap_seconds = (snapshot.timestamp_utc - prior.timestamp_utc).total_seconds()
    if gap_seconds <= 0:
        return
    features[f"{prefix}_change_gap_seconds"] = gap_seconds
    for name in (
        "up_mid",
        "down_mid",
        "up_bid",
        "up_ask",
        "down_bid",
        "down_ask",
        "last_price",
        "liquidity",
        "open_interest",
    ):
        current = getattr(snapshot, name)
        previous = getattr(prior, name)
        if current is not None and previous is not None:
            features[f"{prefix}_{name}_change"] = current - previous
    if (
        snapshot.up_mid is not None
        and snapshot.down_mid is not None
        and prior.up_mid is not None
        and prior.down_mid is not None
    ):
        current_skew = snapshot.up_mid - snapshot.down_mid
        prior_skew = prior.up_mid - prior.down_mid
        skew_change = current_skew - prior_skew
        features[f"{prefix}_direction_skew_change"] = skew_change
        features[f"{prefix}_side_aligned_skew_change"] = (
            skew_change if event.side.value == "BUY" else -skew_change
        )
    current_side_mid = snapshot.up_mid if event.side.value == "BUY" else snapshot.down_mid
    prior_side_mid = prior.up_mid if event.side.value == "BUY" else prior.down_mid
    if current_side_mid is not None and prior_side_mid is not None:
        features[f"{prefix}_side_aligned_mid_change"] = current_side_mid - prior_side_mid


def _add_prediction_market_spot_divergence_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    snapshot: PredictionMarketSnapshot,
) -> None:
    if snapshot.up_mid is None or snapshot.down_mid is None:
        return
    spot_return = features.get(f"return_elapsed_{snapshot.duration}")
    if not isinstance(spot_return, int | float):
        return
    prefix = f"pm_{snapshot.provider}_{snapshot.duration}"
    direction_skew = snapshot.up_mid - snapshot.down_mid
    side_aligned_skew = direction_skew if event.side.value == "BUY" else -direction_skew
    side_aligned_spot = float(spot_return) if event.side.value == "BUY" else -float(spot_return)
    if side_aligned_spot > 0:
        spot_direction = 1.0
    elif side_aligned_spot < 0:
        spot_direction = -1.0
    else:
        spot_direction = 0.0
    features[f"{prefix}_side_spot_return"] = side_aligned_spot
    features[f"{prefix}_side_spot_agreement"] = (
        1.0 if side_aligned_skew * side_aligned_spot > 0 else -1.0 if side_aligned_skew * side_aligned_spot < 0 else 0.0
    )
    features[f"{prefix}_side_skew_minus_spot_direction"] = side_aligned_skew - spot_direction
    features[f"{prefix}_side_skew_to_abs_spot_return"] = (
        side_aligned_skew / abs(side_aligned_spot) if abs(side_aligned_spot) > 1e-9 else 0.0
    )


def _add_prediction_market_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    prediction_markets: PreparedPredictionMarketSnapshots | None,
) -> None:
    if prediction_markets is None:
        return
    latest: dict[tuple[str, str], PredictionMarketSnapshot] = {}
    for provider, duration in sorted(prediction_markets.by_provider_duration):
        snapshot = prediction_markets.latest_active_before(
            provider=provider,
            duration=duration,
            timestamp=event.timestamp_utc,
            max_age=_prediction_market_max_age(duration),
        )
        if snapshot is None:
            features[f"pm_{provider}_{duration}_available"] = 0.0
            continue
        latest[(provider, duration)] = snapshot
        _add_prediction_market_snapshot_features(features, event=event, snapshot=snapshot)
        prior = prediction_markets.latest_active_before(
            provider=provider,
            duration=duration,
            timestamp=snapshot.timestamp_utc - timedelta(microseconds=1),
            max_age=_prediction_market_max_age(duration) * 3,
        )
        _add_prediction_market_change_features(
            features,
            event=event,
            snapshot=snapshot,
            prior=prior,
        )
        _add_prediction_market_spot_divergence_features(features, event=event, snapshot=snapshot)
    for duration in sorted({duration for _, duration in latest}):
        poly = latest.get(("polymarket", duration))
        kalshi = latest.get(("kalshi", duration))
        if poly is None or kalshi is None:
            continue
        prefix = f"pm_cross_polymarket_kalshi_{duration}"
        if poly.up_mid is not None and kalshi.up_mid is not None:
            features[f"{prefix}_up_mid_diff"] = poly.up_mid - kalshi.up_mid
            features[f"{prefix}_up_mid_abs_diff"] = abs(poly.up_mid - kalshi.up_mid)
        if poly.down_mid is not None and kalshi.down_mid is not None:
            features[f"{prefix}_down_mid_diff"] = poly.down_mid - kalshi.down_mid
        if poly.up_mid is not None and poly.down_mid is not None and kalshi.up_mid is not None:
            kalshi_down_mid = kalshi.down_mid if kalshi.down_mid is not None else 1.0 - kalshi.up_mid
            features[f"{prefix}_direction_skew_diff"] = (
                (poly.up_mid - poly.down_mid) - (kalshi.up_mid - kalshi_down_mid)
            )


def _bar_interval_seconds(bars: list[Bar]) -> float:
    if len(bars) < 2:
        return 3600.0
    diffs = [
        (bars[idx].timestamp_utc - bars[idx - 1].timestamp_utc).total_seconds()
        for idx in range(1, min(len(bars), 200))
    ]
    positive = [value for value in diffs if value > 0]
    return median(positive) if positive else 3600.0


def _recent_return_volatility(bars: list[Bar], *, lookback: int) -> float:
    if len(bars) < 3:
        return 0.0
    window = bars[-max(lookback + 1, 2) :]
    returns = [window[idx].close / window[idx - 1].close - 1 for idx in range(1, len(window))]
    return pstdev(returns) if len(returns) > 1 else 0.0


def _event_label_geometry(
    *,
    event: CandidateEvent,
    history_bars: list[Bar],
    config: AppConfig,
    cost_fraction: float,
) -> tuple[float, float, float]:
    interval_seconds = _bar_interval_seconds(history_bars)
    horizon_bars = max(1.0, event.max_holding_period.total_seconds() / interval_seconds)
    horizon_volatility = _recent_return_volatility(
        history_bars,
        lookback=config.labels.volatility_lookback_bars,
    ) * sqrt(horizon_bars)
    min_gross_profit = config.labels.minimum_gross_profit_bps / 10_000
    min_gross_stop = config.labels.minimum_gross_stop_bps / 10_000
    net_profit_target = max(
        config.labels.net_profit_target,
        min_gross_profit - cost_fraction,
        config.labels.profit_volatility_multiplier * horizon_volatility - cost_fraction,
    )
    net_stop_loss = max(
        config.labels.net_stop_loss,
        cost_fraction + min_gross_stop,
        cost_fraction + config.labels.stop_volatility_multiplier * horizon_volatility,
    )
    return max(net_profit_target, 1e-9), max(net_stop_loss, 1e-9), horizon_volatility


def label_geometry_diagnostics(
    *,
    config: AppConfig,
    assumed_spread_bps: float,
    research_notional: float | None = None,
) -> LabelGeometryDiagnostics:
    notional = min(research_notional or config.risk.paper_max_notional, config.risk.paper_max_notional)
    commission_model = CommissionModel(
        tier_rate=config.cost.tier_rate,
        minimum_commission=config.cost.minimum_commission,
        maximum_commission_rate=config.cost.maximum_commission_rate,
    )
    slippage_model = SlippageModel(base_slippage_bps=config.cost.base_slippage_bps)
    cost = estimate_round_trip_cost(
        notional,
        spread_bps=assumed_spread_bps,
        commission_model=commission_model,
        slippage_model=slippage_model,
        safety_margin_bps=config.cost.safety_margin_bps,
    )
    gross_profit_move = max(
        config.labels.net_profit_target + cost.total_return_fraction,
        config.labels.minimum_gross_profit_bps / 10_000,
    )
    gross_stop_distance = max(
        config.labels.net_stop_loss - cost.total_return_fraction,
        config.labels.minimum_gross_stop_bps / 10_000,
    )
    if gross_stop_distance <= 0:
        warning = "invalid_stop_loss_not_above_cost"
    elif gross_stop_distance < 0.005:
        warning = "gross_stop_distance_below_50_bps"
    elif gross_stop_distance < 0.01:
        warning = "gross_stop_distance_below_100_bps"
    else:
        warning = ""
    return LabelGeometryDiagnostics(
        notional=notional,
        round_trip_cost_bps=cost.total_bps,
        round_trip_cost_fraction=cost.total_return_fraction,
        net_profit_target=config.labels.net_profit_target,
        net_stop_loss=config.labels.net_stop_loss,
        gross_profit_move=gross_profit_move,
        gross_stop_distance=gross_stop_distance,
        stop_to_cost_ratio=(
            config.labels.net_stop_loss / cost.total_return_fraction
            if cost.total_return_fraction > 0
            else float("inf")
        ),
        warning=warning,
    )


def build_meta_label_samples(
    bars: list[Bar],
    *,
    config: AppConfig,
    assumed_spread_bps: float,
    research_notional: float | None = None,
    context_bars: Mapping[str, list[Bar]] | None = None,
    prediction_market_snapshots: Sequence[PredictionMarketSnapshot] | None = None,
    candidate_config: CandidateGenerationConfig | None = None,
) -> list[MetaLabelSample]:
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    if len(ordered) < 300:
        raise ValueError("not enough bars for meta-label dataset")
    notional = min(research_notional or config.risk.paper_max_notional, config.risk.paper_max_notional)
    if notional < config.risk.minimum_fee_efficient_notional:
        raise ValueError("research_notional is below fee-efficient minimum")

    diagnostics = label_geometry_diagnostics(
        config=config,
        assumed_spread_bps=assumed_spread_bps,
        research_notional=notional,
    )
    if diagnostics.gross_stop_distance <= 0:
        raise ValueError("label stop geometry must leave positive gross stop distance after round-trip cost")

    events = generate_candidate_events(ordered, config=candidate_config)
    prepared_context = _prepare_context_bars(context_bars)
    prepared_prediction_markets = (
        PreparedPredictionMarketSnapshots.from_snapshots(prediction_market_snapshots)
        if prediction_market_snapshots
        else None
    )
    timestamps = [bar.timestamp_utc for bar in ordered]
    by_timestamp = {bar.timestamp_utc: idx for idx, bar in enumerate(ordered)}
    samples: list[MetaLabelSample] = []
    feature_window_bars = max(
        candidate_config.rolling_window_bars if candidate_config else 500,
        10_080,
    )
    for event in events:
        idx = by_timestamp.get(event.timestamp_utc)
        if idx is None or idx + 1 >= len(ordered):
            continue
        horizon_end = bisect_right(timestamps, event.vertical_barrier_timestamp_utc)
        future = ordered[idx + 1 : horizon_end]
        if not future:
            continue
        entry_bar = ordered[idx + 1]
        feature_start = max(0, idx + 1 - feature_window_bars)
        history = ordered[feature_start : idx + 1]
        net_profit_target, net_stop_loss, horizon_volatility = _event_label_geometry(
            event=event,
            history_bars=history,
            config=config,
            cost_fraction=diagnostics.round_trip_cost_fraction,
        )
        label = label_event(
            event,
            future,
            entry_price=entry_bar.open,
            net_profit_target=net_profit_target,
            net_stop_loss=net_stop_loss,
            round_trip_cost_bps=diagnostics.round_trip_cost_bps,
            conservative_same_bar=config.labels.conservative_same_bar,
        )
        features = build_event_features(event, history)
        features.update(classify_market_regime(history).as_features())
        _add_event_metadata(features, event)
        _add_cross_asset_features(features, event=event, context_bars=prepared_context)
        _add_prediction_market_features(
            features,
            event=event,
            prediction_markets=prepared_prediction_markets,
        )
        features.update(build_kronos_features(history, config=config.kronos))
        features.update(
            {
                "round_trip_cost_bps": diagnostics.round_trip_cost_bps,
                "assumed_spread_bps": assumed_spread_bps,
                "net_profit_target": net_profit_target,
                "net_stop_loss": net_stop_loss,
                "max_holding_hours": float(event.max_holding_hours),
                "gross_profit_move": net_profit_target + diagnostics.round_trip_cost_fraction,
                "gross_stop_distance": net_stop_loss - diagnostics.round_trip_cost_fraction,
                "horizon_volatility": horizon_volatility,
            }
        )
        samples.append(
            MetaLabelSample(
                event_id=event.event_id,
                timestamp_utc=event.timestamp_utc,
                t1=label.t1,
                candidate_type=event.candidate_type,
                side=event.side.value,
                net_profit_target=net_profit_target,
                net_stop_loss=net_stop_loss,
                features=features,
                label=label.label,
                net_return=label.net_return,
                notional=notional,
                round_trip_cost_bps=diagnostics.round_trip_cost_bps,
                outcome_type=label.outcome_type,
                label_detail=label,
            )
        )
    return samples
