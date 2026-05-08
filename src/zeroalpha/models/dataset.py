"""Meta-label event dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from bisect import bisect_right
from math import isfinite, log1p, sqrt
from statistics import median, pstdev
from typing import Mapping, Sequence

from zeroalpha.bars import bar_start_timestamp_utc
from zeroalpha.candidates.events import (
    CandidateGenerationConfig,
    candidates_for_index,
    generate_candidate_events,
    validate_candidate_generation_config,
)
from zeroalpha.config import AppConfig
from zeroalpha.costs import CommissionModel, SlippageModel, estimate_round_trip_cost
from zeroalpha.data.external.prediction_markets import (
    PreparedPredictionMarketSnapshots,
    PredictionMarketSnapshot,
    prediction_market_duration_seconds,
)
from zeroalpha.domain import Bar, CandidateEvent, MarketQuote, Side, TripleBarrierLabel
from zeroalpha.features.basic import build_event_features
from zeroalpha.features.kronos import build_kronos_features
from zeroalpha.features.regime import classify_market_regime
from zeroalpha.labels.triple_barrier import label_event
from zeroalpha.timeutils import ensure_utc


PREDICTION_MARKET_FEATURE_PROFILES = ("stable", "full")
FEATURE_ASOF_CHOICES = ("signal", "entry")


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
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    time_to_exit_seconds: float = 0.0
    early_adverse_label: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        object.__setattr__(self, "t1", ensure_utc(self.t1))


@dataclass(frozen=True, slots=True)
class ScoringSample:
    event_id: str
    timestamp_utc: datetime
    candidate_type: str
    side: str
    reference_price: float
    features: dict[str, float | str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))


@dataclass(frozen=True, slots=True)
class LabelGeometryDiagnostics:
    notional: float
    round_trip_cost_bps: float
    round_trip_cost_fraction: float
    round_trip_commission_bps: float
    spread_bps: float
    slippage_bps: float
    safety_margin_bps: float
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
    dollar_volumes: tuple[float, ...]
    trade_counts: tuple[float, ...]
    vwaps: tuple[float | None, ...]
    taker_buy_base_volumes: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class _PreparedMarketQuotes:
    timestamps: tuple[datetime, ...]
    market_timestamps: tuple[datetime, ...]
    bids: tuple[float, ...]
    asks: tuple[float, ...]
    mids: tuple[float, ...]
    spread_bps: tuple[float, ...]
    bid_sizes: tuple[float | None, ...]
    ask_sizes: tuple[float | None, ...]


def _prepare_market_quotes(quotes: Sequence[MarketQuote] | None) -> _PreparedMarketQuotes | None:
    if not quotes:
        return None
    ordered = sorted(quotes, key=lambda quote: quote.received_timestamp_utc)
    return _PreparedMarketQuotes(
        timestamps=tuple(quote.received_timestamp_utc for quote in ordered),
        market_timestamps=tuple(quote.timestamp_utc for quote in ordered),
        bids=tuple(quote.bid for quote in ordered),
        asks=tuple(quote.ask for quote in ordered),
        mids=tuple(quote.midpoint for quote in ordered),
        spread_bps=tuple(quote.spread_bps for quote in ordered),
        bid_sizes=tuple(quote.bid_size for quote in ordered),
        ask_sizes=tuple(quote.ask_size for quote in ordered),
    )


def _quote_index_at_or_before(quotes: _PreparedMarketQuotes, timestamp: datetime) -> int:
    return bisect_right(quotes.timestamps, ensure_utc(timestamp)) - 1


def _quote_index_before_lookback(
    quotes: _PreparedMarketQuotes,
    timestamp: datetime,
    lookback: timedelta,
) -> int:
    return bisect_right(quotes.timestamps, ensure_utc(timestamp) - lookback) - 1


def _quote_feature_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _add_market_quote_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    market_quotes: _PreparedMarketQuotes | None,
    prefix: str = "ibkr",
) -> None:
    if market_quotes is None:
        return
    idx = _quote_index_at_or_before(market_quotes, event.timestamp_utc)
    features[_quote_feature_name(prefix, "quote_available")] = 1.0 if idx >= 0 else 0.0
    if idx < 0:
        return

    quote_time = market_quotes.timestamps[idx]
    market_time = market_quotes.market_timestamps[idx]
    age_seconds = max(0.0, (ensure_utc(event.timestamp_utc) - quote_time).total_seconds())
    market_age_seconds = max(0.0, (ensure_utc(event.timestamp_utc) - market_time).total_seconds())
    bid = market_quotes.bids[idx]
    ask = market_quotes.asks[idx]
    mid = market_quotes.mids[idx]
    spread = market_quotes.spread_bps[idx]
    bar_close = float(features.get("bar_close", 0.0) or 0.0)
    features[_quote_feature_name(prefix, "quote_age_seconds")] = age_seconds
    features[_quote_feature_name(prefix, "market_quote_age_seconds")] = market_age_seconds
    features[_quote_feature_name(prefix, "quote_stale_5s")] = 1.0 if age_seconds > 5 else 0.0
    features[_quote_feature_name(prefix, "quote_stale_30s")] = 1.0 if age_seconds > 30 else 0.0
    features[_quote_feature_name(prefix, "spread_bps")] = spread
    features[_quote_feature_name(prefix, "log_spread_bps")] = log1p(max(0.0, spread))
    if bar_close > 0:
        features[_quote_feature_name(prefix, "mid_to_bar_close_bps")] = 10_000 * (
            mid / bar_close - 1
        )
        features[_quote_feature_name(prefix, "bid_to_bar_close_bps")] = 10_000 * (
            bid / bar_close - 1
        )
        features[_quote_feature_name(prefix, "ask_to_bar_close_bps")] = 10_000 * (
            ask / bar_close - 1
        )

    bid_size = market_quotes.bid_sizes[idx]
    ask_size = market_quotes.ask_sizes[idx]
    if bid_size is not None or ask_size is not None:
        bid_size_value = float(bid_size or 0.0)
        ask_size_value = float(ask_size or 0.0)
        total_size = bid_size_value + ask_size_value
        features[_quote_feature_name(prefix, "quote_size_available")] = 1.0
        features[_quote_feature_name(prefix, "bid_size_log")] = log1p(bid_size_value)
        features[_quote_feature_name(prefix, "ask_size_log")] = log1p(ask_size_value)
        features[_quote_feature_name(prefix, "top_of_book_size_log")] = log1p(total_size)
        features[_quote_feature_name(prefix, "top_of_book_imbalance")] = (
            (bid_size_value - ask_size_value) / total_size if total_size > 0 else 0.0
        )
        if total_size > 0 and mid > 0:
            microprice = (ask * bid_size_value + bid * ask_size_value) / total_size
            microprice_edge_bps = 10_000 * (microprice / mid - 1)
            features[_quote_feature_name(prefix, "microprice")] = microprice
            features[_quote_feature_name(prefix, "microprice_edge_bps")] = microprice_edge_bps
            features[_quote_feature_name(prefix, "side_microprice_edge_bps")] = (
                microprice_edge_bps if event.side.value == "BUY" else -microprice_edge_bps
            )
            features[_quote_feature_name(prefix, "microprice_to_spread_ratio")] = (
                (microprice - mid) / (ask - bid) if ask > bid else 0.0
            )
    else:
        features[_quote_feature_name(prefix, "quote_size_available")] = 0.0

    for label, lookback in {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
    }.items():
        previous_idx = _quote_index_before_lookback(market_quotes, event.timestamp_utc, lookback)
        if previous_idx < 0 or previous_idx >= idx:
            continue
        previous_mid = market_quotes.mids[previous_idx]
        if previous_mid > 0:
            features[_quote_feature_name(prefix, f"mid_return_{label}")] = mid / previous_mid - 1
        features[_quote_feature_name(prefix, f"spread_change_bps_{label}")] = (
            spread - market_quotes.spread_bps[previous_idx]
        )


def _bar_dollar_volume(bar: Bar) -> float:
    if bar.quote_volume is not None:
        return max(0.0, float(bar.quote_volume))
    price = bar.vwap or bar.close
    return max(0.0, float(price) * float(bar.volume))


def _bar_taker_buy_base_volume(bar: Bar) -> float | None:
    raw = (bar.extra or {}).get("taker_buy_base_volume", (bar.extra or {}).get("taker_buy_volume"))
    return float(raw) if isinstance(raw, int | float) else None


def _prepared_context_from_bars(bars: Sequence[Bar]) -> _PreparedContext:
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    return _PreparedContext(
        timestamps=tuple(bar.timestamp_utc for bar in ordered),
        closes=tuple(bar.close for bar in ordered),
        volumes=tuple(bar.volume for bar in ordered),
        dollar_volumes=tuple(_bar_dollar_volume(bar) for bar in ordered),
        trade_counts=tuple(float(bar.trade_count or 0) for bar in ordered),
        vwaps=tuple(bar.vwap for bar in ordered),
        taker_buy_base_volumes=tuple(_bar_taker_buy_base_volume(bar) for bar in ordered),
    )


def _prepare_context_bars(context_bars: Mapping[str, list[Bar]] | None) -> dict[str, _PreparedContext]:
    prepared: dict[str, _PreparedContext] = {}
    if not context_bars:
        return prepared
    for raw_name, bars in context_bars.items():
        name = raw_name.lower().replace("/", "").replace("-", "").replace("_", "")
        prepared[name] = _prepared_context_from_bars(bars)
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


def _context_value_ratio_before(
    context: _PreparedContext,
    values: Sequence[float],
    timestamp: datetime,
    periods: int,
) -> float:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx <= 0:
        return 0.0
    start = max(0, idx - periods)
    baseline = [value for value in values[start:idx] if value > 0]
    average = sum(baseline) / len(baseline) if baseline else 0.0
    current = values[idx]
    return current / average if average > 0 and current > 0 else 0.0


def _context_value_ratio_before_duration(
    context: _PreparedContext,
    values: Sequence[float],
    timestamp: datetime,
    lookback: timedelta,
) -> float:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx <= 0:
        return 0.0
    start = max(0, bisect_right(context.timestamps, ensure_utc(timestamp) - lookback) - 1)
    baseline = [value for value in values[start:idx] if value > 0]
    average = sum(baseline) / len(baseline) if baseline else 0.0
    current = values[idx]
    return current / average if average > 0 and current > 0 else 0.0


def _context_taker_buy_share_at(context: _PreparedContext, idx: int) -> float | None:
    if idx < 0 or idx >= len(context.timestamps):
        return None
    taker_buy = context.taker_buy_base_volumes[idx]
    volume = context.volumes[idx]
    if taker_buy is None or volume <= 0:
        return None
    return max(0.0, min(1.0, taker_buy / volume))


def _context_taker_buy_share_mean_before(
    context: _PreparedContext,
    *,
    idx: int,
    start: int,
) -> float | None:
    shares = [
        share
        for position in range(max(0, start), max(0, idx))
        if (share := _context_taker_buy_share_at(context, position)) is not None
    ]
    if not shares:
        return None
    return sum(shares) / len(shares)


def _basis_before(
    primary: _PreparedContext,
    context: _PreparedContext,
    timestamp: datetime,
) -> float | None:
    primary_close = _latest_close_value_before(primary, timestamp)
    context_close = _latest_close_value_before(context, timestamp)
    if primary_close is None or context_close is None or primary_close <= 0:
        return None
    return context_close / primary_close - 1


def _basis_values_since(
    primary: _PreparedContext,
    context: _PreparedContext,
    *,
    timestamp: datetime,
    lookback: timedelta,
) -> list[float]:
    end_time = ensure_utc(timestamp)
    start_time = end_time - lookback
    start_idx = max(0, bisect_right(context.timestamps, start_time) - 1)
    end_idx = bisect_right(context.timestamps, end_time) - 1
    if end_idx < start_idx:
        return []
    values: list[float] = []
    for idx in range(start_idx, end_idx + 1):
        context_close = context.closes[idx]
        primary_close = _latest_close_value_before(primary, context.timestamps[idx])
        if primary_close is not None and primary_close > 0 and context_close > 0:
            values.append(context_close / primary_close - 1)
    return values


def _is_btc_basis_context(name: str) -> bool:
    normalized = name.lower().replace("/", "").replace("-", "").replace("_", "")
    if normalized in {"ethbtc", "ethusdt", "solusdt"}:
        return False
    if any(token in normalized for token in ("funding", "openinterest", "longshort", "takerflow", "basis")):
        return False
    return any(token in normalized for token in ("btcusd", "btcusdt", "xbtusd", "mbt", "btcspot", "ibkrspot"))


def _context_pair_asset(name: str) -> str | None:
    normalized = name.lower().replace("/", "").replace("-", "").replace("_", "")
    if any(token in normalized for token in ("funding", "openinterest", "longshort", "takerflow", "basis")):
        return None
    if "ethbtc" in normalized:
        return None
    if "met" in normalized or "eth" in normalized:
        return "eth"
    if "mbt" in normalized or "btc" in normalized or "xbt" in normalized:
        return "btc"
    if "sol" in normalized:
        return "sol"
    return None


def _is_futures_price_context(name: str) -> bool:
    normalized = name.lower().replace("/", "").replace("-", "").replace("_", "")
    if any(token in normalized for token in ("funding", "openinterest", "longshort", "takerflow", "basis")):
        return False
    return any(token in normalized for token in ("futures", "binanceum", "mbt", "met", "perpetual"))


def _is_spot_price_context(name: str) -> bool:
    normalized = name.lower().replace("/", "").replace("-", "").replace("_", "")
    if _is_futures_price_context(normalized):
        return False
    if "ethbtc" in normalized:
        return False
    return any(token in normalized for token in ("spot", "paxos", "zerohash", "coinbase", "usdt", "usd"))


def _add_cross_context_pair_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    context_bars: Mapping[str, _PreparedContext],
) -> None:
    futures_contexts: list[tuple[str, str, _PreparedContext]] = []
    spot_contexts: list[tuple[str, str, _PreparedContext]] = []
    for name, context in context_bars.items():
        asset = _context_pair_asset(name)
        if asset is None:
            continue
        if _is_futures_price_context(name):
            futures_contexts.append((asset, name, context))
        elif _is_spot_price_context(name):
            spot_contexts.append((asset, name, context))

    duration_lookbacks = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "24h": timedelta(hours=24),
    }
    timestamp = ensure_utc(event.timestamp_utc)
    for asset, futures_name, futures_context in futures_contexts:
        for spot_asset, spot_name, spot_context in spot_contexts:
            if spot_asset != asset:
                continue
            futures_idx = bisect_right(futures_context.timestamps, timestamp) - 1
            spot_idx = bisect_right(spot_context.timestamps, timestamp) - 1
            if futures_idx < 0 or spot_idx < 0:
                continue
            futures_close = futures_context.closes[futures_idx]
            spot_close = spot_context.closes[spot_idx]
            if futures_close <= 0 or spot_close <= 0:
                continue
            prefix = f"{asset}_futures_spot_{futures_name}_vs_{spot_name}"
            basis = futures_close / spot_close - 1
            features[f"{prefix}_basis_bps"] = basis * 10_000
            features[f"{prefix}_side_basis_bps"] = (
                basis * 10_000 if event.side.value == "BUY" else -basis * 10_000
            )
            futures_dollar_volume = futures_context.dollar_volumes[futures_idx]
            spot_dollar_volume = spot_context.dollar_volumes[spot_idx]
            if futures_dollar_volume > 0 and spot_dollar_volume > 0:
                features[f"{prefix}_dollar_volume_log_ratio"] = log1p(futures_dollar_volume) - log1p(
                    spot_dollar_volume
                )
                features[f"{prefix}_futures_dollar_volume_share"] = futures_dollar_volume / (
                    futures_dollar_volume + spot_dollar_volume
                )
            futures_trade_count = futures_context.trade_counts[futures_idx]
            spot_trade_count = spot_context.trade_counts[spot_idx]
            if futures_trade_count > 0 and spot_trade_count > 0:
                features[f"{prefix}_trade_count_ratio"] = futures_trade_count / spot_trade_count
            for label, lookback in duration_lookbacks.items():
                futures_return = _latest_close_before_duration(futures_context, timestamp, lookback)
                spot_return = _latest_close_before_duration(spot_context, timestamp, lookback)
                if futures_return is not None and spot_return is not None:
                    return_spread = futures_return - spot_return
                    features[f"{prefix}_return_spread_bps_{label}"] = return_spread * 10_000
                    features[f"{prefix}_side_return_spread_bps_{label}"] = (
                        return_spread * 10_000 if event.side.value == "BUY" else -return_spread * 10_000
                    )
                current_basis = _basis_before(spot_context, futures_context, timestamp)
                prior_basis = _basis_before(spot_context, futures_context, timestamp - lookback)
                if current_basis is not None and prior_basis is not None:
                    basis_change = current_basis - prior_basis
                    features[f"{prefix}_basis_change_bps_{label}"] = basis_change * 10_000
                    features[f"{prefix}_side_basis_change_bps_{label}"] = (
                        basis_change * 10_000 if event.side.value == "BUY" else -basis_change * 10_000
                    )
                futures_volume_ratio = _context_value_ratio_before_duration(
                    futures_context,
                    futures_context.dollar_volumes,
                    timestamp,
                    lookback,
                )
                spot_volume_ratio = _context_value_ratio_before_duration(
                    spot_context,
                    spot_context.dollar_volumes,
                    timestamp,
                    lookback,
                )
                if futures_volume_ratio > 0 and spot_volume_ratio > 0:
                    features[f"{prefix}_relative_dollar_volume_ratio_{label}"] = (
                        futures_volume_ratio / spot_volume_ratio
                    )
            for label, lookback in {"4h": timedelta(hours=4), "24h": timedelta(hours=24)}.items():
                basis_values = _basis_values_since(
                    spot_context,
                    futures_context,
                    timestamp=timestamp,
                    lookback=lookback,
                )
                if len(basis_values) > 1:
                    basis_mean = sum(basis_values) / len(basis_values)
                    basis_std = pstdev(basis_values)
                    features[f"{prefix}_basis_mean_bps_{label}"] = basis_mean * 10_000
                    features[f"{prefix}_basis_stdev_bps_{label}"] = basis_std * 10_000
                    features[f"{prefix}_basis_zscore_{label}"] = (
                        (basis_values[-1] - basis_mean) / basis_std if basis_std > 0 else 0.0
                    )


def _context_metric_value(name: str, close: float) -> float:
    normalized = name.lower().replace("/", "").replace("-", "").replace("_", "")
    if "funding" in normalized:
        return close - 1.0
    return close


def _context_metric_change_before(
    context: _PreparedContext,
    timestamp: datetime,
    periods: int,
    *,
    name: str,
) -> float | None:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx < periods:
        return None
    return _context_metric_value(name, context.closes[idx]) - _context_metric_value(name, context.closes[idx - periods])


def _context_metric_change_before_duration(
    context: _PreparedContext,
    timestamp: datetime,
    lookback: timedelta,
    *,
    name: str,
) -> float | None:
    idx = bisect_right(context.timestamps, ensure_utc(timestamp)) - 1
    if idx < 0:
        return None
    target_idx = bisect_right(context.timestamps, ensure_utc(timestamp) - lookback) - 1
    if target_idx < 0 or target_idx >= idx:
        return None
    return _context_metric_value(name, context.closes[idx]) - _context_metric_value(name, context.closes[target_idx])


def _add_cross_asset_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    context_bars: Mapping[str, _PreparedContext] | None,
    history_bars: Sequence[Bar] | None = None,
) -> None:
    if not context_bars:
        return
    primary_context = None
    if history_bars:
        primary_context = _prepared_context_from_bars(history_bars)
    btc_return_24 = float(features.get("return_elapsed_24h", features.get("return_24", 0.0)) or 0.0)
    btc_vol_24 = float(features.get("realized_vol_elapsed_24h", features.get("realized_vol_24", 0.0)) or 0.0)
    btc_vol_72 = float(features.get("realized_vol_elapsed_72h", features.get("realized_vol_72", 0.0)) or 0.0)
    btc_close = float(features.get("bar_close", 0.0) or 0.0)
    duration_lookbacks = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "12h": timedelta(hours=12),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
        "168h": timedelta(hours=168),
    }
    for name, context in context_bars.items():
        context_idx = bisect_right(context.timestamps, ensure_utc(event.timestamp_utc)) - 1
        if context_idx < 0:
            features[f"{name}_available"] = 0.0
            continue
        context_age = event.timestamp_utc - context.timestamps[context_idx]
        features[f"{name}_available"] = 1.0
        features[f"{name}_age_seconds"] = max(0.0, context_age.total_seconds())
        context_close = context.closes[context_idx]
        metric_value = _context_metric_value(name, context_close)
        context_dollar_volume = context.dollar_volumes[context_idx]
        context_trade_count = context.trade_counts[context_idx]
        is_btc_basis_context = _is_btc_basis_context(name)
        features[f"{name}_value"] = metric_value
        if metric_value > 0:
            features[f"{name}_value_log"] = log1p(metric_value)
        if btc_close > 0 and context_close > 0:
            features[f"{name}_close_to_btc"] = context_close / btc_close - 1
            if is_btc_basis_context:
                features[f"{name}_basis_bps"] = 10_000 * (context_close / btc_close - 1)
        if context_dollar_volume > 0:
            features[f"{name}_dollar_volume_log"] = log1p(context_dollar_volume)
        if context_trade_count > 0:
            features[f"{name}_volume_per_trade"] = context.volumes[context_idx] / context_trade_count
            features[f"{name}_dollar_volume_per_trade"] = context_dollar_volume / context_trade_count
        if context.vwaps[context_idx] is not None and context.vwaps[context_idx] > 0:
            features[f"{name}_vwap_distance_bps"] = 10_000 * (
                context_close / float(context.vwaps[context_idx]) - 1
            )
        taker_buy_share = _context_taker_buy_share_at(context, context_idx)
        if taker_buy_share is not None:
            signed_taker_buy_imbalance = 2 * taker_buy_share - 1
            features[f"{name}_taker_buy_share"] = taker_buy_share
            features[f"{name}_signed_taker_buy_imbalance"] = signed_taker_buy_imbalance
            features[f"{name}_side_taker_buy_imbalance"] = (
                signed_taker_buy_imbalance
                if event.side.value == "BUY"
                else -signed_taker_buy_imbalance
            )
        for periods in (1, 4, 24, 72):
            value = _latest_close_before(context, event.timestamp_utc, periods)
            if value is not None:
                features[f"{name}_return_{periods}"] = value
                btc_value = features.get(f"return_{periods}")
                if isinstance(btc_value, int | float):
                    spread = value - float(btc_value)
                    features[f"{name}_return_spread_{periods}"] = spread
                    features[f"{name}_return_spread_bps_{periods}"] = spread * 10_000
                    features[f"{name}_side_return_spread_bps_{periods}"] = (
                        spread * 10_000 if event.side.value == "BUY" else -spread * 10_000
                    )
            metric_change = _context_metric_change_before(
                context,
                event.timestamp_utc,
                periods,
                name=name,
            )
            if metric_change is not None:
                features[f"{name}_value_change_{periods}"] = metric_change
                features[f"{name}_side_value_change_{periods}"] = (
                    metric_change if event.side.value == "BUY" else -metric_change
                )
            features[f"{name}_dollar_volume_ratio_{periods}"] = _context_value_ratio_before(
                context,
                context.dollar_volumes,
                event.timestamp_utc,
                periods,
            )
            features[f"{name}_trade_count_ratio_{periods}"] = _context_value_ratio_before(
                context,
                context.trade_counts,
                event.timestamp_utc,
                periods,
            )
            taker_mean = _context_taker_buy_share_mean_before(
                context,
                idx=context_idx,
                start=context_idx - periods,
            )
            if taker_buy_share is not None and taker_mean is not None:
                taker_delta = taker_buy_share - taker_mean
                features[f"{name}_taker_buy_share_delta_{periods}"] = taker_delta
                features[f"{name}_side_taker_buy_share_delta_{periods}"] = (
                    taker_delta if event.side.value == "BUY" else -taker_delta
                )
        for label, lookback in duration_lookbacks.items():
            value = _latest_close_before_duration(context, event.timestamp_utc, lookback)
            if value is not None:
                features[f"{name}_return_elapsed_{label}"] = value
                btc_value = features.get(f"return_elapsed_{label}")
                if isinstance(btc_value, int | float):
                    spread = value - float(btc_value)
                    features[f"{name}_return_spread_{label}"] = spread
                    features[f"{name}_return_spread_bps_{label}"] = spread * 10_000
                    features[f"{name}_side_return_spread_bps_{label}"] = (
                        spread * 10_000 if event.side.value == "BUY" else -spread * 10_000
                    )
            metric_change = _context_metric_change_before_duration(
                context,
                event.timestamp_utc,
                lookback,
                name=name,
            )
            if metric_change is not None:
                features[f"{name}_value_change_{label}"] = metric_change
                features[f"{name}_side_value_change_{label}"] = (
                    metric_change if event.side.value == "BUY" else -metric_change
                )
            features[f"{name}_dollar_volume_ratio_{label}"] = _context_value_ratio_before_duration(
                context,
                context.dollar_volumes,
                event.timestamp_utc,
                lookback,
            )
            features[f"{name}_trade_count_ratio_{label}"] = _context_value_ratio_before_duration(
                context,
                context.trade_counts,
                event.timestamp_utc,
                lookback,
            )
            duration_start_idx = bisect_right(
                context.timestamps,
                ensure_utc(event.timestamp_utc) - lookback,
            ) - 1
            taker_mean = _context_taker_buy_share_mean_before(
                context,
                idx=context_idx,
                start=duration_start_idx,
            )
            if taker_buy_share is not None and taker_mean is not None:
                taker_delta = taker_buy_share - taker_mean
                features[f"{name}_taker_buy_share_delta_{label}"] = taker_delta
                features[f"{name}_side_taker_buy_share_delta_{label}"] = (
                    taker_delta if event.side.value == "BUY" else -taker_delta
                )
            if primary_context is not None and is_btc_basis_context:
                current_basis = _basis_before(primary_context, context, event.timestamp_utc)
                prior_basis = _basis_before(primary_context, context, event.timestamp_utc - lookback)
                if current_basis is not None and prior_basis is not None:
                    basis_change_bps = (current_basis - prior_basis) * 10_000
                    features[f"{name}_basis_change_bps_{label}"] = basis_change_bps
                    features[f"{name}_side_basis_change_bps_{label}"] = (
                        basis_change_bps if event.side.value == "BUY" else -basis_change_bps
                    )
        if primary_context is not None and is_btc_basis_context:
            for label, lookback in {"4h": timedelta(hours=4), "24h": timedelta(hours=24)}.items():
                basis_values = _basis_values_since(
                    primary_context,
                    context,
                    timestamp=event.timestamp_utc,
                    lookback=lookback,
                )
                if len(basis_values) > 1:
                    basis_mean = sum(basis_values) / len(basis_values)
                    basis_std = pstdev(basis_values)
                    current_basis = basis_values[-1]
                    features[f"{name}_basis_mean_bps_{label}"] = basis_mean * 10_000
                    features[f"{name}_basis_stdev_bps_{label}"] = basis_std * 10_000
                    features[f"{name}_basis_zscore_{label}"] = (
                        (current_basis - basis_mean) / basis_std if basis_std > 0 else 0.0
                    )
        alt_return_24 = features.get(f"{name}_return_24")
        alt_return_24 = features.get(f"{name}_return_elapsed_24h", alt_return_24)
        if isinstance(alt_return_24, float):
            features[f"btc_relative_strength_vs_{name}_24"] = btc_return_24 - alt_return_24
            features[f"{name}_return_spread_24"] = alt_return_24 - btc_return_24
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
    _add_cross_context_pair_features(features, event=event, context_bars=context_bars)


def _add_event_metadata(features: dict[str, float | str], event: CandidateEvent) -> None:
    for key, value in event.metadata.items():
        if isinstance(value, bool):
            features[f"event_{key}"] = float(value)
        elif isinstance(value, int | float):
            features[f"event_{key}"] = float(value)
        elif isinstance(value, str):
            features[f"event_{key}"] = value


def _event_min_holding_seconds(event: CandidateEvent) -> float:
    try:
        value = float(event.metadata.get("min_holding_seconds", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) and value > 0 else 0.0


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
    feature_profile: str = "full",
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
        ("down_mid", snapshot.down_mid),
        ("last_price", snapshot.last_price),
        ("volume", snapshot.volume),
        ("volume_24h", snapshot.volume_24h),
        ("liquidity", snapshot.liquidity),
        ("open_interest", snapshot.open_interest),
    ):
        _set_float_feature(features, f"{prefix}_{name}", value)
    if feature_profile == "full":
        for name, value in (
            ("down_ask", snapshot.down_ask),
            ("up_bid_size", snapshot.up_bid_size),
            ("up_ask_size", snapshot.up_ask_size),
            ("down_bid_size", snapshot.down_bid_size),
            ("down_ask_size", snapshot.down_ask_size),
            ("up_bid_depth_1c", snapshot.up_bid_depth_1c),
            ("up_bid_depth_3c", snapshot.up_bid_depth_3c),
            ("up_bid_depth_5c", snapshot.up_bid_depth_5c),
            ("up_ask_depth_1c", snapshot.up_ask_depth_1c),
            ("up_ask_depth_3c", snapshot.up_ask_depth_3c),
            ("up_ask_depth_5c", snapshot.up_ask_depth_5c),
            ("orderbook_imbalance_1c", snapshot.orderbook_imbalance_1c),
            ("orderbook_imbalance_3c", snapshot.orderbook_imbalance_3c),
            ("orderbook_imbalance_5c", snapshot.orderbook_imbalance_5c),
            ("last_trade_size", snapshot.last_trade_size),
        ):
            _set_float_feature(features, f"{prefix}_{name}", value)
        for name, value in (
            ("volume", snapshot.volume),
            ("volume_24h", snapshot.volume_24h),
            ("liquidity", snapshot.liquidity),
            ("open_interest", snapshot.open_interest),
        ):
            if value is not None:
                features[f"{prefix}_{name}_log"] = log1p(max(0.0, float(value)))
        if snapshot.liquidity is not None and snapshot.volume_24h is not None:
            features[f"{prefix}_volume_24h_to_liquidity"] = (
                snapshot.volume_24h / snapshot.liquidity if snapshot.liquidity > 0 else 0.0
            )
    if snapshot.up_bid is not None and snapshot.up_ask is not None:
        features[f"{prefix}_up_spread"] = snapshot.up_ask - snapshot.up_bid
    if feature_profile == "full" and snapshot.down_bid is not None and snapshot.down_ask is not None:
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
    if feature_profile != "full":
        return
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
    if snapshot.up_bid_depth_5c is not None and snapshot.up_ask_depth_5c is not None:
        total_depth = snapshot.up_bid_depth_5c + snapshot.up_ask_depth_5c
        features[f"{prefix}_book_depth_5c_total_log"] = log1p(max(total_depth, 0.0))
        features[f"{prefix}_side_book_imbalance_5c"] = (
            snapshot.orderbook_imbalance_5c
            if event.side.value == "BUY"
            else -(snapshot.orderbook_imbalance_5c or 0.0)
        )
    if snapshot.last_trade_side:
        features[f"{prefix}_last_trade_buy_side"] = (
            1.0 if snapshot.last_trade_side.upper() in {"BUY", "YES"} else -1.0
        )


def _add_prediction_market_change_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    snapshot: PredictionMarketSnapshot,
    prior: PredictionMarketSnapshot | None,
    feature_profile: str = "full",
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
        "last_price",
        "liquidity",
        "open_interest",
    ):
        current = getattr(snapshot, name)
        previous = getattr(prior, name)
        if current is not None and previous is not None:
            change = current - previous
            features[f"{prefix}_{name}_change"] = change
            if feature_profile == "full":
                features[f"{prefix}_{name}_change_per_minute"] = change / (gap_seconds / 60)
    if feature_profile == "full":
        current = snapshot.down_ask
        previous = prior.down_ask
        if current is not None and previous is not None:
            change = current - previous
            features[f"{prefix}_down_ask_change"] = change
            features[f"{prefix}_down_ask_change_per_minute"] = change / (gap_seconds / 60)
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
        if feature_profile == "full":
            features[f"{prefix}_direction_skew_change_per_minute"] = skew_change / (gap_seconds / 60)
            features[f"{prefix}_side_aligned_skew_change_per_minute"] = (
                features[f"{prefix}_side_aligned_skew_change"] / (gap_seconds / 60)
            )
    current_side_mid = snapshot.up_mid if event.side.value == "BUY" else snapshot.down_mid
    prior_side_mid = prior.up_mid if event.side.value == "BUY" else prior.down_mid
    if current_side_mid is not None and prior_side_mid is not None:
        side_mid_change = current_side_mid - prior_side_mid
        features[f"{prefix}_side_aligned_mid_change"] = side_mid_change
        if feature_profile == "full":
            features[f"{prefix}_side_aligned_mid_change_per_minute"] = side_mid_change / (gap_seconds / 60)


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


def _prediction_market_liquidity_weight(snapshot: PredictionMarketSnapshot) -> float:
    return log1p(
        max(
            0.0,
            float(snapshot.volume_24h or 0.0),
            float(snapshot.volume or 0.0),
            float(snapshot.liquidity or 0.0),
            float(snapshot.open_interest or 0.0),
        )
    )


def _add_prediction_market_lead_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    snapshot: PredictionMarketSnapshot,
    history: _PreparedContext | None,
    lead_floor_seconds: float,
    emit_per_snapshot: bool = True,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    prefix = f"pm_{snapshot.provider}_{snapshot.duration}"
    event_time = ensure_utc(event.timestamp_utc)
    seconds_to_close: float | None = None
    progress: float | None = None
    if snapshot.window_start_utc is not None and snapshot.window_end_utc is not None:
        total_seconds = (snapshot.window_end_utc - snapshot.window_start_utc).total_seconds()
        seconds_to_close = (snapshot.window_end_utc - event_time).total_seconds()
        elapsed_seconds = (event_time - snapshot.window_start_utc).total_seconds()
        progress = min(max(elapsed_seconds / total_seconds, 0.0), 1.0) if total_seconds > 0 else 0.0
        if emit_per_snapshot:
            features[f"{prefix}_lead_seconds"] = max(seconds_to_close, 0.0)
            features[f"{prefix}_remaining_fraction"] = (
                max(seconds_to_close, 0.0) / total_seconds if total_seconds > 0 else 0.0
            )
            features[f"{prefix}_has_usable_lead"] = 1.0 if seconds_to_close >= lead_floor_seconds else 0.0
            event_horizon = event.max_holding_period.total_seconds()
            features[f"{prefix}_lead_to_event_horizon"] = (
                max(seconds_to_close, 0.0) / event_horizon if event_horizon > 0 else 0.0
            )

    side_mid = snapshot.up_mid if event.side.value == "BUY" else snapshot.down_mid
    side_skew: float | None = None
    if snapshot.up_mid is not None and snapshot.down_mid is not None:
        direction_skew = snapshot.up_mid - snapshot.down_mid
        side_skew = direction_skew if event.side.value == "BUY" else -direction_skew

    side_elapsed_return: float | None = None
    residual_edge: float | None = None
    if history is not None and snapshot.window_start_utc is not None:
        event_close = _latest_close_value_before(history, event_time)
        window_start_close = _latest_close_value_before(history, snapshot.window_start_utc)
        if event_close is not None and window_start_close is not None and window_start_close > 0:
            elapsed_return = event_close / window_start_close - 1
            side_elapsed_return = elapsed_return if event.side.value == "BUY" else -elapsed_return
            if emit_per_snapshot:
                features[f"{prefix}_contract_elapsed_return"] = elapsed_return
                features[f"{prefix}_side_contract_elapsed_return"] = side_elapsed_return
                features[f"{prefix}_side_contract_elapsed_bps"] = side_elapsed_return * 10_000
            if side_mid is not None:
                elapsed_direction_mid = 1.0 if side_elapsed_return > 0 else 0.0 if side_elapsed_return < 0 else 0.5
                residual_edge = side_mid - elapsed_direction_mid
                if emit_per_snapshot:
                    features[f"{prefix}_side_mid_minus_elapsed_direction"] = residual_edge
                    features[f"{prefix}_contrarian_side_mid"] = side_mid if side_elapsed_return < 0 else 0.0
                    features[f"{prefix}_continuation_side_mid"] = side_mid if side_elapsed_return > 0 else 0.0
                if emit_per_snapshot and seconds_to_close is not None and seconds_to_close > 0:
                    features[f"{prefix}_residual_edge_per_remaining_hour"] = (
                        residual_edge / (seconds_to_close / 3600)
                    )

    if emit_per_snapshot and side_mid is not None:
        features[f"{prefix}_side_mid_edge"] = side_mid - 0.5
        if side_skew is not None:
            features[f"{prefix}_side_skew_edge"] = side_skew
    return side_mid, side_skew, seconds_to_close, progress, residual_edge


def _add_prediction_market_features(
    features: dict[str, float | str],
    *,
    event: CandidateEvent,
    prediction_markets: PreparedPredictionMarketSnapshots | None,
    history_bars: Sequence[Bar] | None = None,
    feature_profile: str = "full",
) -> None:
    if prediction_markets is None:
        return
    if feature_profile not in PREDICTION_MARKET_FEATURE_PROFILES:
        raise ValueError(f"unsupported prediction-market feature profile {feature_profile}")
    latest: dict[tuple[str, str], PredictionMarketSnapshot] = {}
    side_aligned_mids: list[float] = []
    side_aligned_skews: list[float] = []
    side_aligned_mids_by_duration: list[tuple[int, float]] = []
    short_horizon_mids: list[float] = []
    micro_horizon_mids: list[float] = []
    leading_side_aligned_mids: list[float] = []
    leading_side_aligned_skews: list[float] = []
    leading_side_aligned_mids_by_duration: list[tuple[int, float]] = []
    leading_residual_edges: list[float] = []
    leading_liquidity_weights: list[float] = []
    leading_weighted_mid_sum = 0.0
    leading_weighted_skew_sum = 0.0
    leading_weighted_residual_sum = 0.0
    leading_weight_total = 0.0
    seconds_to_close_values: list[float] = []
    progress_values: list[float] = []
    history_context = None
    lead_floor_seconds = 1.0
    if history_bars:
        ordered_history = sorted(history_bars, key=lambda bar: bar.timestamp_utc)
        history_context = _prepared_context_from_bars(ordered_history)
        lead_floor_seconds = max(_bar_interval_seconds(ordered_history), 1.0)
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
        _add_prediction_market_snapshot_features(
            features,
            event=event,
            snapshot=snapshot,
            feature_profile=feature_profile,
        )
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
            feature_profile=feature_profile,
        )
        _add_prediction_market_spot_divergence_features(features, event=event, snapshot=snapshot)
        lead_side_mid, lead_side_skew, seconds_to_close, progress, residual_edge = _add_prediction_market_lead_features(
            features,
            event=event,
            snapshot=snapshot,
            history=history_context,
            lead_floor_seconds=lead_floor_seconds,
            emit_per_snapshot=feature_profile == "full",
        )
        if seconds_to_close is not None:
            seconds_to_close_values.append(seconds_to_close)
            if seconds_to_close >= lead_floor_seconds and lead_side_mid is not None:
                leading_side_aligned_mids.append(lead_side_mid)
                try:
                    lead_duration_seconds = prediction_market_duration_seconds(duration)
                except ValueError:
                    lead_duration_seconds = 0
                if lead_duration_seconds > 0:
                    leading_side_aligned_mids_by_duration.append((lead_duration_seconds, lead_side_mid))
                liquidity_weight = _prediction_market_liquidity_weight(snapshot)
                leading_liquidity_weights.append(liquidity_weight)
                if liquidity_weight > 0:
                    leading_weighted_mid_sum += lead_side_mid * liquidity_weight
                    leading_weight_total += liquidity_weight
                if lead_side_skew is not None:
                    leading_side_aligned_skews.append(lead_side_skew)
                    if liquidity_weight > 0:
                        leading_weighted_skew_sum += lead_side_skew * liquidity_weight
                if residual_edge is not None:
                    leading_residual_edges.append(residual_edge)
                    if liquidity_weight > 0:
                        leading_weighted_residual_sum += residual_edge * liquidity_weight
        if progress is not None:
            progress_values.append(progress)
        side_mid = snapshot.up_mid if event.side.value == "BUY" else snapshot.down_mid
        if side_mid is not None:
            side_aligned_mids.append(side_mid)
            try:
                duration_seconds = prediction_market_duration_seconds(duration)
            except ValueError:
                duration_seconds = 0
            if duration_seconds > 0:
                side_aligned_mids_by_duration.append((duration_seconds, side_mid))
            if duration_seconds <= 3600:
                short_horizon_mids.append(side_mid)
            if duration_seconds <= 900:
                micro_horizon_mids.append(side_mid)
        if snapshot.up_mid is not None and snapshot.down_mid is not None:
            direction_skew = snapshot.up_mid - snapshot.down_mid
            side_aligned_skews.append(direction_skew if event.side.value == "BUY" else -direction_skew)
    if side_aligned_mids:
        features["pm_available_count"] = float(len(side_aligned_mids))
        features["pm_side_aligned_mid_mean"] = sum(side_aligned_mids) / len(side_aligned_mids)
        features["pm_side_aligned_mid_max"] = max(side_aligned_mids)
        features["pm_side_aligned_mid_min"] = min(side_aligned_mids)
    if side_aligned_skews:
        features["pm_side_aligned_skew_mean"] = sum(side_aligned_skews) / len(side_aligned_skews)
        features["pm_side_aligned_skew_max"] = max(side_aligned_skews)
        features["pm_side_aligned_skew_min"] = min(side_aligned_skews)
    if short_horizon_mids:
        features["pm_short_side_aligned_mid_mean"] = sum(short_horizon_mids) / len(short_horizon_mids)
        features["pm_short_side_aligned_mid_max"] = max(short_horizon_mids)
    if micro_horizon_mids:
        features["pm_micro_side_aligned_mid_mean"] = sum(micro_horizon_mids) / len(micro_horizon_mids)
        features["pm_micro_side_aligned_mid_max"] = max(micro_horizon_mids)
    if side_aligned_mids_by_duration:
        micro = [value for seconds, value in side_aligned_mids_by_duration if seconds <= 900]
        intraday = [value for seconds, value in side_aligned_mids_by_duration if 900 < seconds <= 14_400]
        long = [value for seconds, value in side_aligned_mids_by_duration if seconds >= 3600]
        if micro and long:
            features["pm_term_micro_minus_long_mid"] = (
                sum(micro) / len(micro) - sum(long) / len(long)
            )
        if micro and intraday:
            features["pm_term_micro_minus_intraday_mid"] = (
                sum(micro) / len(micro) - sum(intraday) / len(intraday)
            )
        if len(side_aligned_mids_by_duration) > 1:
            duration_mean = sum(seconds for seconds, _ in side_aligned_mids_by_duration) / len(
                side_aligned_mids_by_duration
            )
            mid_mean = sum(value for _, value in side_aligned_mids_by_duration) / len(
                side_aligned_mids_by_duration
            )
            denominator = sum((seconds - duration_mean) ** 2 for seconds, _ in side_aligned_mids_by_duration)
            if denominator > 0:
                features["pm_term_mid_slope_per_hour"] = (
                    sum(
                        (seconds - duration_mean) * (value - mid_mean)
                        for seconds, value in side_aligned_mids_by_duration
                    )
                    / denominator
                    * 3600
                )
    if leading_side_aligned_mids:
        features["pm_leading_available_count"] = float(len(leading_side_aligned_mids))
        features["pm_leading_side_aligned_mid_mean"] = (
            sum(leading_side_aligned_mids) / len(leading_side_aligned_mids)
        )
        features["pm_leading_side_aligned_mid_max"] = max(leading_side_aligned_mids)
        features["pm_leading_side_aligned_mid_min"] = min(leading_side_aligned_mids)
    if leading_side_aligned_mids_by_duration:
        leading_micro = [
            value for seconds, value in leading_side_aligned_mids_by_duration if seconds <= 900
        ]
        leading_long = [
            value for seconds, value in leading_side_aligned_mids_by_duration if seconds >= 3600
        ]
        if leading_micro and leading_long:
            features["pm_leading_term_micro_minus_long_mid"] = (
                sum(leading_micro) / len(leading_micro) - sum(leading_long) / len(leading_long)
            )
    if leading_weight_total > 0:
        features["pm_leading_liquidity_weight_total"] = leading_weight_total
        features["pm_leading_side_aligned_mid_liquidity_weighted"] = (
            leading_weighted_mid_sum / leading_weight_total
        )
    if leading_side_aligned_skews:
        features["pm_leading_side_aligned_skew_mean"] = (
            sum(leading_side_aligned_skews) / len(leading_side_aligned_skews)
        )
        features["pm_leading_side_aligned_skew_max"] = max(leading_side_aligned_skews)
        features["pm_leading_side_aligned_skew_min"] = min(leading_side_aligned_skews)
        if leading_weight_total > 0:
            features["pm_leading_side_aligned_skew_liquidity_weighted"] = (
                leading_weighted_skew_sum / leading_weight_total
            )
    if leading_residual_edges:
        features["pm_leading_residual_edge_mean"] = sum(leading_residual_edges) / len(leading_residual_edges)
        features["pm_leading_residual_edge_max"] = max(leading_residual_edges)
        features["pm_leading_residual_edge_min"] = min(leading_residual_edges)
        if leading_weight_total > 0:
            features["pm_leading_residual_edge_liquidity_weighted"] = (
                leading_weighted_residual_sum / leading_weight_total
            )
    if leading_liquidity_weights:
        features["pm_leading_liquidity_weight_mean"] = sum(leading_liquidity_weights) / len(
            leading_liquidity_weights
        )
    if seconds_to_close_values:
        features["pm_seconds_to_close_min"] = min(seconds_to_close_values)
        features["pm_seconds_to_close_max"] = max(seconds_to_close_values)
        features["pm_seconds_to_close_mean"] = sum(seconds_to_close_values) / len(seconds_to_close_values)
    if progress_values:
        features["pm_contract_progress_mean"] = sum(progress_values) / len(progress_values)
    for duration in sorted({duration for _, duration in latest}):
        poly = latest.get(("polymarket", duration))
        kalshi = latest.get(("kalshi", duration))
        if poly is None or kalshi is None:
            continue
        prefix = f"pm_cross_polymarket_kalshi_{duration}"
        poly_side_mid = poly.up_mid if event.side.value == "BUY" else poly.down_mid
        kalshi_side_mid = kalshi.up_mid if event.side.value == "BUY" else kalshi.down_mid
        if poly_side_mid is not None and kalshi_side_mid is not None:
            side_mid_diff = poly_side_mid - kalshi_side_mid
            features[f"{prefix}_side_mid_diff"] = side_mid_diff
            features[f"{prefix}_side_mid_abs_diff"] = abs(side_mid_diff)
            features[f"{prefix}_side_mid_mean"] = (poly_side_mid + kalshi_side_mid) / 2
            features[f"{prefix}_side_mid_max"] = max(poly_side_mid, kalshi_side_mid)
            features[f"{prefix}_side_mid_min"] = min(poly_side_mid, kalshi_side_mid)
            features[f"{prefix}_side_mid_range"] = abs(side_mid_diff)
            features[f"{prefix}_side_mid_agreement"] = (
                1.0
                if poly_side_mid >= 0.5 and kalshi_side_mid >= 0.5
                else -1.0
                if poly_side_mid < 0.5 and kalshi_side_mid < 0.5
                else 0.0
            )
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


def _bar_side_extreme_return(
    side: Side,
    *,
    entry_price: float,
    bar: Bar,
) -> tuple[float, float]:
    if entry_price <= 0:
        return 0.0, 0.0
    if side == Side.SELL:
        favorable = (entry_price - bar.low) / entry_price
        adverse = (entry_price - bar.high) / entry_price
    else:
        favorable = bar.high / entry_price - 1
        adverse = bar.low / entry_price - 1
    return favorable, adverse


def _label_path_targets(
    *,
    event: CandidateEvent,
    label: TripleBarrierLabel,
    future_bars: Sequence[Bar],
    round_trip_cost_bps: float,
) -> tuple[float, float, float, int]:
    entry_price = label.entry_price
    if entry_price <= 0:
        return 0.0, 0.0, 0.0, 0
    adverse = 0.0
    favorable = 0.0
    early_adverse = 0
    entry_timestamp = ensure_utc(label.entry_timestamp_utc)
    exit_timestamp = ensure_utc(label.exit_timestamp_utc)
    early_cutoff = min(exit_timestamp, entry_timestamp + timedelta(seconds=60))
    early_threshold = max(round_trip_cost_bps / 10_000, 0.0005)
    for bar in sorted(future_bars, key=lambda row: row.timestamp_utc):
        bar_time = bar_start_timestamp_utc(bar)
        if bar_time < entry_timestamp:
            continue
        if bar_time > exit_timestamp:
            break
        bar_favorable, bar_adverse = _bar_side_extreme_return(
            event.side,
            entry_price=entry_price,
            bar=bar,
        )
        favorable = max(favorable, bar_favorable)
        adverse = max(adverse, -bar_adverse)
        if bar_time <= early_cutoff and -bar_adverse >= early_threshold:
            early_adverse = 1
    time_to_exit_seconds = max(0.0, (exit_timestamp - entry_timestamp).total_seconds())
    return adverse, favorable, time_to_exit_seconds, early_adverse


def _safe_ratio(current: float, baseline: float) -> float:
    return current / baseline if baseline > 0 else 0.0


def _bar_extra_float(bar: Bar, *keys: str) -> float | None:
    for key in keys:
        value = bar.extra.get(key)
        if isinstance(value, int | float) and isfinite(float(value)):
            return float(value)
    return None


def _bar_tick_count(bar: Bar) -> float:
    if bar.trade_count is not None:
        return max(0.0, float(bar.trade_count))
    value = _bar_extra_float(bar, "tick_count", "trade_count")
    return max(0.0, value or 0.0)


def _bar_quote_age_seconds(bar: Bar) -> float | None:
    return _bar_extra_float(bar, "quote_age_seconds", "market_quote_age_seconds")


def _bar_spread_proxy_bps(bar: Bar) -> float:
    spread = _bar_extra_float(bar, "spread_bps")
    if spread is not None and spread > 0:
        return spread
    if bar.close <= 0:
        return 0.0
    return 10_000 * max(bar.high - bar.low, 0.0) / bar.close


def _signed_bar_flow(bar: Bar, *, use_volume: bool = False) -> float:
    direction = 1.0 if bar.close > bar.open else -1.0 if bar.close < bar.open else 0.0
    weight = max(0.0, float(bar.volume)) if use_volume else _bar_tick_count(bar)
    if weight <= 0 and not use_volume:
        weight = 1.0 if direction else 0.0
    return direction * weight


def _return_autocorr(returns: list[float]) -> float:
    if len(returns) < 3:
        return 0.0
    left = returns[:-1]
    right = returns[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    denominator = sqrt(left_var * right_var)
    if denominator <= 0:
        return 0.0
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)) / denominator


def _add_primary_microstructure_features(
    features: dict[str, float | str],
    *,
    history: Sequence[Bar],
) -> None:
    ordered = sorted(history, key=lambda bar: bar.timestamp_utc)
    if len(ordered) < 2:
        return
    interval_seconds = max(_bar_interval_seconds(list(ordered)), 1.0)
    latest = ordered[-1]
    tick_count = _bar_tick_count(latest)
    spread_proxy = _bar_spread_proxy_bps(latest)
    features["micro_bar_interval_seconds"] = interval_seconds
    features["micro_is_one_second_bar"] = 1.0 if interval_seconds <= 1.5 else 0.0
    features["micro_is_five_second_or_faster_bar"] = 1.0 if interval_seconds <= 5.5 else 0.0
    features["micro_tick_count"] = tick_count
    features["micro_tick_count_log"] = log1p(tick_count)
    features["micro_spread_proxy_bps"] = spread_proxy
    features["micro_bar_range_proxy_bps"] = _bar_spread_proxy_bps(latest)
    quote_age = _bar_quote_age_seconds(latest)
    if quote_age is not None:
        features["micro_quote_age_seconds"] = quote_age
        features["micro_quote_stale_5s"] = 1.0 if quote_age > 5 else 0.0

    returns = [
        ordered[idx].close / ordered[idx - 1].close - 1
        for idx in range(1, len(ordered))
        if ordered[idx - 1].close > 0
    ]
    for seconds in (5, 30, 60, 300):
        bars_needed = max(1, int(round(seconds / interval_seconds)))
        window = ordered[-bars_needed:]
        if not window:
            continue
        tick_sum = sum(_bar_tick_count(bar) for bar in window)
        volume_sum = sum(max(0.0, float(bar.volume)) for bar in window)
        signed_tick = sum(_signed_bar_flow(bar) for bar in window)
        signed_volume = sum(_signed_bar_flow(bar, use_volume=True) for bar in window)
        range_values = [_bar_spread_proxy_bps(bar) for bar in window]
        features[f"micro_tick_intensity_{seconds}s"] = tick_sum / seconds
        features[f"micro_volume_intensity_{seconds}s"] = volume_sum / seconds
        features[f"micro_signed_tick_imbalance_{seconds}s"] = (
            signed_tick / tick_sum if tick_sum > 0 else 0.0
        )
        features[f"micro_signed_volume_imbalance_{seconds}s"] = (
            signed_volume / volume_sum if volume_sum > 0 else 0.0
        )
        features[f"micro_range_proxy_bps_mean_{seconds}s"] = (
            sum(range_values) / len(range_values) if range_values else 0.0
        )
        if len(window) >= 2 and window[0].close > 0:
            features[f"micro_return_bps_{seconds}s"] = 10_000 * (window[-1].close / window[0].close - 1)
        window_returns = returns[-bars_needed:]
        if window_returns:
            zero_share = sum(1 for value in window_returns if abs(value) < 1e-12) / len(window_returns)
            features[f"micro_zero_return_share_{seconds}s"] = zero_share
            if len(window_returns) > 1:
                features[f"micro_realized_vol_bps_{seconds}s"] = pstdev(window_returns) * 10_000
    vol_10 = features.get("micro_realized_vol_bps_5s", 0.0)
    vol_60 = features.get("micro_realized_vol_bps_60s", 0.0)
    if isinstance(vol_10, int | float) and isinstance(vol_60, int | float):
        features["micro_realized_vol_burst_5s_vs_60s"] = _safe_ratio(float(vol_10), float(vol_60))
    features["micro_return_autocorr_60s"] = _return_autocorr(
        returns[-max(3, int(round(60 / interval_seconds))) :]
    )


def label_geometry_diagnostics(
    *,
    config: AppConfig,
    assumed_spread_bps: float,
    research_notional: float | None = None,
    reference_price: float | None = None,
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
        futures_fee_per_contract=config.cost.futures_fee_per_contract,
        futures_contract_multiplier=config.cost.futures_contract_multiplier,
        reference_price=reference_price,
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
        round_trip_commission_bps=cost.commission_bps,
        spread_bps=cost.spread_bps,
        slippage_bps=cost.slippage_bps,
        safety_margin_bps=cost.safety_margin_bps,
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
    label_bars: Sequence[Bar] | None = None,
    context_bars: Mapping[str, list[Bar]] | None = None,
    market_quotes: Sequence[MarketQuote] | None = None,
    futures_market_quotes: Sequence[MarketQuote] | None = None,
    prediction_market_snapshots: Sequence[PredictionMarketSnapshot] | None = None,
    prediction_market_feature_profile: str = "full",
    feature_asof: str = "signal",
    external_feature_latency_seconds: float = 0.0,
    candidate_config: CandidateGenerationConfig | None = None,
) -> list[MetaLabelSample]:
    if prediction_market_feature_profile not in PREDICTION_MARKET_FEATURE_PROFILES:
        raise ValueError(
            f"prediction_market_feature_profile must be one of {PREDICTION_MARKET_FEATURE_PROFILES}"
        )
    if feature_asof not in FEATURE_ASOF_CHOICES:
        raise ValueError(f"feature_asof must be one of {FEATURE_ASOF_CHOICES}")
    if external_feature_latency_seconds < 0:
        raise ValueError("external_feature_latency_seconds must be nonnegative")
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    if len(ordered) < 300:
        raise ValueError("not enough bars for meta-label dataset")
    bar_interval_seconds = _bar_interval_seconds(ordered)
    label_ordered = sorted(label_bars, key=lambda bar: bar.timestamp_utc) if label_bars else ordered
    label_interval_seconds = _bar_interval_seconds(label_ordered) if len(label_ordered) >= 2 else bar_interval_seconds
    if (
        config.labels.max_holding_seconds is not None
        and config.labels.max_holding_seconds < label_interval_seconds
    ):
        raise ValueError(
            "max_holding_seconds is shorter than the input bar interval or label bar interval; "
            "use tick or sub-minute bars before researching sub-bar exits"
        )
    notional = min(research_notional or config.risk.paper_max_notional, config.risk.paper_max_notional)
    if notional < config.risk.minimum_fee_efficient_notional:
        raise ValueError("research_notional is below fee-efficient minimum")

    diagnostics = label_geometry_diagnostics(
        config=config,
        assumed_spread_bps=assumed_spread_bps,
        research_notional=notional,
        reference_price=ordered[0].close if ordered else None,
    )
    if diagnostics.gross_stop_distance <= 0:
        raise ValueError("label stop geometry must leave positive gross stop distance after round-trip cost")

    events = generate_candidate_events(ordered, config=candidate_config)
    prepared_context = _prepare_context_bars(context_bars)
    prepared_market_quotes = _prepare_market_quotes(market_quotes)
    prepared_futures_market_quotes = _prepare_market_quotes(futures_market_quotes)
    prepared_prediction_markets = (
        PreparedPredictionMarketSnapshots.from_snapshots(prediction_market_snapshots)
        if prediction_market_snapshots
        else None
    )
    label_timestamps = [bar.timestamp_utc for bar in label_ordered]
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
        label_start = bisect_right(label_timestamps, event.timestamp_utc)
        horizon_end = bisect_right(label_timestamps, event.vertical_barrier_timestamp_utc)
        future = label_ordered[label_start:horizon_end]
        if not future:
            continue
        entry_bar = future[0]
        entry_timestamp = bar_start_timestamp_utc(entry_bar)
        if entry_timestamp < event.timestamp_utc:
            entry_timestamp = event.timestamp_utc
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
            entry_timestamp_utc=entry_timestamp,
            net_profit_target=net_profit_target,
            net_stop_loss=net_stop_loss,
            round_trip_cost_bps=diagnostics.round_trip_cost_bps,
            conservative_same_bar=config.labels.conservative_same_bar,
        )
        max_adverse_excursion, max_favorable_excursion, time_to_exit_seconds, early_adverse_label = (
            _label_path_targets(
                event=event,
                label=label,
                future_bars=future,
                round_trip_cost_bps=diagnostics.round_trip_cost_bps,
            )
        )
        features = build_event_features(event, history)
        features.update(classify_market_regime(history).as_features())
        _add_event_metadata(features, event)
        _add_primary_microstructure_features(features, history=history)
        point_in_time_event = (
            replace(event, timestamp_utc=label.entry_timestamp_utc)
            if feature_asof == "entry"
            else event
        )
        external_event = replace(
            point_in_time_event,
            timestamp_utc=point_in_time_event.timestamp_utc
            - timedelta(seconds=external_feature_latency_seconds),
        )
        features["point_signal_delay_seconds"] = max(
            0.0,
            (label.entry_timestamp_utc - event.timestamp_utc).total_seconds(),
        )
        features["external_feature_latency_seconds"] = external_feature_latency_seconds
        features["external_feature_delay_from_entry_seconds"] = max(
            0.0,
            (label.entry_timestamp_utc - external_event.timestamp_utc).total_seconds(),
        )
        features["feature_asof_is_entry"] = 1.0 if feature_asof == "entry" else 0.0
        _add_cross_asset_features(
            features,
            event=event,
            context_bars=prepared_context,
            history_bars=history,
        )
        _add_market_quote_features(
            features,
            event=external_event,
            market_quotes=prepared_market_quotes,
        )
        _add_market_quote_features(
            features,
            event=external_event,
            market_quotes=prepared_futures_market_quotes,
            prefix="ibkr_futures",
        )
        _add_prediction_market_features(
            features,
            event=external_event,
            prediction_markets=prepared_prediction_markets,
            history_bars=history,
            feature_profile=prediction_market_feature_profile,
        )
        features.update(build_kronos_features(history, config=config.kronos))
        features.update(
            {
                "round_trip_cost_bps": diagnostics.round_trip_cost_bps,
                "assumed_spread_bps": assumed_spread_bps,
                "net_profit_target": net_profit_target,
                "net_stop_loss": net_stop_loss,
                "max_holding_hours": float(event.max_holding_hours),
                "max_holding_seconds": event.max_holding_period.total_seconds(),
                "min_holding_seconds": _event_min_holding_seconds(event),
                "gross_profit_move": net_profit_target + diagnostics.round_trip_cost_fraction,
                "gross_stop_distance": net_stop_loss - diagnostics.round_trip_cost_fraction,
                "horizon_volatility": horizon_volatility,
                "label_bar_interval_seconds": label_interval_seconds,
                "feature_bar_interval_seconds": bar_interval_seconds,
            }
        )
        samples.append(
            MetaLabelSample(
                event_id=event.event_id,
                timestamp_utc=point_in_time_event.timestamp_utc,
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
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                time_to_exit_seconds=time_to_exit_seconds,
                early_adverse_label=early_adverse_label,
            )
        )
    return samples


def build_scoring_samples(
    bars: list[Bar],
    *,
    config: AppConfig,
    candidate_config: CandidateGenerationConfig | None = None,
    context_bars: Mapping[str, list[Bar]] | None = None,
    market_quotes: Sequence[MarketQuote] | None = None,
    futures_market_quotes: Sequence[MarketQuote] | None = None,
    prediction_market_snapshots: Sequence[PredictionMarketSnapshot] | None = None,
    prediction_market_feature_profile: str = "full",
    quote: MarketQuote | None = None,
    max_samples: int = 20,
    assumed_spread_bps: float | None = None,
    research_notional: float | None = None,
    feature_asof: str = "signal",
    external_feature_latency_seconds: float = 0.0,
) -> list[ScoringSample]:
    """Build feature-only samples from completed bars for live/paper scoring."""
    if prediction_market_feature_profile not in PREDICTION_MARKET_FEATURE_PROFILES:
        raise ValueError(
            f"prediction_market_feature_profile must be one of {PREDICTION_MARKET_FEATURE_PROFILES}"
        )
    if feature_asof not in FEATURE_ASOF_CHOICES:
        raise ValueError(f"feature_asof must be one of {FEATURE_ASOF_CHOICES}")
    if external_feature_latency_seconds < 0:
        raise ValueError("external_feature_latency_seconds must be nonnegative")
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    if len(ordered) < 2:
        return []
    if max_samples <= 0:
        return []
    bar_interval_seconds = _bar_interval_seconds(ordered)
    event_config = candidate_config or CandidateGenerationConfig()
    validate_candidate_generation_config(event_config)
    first_idx = max(event_config.min_history_bars - 1, 0)
    if len(ordered) <= first_idx:
        return []
    recent_index_budget = max(max_samples * 8, max_samples + 8, 16)
    candidate_indices: list[int] = []
    for idx in range(len(ordered) - 1, first_idx - 1, -1):
        if (idx - first_idx) % event_config.dense_stride_bars:
            continue
        candidate_indices.append(idx)
        if len(candidate_indices) >= recent_index_budget:
            break
    candidate_indices.reverse()
    seen_event_ids: set[str] = set()
    events = []
    for idx in candidate_indices:
        for event in candidates_for_index(ordered, idx, config=event_config):
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            events.append(event)
    by_timestamp = {bar.timestamp_utc: idx for idx, bar in enumerate(ordered)}
    prepared_context = _prepare_context_bars(context_bars)
    prepared_market_quotes = _prepare_market_quotes(market_quotes)
    prepared_futures_market_quotes = _prepare_market_quotes(futures_market_quotes)
    prepared_prediction_markets = (
        PreparedPredictionMarketSnapshots.from_snapshots(prediction_market_snapshots)
        if prediction_market_snapshots
        else None
    )
    feature_window_bars = max(
        event_config.rolling_window_bars,
        10_080,
    )
    diagnostics = None
    effective_spread_bps = (
        assumed_spread_bps
        if assumed_spread_bps is not None
        else (quote.spread_bps if quote is not None else None)
    )
    if effective_spread_bps is not None:
        diagnostics = label_geometry_diagnostics(
            config=config,
            assumed_spread_bps=effective_spread_bps,
            research_notional=research_notional,
            reference_price=ordered[-1].close if ordered else None,
        )
    samples: list[ScoringSample] = []
    for event in events[-max(max_samples * 4, max_samples) :]:
        idx = by_timestamp.get(event.timestamp_utc)
        if idx is None:
            continue
        feature_start = max(0, idx + 1 - feature_window_bars)
        history = ordered[feature_start : idx + 1]
        if len(history) < 2:
            continue
        try:
            features = build_event_features(event, history, quote=quote)
        except ValueError:
            continue
        net_profit_target = config.labels.net_profit_target
        net_stop_loss = config.labels.net_stop_loss
        horizon_volatility = 0.0
        if diagnostics is not None:
            net_profit_target, net_stop_loss, horizon_volatility = _event_label_geometry(
                event=event,
                history_bars=history,
                config=config,
                cost_fraction=diagnostics.round_trip_cost_fraction,
            )
        features.update(classify_market_regime(history).as_features())
        _add_event_metadata(features, event)
        _add_primary_microstructure_features(features, history=history)
        external_event = replace(
            event,
            timestamp_utc=event.timestamp_utc - timedelta(seconds=external_feature_latency_seconds),
        )
        _add_cross_asset_features(
            features,
            event=event,
            context_bars=prepared_context,
            history_bars=history,
        )
        _add_market_quote_features(
            features,
            event=external_event,
            market_quotes=prepared_market_quotes,
        )
        _add_market_quote_features(
            features,
            event=external_event,
            market_quotes=prepared_futures_market_quotes,
            prefix="ibkr_futures",
        )
        _add_prediction_market_features(
            features,
            event=external_event,
            prediction_markets=prepared_prediction_markets,
            history_bars=history,
            feature_profile=prediction_market_feature_profile,
        )
        features.update(build_kronos_features(history, config=config.kronos))
        point_signal_delay_seconds = bar_interval_seconds if feature_asof == "entry" else 0.0
        features.update(
            {
                "max_holding_hours": float(event.max_holding_hours),
                "max_holding_seconds": event.max_holding_period.total_seconds(),
                "min_holding_seconds": _event_min_holding_seconds(event),
                "net_profit_target": net_profit_target,
                "net_stop_loss": net_stop_loss,
                "horizon_volatility": horizon_volatility,
                "point_signal_delay_seconds": point_signal_delay_seconds,
                "external_feature_latency_seconds": external_feature_latency_seconds,
                "external_feature_delay_from_entry_seconds": (
                    external_feature_latency_seconds if feature_asof == "entry" else 0.0
                ),
                "feature_asof_is_entry": 1.0 if feature_asof == "entry" else 0.0,
            }
        )
        if diagnostics is not None:
            features.update(
                {
                    "round_trip_cost_bps": diagnostics.round_trip_cost_bps,
                    "assumed_spread_bps": effective_spread_bps or 0.0,
                    "gross_profit_move": (
                        net_profit_target + diagnostics.round_trip_cost_fraction
                    ),
                    "gross_stop_distance": (
                        net_stop_loss - diagnostics.round_trip_cost_fraction
                    ),
                }
            )
        samples.append(
            ScoringSample(
                event_id=event.event_id,
                timestamp_utc=event.timestamp_utc,
                candidate_type=event.candidate_type,
                side=event.side.value,
                reference_price=event.reference_price,
                features=features,
            )
        )
    return samples[-max_samples:]
