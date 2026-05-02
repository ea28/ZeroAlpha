"""Public prediction-market data helpers for BTC directional contracts."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

from zeroalpha.timeutils import ensure_utc, parse_unix_timestamp


POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_V2_BASE = "https://clob.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "ZeroAlpha prediction-market research"

BTC_PREDICTION_MARKET_DURATIONS: tuple[str, ...] = (
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "24h",
)

POLYMARKET_BTC_UPDOWN_DURATIONS: tuple[str, ...] = ("5m", "15m", "1h", "4h")
KALSHI_BTC_DURATION_SERIES: dict[str, tuple[str, ...]] = {
    "5m": ("KXBTC5M",),
    "15m": ("KXBTC15M",),
    "1h": ("KXBTC", "KXBTCD"),
    "24h": ("KXBTCD",),
}
KALSHI_BTC_SIGNAL_SERIES: tuple[str, ...] = tuple(
    dict.fromkeys(series for values in KALSHI_BTC_DURATION_SERIES.values() for series in values)
)


@dataclass(frozen=True, slots=True)
class PredictionMarketSnapshot:
    provider: str
    duration: str
    timestamp_utc: datetime
    market_id: str
    market_slug: str
    market_title: str
    condition_id: str
    window_start_utc: datetime | None
    window_end_utc: datetime | None
    up_bid: float | None = None
    up_ask: float | None = None
    up_mid: float | None = None
    down_bid: float | None = None
    down_ask: float | None = None
    down_mid: float | None = None
    up_bid_size: float | None = None
    up_ask_size: float | None = None
    down_bid_size: float | None = None
    down_ask_size: float | None = None
    last_price: float | None = None
    volume: float | None = None
    volume_24h: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", ensure_utc(self.timestamp_utc))
        if self.window_start_utc is not None:
            object.__setattr__(self, "window_start_utc", ensure_utc(self.window_start_utc))
        if self.window_end_utc is not None:
            object.__setattr__(self, "window_end_utc", ensure_utc(self.window_end_utc))

    def as_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("timestamp_utc", "window_start_utc", "window_end_utc"):
            value = payload[key]
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "PredictionMarketSnapshot":
        data = dict(payload)
        for key in ("timestamp_utc", "window_start_utc", "window_end_utc"):
            value = data.get(key)
            data[key] = datetime.fromisoformat(value) if isinstance(value, str) and value else None
        if data["timestamp_utc"] is None:
            raise ValueError("prediction-market snapshot requires timestamp_utc")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PredictionMarketLoadResult:
    snapshots: list[PredictionMarketSnapshot]
    coverage: dict[str, Any]


def _request_json(
    url: str,
    *,
    timeout: float = 30.0,
    method: str = "GET",
    payload: Any | None = None,
) -> Any:
    data = None
    headers = {"User-Agent": USER_AGENT}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _url(base: str, path: str, params: dict[str, Any] | None = None) -> str:
    query = urllib.parse.urlencode(
        {key: value for key, value in (params or {}).items() if value is not None}
    )
    return f"{base.rstrip('/')}/{path.lstrip('/')}" + (f"?{query}" if query else "")


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _duration_seconds(duration: str) -> int:
    value = duration.strip().lower()
    if value.endswith("m"):
        return int(value[:-1]) * 60
    if value.endswith("h"):
        return int(value[:-1]) * 3600
    if value.endswith("d"):
        return int(value[:-1]) * 86_400
    raise ValueError(f"unsupported prediction-market duration {duration}")


def prediction_market_duration_seconds(duration: str) -> int:
    return _duration_seconds(duration)


def _duration_minutes(duration: str) -> int:
    seconds = _duration_seconds(duration)
    if seconds % 60:
        raise ValueError(f"duration {duration} is not minute aligned")
    return seconds // 60


def _floor_time(value: datetime, duration: str) -> datetime:
    value = ensure_utc(value)
    seconds = _duration_seconds(duration)
    epoch = int(value.timestamp())
    return parse_unix_timestamp(epoch - epoch % seconds)


def _iter_window_starts(start: datetime, end: datetime, duration: str) -> Iterable[datetime]:
    seconds = _duration_seconds(duration)
    current = _floor_time(start, duration)
    while current <= end:
        yield current
        current += timedelta(seconds=seconds)


def _iter_window_starts_reverse(start: datetime, end: datetime, duration: str) -> Iterable[datetime]:
    seconds = _duration_seconds(duration)
    current = _floor_time(end, duration)
    start = ensure_utc(start)
    while current >= start:
        yield current
        current -= timedelta(seconds=seconds)


def _polymarket_hourly_slug(start: datetime) -> str:
    local = ensure_utc(start).astimezone(ZoneInfo("America/New_York"))
    hour = local.hour % 12 or 12
    meridiem = "am" if local.hour < 12 else "pm"
    month = local.strftime("%B").lower()
    return f"bitcoin-up-or-down-{month}-{local.day}-{local.year}-{hour}{meridiem}-et"


def _polymarket_slug(duration: str, start: datetime) -> str:
    epoch = int(ensure_utc(start).timestamp())
    if duration == "1h":
        return _polymarket_hourly_slug(start)
    return f"btc-updown-{duration}-{epoch}"


def _best_bid(book: dict[str, Any]) -> tuple[float | None, float | None]:
    levels = [
        (_float(level.get("price")), _float(level.get("size")))
        for level in book.get("bids", [])
        if isinstance(level, dict)
    ]
    valid = [(price, size) for price, size in levels if price is not None]
    if not valid:
        return None, None
    price, size = max(valid, key=lambda item: item[0])
    return price, size


def _best_ask(book: dict[str, Any]) -> tuple[float | None, float | None]:
    levels = [
        (_float(level.get("price")), _float(level.get("size")))
        for level in book.get("asks", [])
        if isinstance(level, dict)
    ]
    valid = [(price, size) for price, size in levels if price is not None]
    if not valid:
        return None, None
    price, size = min(valid, key=lambda item: item[0])
    return price, size


def _midpoint(bid: float | None, ask: float | None, fallback: float | None = None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return fallback if fallback is not None else bid if bid is not None else ask


class PolymarketClobV2Client:
    """Public Polymarket data client using production CLOB v2 market-data endpoints."""

    def __init__(
        self,
        *,
        gamma_base_url: str = POLYMARKET_GAMMA_BASE,
        clob_base_url: str = POLYMARKET_CLOB_V2_BASE,
    ) -> None:
        self.gamma_base_url = gamma_base_url
        self.clob_base_url = clob_base_url

    def market_by_slug(self, slug: str, *, prefer_closed: bool = False) -> dict[str, Any] | None:
        closed_order = (True, False) if prefer_closed else (False, True)
        for closed in closed_order:
            url = _url(self.gamma_base_url, "markets", {"slug": slug, "closed": str(closed).lower()})
            try:
                payload = _request_json(url)
            except HTTPError as exc:
                if exc.code in {400, 404, 429}:
                    return None
                raise
            except (TimeoutError, URLError):
                return None
            if isinstance(payload, list) and payload:
                return payload[0] if isinstance(payload[0], dict) else None
        return None

    def discover_btc_updown_markets(
        self,
        *,
        start: datetime,
        end: datetime,
        durations: Sequence[str],
        max_markets: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        markets: list[dict[str, Any]] = []
        attempted: dict[str, int] = {}
        found: dict[str, int] = {}
        supported = [duration for duration in durations if duration in POLYMARKET_BTC_UPDOWN_DURATIONS]
        unsupported = [duration for duration in durations if duration not in POLYMARKET_BTC_UPDOWN_DURATIONS]
        per_duration_limit = max(1, max_markets // max(len(supported), 1))
        for duration in durations:
            if duration not in POLYMARKET_BTC_UPDOWN_DURATIONS:
                continue
            duration_markets = 0
            for window_start in _iter_window_starts_reverse(start, end, duration):
                if len(markets) >= max_markets or duration_markets >= per_duration_limit:
                    break
                slug = _polymarket_slug(duration, window_start)
                attempted[duration] = attempted.get(duration, 0) + 1
                window_end = window_start + timedelta(seconds=_duration_seconds(duration))
                market = self.market_by_slug(slug, prefer_closed=window_end < datetime.now(tz=UTC))
                if market is None:
                    continue
                market["_zeroalpha_duration"] = duration
                market["_zeroalpha_window_start_utc"] = window_start.isoformat()
                market["_zeroalpha_window_end_utc"] = window_end.isoformat()
                markets.append(market)
                duration_markets += 1
                found[duration] = found.get(duration, 0) + 1
            if len(markets) >= max_markets:
                break
        return markets, {
            "attempted": attempted,
            "found": found,
            "unsupported_durations": unsupported,
            "max_markets": max_markets,
            "truncated": len(markets) >= max_markets,
        }

    def order_books(self, token_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not token_ids:
            return {}
        body = [{"token_id": token_id} for token_id in token_ids]
        try:
            payload = _request_json(_url(self.clob_base_url, "books"), method="POST", payload=body)
        except HTTPError as exc:
            if exc.code in {400, 404, 429}:
                return {}
            raise
        except (TimeoutError, URLError):
            return {}
        if not isinstance(payload, list):
            return {}
        return {
            str(book.get("asset_id")): book
            for book in payload
            if isinstance(book, dict) and book.get("asset_id") is not None
        }

    def prices_history(
        self,
        token_id: str,
        *,
        start: datetime,
        end: datetime,
        fidelity_minutes: int,
    ) -> list[dict[str, float]]:
        params = {
            "market": token_id,
            "startTs": int(ensure_utc(start).timestamp()),
            "endTs": int(ensure_utc(end).timestamp()),
            "interval": "all",
            "fidelity": max(fidelity_minutes, 1),
        }
        try:
            payload = _request_json(_url(self.clob_base_url, "prices-history", params))
        except HTTPError as exc:
            if exc.code in {400, 404, 429}:
                return []
            raise
        except (TimeoutError, URLError):
            return []
        history = payload.get("history", []) if isinstance(payload, dict) else []
        rows: list[dict[str, float]] = []
        for row in history:
            if not isinstance(row, dict):
                continue
            ts = _float(row.get("t"))
            price = _float(row.get("p"))
            if ts is not None and price is not None:
                rows.append({"t": ts, "p": price})
        return rows

    def batch_prices_history(
        self,
        token_ids: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        fidelity_minutes: int,
    ) -> dict[str, list[dict[str, float]]]:
        if not token_ids:
            return {}
        body = {
            "markets": list(token_ids),
            "start_ts": int(ensure_utc(start).timestamp()),
            "end_ts": int(ensure_utc(end).timestamp()),
            "interval": "1m" if fidelity_minutes <= 1 else "all",
            "fidelity": max(fidelity_minutes, 1),
        }
        try:
            payload = _request_json(
                _url(self.clob_base_url, "batch-prices-history"),
                method="POST",
                payload=body,
            )
        except HTTPError as exc:
            if exc.code in {400, 404, 429}:
                return {}
            raise
        except (TimeoutError, URLError):
            return {}
        history = payload.get("history", {}) if isinstance(payload, dict) else {}
        if not isinstance(history, dict):
            return {}
        results: dict[str, list[dict[str, float]]] = {}
        for token_id, rows in history.items():
            parsed_rows: list[dict[str, float]] = []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ts = _float(row.get("t"))
                price = _float(row.get("p"))
                if ts is not None and price is not None:
                    parsed_rows.append({"t": ts, "p": price})
            results[str(token_id)] = parsed_rows
        return results

    def snapshots_from_markets(
        self,
        markets: Sequence[dict[str, Any]],
        *,
        start: datetime,
        end: datetime,
        fidelity_minutes: int = 1,
        include_orderbooks: bool = True,
    ) -> list[PredictionMarketSnapshot]:
        snapshots: list[PredictionMarketSnapshot] = []
        token_ids: list[str] = []
        for market in markets:
            token_ids.extend(str(token) for token in _json_array(market.get("clobTokenIds"))[:2])
        books = self.order_books(token_ids) if include_orderbooks else {}
        history_by_token: dict[str, list[dict[str, float]]] = {}
        unique_token_ids = list(dict.fromkeys(token_ids))
        for idx in range(0, len(unique_token_ids), 20):
            history_by_token.update(
                self.batch_prices_history(
                    unique_token_ids[idx : idx + 20],
                    start=start,
                    end=end,
                    fidelity_minutes=fidelity_minutes,
                )
            )
        for market in markets:
            duration = str(market.get("_zeroalpha_duration") or "")
            outcomes = [str(value).lower() for value in _json_array(market.get("outcomes"))]
            tokens = [str(value) for value in _json_array(market.get("clobTokenIds"))]
            if len(tokens) < 2:
                continue
            try:
                up_index = outcomes.index("up")
                down_index = outcomes.index("down")
            except ValueError:
                up_index, down_index = 0, 1
            up_token = tokens[up_index]
            down_token = tokens[down_index]
            window_start = datetime.fromisoformat(str(market["_zeroalpha_window_start_utc"]))
            window_end = datetime.fromisoformat(str(market["_zeroalpha_window_end_utc"]))
            market_start = max(start, window_start)
            market_end = min(end, window_end)
            if market_end <= market_start:
                continue
            up_history = history_by_token.get(up_token)
            if up_history is None:
                up_history = self.prices_history(
                    up_token,
                    start=market_start,
                    end=market_end,
                    fidelity_minutes=fidelity_minutes,
                )
            down_history = history_by_token.get(down_token)
            if down_history is None:
                down_history = self.prices_history(
                    down_token,
                    start=market_start,
                    end=market_end,
                    fidelity_minutes=fidelity_minutes,
                )
            by_ts: dict[int, dict[str, float | None]] = {}
            for row in up_history:
                by_ts.setdefault(int(row["t"]), {})["up_mid"] = row["p"]
            for row in down_history:
                by_ts.setdefault(int(row["t"]), {})["down_mid"] = row["p"]
            for ts, values in sorted(by_ts.items()):
                snapshot_time = parse_unix_timestamp(ts)
                if not market_start <= snapshot_time <= market_end:
                    continue
                up_mid = values.get("up_mid")
                down_mid = values.get("down_mid")
                snapshots.append(
                    PredictionMarketSnapshot(
                        provider="polymarket",
                        duration=duration,
                        timestamp_utc=snapshot_time,
                        market_id=str(market.get("id", "")),
                        market_slug=str(market.get("slug", "")),
                        market_title=str(market.get("question", "")),
                        condition_id=str(market.get("conditionId", "")),
                        window_start_utc=window_start,
                        window_end_utc=window_end,
                        up_mid=up_mid,
                        down_mid=down_mid,
                        last_price=_float(market.get("lastTradePrice")),
                        volume=_float(market.get("volume")),
                        volume_24h=_float(market.get("volume24hr")),
                        liquidity=_float(market.get("liquidityClob") or market.get("liquidity")),
                        open_interest=_float(market.get("openInterest")),
                        source="clob_v2_prices_history",
                    )
                )
            if include_orderbooks:
                up_book = books.get(up_token, {})
                down_book = books.get(down_token, {})
                up_bid, up_bid_size = _best_bid(up_book)
                up_ask, up_ask_size = _best_ask(up_book)
                down_bid, down_bid_size = _best_bid(down_book)
                down_ask, down_ask_size = _best_ask(down_book)
                book_ts = _float(up_book.get("timestamp")) or _float(down_book.get("timestamp"))
                timestamp = (
                    parse_unix_timestamp(book_ts / 1000)
                    if book_ts and book_ts > 10_000_000_000
                    else ensure_utc(datetime.now(tz=UTC))
                )
                if start <= timestamp <= end + timedelta(minutes=5):
                    snapshots.append(
                        PredictionMarketSnapshot(
                            provider="polymarket",
                            duration=duration,
                            timestamp_utc=timestamp,
                            market_id=str(market.get("id", "")),
                            market_slug=str(market.get("slug", "")),
                            market_title=str(market.get("question", "")),
                            condition_id=str(market.get("conditionId", "")),
                            window_start_utc=window_start,
                            window_end_utc=window_end,
                            up_bid=up_bid,
                            up_ask=up_ask,
                            up_mid=_midpoint(up_bid, up_ask, _float(market.get("bestBid"))),
                            down_bid=down_bid,
                            down_ask=down_ask,
                            down_mid=_midpoint(down_bid, down_ask),
                            up_bid_size=up_bid_size,
                            up_ask_size=up_ask_size,
                            down_bid_size=down_bid_size,
                            down_ask_size=down_ask_size,
                            last_price=_float(up_book.get("last_trade_price"))
                            or _float(market.get("lastTradePrice")),
                            volume=_float(market.get("volume")),
                            volume_24h=_float(market.get("volume24hr")),
                            liquidity=_float(market.get("liquidityClob") or market.get("liquidity")),
                            open_interest=_float(market.get("openInterest")),
                            source="clob_v2_orderbook",
                        )
                    )
        return snapshots


class KalshiPublicDataClient:
    def __init__(self, *, base_url: str = KALSHI_BASE) -> None:
        self.base_url = base_url

    def get_markets(
        self,
        *,
        series_ticker: str,
        status: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        cursor = ""
        markets: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {
                "limit": min(max(limit, 1), 1000),
                "series_ticker": series_ticker,
                "status": status,
                "cursor": cursor or None,
            }
            if start is not None:
                params["min_close_ts"] = int(ensure_utc(start).timestamp())
            if end is not None:
                params["max_close_ts"] = int(ensure_utc(end).timestamp())
            try:
                payload = _request_json(_url(self.base_url, "markets", params))
            except HTTPError as exc:
                if exc.code == 429:
                    break
                raise
            except (TimeoutError, URLError):
                break
            rows = payload.get("markets", []) if isinstance(payload, dict) else []
            markets.extend(row for row in rows if isinstance(row, dict))
            cursor = str(payload.get("cursor") or "") if isinstance(payload, dict) else ""
            if not cursor or len(rows) < limit:
                break
        return markets

    def market_candlesticks(
        self,
        *,
        series_ticker: str,
        ticker: str,
        start: datetime,
        end: datetime,
        period_interval_minutes: int = 1,
    ) -> list[dict[str, Any]]:
        params = {
            "start_ts": int(ensure_utc(start).timestamp()),
            "end_ts": int(ensure_utc(end).timestamp()),
            "period_interval": period_interval_minutes,
            "include_latest_before_start": "true",
        }
        path = f"series/{urllib.parse.quote(series_ticker)}/markets/{urllib.parse.quote(ticker)}/candlesticks"
        try:
            payload = _request_json(_url(self.base_url, path, params))
        except HTTPError as exc:
            if exc.code in {400, 404, 429}:
                return []
            raise
        except (TimeoutError, URLError):
            return []
        rows = payload.get("candlesticks", []) if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    def snapshots_from_btc_series(
        self,
        *,
        start: datetime,
        end: datetime,
        max_markets: int,
        durations: Sequence[str] = BTC_PREDICTION_MARKET_DURATIONS,
    ) -> tuple[list[PredictionMarketSnapshot], dict[str, Any]]:
        snapshots: list[PredictionMarketSnapshot] = []
        requested_durations = tuple(dict.fromkeys(duration.strip().lower() for duration in durations))
        requested_series = tuple(
            dict.fromkeys(
                series
                for duration in requested_durations
                for series in KALSHI_BTC_DURATION_SERIES.get(duration, ())
            )
        )
        coverage: dict[str, Any] = {
            "series": {},
            "durations": list(requested_durations),
            "unsupported_durations": [
                duration for duration in requested_durations if duration not in KALSHI_BTC_DURATION_SERIES
            ],
            "max_markets": max_markets,
        }
        for series_ticker in requested_series:
            statuses = (
                ("open", "closed", "settled")
                if series_ticker in {"KXBTC5M", "KXBTC15M"}
                else ("open",)
            )
            markets: list[dict[str, Any]] = []
            for status in statuses:
                markets.extend(
                    self.get_markets(
                        series_ticker=series_ticker,
                        status=status,
                        start=start,
                        end=end,
                    )
                )
                if len(markets) >= max_markets:
                    break
            markets = markets[:max_markets]
            coverage["series"][series_ticker] = {"markets": len(markets), "statuses": list(statuses)}
            for market in markets:
                if series_ticker in {"KXBTC5M", "KXBTC15M"}:
                    duration = "5m" if series_ticker == "KXBTC5M" else "15m"
                    snapshots.extend(
                        self._snapshots_from_directional_market(
                            series_ticker,
                            market,
                            start,
                            end,
                            duration=duration,
                        )
                    )
                else:
                    snapshot = self._snapshot_from_market_ladder(series_ticker, market)
                    if snapshot is not None:
                        snapshots.append(snapshot)
        return snapshots, coverage

    def _snapshots_from_directional_market(
        self,
        series_ticker: str,
        market: dict[str, Any],
        start: datetime,
        end: datetime,
        *,
        duration: str,
    ) -> list[PredictionMarketSnapshot]:
        ticker = str(market.get("ticker", ""))
        close_time = _parse_iso(market.get("close_time"))
        open_time = _parse_iso(market.get("open_time"))
        if not ticker or close_time is None:
            return []
        market_start = max(start, open_time or start)
        market_end = min(end, close_time)
        rows = self.market_candlesticks(
            series_ticker=series_ticker,
            ticker=ticker,
            start=market_start,
            end=market_end,
            period_interval_minutes=1,
        )
        snapshots: list[PredictionMarketSnapshot] = []
        for row in rows:
            ts = _float(row.get("end_period_ts"))
            if ts is None:
                continue
            price = _candlestick_close(row.get("price"))
            yes_bid = _candlestick_close(row.get("yes_bid"))
            yes_ask = _candlestick_close(row.get("yes_ask"))
            up_mid = price if price is not None else _midpoint(yes_bid, yes_ask)
            snapshots.append(
                PredictionMarketSnapshot(
                    provider="kalshi",
                    duration=duration,
                    timestamp_utc=parse_unix_timestamp(ts),
                    market_id=ticker,
                    market_slug=ticker,
                    market_title=str(market.get("title", "")),
                    condition_id=str(market.get("event_ticker", "")),
                    window_start_utc=open_time,
                    window_end_utc=close_time,
                    up_bid=yes_bid,
                    up_ask=yes_ask,
                    up_mid=up_mid,
                    down_mid=1.0 - up_mid if up_mid is not None else None,
                    last_price=price,
                    volume=_float(row.get("volume_fp") or row.get("volume")),
                    volume_24h=_float(market.get("volume_24h_fp")),
                    liquidity=_float(market.get("liquidity_dollars")),
                    open_interest=_float(row.get("open_interest_fp") or row.get("open_interest")),
                    source="kalshi_market_candlesticks",
                )
            )
        if not snapshots:
            snapshot = self._snapshot_from_directional_market_quote(market, duration=duration)
            return [snapshot] if snapshot is not None else []
        return snapshots

    def _snapshot_from_directional_market_quote(
        self,
        market: dict[str, Any],
        *,
        duration: str,
    ) -> PredictionMarketSnapshot | None:
        timestamp = _parse_iso(market.get("updated_time")) or ensure_utc(datetime.now(tz=UTC))
        close_time = _parse_iso(market.get("close_time"))
        open_time = _parse_iso(market.get("open_time"))
        yes_bid = _float(market.get("yes_bid_dollars"))
        yes_ask = _float(market.get("yes_ask_dollars"))
        last = _float(market.get("last_price_dollars"))
        up_mid = _midpoint(yes_bid, yes_ask, last)
        return PredictionMarketSnapshot(
            provider="kalshi",
            duration=duration,
            timestamp_utc=timestamp,
            market_id=str(market.get("ticker", "")),
            market_slug=str(market.get("ticker", "")),
            market_title=str(market.get("title", "")),
            condition_id=str(market.get("event_ticker", "")),
            window_start_utc=open_time,
            window_end_utc=close_time,
            up_bid=yes_bid,
            up_ask=yes_ask,
            up_mid=up_mid,
            down_bid=_float(market.get("no_bid_dollars")),
            down_ask=_float(market.get("no_ask_dollars")),
            down_mid=1.0 - up_mid if up_mid is not None else None,
            last_price=last,
            volume=_float(market.get("volume_fp")),
            volume_24h=_float(market.get("volume_24h_fp")),
            liquidity=_float(market.get("liquidity_dollars")),
            open_interest=_float(market.get("open_interest_fp")),
            source="kalshi_market_quote",
        )

    def _snapshot_from_market_ladder(
        self,
        series_ticker: str,
        market: dict[str, Any],
    ) -> PredictionMarketSnapshot | None:
        close_time = _parse_iso(market.get("close_time"))
        open_time = _parse_iso(market.get("open_time"))
        timestamp = _parse_iso(market.get("updated_time")) or ensure_utc(datetime.now(tz=UTC))
        yes_bid = _float(market.get("yes_bid_dollars"))
        yes_ask = _float(market.get("yes_ask_dollars"))
        last = _float(market.get("last_price_dollars"))
        mid = _midpoint(yes_bid, yes_ask, last)
        duration = "1h" if series_ticker in {"KXBTC", "KXBTCD"} else "unknown"
        return PredictionMarketSnapshot(
            provider="kalshi",
            duration=duration,
            timestamp_utc=timestamp,
            market_id=str(market.get("ticker", "")),
            market_slug=str(market.get("ticker", "")),
            market_title=str(market.get("title", "")),
            condition_id=str(market.get("event_ticker", "")),
            window_start_utc=open_time,
            window_end_utc=close_time,
            up_bid=yes_bid,
            up_ask=yes_ask,
            up_mid=mid,
            down_bid=_float(market.get("no_bid_dollars")),
            down_ask=_float(market.get("no_ask_dollars")),
            down_mid=1.0 - mid if mid is not None else None,
            last_price=last,
            volume=_float(market.get("volume_fp")),
            volume_24h=_float(market.get("volume_24h_fp")),
            liquidity=_float(market.get("liquidity_dollars")),
            open_interest=_float(market.get("open_interest_fp")),
            source=f"kalshi_{series_ticker.lower()}_quote",
        )


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _candlestick_close(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    for key in ("close_dollars", "close", "previous_dollars", "previous"):
        parsed = _float(value.get(key))
        if parsed is not None:
            return parsed
    return None


def write_prediction_market_snapshots(path: Path, snapshots: Sequence[PredictionMarketSnapshot]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(snapshot.as_json_dict(), sort_keys=True) + "\n")
    return len(snapshots)


def read_prediction_market_snapshots(path: Path) -> list[PredictionMarketSnapshot]:
    snapshots: list[PredictionMarketSnapshot] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                snapshots.append(PredictionMarketSnapshot.from_json_dict(json.loads(line)))
    return snapshots


def load_prediction_market_snapshots(
    *,
    start: datetime,
    end: datetime,
    durations: Sequence[str],
    cache_dir: Path,
    max_markets: int = 500,
    fidelity_minutes: int = 1,
    refresh: bool = False,
) -> PredictionMarketLoadResult:
    start = ensure_utc(start)
    end = ensure_utc(end)
    durations = tuple(dict.fromkeys(duration.strip().lower() for duration in durations if duration.strip()))
    duration_key = "-".join(durations) or "none"
    cache_path = (
        cache_dir
        / f"btc_prediction_market_snapshots_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_{duration_key}_{max_markets}.jsonl"
    )
    if cache_path.exists() and not refresh:
        snapshots = read_prediction_market_snapshots(cache_path)
        provider_duration_counts: dict[str, dict[str, int]] = {}
        for snapshot in snapshots:
            durations_for_provider = provider_duration_counts.setdefault(snapshot.provider, {})
            durations_for_provider[snapshot.duration] = durations_for_provider.get(snapshot.duration, 0) + 1
        return PredictionMarketLoadResult(
            snapshots=snapshots,
            coverage={
                "cache": {"path": str(cache_path), "hit": True},
                "snapshots": len(snapshots),
                "durations": list(durations),
                "polymarket": {
                    "api": "gamma_discovery_plus_clob_v2_market_data",
                    "snapshots": sum(provider_duration_counts.get("polymarket", {}).values()),
                    "duration_snapshots": provider_duration_counts.get("polymarket", {}),
                },
                "kalshi": {
                    "api": "trade_api_v2_public_market_data",
                    "snapshots": sum(provider_duration_counts.get("kalshi", {}).values()),
                    "duration_snapshots": provider_duration_counts.get("kalshi", {}),
                },
            },
        )

    polymarket = PolymarketClobV2Client()
    poly_markets, poly_coverage = polymarket.discover_btc_updown_markets(
        start=start,
        end=end,
        durations=durations,
        max_markets=max_markets,
    )
    poly_snapshots = polymarket.snapshots_from_markets(
        poly_markets,
        start=start,
        end=end,
        fidelity_minutes=fidelity_minutes,
        include_orderbooks=True,
    )

    kalshi = KalshiPublicDataClient()
    kalshi_snapshots, kalshi_coverage = kalshi.snapshots_from_btc_series(
        start=start,
        end=end,
        max_markets=max_markets,
        durations=durations,
    )

    snapshots = sorted(
        [*poly_snapshots, *kalshi_snapshots],
        key=lambda snapshot: (
            snapshot.timestamp_utc,
            snapshot.provider,
            snapshot.duration,
            snapshot.market_id,
        ),
    )
    write_prediction_market_snapshots(cache_path, snapshots)
    return PredictionMarketLoadResult(
        snapshots=snapshots,
        coverage={
            "cache": {"path": str(cache_path), "hit": False},
            "durations": list(durations),
            "snapshots": len(snapshots),
            "polymarket": {
                **poly_coverage,
                "markets": len(poly_markets),
                "snapshots": len(poly_snapshots),
                "api": "gamma_discovery_plus_clob_v2_market_data",
            },
            "kalshi": {
                **kalshi_coverage,
                "snapshots": len(kalshi_snapshots),
                "api": "trade_api_v2_public_market_data",
            },
        },
    )


@dataclass(frozen=True, slots=True)
class PreparedPredictionMarketSnapshots:
    by_provider_duration: dict[tuple[str, str], tuple[PredictionMarketSnapshot, ...]]
    timestamps: dict[tuple[str, str], tuple[datetime, ...]]

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[PredictionMarketSnapshot] | None,
    ) -> "PreparedPredictionMarketSnapshots":
        grouped: dict[tuple[str, str], list[PredictionMarketSnapshot]] = {}
        for snapshot in snapshots or []:
            grouped.setdefault((snapshot.provider, snapshot.duration), []).append(snapshot)
        ordered = {
            key: tuple(sorted(values, key=lambda snapshot: snapshot.timestamp_utc))
            for key, values in grouped.items()
        }
        return cls(
            by_provider_duration=ordered,
            timestamps={key: tuple(snapshot.timestamp_utc for snapshot in values) for key, values in ordered.items()},
        )

    def latest_active_before(
        self,
        *,
        provider: str,
        duration: str,
        timestamp: datetime,
        max_age: timedelta,
    ) -> PredictionMarketSnapshot | None:
        key = (provider, duration)
        times = self.timestamps.get(key)
        snapshots = self.by_provider_duration.get(key)
        if not times or not snapshots:
            return None
        event_time = ensure_utc(timestamp)
        idx = bisect_right(times, event_time) - 1
        while idx >= 0:
            snapshot = snapshots[idx]
            age = event_time - snapshot.timestamp_utc
            if age > max_age:
                return None
            if snapshot.window_start_utc and event_time < snapshot.window_start_utc:
                idx -= 1
                continue
            if snapshot.window_end_utc and event_time > snapshot.window_end_utc:
                idx -= 1
                continue
            return snapshot
        return None
