"""Binance public data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
import csv
import hashlib
import urllib.request
from urllib.error import HTTPError
import zipfile

from zeroalpha.domain import Bar
from zeroalpha.timeutils import ensure_utc, parse_unix_timestamp


BINANCE_DATA_BASE = "https://data.binance.vision/data"
BINANCE_SPOT_REST_BASE = "https://api.binance.com"
BINANCE_FAPI_BASE = "https://fapi.binance.com"


def _interval_delta(interval: str) -> timedelta:
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported Binance interval {interval}") from exc
    if value <= 0:
        raise ValueError("interval must be positive")
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"unsupported Binance interval {interval}")


@dataclass(frozen=True, slots=True)
class BinancePublicDataClient:
    base_url: str = BINANCE_DATA_BASE
    spot_rest_base_url: str = BINANCE_SPOT_REST_BASE

    def monthly_klines_url(self, symbol: str, interval: str, month: str) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/spot/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{month}.zip"
        )

    def monthly_futures_klines_url(self, symbol: str, interval: str, month: str, *, market_type: str = "um") -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/futures/{market_type}/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{month}.zip"
        )

    def daily_klines_url(self, symbol: str, interval: str, day: date) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/spot/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{day.isoformat()}.zip"
        )

    def daily_futures_klines_url(self, symbol: str, interval: str, day: date, *, market_type: str = "um") -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/futures/{market_type}/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{day.isoformat()}.zip"
        )

    def monthly_agg_trades_url(self, symbol: str, month: str) -> str:
        symbol = symbol.upper()
        return f"{self.base_url}/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{month}.zip"

    def spot_depth_url(self, symbol: str, *, limit: int = 100) -> str:
        symbol = symbol.upper()
        return f"{self.spot_rest_base_url}/api/v3/depth?symbol={symbol}&limit={min(max(limit, 1), 5000)}"

    def spot_book_ticker_url(self, symbol: str) -> str:
        symbol = symbol.upper()
        return f"{self.spot_rest_base_url}/api/v3/ticker/bookTicker?symbol={symbol}"

    def spot_recent_trades_url(self, symbol: str, *, limit: int = 1000) -> str:
        symbol = symbol.upper()
        return f"{self.spot_rest_base_url}/api/v3/trades?symbol={symbol}&limit={min(max(limit, 1), 1000)}"

    def spot_agg_trades_url(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> str:
        symbol = symbol.upper()
        return (
            f"{self.spot_rest_base_url}/api/v3/aggTrades?symbol={symbol}"
            f"&startTime={int(ensure_utc(start).timestamp() * 1000)}"
            f"&endTime={int(ensure_utc(end).timestamp() * 1000)}"
            f"&limit={min(max(limit, 1), 1000)}"
        )

    def get_json(self, url: str, timeout: float = 30.0):
        import json

        request = urllib.request.Request(url, headers={"User-Agent": "ZeroAlpha/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_spot_book_ticker(self, symbol: str, timeout: float = 30.0) -> dict:
        payload = self.get_json(self.spot_book_ticker_url(symbol), timeout=timeout)
        return payload if isinstance(payload, dict) else {}

    def download(self, url: str, timeout: float = 60.0) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()

    def checksum_url(self, url: str) -> str:
        return f"{url}.CHECKSUM"


@dataclass(frozen=True, slots=True)
class BinanceFuturesMetricsClient:
    base_url: str = BINANCE_FAPI_BASE

    def funding_rate_url(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = 1000,
    ) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/fapi/v1/fundingRate?"
            f"symbol={symbol}&startTime={int(ensure_utc(start).timestamp() * 1000)}"
            f"&endTime={int(ensure_utc(end).timestamp() * 1000)}&limit={limit}"
        )

    def open_interest_hist_url(
        self,
        symbol: str,
        period: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = 500,
    ) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/futures/data/openInterestHist?"
            f"symbol={symbol}&period={period}&startTime={int(ensure_utc(start).timestamp() * 1000)}"
            f"&endTime={int(ensure_utc(end).timestamp() * 1000)}&limit={limit}"
        )

    def taker_buy_sell_volume_url(
        self,
        symbol: str,
        period: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = 500,
    ) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/futures/data/takerlongshortRatio?"
            f"symbol={symbol}&period={period}&startTime={int(ensure_utc(start).timestamp() * 1000)}"
            f"&endTime={int(ensure_utc(end).timestamp() * 1000)}&limit={limit}"
        )

    def basis_url(
        self,
        pair: str,
        period: str,
        start: datetime,
        end: datetime,
        *,
        contract_type: str = "PERPETUAL",
        limit: int = 500,
    ) -> str:
        pair = pair.upper()
        return (
            f"{self.base_url}/futures/data/basis?"
            f"pair={pair}&contractType={contract_type}&period={period}"
            f"&startTime={int(ensure_utc(start).timestamp() * 1000)}"
            f"&endTime={int(ensure_utc(end).timestamp() * 1000)}&limit={limit}"
        )

    def get_json(self, url: str, timeout: float = 30.0):
        import json

        request = urllib.request.Request(url, headers={"User-Agent": "ZeroAlpha/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_funding_rates(
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        timeout: float = 30.0,
    ) -> list[Bar]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        bars: list[Bar] = []
        cursor = start
        while cursor < end:
            payload = self.get_json(self.funding_rate_url(symbol, cursor, end), timeout=timeout)
            rows = payload if isinstance(payload, list) else []
            if not rows:
                break
            last_timestamp = None
            for row in rows:
                funding_rate = float(row.get("fundingRate", 0.0) or 0.0)
                mark_price = float(row.get("markPrice", 0.0) or 0.0)
                metric_value = 1.0 + funding_rate
                timestamp = parse_unix_timestamp(int(row["fundingTime"]) // 1000)
                if timestamp < cursor:
                    continue
                last_timestamp = timestamp
                bars.append(
                    Bar(
                        timestamp_utc=timestamp,
                        symbol=symbol.upper(),
                        bar_size="8h",
                        open=metric_value,
                        high=metric_value,
                        low=metric_value,
                        close=metric_value,
                        volume=0.0,
                        source="BINANCE_UM_FUNDING",
                        extra={
                            "funding_rate": funding_rate,
                            "mark_price": mark_price,
                        },
                    )
                )
            if last_timestamp is None or last_timestamp <= cursor:
                break
            cursor = last_timestamp + timedelta(milliseconds=1)
        return [
            bar
            for bar in sorted(bars, key=lambda item: item.timestamp_utc)
            if start <= bar.timestamp_utc < end
        ]

    def fetch_open_interest_history(
        self,
        *,
        symbol: str,
        period: str,
        start: datetime,
        end: datetime,
        timeout: float = 30.0,
    ) -> list[Bar]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        bars: list[Bar] = []
        cursor = start
        while cursor < end:
            payload = self.get_json(self.open_interest_hist_url(symbol, period, cursor, end), timeout=timeout)
            rows = payload if isinstance(payload, list) else []
            if not rows:
                break
            last_timestamp = None
            for row in rows:
                open_interest = float(row.get("sumOpenInterest", 0.0) or 0.0)
                open_interest_value = float(row.get("sumOpenInterestValue", 0.0) or 0.0)
                timestamp = parse_unix_timestamp(int(row["timestamp"]) // 1000)
                if timestamp < cursor:
                    continue
                last_timestamp = timestamp
                close = max(open_interest_value, open_interest, 1e-12)
                bars.append(
                    Bar(
                        timestamp_utc=timestamp,
                        symbol=symbol.upper(),
                        bar_size=period,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=open_interest,
                        quote_volume=open_interest_value if open_interest_value > 0 else None,
                        source="BINANCE_UM_OPEN_INTEREST",
                        extra={
                            "open_interest": open_interest,
                            "open_interest_value": open_interest_value,
                        },
                    )
                )
            if last_timestamp is None or last_timestamp <= cursor:
                break
            cursor = last_timestamp + timedelta(milliseconds=1)
        return [
            bar
            for bar in sorted(bars, key=lambda item: item.timestamp_utc)
            if start <= bar.timestamp_utc < end
        ]

    def fetch_taker_buy_sell_volume(
        self,
        *,
        symbol: str,
        period: str,
        start: datetime,
        end: datetime,
        timeout: float = 30.0,
    ) -> list[Bar]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        bars: list[Bar] = []
        cursor = start
        while cursor < end:
            payload = self.get_json(
                self.taker_buy_sell_volume_url(symbol, period, cursor, end),
                timeout=timeout,
            )
            rows = payload if isinstance(payload, list) else []
            if not rows:
                break
            last_timestamp = None
            for row in rows:
                timestamp = parse_unix_timestamp(int(row["timestamp"]) // 1000)
                if timestamp < cursor:
                    continue
                buy_volume = float(row.get("buyVol", 0.0) or 0.0)
                sell_volume = float(row.get("sellVol", 0.0) or 0.0)
                ratio = float(row.get("buySellRatio", 0.0) or 0.0)
                total_volume = buy_volume + sell_volume
                close = ratio if ratio > 0 else (buy_volume / sell_volume if sell_volume > 0 else 1.0)
                last_timestamp = timestamp
                bars.append(
                    Bar(
                        timestamp_utc=timestamp,
                        symbol=symbol.upper(),
                        bar_size=period,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=total_volume,
                        source="BINANCE_UM_TAKERFLOW",
                        extra={
                            "buy_volume": buy_volume,
                            "sell_volume": sell_volume,
                            "taker_buy_volume": buy_volume,
                            "taker_buy_base_volume": buy_volume,
                            "buy_sell_ratio": close,
                            "signed_taker_volume": buy_volume - sell_volume,
                        },
                    )
                )
            if last_timestamp is None or last_timestamp <= cursor:
                break
            cursor = last_timestamp + timedelta(milliseconds=1)
        return [
            bar
            for bar in sorted(bars, key=lambda item: item.timestamp_utc)
            if start <= bar.timestamp_utc < end
        ]

    def fetch_basis_history(
        self,
        *,
        pair: str,
        period: str,
        start: datetime,
        end: datetime,
        contract_type: str = "PERPETUAL",
        timeout: float = 30.0,
    ) -> list[Bar]:
        start = ensure_utc(start)
        end = ensure_utc(end)
        bars: list[Bar] = []
        cursor = start
        while cursor < end:
            payload = self.get_json(
                self.basis_url(pair, period, cursor, end, contract_type=contract_type),
                timeout=timeout,
            )
            rows = payload if isinstance(payload, list) else []
            if not rows:
                break
            last_timestamp = None
            for row in rows:
                timestamp = parse_unix_timestamp(int(row["timestamp"]) // 1000)
                if timestamp < cursor:
                    continue
                basis = float(row.get("basis", 0.0) or 0.0)
                basis_rate = float(row.get("basisRate", 0.0) or 0.0)
                futures_price = float(row.get("futuresPrice", 0.0) or 0.0)
                index_price = float(row.get("indexPrice", 0.0) or 0.0)
                annualized_basis_rate = float(row.get("annualizedBasisRate", 0.0) or 0.0)
                last_timestamp = timestamp
                bars.append(
                    Bar(
                        timestamp_utc=timestamp,
                        symbol=pair.upper(),
                        bar_size=period,
                        open=basis_rate,
                        high=basis_rate,
                        low=basis_rate,
                        close=basis_rate,
                        volume=0.0,
                        source="BINANCE_UM_BASIS",
                        extra={
                            "basis": basis,
                            "basis_rate": basis_rate,
                            "futures_price": futures_price,
                            "index_price": index_price,
                            "annualized_basis_rate": annualized_basis_rate,
                        },
                    )
                )
            if last_timestamp is None or last_timestamp <= cursor:
                break
            cursor = last_timestamp + timedelta(milliseconds=1)
        return [
            bar
            for bar in sorted(bars, key=lambda item: item.timestamp_utc)
            if start <= bar.timestamp_utc < end
        ]


def verify_sha256(payload: bytes, checksum_text: str) -> bool:
    expected = checksum_text.strip().split()[0]
    actual = hashlib.sha256(payload).hexdigest()
    return actual == expected


def parse_kline_zip(payload: bytes, *, symbol: str, interval: str, source: str = "BINANCE") -> list[Bar]:
    rows: list[Bar] = []
    interval_delta = _interval_delta(interval)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError("zip payload does not contain a CSV file")
        with archive.open(csv_names[0]) as raw:
            reader = csv.reader(TextIOWrapper(raw, encoding="utf-8"))
            for fields in reader:
                if not fields or fields[0] == "open_time":
                    continue
                bar_start = parse_unix_timestamp(int(fields[0]))
                bar_close = bar_start + interval_delta
                rows.append(
                    Bar(
                        timestamp_utc=bar_close,
                        symbol=symbol.upper(),
                        bar_size=interval,
                        open=float(fields[1]),
                        high=float(fields[2]),
                        low=float(fields[3]),
                        close=float(fields[4]),
                        volume=float(fields[5]),
                        quote_volume=float(fields[7]),
                        trade_count=int(fields[8]),
                        source=source,
                        extra={
                            "bar_start_timestamp_utc": bar_start.isoformat(),
                            "bar_close_timestamp_utc": bar_close.isoformat(),
                            "taker_buy_base_volume": float(fields[9]) if len(fields) > 9 and fields[9] else 0.0,
                            "taker_buy_quote_volume": float(fields[10]) if len(fields) > 10 and fields[10] else 0.0,
                        },
                    )
                )
    return rows


def read_kline_zip(path: str | Path, *, symbol: str, interval: str) -> list[Bar]:
    return parse_kline_zip(Path(path).read_bytes(), symbol=symbol, interval=interval)


def month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    months: list[date] = []
    while current <= end:
        months.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def first_day_next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def is_full_month(month_start: date, start: date, end: date) -> bool:
    return month_start >= date(start.year, start.month, 1) and (
        month_start > start and first_day_next_month(month_start) - timedelta(days=1) < end
    )


def download_zip_with_checksum(
    client: BinancePublicDataClient,
    url: str,
    cache_dir: Path,
    *,
    timeout: float = 60.0,
) -> bytes | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    zip_path = cache_dir / filename
    checksum_path = cache_dir / f"{filename}.CHECKSUM"

    if zip_path.exists() and checksum_path.exists():
        payload = zip_path.read_bytes()
        if verify_sha256(payload, checksum_path.read_text(encoding="utf-8")):
            return payload

    try:
        payload = client.download(url, timeout=timeout)
        checksum_text = client.download(client.checksum_url(url), timeout=timeout).decode("utf-8")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise

    if not verify_sha256(payload, checksum_text):
        raise ValueError(f"checksum failed for {url}")
    zip_path.write_bytes(payload)
    checksum_path.write_text(checksum_text, encoding="utf-8")
    return payload


def fetch_klines_archive_range(
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    client: BinancePublicDataClient | None = None,
) -> list[Bar]:
    """Fetch Binance archive klines for a UTC range using monthly files when possible."""
    client = client or BinancePublicDataClient()
    start = ensure_utc(start)
    end = ensure_utc(end)
    if end <= start:
        raise ValueError("end must be after start")

    start_date = start.date()
    end_date = end.date()
    bars: list[Bar] = []
    for month_start in month_starts(start_date, end_date):
        month_end = first_day_next_month(month_start) - timedelta(days=1)
        if start_date <= month_start and month_end <= end_date and month_end < date.today().replace(day=1):
            url = client.monthly_klines_url(symbol, interval, month_start.strftime("%Y-%m"))
            payload = download_zip_with_checksum(client, url, cache_dir / "monthly")
            if payload is not None:
                bars.extend(parse_kline_zip(payload, symbol=symbol, interval=interval))
            continue

        day = max(start_date, month_start)
        last = min(end_date, month_end)
        while day <= last:
            # Daily files are usually available the next day. Missing future/current files are skipped.
            url = client.daily_klines_url(symbol, interval, day)
            payload = download_zip_with_checksum(client, url, cache_dir / "daily")
            if payload is not None:
                bars.extend(parse_kline_zip(payload, symbol=symbol, interval=interval))
            day += timedelta(days=1)

    deduped = {bar.timestamp_utc: bar for bar in bars}
    return [
        bar
        for bar in sorted(deduped.values(), key=lambda item: item.timestamp_utc)
        if start <= bar.timestamp_utc < end
    ]


def fetch_futures_klines_archive_range(
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    market_type: str = "um",
    client: BinancePublicDataClient | None = None,
) -> list[Bar]:
    """Fetch Binance USD-M/COIN-M futures archive klines for a UTC range."""

    if market_type not in {"um", "cm"}:
        raise ValueError("market_type must be um or cm")
    client = client or BinancePublicDataClient()
    start = ensure_utc(start)
    end = ensure_utc(end)
    if end <= start:
        raise ValueError("end must be after start")

    start_date = start.date()
    end_date = end.date()
    bars: list[Bar] = []
    source = f"BINANCE_{market_type.upper()}_FUTURES"
    for month_start in month_starts(start_date, end_date):
        month_end = first_day_next_month(month_start) - timedelta(days=1)
        if start_date <= month_start and month_end <= end_date and month_end < date.today().replace(day=1):
            url = client.monthly_futures_klines_url(
                symbol,
                interval,
                month_start.strftime("%Y-%m"),
                market_type=market_type,
            )
            payload = download_zip_with_checksum(client, url, cache_dir / market_type / "monthly")
            if payload is not None:
                bars.extend(parse_kline_zip(payload, symbol=symbol, interval=interval, source=source))
            continue

        day = max(start_date, month_start)
        last = min(end_date, month_end)
        while day <= last:
            url = client.daily_futures_klines_url(symbol, interval, day, market_type=market_type)
            payload = download_zip_with_checksum(client, url, cache_dir / market_type / "daily")
            if payload is not None:
                bars.extend(parse_kline_zip(payload, symbol=symbol, interval=interval, source=source))
            day += timedelta(days=1)

    deduped = {bar.timestamp_utc: bar for bar in bars}
    return [
        bar
        for bar in sorted(deduped.values(), key=lambda item: item.timestamp_utc)
        if start <= bar.timestamp_utc < end
    ]
