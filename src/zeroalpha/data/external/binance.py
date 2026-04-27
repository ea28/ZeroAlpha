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


@dataclass(frozen=True, slots=True)
class BinancePublicDataClient:
    base_url: str = BINANCE_DATA_BASE

    def monthly_klines_url(self, symbol: str, interval: str, month: str) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/spot/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{month}.zip"
        )

    def daily_klines_url(self, symbol: str, interval: str, day: date) -> str:
        symbol = symbol.upper()
        return (
            f"{self.base_url}/spot/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{day.isoformat()}.zip"
        )

    def monthly_agg_trades_url(self, symbol: str, month: str) -> str:
        symbol = symbol.upper()
        return f"{self.base_url}/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{month}.zip"

    def download(self, url: str, timeout: float = 60.0) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()

    def checksum_url(self, url: str) -> str:
        return f"{url}.CHECKSUM"


def verify_sha256(payload: bytes, checksum_text: str) -> bool:
    expected = checksum_text.strip().split()[0]
    actual = hashlib.sha256(payload).hexdigest()
    return actual == expected


def parse_kline_zip(payload: bytes, *, symbol: str, interval: str, source: str = "BINANCE") -> list[Bar]:
    rows: list[Bar] = []
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError("zip payload does not contain a CSV file")
        with archive.open(csv_names[0]) as raw:
            reader = csv.reader(TextIOWrapper(raw, encoding="utf-8"))
            for fields in reader:
                if not fields or fields[0] == "open_time":
                    continue
                rows.append(
                    Bar(
                        timestamp_utc=parse_unix_timestamp(int(fields[0])),
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
