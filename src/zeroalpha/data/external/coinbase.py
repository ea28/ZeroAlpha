"""Coinbase Exchange public data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import urllib.parse
import urllib.request

from zeroalpha.domain import Bar
from zeroalpha.timeutils import ensure_utc, parse_unix_timestamp


COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"


@dataclass(frozen=True, slots=True)
class CoinbaseExchangeClient:
    base_url: str = COINBASE_EXCHANGE_BASE

    def candles_url(
        self,
        product_id: str,
        granularity: int,
        start: datetime,
        end: datetime,
    ) -> str:
        params = urllib.parse.urlencode(
            {
                "granularity": str(granularity),
                "start": ensure_utc(start).isoformat().replace("+00:00", "Z"),
                "end": ensure_utc(end).isoformat().replace("+00:00", "Z"),
            }
        )
        return f"{self.base_url}/products/{product_id}/candles?{params}"

    def fetch_candles(
        self,
        product_id: str,
        granularity: int,
        start: datetime,
        end: datetime,
        timeout: float = 30.0,
    ) -> list[Bar]:
        url = self.candles_url(product_id, granularity, start, end)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "ZeroAlpha/0.1",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        bars = [
            Bar(
                timestamp_utc=parse_unix_timestamp(row[0]),
                symbol=product_id,
                bar_size=f"{granularity}s",
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                source="COINBASE",
            )
            for row in payload
        ]
        return sorted(bars, key=lambda bar: bar.timestamp_utc)

    def fetch_candles_range(
        self,
        product_id: str,
        granularity: int,
        start: datetime,
        end: datetime,
        timeout: float = 30.0,
    ) -> list[Bar]:
        bars: list[Bar] = []
        for window_start, window_end in candle_windows(start, end, granularity):
            bars.extend(
                self.fetch_candles(
                    product_id,
                    granularity,
                    window_start,
                    window_end,
                    timeout=timeout,
                )
            )
        deduped = {bar.timestamp_utc: bar for bar in bars}
        return [
            bar
            for bar in sorted(deduped.values(), key=lambda row: row.timestamp_utc)
            if ensure_utc(start) <= bar.timestamp_utc < ensure_utc(end)
        ]


def candle_windows(start: datetime, end: datetime, granularity_seconds: int) -> list[tuple[datetime, datetime]]:
    """Split Coinbase candle requests into <=300 candle windows."""
    start = ensure_utc(start)
    end = ensure_utc(end)
    if end <= start:
        raise ValueError("end must be after start")
    if granularity_seconds <= 0:
        raise ValueError("granularity_seconds must be positive")
    max_window = timedelta(seconds=granularity_seconds * 300)
    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + max_window, end)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows
