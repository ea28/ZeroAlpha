"""External data-source health checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import urllib.parse
import urllib.request

from zeroalpha.data.external.binance import (
    BinancePublicDataClient,
    download_zip_with_checksum,
    parse_kline_zip,
)
from zeroalpha.data.external.coinbase import CoinbaseExchangeClient


@dataclass(frozen=True, slots=True)
class DataSourceHealth:
    source: str
    ok: bool
    detail: str
    rows: int = 0
    url: str | None = None


def _read_json_url(url: str, *, timeout: float = 20.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "ZeroAlpha/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_binance_archive(
    *,
    cache_dir: Path,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> DataSourceHealth:
    client = BinancePublicDataClient()
    last_error: Exception | None = None
    for days_back in range(2, 8):
        day = (datetime.now(tz=UTC) - timedelta(days=days_back)).date()
        url = client.daily_klines_url(symbol, interval, day)
        try:
            payload = download_zip_with_checksum(client, url, cache_dir / "health" / symbol / interval)
            if payload is None:
                continue
            bars = parse_kline_zip(payload, symbol=symbol, interval=interval)
            return DataSourceHealth("binance_archive", bool(bars), "checksum_ok", len(bars), url)
        except Exception as exc:
            last_error = exc
    return DataSourceHealth("binance_archive", False, f"{type(last_error).__name__}: {last_error}")


def check_coinbase_candles(product_id: str = "BTC-USD", granularity: int = 3600) -> DataSourceHealth:
    client = CoinbaseExchangeClient()
    end = datetime.now(tz=UTC) - timedelta(hours=1)
    start = end - timedelta(hours=6)
    url = client.candles_url(product_id, granularity, start, end)
    try:
        bars = client.fetch_candles_range(product_id, granularity, start, end)
        return DataSourceHealth("coinbase_candles", bool(bars), "rest_ok", len(bars), url)
    except Exception as exc:
        return DataSourceHealth("coinbase_candles", False, f"{type(exc).__name__}: {exc}", url=url)


def check_kraken_ohlc(pair: str = "XBTUSD", interval: int = 60) -> DataSourceHealth:
    params = urllib.parse.urlencode({"pair": pair, "interval": str(interval)})
    url = f"https://api.kraken.com/0/public/OHLC?{params}"
    try:
        payload = _read_json_url(url)
        if not isinstance(payload, dict) or payload.get("error"):
            return DataSourceHealth("kraken_ohlc", False, f"api_error={payload}", url=url)
        result = payload.get("result", {})
        rows = 0
        if isinstance(result, dict):
            rows = sum(len(value) for key, value in result.items() if key != "last" and isinstance(value, list))
        return DataSourceHealth("kraken_ohlc", rows > 0, "rest_ok", rows, url)
    except Exception as exc:
        return DataSourceHealth("kraken_ohlc", False, f"{type(exc).__name__}: {exc}", url=url)


def run_external_data_health_checks(*, cache_dir: Path) -> list[DataSourceHealth]:
    return [
        check_binance_archive(cache_dir=cache_dir),
        check_coinbase_candles(),
        check_kraken_ohlc(),
    ]


def health_checks_as_dict(checks: list[DataSourceHealth]) -> list[dict[str, object]]:
    return [asdict(check) for check in checks]
