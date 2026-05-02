from datetime import datetime, UTC
from io import BytesIO
import zipfile

from zeroalpha.data.external.binance import BinancePublicDataClient, parse_kline_zip, verify_sha256
from zeroalpha.data.external.coinbase import CoinbaseExchangeClient, candle_windows
from zeroalpha.data.external.kraken import missing_ohlcvt_intervals, parse_ohlcvt_zip


def test_binance_monthly_url() -> None:
    client = BinancePublicDataClient()
    assert (
        client.monthly_klines_url("btcusdt", "1h", "2026-04")
        == "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-04.zip"
    )
    assert (
        client.daily_futures_klines_url("btcusdt", "15m", datetime(2026, 4, 27, tzinfo=UTC).date())
        == "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2026-04-27.zip"
    )


def test_binance_kline_parser_exposes_taker_flow() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "BTCUSDT-15m.csv",
            "1767225600000,100,101,99,100.5,10,1767226499999,1005,4,6,603,0\n",
        )

    bars = parse_kline_zip(payload.getvalue(), symbol="BTCUSDT", interval="15m")

    assert bars[0].extra["taker_buy_base_volume"] == 6
    assert bars[0].extra["taker_buy_quote_volume"] == 603


def test_verify_sha256() -> None:
    payload = b"zeroalpha"
    checksum = "ee126f262a24ef803938a1e7836e78a332cb9864ababc826e1a861c1ba8a8eac  file.zip"
    assert verify_sha256(payload, checksum)


def test_coinbase_windows_are_capped_at_300_candles() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    windows = candle_windows(start, end, 60)
    assert len(windows) == 5
    assert all((b - a).total_seconds() <= 18_000 for a, b in windows)


def test_coinbase_candles_url_uses_iso_utc() -> None:
    client = CoinbaseExchangeClient()
    url = client.candles_url(
        "BTC-USD",
        3600,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert "granularity=3600" in url
    assert "start=2026-01-01T00%3A00%3A00Z" in url


def test_kraken_zip_parser_and_missing_intervals() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "XBTUSD_60.csv",
            "1767225600,100,101,99,100.5,10,4\n1767232800,102,103,101,102.5,12,5\n",
        )
    bars = parse_ohlcvt_zip(payload.getvalue(), symbol="XBT/USD", interval_minutes=60)
    assert len(bars) == 2
    assert bars[0].trade_count == 4
    assert len(missing_ohlcvt_intervals(bars, interval_minutes=60)) == 1
