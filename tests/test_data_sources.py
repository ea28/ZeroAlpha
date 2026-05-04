from datetime import datetime, UTC
from io import BytesIO
import zipfile

from zeroalpha.data.external.binance import (
    BinanceFuturesMetricsClient,
    BinancePublicDataClient,
    parse_kline_zip,
    verify_sha256,
)
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

    assert bars[0].timestamp_utc == datetime(2026, 1, 1, 0, 15, tzinfo=UTC)
    assert bars[0].extra["bar_start_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
    assert bars[0].extra["taker_buy_base_volume"] == 6
    assert bars[0].extra["taker_buy_quote_volume"] == 603


def test_binance_futures_metrics_urls_and_parsers() -> None:
    class FakeMetricsClient(BinanceFuturesMetricsClient):
        def get_json(self, url: str, timeout: float = 30.0):  # type: ignore[no-untyped-def]
            if "fundingRate" in url:
                return [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingTime": 1767225600000,
                        "markPrice": "100.0",
                    }
                ]
            if "takerlongshortRatio" in url:
                return [
                    {
                        "symbol": "BTCUSDT",
                        "buySellRatio": "1.5",
                        "buyVol": "15.0",
                        "sellVol": "10.0",
                        "timestamp": "1767225600000",
                    }
                ]
            if "/basis" in url:
                return [
                    {
                        "pair": "BTCUSDT",
                        "basis": "12.5",
                        "basisRate": "0.0002",
                        "futuresPrice": "100012.5",
                        "indexPrice": "100000.0",
                        "annualizedBasisRate": "0.012",
                        "timestamp": "1767225600000",
                    }
                ]
            return [
                {
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "10.0",
                    "sumOpenInterestValue": "1000.0",
                    "timestamp": "1767225600000",
                }
            ]

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    client = FakeMetricsClient()

    assert "/fapi/v1/fundingRate?" in client.funding_rate_url("btcusdt", start, end)
    assert "/futures/data/openInterestHist?" in client.open_interest_hist_url("btcusdt", "5m", start, end)
    assert "/futures/data/takerlongshortRatio?" in client.taker_buy_sell_volume_url("btcusdt", "5m", start, end)
    assert "/futures/data/basis?" in client.basis_url("btcusdt", "5m", start, end)
    funding = client.fetch_funding_rates(symbol="BTCUSDT", start=start, end=end)
    open_interest = client.fetch_open_interest_history(symbol="BTCUSDT", period="5m", start=start, end=end)
    taker_flow = client.fetch_taker_buy_sell_volume(symbol="BTCUSDT", period="5m", start=start, end=end)
    basis = client.fetch_basis_history(pair="BTCUSDT", period="5m", start=start, end=end)

    assert funding[0].source == "BINANCE_UM_FUNDING"
    assert funding[0].extra["funding_rate"] == 0.0001
    assert funding[0].close == 1.0001
    assert open_interest[0].source == "BINANCE_UM_OPEN_INTEREST"
    assert open_interest[0].close == 1000.0
    assert open_interest[0].extra["open_interest"] == 10.0
    assert taker_flow[0].source == "BINANCE_UM_TAKERFLOW"
    assert taker_flow[0].extra["taker_buy_base_volume"] == 15.0
    assert taker_flow[0].volume == 25.0
    assert basis[0].source == "BINANCE_UM_BASIS"
    assert basis[0].close == 0.0002
    assert basis[0].extra["futures_price"] == 100012.5


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


def test_coinbase_candles_use_completed_bar_timestamps(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def read(self) -> bytes:
            return b"[[1767225600,99,101,100,100.5,10]]"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    bars = CoinbaseExchangeClient().fetch_candles(
        "BTC-USD",
        3600,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 1, tzinfo=UTC),
    )

    assert bars[0].timestamp_utc == datetime(2026, 1, 1, 1, tzinfo=UTC)
    assert bars[0].extra["bar_start_timestamp_utc"] == "2026-01-01T00:00:00+00:00"


def test_spot_orderbook_urls_cover_binance_and_coinbase() -> None:
    binance = BinancePublicDataClient()
    coinbase = CoinbaseExchangeClient()

    assert binance.spot_depth_url("btcusdt", limit=500).endswith("/api/v3/depth?symbol=BTCUSDT&limit=500")
    assert binance.spot_book_ticker_url("btcusdt").endswith("/api/v3/ticker/bookTicker?symbol=BTCUSDT")
    assert coinbase.product_book_url("BTC-USD", level=2).endswith("/products/BTC-USD/book?level=2")
    assert coinbase.product_trades_url("BTC-USD", limit=50).endswith("/products/BTC-USD/trades?limit=50")


def test_kraken_zip_parser_and_missing_intervals() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "XBTUSD_60.csv",
            "1767225600,100,101,99,100.5,10,4\n1767232800,102,103,101,102.5,12,5\n",
        )
    bars = parse_ohlcvt_zip(payload.getvalue(), symbol="XBT/USD", interval_minutes=60)
    assert len(bars) == 2
    assert bars[0].timestamp_utc == datetime(2026, 1, 1, 1, tzinfo=UTC)
    assert bars[0].extra["bar_start_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
    assert bars[0].trade_count == 4
    assert len(missing_ohlcvt_intervals(bars, interval_minutes=60)) == 1
