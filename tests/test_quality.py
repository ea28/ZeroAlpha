from datetime import datetime, UTC

from zeroalpha.data.quality import validate_bars, validate_source_divergence
from zeroalpha.domain import Bar


def test_duplicate_bar_detection() -> None:
    bar = Bar(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        bar_size="1h",
        open=100,
        high=110,
        low=90,
        close=105,
        volume=1,
        source="BINANCE",
    )
    report = validate_bars([bar, bar])
    assert not report.ok
    assert report.issues[0].code == "duplicate_bar"


def test_bar_gap_and_coverage_detection() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            timestamp_utc=start,
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
        Bar(
            timestamp_utc=start.replace(hour=3),
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="BINANCE",
        ),
    ]

    report = validate_bars(
        bars,
        expected_interval="1h",
        start=start,
        end=start.replace(hour=6),
        minimum_coverage_ratio=0.75,
    )

    assert not report.ok
    assert {issue.code for issue in report.issues} >= {"bar_gap", "insufficient_coverage"}


def test_source_divergence_detection_uses_latest_causal_reference() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    primary = [
        Bar(
            timestamp_utc=start,
            symbol="BTCUSDT",
            bar_size="1h",
            open=100,
            high=111,
            low=99,
            close=110,
            volume=1,
            source="BINANCE",
        )
    ]
    reference = [
        Bar(
            timestamp_utc=start,
            symbol="BTC-USD",
            bar_size="3600s",
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            source="COINBASE",
        )
    ]

    report = validate_source_divergence(
        primary,
        reference,
        expected_interval="1h",
        max_divergence_bps=500,
    )

    assert not report.ok
    assert report.issues[0].code == "source_divergence"
