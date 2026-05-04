"""Kraken downloadable OHLCVT helpers."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO, StringIO, TextIOWrapper
import csv
import zipfile

from zeroalpha.domain import Bar
from zeroalpha.timeutils import parse_unix_timestamp


def parse_ohlcvt_csv(text: str, *, symbol: str, interval_minutes: int) -> list[Bar]:
    bars: list[Bar] = []
    reader = csv.reader(StringIO(text))
    for fields in reader:
        if not fields or fields[0].lower() in {"time", "timestamp"}:
            continue
        bar_start = parse_unix_timestamp(int(fields[0]))
        bar_close = bar_start + timedelta(minutes=interval_minutes)
        bars.append(
            Bar(
                timestamp_utc=bar_close,
                symbol=symbol,
                bar_size=f"{interval_minutes}m",
                open=float(fields[1]),
                high=float(fields[2]),
                low=float(fields[3]),
                close=float(fields[4]),
                volume=float(fields[5]),
                trade_count=int(float(fields[6])) if len(fields) > 6 and fields[6] else None,
                source="KRAKEN",
                extra={
                    "bar_start_timestamp_utc": bar_start.isoformat(),
                    "bar_close_timestamp_utc": bar_close.isoformat(),
                },
            )
        )
    return bars


def parse_ohlcvt_zip(payload: bytes, *, symbol: str, interval_minutes: int) -> list[Bar]:
    rows: list[Bar] = []
    interval_token = str(interval_minutes)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.endswith(".csv") and interval_token in name.rsplit("/", 1)[-1]
        ]
        if not csv_names:
            raise ValueError(f"zip payload does not contain interval {interval_minutes} CSV data")
        for name in csv_names:
            with archive.open(name) as raw:
                text = TextIOWrapper(raw, encoding="utf-8").read()
                rows.extend(parse_ohlcvt_csv(text, symbol=symbol, interval_minutes=interval_minutes))
    deduped = {bar.timestamp_utc: bar for bar in rows}
    return sorted(deduped.values(), key=lambda bar: bar.timestamp_utc)


def missing_ohlcvt_intervals(bars: list[Bar], *, interval_minutes: int) -> list[tuple[Bar, Bar]]:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    expected = timedelta(minutes=interval_minutes)
    gaps: list[tuple[Bar, Bar]] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.timestamp_utc - previous.timestamp_utc > expected:
            gaps.append((previous, current))
    return gaps
