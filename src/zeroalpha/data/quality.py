"""Data quality checks before features, labels, or trading decisions."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

from zeroalpha.domain import Bar, MarketQuote
from zeroalpha.timeutils import ensure_utc, utc_now


@dataclass(frozen=True, slots=True)
class DataIssue:
    code: str
    message: str
    timestamp_utc: datetime | None = None
    symbol: str | None = None


@dataclass(slots=True)
class DataQualityReport:
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(
        self,
        code: str,
        message: str,
        timestamp_utc: datetime | None = None,
        symbol: str | None = None,
    ) -> None:
        self.issues.append(
            DataIssue(
                code=code,
                message=message,
                timestamp_utc=ensure_utc(timestamp_utc) if timestamp_utc else None,
                symbol=symbol,
            )
        )

    def extend(self, other: "DataQualityReport") -> None:
        self.issues.extend(other.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "issue_count": len(self.issues),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "timestamp_utc": issue.timestamp_utc.isoformat() if issue.timestamp_utc else None,
                    "symbol": issue.symbol,
                }
                for issue in self.issues
            ],
        }


def interval_to_timedelta(interval: str) -> timedelta:
    """Parse common crypto bar intervals such as 1m, 3600s, 4h, or 1d."""

    match = re.fullmatch(r"(\d+)(s|m|h|d)", interval.strip().lower())
    if not match:
        raise ValueError(f"unsupported interval {interval!r}")
    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        raise ValueError("interval value must be positive")
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(days=value)


def validate_bars(
    bars: list[Bar],
    *,
    now: datetime | None = None,
    expected_interval: str | timedelta | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    minimum_coverage_ratio: float = 0.0,
    max_gap_intervals: float = 1.5,
    max_return_bps: float | None = None,
) -> DataQualityReport:
    report = DataQualityReport()
    now = ensure_utc(now or utc_now())
    if not 0 <= minimum_coverage_ratio <= 1:
        raise ValueError("minimum_coverage_ratio must be in [0, 1]")
    if max_gap_intervals <= 0:
        raise ValueError("max_gap_intervals must be positive")
    interval = (
        interval_to_timedelta(expected_interval)
        if isinstance(expected_interval, str)
        else expected_interval
    )
    if not bars:
        report.add("empty_bars", "bar collection is empty")
        return report
    seen: set[tuple[str, str, datetime, str]] = set()
    groups: dict[tuple[str, str, str], list[Bar]] = {}
    for bar in bars:
        key = (bar.source, bar.symbol, bar.timestamp_utc, bar.bar_size)
        if key in seen:
            report.add("duplicate_bar", "duplicate source/symbol/timestamp/bar_size", bar.timestamp_utc, bar.symbol)
        seen.add(key)
        groups.setdefault((bar.source, bar.symbol, bar.bar_size), []).append(bar)
        if bar.timestamp_utc > now:
            report.add("future_bar", "bar timestamp is in the future", bar.timestamp_utc, bar.symbol)
        if bar.volume < 0:
            report.add("negative_volume", "bar volume is negative", bar.timestamp_utc, bar.symbol)
        if bar.low > min(bar.open, bar.close, bar.high) or bar.high < max(bar.open, bar.close, bar.low):
            report.add("invalid_ohlc", "OHLC values are internally inconsistent", bar.timestamp_utc, bar.symbol)
    if interval is not None:
        interval_seconds = interval.total_seconds()
        for (_, symbol, _), group in groups.items():
            ordered = sorted(group, key=lambda row: row.timestamp_utc)
            for previous, current in zip(ordered, ordered[1:], strict=False):
                delta = current.timestamp_utc - previous.timestamp_utc
                if delta.total_seconds() > interval_seconds * max_gap_intervals:
                    missing = max(0, round(delta.total_seconds() / interval_seconds) - 1)
                    report.add(
                        "bar_gap",
                        f"gap of {delta} between bars; approximately {missing} missing interval(s)",
                        current.timestamp_utc,
                        symbol,
                    )
                if max_return_bps is not None and previous.close > 0:
                    return_bps = abs(current.close / previous.close - 1) * 10_000
                    if return_bps > max_return_bps:
                        report.add(
                            "extreme_return",
                            f"close-to-close move {return_bps:.1f} bps exceeds {max_return_bps:.1f} bps",
                            current.timestamp_utc,
                            symbol,
                        )
            if start is not None and end is not None and minimum_coverage_ratio > 0:
                start_utc = ensure_utc(start)
                end_utc = ensure_utc(end)
                expected = max(1, int((end_utc - start_utc).total_seconds() // interval_seconds))
                actual = sum(1 for bar in ordered if start_utc <= bar.timestamp_utc < end_utc)
                coverage = actual / expected
                if coverage < minimum_coverage_ratio:
                    report.add(
                        "insufficient_coverage",
                        f"coverage {coverage:.3f} below required {minimum_coverage_ratio:.3f}",
                        ordered[-1].timestamp_utc if ordered else start_utc,
                        symbol,
                    )
    return report


def validate_source_divergence(
    primary_bars: list[Bar],
    reference_bars: list[Bar],
    *,
    max_divergence_bps: float,
    expected_interval: str | timedelta | None = None,
    max_stale_intervals: float = 2.0,
) -> DataQualityReport:
    """Compare primary bars against an independent reference close series.

    The reference is sampled at or before the primary timestamp. This avoids
    lookahead while still catching bad files, symbol mixups, and stale reference
    feeds before model training.
    """

    report = DataQualityReport()
    if max_divergence_bps <= 0:
        raise ValueError("max_divergence_bps must be positive")
    if max_stale_intervals <= 0:
        raise ValueError("max_stale_intervals must be positive")
    if not primary_bars or not reference_bars:
        report.add("missing_reference", "source divergence check requires primary and reference bars")
        return report
    interval = (
        interval_to_timedelta(expected_interval)
        if isinstance(expected_interval, str)
        else expected_interval
    )
    max_stale = interval * max_stale_intervals if interval is not None else None
    reference = sorted(reference_bars, key=lambda row: row.timestamp_utc)
    reference_timestamps = [bar.timestamp_utc for bar in reference]
    for primary in sorted(primary_bars, key=lambda row: row.timestamp_utc):
        idx = bisect_right(reference_timestamps, primary.timestamp_utc) - 1
        if idx < 0:
            report.add("stale_reference", "no reference bar available before primary timestamp", primary.timestamp_utc, primary.symbol)
            continue
        ref = reference[idx]
        if max_stale is not None and primary.timestamp_utc - ref.timestamp_utc > max_stale:
            report.add(
                "stale_reference",
                f"reference bar is stale by {primary.timestamp_utc - ref.timestamp_utc}",
                primary.timestamp_utc,
                primary.symbol,
            )
            continue
        divergence_bps = abs(primary.close / ref.close - 1) * 10_000
        if divergence_bps > max_divergence_bps:
            report.add(
                "source_divergence",
                f"primary/reference close divergence {divergence_bps:.1f} bps exceeds {max_divergence_bps:.1f} bps",
                primary.timestamp_utc,
                primary.symbol,
            )
    return report


def validate_quotes(quotes: list[MarketQuote], *, now: datetime | None = None) -> DataQualityReport:
    report = DataQualityReport()
    now = ensure_utc(now or utc_now())
    for quote in quotes:
        if quote.timestamp_utc > now:
            report.add("future_quote", "quote timestamp is in the future", quote.timestamp_utc, quote.symbol)
        if quote.ask < quote.bid:
            report.add("negative_spread", "ask is below bid", quote.timestamp_utc, quote.symbol)
        if quote.bid <= 0 or quote.ask <= 0:
            report.add("nonpositive_quote", "bid/ask must be positive", quote.timestamp_utc, quote.symbol)
    return report
