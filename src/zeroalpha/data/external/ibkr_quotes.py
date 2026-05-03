"""IBKR quote-record loading for research features."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from zeroalpha.domain import MarketQuote
from zeroalpha.timeutils import ensure_utc


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO datetime string")
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def quote_record_to_market_quote(record: dict[str, Any]) -> MarketQuote:
    """Convert a JSONL quote-recorder row into the domain quote type."""

    return MarketQuote(
        timestamp_utc=_parse_timestamp(record["timestamp_utc"]),
        received_timestamp_utc=_parse_timestamp(
            record.get("received_timestamp_utc", record["timestamp_utc"])
        ),
        symbol=str(record.get("symbol") or "BTC/USD"),
        bid=float(record["bid"]),
        ask=float(record["ask"]),
        source=str(record.get("source") or "IBKR"),
        bid_size=_optional_float(record.get("bid_size")),
        ask_size=_optional_float(record.get("ask_size")),
        market_data_type=(
            str(record["market_data_type"]) if record.get("market_data_type") is not None else None
        ),
    )


def read_ibkr_quote_records(path: Path) -> list[MarketQuote]:
    """Read IBKR quote-recorder JSONL, skipping malformed rows."""

    quotes: list[MarketQuote] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    quotes.append(quote_record_to_market_quote(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return sorted(quotes, key=lambda quote: quote.timestamp_utc)
