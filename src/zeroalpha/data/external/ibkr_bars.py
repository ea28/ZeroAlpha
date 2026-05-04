"""IBKR historical bar JSONL helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

from zeroalpha.domain import Bar
from zeroalpha.timeutils import ensure_utc


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO datetime string")
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def bar_to_json_dict(bar: Bar) -> dict[str, Any]:
    payload = asdict(bar)
    payload["timestamp_utc"] = bar.timestamp_utc.isoformat()
    return payload


def bar_from_json_dict(payload: dict[str, Any]) -> Bar:
    extra = payload.get("extra")
    return Bar(
        timestamp_utc=_parse_timestamp(payload["timestamp_utc"]),
        symbol=str(payload["symbol"]),
        bar_size=str(payload["bar_size"]),
        open=float(payload["open"]),
        high=float(payload["high"]),
        low=float(payload["low"]),
        close=float(payload["close"]),
        volume=float(payload.get("volume", 0.0) or 0.0),
        quote_volume=(
            float(payload["quote_volume"]) if payload.get("quote_volume") is not None else None
        ),
        trade_count=(
            int(payload["trade_count"]) if payload.get("trade_count") is not None else None
        ),
        vwap=float(payload["vwap"]) if payload.get("vwap") is not None else None,
        source=str(payload.get("source") or "IBKR"),
        extra=extra if isinstance(extra, dict) else {},
    )


def write_ibkr_bars(path: Path, bars: Sequence[Bar]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for bar in bars:
            handle.write(json.dumps(bar_to_json_dict(bar), sort_keys=True) + "\n")
    return len(bars)


def read_ibkr_bars(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    bars.append(bar_from_json_dict(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return sorted(bars, key=lambda bar: bar.timestamp_utc)
