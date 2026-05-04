"""Runtime event streaming for broker and strategy runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
import json
import sys
import time
import uuid


def _json_default(value: object) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    return str(value)


def _compact_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, default=_json_default, separators=(",", ":"))
    return str(value)


@dataclass(slots=True)
class RuntimeEventStream:
    """Emit human-readable progress and machine-readable JSONL runtime events."""

    run_name: str
    run_id: str = ""
    console: bool = True
    console_format: str = "text"
    output_path: Path | None = None
    stream: TextIO | None = None
    _started_monotonic: float = field(default=0.0, init=False, repr=False)
    _handle: TextIO | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]
        if self.console_format not in {"text", "json"}:
            raise ValueError("console_format must be text or json")
        self._started_monotonic = time.monotonic()
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("a", encoding="utf-8")

    def emit(self, event: str, message: str = "", **fields: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self._started_monotonic, 3),
            "run_id": self.run_id,
            "run_name": self.run_name,
            "event": event,
        }
        if message:
            payload["message"] = message
        payload.update({key: value for key, value in fields.items() if value is not None})
        line = json.dumps(payload, sort_keys=True, default=_json_default)
        if self._handle is not None:
            self._handle.write(line)
            self._handle.write("\n")
            self._handle.flush()
        if self.console:
            target = self.stream or sys.stderr
            if self.console_format == "json":
                print(line, file=target, flush=True)
            else:
                print(self._format_text(payload), file=target, flush=True)
        return payload

    def _format_text(self, payload: dict[str, object]) -> str:
        timestamp = str(payload["timestamp_utc"])
        event = str(payload["event"])
        message = str(payload.get("message", ""))
        skip = {"timestamp_utc", "elapsed_seconds", "run_id", "run_name", "event", "message"}
        details = " ".join(
            f"{key}={_compact_value(payload[key])}"
            for key in sorted(payload)
            if key not in skip
        )
        pieces = [timestamp, event]
        if message:
            pieces.append(message)
        if details:
            pieces.append(details)
        return " | ".join(pieces)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> RuntimeEventStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
