"""IBKR historical data pacing guard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from zeroalpha.timeutils import ensure_utc, utc_now


class PacingViolation(RuntimeError):
    pass


@dataclass(slots=True)
class HistoricalPacingGuard:
    duplicate_window: timedelta = timedelta(seconds=15)
    burst_window: timedelta = timedelta(seconds=2)
    total_window: timedelta = timedelta(minutes=10)
    max_same_key_in_burst: int = 5
    max_total: int = 60
    _requests: deque[tuple[datetime, tuple[str, str, str]]] = field(default_factory=deque)

    def check(self, key: tuple[str, str, str], now: datetime | None = None, weight: int = 1) -> None:
        now = ensure_utc(now or utc_now())
        while self._requests and now - self._requests[0][0] > self.total_window:
            self._requests.popleft()

        for ts, previous_key in reversed(self._requests):
            if previous_key == key and now - ts < self.duplicate_window:
                raise PacingViolation("identical historical data request within 15 seconds")

        same_key_recent = sum(
            1
            for ts, previous_key in self._requests
            if previous_key == key and now - ts <= self.burst_window
        )
        if same_key_recent + weight > self.max_same_key_in_burst:
            raise PacingViolation("too many requests for same contract/exchange/tick type")

        if len(self._requests) + weight > self.max_total:
            raise PacingViolation("more than 60 historical requests in 10 minutes")

        for _ in range(weight):
            self._requests.append((now, key))
