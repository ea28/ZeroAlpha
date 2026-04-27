from datetime import datetime, timedelta, UTC

import pytest

from zeroalpha.broker.pacing import HistoricalPacingGuard, PacingViolation


def test_duplicate_historical_request_rejected() -> None:
    guard = HistoricalPacingGuard()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    guard.check(("BTC", "PAXOS", "TRADES"), now=now)
    with pytest.raises(PacingViolation):
        guard.check(("BTC", "PAXOS", "TRADES"), now=now + timedelta(seconds=1))
