from datetime import UTC, datetime

from zeroalpha.domain import CandidateEvent, Side
from zeroalpha.models.dataset import _add_event_metadata


def test_event_metadata_preserves_setup_family_for_specialist_routing() -> None:
    event = CandidateEvent(
        event_id="event-1",
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        candidate_type="active_liquidity_reversal",
        side=Side.BUY,
        bar_size="15m",
        signal_strength=1.0,
        reference_price=100.0,
        max_holding_hours=4,
        metadata={"setup_family": "liquidation_reversal", "volume_ratio": 1.2},
    )
    features: dict[str, float | str] = {}

    _add_event_metadata(features, event)

    assert features["event_setup_family"] == "liquidation_reversal"
    assert features["event_volume_ratio"] == 1.2
