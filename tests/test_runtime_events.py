import io
import json

from zeroalpha.monitoring.events import RuntimeEventStream
from zeroalpha.domain import BotState
from zeroalpha.execution.state_machine import StateMachine


def test_runtime_event_stream_writes_jsonl_and_text(tmp_path) -> None:
    event_log = tmp_path / "events.jsonl"
    console = io.StringIO()

    with RuntimeEventStream(
        run_name="test.run",
        run_id="run123",
        output_path=event_log,
        stream=console,
    ) as events:
        payload = events.emit("order.submitted", "sent order", order_id=7, notional=100.0)

    assert payload["event"] == "order.submitted"
    assert "order.submitted" in console.getvalue()
    assert "order_id=7" in console.getvalue()

    rows = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert rows == [
        {
            **rows[0],
            "run_id": "run123",
            "run_name": "test.run",
            "event": "order.submitted",
            "message": "sent order",
            "order_id": 7,
            "notional": 100.0,
        }
    ]


def test_runtime_event_stream_can_emit_json_console() -> None:
    console = io.StringIO()
    events = RuntimeEventStream(
        run_name="test.run",
        run_id="run123",
        console_format="json",
        output_path=None,
        stream=console,
    )
    try:
        events.emit("account.snapshot", equity=10_000.0)
    finally:
        events.close()

    payload = json.loads(console.getvalue())
    assert payload["event"] == "account.snapshot"
    assert payload["equity"] == 10_000.0


def test_state_machine_streams_transitions() -> None:
    console = io.StringIO()
    events = RuntimeEventStream(
        run_name="bot",
        run_id="run123",
        console_format="json",
        output_path=None,
        stream=console,
    )
    try:
        machine = StateMachine(events=events)
        machine.transition(BotState.IDLE, reason="startup_complete")
    finally:
        events.close()

    payload = json.loads(console.getvalue())
    assert payload["event"] == "bot.state_transition"
    assert payload["from_state"] == "INITIALIZING"
    assert payload["to_state"] == "IDLE"
    assert payload["reason"] == "startup_complete"
