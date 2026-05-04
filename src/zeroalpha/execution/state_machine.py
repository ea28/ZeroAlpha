"""Explicit bot state machine."""

from __future__ import annotations

from dataclasses import dataclass

from zeroalpha.domain import BotState
from zeroalpha.monitoring.events import RuntimeEventStream


ALLOWED_TRANSITIONS: dict[BotState, set[BotState]] = {
    BotState.INITIALIZING: {BotState.IDLE, BotState.DISABLED},
    BotState.IDLE: {BotState.SIGNAL_DETECTED, BotState.DISABLED},
    BotState.SIGNAL_DETECTED: {BotState.SCORING, BotState.IDLE, BotState.DISABLED},
    BotState.SCORING: {BotState.RISK_CHECK, BotState.IDLE, BotState.DISABLED},
    BotState.RISK_CHECK: {BotState.ORDER_PENDING, BotState.IDLE, BotState.DISABLED},
    BotState.ORDER_PENDING: {BotState.IN_POSITION, BotState.IDLE, BotState.EXIT_PENDING, BotState.DISABLED},
    BotState.IN_POSITION: {BotState.EXIT_PENDING, BotState.DISABLED},
    BotState.EXIT_PENDING: {BotState.IDLE, BotState.COOLDOWN, BotState.DISABLED},
    BotState.COOLDOWN: {BotState.IDLE, BotState.DISABLED},
    BotState.DISABLED: {BotState.INITIALIZING},
}


@dataclass(slots=True)
class StateMachine:
    state: BotState = BotState.INITIALIZING
    events: RuntimeEventStream | None = None

    def transition(self, new_state: BotState, *, reason: str = "") -> None:
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition {self.state.value} -> {new_state.value}")
        old_state = self.state
        self.state = new_state
        if self.events is not None:
            self.events.emit(
                "bot.state_transition",
                "bot state changed",
                from_state=old_state.value,
                to_state=new_state.value,
                reason=reason,
            )
