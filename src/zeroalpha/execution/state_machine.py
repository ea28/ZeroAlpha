"""Explicit bot state machine."""

from __future__ import annotations

from dataclasses import dataclass

from zeroalpha.domain import BotState


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

    def transition(self, new_state: BotState) -> None:
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition {self.state.value} -> {new_state.value}")
        self.state = new_state
