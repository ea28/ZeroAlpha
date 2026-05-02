"""Order intent factories and synthetic exit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from zeroalpha.domain import OrderIntent, OrderType, Side, TimeInForce


def new_order_id(prefix: str = "za") -> str:
    return f"{prefix}_{uuid4().hex}"


def _opposite_side(side: Side) -> Side:
    return Side.SELL if side == Side.BUY else Side.BUY


class CryptoOrderFactory:
    @staticmethod
    def limit_entry(
        event_id: str,
        symbol: str,
        quantity: float,
        limit_price: float,
        *,
        side: Side = Side.BUY,
        transmit: bool = True,
    ) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("entry"),
            event_id=event_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LMT,
            time_in_force=TimeInForce.MINUTES,
            quantity=quantity,
            limit_price=limit_price,
            transmit=transmit,
            reason="normal_limit_entry",
        )

    @staticmethod
    def limit_exit(
        event_id: str | None,
        symbol: str,
        quantity: float,
        limit_price: float,
        *,
        side: Side = Side.SELL,
        parent_internal_order_id: str | None = None,
        transmit: bool = True,
    ) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("exit"),
            event_id=event_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LMT,
            time_in_force=TimeInForce.MINUTES,
            quantity=quantity,
            limit_price=limit_price,
            parent_internal_order_id=parent_internal_order_id,
            transmit=transmit,
            reason="normal_limit_exit",
        )

    @staticmethod
    def stop_loss_exit(
        event_id: str | None,
        symbol: str,
        quantity: float,
        stop_price: float,
        *,
        side: Side = Side.SELL,
        parent_internal_order_id: str | None = None,
        transmit: bool = True,
    ) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("stop"),
            event_id=event_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.STP,
            time_in_force=TimeInForce.GTC,
            quantity=quantity,
            aux_price=stop_price,
            parent_internal_order_id=parent_internal_order_id,
            transmit=transmit,
            reason="attached_stop_loss",
        )

    @staticmethod
    def bracket_entry(
        event_id: str,
        symbol: str,
        quantity: float,
        limit_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        *,
        side: Side = Side.BUY,
    ) -> tuple[OrderIntent, OrderIntent, OrderIntent]:
        parent = CryptoOrderFactory.limit_entry(
            event_id,
            symbol,
            quantity,
            limit_price,
            side=side,
            transmit=False,
        )
        exit_side = _opposite_side(side)
        take_profit = CryptoOrderFactory.limit_exit(
            event_id,
            symbol,
            quantity,
            take_profit_price,
            side=exit_side,
            parent_internal_order_id=parent.internal_order_id,
            transmit=False,
        )
        stop_loss = CryptoOrderFactory.stop_loss_exit(
            event_id,
            symbol,
            quantity,
            stop_loss_price,
            side=exit_side,
            parent_internal_order_id=parent.internal_order_id,
            transmit=True,
        )
        return parent, take_profit, stop_loss

    @staticmethod
    def market_buy_cash(event_id: str, symbol: str, cash_qty: float) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("mktbuy"),
            event_id=event_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MKT,
            time_in_force=TimeInForce.IOC,
            cash_qty=cash_qty,
            reason="explicit_market_buy_cash",
        )

    @staticmethod
    def urgent_market_exit(
        symbol: str,
        quantity: float,
        reason: str,
        *,
        side: Side = Side.SELL,
    ) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("urgent"),
            event_id=None,
            symbol=symbol,
            side=side,
            order_type=OrderType.MKT,
            time_in_force=TimeInForce.IOC,
            quantity=quantity,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class SyntheticExitLevels:
    entry_price: float
    profit_target: float
    stop_distance: float

    @property
    def profit_price(self) -> float:
        return self.entry_price * (1 + self.profit_target)

    @property
    def stop_price(self) -> float:
        return self.entry_price * (1 - self.stop_distance)

    def triggered(self, bid: float) -> str | None:
        if bid >= self.profit_price:
            return "profit_target"
        if bid <= self.stop_price:
            return "stop_loss"
        return None
