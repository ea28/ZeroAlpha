"""Order intent factories and synthetic exit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from zeroalpha.domain import OrderIntent, OrderType, Side, TimeInForce


def new_order_id(prefix: str = "za") -> str:
    return f"{prefix}_{uuid4().hex}"


class CryptoOrderFactory:
    @staticmethod
    def limit_entry(event_id: str, symbol: str, quantity: float, limit_price: float) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("entry"),
            event_id=event_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LMT,
            time_in_force=TimeInForce.MINUTES,
            quantity=quantity,
            limit_price=limit_price,
            reason="normal_limit_entry",
        )

    @staticmethod
    def limit_exit(event_id: str | None, symbol: str, quantity: float, limit_price: float) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("exit"),
            event_id=event_id,
            symbol=symbol,
            side=Side.SELL,
            order_type=OrderType.LMT,
            time_in_force=TimeInForce.MINUTES,
            quantity=quantity,
            limit_price=limit_price,
            reason="normal_limit_exit",
        )

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
    def urgent_market_exit(symbol: str, quantity: float, reason: str) -> OrderIntent:
        return OrderIntent(
            internal_order_id=new_order_id("urgent"),
            event_id=None,
            symbol=symbol,
            side=Side.SELL,
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
