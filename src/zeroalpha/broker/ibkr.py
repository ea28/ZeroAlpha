"""IBKR Gateway adapter.

The adapter keeps `ib_async` optional so core tests can run before Gateway and
broker dependencies are installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from zeroalpha.config import AppConfig, ConfigError
from zeroalpha.domain import Bar, MarketQuote, OrderIntent, OrderType, RuntimeMode
from zeroalpha.broker.pacing import HistoricalPacingGuard
from zeroalpha.timeutils import ensure_utc, utc_now


class BrokerDependencyError(RuntimeError):
    pass


class BrokerConnectionError(RuntimeError):
    pass


def _ib_async() -> Any:
    try:
        import ib_async  # type: ignore
    except ImportError as exc:
        raise BrokerDependencyError("install zeroalpha[broker] to use IBKR Gateway") from exc
    return ib_async


@dataclass(slots=True)
class QualifiedCryptoContract:
    symbol: str
    currency: str
    exchange: str
    con_id: int | None
    raw: Any


class IBKRBroker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.validate()
        self._ib: Any | None = None
        self._read_only = config.broker.read_only
        self.pacing = HistoricalPacingGuard()

    @property
    def connected(self) -> bool:
        return bool(self._ib and self._ib.isConnected())

    def _validate_runtime_for_orders(self) -> None:
        if self.config.runtime.mode == RuntimeMode.RESEARCH:
            raise ConfigError("research mode cannot connect to broker")
        if self.config.runtime.mode == RuntimeMode.SHADOW and not self.config.broker.read_only:
            raise ConfigError("shadow mode must be read-only")

    async def connect(self, *, read_only: bool | None = None) -> None:
        self._validate_runtime_for_orders()
        ib_async = _ib_async()
        read_only = self.config.broker.read_only if read_only is None else read_only
        self._read_only = read_only
        self._ib = ib_async.IB()
        await self._ib.connectAsync(
            self.config.broker.host,
            self.config.broker.port,
            clientId=self.config.broker.client_id,
            readonly=read_only,
            account=self.config.broker.account or None,
        )

    async def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()

    async def qualify_crypto_contract(self) -> QualifiedCryptoContract:
        if not self._ib:
            raise BrokerConnectionError("connect before qualifying contracts")
        ib_async = _ib_async()
        last_error: Exception | None = None
        for exchange in self.config.broker.crypto_exchanges:
            contract = ib_async.Contract(
                symbol=self.config.contract.symbol,
                secType=self.config.contract.security_type,
                currency=self.config.contract.currency,
                exchange=exchange,
            )
            try:
                qualified = await self._ib.qualifyContractsAsync(contract)
            except Exception as exc:  # pragma: no cover - depends on Gateway
                last_error = exc
                continue
            if qualified:
                contract = qualified[0]
                return QualifiedCryptoContract(
                    symbol=self.config.contract.symbol,
                    currency=self.config.contract.currency,
                    exchange=exchange,
                    con_id=getattr(contract, "conId", None),
                    raw=contract,
                )
        raise BrokerConnectionError(f"could not qualify BTC crypto contract: {last_error}")

    async def snapshot_quote(self, contract: QualifiedCryptoContract) -> MarketQuote:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting quotes")
        ticker = self._ib.reqMktData(contract.raw, "", False, False)
        bid = ask = None
        for _ in range(20):
            await asyncio.sleep(0.25)
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            if bid and ask and float(bid) > 0 and float(ask) >= float(bid):
                break
        else:
            self._ib.cancelMktData(contract.raw)
            raise BrokerConnectionError("did not receive a usable bid/ask quote")
        bid = float(ticker.bid)
        ask = float(ticker.ask)
        self._ib.cancelMktData(contract.raw)
        now = utc_now()
        return MarketQuote(
            timestamp_utc=now,
            received_timestamp_utc=now,
            symbol=f"{contract.symbol}/{contract.currency}",
            bid=bid,
            ask=ask,
            source="IBKR",
            bid_size=getattr(ticker, "bidSize", None),
            ask_size=getattr(ticker, "askSize", None),
            market_data_type=self.config.broker.market_data_type,
        )

    async def historical_bars(
        self,
        contract: QualifiedCryptoContract,
        *,
        end: datetime,
        duration: str,
        bar_size: str,
        what_to_show: str,
    ) -> list[Bar]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting historical bars")
        key = (contract.symbol, contract.exchange, what_to_show)
        weight = 2 if what_to_show == "BID_ASK" else 1
        self.pacing.check(key, weight=weight)
        rows = await self._ib.reqHistoricalDataAsync(
            contract.raw,
            endDateTime=ensure_utc(end),
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        bars: list[Bar] = []
        for row in rows:
            bars.append(
                Bar(
                    timestamp_utc=ensure_utc(row.date),
                    symbol=f"{contract.symbol}/{contract.currency}",
                    bar_size=bar_size,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume or 0),
                    vwap=float(row.average) if getattr(row, "average", None) else None,
                    trade_count=int(row.barCount) if getattr(row, "barCount", None) else None,
                    source=f"IBKR:{what_to_show}",
                )
            )
        return bars

    def _order_from_intent(
        self,
        intent: OrderIntent,
        *,
        parent_order_ids: dict[str, int] | None = None,
    ) -> Any:
        ib_async = _ib_async()
        order = ib_async.Order()
        order.action = intent.side.value
        order.orderType = intent.order_type.value
        order.tif = intent.time_in_force.value
        order.transmit = intent.transmit
        if intent.order_type == OrderType.MKT:
            order.tif = "IOC"
        if intent.limit_price is not None:
            order.lmtPrice = intent.limit_price
        if intent.aux_price is not None:
            order.auxPrice = intent.aux_price
        if intent.cash_qty is not None:
            order.cashQty = intent.cash_qty
        if intent.quantity is not None:
            order.totalQuantity = intent.quantity
        if intent.parent_internal_order_id and parent_order_ids:
            parent_id = parent_order_ids.get(intent.parent_internal_order_id)
            if parent_id is not None:
                order.parentId = parent_id
        if intent.order_type == OrderType.MKT and intent.side.value == "BUY" and intent.cash_qty is None:
            raise ValueError("IBKR crypto BUY market orders require cash_qty")
        return order

    def place_order_intent(self, contract: QualifiedCryptoContract, intent: OrderIntent) -> Any:
        if not self._ib:
            raise BrokerConnectionError("connect before placing orders")
        if self._read_only:
            raise ConfigError("broker connection is read-only; refusing to place order")
        if self.config.runtime.mode in {RuntimeMode.RESEARCH, RuntimeMode.SHADOW}:
            raise ConfigError(f"{self.config.runtime.mode.value} mode cannot submit orders")
        order = self._order_from_intent(intent)
        return self._ib.placeOrder(contract.raw, order)

    def place_order_intents(self, contract: QualifiedCryptoContract, intents: list[OrderIntent]) -> list[Any]:
        if not self._ib:
            raise BrokerConnectionError("connect before placing orders")
        if self._read_only:
            raise ConfigError("broker connection is read-only; refusing to place order")
        if self.config.runtime.mode in {RuntimeMode.RESEARCH, RuntimeMode.SHADOW}:
            raise ConfigError(f"{self.config.runtime.mode.value} mode cannot submit orders")
        parent_order_ids: dict[str, int] = {}
        trades: list[Any] = []
        for intent in intents:
            order = self._order_from_intent(intent, parent_order_ids=parent_order_ids)
            trade = self._ib.placeOrder(contract.raw, order)
            trades.append(trade)
            trade_order = getattr(trade, "order", order)
            order_id = getattr(trade_order, "orderId", None)
            if isinstance(order_id, int):
                parent_order_ids[intent.internal_order_id] = order_id
        return trades

    async def wait(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def cancel_trade(self, trade: Any) -> Any:
        if not self._ib:
            raise BrokerConnectionError("connect before canceling orders")
        order = getattr(trade, "order", trade)
        return self._ib.cancelOrder(order)
