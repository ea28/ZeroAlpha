"""IBKR Gateway adapter.

The adapter keeps `ib_async` optional so core tests can run before Gateway and
broker dependencies are installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any

from zeroalpha.config import AppConfig, ConfigError
from zeroalpha.domain import Bar, MarketQuote, OrderIntent, OrderType, RuntimeMode
from zeroalpha.broker.pacing import HistoricalPacingGuard
from zeroalpha.timeutils import ensure_utc, utc_now


class BrokerDependencyError(RuntimeError):
    pass


class BrokerConnectionError(RuntimeError):
    pass


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_optional_float(value: Any) -> float | None:
    converted = _optional_float(value)
    if converted is None or not math.isfinite(converted):
        return None
    return converted


def _contract_identity(contract: Any) -> dict[str, Any]:
    return {
        "con_id": getattr(contract, "conId", None),
        "symbol": getattr(contract, "symbol", ""),
        "security_type": getattr(contract, "secType", ""),
        "currency": getattr(contract, "currency", ""),
        "exchange": getattr(contract, "exchange", ""),
        "local_symbol": getattr(contract, "localSymbol", ""),
        "last_trade_date_or_contract_month": getattr(contract, "lastTradeDateOrContractMonth", ""),
        "multiplier": getattr(contract, "multiplier", ""),
    }


def _account_value_asdict(row: Any) -> dict[str, Any]:
    return {
        "account": getattr(row, "account", ""),
        "tag": getattr(row, "tag", ""),
        "value": getattr(row, "value", ""),
        "currency": getattr(row, "currency", ""),
        "model_code": getattr(row, "modelCode", ""),
    }


def _portfolio_item_asdict(row: Any) -> dict[str, Any]:
    contract = getattr(row, "contract", None)
    payload = _contract_identity(contract)
    payload.update(
        {
            "account": getattr(row, "account", ""),
            "position": _finite_optional_float(getattr(row, "position", None)),
            "market_price": _finite_optional_float(getattr(row, "marketPrice", None)),
            "market_value": _finite_optional_float(getattr(row, "marketValue", None)),
            "average_cost": _finite_optional_float(getattr(row, "averageCost", None)),
            "unrealized_pnl": _finite_optional_float(getattr(row, "unrealizedPNL", None)),
            "realized_pnl": _finite_optional_float(getattr(row, "realizedPNL", None)),
        }
    )
    return payload


def _position_asdict(row: Any) -> dict[str, Any]:
    contract = getattr(row, "contract", None)
    payload = _contract_identity(contract)
    payload.update(
        {
            "account": getattr(row, "account", ""),
            "position": _finite_optional_float(getattr(row, "position", None)),
            "average_cost": _finite_optional_float(getattr(row, "avgCost", None)),
        }
    )
    return payload


def _pnl_asdict(row: Any) -> dict[str, Any]:
    return {
        "account": getattr(row, "account", ""),
        "model_code": getattr(row, "modelCode", ""),
        "daily_pnl": _finite_optional_float(getattr(row, "dailyPnL", None)),
        "unrealized_pnl": _finite_optional_float(getattr(row, "unrealizedPnL", None)),
        "realized_pnl": _finite_optional_float(getattr(row, "realizedPnL", None)),
    }


def _historical_bar_volume(value: Any) -> float:
    volume = _optional_float(value)
    if volume is None or not math.isfinite(volume) or volume < 0:
        return 0.0
    return volume


def _bar_size_delta(bar_size: str) -> timedelta:
    parts = bar_size.strip().lower().split()
    if len(parts) < 2:
        return timedelta(0)
    try:
        value = int(parts[0])
    except ValueError:
        return timedelta(0)
    unit = parts[1].rstrip("s")
    if unit in {"sec", "second"}:
        return timedelta(seconds=value)
    if unit in {"min", "minute"}:
        return timedelta(minutes=value)
    if unit in {"hour"}:
        return timedelta(hours=value)
    if unit in {"day"}:
        return timedelta(days=value)
    return timedelta(0)


MARKET_DATA_TYPE_IDS = {
    "live": 1,
    "frozen": 2,
    "delayed": 3,
    "delayed_frozen": 4,
    "delayed-frozen": 4,
}


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

    def _validate_runtime_for_orders(self, *, read_only: bool | None = None) -> None:
        if self.config.runtime.mode == RuntimeMode.RESEARCH:
            raise ConfigError("research mode cannot connect to broker")
        effective_read_only = self.config.broker.read_only if read_only is None else read_only
        if self.config.runtime.mode == RuntimeMode.SHADOW and not effective_read_only:
            raise ConfigError("shadow mode must be read-only")

    def _configured_account(self) -> str:
        return self.config.broker.account.strip()

    def _require_configured_order_account(self) -> str:
        account = self._configured_account()
        if not account:
            raise ConfigError("broker.account is required for paper/live order submission or cancellation")
        return account

    def _validate_configured_account_is_managed(self) -> None:
        account = self._require_configured_order_account()
        accounts = self.managed_accounts()
        if account not in accounts:
            available = ", ".join(accounts) or "none"
            raise ConfigError(
                f"configured broker.account {account!r} is not in TWS managedAccounts(); "
                f"available accounts: {available}"
            )

    def _assert_can_mutate_orders(self) -> None:
        if not self._ib:
            raise BrokerConnectionError("connect before mutating orders")
        if self._read_only:
            raise ConfigError("broker connection is read-only; refusing to mutate orders")
        if self.config.runtime.mode in {RuntimeMode.RESEARCH, RuntimeMode.SHADOW}:
            raise ConfigError(f"{self.config.runtime.mode.value} mode cannot mutate orders")
        self._validate_configured_account_is_managed()

    async def connect(self, *, read_only: bool | None = None) -> None:
        ib_async = _ib_async()
        read_only = self.config.broker.read_only if read_only is None else read_only
        self._validate_runtime_for_orders(read_only=read_only)
        self._read_only = read_only
        self._ib = ib_async.IB()
        try:
            await self._ib.connectAsync(
                self.config.broker.host,
                self.config.broker.port,
                clientId=self.config.broker.client_id,
                readonly=read_only,
                account=self.config.broker.account or None,
            )
            market_data_type = self.config.broker.market_data_type.strip().lower()
            if market_data_type:
                self._ib.reqMarketDataType(MARKET_DATA_TYPE_IDS[market_data_type])
            if not read_only:
                self._validate_configured_account_is_managed()
        except Exception:
            if self._ib and self._ib.isConnected():
                self._ib.disconnect()
            raise

    async def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()

    def managed_accounts(self) -> list[str]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting accounts")
        return list(self._ib.managedAccounts())

    def resolved_account(self, account: str = "", *, require_explicit: bool = False) -> str:
        if account:
            return account
        if self._configured_account():
            return self._configured_account()
        if require_explicit:
            raise ConfigError("broker.account is required for this broker operation")
        accounts = self.managed_accounts()
        return accounts[0] if accounts else ""

    async def account_summary(
        self,
        *,
        account: str = "",
        timeout_seconds: float = 5.0,
    ) -> list[dict[str, Any]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting account summary")
        resolved = self.resolved_account(account)
        try:
            rows = await asyncio.wait_for(
                self._ib.accountSummaryAsync(resolved),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return [
                {
                    "account": resolved,
                    "tag": "account_summary_timeout",
                    "value": "",
                    "currency": "",
                    "model_code": "",
                }
            ]
        return [_account_value_asdict(row) for row in rows]

    async def refresh_account_updates(
        self,
        *,
        account: str = "",
        timeout_seconds: float = 5.0,
    ) -> None:
        if not self._ib:
            raise BrokerConnectionError("connect before refreshing account updates")
        if timeout_seconds <= 0:
            return
        try:
            await asyncio.wait_for(
                self._ib.reqAccountUpdatesAsync(self.resolved_account(account)),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return

    async def portfolio_items(
        self,
        *,
        account: str = "",
        refresh: bool = True,
        timeout_seconds: float = 5.0,
    ) -> list[dict[str, Any]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting portfolio")
        resolved = self.resolved_account(account)
        if refresh:
            await self.refresh_account_updates(account=resolved, timeout_seconds=timeout_seconds)
        return [_portfolio_item_asdict(row) for row in self._ib.portfolio(resolved)]

    async def positions(
        self,
        *,
        account: str = "",
        timeout_seconds: float = 5.0,
    ) -> list[dict[str, Any]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting positions")
        resolved = self.resolved_account(account)
        try:
            rows = await asyncio.wait_for(self._ib.reqPositionsAsync(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return [
                {
                    "account": resolved,
                    "con_id": None,
                    "symbol": "",
                    "security_type": "",
                    "currency": "",
                    "exchange": "",
                    "local_symbol": "",
                    "last_trade_date_or_contract_month": "",
                    "multiplier": "",
                    "position": None,
                    "average_cost": None,
                    "warning": "positions_timeout",
                }
            ]
        if resolved:
            rows = [row for row in rows if getattr(row, "account", "") == resolved]
        return [_position_asdict(row) for row in rows]

    async def pnl_snapshot(
        self,
        *,
        account: str = "",
        wait_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting PnL")
        resolved = self.resolved_account(account)
        if not resolved:
            return []
        self._ib.reqPnL(resolved)
        await asyncio.sleep(max(0.0, wait_seconds))
        rows = self._ib.pnl(resolved)
        self._ib.cancelPnL(resolved)
        return [_pnl_asdict(row) for row in rows]

    async def qualify_crypto_contract(self) -> QualifiedCryptoContract:
        if not self._ib:
            raise BrokerConnectionError("connect before qualifying contracts")
        last_error: Exception | None = None
        for exchange in self.config.broker.crypto_exchanges:
            try:
                return await self.qualify_contract(
                    symbol=self.config.contract.symbol,
                    security_type=self.config.contract.security_type,
                    currency=self.config.contract.currency,
                    exchange=exchange,
                )
            except Exception as exc:  # pragma: no cover - depends on Gateway
                last_error = exc
                continue
        raise BrokerConnectionError(f"could not qualify BTC crypto contract: {last_error}")

    async def qualify_contract(
        self,
        *,
        symbol: str,
        security_type: str,
        currency: str,
        exchange: str,
        last_trade_date_or_contract_month: str = "",
        local_symbol: str = "",
    ) -> QualifiedCryptoContract:
        if not self._ib:
            raise BrokerConnectionError("connect before qualifying contracts")
        ib_async = _ib_async()
        contract = ib_async.Contract(
            symbol=symbol,
            secType=security_type,
            currency=currency,
            exchange=exchange,
        )
        if last_trade_date_or_contract_month:
            contract.lastTradeDateOrContractMonth = last_trade_date_or_contract_month
        if local_symbol:
            contract.localSymbol = local_symbol
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified or qualified[0] is None:
            raise BrokerConnectionError(
                f"could not qualify contract {security_type}:{symbol}/{currency}@{exchange}"
            )
        contract = qualified[0]
        return QualifiedCryptoContract(
            symbol=getattr(contract, "symbol", symbol),
            currency=getattr(contract, "currency", currency),
            exchange=getattr(contract, "exchange", exchange),
            con_id=getattr(contract, "conId", None),
            raw=contract,
        )

    async def snapshot_quote(
        self,
        contract: QualifiedCryptoContract,
        *,
        max_wait_seconds: float = 10.0,
    ) -> MarketQuote:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting quotes")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive")
        ticker = self._ib.reqMktData(contract.raw, "", False, False)
        bid = ask = None
        for _ in range(max(1, math.ceil(max_wait_seconds / 0.25))):
            await asyncio.sleep(0.25)
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            if bid is not None and ask is not None and float(bid) > 0 and float(ask) >= float(bid):
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
        multiplier = _optional_float(getattr(contract.raw, "multiplier", None))
        security_type = str(getattr(contract.raw, "secType", "") or "")
        local_symbol = str(getattr(contract.raw, "localSymbol", "") or "")
        for row in rows:
            volume = _historical_bar_volume(getattr(row, "volume", 0))
            close = float(row.close)
            bar_start = ensure_utc(row.date)
            bar_close = bar_start + _bar_size_delta(bar_size)
            quote_volume = (
                volume * close * multiplier
                if multiplier is not None and multiplier > 0 and security_type == "FUT"
                else None
            )
            bars.append(
                Bar(
                    timestamp_utc=bar_close if bar_close > bar_start else bar_start,
                    symbol=f"{contract.symbol}/{contract.currency}",
                    bar_size=bar_size,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=close,
                    volume=volume,
                    quote_volume=quote_volume,
                    vwap=float(row.average) if getattr(row, "average", None) else None,
                    trade_count=int(row.barCount) if getattr(row, "barCount", None) else None,
                    source=f"IBKR:{what_to_show}",
                    extra={
                        "bar_start_timestamp_utc": bar_start.isoformat(),
                        "bar_close_timestamp_utc": (
                            bar_close if bar_close > bar_start else bar_start
                        ).isoformat(),
                        "security_type": security_type,
                        "local_symbol": local_symbol,
                        "contract_multiplier": multiplier or 0.0,
                        "what_to_show": what_to_show,
                    },
                )
            )
        return bars

    async def market_depth_snapshot(
        self,
        contract: QualifiedCryptoContract,
        *,
        rows: int = 10,
        max_wait_seconds: float = 5.0,
    ) -> list[dict[str, float | int | str]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting market depth")
        ticker = self._ib.reqMktDepth(contract.raw, numRows=max(1, rows), isSmartDepth=False)
        await asyncio.sleep(max_wait_seconds)
        depth_rows: list[dict[str, float | int | str]] = []
        for side_name, levels in (("bid", getattr(ticker, "domBids", [])), ("ask", getattr(ticker, "domAsks", []))):
            for position, level in enumerate(levels or []):
                price = _optional_float(getattr(level, "price", None))
                size = _optional_float(getattr(level, "size", None))
                if price is None:
                    continue
                depth_rows.append(
                    {
                        "side": side_name,
                        "position": position,
                        "price": price,
                        "size": size or 0.0,
                    }
                )
        self._ib.cancelMktDepth(contract.raw, isSmartDepth=False)
        return depth_rows

    async def tick_by_tick_snapshot(
        self,
        contract: QualifiedCryptoContract,
        *,
        tick_type: str = "BidAsk",
        number_of_ticks: int = 0,
        max_wait_seconds: float = 5.0,
    ) -> list[dict[str, Any]]:
        if not self._ib:
            raise BrokerConnectionError("connect before requesting tick-by-tick data")
        ticker = self._ib.reqTickByTickData(
            contract.raw,
            tickType=tick_type,
            numberOfTicks=max(0, number_of_ticks),
            ignoreSize=False,
        )
        await asyncio.sleep(max_wait_seconds)
        rows: list[dict[str, Any]] = []
        for tick in getattr(ticker, "tickByTicks", []) or []:
            rows.append(
                {
                    "time": ensure_utc(getattr(tick, "time", utc_now())).isoformat(),
                    "tick_type": tick_type,
                    "price": _optional_float(getattr(tick, "price", None)),
                    "size": _optional_float(getattr(tick, "size", None)),
                    "bid_price": _optional_float(getattr(tick, "bidPrice", None)),
                    "ask_price": _optional_float(getattr(tick, "askPrice", None)),
                    "bid_size": _optional_float(getattr(tick, "bidSize", None)),
                    "ask_size": _optional_float(getattr(tick, "askSize", None)),
                }
            )
        self._ib.cancelTickByTickData(contract.raw, tick_type)
        return rows

    def _order_from_intent(
        self,
        intent: OrderIntent,
        *,
        parent_order_ids: dict[str, int] | None = None,
    ) -> Any:
        if self.config.contract.security_type == "CRYPTO" and intent.order_type not in {
            OrderType.LMT,
            OrderType.MKT,
        }:
            raise ConfigError("IBKR crypto supports only market and limit orders; use synthetic stops")
        ib_async = _ib_async()
        order = ib_async.Order()
        order.action = intent.side.value
        order.orderType = intent.order_type.value
        order.tif = intent.time_in_force.value
        order.transmit = intent.transmit
        if self._configured_account():
            order.account = self._configured_account()
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

    def _intent_notional_estimate(
        self,
        intent: OrderIntent,
        *,
        reference_price: float | None = None,
    ) -> float | None:
        if intent.cash_qty is not None:
            return intent.cash_qty
        if intent.quantity is None:
            return None
        price = intent.limit_price if intent.limit_price is not None else intent.aux_price
        if price is None:
            price = reference_price
        if price is None:
            return None
        return abs(intent.quantity * price)

    def _max_order_notional(self) -> float:
        if self.config.runtime.mode == RuntimeMode.LIVE:
            return self.config.risk.live_max_notional
        return self.config.risk.paper_max_notional

    def _validate_order_intent_risk(
        self,
        intent: OrderIntent,
        *,
        reference_price: float | None = None,
        current_position_quantity: float | None = None,
    ) -> None:
        if intent.order_type == OrderType.MKT and intent.quantity is not None and reference_price is None:
            raise ConfigError("quantity-based market orders require a live quote/reference price")
        if intent.side.value == "SELL" and intent.order_type == OrderType.MKT:
            if current_position_quantity is None:
                raise ConfigError("market sell exits require current BTC position verification")
            if intent.quantity is None:
                raise ConfigError("market sell exits require quantity")
            if intent.quantity > current_position_quantity + 1e-8:
                raise ConfigError(
                    f"market sell quantity {intent.quantity:.8f} exceeds current BTC position "
                    f"{current_position_quantity:.8f}"
                )
        estimate = self._intent_notional_estimate(intent, reference_price=reference_price)
        if estimate is None:
            raise ConfigError("order intent notional cannot be estimated")
        max_notional = self._max_order_notional()
        if estimate > max_notional + 1e-9:
            raise ConfigError(
                f"order intent estimated notional {estimate:.2f} exceeds "
                f"configured {self.config.runtime.mode.value} max notional {max_notional:.2f}"
            )

    def place_order_intent(
        self,
        contract: QualifiedCryptoContract,
        intent: OrderIntent,
        *,
        reference_price: float | None = None,
        current_position_quantity: float | None = None,
    ) -> Any:
        self._assert_can_mutate_orders()
        self._validate_order_intent_risk(
            intent,
            reference_price=reference_price,
            current_position_quantity=current_position_quantity,
        )
        order = self._order_from_intent(intent)
        return self._ib.placeOrder(contract.raw, order)

    def place_order_intents(self, contract: QualifiedCryptoContract, intents: list[OrderIntent]) -> list[Any]:
        self._assert_can_mutate_orders()
        parent_order_ids: dict[str, int] = {}
        trades: list[Any] = []
        for intent in intents:
            self._validate_order_intent_risk(intent)
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
        self._assert_can_mutate_orders()
        order = getattr(trade, "order", trade)
        return self._ib.cancelOrder(order)
