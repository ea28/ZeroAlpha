"""IBKR quote recorder for paper/live execution calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import asyncio
import json

from zeroalpha.broker.ibkr import IBKRBroker, QualifiedCryptoContract
from zeroalpha.config import AppConfig
from zeroalpha.domain import MarketQuote
from zeroalpha.monitoring.events import RuntimeEventStream


@dataclass(frozen=True, slots=True)
class QuoteRecord:
    timestamp_utc: datetime
    received_timestamp_utc: datetime
    symbol: str
    exchange: str
    con_id: int
    bid: float
    ask: float
    bid_size: float | None
    ask_size: float | None
    midpoint: float
    spread_bps: float
    quote_age_ms: float
    market_data_type: str | None


def quote_to_record(quote: MarketQuote, contract: QualifiedCryptoContract) -> QuoteRecord:
    return QuoteRecord(
        timestamp_utc=quote.timestamp_utc,
        received_timestamp_utc=quote.received_timestamp_utc,
        symbol=quote.symbol,
        exchange=contract.exchange,
        con_id=contract.con_id,
        bid=quote.bid,
        ask=quote.ask,
        bid_size=quote.bid_size,
        ask_size=quote.ask_size,
        midpoint=quote.midpoint,
        spread_bps=quote.spread_bps,
        quote_age_ms=quote.quote_age_ms(),
        market_data_type=quote.market_data_type,
    )


class IBKRQuoteRecorder:
    def __init__(
        self,
        config: AppConfig,
        *,
        output_path: Path,
        interval_seconds: float = 5.0,
        snapshot_timeout_seconds: float = 10.0,
        events: RuntimeEventStream | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if snapshot_timeout_seconds <= 0:
            raise ValueError("snapshot_timeout_seconds must be positive")
        self.config = config
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.snapshot_timeout_seconds = snapshot_timeout_seconds
        self.events = events

    async def run(
        self,
        *,
        duration_seconds: float | None = None,
        symbol: str | None = None,
        security_type: str | None = None,
        currency: str | None = None,
        exchange: str | None = None,
        last_trade_date_or_contract_month: str = "",
        local_symbol: str = "",
    ) -> int:
        broker = IBKRBroker(self.config)
        if self.events is not None:
            self.events.emit(
                "quote_recorder.start",
                "starting IBKR quote recorder",
                output=str(self.output_path),
                interval_seconds=self.interval_seconds,
                duration_seconds=duration_seconds,
            )
        await broker.connect(read_only=True)
        count = 0
        try:
            if self.events is not None:
                self.events.emit(
                    "broker.connected",
                    "connected to IBKR Gateway/TWS",
                    host=self.config.broker.host,
                    port=self.config.broker.port,
                    read_only=True,
                )
            if security_type or exchange or symbol or currency or last_trade_date_or_contract_month or local_symbol:
                contract = await broker.qualify_contract(
                    symbol=symbol or self.config.contract.symbol,
                    security_type=security_type or self.config.contract.security_type,
                    currency=currency or self.config.contract.currency,
                    exchange=exchange or self.config.broker.crypto_exchanges[0],
                    last_trade_date_or_contract_month=last_trade_date_or_contract_month,
                    local_symbol=local_symbol,
                )
            else:
                contract = await broker.qualify_crypto_contract()
            if self.events is not None:
                self.events.emit(
                    "broker.contract_qualified",
                    "qualified quote-recorder contract",
                    symbol=f"{contract.symbol}/{contract.currency}",
                    exchange=contract.exchange,
                    con_id=contract.con_id,
                )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            started = asyncio.get_running_loop().time()
            with self.output_path.open("a", encoding="utf-8") as handle:
                while True:
                    if duration_seconds is not None:
                        elapsed = asyncio.get_running_loop().time() - started
                        if elapsed >= duration_seconds:
                            break
                    quote = await broker.snapshot_quote(
                        contract,
                        max_wait_seconds=self.snapshot_timeout_seconds,
                    )
                    record = quote_to_record(quote, contract)
                    handle.write(json.dumps(asdict(record), default=str, sort_keys=True) + "\n")
                    handle.flush()
                    count += 1
                    if self.events is not None:
                        self.events.emit(
                            "market.quote_recorded",
                            "recorded IBKR quote",
                            count=count,
                            symbol=record.symbol,
                            bid=record.bid,
                            ask=record.ask,
                            spread_bps=record.spread_bps,
                            bid_size=record.bid_size,
                            ask_size=record.ask_size,
                            quote_age_ms=record.quote_age_ms,
                        )
                    await asyncio.sleep(self.interval_seconds)
        finally:
            await broker.disconnect()
            if self.events is not None:
                self.events.emit(
                    "quote_recorder.finished",
                    "IBKR quote recorder finished",
                    count=count,
                    output=str(self.output_path),
                )
                self.events.emit("broker.disconnected", "disconnected from IBKR Gateway/TWS")
        return count
