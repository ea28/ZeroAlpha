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
    def __init__(self, config: AppConfig, *, output_path: Path, interval_seconds: float = 5.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.config = config
        self.output_path = output_path
        self.interval_seconds = interval_seconds

    async def run(self, *, duration_seconds: float | None = None) -> int:
        broker = IBKRBroker(self.config)
        await broker.connect(read_only=True)
        count = 0
        try:
            contract = await broker.qualify_crypto_contract()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            started = asyncio.get_running_loop().time()
            with self.output_path.open("a", encoding="utf-8") as handle:
                while True:
                    if duration_seconds is not None:
                        elapsed = asyncio.get_running_loop().time() - started
                        if elapsed >= duration_seconds:
                            break
                    quote = await broker.snapshot_quote(contract)
                    record = quote_to_record(quote, contract)
                    handle.write(json.dumps(asdict(record), default=str, sort_keys=True) + "\n")
                    handle.flush()
                    count += 1
                    await asyncio.sleep(self.interval_seconds)
        finally:
            await broker.disconnect()
        return count
