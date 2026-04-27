"""SQLite schema for runtime metadata and research artifacts."""

from __future__ import annotations

from pathlib import Path
import sqlite3


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS raw_ibkr_market_data (
        id INTEGER PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        received_timestamp_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bid REAL,
        ask REAL,
        midpoint REAL,
        last REAL,
        last_size REAL,
        market_data_type TEXT,
        request_id TEXT,
        source TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ibkr_historical_bars (
        id INTEGER PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bar_size TEXT NOT NULL,
        what_to_show TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        wap REAL,
        bar_count INTEGER,
        download_timestamp_utc TEXT NOT NULL,
        request_parameters_hash TEXT NOT NULL,
        UNIQUE(timestamp_utc, symbol, bar_size, what_to_show, request_parameters_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS external_bars (
        id INTEGER PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        source TEXT NOT NULL,
        symbol TEXT NOT NULL,
        bar_size TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        quote_volume REAL,
        trade_count INTEGER,
        vwap REAL,
        download_timestamp_utc TEXT NOT NULL,
        data_version TEXT NOT NULL,
        UNIQUE(timestamp_utc, source, symbol, bar_size, data_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_events (
        event_id TEXT PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        candidate_type TEXT NOT NULL,
        side TEXT NOT NULL,
        bar_size TEXT NOT NULL,
        signal_strength REAL NOT NULL,
        reference_price REAL NOT NULL,
        max_holding_hours INTEGER NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS labels (
        event_id TEXT PRIMARY KEY,
        entry_timestamp_utc TEXT NOT NULL,
        entry_price REAL NOT NULL,
        upper_barrier_price REAL NOT NULL,
        lower_barrier_price REAL NOT NULL,
        vertical_barrier_timestamp_utc TEXT NOT NULL,
        exit_timestamp_utc TEXT NOT NULL,
        exit_price REAL NOT NULL,
        outcome_type TEXT NOT NULL,
        gross_return REAL NOT NULL,
        net_return REAL NOT NULL,
        label INTEGER NOT NULL,
        t1 TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        event_id TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        model_version TEXT NOT NULL,
        fold_id TEXT,
        raw_score REAL,
        uncalibrated_probability REAL,
        calibrated_probability REAL NOT NULL,
        expected_value REAL NOT NULL,
        decision TEXT NOT NULL,
        decision_reason TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        internal_order_id TEXT PRIMARY KEY,
        ibkr_order_id TEXT,
        event_id TEXT,
        timestamp_submitted_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        time_in_force TEXT NOT NULL,
        limit_price REAL,
        cash_qty REAL,
        quantity REAL,
        status TEXT NOT NULL,
        reject_reason TEXT,
        cancel_reason TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id TEXT PRIMARY KEY,
        internal_order_id TEXT NOT NULL,
        ibkr_order_id TEXT,
        timestamp_filled_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        filled_quantity REAL NOT NULL,
        fill_price REAL NOT NULL,
        commission REAL NOT NULL,
        spread_at_fill_bps REAL,
        slippage_bps REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS account_snapshots (
        id INTEGER PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        account_equity REAL,
        available_cash REAL,
        buying_power REAL,
        crypto_position_quantity REAL,
        crypto_position_market_value REAL,
        unrealized_pnl REAL,
        realized_pnl REAL,
        daily_pnl REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY,
        timestamp_utc TEXT NOT NULL,
        severity TEXT NOT NULL,
        code TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
]


def initialize_sqlite(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
