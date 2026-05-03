from __future__ import annotations

from datetime import UTC, datetime, timedelta

from zeroalpha.candidates.events import CandidateGenerationConfig
from zeroalpha.config import AppConfig
from zeroalpha.data.external.ibkr_quotes import read_ibkr_quote_records
from zeroalpha.domain import Bar, MarketQuote
from zeroalpha.models.dataset import build_meta_label_samples


def _bar(idx: int) -> Bar:
    close = 100.0 + idx * 0.1
    return Bar(
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
        symbol="BTCUSDT",
        bar_size="1h",
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=100 + idx,
        trade_count=10 + idx % 3,
        source="TEST",
    )


def test_read_ibkr_quote_records_handles_sizes_and_bad_rows(tmp_path) -> None:
    path = tmp_path / "quotes.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"timestamp_utc":"2024-01-01T00:00:00+00:00",'
                '"received_timestamp_utc":"2024-01-01T00:00:00+00:00",'
                '"symbol":"BTC/USD","bid":42000,"ask":42002,'
                '"bid_size":1.5,"ask_size":2.0,"market_data_type":"LIVE"}',
                "not-json",
            ]
        ),
        encoding="utf-8",
    )

    quotes = read_ibkr_quote_records(path)

    assert len(quotes) == 1
    assert quotes[0].bid == 42000
    assert quotes[0].ask_size == 2.0
    assert quotes[0].market_data_type == "LIVE"


def test_dataset_includes_ibkr_quote_microstructure_features() -> None:
    bars = [_bar(idx) for idx in range(360)]
    quotes = [
        MarketQuote(
            timestamp_utc=bar.timestamp_utc,
            received_timestamp_utc=bar.timestamp_utc,
            symbol="BTC/USD",
            bid=bar.close - 0.05,
            ask=bar.close + 0.05,
            bid_size=1.0 + idx % 3,
            ask_size=2.0 + idx % 2,
        )
        for idx, bar in enumerate(bars)
    ]

    samples = build_meta_label_samples(
        bars,
        config=AppConfig(),
        assumed_spread_bps=1.0,
        research_notional=10_000,
        market_quotes=quotes,
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=240,
            max_holding_hours=24,
            dense_stride_bars=48,
        ),
    )

    features = samples[0].features

    assert features["ibkr_quote_available"] == 1.0
    assert features["ibkr_quote_size_available"] == 1.0
    assert "ibkr_spread_bps" in features
    assert "ibkr_top_of_book_imbalance" in features
    assert "ibkr_mid_return_1h" in features
