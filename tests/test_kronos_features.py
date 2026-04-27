from datetime import UTC, datetime, timedelta

from dataclasses import replace

from zeroalpha.config import AppConfig, KronosConfig
from zeroalpha.domain import Bar
from zeroalpha.features.kronos import build_kronos_features, kronos_import_status
from zeroalpha.models.dataset import build_meta_label_samples
from zeroalpha.candidates.events import CandidateGenerationConfig


def _bar(idx: int, *, bar_size: str = "1h") -> Bar:
    close = 100.0 + idx * 0.1
    return Bar(
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx),
        symbol="BTCUSDT",
        bar_size=bar_size,
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=100 + idx,
        trade_count=10 + idx % 3,
        source="TEST",
    )


def test_kronos_proxy_features_are_causal_and_sized() -> None:
    bars = [_bar(idx) for idx in range(128)]
    features = build_kronos_features(
        bars,
        config=KronosConfig(enabled=True, mode="proxy", lookback_bars=64, embedding_dims=8),
    )

    assert features["kronos_enabled"] == 1.0
    assert features["kronos_proxy_available"] == 1.0
    assert features["kronos_provider"] == "proxy"
    assert len([key for key in features if key.startswith("kronos_embedding_")]) == 8
    assert "kronos_direction_score" in features
    assert "kronos_regime_cluster" in features


def test_kronos_status_reports_missing_official_adapter_without_crashing() -> None:
    status = kronos_import_status()

    assert status.provider == "official"
    assert isinstance(status.available, bool)


def test_dataset_includes_kronos_proxy_features_when_enabled() -> None:
    bars = [_bar(idx) for idx in range(360)]
    cfg = replace(
        AppConfig(),
        kronos=KronosConfig(enabled=True, mode="proxy", lookback_bars=64, embedding_dims=4),
    )
    samples = build_meta_label_samples(
        bars,
        config=cfg,
        assumed_spread_bps=1.0,
        research_notional=10_000,
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=240,
            max_holding_hours=24,
            dense_stride_bars=48,
        ),
    )

    assert samples
    assert samples[0].features["kronos_proxy_available"] == 1.0
    assert "kronos_embedding_04" in samples[0].features


def test_dataset_context_features_use_returns_not_cross_asset_price_basis() -> None:
    bars = [_bar(idx) for idx in range(360)]
    eth_bars = [
        Bar(
            timestamp_utc=bar.timestamp_utc,
            symbol="ETHUSDT",
            bar_size=bar.bar_size,
            open=2_000 + idx,
            high=2_010 + idx,
            low=1_990 + idx,
            close=2_005 + idx,
            volume=1_000 + idx,
            trade_count=100 + idx % 5,
            source="TEST",
        )
        for idx, bar in enumerate(bars)
    ]
    samples = build_meta_label_samples(
        bars,
        config=AppConfig(),
        assumed_spread_bps=1.0,
        research_notional=10_000,
        context_bars={"ETHUSDT": eth_bars},
        candidate_config=CandidateGenerationConfig(
            mode="dense_research",
            min_history_bars=240,
            max_holding_hours=24,
            dense_stride_bars=48,
        ),
    )

    features = samples[0].features

    assert features["ethusdt_available"] == 1.0
    assert "ethusdt_return_elapsed_24h" in features
    assert "ethusdt_realized_vol_24" in features
    assert "ethusdt_volatility_ratio_vs_btc_24" in features
    assert "ethusdt_latest_close" not in features
    assert "ethusdt_basis_bps" not in features
