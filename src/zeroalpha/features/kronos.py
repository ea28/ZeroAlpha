"""Causal Kronos and Kronos-compatible K-line feature generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import cos, log, pi, sqrt
from statistics import mean, pstdev
from typing import Any

from zeroalpha.config import KronosConfig
from zeroalpha.domain import Bar


@dataclass(frozen=True, slots=True)
class KronosFeatureStatus:
    provider: str
    available: bool
    detail: str = ""


def kronos_import_status() -> KronosFeatureStatus:
    try:
        import pandas  # noqa: F401
        import torch  # noqa: F401
        from model import Kronos, KronosTokenizer  # type: ignore  # noqa: F401
        from model.kronos import calc_time_stamps  # type: ignore  # noqa: F401
    except Exception as exc:
        return KronosFeatureStatus("official", False, f"{type(exc).__name__}: {exc}")
    return KronosFeatureStatus("official", True, "official Kronos import ok")


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _zscore(values: list[float]) -> list[float]:
    center = _safe_mean(values)
    scale = _safe_stdev(values)
    if scale <= 1e-12:
        return [0.0 for _ in values]
    return [(value - center) / scale for value in values]


def _tail(values: list[float], count: int) -> list[float]:
    return values[-count:] if len(values) >= count else values[:]


def _autocorr(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    mx = _safe_mean(x)
    my = _safe_mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator > 1e-12 else 0.0


def _proxy_regime(direction_score: float, volatility: float, tail_ratio: float) -> str:
    if volatility > 0.025 or tail_ratio > 3.0:
        return "panic"
    if direction_score > 0.75:
        return "trend_up"
    if direction_score < -0.75:
        return "trend_down"
    return "chop"


def _kline_proxy_features(bars: list[Bar], *, config: KronosConfig) -> dict[str, float | str]:
    dims = config.embedding_dims
    if len(bars) < max(16, min(config.lookback_bars, 32)):
        return {
            "kronos_enabled": 1.0,
            "kronos_official_available": 0.0,
            "kronos_proxy_available": 0.0,
            "kronos_provider": "insufficient_history",
            **{f"kronos_embedding_{idx + 1:02d}": 0.0 for idx in range(dims)},
        }

    window = bars[-config.lookback_bars :]
    returns = [log(window[idx].close / window[idx - 1].close) for idx in range(1, len(window))]
    ranges = [log(bar.high / bar.low) for bar in window if bar.low > 0]
    bodies = [
        (bar.close - bar.open) / bar.open
        for bar in window
        if bar.open > 0
    ]
    upper_wicks = [
        (bar.high - max(bar.open, bar.close)) / bar.open
        for bar in window
        if bar.open > 0
    ]
    lower_wicks = [
        (min(bar.open, bar.close) - bar.low) / bar.open
        for bar in window
        if bar.open > 0
    ]
    log_volumes = [log(max(bar.volume, 1e-12)) for bar in window]
    dollar_volumes = [log(max(bar.volume * bar.close, 1e-12)) for bar in window]

    channels = [
        _zscore(returns),
        _zscore(ranges[-len(returns) :]),
        _zscore(bodies[-len(returns) :]),
        _zscore(upper_wicks[-len(returns) :]),
        _zscore(lower_wicks[-len(returns) :]),
        _zscore(log_volumes[-len(returns) :]),
        _zscore(dollar_volumes[-len(returns) :]),
    ]
    n = min(len(channel) for channel in channels)
    channels = [channel[-n:] for channel in channels]
    embeddings: dict[str, float] = {}
    for dim in range(dims):
        channel = channels[dim % len(channels)]
        frequency = dim // len(channels) + 1
        weights = [cos((idx + 0.5) * frequency * pi / n) for idx in range(n)]
        norm = sqrt(sum(weight * weight for weight in weights)) or 1.0
        embeddings[f"kronos_embedding_{dim + 1:02d}"] = (
            sum(value * weight for value, weight in zip(channel, weights, strict=True)) / norm
        )

    recent_returns = _tail(returns, min(24, len(returns)))
    medium_returns = _tail(returns, min(168, len(returns)))
    recent_volatility = _safe_stdev(recent_returns)
    medium_volatility = _safe_stdev(medium_returns)
    trend_return = sum(recent_returns)
    direction_score = trend_return / (recent_volatility * sqrt(len(recent_returns)) + 1e-12)
    abs_returns = [abs(value) for value in medium_returns]
    tail_ratio = (
        max(abs_returns) / (_safe_mean(abs_returns) + 1e-12)
        if abs_returns
        else 0.0
    )
    range_forecast = _safe_mean(_tail(ranges, min(24, len(ranges))))
    vol_of_vol = _safe_stdev(
        [
            _safe_stdev(medium_returns[max(0, idx - 24) : idx])
            for idx in range(2, len(medium_returns) + 1)
        ]
    )
    return {
        "kronos_enabled": 1.0,
        "kronos_official_available": 0.0,
        "kronos_proxy_available": 1.0,
        "kronos_provider": "proxy",
        "kronos_direction_score": direction_score,
        "kronos_volatility_forecast": recent_volatility,
        "kronos_volatility_ratio": recent_volatility / medium_volatility if medium_volatility > 0 else 0.0,
        "kronos_range_forecast": range_forecast,
        "kronos_uncertainty": vol_of_vol + tail_ratio * recent_volatility,
        "kronos_return_autocorr": _autocorr(medium_returns),
        "kronos_tail_ratio": tail_ratio,
        "kronos_regime_cluster": _proxy_regime(direction_score, recent_volatility, tail_ratio),
        **embeddings,
    }


@lru_cache(maxsize=2)
def _load_official_kronos(
    model_name: str,
    tokenizer_name: str,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from model import Kronos, KronosTokenizer  # type: ignore
    from model.kronos import calc_time_stamps  # type: ignore

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name).to(device)
    model = Kronos.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer.eval()
    return model, tokenizer, calc_time_stamps, torch


def _official_hidden_features(bars: list[Bar], *, config: KronosConfig) -> dict[str, float | str]:
    import numpy as np
    import pandas as pd

    model, tokenizer, calc_time_stamps, torch = _load_official_kronos(
        config.official_model_name,
        config.official_tokenizer_name,
        config.device,
    )
    window = bars[-config.lookback_bars :]
    df = pd.DataFrame(
        {
            "open": [bar.open for bar in window],
            "high": [bar.high for bar in window],
            "low": [bar.low for bar in window],
            "close": [bar.close for bar in window],
            "volume": [bar.volume for bar in window],
            "amount": [bar.volume * bar.close for bar in window],
            "timestamps": pd.to_datetime([bar.timestamp_utc for bar in window]),
        }
    )
    values = df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    centered = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-5)
    stamps = calc_time_stamps(df["timestamps"]).values.astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(centered).unsqueeze(0).to(config.device)
        stamp = torch.from_numpy(stamps).unsqueeze(0).to(config.device)
        s1_ids, s2_ids = tokenizer.encode(torch.clip(x, -5, 5), half=True)
        _, context = model.decode_s1(s1_ids, s2_ids, stamp)
        pooled = context[:, -config.hidden_pool_tail :, :].mean(dim=1).squeeze(0).detach().cpu().numpy()
    dims = min(config.embedding_dims, len(pooled))
    features = {
        "kronos_enabled": 1.0,
        "kronos_official_available": 1.0,
        "kronos_proxy_available": 0.0,
        "kronos_provider": "official",
    }
    features.update({f"kronos_embedding_{idx + 1:02d}": float(pooled[idx]) for idx in range(dims)})
    for idx in range(dims, config.embedding_dims):
        features[f"kronos_embedding_{idx + 1:02d}"] = 0.0
    return features


def build_kronos_features(bars: list[Bar], *, config: KronosConfig) -> dict[str, float | str]:
    if not config.enabled:
        return {"kronos_enabled": 0.0}
    ordered = sorted(bars, key=lambda bar: bar.timestamp_utc)
    if config.mode in {"official", "auto"}:
        try:
            return _official_hidden_features(ordered, config=config)
        except Exception as exc:
            if config.mode == "official":
                raise
            features = _kline_proxy_features(ordered, config=config)
            features["kronos_official_error"] = type(exc).__name__
            return features
    return _kline_proxy_features(ordered, config=config)
