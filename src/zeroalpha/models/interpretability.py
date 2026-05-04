"""Fold-local model interpretability helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

from zeroalpha.models.dataset import MetaLabelSample


@dataclass(frozen=True, slots=True)
class FeatureImportanceRow:
    fold_id: int
    scope: str
    model_name: str
    feature: str
    feature_group: str
    metric: str
    baseline_score: float
    permuted_score_mean: float
    importance_mean: float
    importance_std: float
    rank: int
    n_repeats: int
    sample_count: int
    column_count: int


@dataclass(frozen=True, slots=True)
class ModelNativeImportanceRow:
    fold_id: int
    model_name: str
    feature: str
    feature_group: str
    source: str
    importance: float
    signed_importance: float
    rank: int


@dataclass(frozen=True, slots=True)
class ShapImportanceRow:
    fold_id: int
    scope: str
    model_name: str
    feature: str
    feature_group: str
    explainer: str
    mean_abs_shap: float
    mean_shap: float
    rank: int
    sample_count: int
    background_count: int
    column_count: int


@dataclass(frozen=True, slots=True)
class EncodedFeatureGroup:
    name: str
    family: str
    columns: tuple[int, ...]


def feature_family(feature_name: str) -> str:
    base = feature_name.split("=", 1)[0]
    lowered = base.lower()
    if lowered.startswith("pm_"):
        return "prediction_market"
    if lowered.startswith("ibkr_futures") or "mbt" in lowered or "futures" in lowered:
        return "futures_context"
    if lowered.startswith("ibkr"):
        return "ibkr_spot"
    if lowered.startswith(("eth", "sol")) or "relative_strength" in lowered:
        return "cross_asset"
    if any(token in lowered for token in ("volume", "trade_count", "taker", "liquidity")):
        return "volume_order_flow"
    if any(token in lowered for token in ("spread", "microprice", "order_book", "depth", "imbalance")):
        return "microstructure"
    if any(token in lowered for token in ("rsi", "ema", "macd", "bollinger", "atr", "sma")):
        return "technical"
    if any(token in lowered for token in ("return", "momentum", "slope", "breakout", "breakdown")):
        return "momentum"
    if any(token in lowered for token in ("volatility", "vol_", "range", "drawdown", "distance")):
        return "volatility_range"
    if any(token in lowered for token in ("hour", "day_", "weekend", "session")):
        return "calendar"
    if lowered in {"candidate_type", "side", "setup_family", "market_regime"}:
        return "categorical"
    if any(token in lowered for token in ("cost", "spread_bps", "slippage", "profit", "stop")):
        return "execution_cost"
    return "other"


def encoded_feature_groups(feature_names: Sequence[str]) -> list[EncodedFeatureGroup]:
    grouped: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        group_name = name.split("=", 1)[0] if "=" in name else name
        grouped.setdefault(group_name, []).append(idx)
    return [
        EncodedFeatureGroup(name=name, family=feature_family(name), columns=tuple(columns))
        for name, columns in sorted(grouped.items())
    ]


def encoded_feature_family_groups(feature_names: Sequence[str]) -> list[EncodedFeatureGroup]:
    grouped: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        grouped.setdefault(feature_family(name), []).append(idx)
    return [
        EncodedFeatureGroup(name=f"family:{family}", family=family, columns=tuple(columns))
        for family, columns in sorted(grouped.items())
    ]


def encoded_permutation_groups(
    feature_names: Sequence[str],
    *,
    grouping: str = "feature",
) -> list[EncodedFeatureGroup]:
    normalized = grouping.strip().lower()
    if normalized == "feature":
        return encoded_feature_groups(feature_names)
    if normalized == "family":
        return encoded_feature_family_groups(feature_names)
    if normalized == "both":
        return encoded_feature_groups(feature_names) + encoded_feature_family_groups(feature_names)
    raise ValueError("permutation grouping must be feature, family, or both")


def _sample_rows(
    x: Any,
    y: Any,
    samples: Sequence[MetaLabelSample],
    *,
    sample_limit: int,
    seed: int,
) -> tuple[Any, Any, list[MetaLabelSample]]:
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if sample_limit <= 0 or len(y_arr) <= sample_limit:
        return x_arr.copy(), y_arr.copy(), list(samples)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(y_arr), size=sample_limit, replace=False))
    return x_arr[indices].copy(), y_arr[indices].copy(), [samples[int(idx)] for idx in indices]


def _candidate_groups(x: Any, groups: Sequence[EncodedFeatureGroup], max_features: int) -> list[EncodedFeatureGroup]:
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    scored: list[tuple[float, EncodedFeatureGroup]] = []
    for group in groups:
        values = x_arr[:, list(group.columns)]
        variance = float(np.nanmean(np.nanvar(values, axis=0)))
        if isfinite(variance) and variance > 0:
            scored.append((variance, group))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    if max_features > 0:
        scored = scored[:max_features]
    return [group for _, group in scored]


def _metric_score(
    metric: str,
    labels: Any,
    probabilities: Any,
    samples: Sequence[MetaLabelSample],
    *,
    selection_threshold: float | None,
) -> tuple[float, bool]:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss

    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if metric == "brier":
        return float(brier_score_loss(y, p)), False
    if metric == "log_loss":
        return float(log_loss(y, p, labels=[0, 1])), False
    if metric in {"net_pnl", "threshold_only_net_pnl"}:
        threshold = 0.5 if selection_threshold is None else float(selection_threshold)
        pnl = sum(
            sample.notional * sample.net_return
            for sample, probability in zip(samples, p, strict=True)
            if float(probability) >= threshold
        )
        return float(pnl), True
    raise ValueError(f"unsupported importance metric {metric}")


def compute_permutation_importance(
    *,
    fold_id: int,
    scope: str,
    model_name: str,
    x: Any,
    y: Any,
    samples: Sequence[MetaLabelSample],
    feature_names: Sequence[str],
    predict_probability: Callable[[Any], Any],
    metrics: Iterable[str],
    n_repeats: int,
    max_features: int,
    sample_limit: int,
    top_n: int,
    selection_threshold: float | None,
    grouping: str = "feature",
    seed: int = 1729,
) -> list[FeatureImportanceRow]:
    import numpy as np

    metrics = tuple(
        dict.fromkeys(
            "threshold_only_net_pnl" if metric.strip() == "net_pnl" else metric.strip()
            for metric in metrics
            if metric.strip()
        )
    )
    if not metrics or n_repeats <= 0 or len(samples) == 0:
        return []
    x_eval, y_eval, sample_eval = _sample_rows(
        x,
        y,
        samples,
        sample_limit=sample_limit,
        seed=seed + fold_id,
    )
    groups = _candidate_groups(
        x_eval,
        encoded_permutation_groups(feature_names, grouping=grouping),
        max_features,
    )
    if not groups:
        return []
    baseline_probability = predict_probability(x_eval)
    baseline_scores = {
        metric: _metric_score(
            metric,
            y_eval,
            baseline_probability,
            sample_eval,
            selection_threshold=selection_threshold,
        )
        for metric in metrics
    }
    digest = hashlib.sha256(f"{scope}:{model_name}".encode("utf-8")).hexdigest()
    model_seed = int(digest[:10], 16) % 1_000_003
    rng = np.random.default_rng(seed + 1009 * (fold_id + 1) + model_seed)
    by_metric: dict[str, list[FeatureImportanceRow]] = {metric: [] for metric in metrics}
    for group in groups:
        columns = list(group.columns)
        metric_values: dict[str, list[float]] = {metric: [] for metric in metrics}
        for _ in range(n_repeats):
            permuted = x_eval.copy()
            order = rng.permutation(permuted.shape[0])
            permuted[:, columns] = permuted[order][:, columns]
            permuted_probability = predict_probability(permuted)
            for metric in metrics:
                permuted_score, higher_is_better = _metric_score(
                    metric,
                    y_eval,
                    permuted_probability,
                    sample_eval,
                    selection_threshold=selection_threshold,
                )
                baseline_score = baseline_scores[metric][0]
                importance = (
                    baseline_score - permuted_score
                    if higher_is_better
                    else permuted_score - baseline_score
                )
                metric_values[metric].append(float(importance))
        for metric in metrics:
            baseline_score = baseline_scores[metric][0]
            higher_is_better = baseline_scores[metric][1]
            importances = metric_values[metric]
            importance_mean = mean(importances) if importances else 0.0
            importance_std = pstdev(importances) if len(importances) > 1 else 0.0
            permuted_score_mean = (
                baseline_score - importance_mean
                if higher_is_better
                else baseline_score + importance_mean
            )
            by_metric[metric].append(
                FeatureImportanceRow(
                    fold_id=fold_id,
                    scope=scope,
                    model_name=model_name,
                    feature=group.name,
                    feature_group=group.family,
                    metric=metric,
                    baseline_score=float(baseline_score),
                    permuted_score_mean=float(permuted_score_mean),
                    importance_mean=float(importance_mean),
                    importance_std=float(importance_std),
                    rank=0,
                    n_repeats=n_repeats,
                    sample_count=len(sample_eval),
                    column_count=len(columns),
                )
            )
    rows: list[FeatureImportanceRow] = []
    for metric, metric_rows in by_metric.items():
        metric_rows.sort(key=lambda row: (-row.importance_mean, row.importance_std, row.feature))
        for rank, row in enumerate(metric_rows[:top_n], start=1):
            rows.append(
                FeatureImportanceRow(
                    fold_id=row.fold_id,
                    scope=row.scope,
                    model_name=row.model_name,
                    feature=row.feature,
                    feature_group=row.feature_group,
                    metric=metric,
                    baseline_score=row.baseline_score,
                    permuted_score_mean=row.permuted_score_mean,
                    importance_mean=row.importance_mean,
                    importance_std=row.importance_std,
                    rank=rank,
                    n_repeats=row.n_repeats,
                    sample_count=row.sample_count,
                    column_count=row.column_count,
                )
            )
    return rows


def native_model_importance(
    *,
    fold_id: int,
    model_name: str,
    estimator: Any,
    feature_names: Sequence[str],
    top_n: int,
) -> list[ModelNativeImportanceRow]:
    import numpy as np

    source = ""
    raw: Any | None = None
    native_estimator = estimator
    steps = getattr(estimator, "steps", None)
    if steps:
        native_estimator = steps[-1][1]
    if hasattr(native_estimator, "feature_importances_"):
        raw = getattr(native_estimator, "feature_importances_")
        source = "feature_importances_"
    elif hasattr(native_estimator, "coef_"):
        raw = getattr(native_estimator, "coef_")
        source = "coef_abs"
    elif hasattr(native_estimator, "get_feature_importance"):
        try:
            raw = native_estimator.get_feature_importance()
            source = "catboost_get_feature_importance"
        except Exception:
            raw = None
    if raw is None:
        return []
    values = np.asarray(raw, dtype=float)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[0]
    if len(values) != len(feature_names):
        return []
    rows: list[tuple[float, float, str]] = []
    for feature, value in zip(feature_names, values, strict=True):
        signed = float(value)
        magnitude = abs(signed) if source == "coef_abs" else signed
        if isfinite(magnitude):
            rows.append((float(magnitude), signed, feature))
    rows.sort(key=lambda item: (-item[0], item[2]))
    result: list[ModelNativeImportanceRow] = []
    for rank, (importance, signed, feature) in enumerate(rows[:top_n], start=1):
        result.append(
            ModelNativeImportanceRow(
                fold_id=fold_id,
                model_name=model_name,
                feature=feature,
                feature_group=feature_family(feature),
                source=source,
                importance=importance,
                signed_importance=signed,
                rank=rank,
            )
        )
    return result


def _native_estimator(estimator: Any) -> Any:
    steps = getattr(estimator, "steps", None)
    return steps[-1][1] if steps else estimator


def _preprocess_for_native_estimator(estimator: Any, x: Any) -> Any:
    steps = getattr(estimator, "steps", None)
    if not steps:
        return x
    transformed = x
    for _, step in steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return transformed


def _sample_matrix(x: Any, *, sample_limit: int, seed: int) -> Any:
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    if sample_limit <= 0 or len(x_arr) <= sample_limit:
        return x_arr.copy()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(x_arr), size=sample_limit, replace=False))
    return x_arr[indices].copy()


def _shap_matrix(raw_values: Any) -> Any | None:
    import numpy as np

    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim != 2:
        return None
    return arr


def compute_shap_importance(
    *,
    fold_id: int,
    scope: str,
    model_name: str,
    estimator: Any,
    x_background: Any,
    x: Any,
    feature_names: Sequence[str],
    sample_limit: int,
    background_limit: int,
    top_n: int,
    grouping: str = "feature",
    seed: int = 1729,
) -> list[ShapImportanceRow]:
    import numpy as np

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed") from exc

    x_eval = _sample_matrix(x, sample_limit=sample_limit, seed=seed + fold_id)
    if len(x_eval) == 0:
        return []
    background = _sample_matrix(
        x_background,
        sample_limit=background_limit,
        seed=seed + 7919 + fold_id,
    )
    native = _native_estimator(estimator)
    x_eval = _preprocess_for_native_estimator(estimator, x_eval)
    background = _preprocess_for_native_estimator(estimator, background)
    try:
        explainer = shap.TreeExplainer(native, data=background if len(background) else None)
        try:
            raw_values = explainer.shap_values(x_eval, check_additivity=False)
        except TypeError:
            raw_values = explainer.shap_values(x_eval)
        explainer_name = type(explainer).__name__
    except Exception as tree_exc:
        # Linear models are cheap enough for SHAP's linear path; other unsupported models should
        # fail fast and be reported by the caller instead of silently falling back to a slow kernel.
        if not hasattr(native, "coef_"):
            raise tree_exc
        explainer = shap.LinearExplainer(native, background)
        raw_values = explainer.shap_values(x_eval)
        explainer_name = type(explainer).__name__
    values = _shap_matrix(raw_values)
    if values is None or values.shape[1] != len(feature_names):
        return []

    rows: list[ShapImportanceRow] = []
    groups = encoded_permutation_groups(feature_names, grouping=grouping)
    for group in groups:
        columns = list(group.columns)
        group_values = values[:, columns]
        if group_values.ndim != 2:
            continue
        signed = np.sum(group_values, axis=1)
        abs_values = np.sum(np.abs(group_values), axis=1)
        mean_abs = float(np.nanmean(abs_values))
        mean_signed = float(np.nanmean(signed))
        if not isfinite(mean_abs):
            continue
        rows.append(
            ShapImportanceRow(
                fold_id=fold_id,
                scope=scope,
                model_name=model_name,
                feature=group.name,
                feature_group=group.family,
                explainer=explainer_name,
                mean_abs_shap=mean_abs,
                mean_shap=mean_signed if isfinite(mean_signed) else 0.0,
                rank=0,
                sample_count=len(x_eval),
                background_count=len(background),
                column_count=len(columns),
            )
        )
    rows.sort(key=lambda row: (-row.mean_abs_shap, row.feature))
    return [
        ShapImportanceRow(
            fold_id=row.fold_id,
            scope=row.scope,
            model_name=row.model_name,
            feature=row.feature,
            feature_group=row.feature_group,
            explainer=row.explainer,
            mean_abs_shap=row.mean_abs_shap,
            mean_shap=row.mean_shap,
            rank=rank,
            sample_count=row.sample_count,
            background_count=row.background_count,
            column_count=row.column_count,
        )
        for rank, row in enumerate(rows[:top_n], start=1)
    ]


def summarize_feature_importance(rows: Iterable[FeatureImportanceRow]) -> dict[str, list[dict[str, float | str]]]:
    buckets: dict[tuple[str, str, str, str], list[FeatureImportanceRow]] = {}
    for row in rows:
        buckets.setdefault((row.scope, row.model_name, row.metric, row.feature), []).append(row)
    summary: dict[str, list[dict[str, float | str]]] = {}
    for (scope, model_name, metric, feature), values in buckets.items():
        key = f"{scope}:{model_name}:{metric}"
        importances = [row.importance_mean for row in values]
        ranks = [row.rank for row in values]
        feature_group = values[0].feature_group if values else "other"
        summary.setdefault(key, []).append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "folds": float(len(values)),
                "mean_importance": mean(importances) if importances else 0.0,
                "std_importance": pstdev(importances) if len(importances) > 1 else 0.0,
                "mean_rank": mean(ranks) if ranks else 0.0,
            }
        )
    for values in summary.values():
        values.sort(key=lambda item: (-float(item["mean_importance"]), float(item["mean_rank"]), str(item["feature"])))
    return summary


def summarize_shap_importance(
    rows: Iterable[ShapImportanceRow],
) -> dict[str, list[dict[str, float | str]]]:
    buckets: dict[tuple[str, str, str], list[ShapImportanceRow]] = {}
    for row in rows:
        buckets.setdefault((row.scope, row.model_name, row.feature), []).append(row)
    summary: dict[str, list[dict[str, float | str]]] = {}
    for (scope, model_name, feature), values in buckets.items():
        key = f"{scope}:{model_name}"
        mean_abs_values = [row.mean_abs_shap for row in values]
        mean_values = [row.mean_shap for row in values]
        ranks = [row.rank for row in values]
        feature_group = values[0].feature_group if values else "other"
        summary.setdefault(key, []).append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "folds": float(len(values)),
                "mean_abs_shap": mean(mean_abs_values) if mean_abs_values else 0.0,
                "std_abs_shap": pstdev(mean_abs_values) if len(mean_abs_values) > 1 else 0.0,
                "mean_shap": mean(mean_values) if mean_values else 0.0,
                "mean_rank": mean(ranks) if ranks else 0.0,
            }
        )
    for values in summary.values():
        values.sort(key=lambda item: (-float(item["mean_abs_shap"]), float(item["mean_rank"]), str(item["feature"])))
    return summary


def summarize_native_importance(
    rows: Iterable[ModelNativeImportanceRow],
) -> dict[str, list[dict[str, float | str]]]:
    buckets: dict[tuple[str, str, str], list[ModelNativeImportanceRow]] = {}
    for row in rows:
        buckets.setdefault((row.model_name, row.source, row.feature), []).append(row)
    summary: dict[str, list[dict[str, float | str]]] = {}
    for (model_name, source, feature), values in buckets.items():
        key = f"{model_name}:{source}"
        importances = [row.importance for row in values]
        signed_importances = [row.signed_importance for row in values]
        ranks = [row.rank for row in values]
        feature_group = values[0].feature_group if values else "other"
        summary.setdefault(key, []).append(
            {
                "feature": feature,
                "feature_group": feature_group,
                "folds": float(len(values)),
                "mean_importance": mean(importances) if importances else 0.0,
                "std_importance": pstdev(importances) if len(importances) > 1 else 0.0,
                "mean_signed_importance": mean(signed_importances) if signed_importances else 0.0,
                "mean_rank": mean(ranks) if ranks else 0.0,
            }
        )
    for values in summary.values():
        values.sort(key=lambda item: (-float(item["mean_importance"]), float(item["mean_rank"]), str(item["feature"])))
    return summary
