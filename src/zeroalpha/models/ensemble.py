"""Cost-aware meta-label ensemble training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable
import math
import json
import os
import subprocess
import sys
import tempfile
import warnings

from zeroalpha.config import AppConfig
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.training import ModelDependencyError
from zeroalpha.validation.purged import walk_forward_folds


@dataclass(frozen=True, slots=True)
class FeatureEncoder:
    numeric_features: tuple[str, ...]
    categorical_values: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, samples: list[MetaLabelSample]) -> "FeatureEncoder":
        numeric: set[str] = set()
        categorical: dict[str, set[str]] = {}
        for sample in samples:
            for key, value in sample.features.items():
                if isinstance(value, bool):
                    numeric.add(key)
                elif isinstance(value, int | float):
                    numeric.add(key)
                elif isinstance(value, str):
                    categorical.setdefault(key, set()).add(value)
        return cls(
            numeric_features=tuple(sorted(numeric)),
            categorical_values={key: tuple(sorted(values)) for key, values in sorted(categorical.items())},
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_features)
        for key, values in self.categorical_values.items():
            names.extend(f"{key}={value}" for value in values)
            names.append(f"{key}=__unknown__")
        return names

    def transform(self, samples: list[MetaLabelSample]) -> Any:
        import numpy as np

        rows: list[list[float]] = []
        for sample in samples:
            row: list[float] = []
            for key in self.numeric_features:
                value = sample.features.get(key, 0.0)
                row.append(float(value) if isinstance(value, int | float | bool) else 0.0)
            for key, values in self.categorical_values.items():
                actual = sample.features.get(key)
                row.extend(1.0 if actual == value else 0.0 for value in values)
                row.append(1.0 if isinstance(actual, str) and actual not in values else 0.0)
            rows.append(row)
        return np.asarray(rows, dtype=float)


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    method: str
    model: Any | None = None
    constant_probability: float | None = None

    @classmethod
    def fit(cls, probabilities: Any, labels: Any, *, method: str) -> "ProbabilityCalibrator":
        import numpy as np

        y = np.asarray(labels, dtype=int)
        p_flat = np.asarray(probabilities, dtype=float).reshape(-1)
        p = p_flat.reshape(-1, 1)
        if len(y) == 0:
            raise ValueError("cannot fit calibrator on empty labels")
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration method must be sigmoid or isotonic")
        if len(set(y.tolist())) < 2:
            return cls(method=method, constant_probability=float(y.mean()))
        probability_spread = float(np.nanmax(p_flat) - np.nanmin(p_flat))
        if not math.isfinite(probability_spread) or probability_spread < 0.02:
            return cls(method=f"{method}_identity_low_spread")
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p.ravel(), y)
            return cls(method=method, model=calibrator)
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(p, y)
        return cls(method=method, model=calibrator)

    def predict(self, probabilities: Any) -> Any:
        import numpy as np

        p = np.asarray(probabilities, dtype=float).reshape(-1, 1)
        if self.constant_probability is not None:
            return np.full(p.shape[0], self.constant_probability)
        if self.model is None:
            return p.ravel()
        if self.method == "isotonic":
            calibrated = self.model.predict(p.ravel())
        else:
            calibrated = self.model.predict_proba(p)[:, 1]
        return np.clip(calibrated, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class BaseFoldPredictionSet:
    name: str
    ensemble_probability: Any
    threshold_probability: Any
    test_probability: Any
    selected_params: dict[str, Any]
    validation_utility: float = 0.0
    validation_brier: float = 0.0
    validation_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class FoldPrediction:
    fold_id: int
    event_id: str
    timestamp_utc: str
    candidate_type: str
    label: int
    probability: float
    expected_value: float
    should_trade: bool
    decision_reason: str
    net_return: float
    pnl: float
    side: str = ""
    predicted_return: float = 0.0
    predicted_downside: float = 0.0
    selection_score: float = 0.0
    setup_family: str = ""
    market_regime: str = ""


@dataclass(frozen=True, slots=True)
class ThresholdSweepRow:
    threshold: float
    traded_signals: int
    hit_rate: float
    net_pnl: float
    average_trade_return: float


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    lower: float
    upper: float
    count: int
    average_probability: float
    observed_rate: float


@dataclass(frozen=True, slots=True)
class FoldReport:
    fold_id: int
    train_samples: int
    calibration_samples: int
    test_samples: int
    fitted_models: list[str]
    skipped_models: dict[str, str]
    brier_score: float
    log_loss: float | None
    candidate_trades: int
    traded_signals: int
    trade_hit_rate: float
    net_pnl: float
    average_trade_return: float
    probability_min: float
    probability_median: float
    probability_max: float
    selected_threshold: float | None
    selected_threshold_source: str
    threshold_sweep: list[ThresholdSweepRow]
    selected_model_params: dict[str, dict[str, Any]]
    model_diagnostics: dict[str, dict[str, float]]
    reliability_buckets: list[CalibrationBucket]
    candidate_type_thresholds: dict[str, dict[str, Any]]
    candidate_type_calibration: dict[str, dict[str, float]]
    empirical_payoff: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MetaLabelWalkForwardReport:
    samples: int
    folds: list[FoldReport]
    predictions: list[FoldPrediction]
    feature_names: list[str]
    requested_models: list[str]
    calibration_method: str
    stacker_mode: str
    data_coverage: dict[str, Any]

    @property
    def traded_signals(self) -> int:
        return sum(fold.traded_signals for fold in self.folds)

    @property
    def net_pnl(self) -> float:
        return sum(fold.net_pnl for fold in self.folds)


@dataclass(frozen=True, slots=True)
class ModelSmokeResult:
    model_name: str
    ok: bool
    detail: str
    probability_min: float = 0.0
    probability_max: float = 0.0


def _labels(samples: list[MetaLabelSample]) -> Any:
    import numpy as np

    return np.asarray([sample.label for sample in samples], dtype=int)


def _economic_sample_weights(samples: list[MetaLabelSample]) -> Any:
    """Weight training rows by realized economic consequence, clipped for stability."""
    import numpy as np

    if not samples:
        return np.asarray([], dtype=float)
    raw = np.asarray(
        [
            max(
                abs(sample.net_return),
                0.5 * min(sample.net_profit_target, sample.net_stop_loss),
                1e-6,
            )
            for sample in samples
        ],
        dtype=float,
    )
    median_weight = float(np.median(raw))
    if median_weight <= 0:
        return np.ones(len(samples), dtype=float)
    return np.clip(raw / median_weight, 0.50, 3.00)


def _classification_sample_weights(samples: list[MetaLabelSample]) -> Any:
    """Combine economic consequence with label balance for classifiers."""
    import numpy as np

    weights = np.asarray(_economic_sample_weights(samples), dtype=float)
    if len(samples) == 0:
        return weights
    positives = sum(1 for sample in samples if sample.label == 1)
    negatives = len(samples) - positives
    if positives <= 0 or negatives <= 0:
        return weights
    positive_weight = len(samples) / (2 * positives)
    negative_weight = len(samples) / (2 * negatives)
    class_weights = np.asarray(
        [positive_weight if sample.label == 1 else negative_weight for sample in samples],
        dtype=float,
    )
    combined = weights * class_weights
    median_weight = float(np.median(combined))
    if median_weight <= 0:
        return weights
    return np.clip(combined / median_weight, 0.25, 4.00)


def _fit_estimator(estimator: Any, x: Any, y: Any, *, sample_weight: Any | None = None) -> None:
    if sample_weight is None:
        estimator.fit(x, y)
        return
    try:
        estimator.fit(x, y, sample_weight=sample_weight)
        return
    except (TypeError, ValueError) as exc:
        fit_error = exc
    steps = getattr(estimator, "steps", None)
    if steps:
        final_step_name = steps[-1][0]
        estimator.fit(x, y, **{f"{final_step_name}__sample_weight": sample_weight})
        return
    raise fit_error


def _predict_probability(estimator: Any, x: Any) -> Any:
    import numpy as np

    if hasattr(estimator, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            probability = estimator.predict_proba(x)
        probability = np.asarray(probability)
        if probability.ndim == 1:
            return np.clip(probability, 0.0, 1.0)
        if probability.shape[1] == 1:
            classes = getattr(estimator, "classes_", None)
            if classes is not None and len(classes) == 1:
                return np.ones(probability.shape[0]) if int(classes[0]) == 1 else np.zeros(probability.shape[0])
            return np.clip(probability[:, 0], 0.0, 1.0)
        return np.asarray(probability)[:, 1]
    if hasattr(estimator, "decision_function"):
        margin = np.asarray(estimator.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-margin))
    prediction = np.asarray(estimator.predict(x), dtype=float)
    return np.clip(prediction, 0.0, 1.0)


def _logistic_classifier() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 1)


def _model_n_jobs() -> int:
    return _env_int("ZEROALPHA_MODEL_N_JOBS", default=1)


def _gpu_enabled() -> bool:
    return _env_flag("ZEROALPHA_USE_GPU", default=False)


def _lightgbm_classifier(params: dict[str, Any] | None = None) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ModelDependencyError("LightGBM is not installed") from exc

    defaults = {
        "objective": "binary",
        "learning_rate": 0.03,
        "n_estimators": 150,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": _model_n_jobs(),
        "verbosity": -1,
    }
    return lgb.LGBMClassifier(**{**defaults, **(params or {})})


def _catboost_classifier(params: dict[str, Any] | None = None) -> Any:
    try:
        import catboost as cb
    except ImportError as exc:
        raise ModelDependencyError("CatBoost is not installed") from exc

    defaults = {
        "loss_function": "Logloss",
        "iterations": 150,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    if _gpu_enabled():
        defaults.update({"task_type": "GPU", "devices": os.environ.get("ZEROALPHA_GPU_DEVICES", "0")})
    return cb.CatBoostClassifier(**{**defaults, **(params or {})})


def _xgboost_classifier(params: dict[str, Any] | None = None) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ModelDependencyError("XGBoost is not installed") from exc

    defaults = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "n_estimators": 150,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": _model_n_jobs(),
        "tree_method": "hist",
        "verbosity": 0,
    }
    if _gpu_enabled():
        defaults["device"] = os.environ.get("ZEROALPHA_XGBOOST_DEVICE", "cuda")
    return xgb.XGBClassifier(**{**defaults, **(params or {})})


def _hist_gradient_boosting_classifier(params: dict[str, Any] | None = None) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    defaults = {
        "learning_rate": 0.04,
        "max_iter": 180,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.10,
        "early_stopping": True,
        "random_state": 42,
    }
    return HistGradientBoostingClassifier(**{**defaults, **(params or {})})


def _random_forest_classifier(params: dict[str, Any] | None = None) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    defaults = {
        "n_estimators": 400,
        "max_depth": 6,
        "min_samples_leaf": 12,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": _model_n_jobs(),
    }
    return RandomForestClassifier(**{**defaults, **(params or {})})


def _extra_trees_classifier(params: dict[str, Any] | None = None) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    defaults = {
        "n_estimators": 500,
        "max_depth": 7,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": _model_n_jobs(),
    }
    return ExtraTreesClassifier(**{**defaults, **(params or {})})


def _tabpfn_classifier() -> Any:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as exc:
        raise ModelDependencyError("TabPFN is not installed in this Python environment") from exc

    try:
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    except Exception:
        return TabPFNClassifier()


def _tabicl_classifier() -> Any:
    try:
        from tabicl import TabICLClassifier  # type: ignore
    except ImportError as exc:
        raise ModelDependencyError("TabICL/TabICLv2 package is not installed") from exc

    return TabICLClassifier()


def _model_factories() -> dict[str, Callable[[], Any]]:
    return {
        "logistic": _logistic_classifier,
        "lightgbm": _lightgbm_classifier,
        "catboost": _catboost_classifier,
        "xgboost": _xgboost_classifier,
        "histgb": _hist_gradient_boosting_classifier,
        "randomforest": _random_forest_classifier,
        "extratrees": _extra_trees_classifier,
        "tabpfn": _tabpfn_classifier,
        "tabicl": _tabicl_classifier,
        "tabiclv2": _tabicl_classifier,
    }


def _build_model(name: str, params: dict[str, Any] | None = None) -> Any:
    if name == "lightgbm":
        return _lightgbm_classifier(params)
    if name == "catboost":
        return _catboost_classifier(params)
    if name == "xgboost":
        return _xgboost_classifier(params)
    if name == "histgb":
        return _hist_gradient_boosting_classifier(params)
    if name == "randomforest":
        return _random_forest_classifier(params)
    if name == "extratrees":
        return _extra_trees_classifier(params)
    factory = _model_factories()[name]
    return factory()


def _return_regressor() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=120,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=42,
    )


def _sample_group(sample: MetaLabelSample) -> str:
    family = sample.features.get("event_setup_family")
    if isinstance(family, str) and family:
        return family
    candidate_type = sample.candidate_type
    if "breakout" in candidate_type or "breakdown" in candidate_type:
        return "breakout"
    if "reversion" in candidate_type or "sweep" in candidate_type or "reversal" in candidate_type:
        return "mean_reversion"
    if "pullback" in candidate_type or "reclaim" in candidate_type:
        return "pullback_reclaim"
    if "momentum" in candidate_type or "continuation" in candidate_type:
        return "momentum"
    return "global"


def _sample_regime(sample: MetaLabelSample) -> str:
    regime = sample.features.get("market_regime")
    return regime if isinstance(regime, str) and regime else "unknown"


def _threshold_group_key(sample: MetaLabelSample) -> str:
    regime = _sample_regime(sample)
    if regime == "unknown":
        return sample.candidate_type
    return f"{sample.candidate_type}|{regime}"


def _threshold_lookup(
    thresholds: dict[str, dict[str, Any]],
    sample: MetaLabelSample,
) -> dict[str, Any] | None:
    local = thresholds.get(_threshold_group_key(sample))
    family = thresholds.get(sample.candidate_type)
    if local and local.get("source") == "insufficient_calibration" and family and family.get("abstain"):
        return family
    return local or family


def _fit_return_regression_predictions(
    *,
    x_train: Any,
    train_samples: list[MetaLabelSample],
    x_threshold_selection: Any,
    threshold_samples: list[MetaLabelSample],
    x_test: Any,
    test_samples: list[MetaLabelSample],
    specialist_models: bool,
    min_group_samples: int = 50,
) -> tuple[Any, Any, dict[str, dict[str, float]]]:
    import numpy as np

    if len(train_samples) < 20:
        baseline = float(np.mean([sample.net_return for sample in train_samples])) if train_samples else 0.0
        return (
            np.full(len(threshold_samples), baseline),
            np.full(len(test_samples), baseline),
            {"global": {"samples": float(len(train_samples)), "mode": "baseline", "mean_return": baseline}},
        )

    y_train = np.asarray([sample.net_return for sample in train_samples], dtype=float)
    global_model = _return_regressor()
    global_model.fit(x_train, y_train, sample_weight=_economic_sample_weights(train_samples))
    threshold_predictions = np.asarray(global_model.predict(x_threshold_selection), dtype=float)
    test_predictions = np.asarray(global_model.predict(x_test), dtype=float)
    diagnostics: dict[str, dict[str, float]] = {
        "global": {
            "samples": float(len(train_samples)),
            "mean_return": float(np.mean(y_train)),
            "mode": "regression",
        }
    }
    if not specialist_models:
        return threshold_predictions, test_predictions, diagnostics

    groups: dict[str, list[int]] = {}
    for idx, sample in enumerate(train_samples):
        groups.setdefault(_sample_group(sample), []).append(idx)
    for group, indices in sorted(groups.items()):
        if group == "global" or len(indices) < min_group_samples:
            diagnostics[group] = {
                "samples": float(len(indices)),
                "mean_return": float(np.mean(y_train[indices])) if indices else 0.0,
                "mode": "global_fallback",
            }
            continue
        group_model = _return_regressor()
        group_model.fit(x_train[indices], y_train[indices], sample_weight=_economic_sample_weights([train_samples[i] for i in indices]))
        threshold_mask = [idx for idx, sample in enumerate(threshold_samples) if _sample_group(sample) == group]
        test_mask = [idx for idx, sample in enumerate(test_samples) if _sample_group(sample) == group]
        if threshold_mask:
            threshold_predictions[threshold_mask] = group_model.predict(x_threshold_selection[threshold_mask])
        if test_mask:
            test_predictions[test_mask] = group_model.predict(x_test[test_mask])
        diagnostics[group] = {
            "samples": float(len(indices)),
            "mean_return": float(np.mean(y_train[indices])),
            "mode": "specialist_regression",
        }
    return threshold_predictions, test_predictions, diagnostics


def _downside_estimates_by_group(samples: list[MetaLabelSample], *, min_samples: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        grouped.setdefault(_sample_group(sample), []).append(sample.net_return)
        grouped.setdefault(f"{_sample_group(sample)}|{_sample_regime(sample)}", []).append(sample.net_return)
    estimates: dict[str, float] = {}
    for key, returns in grouped.items():
        losses = [-value for value in returns if value < 0]
        estimates[key] = float(sum(losses) / len(losses)) if len(returns) >= min_samples and losses else 0.0
    return estimates


def _predicted_downside(sample: MetaLabelSample, estimates: dict[str, float]) -> float:
    return estimates.get(f"{_sample_group(sample)}|{_sample_regime(sample)}") or estimates.get(_sample_group(sample), 0.0)


def _selection_score(
    *,
    probability: float,
    expected_value: float,
    predicted_return: float,
    predicted_downside: float,
    mode: str,
) -> float:
    if mode == "probability":
        return probability
    if mode == "expected_value":
        return expected_value
    if mode == "predicted_return":
        return predicted_return
    if mode == "expected_utility":
        return predicted_return + 0.25 * expected_value - 0.25 * predicted_downside
    raise ValueError("selection_score must be probability, expected_value, predicted_return, or expected_utility")


def _class_balance_params(name: str, labels: Any) -> dict[str, Any]:
    if name not in {"lightgbm", "catboost", "xgboost", "randomforest", "extratrees"}:
        return {}
    values = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    positives = sum(1 for value in values if int(value) == 1)
    negatives = len(values) - positives
    if positives <= 0 or negatives <= 0:
        return {}
    if name == "lightgbm":
        return {"class_weight": "balanced"}
    if name == "catboost":
        return {"auto_class_weights": "Balanced"}
    if name == "xgboost":
        return {"scale_pos_weight": negatives / positives}
    if name == "randomforest":
        return {"class_weight": "balanced_subsample"}
    if name == "extratrees":
        return {"class_weight": "balanced"}
    return {}


def _hpo_grid(name: str, *, profile: str = "standard") -> list[dict[str, Any]]:
    if name == "lightgbm":
        grid = [
            {},
            {"learning_rate": 0.02, "n_estimators": 250, "num_leaves": 15, "min_child_samples": 30},
            {"learning_rate": 0.05, "n_estimators": 120, "num_leaves": 31, "min_child_samples": 20},
            {"learning_rate": 0.03, "n_estimators": 180, "num_leaves": 63, "min_child_samples": 40},
            {"learning_rate": 0.01, "n_estimators": 300, "num_leaves": 15, "reg_lambda": 5.0},
            {"learning_rate": 0.04, "n_estimators": 160, "num_leaves": 31, "subsample": 0.70},
            {"learning_rate": 0.03, "n_estimators": 150, "num_leaves": 7, "colsample_bytree": 0.70},
            {"learning_rate": 0.06, "n_estimators": 90, "num_leaves": 15, "reg_lambda": 8.0},
        ]
        if profile == "deep":
            grid.extend(
                [
                    {"learning_rate": 0.015, "n_estimators": 360, "num_leaves": 31, "min_child_samples": 50, "reg_lambda": 10.0},
                    {"learning_rate": 0.025, "n_estimators": 260, "num_leaves": 63, "feature_fraction": 0.75, "bagging_fraction": 0.75},
                    {"learning_rate": 0.04, "n_estimators": 220, "num_leaves": 127, "min_child_samples": 80, "reg_alpha": 0.25},
                ]
            )
        return grid
    if name == "catboost":
        grid = [
            {},
            {"iterations": 220, "learning_rate": 0.02, "depth": 4, "l2_leaf_reg": 8.0},
            {"iterations": 120, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 5.0},
            {"iterations": 180, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 8.0},
            {"iterations": 260, "learning_rate": 0.015, "depth": 4, "l2_leaf_reg": 12.0},
            {"iterations": 100, "learning_rate": 0.06, "depth": 3, "l2_leaf_reg": 4.0},
            {"iterations": 160, "learning_rate": 0.03, "depth": 5, "l2_leaf_reg": 12.0},
            {"iterations": 90, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 15.0},
        ]
        if profile == "deep":
            grid.extend(
                [
                    {"iterations": 320, "learning_rate": 0.018, "depth": 4, "l2_leaf_reg": 16.0, "random_strength": 0.5},
                    {"iterations": 240, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 10.0, "random_strength": 1.0},
                    {"iterations": 180, "learning_rate": 0.045, "depth": 5, "l2_leaf_reg": 20.0, "random_strength": 2.0},
                ]
            )
        return grid
    if name == "xgboost":
        grid = [
            {},
            {"learning_rate": 0.02, "n_estimators": 250, "max_depth": 3, "min_child_weight": 8},
            {"learning_rate": 0.05, "n_estimators": 120, "max_depth": 4, "min_child_weight": 5},
            {"learning_rate": 0.03, "n_estimators": 180, "max_depth": 5, "min_child_weight": 8},
            {"learning_rate": 0.01, "n_estimators": 300, "max_depth": 3, "reg_lambda": 5.0},
            {"learning_rate": 0.04, "n_estimators": 160, "max_depth": 4, "subsample": 0.70},
            {"learning_rate": 0.03, "n_estimators": 140, "max_depth": 2, "colsample_bytree": 0.70},
            {"learning_rate": 0.06, "n_estimators": 90, "max_depth": 3, "reg_lambda": 8.0},
        ]
        if profile == "deep":
            grid.extend(
                [
                    {"learning_rate": 0.015, "n_estimators": 360, "max_depth": 4, "min_child_weight": 12, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0},
                    {"learning_rate": 0.025, "n_estimators": 260, "max_depth": 6, "min_child_weight": 8, "subsample": 0.7, "colsample_bytree": 0.65, "reg_alpha": 0.25},
                    {"learning_rate": 0.05, "n_estimators": 180, "max_depth": 3, "min_child_weight": 16, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 15.0},
                ]
            )
        return grid
    if name == "histgb":
        return [
            {},
            {"learning_rate": 0.02, "max_iter": 260, "max_leaf_nodes": 15, "l2_regularization": 0.25},
            {"learning_rate": 0.04, "max_iter": 180, "max_leaf_nodes": 7, "l2_regularization": 0.40},
            {"learning_rate": 0.06, "max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 0.10},
            {"learning_rate": 0.03, "max_iter": 220, "max_leaf_nodes": 31, "l2_regularization": 0.25},
            {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 11, "min_samples_leaf": 25},
        ]
    if name == "randomforest":
        return [
            {},
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt"},
            {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 12, "max_features": "sqrt"},
            {"n_estimators": 500, "max_depth": 9, "min_samples_leaf": 8, "max_features": 0.35},
            {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 20, "max_features": 0.25},
        ]
    if name == "extratrees":
        return [
            {},
            {"n_estimators": 400, "max_depth": 5, "min_samples_leaf": 16, "max_features": "sqrt"},
            {"n_estimators": 600, "max_depth": 8, "min_samples_leaf": 10, "max_features": 0.35},
            {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 24, "max_features": 0.25},
            {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 6, "max_features": "log2"},
        ]
    return [{}]


def _select_hyperparameters(
    *,
    name: str,
    x_train: Any,
    y_train: Any,
    training_samples: list[MetaLabelSample],
    hpo_profile: str = "standard",
) -> dict[str, Any]:
    if name not in {"lightgbm", "catboost", "xgboost", "histgb", "randomforest", "extratrees"}:
        return {}
    from sklearn.metrics import brier_score_loss

    if len(y_train) < 40:
        return {}
    split_at = max(30, int(len(y_train) * 0.80))
    if split_at >= len(y_train):
        return {}
    inner_x_train = x_train[:split_at]
    inner_y_train = y_train[:split_at]
    inner_x_validation = x_train[split_at:]
    inner_y_validation = y_train[split_at:]
    validation_samples = training_samples[split_at:]
    inner_weights = _classification_sample_weights(training_samples[:split_at])
    validation_weights = _classification_sample_weights(validation_samples)
    if len(set(inner_y_train.tolist())) < 2 or len(set(inner_y_validation.tolist())) < 2:
        return {}

    best_params: dict[str, Any] = {}
    best_score: tuple[float, float, float] | None = None
    for params in _hpo_grid(name, profile=hpo_profile):
        balanced_params = {**_class_balance_params(name, inner_y_train), **params}
        estimator = _build_model(name, balanced_params)
        _fit_estimator(estimator, inner_x_train, inner_y_train, sample_weight=inner_weights)
        raw = _predict_probability(estimator, inner_x_validation)
        brier = float(brier_score_loss(inner_y_validation, raw, sample_weight=validation_weights))
        sweep = _threshold_sweep(
            test_samples=validation_samples,
            probabilities=raw,
            thresholds=tuple(round(idx * 0.05, 2) for idx in range(2, 17)),
        )
        min_validation_trades = max(3, min(12, int(len(validation_samples) * 0.05)))
        viable = [row for row in sweep if row.traded_signals >= min_validation_trades]
        best_row = max(
            viable,
            key=lambda row: (
                row.average_trade_return,
                row.net_pnl / max(row.traded_signals**0.5, 1.0),
                row.hit_rate,
            ),
            default=None,
        )
        if best_row is None:
            score = (0.0, 0.0, -brier)
        else:
            score = (best_row.average_trade_return, best_row.net_pnl, -brier)
        if best_score is None or score > best_score:
            best_score = score
            best_params = balanced_params
    return best_params


def _is_foundation_model(name: str) -> bool:
    return name in {"tabicl", "tabiclv2", "tabpfn"}


def _tail_training_window_with_two_classes(
    x_train: Any,
    y_train: Any,
    *,
    initial_samples: int = 128,
    max_samples: int = 1024,
) -> tuple[Any, Any]:
    if len(y_train) <= initial_samples:
        return x_train, y_train
    window = initial_samples
    while window < len(y_train):
        model_y_train = y_train[-window:]
        if len(set(model_y_train.tolist())) >= 2:
            return x_train[-window:], model_y_train
        if window >= max_samples:
            break
        window = min(window * 2, max_samples, len(y_train))
    capped = min(max_samples, len(y_train))
    return x_train[-capped:], y_train[-capped:]


def _predict_probability_batched(estimator: Any, x: Any, *, batch_size: int = 256) -> Any:
    import numpy as np

    if len(x) <= batch_size:
        return _predict_probability(estimator, x)
    parts = [
        _predict_probability(estimator, x[start : start + batch_size])
        for start in range(0, len(x), batch_size)
    ]
    return np.concatenate(parts)


def _foundation_fit_predict_arrays(
    model_name: str,
    x_train: Any,
    y_train: Any,
    x_base_calibration: Any,
    x_ensemble_calibration: Any,
    x_threshold_selection: Any,
    x_test: Any,
) -> tuple[Any, Any, Any, Any]:
    factory = _model_factories()[model_name]
    estimator = factory()
    estimator.fit(x_train, y_train)
    return (
        _predict_probability_batched(estimator, x_base_calibration),
        _predict_probability_batched(estimator, x_ensemble_calibration),
        _predict_probability_batched(estimator, x_threshold_selection),
        _predict_probability_batched(estimator, x_test),
    )


def _foundation_fit_predict_worker(model_name: str, input_path: str, output_path: str) -> None:
    import numpy as np

    data = np.load(input_path)
    (
        base_calibration_probability,
        ensemble_probability,
        threshold_probability,
        test_probability,
    ) = _foundation_fit_predict_arrays(
        model_name,
        data["x_train"],
        data["y_train"],
        data["x_base_calibration"],
        data["x_ensemble_calibration"],
        data["x_threshold_selection"],
        data["x_test"],
    )
    np.savez(
        output_path,
        base_calibration_probability=base_calibration_probability,
        ensemble_probability=ensemble_probability,
        threshold_probability=threshold_probability,
        test_probability=test_probability,
    )


def _foundation_fit_predict_isolated(
    model_name: str,
    x_train: Any,
    y_train: Any,
    x_base_calibration: Any,
    x_ensemble_calibration: Any,
    x_threshold_selection: Any,
    x_test: Any,
    *,
    timeout_seconds: int = 180,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="zeroalpha_foundation_") as tmp:
        input_path = Path(tmp) / "input.npz"
        output_path = Path(tmp) / "output.npz"
        np.savez(
            input_path,
            x_train=x_train,
            y_train=y_train,
            x_base_calibration=x_base_calibration,
            x_ensemble_calibration=x_ensemble_calibration,
            x_threshold_selection=x_threshold_selection,
            x_test=x_test,
        )
        code = (
            "from zeroalpha.models.ensemble import _foundation_fit_predict_worker;"
            f"_foundation_fit_predict_worker({model_name!r}, {str(input_path)!r}, {str(output_path)!r})"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "TABPFN_DISABLE_TELEMETRY": "1"},
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}"
            raise ModelDependencyError(detail)
        data = np.load(output_path)
        return (
            data["base_calibration_probability"],
            data["ensemble_probability"],
            data["threshold_probability"],
            data["test_probability"],
        )


def _smoke_test_model_stack_in_process(model_names: list[str]) -> list[ModelSmokeResult]:
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 8))
    y = ((x[:, 0] + 0.6 * x[:, 1] - 0.4 * x[:, 2]) > 0).astype(int)
    factories = _model_factories()
    results: list[ModelSmokeResult] = []
    for name in model_names:
        factory = factories.get(name)
        if factory is None:
            results.append(ModelSmokeResult(name, False, "unknown model name"))
            continue
        try:
            estimator = factory()
            estimator.fit(x[:60], y[:60])
            probabilities = _predict_probability(estimator, x[60:70])
            p_min, _, p_max = _probability_summary(probabilities)
            results.append(ModelSmokeResult(name, True, "fit_predict_ok", p_min, p_max))
        except Exception as exc:
            results.append(ModelSmokeResult(name, False, f"{type(exc).__name__}: {exc}"))
    return results


def smoke_test_model_stack(
    model_names: list[str] | None = None,
    *,
    isolated: bool = False,
    timeout_seconds: int = 90,
) -> list[ModelSmokeResult]:
    names = model_names or [
        "logistic",
        "histgb",
        "randomforest",
        "extratrees",
        "lightgbm",
        "catboost",
        "xgboost",
        "tabicl",
        "tabpfn",
    ]
    if not isolated:
        return _smoke_test_model_stack_in_process(names)

    results: list[ModelSmokeResult] = []
    for name in names:
        code = (
            "from dataclasses import asdict;"
            "import json;"
            "from zeroalpha.models.ensemble import smoke_test_model_stack;"
            f"r=smoke_test_model_stack([{name!r}], isolated=False);"
            "print('__ZEROALPHA_SMOKE__'+json.dumps([asdict(x) for x in r]))"
        )
        env = {**os.environ, "TABPFN_DISABLE_TELEMETRY": "1"}
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired:
            results.append(ModelSmokeResult(name, False, f"timeout_after_{timeout_seconds}s"))
            continue
        marker_payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("__ZEROALPHA_SMOKE__"):
                marker_payload = line.removeprefix("__ZEROALPHA_SMOKE__")
        if marker_payload is None:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}"
            results.append(ModelSmokeResult(name, False, detail))
            continue
        for item in json.loads(marker_payload):
            results.append(ModelSmokeResult(**item))
    return results


def _fit_base_prediction_sets(
    *,
    x_train: Any,
    y_train: Any,
    x_base_calibration: Any,
    y_base_calibration: Any,
    x_ensemble_calibration: Any,
    x_threshold_selection: Any,
    x_test: Any,
    train_samples: list[MetaLabelSample],
    model_names: list[str],
    calibration_method: str,
    tune_hyperparameters: bool,
    hpo_profile: str = "standard",
    foundation_max_samples: int = 1024,
) -> tuple[list[BaseFoldPredictionSet], dict[str, str]]:
    prediction_sets: list[BaseFoldPredictionSet] = []
    skipped: dict[str, str] = {}
    if len(set(y_train.tolist())) < 2:
        return prediction_sets, {name: "training fold has one class" for name in model_names}
    factories = _model_factories()
    for name in model_names:
        factory = factories.get(name)
        if factory is None:
            skipped[name] = "unknown model name"
            continue
        try:
            model_x_train = x_train
            model_y_train = y_train
            if _is_foundation_model(name):
                model_x_train, model_y_train = _tail_training_window_with_two_classes(
                    x_train,
                    y_train,
                    max_samples=foundation_max_samples,
                )
                if len(set(model_y_train.tolist())) < 2:
                    skipped[name] = "bounded foundation training window has one class"
                    continue
            selected_params: dict[str, Any] = {}
            if _is_foundation_model(name):
                (
                    raw_base_probability,
                    raw_ensemble_probability,
                    raw_threshold_probability,
                    raw_test_probability,
                ) = _foundation_fit_predict_isolated(
                    name,
                    model_x_train,
                    model_y_train,
                    x_base_calibration,
                    x_ensemble_calibration,
                    x_threshold_selection,
                    x_test,
                )
                calibrator = ProbabilityCalibrator.fit(
                    raw_base_probability,
                    y_base_calibration,
                    method=calibration_method,
                )
                ensemble_probability = calibrator.predict(raw_ensemble_probability)
                threshold_probability = calibrator.predict(raw_threshold_probability)
                test_probability = calibrator.predict(raw_test_probability)
            else:
                if tune_hyperparameters:
                    selected_params = _select_hyperparameters(
                        name=name,
                        x_train=model_x_train,
                        y_train=model_y_train,
                        training_samples=train_samples[-len(model_y_train) :],
                        hpo_profile=hpo_profile,
                    )
                if name in {"lightgbm", "catboost", "xgboost", "randomforest", "extratrees"}:
                    selected_params = {**_class_balance_params(name, model_y_train), **selected_params}
                estimator = _build_model(name, selected_params)
                _fit_estimator(
                    estimator,
                    model_x_train,
                    model_y_train,
                    sample_weight=_classification_sample_weights(train_samples[-len(model_y_train) :]),
                )
                raw_base_probability = _predict_probability(estimator, x_base_calibration)
                calibrator = ProbabilityCalibrator.fit(
                    raw_base_probability,
                    y_base_calibration,
                    method=calibration_method,
                )
                ensemble_probability = calibrator.predict(
                    _predict_probability(estimator, x_ensemble_calibration)
                )
                threshold_probability = calibrator.predict(
                    _predict_probability(estimator, x_threshold_selection)
                )
                test_probability = calibrator.predict(_predict_probability(estimator, x_test))
            prediction_sets.append(
                BaseFoldPredictionSet(
                    name=name,
                    ensemble_probability=ensemble_probability,
                    threshold_probability=threshold_probability,
                    test_probability=test_probability,
                    selected_params=selected_params,
                )
            )
        except subprocess.TimeoutExpired:
            skipped[name] = "timeout_in_isolated_foundation_worker"
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return prediction_sets, skipped


def _fit_stacker(base_probabilities: Any, labels: Any, *, sample_weight: Any | None = None) -> Any | None:
    if base_probabilities.shape[1] < 2 or len(set(labels.tolist())) < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    stacker = LogisticRegression(C=0.25, max_iter=1000)
    if sample_weight is None:
        stacker.fit(base_probabilities, labels)
    else:
        stacker.fit(base_probabilities, labels, sample_weight=sample_weight)
    return stacker


def _stack_probabilities(stacker: Any | None, base_probabilities: Any) -> Any:
    import numpy as np

    if base_probabilities.shape[1] == 0:
        return np.asarray([], dtype=float)
    if stacker is None:
        return np.mean(base_probabilities, axis=1)
    return stacker.predict_proba(base_probabilities)[:, 1]


def _base_validation_scores(
    *,
    calibration_samples: list[MetaLabelSample],
    labels: Any,
    calibration_matrix: Any,
) -> list[tuple[float, float]]:
    from sklearn.metrics import brier_score_loss

    scores: list[tuple[float, float]] = []
    thresholds = tuple(round(idx * 0.05, 2) for idx in range(1, 20))
    for column_idx in range(calibration_matrix.shape[1]):
        probabilities = calibration_matrix[:, column_idx]
        brier = float(brier_score_loss(labels, probabilities))
        sweep = _threshold_sweep(
            test_samples=calibration_samples,
            probabilities=probabilities,
            thresholds=thresholds,
        )
        utility = max((row.net_pnl for row in sweep if row.traded_signals >= 3), default=0.0)
        scores.append((utility, brier))
    return scores


def _weighted_average_probabilities(base_probabilities: Any, scores: list[tuple[float, float]]) -> Any:
    weights = _validation_weights(scores)
    return base_probabilities @ weights


def _validation_weights(scores: list[tuple[float, float]]) -> Any:
    import numpy as np

    utilities = np.asarray([max(0.0, utility) for utility, _ in scores], dtype=float)
    if utilities.sum() > 0:
        return utilities / utilities.sum()
    else:
        inverse_brier = np.asarray([1.0 / max(brier, 1e-9) for _, brier in scores], dtype=float)
        return inverse_brier / inverse_brier.sum()


def _base_model_diagnostics(
    *,
    prediction_sets: list[BaseFoldPredictionSet],
    scores: list[tuple[float, float]],
    weights: Any,
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for idx, prediction_set in enumerate(prediction_sets):
        utility, brier = scores[idx]
        probabilities = prediction_set.ensemble_probability
        probability_min, probability_median, probability_max = _probability_summary(probabilities)
        diagnostics[prediction_set.name] = {
            "validation_utility": float(utility),
            "validation_brier": float(brier),
            "validation_weight": float(weights[idx]) if len(weights) > idx else 0.0,
            "validation_probability_min": probability_min,
            "validation_probability_median": probability_median,
            "validation_probability_max": probability_max,
        }
    return diagnostics


def _probability_summary(probabilities: Any) -> tuple[float, float, float]:
    import numpy as np

    if len(probabilities) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.min(probabilities)),
        float(np.median(probabilities)),
        float(np.max(probabilities)),
    )


def _reliability_buckets(labels: Any, probabilities: Any, *, bucket_count: int = 10) -> list[CalibrationBucket]:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rows: list[CalibrationBucket] = []
    for idx in range(bucket_count):
        lower = idx / bucket_count
        upper = (idx + 1) / bucket_count
        if idx == bucket_count - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            rows.append(CalibrationBucket(lower, upper, 0, 0.0, 0.0))
            continue
        rows.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                count=int(np.sum(mask)),
                average_probability=float(np.mean(p[mask])),
                observed_rate=float(np.mean(y[mask])),
            )
        )
    return rows


def _candidate_type_calibration(
    samples: list[MetaLabelSample],
    probabilities: Any,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for sample, probability in zip(samples, probabilities, strict=True):
        bucket = buckets.setdefault(
            sample.candidate_type,
            {"count": 0.0, "labels": 0.0, "probability_sum": 0.0},
        )
        bucket["count"] += 1
        bucket["labels"] += sample.label
        bucket["probability_sum"] += float(probability)
    for bucket in buckets.values():
        count = bucket["count"]
        bucket["observed_rate"] = bucket["labels"] / count if count else 0.0
        bucket["average_probability"] = bucket["probability_sum"] / count if count else 0.0
    return buckets


def _threshold_sweep(
    *,
    test_samples: list[MetaLabelSample],
    probabilities: Any,
    thresholds: tuple[float, ...],
) -> list[ThresholdSweepRow]:
    rows: list[ThresholdSweepRow] = []
    for threshold in thresholds:
        selected = [
            (sample, float(probability))
            for sample, probability in zip(test_samples, probabilities, strict=True)
            if float(probability) >= threshold
        ]
        if not selected:
            rows.append(
                ThresholdSweepRow(
                    threshold=threshold,
                    traded_signals=0,
                    hit_rate=0.0,
                    net_pnl=0.0,
                    average_trade_return=0.0,
                )
            )
            continue
        labels = [sample.label for sample, _ in selected]
        returns = [sample.net_return for sample, _ in selected]
        pnl = sum(sample.notional * sample.net_return for sample, _ in selected)
        rows.append(
            ThresholdSweepRow(
                threshold=threshold,
                traded_signals=len(selected),
                hit_rate=sum(labels) / len(labels),
                net_pnl=pnl,
                average_trade_return=sum(returns) / len(returns),
            )
        )
    return rows


def _payoff_key(sample: MetaLabelSample) -> str:
    return f"{sample.candidate_type}|{sample.side}"


def _payoff_estimate(
    samples: list[MetaLabelSample],
    *,
    min_samples: int,
) -> dict[str, Any]:
    label_one_returns = [sample.net_return for sample in samples if sample.label == 1]
    label_zero_returns = [sample.net_return for sample in samples if sample.label == 0]
    wins = [value for value in label_one_returns if value > 0]
    losses = [-value for value in label_zero_returns if value < 0]
    if len(samples) < min_samples or not label_one_returns or not label_zero_returns:
        return {
            "count": float(len(samples)),
            "wins": float(len(wins)),
            "losses": float(len(losses)),
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_label_one_return": 0.0,
            "average_label_zero_return": 0.0,
            "source": "static_fallback",
        }
    return {
        "count": float(len(samples)),
        "wins": float(len(wins)),
        "losses": float(len(losses)),
        "average_win": float(sum(wins) / len(wins)) if wins else 0.0,
        "average_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "average_label_one_return": float(sum(label_one_returns) / len(label_one_returns)),
        "average_label_zero_return": float(sum(label_zero_returns) / len(label_zero_returns)),
        "source": "calibration",
    }


def _payoff_estimates_by_type_and_side(
    samples: list[MetaLabelSample],
    *,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[MetaLabelSample]] = {}
    for sample in samples:
        grouped.setdefault(_payoff_key(sample), []).append(sample)
    return {
        key: _payoff_estimate(group_samples, min_samples=min_samples)
        for key, group_samples in sorted(grouped.items())
    }


def _expected_value(
    *,
    probability: float,
    sample: MetaLabelSample,
    empirical_payoff_ev: bool,
    payoff_estimates: dict[str, dict[str, Any]],
) -> float:
    if empirical_payoff_ev:
        estimate = payoff_estimates.get(_payoff_key(sample))
        if estimate and estimate.get("source") == "calibration":
            if "average_label_one_return" in estimate and "average_label_zero_return" in estimate:
                return (
                    probability * float(estimate["average_label_one_return"])
                    + (1.0 - probability) * float(estimate["average_label_zero_return"])
                )
            return (
                probability * float(estimate["average_win"])
                - (1.0 - probability) * float(estimate["average_loss"])
            )
    return probability * sample.net_profit_target - (1.0 - probability) * sample.net_stop_loss


def _type_threshold_allows_ev(
    type_threshold: dict[str, Any] | None,
    *,
    expected_value: float,
    minimum_expected_value: float,
) -> bool:
    if expected_value >= minimum_expected_value:
        return True
    return bool(
        type_threshold
        and type_threshold.get("source") == "candidate_type_calibration"
        and float(type_threshold.get("average_trade_return", 0.0)) > 0
    )


def _passes_selection_and_ev_gate(
    *,
    selection_score_mode: str,
    selection_score: float,
    expected_value: float,
    minimum_expected_value: float,
    type_threshold: dict[str, Any] | None = None,
) -> bool:
    ev_ok = _type_threshold_allows_ev(
        type_threshold,
        expected_value=expected_value,
        minimum_expected_value=minimum_expected_value,
    )
    if selection_score_mode in {"predicted_return", "expected_utility"}:
        return selection_score > 0 and ev_ok
    return ev_ok


def _select_threshold_from_calibration(
    *,
    calibration_samples: list[MetaLabelSample],
    probabilities: Any,
    thresholds: tuple[float, ...],
    min_trades: int,
    minimum_threshold: float,
) -> ThresholdSweepRow | None:
    viable = [
        row
        for row in _threshold_sweep(
            test_samples=calibration_samples,
            probabilities=probabilities,
            thresholds=thresholds,
        )
        if row.traded_signals >= min_trades and row.threshold >= minimum_threshold
    ]
    positive = [row for row in viable if row.net_pnl > 0 and row.average_trade_return > 0]
    if not positive:
        return None
    return max(positive, key=lambda row: (row.average_trade_return, row.threshold, row.net_pnl))


def _select_candidate_type_thresholds(
    *,
    calibration_samples: list[MetaLabelSample],
    probabilities: Any,
    thresholds: tuple[float, ...],
    min_trades: int,
    minimum_threshold: float,
    utility_samples: list[MetaLabelSample] | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[MetaLabelSample, float]]] = {}
    for sample, probability in zip(calibration_samples, probabilities, strict=True):
        grouped.setdefault(_threshold_group_key(sample), []).append((sample, float(probability)))
    prior_grouped: dict[str, list[MetaLabelSample]] = {}
    family_prior_grouped: dict[str, list[MetaLabelSample]] = {}
    for sample in utility_samples or calibration_samples:
        prior_grouped.setdefault(_threshold_group_key(sample), []).append(sample)
        family_prior_grouped.setdefault(sample.candidate_type, []).append(sample)

    selections: dict[str, dict[str, Any]] = {}
    for candidate_type, rows in sorted(grouped.items()):
        group_samples = [sample for sample, _ in rows]
        group_probabilities = [probability for _, probability in rows]
        average_target = sum(sample.net_profit_target for sample in group_samples) / len(group_samples)
        utility_floor = max(0.001, 0.15 * average_target)
        prior_samples = prior_grouped.get(candidate_type, group_samples)
        prior_average_return = (
            sum(sample.net_return for sample in prior_samples) / len(prior_samples)
            if prior_samples
            else 0.0
        )
        prior_hit_rate = (
            sum(sample.label for sample in prior_samples) / len(prior_samples)
            if prior_samples
            else 0.0
        )
        sweep = _threshold_sweep(
            test_samples=group_samples,
            probabilities=group_probabilities,
            thresholds=thresholds,
        )
        viable = [
            row
            for row in sweep
            if row.traded_signals >= min_trades and row.threshold >= minimum_threshold
        ]
        positive = [
            row
            for row in viable
            if row.net_pnl > 0 and row.average_trade_return >= utility_floor
        ]
        best = max(positive, key=lambda row: (row.average_trade_return, row.threshold, row.net_pnl)) if positive else None
        best_observed = max(sweep, key=lambda row: (row.net_pnl, row.average_trade_return, row.threshold))
        if best is not None:
            selections[candidate_type] = {
                "threshold": best.threshold,
                "source": "candidate_type_calibration",
                "abstain": False,
                "calibration_samples": len(group_samples),
                "prior_samples": len(prior_samples),
                "prior_average_trade_return": prior_average_return,
                "prior_hit_rate": prior_hit_rate,
                "utility_floor": utility_floor,
                "traded_signals": best.traded_signals,
                "hit_rate": best.hit_rate,
                "net_pnl": best.net_pnl,
                "average_trade_return": best.average_trade_return,
            }
        elif len(group_samples) >= min_trades:
            selections[candidate_type] = {
                "threshold": None,
                "source": "negative_calibration_utility",
                "abstain": True,
                "calibration_samples": len(group_samples),
                "prior_samples": len(prior_samples),
                "prior_average_trade_return": prior_average_return,
                "prior_hit_rate": prior_hit_rate,
                "utility_floor": utility_floor,
                "traded_signals": best_observed.traded_signals,
                "hit_rate": best_observed.hit_rate,
                "net_pnl": best_observed.net_pnl,
                "average_trade_return": best_observed.average_trade_return,
            }
        else:
            selections[candidate_type] = {
                "threshold": None,
                "source": "insufficient_calibration",
                "abstain": False,
                "calibration_samples": len(group_samples),
                "prior_samples": len(prior_samples),
                "prior_average_trade_return": prior_average_return,
                "prior_hit_rate": prior_hit_rate,
                "utility_floor": utility_floor,
                "traded_signals": 0,
                "hit_rate": 0.0,
                "net_pnl": 0.0,
                "average_trade_return": 0.0,
            }
    for candidate_type, prior_samples in sorted(family_prior_grouped.items()):
        if candidate_type in selections or len(prior_samples) < min_trades:
            continue
        average_return = sum(sample.net_return for sample in prior_samples) / len(prior_samples)
        average_target = sum(sample.net_profit_target for sample in prior_samples) / len(prior_samples)
        utility_floor = max(0.001, 0.15 * average_target)
        selections[candidate_type] = {
            "threshold": None,
            "source": "negative_family_prior_utility" if average_return < utility_floor else "family_prior",
            "abstain": average_return < utility_floor,
            "calibration_samples": 0,
            "prior_samples": len(prior_samples),
            "prior_average_trade_return": average_return,
            "prior_hit_rate": sum(sample.label for sample in prior_samples) / len(prior_samples),
            "utility_floor": utility_floor,
            "traded_signals": 0,
            "hit_rate": 0.0,
            "net_pnl": 0.0,
            "average_trade_return": average_return,
        }
    return selections


def _days_spanned(samples: list[MetaLabelSample]) -> float:
    if len(samples) < 2:
        return 1.0
    ordered = sorted(sample.timestamp_utc for sample in samples)
    seconds = (ordered[-1] - ordered[0]).total_seconds()
    return max(seconds / 86_400, 1 / 24)


def _select_threshold_for_target_frequency(
    *,
    calibration_samples: list[MetaLabelSample],
    probabilities: Any,
    thresholds: tuple[float, ...],
    target_trades_per_day: float,
) -> ThresholdSweepRow | None:
    if target_trades_per_day <= 0:
        return None
    days = _days_spanned(calibration_samples)
    rows = _threshold_sweep(
        test_samples=calibration_samples,
        probabilities=probabilities,
        thresholds=thresholds,
    )
    viable = [row for row in rows if row.traded_signals / days >= target_trades_per_day]
    if viable:
        return max(viable, key=lambda row: (row.threshold, row.average_trade_return, row.net_pnl))
    populated = [row for row in rows if row.traded_signals > 0]
    if not populated:
        return None
    return min(
        populated,
        key=lambda row: (abs(row.traded_signals / days - target_trades_per_day), -row.threshold),
    )


def _select_target_frequency_event_ids(
    *,
    test_samples: list[MetaLabelSample],
    probabilities: Any,
    target_trades_per_day: float,
    selected_threshold: float,
    config: AppConfig,
    allow_negative_ev: bool,
    selection_score_mode: str = "expected_value",
    candidate_type_thresholds: dict[str, dict[str, Any]] | None = None,
    empirical_payoff_ev: bool = False,
    payoff_estimates: dict[str, dict[str, Any]] | None = None,
    predicted_returns: Any | None = None,
    predicted_downsides: Any | None = None,
    require_calibrated_selection: bool = False,
    min_signal_spacing_hours: float = 0.0,
    max_signals_per_group_per_day: int = 0,
    max_signals_per_timestamp: int = 0,
) -> set[str]:
    if target_trades_per_day <= 0:
        return set()
    if predicted_returns is None:
        predicted_returns = [0.0 for _ in test_samples]
    if predicted_downsides is None:
        predicted_downsides = [0.0 for _ in test_samples]
    grouped: dict[object, list[tuple[MetaLabelSample, float, float, float, float, str]]] = {}
    for sample, probability, predicted_return, predicted_downside in zip(
        test_samples,
        probabilities,
        predicted_returns,
        predicted_downsides,
        strict=True,
    ):
        probability = float(probability)
        predicted_return = float(predicted_return)
        predicted_downside = float(predicted_downside)
        expected_value = _expected_value(
            probability=probability,
            sample=sample,
            empirical_payoff_ev=empirical_payoff_ev,
            payoff_estimates=payoff_estimates or {},
        )
        score = _selection_score(
            probability=probability,
            expected_value=expected_value,
            predicted_return=predicted_return,
            predicted_downside=predicted_downside,
            mode=selection_score_mode,
        )
        threshold = selected_threshold
        type_threshold = None
        type_utility = 0.0
        if candidate_type_thresholds:
            type_threshold = _threshold_lookup(candidate_type_thresholds, sample)
            if require_calibrated_selection and (
                type_threshold is None or type_threshold.get("source") != "candidate_type_calibration"
            ):
                continue
            if type_threshold and type_threshold.get("abstain"):
                continue
            if type_threshold and type_threshold.get("threshold") is not None:
                threshold = max(threshold, float(type_threshold["threshold"]))
                type_utility = float(type_threshold.get("average_trade_return", 0.0))
        if probability < threshold:
            continue
        if not allow_negative_ev and not _passes_selection_and_ev_gate(
            selection_score_mode=selection_score_mode,
            selection_score=score,
            expected_value=expected_value,
            minimum_expected_value=config.model.minimum_expected_value,
            type_threshold=type_threshold,
        ):
            continue
        grouped.setdefault(sample.timestamp_utc.date(), []).append(
            (sample, probability, expected_value, type_utility, score, _threshold_group_key(sample))
        )

    selected: set[str] = set()
    spacing = timedelta(hours=max(min_signal_spacing_hours, 0.0))
    whole_day_quota = math.floor(target_trades_per_day)
    fractional_quota = target_trades_per_day - whole_day_quota
    fractional_accumulator = 0.0
    for day in sorted(grouped):
        quota = whole_day_quota
        fractional_accumulator += fractional_quota
        if fractional_accumulator >= 1.0:
            quota += 1
            fractional_accumulator -= 1.0
        if quota <= 0:
            continue
        ranked = sorted(grouped[day], key=lambda row: (row[4], row[3], row[2], row[1]), reverse=True)
        selected_for_day = 0
        selected_times_by_group: dict[str, list[Any]] = {}
        selected_counts_by_group: dict[str, int] = {}
        selected_counts_by_timestamp: dict[Any, int] = {}
        for sample, _, _, _, _, group_key in ranked:
            if selected_for_day >= quota:
                break
            if (
                max_signals_per_timestamp > 0
                and selected_counts_by_timestamp.get(sample.timestamp_utc, 0) >= max_signals_per_timestamp
            ):
                continue
            if max_signals_per_group_per_day > 0 and selected_counts_by_group.get(group_key, 0) >= max_signals_per_group_per_day:
                continue
            if spacing > timedelta(0):
                group_times = selected_times_by_group.get(group_key, [])
                if any(abs(sample.timestamp_utc - timestamp) < spacing for timestamp in group_times):
                    continue
            selected.add(sample.event_id)
            selected_for_day += 1
            selected_times_by_group.setdefault(group_key, []).append(sample.timestamp_utc)
            selected_counts_by_group[group_key] = selected_counts_by_group.get(group_key, 0) + 1
            selected_counts_by_timestamp[sample.timestamp_utc] = selected_counts_by_timestamp.get(sample.timestamp_utc, 0) + 1
    return selected


def _split_calibration_samples(
    calibration_samples: list[MetaLabelSample],
) -> tuple[list[MetaLabelSample], list[MetaLabelSample], list[MetaLabelSample]]:
    """Split calibration into base, ensemble, and threshold-selection slices.

    Small or class-degenerate calibration windows fall back to the full slice for both
    uses. That keeps tiny smoke tests runnable while avoiding calibration reuse on
    normal walk-forward folds.
    """
    if len(calibration_samples) < 18:
        return calibration_samples, calibration_samples, calibration_samples
    first = max(4, len(calibration_samples) // 3)
    second = max(first + 4, (2 * len(calibration_samples)) // 3)
    second = min(second, len(calibration_samples) - 4)
    base = calibration_samples[:first]
    ensemble = calibration_samples[first:second]
    threshold = calibration_samples[second:]
    if min(len(base), len(ensemble), len(threshold)) <= 0:
        return calibration_samples, calibration_samples, calibration_samples
    if (
        len({sample.label for sample in base}) < 2
        or len({sample.label for sample in ensemble}) < 2
        or len({sample.label for sample in threshold}) < 2
    ):
        return calibration_samples, calibration_samples, calibration_samples
    return base, ensemble, threshold


def _score_fold(
    *,
    fold_id: int,
    train_samples: list[MetaLabelSample],
    calibration_samples: list[MetaLabelSample],
    test_samples: list[MetaLabelSample],
    model_names: list[str],
    calibration_method: str,
    config: AppConfig,
    adaptive_threshold: bool,
    min_calibration_trades: int,
    stacker_mode: str,
    adaptive_minimum_threshold: float,
    tune_hyperparameters: bool,
    hpo_profile: str,
    foundation_max_samples: int,
    target_trades_per_day: float | None,
    allow_negative_ev_target_frequency: bool,
    candidate_type_thresholds: bool,
    empirical_payoff_ev: bool,
    selection_score_mode: str,
    specialist_models: bool,
    require_calibrated_selection: bool,
    min_signal_spacing_hours: float,
    max_signals_per_group_per_day: int,
    max_signals_per_timestamp: int,
) -> tuple[FoldReport, list[FoldPrediction], FeatureEncoder]:
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss

    encoder = FeatureEncoder.fit(train_samples)
    base_calibration_samples, ensemble_samples, threshold_samples = _split_calibration_samples(
        calibration_samples
    )
    x_train = encoder.transform(train_samples)
    x_base_calibration = encoder.transform(base_calibration_samples)
    x_ensemble_calibration = encoder.transform(ensemble_samples)
    x_threshold_selection = encoder.transform(threshold_samples)
    x_test = encoder.transform(test_samples)
    y_train = _labels(train_samples)
    y_base_calibration = _labels(base_calibration_samples)
    y_ensemble = _labels(ensemble_samples)
    y_test = _labels(test_samples)

    prediction_sets, skipped_models = _fit_base_prediction_sets(
        x_train=x_train,
        y_train=y_train,
        x_base_calibration=x_base_calibration,
        y_base_calibration=y_base_calibration,
        x_ensemble_calibration=x_ensemble_calibration,
        x_threshold_selection=x_threshold_selection,
        x_test=x_test,
        train_samples=train_samples,
        model_names=model_names,
        calibration_method=calibration_method,
        tune_hyperparameters=tune_hyperparameters,
        hpo_profile=hpo_profile,
        foundation_max_samples=foundation_max_samples,
    )
    if not prediction_sets:
        constant = float(y_train.mean()) if len(y_train) else 0.0
        probabilities = np.full(len(test_samples), constant)
        ensemble_probabilities = np.full(len(ensemble_samples), constant)
        threshold_probabilities = np.full(len(threshold_samples), constant)
        model_diagnostics: dict[str, dict[str, float]] = {}
    else:
        ensemble_matrix = np.column_stack(
            [prediction_set.ensemble_probability for prediction_set in prediction_sets]
        )
        threshold_matrix = np.column_stack(
            [prediction_set.threshold_probability for prediction_set in prediction_sets]
        )
        test_matrix = np.column_stack([prediction_set.test_probability for prediction_set in prediction_sets])
        validation_scores = _base_validation_scores(
            calibration_samples=ensemble_samples,
            labels=y_ensemble,
            calibration_matrix=ensemble_matrix,
        )
        if stacker_mode == "logistic":
            stacker = _fit_stacker(
                ensemble_matrix,
                y_ensemble,
                sample_weight=_classification_sample_weights(ensemble_samples),
            )
            ensemble_probabilities = _stack_probabilities(stacker, ensemble_matrix)
            threshold_probabilities = _stack_probabilities(stacker, threshold_matrix)
            probabilities = _stack_probabilities(stacker, test_matrix)
            validation_weights = np.full(len(prediction_sets), 1 / len(prediction_sets))
        elif stacker_mode == "average":
            ensemble_probabilities = _stack_probabilities(None, ensemble_matrix)
            threshold_probabilities = _stack_probabilities(None, threshold_matrix)
            probabilities = _stack_probabilities(None, test_matrix)
            validation_weights = np.full(len(prediction_sets), 1 / len(prediction_sets))
        elif stacker_mode in {"best", "weighted"}:
            if stacker_mode == "best":
                best_idx = max(range(len(validation_scores)), key=lambda idx: (validation_scores[idx][0], -validation_scores[idx][1]))
                ensemble_probabilities = ensemble_matrix[:, best_idx]
                threshold_probabilities = threshold_matrix[:, best_idx]
                probabilities = test_matrix[:, best_idx]
                validation_weights = np.zeros(len(prediction_sets))
                validation_weights[best_idx] = 1.0
            else:
                validation_weights = _validation_weights(validation_scores)
                ensemble_probabilities = ensemble_matrix @ validation_weights
                threshold_probabilities = threshold_matrix @ validation_weights
                probabilities = test_matrix @ validation_weights
        else:
            raise ValueError("stacker_mode must be average, logistic, best, or weighted")
        model_diagnostics = _base_model_diagnostics(
            prediction_sets=prediction_sets,
            scores=validation_scores,
            weights=validation_weights,
        )

    ensemble_calibrator = ProbabilityCalibrator.fit(
        ensemble_probabilities,
        y_ensemble,
        method=calibration_method,
    )
    threshold_probabilities = ensemble_calibrator.predict(threshold_probabilities)
    probabilities = ensemble_calibrator.predict(probabilities)
    payoff_estimates = _payoff_estimates_by_type_and_side(
        threshold_samples,
        min_samples=min_calibration_trades,
    )
    threshold_predicted_returns, test_predicted_returns, specialist_diagnostics = _fit_return_regression_predictions(
        x_train=x_train,
        train_samples=train_samples,
        x_threshold_selection=x_threshold_selection,
        threshold_samples=threshold_samples,
        x_test=x_test,
        test_samples=test_samples,
        specialist_models=specialist_models,
    )
    downside_estimates = _downside_estimates_by_group(threshold_samples, min_samples=min_calibration_trades)
    test_predicted_downsides = [
        _predicted_downside(sample, downside_estimates) for sample in test_samples
    ]

    threshold_grid = tuple([round(idx * 0.01, 2) for idx in range(101)])
    selected_threshold_row = None
    selected_threshold_source = "config"
    type_thresholds: dict[str, dict[str, Any]] = {}
    if candidate_type_thresholds:
        type_thresholds = _select_candidate_type_thresholds(
            calibration_samples=threshold_samples,
            probabilities=threshold_probabilities,
            thresholds=threshold_grid,
            min_trades=min_calibration_trades,
            minimum_threshold=adaptive_minimum_threshold,
            utility_samples=calibration_samples,
        )
    if target_trades_per_day and target_trades_per_day > 0:
        selected_threshold_row = _select_threshold_for_target_frequency(
            calibration_samples=threshold_samples,
            probabilities=threshold_probabilities,
            thresholds=threshold_grid,
            target_trades_per_day=target_trades_per_day,
        )
        selected_threshold_source = "target_frequency_rank"
    elif adaptive_threshold:
        selected_threshold_row = _select_threshold_from_calibration(
            calibration_samples=threshold_samples,
            probabilities=threshold_probabilities,
            thresholds=threshold_grid,
            min_trades=min_calibration_trades,
            minimum_threshold=adaptive_minimum_threshold,
        )
    selected_threshold = (
        selected_threshold_row.threshold if selected_threshold_row else config.model.minimum_probability
    )
    if adaptive_threshold and selected_threshold_row and selected_threshold_source == "config":
        selected_threshold_source = "calibration"
    target_frequency_event_ids = (
        _select_target_frequency_event_ids(
            test_samples=test_samples,
            probabilities=probabilities,
            predicted_returns=test_predicted_returns,
            predicted_downsides=test_predicted_downsides,
            target_trades_per_day=target_trades_per_day,
            selected_threshold=selected_threshold,
            config=config,
            allow_negative_ev=allow_negative_ev_target_frequency,
            selection_score_mode=selection_score_mode,
            candidate_type_thresholds=type_thresholds if candidate_type_thresholds else None,
            empirical_payoff_ev=empirical_payoff_ev,
            payoff_estimates=payoff_estimates,
            require_calibrated_selection=require_calibrated_selection,
            min_signal_spacing_hours=min_signal_spacing_hours,
            max_signals_per_group_per_day=max_signals_per_group_per_day,
            max_signals_per_timestamp=max_signals_per_timestamp,
        )
        if target_trades_per_day and target_trades_per_day > 0
        else None
    )

    predictions: list[FoldPrediction] = []
    traded_labels: list[int] = []
    traded_returns: list[float] = []
    net_pnl = 0.0
    for sample, probability, predicted_return, predicted_downside in zip(
        test_samples,
        probabilities,
        test_predicted_returns,
        test_predicted_downsides,
        strict=True,
    ):
        probability = float(probability)
        predicted_return = float(predicted_return)
        predicted_downside = float(predicted_downside)
        expected_value = _expected_value(
            probability=probability,
            sample=sample,
            empirical_payoff_ev=empirical_payoff_ev,
            payoff_estimates=payoff_estimates,
        )
        selection_score = _selection_score(
            probability=probability,
            expected_value=expected_value,
            predicted_return=predicted_return,
            predicted_downside=predicted_downside,
            mode=selection_score_mode,
        )
        if target_frequency_event_ids is not None:
            should_trade = sample.event_id in target_frequency_event_ids
            reason = "target_frequency_rank" if should_trade else "target_frequency_rank_not_selected"
        elif candidate_type_thresholds and _threshold_lookup(type_thresholds, sample):
            type_threshold = _threshold_lookup(type_thresholds, sample)
            assert type_threshold is not None
            if type_threshold.get("abstain"):
                should_trade = False
                reason = "candidate_type_negative_calibration_utility"
            else:
                threshold = selected_threshold
                if type_threshold.get("threshold") is not None:
                    threshold = max(selected_threshold, float(type_threshold["threshold"]))
                probability_ok = probability >= threshold
                ev_ok = _passes_selection_and_ev_gate(
                    selection_score_mode=selection_score_mode,
                    selection_score=selection_score,
                    expected_value=expected_value,
                    minimum_expected_value=config.model.minimum_expected_value,
                    type_threshold=type_threshold,
                )
                should_trade = probability_ok and ev_ok
                if should_trade:
                    reason = "candidate_type_calibration"
                elif not probability_ok:
                    reason = "probability_below_candidate_type_threshold"
                else:
                    reason = "expected_value_below_threshold"
        elif adaptive_threshold and selected_threshold_row:
            probability_ok = probability >= selected_threshold
            ev_ok = _passes_selection_and_ev_gate(
                selection_score_mode=selection_score_mode,
                selection_score=selection_score,
                expected_value=expected_value,
                minimum_expected_value=config.model.minimum_expected_value,
            )
            should_trade = probability_ok and ev_ok
            if should_trade:
                reason = "approved"
            elif not probability_ok:
                reason = "probability_below_threshold"
            else:
                reason = "expected_value_below_threshold"
        else:
            utility_ok = _passes_selection_and_ev_gate(
                selection_score_mode=selection_score_mode,
                selection_score=selection_score,
                expected_value=expected_value,
                minimum_expected_value=config.model.minimum_expected_value,
            )
            should_trade = (
                probability >= config.model.minimum_probability
                and utility_ok
            )
            if should_trade:
                reason = "approved"
            elif probability < config.model.minimum_probability:
                reason = "probability_below_threshold"
            else:
                reason = "expected_value_below_threshold"
        pnl = sample.notional * sample.net_return if should_trade else 0.0
        if should_trade:
            traded_labels.append(sample.label)
            traded_returns.append(sample.net_return)
            net_pnl += pnl
        predictions.append(
            FoldPrediction(
                fold_id=fold_id,
                event_id=sample.event_id,
                timestamp_utc=sample.timestamp_utc.isoformat(),
                candidate_type=sample.candidate_type,
                label=sample.label,
                probability=probability,
                expected_value=expected_value,
                should_trade=should_trade,
                decision_reason=reason,
                net_return=sample.net_return,
                pnl=pnl,
                side=sample.side,
                predicted_return=predicted_return,
                predicted_downside=predicted_downside,
                selection_score=selection_score,
                setup_family=_sample_group(sample),
                market_regime=_sample_regime(sample),
            )
        )

    probability_min, probability_median, probability_max = _probability_summary(probabilities)
    threshold_sweep = _threshold_sweep(
        test_samples=test_samples,
        probabilities=probabilities,
        thresholds=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    )
    log_loss_value: float | None
    if len(set(y_test.tolist())) >= 2:
        log_loss_value = float(log_loss(y_test, probabilities, labels=[0, 1]))
    else:
        log_loss_value = None
    fold_report = FoldReport(
        fold_id=fold_id,
        train_samples=len(train_samples),
        calibration_samples=len(calibration_samples),
        test_samples=len(test_samples),
        fitted_models=[prediction_set.name for prediction_set in prediction_sets],
        skipped_models=skipped_models,
        brier_score=float(brier_score_loss(y_test, probabilities)),
        log_loss=log_loss_value,
        candidate_trades=len(test_samples),
        traded_signals=len(traded_labels),
        trade_hit_rate=float(sum(traded_labels) / len(traded_labels)) if traded_labels else 0.0,
        net_pnl=net_pnl,
        average_trade_return=float(sum(traded_returns) / len(traded_returns)) if traded_returns else 0.0,
        probability_min=probability_min,
        probability_median=probability_median,
        probability_max=probability_max,
        selected_threshold=selected_threshold,
        selected_threshold_source=selected_threshold_source,
        threshold_sweep=threshold_sweep,
        selected_model_params={
            prediction_set.name: prediction_set.selected_params for prediction_set in prediction_sets
        },
        model_diagnostics=model_diagnostics,
        reliability_buckets=_reliability_buckets(y_test, probabilities),
        candidate_type_thresholds=type_thresholds,
        candidate_type_calibration=_candidate_type_calibration(test_samples, probabilities),
        empirical_payoff=payoff_estimates,
    )
    fold_report.model_diagnostics["return_regression"] = {
        "specialist_models_enabled": 1.0 if specialist_models else 0.0,
        "threshold_prediction_median": float(np.median(threshold_predicted_returns))
        if len(threshold_predicted_returns)
        else 0.0,
        "test_prediction_median": float(np.median(test_predicted_returns))
        if len(test_predicted_returns)
        else 0.0,
    }
    for name, values in specialist_diagnostics.items():
        fold_report.model_diagnostics[f"specialist_{name}"] = values
    return fold_report, predictions, encoder


def default_fold_sizes(sample_count: int) -> tuple[int, int, int]:
    if sample_count < 60:
        raise ValueError("at least 60 samples are required for walk-forward ML evaluation")
    train_size = max(120, int(sample_count * 0.35)) if sample_count >= 300 else max(30, int(sample_count * 0.45))
    calibration_size = max(40, int(sample_count * 0.10)) if sample_count >= 300 else max(10, int(sample_count * 0.15))
    test_size = max(40, int(sample_count * 0.10)) if sample_count >= 300 else max(10, int(sample_count * 0.15))
    while train_size + calibration_size + test_size > sample_count:
        train_size = max(30, train_size - 5)
        if train_size + calibration_size + test_size <= sample_count:
            break
        calibration_size = max(10, calibration_size - 1)
        test_size = max(10, test_size - 1)
    return train_size, calibration_size, test_size


def run_meta_label_walk_forward(
    samples: list[MetaLabelSample],
    *,
    config: AppConfig,
    model_names: list[str] | None = None,
    train_size: int | None = None,
    calibration_size: int | None = None,
    test_size: int | None = None,
    embargo_hours: int | None = None,
    adaptive_threshold: bool = False,
    min_calibration_trades: int = 5,
    stacker_mode: str = "average",
    adaptive_minimum_threshold: float = 0.0,
    tune_hyperparameters: bool = False,
    hpo_profile: str = "standard",
    foundation_max_samples: int = 1024,
    data_coverage: dict[str, Any] | None = None,
    target_trades_per_day: float | None = None,
    allow_negative_ev_target_frequency: bool = False,
    candidate_type_thresholds: bool = False,
    empirical_payoff_ev: bool = False,
    selection_score_mode: str = "expected_value",
    specialist_models: bool = False,
    require_calibrated_selection: bool = False,
    min_signal_spacing_hours: float = 0.0,
    max_signals_per_group_per_day: int = 0,
    max_signals_per_timestamp: int = 0,
) -> MetaLabelWalkForwardReport:
    if not samples:
        raise ValueError("no samples supplied")
    ordered = sorted(samples, key=lambda sample: sample.timestamp_utc)
    if train_size is None or calibration_size is None or test_size is None:
        train_size, calibration_size, test_size = default_fold_sizes(len(ordered))
    names = model_names or [
        "logistic",
        "histgb",
        "randomforest",
        "extratrees",
        "lightgbm",
        "catboost",
        "xgboost",
        "tabicl",
        "tabpfn",
    ]
    folds = walk_forward_folds(
        ordered,
        train_size=train_size,
        calibration_size=calibration_size,
        test_size=test_size,
        embargo=timedelta(hours=embargo_hours or config.labels.max_holding_hours),
    )
    if not folds:
        raise ValueError("fold sizes produce no walk-forward folds")

    reports: list[FoldReport] = []
    predictions: list[FoldPrediction] = []
    feature_names: list[str] = []
    for fold_id, fold in enumerate(folds):
        train_samples = [ordered[idx] for idx in fold.train_indices]
        calibration_samples = [ordered[idx] for idx in fold.calibration_indices]
        test_samples = [ordered[idx] for idx in fold.test_indices]
        fold_report, fold_predictions, encoder = _score_fold(
            fold_id=fold_id,
            train_samples=train_samples,
            calibration_samples=calibration_samples,
            test_samples=test_samples,
            model_names=names,
            calibration_method=config.model.calibration_method,
            config=config,
            adaptive_threshold=adaptive_threshold,
            min_calibration_trades=min_calibration_trades,
            stacker_mode=stacker_mode,
            adaptive_minimum_threshold=adaptive_minimum_threshold,
            tune_hyperparameters=tune_hyperparameters,
            hpo_profile=hpo_profile,
            foundation_max_samples=foundation_max_samples,
            target_trades_per_day=target_trades_per_day,
            allow_negative_ev_target_frequency=allow_negative_ev_target_frequency,
            candidate_type_thresholds=candidate_type_thresholds,
            empirical_payoff_ev=empirical_payoff_ev,
            selection_score_mode=selection_score_mode,
            specialist_models=specialist_models,
            require_calibrated_selection=require_calibrated_selection,
            min_signal_spacing_hours=min_signal_spacing_hours,
            max_signals_per_group_per_day=max_signals_per_group_per_day,
            max_signals_per_timestamp=max_signals_per_timestamp,
        )
        reports.append(fold_report)
        predictions.extend(fold_predictions)
        if not feature_names:
            feature_names = encoder.feature_names

    return MetaLabelWalkForwardReport(
        samples=len(ordered),
        folds=reports,
        predictions=predictions,
        feature_names=feature_names,
        requested_models=names,
        calibration_method=config.model.calibration_method,
        stacker_mode=stacker_mode,
        data_coverage=data_coverage or {},
    )


def write_meta_label_report(path: Path, report: MetaLabelWalkForwardReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "samples": report.samples,
            "folds": len(report.folds),
            "traded_signals": report.traded_signals,
            "net_pnl": report.net_pnl,
            "requested_models": report.requested_models,
            "calibration_method": report.calibration_method,
            "stacker_mode": report.stacker_mode,
            "data_coverage": report.data_coverage,
            "feature_count": len(report.feature_names),
        },
        "folds": [asdict(fold) for fold in report.folds],
        "predictions": [asdict(prediction) for prediction in report.predictions],
        "feature_names": report.feature_names,
        "side_summary": report_side_summary(report),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def report_candidate_type_summary(report: MetaLabelWalkForwardReport) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for prediction in report.predictions:
        bucket = buckets.setdefault(
            prediction.candidate_type,
            {
                "predictions": 0.0,
                "labels": 0.0,
                "traded_signals": 0.0,
                "traded_labels": 0.0,
                "net_pnl": 0.0,
                "probability_sum": 0.0,
            },
        )
        bucket["predictions"] += 1
        bucket["labels"] += prediction.label
        bucket["probability_sum"] += prediction.probability
        if prediction.should_trade:
            bucket["traded_signals"] += 1
            bucket["traded_labels"] += prediction.label
            bucket["net_pnl"] += prediction.pnl
    for bucket in buckets.values():
        predictions = bucket["predictions"]
        traded = bucket["traded_signals"]
        bucket["base_rate"] = bucket["labels"] / predictions if predictions else 0.0
        bucket["average_probability"] = bucket["probability_sum"] / predictions if predictions else 0.0
        bucket["trade_hit_rate"] = bucket["traded_labels"] / traded if traded else 0.0
    return buckets


def report_side_summary(report: MetaLabelWalkForwardReport) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    for prediction in report.predictions:
        side = prediction.side or "UNKNOWN"
        bucket = buckets.setdefault(
            side,
            {
                "predictions": 0.0,
                "labels": 0.0,
                "traded_signals": 0.0,
                "traded_labels": 0.0,
                "net_pnl": 0.0,
                "probability_sum": 0.0,
            },
        )
        bucket["predictions"] += 1
        bucket["labels"] += prediction.label
        bucket["probability_sum"] += prediction.probability
        if prediction.should_trade:
            bucket["traded_signals"] += 1
            bucket["traded_labels"] += prediction.label
            bucket["net_pnl"] += prediction.pnl
    for bucket in buckets.values():
        predictions = bucket["predictions"]
        traded = bucket["traded_signals"]
        bucket["base_rate"] = bucket["labels"] / predictions if predictions else 0.0
        bucket["average_probability"] = bucket["probability_sum"] / predictions if predictions else 0.0
        bucket["trade_hit_rate"] = bucket["traded_labels"] / traded if traded else 0.0
    return buckets
