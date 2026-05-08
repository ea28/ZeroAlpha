"""Production scoring artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Sequence
import hashlib
import json
import math

from zeroalpha.config import AppConfig
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.ensemble import (
    FeatureEncoder,
    MetaLabelWalkForwardReport,
    ProbabilityCalibrator,
    _base_model_diagnostics,
    _base_validation_scores,
    _classification_sample_weights,
    _downside_estimates_by_group,
    _economic_sample_weights,
    _expected_value,
    _fit_base_prediction_sets,
    _fit_stacker,
    _is_foundation_model,
    _labels,
    _predicted_downside,
    _passes_selection_and_ev_gate,
    _predict_probability,
    _return_regressor,
    _sample_setup_family,
    _sample_setup_family_values,
    _selection_score,
    _stack_probabilities,
    _threshold_lookup,
    _validation_weights,
)


@dataclass(slots=True)
class ProductionBaseModel:
    name: str
    estimator: Any
    calibrator: ProbabilityCalibrator
    selected_params: dict[str, Any] = field(default_factory=dict)
    validation_weight: float = 0.0


@dataclass(slots=True)
class ProductionModelArtifact:
    schema_version: str
    created_at_utc: str
    encoder: FeatureEncoder
    base_models: list[ProductionBaseModel]
    stacker_mode: str
    stacker: Any | None
    ensemble_calibrator: ProbabilityCalibrator
    validation_weights: list[float]
    selected_threshold: float
    feature_names: list[str]
    training_config: dict[str, Any]
    validation_summary: dict[str, Any]
    skipped_models: dict[str, str] = field(default_factory=dict)
    checksum: str = ""
    return_model: Any | None = None
    return_baseline: float = 0.0
    downside_estimates: dict[str, float] = field(default_factory=dict)
    candidate_type_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    payoff_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactScore:
    probability: float
    selected_threshold: float
    should_trade: bool
    model_count: int
    feature_count: int
    expected_value: float = 0.0
    selection_score: float = 0.0
    trade_score: float = 0.0
    decision_reason: str = ""
    missing_feature_count: int = 0
    missing_feature_fraction: float = 0.0


def _artifact_threshold(report: MetaLabelWalkForwardReport, config: AppConfig) -> float:
    thresholds = [
        float(fold.selected_threshold)
        for fold in report.folds
        if fold.selected_threshold is not None
    ]
    if thresholds:
        return median(thresholds)
    return float(config.model.minimum_probability)


def _artifact_selection_score_floor(
    report: MetaLabelWalkForwardReport,
    *,
    selection_score_floor: float | None,
    adaptive_selection_score_floor: bool,
) -> float | None:
    if selection_score_floor is not None:
        return float(selection_score_floor)
    if not adaptive_selection_score_floor:
        return None
    floors: list[float] = []
    for fold in report.folds:
        diagnostics = fold.model_diagnostics.get("target_frequency_selection", {})
        value = diagnostics.get("selection_score_floor")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            floors.append(float(value))
    return median(floors) if floors else None


def _artifact_selection_score_multiplier(
    report: MetaLabelWalkForwardReport,
    *,
    adaptive_selection_score_direction: bool,
) -> float:
    if not adaptive_selection_score_direction:
        return 1.0
    multipliers: list[float] = []
    for fold in report.folds:
        diagnostics = fold.model_diagnostics.get("target_frequency_selection", {})
        value = diagnostics.get("selection_score_multiplier")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            multipliers.append(float(value))
    if not multipliers:
        return 1.0
    positive = sum(1 for value in multipliers if value > 0)
    negative = sum(1 for value in multipliers if value < 0)
    return -1.0 if negative > positive else 1.0


def _median_numeric(values: list[Any], default: float = 0.0) -> float:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    return float(median(numeric)) if numeric else default


def _median_optional_numeric(values: list[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    return float(median(numeric)) if numeric else None


def _artifact_candidate_type_thresholds(
    report: MetaLabelWalkForwardReport,
    *,
    enabled: bool,
) -> dict[str, dict[str, Any]]:
    if not enabled:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fold in report.folds:
        for key, value in fold.candidate_type_thresholds.items():
            if isinstance(value, dict):
                grouped.setdefault(key, []).append(value)
    thresholds: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        abstain = any(bool(row.get("abstain")) for row in rows)
        threshold = _median_optional_numeric([row.get("threshold") for row in rows])
        thresholds[key] = {
            "threshold": threshold,
            "average_trade_return": _median_numeric(
                [row.get("average_trade_return") for row in rows],
                0.0,
            ),
            "trades": _median_numeric(
                [row.get("traded_signals", row.get("trades")) for row in rows],
                0.0,
            ),
            "abstain": abstain,
            "source": "artifact_fold_median",
        }
    return thresholds


def _artifact_payoff_estimates(
    report: MetaLabelWalkForwardReport,
    *,
    enabled: bool,
) -> dict[str, dict[str, Any]]:
    if not enabled:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fold in report.folds:
        for key, value in fold.empirical_payoff.items():
            if isinstance(value, dict):
                grouped.setdefault(key, []).append(value)
    estimates: dict[str, dict[str, Any]] = {}
    numeric_keys = (
        "count",
        "wins",
        "losses",
        "average_win",
        "average_loss",
        "average_label_one_return",
        "average_label_zero_return",
    )
    for key, rows in sorted(grouped.items()):
        estimate = {
            numeric_key: _median_numeric([row.get(numeric_key) for row in rows], 0.0)
            for numeric_key in numeric_keys
        }
        estimate["has_label_one_returns"] = any(bool(row.get("has_label_one_returns")) for row in rows)
        estimate["has_label_zero_returns"] = any(bool(row.get("has_label_zero_returns")) for row in rows)
        estimate["source"] = "calibration"
        estimates[key] = estimate
    return estimates


def fit_production_model_artifact(
    samples: Sequence[MetaLabelSample],
    *,
    config: AppConfig,
    report: MetaLabelWalkForwardReport,
    model_names: list[str],
    stacker_mode: str,
    tune_hyperparameters: bool,
    hpo_profile: str,
    hpo_trials: int,
    foundation_max_samples: int,
    target_trades_per_day: float | None,
    target_frequency_mode: str,
    allow_negative_ev_target_frequency: bool,
    selection_score_mode: str,
    selection_score_floor: float | None,
    selection_score_ceiling: float | None,
    adaptive_selection_score_floor: bool,
    adaptive_selection_score_direction: bool,
    min_signal_spacing_hours: float,
    max_signals_per_group_per_day: int,
    max_signals_per_timestamp: int,
    respect_open_positions: bool,
    capacity_release_mode: str,
    optimize_metric: str,
    candidate_type_thresholds: bool = False,
    empirical_payoff_ev: bool = False,
    specialist_models: bool = False,
    require_calibrated_selection: bool = False,
    selection_setup_families: tuple[str, ...] = (),
    selection_exclude_setup_families: tuple[str, ...] = (),
    feature_asof: str = "signal",
    external_feature_latency_seconds: float = 0.0,
    entry_order_model: str = "limit",
    sizing_mode: str = "fixed",
    sizing_score_field: str = "probability",
    sizing_score_direction: str = "high",
    sizing_base_notional: float = 0.0,
    sizing_mid_notional: float = 0.0,
    sizing_high_notional: float = 0.0,
    sizing_mid_score: float = 0.45,
    sizing_high_score: float = 0.90,
    sizing_max_spread_bps: float = 1.0,
    sizing_min_liquidity_score: float = 0.0,
    candidate_policy: dict[str, Any] | None = None,
    horizon_policy: dict[str, Any] | None = None,
    data_contract: dict[str, Any] | None = None,
    feature_contract: dict[str, Any] | None = None,
) -> ProductionModelArtifact:
    if entry_order_model not in {"limit", "market"}:
        raise ValueError("entry_order_model must be limit or market")
    if sizing_mode not in {"fixed", "confidence", "score_bucket", "liquidity_score_bucket"}:
        raise ValueError("sizing_mode must be fixed, confidence, score_bucket, or liquidity_score_bucket")
    if sizing_score_direction not in {"high", "low"}:
        raise ValueError("sizing_score_direction must be high or low")
    ordered = sorted(samples, key=lambda sample: sample.timestamp_utc)
    if len(ordered) < 60:
        raise ValueError("at least 60 samples are required to save a production model artifact")
    foundation_models = [name for name in model_names if _is_foundation_model(name)]
    if foundation_models:
        raise ValueError(
            "production artifacts cannot serialize foundation models; "
            f"remove {','.join(foundation_models)} from --models before saving"
        )
    if specialist_models:
        raise ValueError("production artifacts do not yet serialize specialist return regressors")
    train_count = max(30, int(len(ordered) * 0.70))
    train_count = min(train_count, len(ordered) - 30)
    train_samples = ordered[:train_count]
    calibration_samples = ordered[train_count:]
    base_calibration_samples, ensemble_samples, threshold_samples = _split_artifact_calibration(
        calibration_samples
    )
    encoder = FeatureEncoder.fit(train_samples)
    x_train = encoder.transform(train_samples)
    y_train = _labels(train_samples)
    x_base_calibration = encoder.transform(base_calibration_samples)
    y_base_calibration = _labels(base_calibration_samples)
    x_ensemble_calibration = encoder.transform(ensemble_samples)
    y_ensemble = _labels(ensemble_samples)
    x_threshold_selection = encoder.transform(threshold_samples)

    prediction_sets, skipped = _fit_base_prediction_sets(
        x_train=x_train,
        y_train=y_train,
        x_base_calibration=x_base_calibration,
        y_base_calibration=y_base_calibration,
        x_ensemble_calibration=x_ensemble_calibration,
        x_threshold_selection=x_threshold_selection,
        x_test=x_threshold_selection,
        train_samples=train_samples,
        config=config,
        model_names=model_names,
        calibration_method=config.model.calibration_method,
        tune_hyperparameters=tune_hyperparameters,
        hpo_profile=hpo_profile,
        hpo_trials=hpo_trials,
        foundation_max_samples=foundation_max_samples,
        target_trades_per_day=target_trades_per_day,
        target_frequency_mode=target_frequency_mode,
        allow_negative_ev_target_frequency=allow_negative_ev_target_frequency,
        selection_score_mode=selection_score_mode,
        selection_score_floor=selection_score_floor,
        selection_score_ceiling=selection_score_ceiling,
        adaptive_selection_score_floor=adaptive_selection_score_floor,
        min_signal_spacing_hours=min_signal_spacing_hours,
        max_signals_per_group_per_day=max_signals_per_group_per_day,
        max_signals_per_timestamp=max_signals_per_timestamp,
        respect_open_positions=respect_open_positions,
        capacity_release_mode=capacity_release_mode,
        optimize_metric=optimize_metric,
        empirical_payoff_ev=empirical_payoff_ev,
        specialist_models=False,
    )
    usable_sets = [
        prediction_set
        for prediction_set in prediction_sets
        if prediction_set.estimator is not None and prediction_set.calibrator is not None
    ]
    if not usable_sets:
        raise ValueError(f"no serializable base models were fitted; skipped={skipped}")

    ensemble_matrix = _column_stack([prediction_set.ensemble_probability for prediction_set in usable_sets])
    validation_scores = _base_validation_scores(
        calibration_samples=ensemble_samples,
        labels=y_ensemble,
        calibration_matrix=ensemble_matrix,
    )
    stacker = None
    if stacker_mode == "logistic":
        stacker = _fit_stacker(
            ensemble_matrix,
            y_ensemble,
            sample_weight=_classification_sample_weights(ensemble_samples),
        )
        ensemble_probabilities = _stack_probabilities(stacker, ensemble_matrix)
        weights = [1.0 / len(usable_sets)] * len(usable_sets)
    elif stacker_mode == "average":
        ensemble_probabilities = _stack_probabilities(None, ensemble_matrix)
        weights = [1.0 / len(usable_sets)] * len(usable_sets)
    elif stacker_mode == "best":
        best_idx = max(
            range(len(validation_scores)),
            key=lambda idx: (validation_scores[idx][0], -validation_scores[idx][1]),
        )
        ensemble_probabilities = ensemble_matrix[:, best_idx]
        weights = [0.0] * len(usable_sets)
        weights[best_idx] = 1.0
    elif stacker_mode == "weighted":
        np_weights = _validation_weights(validation_scores)
        ensemble_probabilities = ensemble_matrix @ np_weights
        weights = [float(value) for value in np_weights]
    else:
        raise ValueError("stacker_mode must be average, logistic, best, or weighted")

    ensemble_calibrator = ProbabilityCalibrator.fit(
        ensemble_probabilities,
        y_ensemble,
        method=config.model.calibration_method,
    )
    diagnostics = _base_model_diagnostics(
        prediction_sets=usable_sets,
        scores=validation_scores,
        weights=weights,
    )
    return_model, return_baseline = _fit_artifact_return_model(
        x_train=x_train,
        train_samples=train_samples,
    )
    downside_estimates = _downside_estimates_by_group(
        calibration_samples,
        min_samples=max(5, min(30, len(calibration_samples) // 4)),
    )
    artifact_type_thresholds = _artifact_candidate_type_thresholds(
        report,
        enabled=candidate_type_thresholds,
    )
    artifact_payoff_estimates = _artifact_payoff_estimates(
        report,
        enabled=empirical_payoff_ev,
    )
    base_models = [
        ProductionBaseModel(
            name=prediction_set.name,
            estimator=prediction_set.estimator,
            calibrator=prediction_set.calibrator,
            selected_params=prediction_set.selected_params,
            validation_weight=float(weights[idx]),
        )
        for idx, prediction_set in enumerate(usable_sets)
    ]
    selected_selection_score_floor = _artifact_selection_score_floor(
        report,
        selection_score_floor=selection_score_floor,
        adaptive_selection_score_floor=adaptive_selection_score_floor,
    )
    selected_selection_score_multiplier = _artifact_selection_score_multiplier(
        report,
        adaptive_selection_score_direction=adaptive_selection_score_direction,
    )
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        encoder=encoder,
        base_models=base_models,
        stacker_mode=stacker_mode,
        stacker=stacker,
        ensemble_calibrator=ensemble_calibrator,
        validation_weights=weights,
        selected_threshold=_artifact_threshold(report, config),
        feature_names=encoder.feature_names,
        training_config={
            "samples": len(ordered),
            "train_samples": len(train_samples),
            "calibration_samples": len(calibration_samples),
            "requested_models": model_names,
            "calibration_method": config.model.calibration_method,
            "stacker_mode": stacker_mode,
            "target_trades_per_day": target_trades_per_day or 0.0,
            "target_frequency_mode": target_frequency_mode,
            "allow_negative_ev_target_frequency": allow_negative_ev_target_frequency,
            "selection_score_mode": selection_score_mode,
            "selection_score_floor": selection_score_floor,
            "selection_score_ceiling": selection_score_ceiling,
            "selected_selection_score_floor": selected_selection_score_floor,
            "adaptive_selection_score_floor": adaptive_selection_score_floor,
            "adaptive_selection_score_direction": adaptive_selection_score_direction,
            "selected_selection_score_multiplier": selected_selection_score_multiplier,
            "minimum_probability": config.model.minimum_probability,
            "minimum_expected_value": config.model.minimum_expected_value,
            "label_net_profit_target": config.labels.net_profit_target,
            "label_net_stop_loss": config.labels.net_stop_loss,
            "capacity_release_mode": capacity_release_mode,
            "optimize_metric": optimize_metric,
            "candidate_type_thresholds": candidate_type_thresholds,
            "empirical_payoff_ev": empirical_payoff_ev,
            "require_calibrated_selection": require_calibrated_selection,
            "selection_setup_families": list(selection_setup_families),
            "selection_exclude_setup_families": list(selection_exclude_setup_families),
            "setup_family_policy": {
                "setup_family_fields": [
                    "event_specialist_setup_family",
                    "event_setup_family",
                    "event_dense_setup_family",
                    "specialist_setup_family",
                    "setup_family",
                    "dense_setup_family",
                ],
                "score_direction_field": "event_setup_score_direction",
                "allowed_score_directions": ["high", "low", "none"],
                "selection_setup_families": list(selection_setup_families),
                "selection_exclude_setup_families": list(selection_exclude_setup_families),
            },
            "feature_asof": feature_asof,
            "external_feature_latency_seconds": external_feature_latency_seconds,
            "candidate_policy": dict(candidate_policy or {}),
            "horizon_policy": dict(horizon_policy or {}),
            "data_contract": dict(data_contract or {}),
            "feature_contract": {
                "feature_count": len(encoder.feature_names),
                **dict(feature_contract or {}),
            },
            "entry_order_model": entry_order_model,
            "execution_policy": {
                "entry_order_model": entry_order_model,
            },
            "sizing_mode": sizing_mode,
            "sizing_score_field": sizing_score_field,
            "sizing_score_direction": sizing_score_direction,
            "sizing_base_notional": sizing_base_notional,
            "sizing_mid_notional": sizing_mid_notional,
            "sizing_high_notional": sizing_high_notional,
            "sizing_mid_score": sizing_mid_score,
            "sizing_high_score": sizing_high_score,
            "sizing_max_spread_bps": sizing_max_spread_bps,
            "sizing_min_liquidity_score": sizing_min_liquidity_score,
            "sizing_policy": {
                "mode": sizing_mode,
                "score_field": sizing_score_field,
                "score_direction": sizing_score_direction,
                "base_notional": sizing_base_notional,
                "mid_notional": sizing_mid_notional,
                "high_notional": sizing_high_notional,
                "mid_score": sizing_mid_score,
                "high_score": sizing_high_score,
                "max_spread_bps": sizing_max_spread_bps,
                "min_liquidity_score": sizing_min_liquidity_score,
            },
        },
        validation_summary={
            "walk_forward_samples": report.samples,
            "walk_forward_folds": len(report.folds),
            "walk_forward_traded_signals": report.traded_signals,
            "walk_forward_net_pnl": report.net_pnl,
            "selected_threshold": _artifact_threshold(report, config),
            "selected_selection_score_floor": selected_selection_score_floor,
            "selected_selection_score_multiplier": selected_selection_score_multiplier,
            "model_diagnostics": diagnostics,
            "return_model": {
                "enabled": return_model is not None,
                "baseline": return_baseline,
                "downside_estimates": len(downside_estimates),
            },
            "candidate_type_thresholds": len(artifact_type_thresholds),
            "payoff_estimates": len(artifact_payoff_estimates),
        },
        skipped_models=skipped,
        return_model=return_model,
        return_baseline=return_baseline,
        downside_estimates=downside_estimates,
        candidate_type_thresholds=artifact_type_thresholds,
        payoff_estimates=artifact_payoff_estimates,
    )
    return artifact


def score_production_artifact(
    artifact: ProductionModelArtifact,
    features: dict[str, float | str],
    *,
    threshold_override: float | None = None,
) -> ArtifactScore:
    import numpy as np

    candidate_type = str(features.get("candidate_type", ""))
    side = str(features.get("side", "BUY"))
    row = SimpleNamespace(features=features, candidate_type=candidate_type, side=side)
    x = artifact.encoder.transform([row])  # type: ignore[list-item]
    expected_feature_keys = [
        *artifact.encoder.numeric_features,
        *artifact.encoder.categorical_values.keys(),
    ]
    missing_feature_count = sum(1 for key in expected_feature_keys if key not in features)
    missing_feature_fraction = (
        missing_feature_count / len(expected_feature_keys)
        if expected_feature_keys
        else 0.0
    )
    base_probabilities = []
    for model in artifact.base_models:
        raw = _predict_probability(model.estimator, x)
        base_probabilities.append(model.calibrator.predict(raw))
    base_matrix = np.column_stack(base_probabilities)
    if artifact.stacker_mode == "logistic":
        stacked = _stack_probabilities(artifact.stacker, base_matrix)
    elif artifact.stacker_mode == "average":
        stacked = _stack_probabilities(None, base_matrix)
    elif artifact.stacker_mode in {"best", "weighted"}:
        stacked = base_matrix @ np.asarray(artifact.validation_weights, dtype=float)
    else:
        stacked = _stack_probabilities(None, base_matrix)
    probability = float(artifact.ensemble_calibrator.predict(stacked)[0])
    selected_threshold = (
        float(artifact.selected_threshold)
        if threshold_override is None
        else float(threshold_override)
    )
    if not 0.0 <= selected_threshold <= 1.0:
        raise ValueError("selected threshold must be between 0 and 1")
    training_config = getattr(artifact, "training_config", {}) or {}
    net_profit_target = _artifact_feature_float(
        features,
        "net_profit_target",
        float(training_config.get("label_net_profit_target", 0.0) or 0.0),
    )
    net_stop_loss = _artifact_feature_float(
        features,
        "net_stop_loss",
        float(training_config.get("label_net_stop_loss", 0.0) or 0.0),
    )
    payoff_estimates = getattr(artifact, "payoff_estimates", {}) or {}
    expected_value = _expected_value(
        probability=probability,
        sample=SimpleNamespace(
            candidate_type=candidate_type,
            side=side,
            features=features,
            net_profit_target=net_profit_target,
            net_stop_loss=net_stop_loss,
        ),
        empirical_payoff_ev=bool(training_config.get("empirical_payoff_ev", False)),
        payoff_estimates=payoff_estimates,
    )
    selection_score_mode = str(training_config.get("selection_score_mode", "probability"))
    predicted_return = _artifact_predicted_return(
        artifact,
        x,
        features,
        default=expected_value,
    )
    predicted_downside = _artifact_predicted_downside(
        artifact,
        features,
        default=0.0,
    )
    predicted_time_to_exit_seconds = _artifact_feature_float(
        features,
        "max_holding_seconds",
        0.0,
    )
    try:
        selection_score = _selection_score(
            probability=probability,
            expected_value=expected_value,
            predicted_return=predicted_return,
            predicted_downside=predicted_downside,
            predicted_mae=predicted_downside,
            predicted_mfe=max(predicted_return, 0.0),
            predicted_time_to_exit_seconds=predicted_time_to_exit_seconds,
            predicted_early_adverse_probability=0.0,
            mode=selection_score_mode,
        )
    except ValueError:
        selection_score_mode = "probability"
        selection_score = probability
    selection_score *= float(training_config.get("selected_selection_score_multiplier", 1.0) or 1.0)
    trade_score = _selection_score(
        probability=probability,
        expected_value=expected_value,
        predicted_return=predicted_return,
        predicted_downside=predicted_downside,
        predicted_mae=predicted_downside,
        predicted_mfe=max(predicted_return, 0.0),
        predicted_time_to_exit_seconds=predicted_time_to_exit_seconds,
        predicted_early_adverse_probability=0.0,
        mode="capital_efficiency",
    )
    selection_score_floor = training_config.get("selected_selection_score_floor")
    if selection_score_floor is None:
        selection_score_floor = training_config.get("selection_score_floor")
    score_floor_ok = (
        selection_score_floor is None
        or selection_score >= float(selection_score_floor)
    )
    selection_score_ceiling = training_config.get("selection_score_ceiling")
    score_ceiling_ok = (
        selection_score_ceiling is None
        or selection_score <= float(selection_score_ceiling)
    )
    sample_for_gates = SimpleNamespace(
        candidate_type=candidate_type,
        side=side,
        features=features,
    )
    setup_family = _sample_setup_family(sample_for_gates)  # type: ignore[arg-type]
    setup_family_values = _sample_setup_family_values(sample_for_gates)  # type: ignore[arg-type]
    if not setup_family_values:
        setup_family_values = {setup_family}
    setup_policy = training_config.get("setup_family_policy")
    if not isinstance(setup_policy, dict):
        setup_policy = {}
    setup_allow = tuple(
        training_config.get("selection_setup_families")
        or setup_policy.get("selection_setup_families")
        or ()
    )
    setup_exclude = tuple(
        training_config.get("selection_exclude_setup_families")
        or setup_policy.get("selection_exclude_setup_families")
        or ()
    )
    setup_family_ok = (
        (not setup_allow or bool(setup_family_values & set(setup_allow)))
        and not bool(setup_family_values & set(setup_exclude))
    )
    type_thresholds = getattr(artifact, "candidate_type_thresholds", {}) or {}
    type_threshold = _threshold_lookup(type_thresholds, sample_for_gates) if type_thresholds else None  # type: ignore[arg-type]
    type_threshold_ok = True
    candidate_type_probability_floor = None
    if bool(training_config.get("require_calibrated_selection", False)) and type_threshold is None:
        type_threshold_ok = False
    if type_threshold is not None:
        if type_threshold.get("abstain"):
            type_threshold_ok = False
        elif type_threshold.get("threshold") is not None:
            candidate_type_probability_floor = float(type_threshold["threshold"])
            selected_threshold = max(selected_threshold, candidate_type_probability_floor)
    minimum_expected_value = float(training_config.get("minimum_expected_value", 0.0) or 0.0)
    utility_ok = _passes_selection_and_ev_gate(
        selection_score_mode=selection_score_mode,
        selection_score=selection_score,
        expected_value=expected_value,
        minimum_expected_value=minimum_expected_value,
        type_threshold=type_threshold,
    )
    expected_feature_asof = str(training_config.get("feature_asof", "") or "")
    asof_contract_ok = True
    if expected_feature_asof in {"signal", "entry"}:
        observed_entry = float(features.get("feature_asof_is_entry", 0.0) or 0.0)
        expected_entry = 1.0 if expected_feature_asof == "entry" else 0.0
        asof_contract_ok = abs(observed_entry - expected_entry) < 0.5
    probability_ok = (
        probability >= selected_threshold
        if threshold_override is not None
        else (
            probability >= candidate_type_probability_floor
            if selection_score_mode == "return_first"
            and candidate_type_probability_floor is not None
            else selection_score_mode == "return_first" or probability >= selected_threshold
        )
    )
    should_trade = (
        probability_ok
        and score_floor_ok
        and score_ceiling_ok
        and utility_ok
        and setup_family_ok
        and type_threshold_ok
        and asof_contract_ok
    )
    if should_trade:
        decision_reason = "approved"
    elif not asof_contract_ok:
        decision_reason = "feature_asof_contract_mismatch"
    elif not setup_family_ok:
        decision_reason = "setup_family_filtered"
    elif not type_threshold_ok:
        decision_reason = "candidate_type_threshold_filtered"
    elif not probability_ok:
        decision_reason = "probability_below_threshold"
    elif not score_floor_ok:
        decision_reason = "selection_score_below_floor"
    elif not score_ceiling_ok:
        decision_reason = "selection_score_above_ceiling"
    else:
        decision_reason = "expected_value_below_threshold"
    return ArtifactScore(
        probability=probability,
        selected_threshold=selected_threshold,
        should_trade=should_trade,
        model_count=len(artifact.base_models),
        feature_count=len(artifact.feature_names),
        expected_value=expected_value,
        selection_score=selection_score,
        trade_score=trade_score,
        decision_reason=decision_reason,
        missing_feature_count=missing_feature_count,
        missing_feature_fraction=missing_feature_fraction,
    )


def save_production_model_artifact(path: Path, artifact: ProductionModelArtifact) -> str:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = {
        "path": str(path),
        "sha256": checksum,
        "schema_version": artifact.schema_version,
        "created_at_utc": artifact.created_at_utc,
        "feature_count": len(artifact.feature_names),
        "base_models": [model.name for model in artifact.base_models],
        "selected_threshold": artifact.selected_threshold,
        "training_config": artifact.training_config,
        "validation_summary": artifact.validation_summary,
        "skipped_models": artifact.skipped_models,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return checksum


def load_production_model_artifact(path: Path) -> ProductionModelArtifact:
    import joblib

    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("sha256", ""))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"{path} checksum mismatch against {manifest_path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, ProductionModelArtifact):
        raise TypeError(f"{path} is not a ZeroAlpha production model artifact")
    return artifact


def _artifact_feature_float(
    features: dict[str, float | str],
    key: str,
    default: float,
) -> float:
    value = features.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return float(default)


def _fit_artifact_return_model(
    *,
    x_train: Any,
    train_samples: list[MetaLabelSample],
) -> tuple[Any | None, float]:
    if not train_samples:
        return None, 0.0
    import numpy as np

    y_train = np.asarray([sample.net_return for sample in train_samples], dtype=float)
    baseline = float(np.mean(y_train)) if len(y_train) else 0.0
    if len(train_samples) < 20:
        return None, baseline
    model = _return_regressor()
    model.fit(x_train, y_train, sample_weight=_economic_sample_weights(train_samples))
    return model, baseline


def _artifact_predicted_return(
    artifact: ProductionModelArtifact,
    x: Any,
    features: dict[str, float | str],
    *,
    default: float,
) -> float:
    model = getattr(artifact, "return_model", None)
    if model is not None:
        try:
            value = float(model.predict(x)[0])
            if math.isfinite(value):
                return value
        except Exception:
            pass
    baseline = getattr(artifact, "return_baseline", None)
    if isinstance(baseline, int | float) and math.isfinite(float(baseline)):
        return float(baseline)
    return _artifact_feature_float(features, "predicted_return", default)


def _artifact_predicted_downside(
    artifact: ProductionModelArtifact,
    features: dict[str, float | str],
    *,
    default: float,
) -> float:
    estimates = getattr(artifact, "downside_estimates", {}) or {}
    if estimates:
        candidate_type = features.get("candidate_type")
        side = features.get("side")
        sample = SimpleNamespace(
            features=features,
            candidate_type=candidate_type if isinstance(candidate_type, str) else "",
            side=side if isinstance(side, str) else "BUY",
        )
        value = _predicted_downside(sample, estimates)
        if math.isfinite(float(value)):
            return float(value)
    return _artifact_feature_float(features, "predicted_downside", default)


def production_model_manifest(artifact: ProductionModelArtifact) -> dict[str, Any]:
    return {
        "schema_version": artifact.schema_version,
        "created_at_utc": artifact.created_at_utc,
        "selected_threshold": artifact.selected_threshold,
        "feature_count": len(artifact.feature_names),
        "base_models": [
            {
                "name": model.name,
                "estimator": type(model.estimator).__name__,
                "calibrator_method": model.calibrator.method,
                "selected_params": model.selected_params,
                "validation_weight": model.validation_weight,
            }
            for model in artifact.base_models
        ],
        "training_config": artifact.training_config,
        "validation_summary": artifact.validation_summary,
        "skipped_models": artifact.skipped_models,
    }


def _split_artifact_calibration(
    samples: list[MetaLabelSample],
) -> tuple[list[MetaLabelSample], list[MetaLabelSample], list[MetaLabelSample]]:
    if len(samples) < 30:
        raise ValueError("at least 30 calibration samples are required")
    first = max(10, len(samples) // 3)
    second = max(first + 10, (2 * len(samples)) // 3)
    second = min(second, len(samples) - 10)
    return samples[:first], samples[first:second], samples[second:]


def _column_stack(columns: Sequence[Any]) -> Any:
    import numpy as np

    return np.column_stack(columns)
