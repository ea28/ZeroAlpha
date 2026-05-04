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

from zeroalpha.config import AppConfig
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.ensemble import (
    FeatureEncoder,
    MetaLabelWalkForwardReport,
    ProbabilityCalibrator,
    _base_model_diagnostics,
    _base_validation_scores,
    _classification_sample_weights,
    _fit_base_prediction_sets,
    _fit_stacker,
    _labels,
    _predict_probability,
    _stack_probabilities,
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


@dataclass(frozen=True, slots=True)
class ArtifactScore:
    probability: float
    selected_threshold: float
    should_trade: bool
    model_count: int
    feature_count: int


def _artifact_threshold(report: MetaLabelWalkForwardReport, config: AppConfig) -> float:
    thresholds = [
        float(fold.selected_threshold)
        for fold in report.folds
        if fold.selected_threshold is not None
    ]
    research_threshold = median(thresholds) if thresholds else float(config.model.minimum_probability)
    return max(research_threshold, float(config.model.minimum_probability))


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
    adaptive_selection_score_floor: bool,
    min_signal_spacing_hours: float,
    max_signals_per_group_per_day: int,
    max_signals_per_timestamp: int,
    respect_open_positions: bool,
    capacity_release_mode: str,
    optimize_metric: str,
) -> ProductionModelArtifact:
    ordered = sorted(samples, key=lambda sample: sample.timestamp_utc)
    if len(ordered) < 60:
        raise ValueError("at least 60 samples are required to save a production model artifact")
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
        adaptive_selection_score_floor=adaptive_selection_score_floor,
        min_signal_spacing_hours=min_signal_spacing_hours,
        max_signals_per_group_per_day=max_signals_per_group_per_day,
        max_signals_per_timestamp=max_signals_per_timestamp,
        respect_open_positions=respect_open_positions,
        capacity_release_mode=capacity_release_mode,
        optimize_metric=optimize_metric,
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
            "capacity_release_mode": capacity_release_mode,
            "optimize_metric": optimize_metric,
        },
        validation_summary={
            "walk_forward_samples": report.samples,
            "walk_forward_folds": len(report.folds),
            "walk_forward_traded_signals": report.traded_signals,
            "walk_forward_net_pnl": report.net_pnl,
            "selected_threshold": _artifact_threshold(report, config),
            "model_diagnostics": diagnostics,
        },
        skipped_models=skipped,
    )
    return artifact


def score_production_artifact(
    artifact: ProductionModelArtifact,
    features: dict[str, float | str],
) -> ArtifactScore:
    import numpy as np

    row = SimpleNamespace(features=features)
    x = artifact.encoder.transform([row])  # type: ignore[list-item]
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
    return ArtifactScore(
        probability=probability,
        selected_threshold=float(artifact.selected_threshold),
        should_trade=probability >= artifact.selected_threshold,
        model_count=len(artifact.base_models),
        feature_count=len(artifact.feature_names),
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

    artifact = joblib.load(path)
    if not isinstance(artifact, ProductionModelArtifact):
        raise TypeError(f"{path} is not a ZeroAlpha production model artifact")
    return artifact


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
