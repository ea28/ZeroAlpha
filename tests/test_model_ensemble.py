from datetime import UTC, datetime, timedelta
from dataclasses import replace
from types import SimpleNamespace

import pytest

import zeroalpha.models.ensemble as ensemble_module
from zeroalpha.config import AppConfig, LabelConfig, ModelConfig, RiskConfig
from zeroalpha.domain import TripleBarrierLabel
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.artifact import (
    ProductionBaseModel,
    ProductionModelArtifact,
    _artifact_candidate_type_thresholds,
    _artifact_selection_score_multiplier,
    fit_production_model_artifact,
    load_production_model_artifact,
    save_production_model_artifact,
    score_production_artifact,
)
from zeroalpha.models.ensemble import (
    FeatureEncoder,
    ProbabilityCalibrator,
    SelectionExecutionPolicy,
    _economic_sample_weights,
    _expected_value,
    _hpo_grid,
    _limit_hpo_grid,
    _online_threshold_frequency_returns,
    _payoff_estimate,
    _purged_hpo_inner_training_indices,
    _quota_frequency_returns,
    _select_candidate_type_thresholds,
    _select_selection_score_floor_for_target_frequency,
    _select_target_frequency_event_ids,
    _select_threshold_for_target_frequency,
    _split_calibration_samples,
    default_fold_sizes,
    report_feature_importance_summary,
    report_native_importance_summary,
    report_shap_importance_summary,
    run_meta_label_walk_forward,
)


def _sample(i: int) -> MetaLabelSample:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i)
    label = 1 if i % 4 in (0, 1, 2) else 0
    detail = TripleBarrierLabel(
        event_id=f"e{i}",
        entry_timestamp_utc=ts + timedelta(hours=1),
        entry_price=100,
        upper_barrier_price=103,
        lower_barrier_price=99,
        vertical_barrier_timestamp_utc=ts + timedelta(hours=24),
        exit_timestamp_utc=ts + timedelta(hours=12),
        exit_price=103 if label else 99,
        outcome_type="upper" if label else "lower",
        gross_return=0.03 if label else -0.01,
        net_return=0.02 if label else -0.02,
        label=label,
        t1=ts + timedelta(hours=12),
    )
    return MetaLabelSample(
        event_id=f"e{i}",
        timestamp_utc=ts,
        t1=detail.t1,
        candidate_type="trend_continuation" if i % 2 else "volatility_breakout",
        side="BUY",
        net_profit_target=0.02,
        net_stop_loss=0.02,
        features={
            "candidate_type": "trend_continuation" if i % 2 else "volatility_breakout",
            "signal_strength": float(label) + (i % 7) * 0.01,
            "return_24": 0.02 if label else -0.02,
            "realized_vol_24": 0.01 + (i % 5) * 0.001,
        },
        label=label,
        net_return=detail.net_return,
        notional=1_000,
        round_trip_cost_bps=86,
        outcome_type=detail.outcome_type,
        label_detail=detail,
    )


class _ConstantProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, x):
        import numpy as np

        positive = np.full(len(x), self.probability)
        return np.column_stack([1.0 - positive, positive])


class _ConstantReturnEstimator:
    def __init__(self, predicted_return: float) -> None:
        self.predicted_return = predicted_return

    def predict(self, x):
        import numpy as np

        return np.full(len(x), self.predicted_return)


def test_meta_label_walk_forward_trains_logistic_stack() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
    )
    assert report.samples == 120
    assert report.folds
    assert any("logistic" in fold.fitted_models for fold in report.folds)
    assert report.predictions


def test_meta_label_walk_forward_reports_fold_local_permutation_importance() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        permutation_importance=True,
        permutation_repeats=2,
        permutation_max_features=5,
        permutation_sample_limit=20,
        interpretability_top_n=3,
        importance_scoring=("brier", "log_loss", "net_pnl"),
        permutation_grouping="both",
    )

    rows = [row for fold in report.folds for row in fold.permutation_importance]

    assert rows
    assert {row.scope for row in rows} >= {"base_model", "final_ensemble"}
    assert all(row.sample_count <= 20 for row in rows)
    assert all(row.rank <= 3 for row in rows)
    assert any(row.feature.startswith("family:") for row in rows)
    assert any(row.metric == "threshold_only_net_pnl" for row in rows)
    assert any(
        "threshold_only" in warning
        for fold in report.folds
        for warning in fold.interpretability_warnings
    )
    assert report_feature_importance_summary(report)
    assert report_native_importance_summary(report)


def test_meta_label_walk_forward_reports_fold_local_shap_importance() -> None:
    pytest.importorskip("shap")
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["extratrees"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        shap_importance=True,
        shap_sample_limit=12,
        shap_background_limit=20,
        shap_top_n=4,
        shap_grouping="both",
    )

    rows = [row for fold in report.folds for row in fold.shap_importance]

    assert rows
    assert all(row.sample_count <= 12 for row in rows)
    assert all(row.background_count <= 20 for row in rows)
    assert any(row.feature.startswith("family:") for row in rows)
    assert report_shap_importance_summary(report)


def test_production_model_artifact_can_be_saved_loaded_and_scored(tmp_path) -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    samples = [_sample(i) for i in range(120)]
    report = run_meta_label_walk_forward(
        samples,
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
    )
    artifact = fit_production_model_artifact(
        samples,
        config=config,
        report=report,
        model_names=["logistic"],
        stacker_mode="average",
        tune_hyperparameters=False,
        hpo_profile="standard",
        hpo_trials=0,
        foundation_max_samples=128,
        target_trades_per_day=None,
        target_frequency_mode="online",
        allow_negative_ev_target_frequency=False,
        selection_score_mode="expected_value",
        selection_score_floor=None,
        selection_score_ceiling=None,
        adaptive_selection_score_floor=False,
        adaptive_selection_score_direction=False,
        min_signal_spacing_hours=0.0,
        max_signals_per_group_per_day=0,
        max_signals_per_timestamp=0,
        respect_open_positions=False,
        capacity_release_mode="planned",
        optimize_metric="sharpe",
        entry_order_model="market",
        sizing_mode="score_bucket",
        sizing_score_field="selection_score",
        sizing_score_direction="low",
        sizing_base_notional=1_250,
        sizing_mid_notional=2_500,
        sizing_high_notional=5_000,
        sizing_mid_score=0.30,
        sizing_high_score=0.75,
        candidate_policy={"mode": "dense", "side_mode": "long"},
        horizon_policy={"min_holding_seconds": 1.0, "max_holding_hours": 6.0},
        data_contract={"requires_one_second_execution": True, "execution_interval": "1s"},
        feature_contract={"max_missing_feature_fraction": 0.0},
    )
    path = tmp_path / "prod.joblib"

    checksum = save_production_model_artifact(path, artifact)
    loaded = load_production_model_artifact(path)
    score = score_production_artifact(loaded, samples[-1].features)

    assert checksum
    assert path.exists()
    assert path.with_suffix(".joblib.manifest.json").exists()
    assert loaded.base_models[0].name == "logistic"
    assert loaded.selected_threshold >= config.model.minimum_probability
    assert loaded.training_config["execution_policy"]["entry_order_model"] == "market"
    assert loaded.training_config["sizing_policy"]["mode"] == "score_bucket"
    assert loaded.training_config["sizing_policy"]["score_direction"] == "low"
    assert loaded.training_config["sizing_policy"]["high_notional"] == 5_000
    assert loaded.training_config["candidate_policy"]["side_mode"] == "long"
    assert loaded.training_config["setup_family_policy"]["score_direction_field"] == "event_setup_score_direction"
    assert loaded.training_config["horizon_policy"]["min_holding_seconds"] == 1.0
    assert loaded.training_config["data_contract"]["requires_one_second_execution"] is True
    assert loaded.training_config["feature_contract"]["max_missing_feature_fraction"] == 0.0
    assert 0.0 <= score.probability <= 1.0
    assert score.feature_count == len(loaded.feature_names)


def test_production_model_load_verifies_manifest_checksum(tmp_path) -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    samples = [_sample(i) for i in range(120)]
    report = run_meta_label_walk_forward(
        samples,
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
    )
    artifact = fit_production_model_artifact(
        samples,
        config=config,
        report=report,
        model_names=["logistic"],
        stacker_mode="average",
        tune_hyperparameters=False,
        hpo_profile="standard",
        hpo_trials=0,
        foundation_max_samples=128,
        target_trades_per_day=None,
        target_frequency_mode="online",
        allow_negative_ev_target_frequency=False,
        selection_score_mode="expected_value",
        selection_score_floor=None,
        selection_score_ceiling=None,
        adaptive_selection_score_floor=False,
        adaptive_selection_score_direction=False,
        min_signal_spacing_hours=0.0,
        max_signals_per_group_per_day=0,
        max_signals_per_timestamp=0,
        respect_open_positions=False,
        capacity_release_mode="planned",
        optimize_metric="sharpe",
    )
    path = tmp_path / "prod.joblib"
    save_production_model_artifact(path, artifact)
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_production_model_artifact(path)


def test_production_artifact_preserves_target_frequency_threshold() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    samples = [_sample(i) for i in range(120)]
    report = run_meta_label_walk_forward(
        samples,
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
    )
    target_threshold = 0.05
    report = replace(
        report,
        folds=[
            replace(
                fold,
                selected_threshold=target_threshold,
                selected_threshold_source="target_frequency_online",
            )
            for fold in report.folds
        ],
    )

    artifact = fit_production_model_artifact(
        samples,
        config=config,
        report=report,
        model_names=["logistic"],
        stacker_mode="average",
        tune_hyperparameters=False,
        hpo_profile="standard",
        hpo_trials=0,
        foundation_max_samples=128,
        target_trades_per_day=3.0,
        target_frequency_mode="online",
        allow_negative_ev_target_frequency=False,
        selection_score_mode="expected_value",
        selection_score_floor=None,
        selection_score_ceiling=None,
        adaptive_selection_score_floor=False,
        adaptive_selection_score_direction=False,
        min_signal_spacing_hours=0.0,
        max_signals_per_group_per_day=0,
        max_signals_per_timestamp=0,
        respect_open_positions=False,
        capacity_release_mode="planned",
        optimize_metric="sharpe",
    )
    score = score_production_artifact(
        artifact,
        samples[-1].features,
        threshold_override=0.01,
    )

    assert artifact.selected_threshold == target_threshold
    assert artifact.validation_summary["selected_threshold"] == target_threshold
    assert score.selected_threshold == 0.01
    assert artifact.training_config["selection_score_mode"] == "expected_value"


def test_production_artifact_applies_cost_aware_ev_gate() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "expected_value",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.0005, "net_stop_loss": 0.008},
    )

    assert score.probability >= score.selected_threshold
    assert score.expected_value < 0
    assert not score.should_trade
    assert score.decision_reason == "expected_value_below_threshold"


def test_production_artifact_applies_learned_selection_score_floor() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "expected_value",
            "selected_selection_score_floor": 0.019,
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.02, "net_stop_loss": 0.02},
    )

    assert 0 < score.selection_score < 0.019
    assert not score.should_trade
    assert score.decision_reason == "selection_score_below_floor"


def test_production_artifact_applies_selection_score_ceiling() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "expected_value",
            "selection_score_ceiling": 0.01,
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.02, "net_stop_loss": 0.02},
    )

    assert score.selection_score > 0.01
    assert not score.should_trade
    assert score.decision_reason == "selection_score_above_ceiling"


def test_production_artifact_reports_missing_feature_fraction() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss", "eth_available"),
            categorical_values={"candidate_type": ("dense_research_bar",)},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss", "eth_available"],
        training_config={
            "selection_score_mode": "probability",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.02, "net_stop_loss": 0.02},
    )

    assert score.missing_feature_count == 2
    assert score.missing_feature_fraction == pytest.approx(0.5)


def test_production_artifact_enforces_serialized_setup_filters() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(numeric_features=("net_profit_target", "net_stop_loss"), categorical_values={}),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "probability",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
            "selection_exclude_setup_families": ["dense_breakout_momentum"],
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {
            "candidate_type": "dense_research_bar",
            "side": "BUY",
            "event_setup_family": "dense_breakout_momentum",
            "net_profit_target": 0.02,
            "net_stop_loss": 0.02,
        },
    )

    assert not score.should_trade
    assert score.decision_reason == "setup_family_filtered"


def test_production_artifact_enforces_serialized_candidate_type_abstain() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(numeric_features=("net_profit_target", "net_stop_loss"), categorical_values={}),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "probability",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
        },
        validation_summary={},
        candidate_type_thresholds={
            "dense_research_bar": {"abstain": True, "threshold": 0.0, "source": "test"}
        },
    )

    score = score_production_artifact(
        artifact,
        {
            "candidate_type": "dense_research_bar",
            "side": "BUY",
            "net_profit_target": 0.02,
            "net_stop_loss": 0.02,
        },
    )

    assert not score.should_trade
    assert score.decision_reason == "candidate_type_threshold_filtered"


def test_artifact_candidate_type_thresholds_preserve_none_threshold() -> None:
    report = SimpleNamespace(
        folds=[
            SimpleNamespace(
                candidate_type_thresholds={
                    "dense_research_bar": {
                        "threshold": None,
                        "source": "insufficient_calibration",
                        "abstain": False,
                        "average_trade_return": 0.0,
                        "traded_signals": 0,
                    }
                }
            )
        ]
    )

    thresholds = _artifact_candidate_type_thresholds(report, enabled=True)

    assert thresholds["dense_research_bar"]["threshold"] is None
    assert thresholds["dense_research_bar"]["abstain"] is False
    assert thresholds["dense_research_bar"]["trades"] == pytest.approx(0.0)


def test_artifact_selection_score_multiplier_majority_vote_avoids_zero() -> None:
    report = SimpleNamespace(
        folds=[
            SimpleNamespace(
                model_diagnostics={
                    "target_frequency_selection": {"selection_score_multiplier": -1.0}
                }
            ),
            SimpleNamespace(
                model_diagnostics={
                    "target_frequency_selection": {"selection_score_multiplier": 1.0}
                }
            ),
        ]
    )

    multiplier = _artifact_selection_score_multiplier(
        report,
        adaptive_selection_score_direction=True,
    )

    assert multiplier == pytest.approx(1.0)


def test_production_artifact_uses_serialized_return_model_for_expected_utility() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "expected_utility",
            "selected_selection_score_floor": 0.003,
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.001,
            "label_net_stop_loss": 0.001,
        },
        validation_summary={},
        return_model=_ConstantReturnEstimator(0.004),
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.001, "net_stop_loss": 0.001},
    )

    assert score.expected_value == pytest.approx(0.0008)
    assert score.selection_score == pytest.approx(0.0042)
    assert score.should_trade


def test_production_artifact_return_first_bypasses_static_probability_threshold() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.10),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.60,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "return_first",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.001,
            "label_net_stop_loss": 0.001,
        },
        validation_summary={},
        return_model=_ConstantReturnEstimator(0.006),
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.001, "net_stop_loss": 0.001},
    )

    assert score.probability < score.selected_threshold
    assert score.expected_value < 0
    assert score.selection_score > 0
    assert score.should_trade


def test_production_artifact_return_first_respects_candidate_type_probability_floor() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.10),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.60,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "return_first",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.001,
            "label_net_stop_loss": 0.001,
        },
        validation_summary={},
        return_model=_ConstantReturnEstimator(0.006),
        candidate_type_thresholds={
            "dense_research_bar": {"threshold": 0.80, "abstain": False, "source": "test"}
        },
    )

    score = score_production_artifact(
        artifact,
        {
            "candidate_type": "dense_research_bar",
            "net_profit_target": 0.001,
            "net_stop_loss": 0.001,
        },
    )

    assert not score.should_trade
    assert score.decision_reason == "probability_below_threshold"


def test_production_artifact_requires_calibrated_candidate_type_when_configured() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.50,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "probability",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.02,
            "label_net_stop_loss": 0.02,
            "require_calibrated_selection": True,
        },
        validation_summary={},
    )

    score = score_production_artifact(
        artifact,
        {
            "candidate_type": "unseen_type",
            "net_profit_target": 0.02,
            "net_stop_loss": 0.02,
        },
    )

    assert not score.should_trade
    assert score.decision_reason == "candidate_type_threshold_filtered"


def test_production_artifact_threshold_override_gates_return_first_mode() -> None:
    artifact = ProductionModelArtifact(
        schema_version="zeroalpha.production_model.v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        encoder=FeatureEncoder(
            numeric_features=("net_profit_target", "net_stop_loss"),
            categorical_values={},
        ),
        base_models=[
            ProductionBaseModel(
                name="constant",
                estimator=_ConstantProbabilityEstimator(0.90),
                calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
                validation_weight=1.0,
            )
        ],
        stacker_mode="average",
        stacker=None,
        ensemble_calibrator=ProbabilityCalibrator(method="sigmoid_identity_low_spread"),
        validation_weights=[1.0],
        selected_threshold=0.60,
        feature_names=["net_profit_target", "net_stop_loss"],
        training_config={
            "selection_score_mode": "return_first",
            "minimum_expected_value": 0.0,
            "label_net_profit_target": 0.001,
            "label_net_stop_loss": 0.001,
        },
        validation_summary={},
        return_model=_ConstantReturnEstimator(0.006),
    )

    score = score_production_artifact(
        artifact,
        {"net_profit_target": 0.001, "net_stop_loss": 0.001},
        threshold_override=1.0,
    )

    assert score.probability < score.selected_threshold
    assert score.selection_score > 0
    assert not score.should_trade
    assert score.decision_reason == "probability_below_threshold"


def test_meta_label_walk_forward_can_prune_feature_families() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    samples = [
        replace(
            _sample(i),
            features={
                **_sample(i).features,
                "rsi_14": float(i % 100),
                "ibkrmbt_value": float(i),
            },
        )
        for i in range(120)
    ]

    report = run_meta_label_walk_forward(
        samples,
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        feature_exclude_families=("technical",),
    )

    assert "rsi_14" not in report.feature_names
    assert "ibkrmbt_value" in report.feature_names
    assert report.data_coverage["feature_family_filter"]["exclude_families"] == ["technical"]


def test_meta_label_walk_forward_can_prune_feature_patterns() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.55, minimum_expected_value=0.0),
    )
    samples = [
        replace(
            _sample(i),
            features={
                **_sample(i).features,
                "rsi_14": float(i % 100),
                "sma_distance_8": float(i) / 100,
            },
        )
        for i in range(120)
    ]

    report = run_meta_label_walk_forward(
        samples,
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        feature_exclude_patterns=("rsi_*",),
    )

    assert "rsi_14" not in report.feature_names
    assert "sma_distance_8" in report.feature_names
    assert report.data_coverage["feature_family_filter"]["exclude_patterns"] == ["rsi_*"]


def test_feature_encoder_marks_unseen_categorical_values() -> None:
    encoder = FeatureEncoder.fit(
        [
            replace(
                _sample(0),
                features={**_sample(0).features, "market_regime": "range_day"},
            )
        ]
    )
    transformed = encoder.transform(
        [
            replace(
                _sample(1),
                features={**_sample(1).features, "market_regime": "trend_day"},
            )
        ]
    )
    unknown_idx = encoder.feature_names.index("market_regime=__unknown__")

    assert transformed[0, unknown_idx] == 1.0


def test_meta_label_walk_forward_adaptive_threshold_reports_selection() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.75, minimum_expected_value=0.0),
    )
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        adaptive_threshold=True,
        min_calibration_trades=3,
    )
    assert report.folds[0].selected_threshold is not None
    assert report.folds[0].threshold_sweep


def test_adaptive_threshold_still_respects_expected_value_floor() -> None:
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.03),
    )
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        adaptive_threshold=True,
        min_calibration_trades=1,
    )

    assert report.folds[0].selected_threshold is not None
    assert report.traded_signals == 0
    assert all(prediction.expected_value < config.model.minimum_expected_value for prediction in report.predictions)
    assert "expected_value_below_threshold" in {
        prediction.decision_reason for prediction in report.predictions
    }


def test_economic_sample_weights_emphasize_larger_outcomes_but_clip_extremes() -> None:
    quiet = _sample(0)
    medium = _sample(1)
    loud = _sample(2)
    samples = [
        replace(quiet, net_return=0.001),
        replace(medium, net_return=0.02),
        replace(loud, net_return=-0.20),
    ]

    weights = _economic_sample_weights(samples)

    assert weights[0] < weights[1] < weights[2]
    assert weights[0] >= 0.5
    assert weights[2] <= 3.0


def test_candidate_type_thresholds_abstain_negative_calibration_utility() -> None:
    samples = [
        replace(_sample(i), candidate_type="weak_type", label=0, net_return=-0.02)
        for i in range(12)
    ]

    thresholds = _select_candidate_type_thresholds(
        calibration_samples=samples,
        probabilities=[0.9] * len(samples),
        thresholds=(0.5, 0.7, 0.9),
        min_trades=5,
        minimum_threshold=0.0,
    )

    assert thresholds["weak_type"]["abstain"] is True
    assert thresholds["weak_type"]["source"] == "negative_calibration_utility"


def test_candidate_type_thresholds_report_broad_prior_without_vetoing_local_edge() -> None:
    positive_slice = [
        replace(_sample(i), candidate_type="unstable_type", label=1, net_return=0.02)
        for i in range(6)
    ]
    broad_prior = [
        replace(_sample(i), candidate_type="unstable_type", label=0, net_return=-0.02)
        for i in range(12)
    ]

    thresholds = _select_candidate_type_thresholds(
        calibration_samples=positive_slice,
        probabilities=[0.8] * len(positive_slice),
        thresholds=(0.5, 0.7),
        min_trades=5,
        minimum_threshold=0.0,
        utility_samples=broad_prior,
    )

    assert thresholds["unstable_type"]["abstain"] is False
    assert thresholds["unstable_type"]["source"] == "candidate_type_calibration"
    assert thresholds["unstable_type"]["prior_average_trade_return"] == pytest.approx(-0.02)


def test_target_frequency_abstains_thin_local_bucket_when_family_prior_is_negative() -> None:
    sample = replace(
        _sample(0),
        candidate_type="weak_type",
        features={**_sample(0).features, "candidate_type": "weak_type", "market_regime": "range_day"},
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.90],
        predicted_returns=[0.02],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        candidate_type_thresholds={
            "weak_type|range_day": {
                "threshold": None,
                "source": "insufficient_calibration",
                "abstain": False,
            },
            "weak_type": {
                "threshold": None,
                "source": "negative_family_prior_utility",
                "abstain": True,
            },
        },
    )

    assert selected == set()


def test_target_frequency_uses_side_specific_short_calibration_bucket() -> None:
    sample = replace(
        _sample(0),
        side="SELL",
        candidate_type="dense_research_bar",
        features={
            **_sample(0).features,
            "candidate_type": "dense_research_bar",
            "market_regime": "range_day",
        },
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.90],
        predicted_returns=[0.02],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        candidate_type_thresholds={
            "dense_research_bar|SELL|range_day": {
                "threshold": None,
                "source": "negative_calibration_utility",
                "abstain": True,
            },
            "dense_research_bar": {
                "threshold": 0.10,
                "source": "candidate_type_calibration",
                "abstain": False,
                "average_trade_return": 0.02,
            },
        },
    )

    assert selected == set()


def test_candidate_type_thresholds_reject_thin_positive_utility() -> None:
    samples = [
        replace(_sample(i), candidate_type="thin_type", label=1, net_return=0.0005)
        for i in range(8)
    ]

    thresholds = _select_candidate_type_thresholds(
        calibration_samples=samples,
        probabilities=[0.8] * len(samples),
        thresholds=(0.5, 0.7),
        min_trades=5,
        minimum_threshold=0.0,
    )

    assert thresholds["thin_type"]["abstain"] is True
    assert thresholds["thin_type"]["source"] == "negative_calibration_utility"
    assert thresholds["thin_type"]["utility_floor"] == pytest.approx(0.003)


def test_empirical_expected_value_uses_calibration_payoff_when_available() -> None:
    sample = _sample(0)
    payoff_estimates = {
        "volatility_breakout|BUY": {
            "source": "calibration",
            "average_win": 0.06,
            "average_loss": 0.01,
        }
    }

    empirical_ev = _expected_value(
        probability=0.50,
        sample=sample,
        empirical_payoff_ev=True,
        payoff_estimates=payoff_estimates,
    )
    static_ev = _expected_value(
        probability=0.50,
        sample=sample,
        empirical_payoff_ev=False,
        payoff_estimates=payoff_estimates,
    )

    assert empirical_ev == pytest.approx(0.025)
    assert static_ev == pytest.approx(0.0)


def test_empirical_payoff_estimate_uses_all_label_zero_returns() -> None:
    samples = [
        replace(_sample(0), label=1, net_return=0.02),
        replace(_sample(1), label=0, net_return=0.004),
        replace(_sample(2), label=0, net_return=-0.006),
    ]

    estimate = _payoff_estimate(samples, min_samples=3)
    ev = _expected_value(
        probability=0.50,
        sample=samples[0],
        empirical_payoff_ev=True,
        payoff_estimates={"volatility_breakout|BUY": estimate},
    )

    assert estimate["average_label_one_return"] == pytest.approx(0.02)
    assert estimate["average_label_zero_return"] == pytest.approx(-0.001)
    assert ev == pytest.approx(0.0095)


def test_empirical_payoff_estimate_keeps_all_win_calibration() -> None:
    samples = [
        replace(_sample(0), label=1, net_return=0.015),
        replace(_sample(1), label=1, net_return=0.020),
        replace(_sample(2), label=1, net_return=0.025),
    ]

    estimate = _payoff_estimate(samples, min_samples=3)
    ev = _expected_value(
        probability=0.80,
        sample=samples[0],
        empirical_payoff_ev=True,
        payoff_estimates={"volatility_breakout|BUY": estimate},
    )

    assert estimate["source"] == "partial_calibration"
    assert estimate["average_label_one_return"] == pytest.approx(0.02)
    assert estimate["has_label_zero_returns"] is False
    assert ev == pytest.approx(0.012)


def test_positive_candidate_type_threshold_can_override_positive_compressed_ev_for_research_rank() -> None:
    sample = _sample(0)
    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.70],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.015),
        ),
        allow_negative_ev=False,
        candidate_type_thresholds={
            "volatility_breakout": {
                "threshold": 0.15,
                "source": "candidate_type_calibration",
                "abstain": False,
                "average_trade_return": 0.004,
            }
        },
    )

    assert selected == {sample.event_id}


def test_target_frequency_ranks_signal_score_before_bucket_prior() -> None:
    high_signal = replace(
        _sample(0),
        event_id="high-signal",
        candidate_type="lower_prior",
        features={**_sample(0).features, "candidate_type": "lower_prior"},
    )
    stale_prior = replace(
        _sample(1),
        event_id="stale-prior",
        candidate_type="higher_prior",
        features={**_sample(1).features, "candidate_type": "higher_prior"},
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[high_signal, stale_prior],
        probabilities=[0.80, 0.80],
        predicted_returns=[0.05, 0.001],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        candidate_type_thresholds={
            "lower_prior": {
                "threshold": 0.10,
                "source": "candidate_type_calibration",
                "abstain": False,
                "average_trade_return": 0.001,
            },
            "higher_prior": {
                "threshold": 0.10,
                "source": "candidate_type_calibration",
                "abstain": False,
                "average_trade_return": 0.02,
            },
        },
    )

    assert selected == {"high-signal"}


def test_expected_utility_rank_can_select_positive_return_forecast_with_compressed_probability() -> None:
    sample = replace(
        _sample(0),
        features={
            **_sample(0).features,
            "event_setup_family": "breakout",
            "market_regime": "trend_day",
        },
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.50],
        predicted_returns=[0.02],
        predicted_downsides=[0.002],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="expected_utility",
    )

    assert selected == {sample.event_id}


def test_default_expected_value_floor_does_not_block_positive_short_horizon_edge() -> None:
    sample = replace(
        _sample(0),
        net_profit_target=0.006,
        net_stop_loss=0.012,
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.80],
        predicted_returns=[0.001],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.001, net_stop_loss=0.001),
        ),
        allow_negative_ev=False,
        selection_score_mode="expected_utility",
        target_frequency_mode="online",
    )

    assert selected == {sample.event_id}


def test_expected_utility_rank_still_requires_positive_probability_ev() -> None:
    sample = _sample(0)

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.16],
        predicted_returns=[0.02],
        predicted_downsides=[0.002],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="expected_utility",
    )

    assert selected == set()


def test_return_first_rank_can_select_positive_return_despite_negative_probability_ev() -> None:
    sample = _sample(0)

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.16],
        predicted_returns=[0.02],
        predicted_downsides=[0.002],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="return_first",
    )

    assert selected == {sample.event_id}


def test_quota_target_frequency_probe_can_rank_below_static_probability_and_ev_gates() -> None:
    strong_rank = replace(_sample(0), event_id="strong-rank")
    weak_rank = replace(_sample(1), event_id="weak-rank")

    selected = _select_target_frequency_event_ids(
        test_samples=[strong_rank, weak_rank],
        probabilities=[0.10, 0.09],
        predicted_returns=[0.02, -0.01],
        target_trades_per_day=1,
        selected_threshold=0.90,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.03),
        ),
        allow_negative_ev=True,
        selection_score_mode="predicted_return",
        target_frequency_mode="quota",
        selection_score_floor=0.0,
    )

    assert selected == {"strong-rank"}


def test_online_target_frequency_selects_chronologically_without_future_rank() -> None:
    early = replace(_sample(0), event_id="early")
    later = replace(_sample(1), event_id="later")

    selected = _select_target_frequency_event_ids(
        test_samples=[early, later],
        probabilities=[0.55, 0.95],
        predicted_returns=[0.01, 0.05],
        target_trades_per_day=1,
        selected_threshold=0.50,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
    )

    assert selected == {"early"}


def test_online_target_frequency_respects_selection_score_floor() -> None:
    early = replace(_sample(0), event_id="early")
    later = replace(_sample(1), event_id="later")

    selected = _select_target_frequency_event_ids(
        test_samples=[early, later],
        probabilities=[0.55, 0.80],
        predicted_returns=[0.01, 0.05],
        target_trades_per_day=1,
        selected_threshold=0.50,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        selection_score_floor=0.70,
    )

    assert selected == {"later"}


def test_online_target_frequency_abstains_negative_threshold_calibration_by_default() -> None:
    samples = [
        replace(_sample(i), label=0, net_return=-0.02)
        for i in range(4)
    ]

    row = _select_threshold_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.75, 0.70, 0.65],
        thresholds=(0.50, 0.60, 0.70),
        target_trades_per_day=2,
        allow_negative_ev=False,
    )
    probe_row = _select_threshold_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.75, 0.70, 0.65],
        thresholds=(0.50, 0.60, 0.70),
        target_trades_per_day=2,
        allow_negative_ev=True,
    )

    assert row is None
    assert probe_row is not None
    assert probe_row.traded_signals > 0


def test_target_frequency_probe_does_not_choose_zero_trade_threshold_for_net_pnl() -> None:
    samples = [
        replace(_sample(i), label=0, net_return=-0.02)
        for i in range(4)
    ]

    row = _select_threshold_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.75, 0.70, 0.65],
        thresholds=(0.50, 0.70, 0.99),
        target_trades_per_day=2,
        optimize_metric="net_pnl",
        allow_negative_ev=True,
    )

    assert row is not None
    assert row.traded_signals > 0
    assert row.threshold < 0.99


def test_target_frequency_probe_softens_tiny_probability_threshold() -> None:
    samples = [
        replace(_sample(0), label=1, net_return=0.02),
        replace(_sample(1), label=0, net_return=-0.02),
    ]

    row = _select_threshold_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.04, 0.03],
        thresholds=(0.00, 0.04),
        target_trades_per_day=24,
        optimize_metric="net_pnl",
        allow_negative_ev=True,
    )

    assert row is not None
    assert row.threshold == pytest.approx(0.0)


def test_target_frequency_allows_tiny_gate_roundoff() -> None:
    sample = replace(_sample(0), event_id="near-boundary")
    probability = 0.50 - 5e-8

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[probability],
        predicted_returns=[0.01],
        target_trades_per_day=1,
        selected_threshold=0.50,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        ),
        allow_negative_ev=True,
        selection_score_mode="probability",
        target_frequency_mode="online",
        selection_score_floor=0.50,
    )

    assert selected == {"near-boundary"}


def test_online_target_frequency_can_respect_open_position_capacity() -> None:
    samples = [replace(_sample(i), event_id=f"s{i}") for i in range(8)]
    selected = _select_target_frequency_event_ids(
        test_samples=samples,
        probabilities=[0.80 for _ in samples],
        target_trades_per_day=8,
        selected_threshold=0.50,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        respect_open_positions=True,
    )

    assert selected == {"s0"}


def test_online_target_frequency_reserves_spot_cash_notional() -> None:
    samples = [replace(_sample(i), event_id=f"s{i}") for i in range(3)]
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        risk=RiskConfig(
            max_open_positions=10,
            paper_max_notional=10_000,
            risk_per_trade=1.0,
        ),
    )

    selection = ensemble_module._select_target_frequency_events(
        test_samples=samples,
        probabilities=[0.80 for _ in samples],
        target_trades_per_day=3,
        selected_threshold=0.50,
        config=config,
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        respect_open_positions=True,
        selection_execution_policy=SelectionExecutionPolicy(
            starting_equity=10_000,
            requested_notional=6_000,
            sizing_mode="fixed",
        ),
    )

    assert selection.event_ids == {"s0", "s1"}
    assert selection.notional_by_event["s0"] == pytest.approx(6_000)
    assert selection.notional_by_event["s1"] == pytest.approx(4_000)


def test_online_capacity_can_release_at_actual_exit() -> None:
    first_ts = datetime(2024, 1, 1, 0, tzinfo=UTC)
    second_ts = datetime(2024, 1, 1, 3, tzinfo=UTC)
    first_detail = replace(
        _sample(0).label_detail,
        entry_timestamp_utc=first_ts,
        exit_timestamp_utc=first_ts + timedelta(hours=2),
        vertical_barrier_timestamp_utc=first_ts + timedelta(hours=6),
        t1=first_ts + timedelta(hours=2),
    )
    second_detail = replace(
        _sample(1).label_detail,
        entry_timestamp_utc=second_ts,
        exit_timestamp_utc=second_ts + timedelta(hours=2),
        vertical_barrier_timestamp_utc=second_ts + timedelta(hours=6),
        t1=second_ts + timedelta(hours=2),
    )
    first = replace(_sample(0), event_id="first", timestamp_utc=first_ts, label_detail=first_detail)
    second = replace(_sample(1), event_id="second", timestamp_utc=second_ts, label_detail=second_detail)
    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        risk=RiskConfig(max_open_positions=1),
    )

    planned = _select_target_frequency_event_ids(
        test_samples=[first, second],
        probabilities=[0.80, 0.80],
        target_trades_per_day=2,
        selected_threshold=0.50,
        config=config,
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        respect_open_positions=True,
    )
    actual = _select_target_frequency_event_ids(
        test_samples=[first, second],
        probabilities=[0.80, 0.80],
        target_trades_per_day=2,
        selected_threshold=0.50,
        config=config,
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        respect_open_positions=True,
        capacity_release_mode="actual",
    )

    assert planned == {"first"}
    assert actual == {"first", "second"}


def test_online_position_capacity_carries_across_days() -> None:
    first_ts = datetime(2024, 1, 1, 23, tzinfo=UTC)
    second_ts = datetime(2024, 1, 2, 1, tzinfo=UTC)
    first_detail = replace(
        _sample(0).label_detail,
        entry_timestamp_utc=first_ts,
        vertical_barrier_timestamp_utc=first_ts + timedelta(hours=12),
    )
    second_detail = replace(
        _sample(1).label_detail,
        entry_timestamp_utc=second_ts,
        vertical_barrier_timestamp_utc=second_ts + timedelta(hours=12),
    )
    first = replace(_sample(0), event_id="first", timestamp_utc=first_ts, label_detail=first_detail)
    second = replace(_sample(1), event_id="second", timestamp_utc=second_ts, label_detail=second_detail)

    selected = _select_target_frequency_event_ids(
        test_samples=[first, second],
        probabilities=[0.80, 0.80],
        target_trades_per_day=1,
        selected_threshold=0.50,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="probability",
        target_frequency_mode="online",
        respect_open_positions=True,
    )

    assert selected == {"first"}


def test_adaptive_selection_score_floor_targets_turnover_without_future_rank() -> None:
    samples = [
        replace(_sample(i), event_id=f"s{i}", net_return=0.02 if i < 2 else -0.02)
        for i in range(4)
    ]
    samples[-1] = replace(samples[-1], timestamp_utc=samples[0].timestamp_utc + timedelta(days=1))

    row = _select_selection_score_floor_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.80, 0.80, 0.80],
        predicted_returns=[0.05, 0.04, 0.01, -0.02],
        target_trades_per_day=2,
        selected_threshold=0.50,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        allow_negative_ev=True,
        selection_score_mode="predicted_return",
        target_frequency_mode="online",
    )

    assert row is not None
    assert row.threshold == pytest.approx(0.04)
    assert row.traded_signals == 2
    assert row.average_trade_return == pytest.approx(0.02)


def test_adaptive_selection_score_floor_tries_to_fill_positive_turnover_goal() -> None:
    samples = [
        replace(_sample(0), event_id="first", net_return=0.02),
        replace(
            _sample(1),
            event_id="second",
            net_return=0.02,
            timestamp_utc=_sample(0).timestamp_utc + timedelta(days=1),
        ),
    ]

    row = _select_selection_score_floor_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.80],
        predicted_returns=[0.05, 0.04],
        target_trades_per_day=1,
        selected_threshold=0.50,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        target_frequency_mode="online",
        optimize_metric="net_pnl",
    )

    assert row is not None
    assert row.threshold == pytest.approx(0.04)
    assert row.traded_signals == 2


def test_walk_forward_forwards_optimize_metric_to_adaptive_score_floor(monkeypatch) -> None:
    captured: list[str] = []
    original = ensemble_module._select_selection_score_floor_for_target_frequency

    def spy(*args, **kwargs):
        captured.append(kwargs["optimize_metric"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ensemble_module,
        "_select_selection_score_floor_for_target_frequency",
        spy,
    )

    config = AppConfig(
        labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
        model=ModelConfig(minimum_probability=0.50, minimum_expected_value=0.0),
    )

    run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=config,
        model_names=["logistic"],
        train_size=50,
        calibration_size=30,
        test_size=20,
        embargo_hours=0,
        target_trades_per_day=2,
        target_frequency_mode="online",
        adaptive_selection_score_floor=True,
        optimize_metric="net_pnl",
    )

    assert captured
    assert set(captured) == {"net_pnl"}


def test_adaptive_selection_score_floor_abstains_negative_calibration_by_default() -> None:
    samples = [
        replace(_sample(i), event_id=f"s{i}", net_return=-0.02, label=0)
        for i in range(4)
    ]

    row = _select_selection_score_floor_for_target_frequency(
        calibration_samples=samples,
        probabilities=[0.80, 0.80, 0.80, 0.80],
        predicted_returns=[-0.01, -0.02, -0.03, -0.04],
        target_trades_per_day=2,
        selected_threshold=0.50,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        target_frequency_mode="online",
    )

    assert row is None


def test_quota_target_frequency_respects_selection_score_floor() -> None:
    sample = replace(_sample(0), event_id="negative-rank")

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.80],
        predicted_returns=[-0.001],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        allow_negative_ev=True,
        selection_score_mode="predicted_return",
        target_frequency_mode="quota",
        selection_score_floor=0.0,
    )

    assert selected == set()


def test_quota_target_frequency_respects_selection_score_ceiling() -> None:
    over_ceiling = replace(_sample(0), event_id="over-ceiling")
    under_ceiling = replace(_sample(1), event_id="under-ceiling")

    selected = _select_target_frequency_event_ids(
        test_samples=[over_ceiling, under_ceiling],
        probabilities=[0.90, 0.80],
        predicted_returns=[0.02, 0.005],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        target_frequency_mode="quota",
        selection_score_ceiling=0.01,
    )

    assert selected == {"under-ceiling"}


def test_quota_target_frequency_rejects_negative_ev_by_default() -> None:
    sample = replace(_sample(0), event_id="negative-ev")

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.10],
        target_trades_per_day=1,
        selected_threshold=0.0,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="expected_value",
        target_frequency_mode="quota",
    )
    forced = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.10],
        target_trades_per_day=1,
        selected_threshold=0.0,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_expected_value=0.0),
        ),
        allow_negative_ev=True,
        selection_score_mode="expected_value",
        target_frequency_mode="quota",
    )

    assert selected == set()
    assert forced == {"negative-ev"}


def test_target_frequency_rejects_negative_signal_ev_even_with_positive_bucket_prior() -> None:
    sample = _sample(0)

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.49],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_expected_value=0.0),
        ),
        allow_negative_ev=False,
        selection_score_mode="expected_value",
        target_frequency_mode="online",
        candidate_type_thresholds={
            "volatility_breakout": {
                "threshold": 0.10,
                "source": "candidate_type_calibration",
                "abstain": False,
                "average_trade_return": 0.004,
            }
        },
    )

    assert selected == set()


def test_target_frequency_can_require_calibrated_candidate_bucket() -> None:
    sample = _sample(0)

    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.80],
        predicted_returns=[0.02],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        candidate_type_thresholds={
            "volatility_breakout": {
                "threshold": None,
                "source": "insufficient_calibration",
                "abstain": False,
            }
        },
        require_calibrated_selection=True,
    )

    assert selected == set()


def test_target_frequency_respects_group_spacing_and_daily_caps() -> None:
    samples = [
        replace(
            _sample(i),
            candidate_type="volatility_breakout",
            features={**_sample(i).features, "candidate_type": "volatility_breakout", "market_regime": "range_day"},
        )
        for i in range(4)
    ]

    selected = _select_target_frequency_event_ids(
        test_samples=samples,
        probabilities=[0.80, 0.79, 0.78, 0.77],
        predicted_returns=[0.02, 0.019, 0.018, 0.017],
        target_trades_per_day=4,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        min_signal_spacing_hours=2,
        max_signals_per_group_per_day=2,
    )

    assert selected == {samples[0].event_id, samples[2].event_id}


def test_target_frequency_can_veto_setup_families_at_selection_time() -> None:
    strong_blocked = replace(
        _sample(0),
        event_id="blocked",
        features={**_sample(0).features, "event_setup_family": "dense_trend_continuation"},
    )
    weaker_allowed = replace(
        _sample(1),
        event_id="allowed",
        features={**_sample(1).features, "event_setup_family": "dense_baseline"},
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[strong_blocked, weaker_allowed],
        probabilities=[0.95, 0.70],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=True,
        target_frequency_mode="quota",
        selection_exclude_setup_families=("dense_trend_continuation",),
    )

    assert selected == {"allowed"}


def test_target_frequency_matches_explicit_specialist_setup_families() -> None:
    legacy_blocked_specialist_allowed = replace(
        _sample(0),
        event_id="specialist",
        features={
            **_sample(0).features,
            "event_setup_family": "dense_breakout_momentum",
            "event_specialist_setup_family": "liquidity_vacuum_breakout",
        },
    )
    legacy_allowed_specialist_blocked = replace(
        _sample(1),
        event_id="legacy",
        features={
            **_sample(1).features,
            "event_setup_family": "dense_baseline",
            "event_specialist_setup_family": "chop_no_trade",
        },
    )

    selected = _select_target_frequency_event_ids(
        test_samples=[legacy_blocked_specialist_allowed, legacy_allowed_specialist_blocked],
        probabilities=[0.70, 0.95],
        target_trades_per_day=1,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=True,
        target_frequency_mode="quota",
        selection_setup_families=("liquidity_vacuum_breakout",),
    )

    assert selected == {"specialist"}


def test_target_frequency_limits_duplicate_timestamp_bets() -> None:
    first = _sample(0)
    second = replace(_sample(0), event_id="same-bar-other-setup", candidate_type="active_squeeze_breakout")

    selected = _select_target_frequency_event_ids(
        test_samples=[first, second],
        probabilities=[0.80, 0.79],
        predicted_returns=[0.02, 0.019],
        target_trades_per_day=2,
        selected_threshold=0.10,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.0045, net_stop_loss=0.003)),
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        max_signals_per_timestamp=1,
    )

    assert selected == {first.event_id}


def test_low_spread_probability_calibration_preserves_rank_signal() -> None:
    calibrator = ProbabilityCalibrator.fit(
        probabilities=[0.151, 0.153, 0.154, 0.156],
        labels=[0, 1, 0, 1],
        method="sigmoid",
    )

    calibrated = calibrator.predict([0.151, 0.156])

    assert calibrator.model is None
    assert calibrator.method == "sigmoid_identity_low_spread"
    assert calibrated[0] == pytest.approx(0.151)
    assert calibrated[1] == pytest.approx(0.156)


def test_one_class_probability_calibration_preserves_raw_signal() -> None:
    calibrator = ProbabilityCalibrator.fit(
        probabilities=[0.22, 0.46, 0.71],
        labels=[0, 0, 0],
        method="sigmoid",
    )

    calibrated = calibrator.predict([0.31, 0.82])

    assert calibrator.model is None
    assert calibrator.method == "sigmoid_identity_one_class"
    assert calibrated[0] == pytest.approx(0.31)
    assert calibrated[1] == pytest.approx(0.82)


def test_meta_label_report_stays_serializable_with_new_fields() -> None:
    samples = [_sample(i) for i in range(120)]
    # Create synthetic bars from the sample feature timestamps indirectly by reusing the
    # dedicated model walk-forward test above for speed. The sweep integration is covered
    # by CLI smoke tests; here we assert the result dataclass can rank reports.
    report = run_meta_label_walk_forward(
        samples,
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        model_names=["logistic"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
    )
    assert report.samples == 120
    assert report.folds[0].reliability_buckets
    assert report.folds[0].candidate_type_calibration
    assert isinstance(report.folds[0].candidate_type_thresholds, dict)
    assert isinstance(report.folds[0].empirical_payoff, dict)
    assert isinstance(report.folds[0].model_diagnostics, dict)
    assert all(0 <= prediction.probability <= 1 for prediction in report.predictions)


def test_meta_label_walk_forward_can_tune_lightgbm_fold_locally() -> None:
    pytest.importorskip("lightgbm")
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(120)],
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        model_names=["lightgbm"],
        train_size=50,
        calibration_size=20,
        test_size=20,
        embargo_hours=24,
        tune_hyperparameters=True,
    )
    assert report.folds
    assert "lightgbm" in report.folds[0].selected_model_params
    assert report.folds[0].selected_model_params["lightgbm"].get("class_weight") == "balanced"


def test_wide_hpo_profile_adds_regularized_model_candidates() -> None:
    lightgbm_deep = _hpo_grid("lightgbm", profile="deep")
    lightgbm_wide = _hpo_grid("lightgbm", profile="wide")
    lightgbm_quota = _hpo_grid("lightgbm", profile="quota")
    lightgbm_capacity = _hpo_grid("lightgbm", profile="capacity")
    histgb_wide = _hpo_grid("histgb", profile="wide")
    gboost_wide = _hpo_grid("gboost", profile="wide")
    forest_wide = _hpo_grid("extratrees", profile="wide")
    randomforest_wide = _hpo_grid("randomforest", profile="wide")

    assert len(lightgbm_wide) > len(lightgbm_deep)
    assert lightgbm_quota == lightgbm_wide
    assert lightgbm_capacity == lightgbm_wide
    assert any("reg_alpha" in params for params in lightgbm_wide)
    assert any(params.get("min_samples_leaf", 0) >= 60 for params in histgb_wide)
    assert any(params.get("subsample", 1.0) < 1.0 for params in gboost_wide)
    assert any(params.get("max_depth") == 1 for params in gboost_wide)
    assert any(params.get("bootstrap") is True for params in forest_wide)
    assert any(params.get("criterion") == "log_loss" for params in randomforest_wide)
    assert any(params.get("class_weight") is None for params in randomforest_wide)


def test_hpo_trial_cap_samples_across_wide_grid() -> None:
    grid = _hpo_grid("lightgbm", profile="wide")
    limited = _limit_hpo_grid(grid, 4)

    assert len(limited) == 4
    assert limited[0] == grid[0]
    assert limited[-1] == grid[-1]


def test_hpo_inner_validation_purges_overlapping_training_labels() -> None:
    samples = [_sample(i) for i in range(80)]
    split_at = 64
    config = AppConfig(labels=LabelConfig(max_holding_hours=12))

    kept = _purged_hpo_inner_training_indices(samples, split_at=split_at, config=config)
    validation_start = samples[split_at].timestamp_utc

    assert kept
    assert max(kept) < split_at
    assert all(samples[idx].t1 < validation_start for idx in kept)
    assert split_at - 1 not in kept


def test_quota_frequency_returns_match_spaced_daily_selection() -> None:
    samples = [
        replace(
            _sample(i),
            candidate_type="volatility_breakout",
            features={
                **_sample(i).features,
                "candidate_type": "volatility_breakout",
                "market_regime": "range_day",
            },
            net_return=0.02 if i in {0, 2} else -0.02,
        )
        for i in range(4)
    ]

    returns = _quota_frequency_returns(
        samples=samples,
        scores=[0.90, 0.89, 0.88, 0.87],
        target_trades_per_day=4,
        min_signal_spacing_hours=2,
    )

    assert returns == [0.02, 0.02]


def test_online_threshold_frequency_returns_can_respect_capacity() -> None:
    samples = []
    for i in range(4):
        base_sample = _sample(i)
        net_return = 0.009 if i in {0, 3} else -0.009
        label = 1 if i in {0, 3} else 0
        detail = replace(
            base_sample.label_detail,
            vertical_barrier_timestamp_utc=base_sample.timestamp_utc + timedelta(hours=3),
            exit_timestamp_utc=base_sample.timestamp_utc + timedelta(hours=3),
            t1=base_sample.timestamp_utc + timedelta(hours=3),
            outcome_type="upper" if label else "lower",
            gross_return=net_return,
            net_return=net_return,
            label=label,
        )
        samples.append(
            replace(
                base_sample,
                t1=detail.t1,
                label_detail=detail,
                net_return=net_return,
                label=label,
                outcome_type=detail.outcome_type,
            )
        )

    returns = _online_threshold_frequency_returns(
        samples=samples,
        scores=[0.90, 0.89, 0.80, 0.88],
        target_trades_per_day=4,
        config=AppConfig(),
        respect_open_positions=True,
    )

    assert sorted(returns) == [0.009, 0.009]


def test_online_threshold_frequency_returns_can_use_expected_value_floor() -> None:
    base = _sample(0)
    low_ev = replace(
        base,
        event_id="low-ev",
        net_profit_target=0.001,
        net_stop_loss=0.02,
        net_return=-0.01,
    )
    high_ev = replace(
        base,
        event_id="high-ev",
        net_profit_target=0.03,
        net_stop_loss=0.005,
        net_return=0.02,
    )

    returns = _online_threshold_frequency_returns(
        samples=[low_ev, high_ev],
        scores=[0.60, 0.60],
        target_trades_per_day=1,
        config=AppConfig(),
        selection_score_mode="expected_value",
        selection_score_floor=0.005,
    )

    assert returns == [0.02]


def test_target_frequency_hpo_returns_can_rank_with_predicted_returns() -> None:
    early = replace(_sample(0), event_id="early", net_return=-0.02)
    late = replace(_sample(1), event_id="late", net_return=0.02)

    returns = _online_threshold_frequency_returns(
        samples=[early, late],
        scores=[0.70, 0.70],
        target_trades_per_day=1,
        config=AppConfig(model=ModelConfig(minimum_expected_value=0.0)),
        selection_score_mode="return_first",
        target_frequency_mode="quota",
        predicted_returns=[-0.02, 0.02],
        predicted_downsides=[0.01, 0.001],
    )

    assert returns == [0.02]


def test_default_fold_sizes_cover_more_walk_forward_windows_for_large_datasets() -> None:
    train_size, calibration_size, test_size = default_fold_sizes(750)

    assert train_size == 262
    assert calibration_size == 75
    assert test_size == 75
    assert (750 - (train_size + calibration_size + test_size)) // test_size + 1 >= 5


def test_target_trade_frequency_uses_daily_rank_not_zero_threshold() -> None:
    report = run_meta_label_walk_forward(
        [_sample(i) for i in range(160)],
        config=AppConfig(labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02)),
        model_names=["logistic"],
        train_size=60,
        calibration_size=24,
        test_size=48,
        embargo_hours=24,
        target_trades_per_day=2,
        allow_negative_ev_target_frequency=True,
    )

    assert report.folds[0].selected_threshold_source == "target_frequency_rank"
    by_day: dict[object, int] = {}
    for prediction in report.predictions:
        if prediction.should_trade:
            day = datetime.fromisoformat(prediction.timestamp_utc).date()
            by_day[day] = by_day.get(day, 0) + 1
            assert prediction.decision_reason == "target_frequency_rank"
    assert by_day
    assert max(by_day.values()) <= 2


def test_calibration_split_uses_later_selection_slice_when_viable() -> None:
    samples = [_sample(i) for i in range(40)]
    base, ensemble, threshold = _split_calibration_samples(samples)

    assert base
    assert ensemble
    assert threshold
    assert base != samples
    assert ensemble != samples
    assert threshold != samples
    assert base[-1].timestamp_utc < ensemble[0].timestamp_utc
    assert ensemble[-1].timestamp_utc < threshold[0].timestamp_utc
