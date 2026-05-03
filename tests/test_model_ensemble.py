from datetime import UTC, datetime, timedelta
from dataclasses import replace

import pytest

from zeroalpha.config import AppConfig, LabelConfig, ModelConfig
from zeroalpha.domain import TripleBarrierLabel
from zeroalpha.models.dataset import MetaLabelSample
from zeroalpha.models.ensemble import (
    FeatureEncoder,
    ProbabilityCalibrator,
    _economic_sample_weights,
    _expected_value,
    _hpo_grid,
    _limit_hpo_grid,
    _payoff_estimate,
    _quota_frequency_returns,
    _select_candidate_type_thresholds,
    _select_target_frequency_event_ids,
    _split_calibration_samples,
    default_fold_sizes,
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


def test_positive_candidate_type_threshold_can_override_compressed_ev_for_research_rank() -> None:
    sample = _sample(0)
    selected = _select_target_frequency_event_ids(
        test_samples=[sample],
        probabilities=[0.16],
        target_trades_per_day=1,
        selected_threshold=0.15,
        config=AppConfig(
            labels=LabelConfig(net_profit_target=0.02, net_stop_loss=0.02),
            model=ModelConfig(minimum_probability=0.60, minimum_expected_value=0.0075),
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


def test_quota_target_frequency_can_rank_below_static_probability_and_ev_gates() -> None:
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
        allow_negative_ev=False,
        selection_score_mode="predicted_return",
        target_frequency_mode="quota",
        selection_score_floor=0.0,
    )

    assert selected == {"strong-rank"}


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
    histgb_wide = _hpo_grid("histgb", profile="wide")
    forest_wide = _hpo_grid("extratrees", profile="wide")

    assert len(lightgbm_wide) > len(lightgbm_deep)
    assert lightgbm_quota == lightgbm_wide
    assert any("reg_alpha" in params for params in lightgbm_wide)
    assert any(params.get("min_samples_leaf", 0) >= 60 for params in histgb_wide)
    assert any(params.get("bootstrap") is True for params in forest_wide)


def test_hpo_trial_cap_samples_across_wide_grid() -> None:
    grid = _hpo_grid("lightgbm", profile="wide")
    limited = _limit_hpo_grid(grid, 4)

    assert len(limited) == 4
    assert limited[0] == grid[0]
    assert limited[-1] == grid[-1]


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
