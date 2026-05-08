from pathlib import Path
from dataclasses import replace
import json

from zeroalpha.models.ibkr_experiments import (
    BASE_CONTEXT_BARS,
    FULL_CONTEXT_BARS,
    build_backtest_command,
    build_ibkr_experiment_suite,
    build_live_valid_strict_10k_experiment_suite,
    experiment_artifact_path,
    filter_experiments,
    parse_backtest_artifact,
    summarize_results,
    write_manifest,
)


def _last_value(command: list[str], option: str) -> str:
    return command[len(command) - 1 - command[::-1].index(option) + 1]


def test_ibkr_experiment_suite_has_unique_named_categories() -> None:
    experiments = build_ibkr_experiment_suite()
    names = [experiment.name for experiment in experiments]
    categories = {experiment.category for experiment in experiments}

    assert len(names) == len(set(names))
    assert "anchor" in categories
    assert "hpo" in categories
    assert "capacity_sizing" in categories
    assert "adaptive_exit" in categories
    assert "data_features" in categories
    assert "phase17_revival" in categories
    assert "fast_hold_sweep" in categories
    assert "live_aligned" in categories
    assert "live_aligned_deep" in categories
    assert "live_aligned_dynamic" in categories
    assert "live_valid_strict_10k" in categories
    assert "live_valid_strict_10k_models" in categories
    assert "live_valid_strict_10k_features" in categories
    assert "live_valid_strict_10k_horizons" in categories
    assert "live_valid_strict_10k_specialists" in categories
    assert "stretch_goal" in categories
    assert any(experiment.name == "champion_repro_extratrees_return_first_1250x8" for experiment in experiments)
    assert any(experiment.name == "phase17_no_pm_bucket_repro" for experiment in experiments)
    assert any(experiment.name == "phase17_fixed_10k_slots3_quality" for experiment in experiments)
    assert any(experiment.name == "live_aligned_phase17_market_entry_bucket" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_market_entry_fixed2500_forest_mix" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_inverse_ev_bucket_1000_2000_3333" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_inverse_ev_1000_3000_4000_hpo28" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_inverse_ev_1000_3000_4000_ceiling0" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_inverse_ev_1000_3000_4000_stride10_spacing15" for experiment in experiments)
    assert any(experiment.name == "live_aligned_strict_10k_inverse_ev_1000_3000_3333_tpd20" for experiment in experiments)
    assert any(experiment.name == "lv10k_model_alltrees_sigmoid" for experiment in experiments)


def test_build_backtest_command_keeps_spot_crypto_execution(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["champion_repro_extratrees_return_first_1250x8"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert command[:5] == ["python", "-m", "zeroalpha.cli", "backtest", "ml"]
    assert command[command.index("--instrument-model") + 1] == "spot_crypto"
    assert command[command.index("--side-mode") + 1] == "long"
    assert command[command.index("--paper-max-notional") + 1] == "10000"
    assert command[command.index("--minimum-commission") + 1] == "1.75"
    assert command[command.index("--tier-rate") + 1] == "0.0018"
    assert command[command.index("--context-bars-jsonl") + 1] == BASE_CONTEXT_BARS
    assert "--research-gate" in command
    assert "--allow-negative-ev-frequency-probe" in command


def test_live_valid_strict_10k_suite_enforces_market_spot_one_second_replay(tmp_path: Path) -> None:
    experiments = build_live_valid_strict_10k_experiment_suite()
    commands = [
        build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")
        for experiment in experiments
    ]

    assert len(experiments) >= 300
    assert {"6", "8", "10", "12", "16", "20"}.issubset(
        {_last_value(command, "--target-trades-per-day") for command in commands}
    )
    assert {"1", "2", "3", "4", "5"}.issubset(
        {_last_value(command, "--max-open-positions") for command in commands}
    )
    assert {"probability", "expected_value", "selection_score", "predicted_return", "trade_score"}.issubset(
        {_last_value(command, "--sizing-score-field") for command in commands}
    )
    assert {"high", "low"}.issubset(
        {_last_value(command, "--sizing-score-direction") for command in commands}
    )
    for command in commands[:20]:
        assert _last_value(command, "--entry-order-model") == "market"
        assert _last_value(command, "--instrument-model") == "spot_crypto"
        assert _last_value(command, "--side-mode") == "long"
        assert _last_value(command, "--starting-equity") == "10000"
        assert _last_value(command, "--paper-max-notional") == "10000"
        assert _last_value(command, "--label-bars-jsonl").endswith("_1s_20260504_6h_merged.jsonl")
        assert _last_value(command, "--execution-bars-jsonl").endswith("_1s_20260504_6h_merged.jsonl")
        assert "--strict-live-valid-1s" in command
        assert _last_value(command, "--min-live-valid-1s-days") == "7"
        assert _last_value(command, "--preferred-live-valid-1s-days") == "14"
        assert _last_value(command, "--min-live-valid-folds") == "2"
        assert _last_value(command, "--selection-score") == "capital_efficiency"
        assert _last_value(command, "--min-expected-gross-edge-bps") == "38.2"
        assert "--adaptive-horizon" in command
        assert "--dynamic-exit-overlay" in command
        assert _last_value(command, "--min-holding-seconds") == "1"
        assert _last_value(command, "--adaptive-horizon-granularity-seconds") == "1"


def test_data_feature_experiment_adds_full_ibkr_context(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["data_context_full_ibkr_bidask_futures"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert command[command.index("--context-bars-jsonl") + 1] == BASE_CONTEXT_BARS
    assert command[command.index("--context-bars-jsonl", command.index("--context-bars-jsonl") + 1) + 1] == FULL_CONTEXT_BARS
    assert "BTC_BIDASK=" in FULL_CONTEXT_BARS
    assert "MET_BIDASK=" in FULL_CONTEXT_BARS


def test_phase17_revival_experiment_uses_online_bucket_sizing(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["phase17_no_pm_bucket_repro"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert command[command.index("--instrument-model") + 1] == "spot_crypto"
    assert command[command.index("--target-frequency-mode", command.index("--target-frequency-mode") + 1) + 1] == "online"
    assert command[command.index("--selection-score", command.index("--selection-score") + 1) + 1] == "expected_value"
    assert _last_value(command, "--min-history-bars") == "1438"
    assert _last_value(command, "--train-size") == "0"
    assert _last_value(command, "--calibration-size") == "0"
    assert _last_value(command, "--test-size") == "0"
    assert command[command.index("--sizing-mode") + 1] == "score_bucket"
    assert command[command.index("--sizing-base-notional") + 1] == "5000"
    assert command[command.index("--sizing-high-notional") + 1] == "15000"
    assert "--adaptive-selection-score-floor" in command


def test_phase17_fixed_quality_experiment_uses_three_ten_k_slots(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["phase17_fixed_10k_slots3_quality"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert command[command.index("--sizing-mode") + 1] == "fixed"
    assert _last_value(command, "--notional") == "10000"
    assert _last_value(command, "--paper-max-notional") == "30000"
    assert _last_value(command, "--max-open-positions") == "3"
    assert command[command.index("--instrument-model") + 1] == "spot_crypto"


def test_phase17_micro_experiment_replays_on_one_second_bars(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["phase17_label_replay_1s_micro_window"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert command[command.index("--interval", command.index("--interval") + 1) + 1] == "1s"
    assert command[command.index("--label-bars-jsonl") + 1].endswith("_6h_merged.jsonl")
    assert command[command.index("--execution-bars-jsonl") + 1].endswith("_6h_merged.jsonl")
    assert command[command.index("--adaptive-horizon-granularity-seconds") + 1] == "1"


def test_live_aligned_experiments_model_market_entries(tmp_path: Path) -> None:
    experiment = filter_experiments(
        build_ibkr_experiment_suite(),
        names=["live_aligned_strict_10k_market_entry_gboost_hpo"],
    )[0]
    command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")

    assert _last_value(command, "--entry-order-model") == "market"
    assert "gboost" in _last_value(command, "--models")
    assert _last_value(command, "--hpo-profile") == "capacity"
    assert _last_value(command, "--capacity-release-mode") == "planned"
    assert command[command.index("--instrument-model") + 1] == "spot_crypto"


def test_live_aligned_extended_experiments_try_new_models_and_sizing(tmp_path: Path) -> None:
    experiments = filter_experiments(
        build_ibkr_experiment_suite(),
        names=[
            "live_aligned_strict_10k_market_entry_fixed2500_forest_mix",
            "live_aligned_strict_10k_market_entry_fixed2000_slots5",
            "live_aligned_strict_10k_market_entry_best_stacker_fixed2500",
            "live_aligned_strict_10k_inverse_ev_bucket_1000_2000_3333",
        ],
    )
    commands_by_name = {
        experiment.name: build_backtest_command(
            experiment,
            output_dir=tmp_path,
            python_executable="python",
        )
        for experiment in experiments
    }
    commands = list(commands_by_name.values())

    assert any("randomforest" in _last_value(command, "--models") for command in commands)
    assert any(_last_value(command, "--notional") == "2000" for command in commands)
    assert any(_last_value(command, "--stacker") == "best" for command in commands)
    inverse_command = commands_by_name["live_aligned_strict_10k_inverse_ev_bucket_1000_2000_3333"]
    assert _last_value(inverse_command, "--sizing-mode") == "score_bucket"
    assert _last_value(inverse_command, "--sizing-score-field") == "expected_value"
    assert _last_value(inverse_command, "--sizing-score-direction") == "low"
    assert _last_value(inverse_command, "--notional") == "3333"
    assert all(_last_value(command, "--entry-order-model") == "market" for command in commands)
    assert all(command[command.index("--instrument-model") + 1] == "spot_crypto" for command in commands)


def test_live_aligned_dynamic_experiments_include_true_one_second_replay(tmp_path: Path) -> None:
    experiments = filter_experiments(
        build_ibkr_experiment_suite(),
        names=[
            "live_aligned_strict_10k_adaptive_1m_move50_fixed2000",
            "live_aligned_strict_10k_1s_replay_move60_fixed2000",
        ],
    )
    commands = [
        build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")
        for experiment in experiments
    ]
    adaptive_1m, replay_1s = commands

    assert "--adaptive-horizon" in adaptive_1m
    assert _last_value(adaptive_1m, "--min-holding-seconds") == "1"
    assert _last_value(adaptive_1m, "--notional") == "2000"
    assert _last_value(replay_1s, "--label-bars-jsonl").endswith("_1s_20260504_6h_merged.jsonl")
    assert _last_value(replay_1s, "--execution-bars-jsonl").endswith("_1s_20260504_6h_merged.jsonl")
    assert _last_value(replay_1s, "--adaptive-horizon-granularity-seconds") == "1"
    assert _last_value(replay_1s, "--entry-order-model") == "market"


def test_fast_hold_sweep_enforces_one_or_five_second_minimums(tmp_path: Path) -> None:
    experiments = filter_experiments(build_ibkr_experiment_suite(), categories=["fast_hold_sweep"])
    min_holds = set()

    for experiment in experiments:
        command = build_backtest_command(experiment, output_dir=tmp_path, python_executable="python")
        min_holds.add(command[command.index("--min-holding-seconds") + 1])
        assert command[command.index("--interval", command.index("--interval") + 1) + 1] == "1s"
        assert command[command.index("--label-bars-jsonl") + 1].endswith("_6h_merged.jsonl")
        assert command[command.index("--execution-bars-jsonl") + 1].endswith("_6h_merged.jsonl")
        assert command[command.index("--instrument-model") + 1] == "spot_crypto"

    assert min_holds == {"1", "5"}


def test_write_manifest_includes_commands(tmp_path: Path) -> None:
    experiments = build_ibkr_experiment_suite()[:3]
    path = write_manifest(experiments, tmp_path, python_executable="python")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["summary"]["total"] == 3
    assert payload["experiments"][0]["artifact"] == str(experiment_artifact_path(tmp_path, experiments[0]))
    assert payload["experiments"][0]["command"][:4] == ["python", "-m", "zeroalpha.cli", "backtest"]


def test_parse_backtest_artifact_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "anchor" / "smoke.json"
    artifact.parent.mkdir()
    artifact.write_text(
        """
        {
          "summary": {
            "trades": 25,
            "trades_per_prediction_day": 10.02,
            "trades_per_active_day": 9.0,
            "candidate_predictions": 270,
            "model_approved_signals": 25,
            "rejected_signals": 12,
            "net_pnl": 428.43,
            "gross_pnl": 547.81,
            "total_return": 0.042843,
            "sharpe": 16.6,
            "profit_factor": 25.8,
            "max_drawdown": 0.0017,
            "hit_rate": 0.96,
            "total_commission_estimate": 112.5,
            "total_slippage_cost_estimate": 31.25,
            "total_spread_cost_estimate": 6.25
          }
        }
        """,
        encoding="utf-8",
    )

    result = parse_backtest_artifact(artifact)

    assert result.status == "ok"
    assert result.trades == 25
    assert result.net_pnl == 428.43
    assert result.rank_key[0] == 428.43


def test_summarize_results_applies_multiple_testing_haircut(tmp_path: Path) -> None:
    first = parse_backtest_artifact(tmp_path / "missing_first.json")
    second = parse_backtest_artifact(tmp_path / "missing_second.json")
    first = replace(first, status="ok", daily_sharpe=20.0, sharpe=20.0, net_pnl=100.0)
    second = replace(second, status="ok", daily_sharpe=10.0, sharpe=10.0, net_pnl=50.0)

    ranked = summarize_results([first, second])

    assert ranked[0].multiple_testing_trials == 2
    assert ranked[0].multiple_testing_haircut > 0
    assert ranked[0].deflated_sharpe < ranked[0].daily_sharpe
