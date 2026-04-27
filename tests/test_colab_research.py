from pathlib import Path

from zeroalpha.models.colab_research import (
    ResearchExperiment,
    build_experiment_matrix,
    build_signal_audit_command,
    parse_signal_audit_artifact,
)


def test_build_signal_audit_command_includes_research_safety_flags(tmp_path: Path) -> None:
    experiment = ResearchExperiment(
        name="smoke",
        interval="15m",
        years=3,
        context_interval="1h",
        candidate_mode="active",
        candidate_types="active_pullback_reclaim",
        max_holding_hours=4,
        net_profit_target=0.0035,
        net_stop_loss=0.003,
        minimum_gross_profit_bps=45,
        minimum_gross_stop_bps=22,
        selection_score="expected_utility",
        calibration_method="isotonic",
        stacker="weighted",
    )

    command = build_signal_audit_command(
        experiment,
        artifact_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        python_executable="python",
    )

    assert command[:4] == ["python", "-m", "zeroalpha.cli", "model"]
    assert "--research-gate" in command
    assert "--hpo" in command
    assert "--empirical-payoff-ev" in command
    assert "--candidate-types" in command
    assert "active_pullback_reclaim" in command


def test_build_experiment_matrix_has_lower_timeframe_runs() -> None:
    experiments = build_experiment_matrix(models="logistic,histgb")
    names = {experiment.name for experiment in experiments}

    assert any(name.startswith("15m_") for name in names)
    assert any(name.startswith("5m_") for name in names)
    assert any(name.startswith("1m_") for name in names)
    assert any("forced_frequency_probe" in name for name in names)


def test_parse_signal_audit_artifact_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text(
        """
        {
          "samples": 123,
          "raw_candidates_per_day": 8.5,
          "backtest_summary": {
            "trades": 12,
            "trades_per_prediction_day": 0.4,
            "net_pnl": 42.0,
            "total_return": 0.0042,
            "sharpe": 1.7,
            "profit_factor": 1.4,
            "max_drawdown": 0.02,
            "hit_rate": 0.58
          }
        }
        """,
        encoding="utf-8",
    )

    result = parse_signal_audit_artifact(artifact)

    assert result.status == "ok"
    assert result.trades == 12
    assert result.raw_candidates_per_day == 8.5
    assert result.sharpe == 1.7
