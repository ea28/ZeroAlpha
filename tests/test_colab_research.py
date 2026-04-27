from pathlib import Path

from zeroalpha.models.colab_research import (
    ResearchExperiment,
    build_experiment_matrix,
    build_signal_audit_command,
    parse_signal_audit_artifact,
    write_experiment_manifest,
    write_research_report,
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
    experiments = build_experiment_matrix(models="logistic,histgb", years_15m=7, years_5m=5, years_1m=3)
    names = {experiment.name for experiment in experiments}

    assert any(name.startswith("15m_7y_") for name in names)
    assert any(name.startswith("5m_5y_") for name in names)
    assert any(name.startswith("1m_3y_") for name in names)
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


def test_write_manifest_and_research_report(tmp_path: Path) -> None:
    experiments = build_experiment_matrix(
        include_15m=True,
        include_5m=False,
        include_1m=False,
        models="logistic,histgb",
    )[:2]
    manifest = tmp_path / "manifest.json"
    write_experiment_manifest(experiments, manifest)

    assert "15m_6y" in manifest.read_text(encoding="utf-8")

    artifact = tmp_path / "result.json"
    artifact.write_text(
        """
        {
          "samples": 123,
          "raw_candidates_per_day": 8.5,
          "backtest_summary": {
            "trades": 12,
            "trades_per_prediction_day": 4.2,
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
    report = tmp_path / "report.json"
    write_research_report([parse_signal_audit_artifact(artifact)], report)

    text = report.read_text(encoding="utf-8")
    assert '"tpd_ge_4"' in text
    assert '"ok_results": 1' in text
