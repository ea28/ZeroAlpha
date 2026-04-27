from pathlib import Path
import json

from zeroalpha.models.colab_research import (
    FOUNDATION_MODELS,
    ResearchExperiment,
    build_experiment_matrix,
    build_signal_audit_command,
    foundation_model_stack_from_smoke,
    make_cost_stress_experiments,
    make_foundation_kronos_experiments,
    parse_signal_audit_artifact,
    prefetch_binance_cache,
    prefetch_requests_for_experiments,
    summarize_artifacts,
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
    assert "--allow-data-gaps" in command
    assert "--assumed-spread-bps" in command
    assert "ETHUSDT,SOLUSDT,ETHBTC,BNBUSDT,XRPUSDT" in command


def test_build_experiment_matrix_has_lower_timeframe_runs() -> None:
    experiments = build_experiment_matrix(models="logistic,histgb", years_15m=7, years_5m=5, years_1m=3)
    names = {experiment.name for experiment in experiments}

    assert any(name.startswith("15m_7y_") for name in names)
    assert any(name.startswith("5m_5y_") for name in names)
    assert any(name.startswith("1m_3y_") for name in names)
    assert any("forced_frequency_probe" in name for name in names)


def test_build_experiment_matrix_keeps_1m_research_runs() -> None:
    experiments = build_experiment_matrix(models="logistic,histgb", years_1m=3)
    one_minute = [experiment for experiment in experiments if experiment.interval == "1m"]

    assert one_minute
    assert all(experiment.years == 3 for experiment in one_minute)
    assert all(experiment.dense_stride_bars == 5 for experiment in one_minute)


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


def test_summarize_artifacts_skips_manifest_json(tmp_path: Path) -> None:
    (tmp_path / "experiment_manifest.json").write_text("[]", encoding="utf-8")
    (tmp_path / "research_report.json").write_text('{"top_overall": []}', encoding="utf-8")
    artifact = tmp_path / "result.json"
    artifact.write_text(
        """
        {
          "samples": 1,
          "raw_candidates_per_day": 1,
          "backtest_summary": {
            "trades": 1,
            "trades_per_prediction_day": 1,
            "net_pnl": 1,
            "total_return": 0.1,
            "sharpe": 1,
            "profit_factor": 1,
            "max_drawdown": 0,
            "hit_rate": 1
          }
        }
        """,
        encoding="utf-8",
    )

    results = summarize_artifacts(tmp_path)

    assert [result.name for result in results] == ["result"]


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
          "candidate_type_summary": {
            "active_pullback_reclaim": {"samples": 100, "label_rate": 0.55}
          },
          "regime_summary": {
            "trend_day": {"samples": 50, "label_rate": 0.60}
          },
          "backtest_summary": {
            "trades": 12,
            "trades_per_prediction_day": 4.2,
            "net_pnl": 42.0,
            "total_return": 0.0042,
            "sharpe": 1.7,
            "profit_factor": 1.4,
            "max_drawdown": 0.02,
            "hit_rate": 0.58
          },
          "folds": [
            {
              "fold_id": 0,
              "train_samples": 80,
              "calibration_samples": 20,
              "test_samples": 23,
              "traded_signals": 12,
              "net_pnl": 42.0,
              "trade_hit_rate": 0.58,
              "average_trade_return": 0.002,
              "brier_score": 0.1,
              "log_loss": 0.2,
              "selected_threshold": 0.55,
              "selected_threshold_source": "adaptive",
              "fitted_models": ["xgboost"],
              "skipped_models": {},
              "selected_model_params": {"xgboost": {"max_depth": 4}},
              "model_diagnostics": {
                "xgboost": {
                  "validation_weight": 1.0,
                  "validation_utility": 42.0,
                  "validation_brier": 0.1
                }
              }
            }
          ],
          "trades": [
            {
              "candidate_type": "active_pullback_reclaim",
              "side": "BUY",
              "outcome_type": "upper",
              "pnl": 42.0
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    write_research_report([parse_signal_audit_artifact(artifact)], report)

    text = report.read_text(encoding="utf-8")
    assert '"tpd_ge_4"' in text
    assert '"ok_results": 1' in text
    assert '"selected_param_counts"' in text
    assert '"trade_by_candidate_type"' in text
    assert '"top_foundation_kronos"' in text


def test_make_cost_stress_experiments_uses_positive_champions(tmp_path: Path) -> None:
    experiments = build_experiment_matrix(
        include_15m=True,
        include_5m=False,
        include_1m=False,
        models="logistic,histgb",
    )
    source = experiments[0]
    artifact = tmp_path / f"{source.name}.json"
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

    stressed = make_cost_stress_experiments([parse_signal_audit_artifact(artifact)], experiments, top_n=1)

    assert len(stressed) == 2
    assert stressed[0].name.endswith("_stress_cost")
    assert stressed[0].assumed_spread_bps >= source.assumed_spread_bps
    assert stressed[1].tier_rate >= source.tier_rate


def test_make_foundation_kronos_experiments_uses_positive_champions(tmp_path: Path) -> None:
    experiments = build_experiment_matrix(
        include_15m=True,
        include_5m=False,
        include_1m=True,
        models="logistic,histgb",
    )
    source = next(experiment for experiment in experiments if experiment.interval == "1m")
    artifact = tmp_path / f"{source.name}.json"
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

    foundation = make_foundation_kronos_experiments(
        [parse_signal_audit_artifact(artifact)],
        experiments,
        top_n=1,
        sample_windows=(512,),
        kronos_configs=((128, 16),),
    )

    assert len(foundation) == 1
    champion = foundation[0]
    assert champion.interval == "1m"
    assert champion.models == FOUNDATION_MODELS
    assert champion.kronos_features is True
    assert champion.kronos_mode == "proxy"
    assert champion.kronos_lookback_bars == 128
    assert champion.kronos_embedding_dims == 16
    assert champion.foundation_max_samples == 512
    assert champion.hpo_profile == "deep"


def test_foundation_signal_audit_command_includes_kronos_and_hpo_flags(tmp_path: Path) -> None:
    experiment = ResearchExperiment(
        name="foundation",
        interval="1m",
        years=3,
        context_interval="1h",
        candidate_mode="active",
        candidate_types="active_pullback_reclaim",
        max_holding_hours=2,
        net_profit_target=0.0025,
        net_stop_loss=0.002,
        minimum_gross_profit_bps=36,
        minimum_gross_stop_bps=18,
        selection_score="expected_utility",
        calibration_method="isotonic",
        stacker="weighted",
        models=FOUNDATION_MODELS,
        kronos_features=True,
        kronos_lookback_bars=128,
        kronos_embedding_dims=16,
        foundation_max_samples=2048,
        hpo_profile="deep",
    )

    command = build_signal_audit_command(
        experiment,
        artifact_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        python_executable="python",
    )

    assert FOUNDATION_MODELS in command
    assert "--kronos-features" in command
    assert command[command.index("--kronos-lookback-bars") + 1] == "128"
    assert command[command.index("--kronos-embedding-dims") + 1] == "16"
    assert command[command.index("--foundation-max-samples") + 1] == "2048"
    assert command[command.index("--hpo-profile") + 1] == "deep"


def test_foundation_model_stack_excludes_failed_optional_smoke_models() -> None:
    smoke_stdout = json.dumps(
        [
            {"model_name": "tabicl", "ok": True},
            {"model_name": "tabpfn", "ok": False, "detail": "timeout"},
        ]
    )

    stack = foundation_model_stack_from_smoke(smoke_stdout, include_catboost=False)

    assert stack == "lightgbm,xgboost,tabicl"


def test_prefetch_requests_include_primary_context_and_1m() -> None:
    experiments = build_experiment_matrix(models="logistic,histgb", years_15m=7, years_5m=5, years_1m=3)

    requests = prefetch_requests_for_experiments(experiments)
    triples = {(request.symbol, request.interval, request.role) for request in requests}

    assert ("BTCUSDT", "1m", "primary") in triples
    assert ("BTCUSDT", "5m", "primary") in triples
    assert ("BTCUSDT", "15m", "primary") in triples
    assert ("ETHUSDT", "1h", "context") in triples
    assert ("SOLUSDT", "1h", "context") in triples
    assert max(request.years for request in requests if request.role == "context") == 7


def test_prefetch_manifest_records_optional_context_failures(monkeypatch, tmp_path: Path) -> None:
    experiments = [
        ResearchExperiment(
            name="smoke",
            interval="1m",
            years=1,
            context_interval="1h",
            candidate_mode="active",
            candidate_types="",
            max_holding_hours=2,
            net_profit_target=0.0025,
            net_stop_loss=0.002,
            minimum_gross_profit_bps=36,
            minimum_gross_stop_bps=18,
            selection_score="expected_utility",
            calibration_method="isotonic",
            stacker="weighted",
        )
    ]

    def fake_fetch(*, symbol, interval, start, end, cache_dir, client=None):
        if symbol == "BTCUSDT":
            return [object()]
        raise RuntimeError("optional context unavailable")

    monkeypatch.setattr("zeroalpha.models.colab_research.fetch_klines_archive_range", fake_fetch)
    manifest_path = tmp_path / "cache_manifest.json"

    payload = prefetch_binance_cache(
        experiments,
        cache_dir=tmp_path / "cache",
        manifest_path=manifest_path,
        context_symbols="SOLUSDT",
    )

    assert manifest_path.exists()
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["failed_optional"] == 1


def test_colab_notebook_dependency_cell_does_not_reinstall_core_stack() -> None:
    notebook = json.loads(Path("src/zeroalpha/models/train.ipynb").read_text(encoding="utf-8"))
    dependency_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "OPTIONAL_MODEL_PACKAGES" in "".join(cell.get("source", []))
    )

    assert "--no-deps" in dependency_cell
    assert "OPTIONAL_MODEL_PACKAGES" in dependency_cell
    assert "numpy==" not in dependency_cell
    assert "pandas==" not in dependency_cell
    assert "scipy==" not in dependency_cell
    assert "os.kill" not in dependency_cell


def test_colab_notebook_bootstrap_refreshes_existing_checkout() -> None:
    notebook = json.loads(Path("src/zeroalpha/models/train.ipynb").read_text(encoding="utf-8"))
    bootstrap_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "refresh_existing_git_repo" in "".join(cell.get("source", []))
    )

    assert "/Users/ethan/" not in bootstrap_cell
    assert "IN_COLAB_CONTENT" in bootstrap_cell
    assert 'BOOTSTRAP_MODE == "git" and IN_COLAB_CONTENT' in bootstrap_cell
    assert "refresh_existing_git_repo" in bootstrap_cell
    assert '"fetch", "--depth", str(GIT_DEPTH), "origin", GIT_BRANCH' in bootstrap_cell
    assert '"reset", "--hard", "FETCH_HEAD"' in bootstrap_cell
    assert "PROJECT_COMMIT" in bootstrap_cell


def test_colab_notebook_pilot_is_clean_and_gap_safe() -> None:
    notebook = json.loads(Path("src/zeroalpha/models/train.ipynb").read_text(encoding="utf-8"))
    import_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "Build the H100 lower-timeframe HPO matrix" in "".join(cell.get("source", []))
    )
    pilot_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "PILOT_RESUME = False" in "".join(cell.get("source", []))
    )

    assert "build_signal_audit_command" in import_cell
    assert "experiment_artifact_path" in import_cell
    assert "PILOT_RESUME = False" in pilot_cell
    assert '"--allow-data-gaps" in pilot_command' in pilot_cell
    assert "artifact_path.unlink(missing_ok=True)" in pilot_cell
    assert "log_path.unlink(missing_ok=True)" in pilot_cell
    assert "resume=PILOT_RESUME" in pilot_cell


def test_colab_notebook_has_prefetch_and_foundation_cells() -> None:
    notebook = json.loads(Path("src/zeroalpha/models/train.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    full_text = "\n".join(sources)

    assert "prefetch_binance_cache" in full_text
    assert "cache_manifest.json" in full_text
    assert "Final champion stage: foundation models + Kronos" in full_text
    assert "make_foundation_kronos_experiments" in full_text
    assert "FOUNDATION_TOP_N = 12" in full_text
    assert "FOUNDATION_PACKAGE_SPECS" in full_text
    assert "foundation_summary.csv" in full_text
    assert "foundation_report.json" in full_text
