from pathlib import Path
from dataclasses import replace

import pytest

from zeroalpha.config import AppConfig, ConfigError, ContractConfig, LabelConfig, RiskConfig, load_config
from zeroalpha.domain import RuntimeMode


def test_load_example_config() -> None:
    cfg = load_config(Path("configs/paper.example.toml"))
    assert cfg.runtime.mode == RuntimeMode.PAPER
    assert cfg.broker.port == 4002
    assert cfg.contract.symbol == "BTC"
    assert cfg.model.minimum_expected_value == 0.0


def test_futures_research_can_validate_multiple_open_positions() -> None:
    cfg = AppConfig(
        contract=ContractConfig(instrument_model="futures"),
        risk=RiskConfig(max_open_positions=4),
    )

    cfg.validate()


def test_spot_crypto_allows_multiple_inventory_lots() -> None:
    cfg = replace(AppConfig(), risk=RiskConfig(max_open_positions=2))

    cfg.validate()


def test_second_level_holding_period_validates_when_set() -> None:
    AppConfig(labels=LabelConfig(max_holding_seconds=1.0)).validate()

    with pytest.raises(ConfigError, match="max_holding_seconds"):
        AppConfig(labels=LabelConfig(max_holding_seconds=0.0)).validate()


def test_zero_consecutive_loss_limit_disables_cooldown_validation() -> None:
    AppConfig(risk=RiskConfig(consecutive_loss_limit=0, cooldown_hours_after_stopouts=0)).validate()

    with pytest.raises(ConfigError, match="cooldown settings"):
        AppConfig(risk=RiskConfig(consecutive_loss_limit=-1)).validate()
