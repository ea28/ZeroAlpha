from pathlib import Path

from zeroalpha.config import load_config
from zeroalpha.domain import RuntimeMode


def test_load_example_config() -> None:
    cfg = load_config(Path("configs/paper.example.toml"))
    assert cfg.runtime.mode == RuntimeMode.PAPER
    assert cfg.broker.port == 4002
    assert cfg.contract.symbol == "BTC"
