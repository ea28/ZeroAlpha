"""TOML-backed configuration and safety validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from zeroalpha.domain import RuntimeMode


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: RuntimeMode = RuntimeMode.PAPER
    timezone: str = "UTC"
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    kill_switch_file: Path = Path(".zeroalpha/kill_switch.enabled")
    enable_live_trading: bool = False
    live_confirmation: str = ""


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 701
    account: str = ""
    read_only: bool = True
    allow_custom_port: bool = False
    market_data_type: str = "live"
    crypto_exchanges: tuple[str, ...] = ("PAXOS", "ZEROHASH")


@dataclass(frozen=True, slots=True)
class ContractConfig:
    symbol: str = "BTC"
    security_type: str = "CRYPTO"
    currency: str = "USD"
    instrument_model: str = "spot_crypto"


@dataclass(frozen=True, slots=True)
class CostConfig:
    tier_rate: float = 0.0018
    minimum_commission: float = 1.75
    maximum_commission_rate: float = 0.01
    base_slippage_bps: float = 5.0
    safety_margin_bps: float = 10.0
    futures_fee_per_contract: float = 0.0
    futures_contract_multiplier: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    entry_order_type: str = "LMT"
    normal_exit_order_type: str = "LMT"
    urgent_exit_order_type: str = "MKT"
    entry_timeout_seconds: int = 300
    exit_timeout_seconds: int = 300
    max_cancel_replace_per_signal: int = 3
    simulated_latency_seconds: float = 0.0
    limit_trade_through_bps: float = 0.0
    limit_fill_probability: float = 1.0
    partial_fill_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class RiskConfig:
    account_equity: float = 10_000.0
    risk_per_trade: float = 0.0035
    paper_max_notional: float = 10_000.0
    live_max_notional: float = 5_000.0
    daily_loss_stop: float = 0.01
    weekly_loss_stop: float = 0.03
    rolling_drawdown_stop: float = 0.08
    max_open_positions: int = 1
    minimum_fee_efficient_notional: float = 1_000.0
    max_spread_bps: float = 75.0
    max_quote_age_ms: int = 5_000
    consecutive_loss_limit: int = 3
    cooldown_hours_after_stopouts: int = 24


@dataclass(frozen=True, slots=True)
class LabelConfig:
    bar_size: str = "1h"
    max_holding_hours: int = 72
    max_holding_seconds: float | None = None
    net_profit_target: float = 0.02
    net_stop_loss: float = 0.02
    conservative_same_bar: bool = True
    volatility_lookback_bars: int = 240
    profit_volatility_multiplier: float = 0.0
    stop_volatility_multiplier: float = 0.0
    minimum_gross_profit_bps: float = 0.0
    minimum_gross_stop_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    minimum_probability: float = 0.60
    minimum_expected_value: float = 0.0
    calibration_method: str = "sigmoid"
    feature_set_version: str = "v1"
    label_version: str = "v1"


@dataclass(frozen=True, slots=True)
class KronosConfig:
    enabled: bool = False
    mode: str = "proxy"
    lookback_bars: int = 512
    embedding_dims: int = 12
    official_model_name: str = "NeoQuasar/Kronos-small"
    official_tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = "cpu"
    hidden_pool_tail: int = 64


@dataclass(frozen=True, slots=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)

    def validate(self) -> None:
        if self.runtime.mode == RuntimeMode.LIVE:
            if not self.runtime.enable_live_trading:
                raise ConfigError("live mode requires enable_live_trading = true")
            if self.runtime.live_confirmation != "ZEROALPHA_LIVE":
                raise ConfigError('live mode requires live_confirmation = "ZEROALPHA_LIVE"')
        expected_ports = {
            RuntimeMode.PAPER: {4002, 7497},
            RuntimeMode.LIVE: {4001, 7496},
        }
        if not self.broker.allow_custom_port and self.runtime.mode in expected_ports:
            if self.broker.port not in expected_ports[self.runtime.mode]:
                raise ConfigError(
                    f"{self.runtime.mode.value} mode should use one of "
                    f"{sorted(expected_ports[self.runtime.mode])}; got {self.broker.port}"
                )
        if self.contract.security_type != "CRYPTO":
            raise ConfigError("v1 only supports CRYPTO contracts")
        if self.contract.currency != "USD":
            raise ConfigError("v1 only supports USD crypto contracts")
        if self.contract.instrument_model not in {"spot_crypto", "futures"}:
            raise ConfigError("contract instrument_model must be spot_crypto or futures")
        if self.broker.market_data_type not in {"live", "frozen", "delayed", "delayed_frozen", "delayed-frozen"}:
            raise ConfigError("broker market_data_type must be live, frozen, delayed, or delayed_frozen")
        if not self.broker.crypto_exchanges:
            raise ConfigError("at least one crypto exchange candidate is required")
        if self.risk.max_open_positions < 1:
            raise ConfigError("risk max_open_positions must be positive")
        if self.cost.tier_rate <= 0 or self.cost.minimum_commission < 0:
            raise ConfigError("invalid cost settings")
        if self.cost.futures_fee_per_contract < 0 or self.cost.futures_contract_multiplier < 0:
            raise ConfigError("invalid futures cost settings")
        if (self.cost.futures_fee_per_contract > 0) != (self.cost.futures_contract_multiplier > 0):
            raise ConfigError(
                "futures_fee_per_contract and futures_contract_multiplier must be set together"
            )
        if self.labels.net_profit_target <= 0 or self.labels.net_stop_loss <= 0:
            raise ConfigError("label net target and stop must be positive")
        if self.labels.max_holding_seconds is not None and self.labels.max_holding_seconds <= 0:
            raise ConfigError("label max_holding_seconds must be positive when set")
        if self.labels.max_holding_seconds is None and self.labels.max_holding_hours <= 0:
            raise ConfigError("label max_holding_hours must be positive")
        if self.labels.volatility_lookback_bars <= 1:
            raise ConfigError("label volatility_lookback_bars must be greater than 1")
        if min(
            self.labels.profit_volatility_multiplier,
            self.labels.stop_volatility_multiplier,
            self.labels.minimum_gross_profit_bps,
            self.labels.minimum_gross_stop_bps,
        ) < 0:
            raise ConfigError("dynamic label controls must be nonnegative")
        if self.risk.minimum_fee_efficient_notional <= 0:
            raise ConfigError("minimum fee-efficient notional must be positive")
        if self.risk.consecutive_loss_limit < 0 or self.risk.cooldown_hours_after_stopouts < 0:
            raise ConfigError("invalid cooldown settings")
        if self.execution.simulated_latency_seconds < 0:
            raise ConfigError("execution simulated_latency_seconds must be nonnegative")
        if self.execution.limit_trade_through_bps < 0:
            raise ConfigError("execution limit_trade_through_bps must be nonnegative")
        if not 0 <= self.execution.limit_fill_probability <= 1:
            raise ConfigError("execution limit_fill_probability must be in [0, 1]")
        if not 0 < self.execution.partial_fill_fraction <= 1:
            raise ConfigError("execution partial_fill_fraction must be in (0, 1]")
        if self.kronos.mode not in {"proxy", "auto", "official"}:
            raise ConfigError("kronos mode must be proxy, auto, or official")
        if self.kronos.lookback_bars < 16:
            raise ConfigError("kronos lookback_bars must be at least 16")
        if not 1 <= self.kronos.embedding_dims <= 64:
            raise ConfigError("kronos embedding_dims must be in [1, 64]")
        if self.kronos.hidden_pool_tail <= 0:
            raise ConfigError("kronos hidden_pool_tail must be positive")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"config section [{name}] must be a table")
    return raw


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    runtime_raw = _section(data, "runtime")
    broker_raw = _section(data, "broker")
    cfg = AppConfig(
        runtime=RuntimeConfig(
            **{
                **runtime_raw,
                "mode": RuntimeMode(runtime_raw.get("mode", RuntimeMode.PAPER)),
                "data_root": Path(runtime_raw.get("data_root", "data")),
                "artifact_root": Path(runtime_raw.get("artifact_root", "artifacts")),
                "kill_switch_file": Path(
                    runtime_raw.get("kill_switch_file", ".zeroalpha/kill_switch.enabled")
                ),
            }
        ),
        broker=BrokerConfig(
            **{
                **broker_raw,
                "crypto_exchanges": tuple(broker_raw.get("crypto_exchanges", ["PAXOS", "ZEROHASH"])),
            }
        ),
        contract=ContractConfig(**_section(data, "contract")),
        cost=CostConfig(**_section(data, "cost")),
        execution=ExecutionConfig(**_section(data, "execution")),
        risk=RiskConfig(**_section(data, "risk")),
        labels=LabelConfig(**_section(data, "labels")),
        model=ModelConfig(**_section(data, "model")),
        kronos=KronosConfig(**_section(data, "kronos")),
    )
    cfg.validate()
    return cfg
