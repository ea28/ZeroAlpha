import math
from types import SimpleNamespace

import pytest

from zeroalpha.broker.ibkr import IBKRBroker, _bar_size_delta, _historical_bar_volume
from zeroalpha.config import AppConfig, BrokerConfig, ConfigError, RiskConfig
from zeroalpha.execution.orders import CryptoOrderFactory


def test_historical_bar_volume_clamps_ibkr_bid_ask_sentinels() -> None:
    assert _historical_bar_volume(-1) == 0.0
    assert _historical_bar_volume(None) == 0.0
    assert _historical_bar_volume(math.nan) == 0.0
    assert _historical_bar_volume(12.5) == 12.5


def test_ibkr_bar_size_delta_parses_completed_bar_sizes() -> None:
    assert _bar_size_delta("1 min").total_seconds() == 60
    assert _bar_size_delta("5 secs").total_seconds() == 5
    assert _bar_size_delta("1 hour").total_seconds() == 3600


def test_ibkr_broker_rejects_order_intent_above_configured_notional_cap() -> None:
    broker = IBKRBroker(AppConfig(risk=RiskConfig(paper_max_notional=100.0)))
    intent = CryptoOrderFactory.limit_entry(
        event_id="test",
        symbol="BTC/USD",
        quantity=2.0,
        limit_price=100.0,
    )

    with pytest.raises(ConfigError, match="max notional"):
        broker._validate_order_intent_risk(intent)


def test_ibkr_broker_requires_reference_price_for_quantity_market_order() -> None:
    broker = IBKRBroker(AppConfig())
    intent = CryptoOrderFactory.urgent_market_exit(
        symbol="BTC/USD",
        quantity=0.01,
        reason="test",
    )

    with pytest.raises(ConfigError, match="reference price"):
        broker._validate_order_intent_risk(intent)


def test_ibkr_broker_caps_market_sell_to_current_position() -> None:
    broker = IBKRBroker(AppConfig())
    intent = CryptoOrderFactory.urgent_market_exit(
        symbol="BTC/USD",
        quantity=0.02,
        reason="test",
    )

    with pytest.raises(ConfigError, match="current BTC position"):
        broker._validate_order_intent_risk(intent, reference_price=50_000.0)
    with pytest.raises(ConfigError, match="exceeds current BTC position"):
        broker._validate_order_intent_risk(
            intent,
            reference_price=50_000.0,
            current_position_quantity=0.01,
        )

    broker._validate_order_intent_risk(
        intent,
        reference_price=50_000.0,
        current_position_quantity=0.02,
    )


def test_ibkr_broker_requires_explicit_account_before_mutating_orders() -> None:
    broker = IBKRBroker(AppConfig(broker=BrokerConfig(account="")))
    broker._ib = SimpleNamespace(
        managedAccounts=lambda: ["DU123456"],
        isConnected=lambda: True,
    )
    broker._read_only = False

    with pytest.raises(ConfigError, match="broker.account is required"):
        broker._assert_can_mutate_orders()


def test_ibkr_broker_validates_configured_account_is_managed() -> None:
    broker = IBKRBroker(AppConfig(broker=BrokerConfig(account="DU999999")))
    broker._ib = SimpleNamespace(
        managedAccounts=lambda: ["DU123456"],
        isConnected=lambda: True,
    )
    broker._read_only = False

    with pytest.raises(ConfigError, match="not in TWS managedAccounts"):
        broker._assert_can_mutate_orders()


def test_ibkr_broker_cancel_is_guarded_by_read_only_runtime() -> None:
    broker = IBKRBroker(AppConfig(broker=BrokerConfig(account="DU123456")))
    broker._ib = SimpleNamespace(
        managedAccounts=lambda: ["DU123456"],
        isConnected=lambda: True,
        cancelOrder=lambda order: order,
    )
    broker._read_only = True

    with pytest.raises(ConfigError, match="read-only"):
        broker.cancel_trade(SimpleNamespace(order=SimpleNamespace(orderId=1)))


def test_ibkr_broker_rejects_native_stop_orders_for_crypto() -> None:
    broker = IBKRBroker(AppConfig())
    intent = CryptoOrderFactory.stop_loss_exit(
        event_id="test",
        symbol="BTC/USD",
        quantity=0.1,
        stop_price=95_000.0,
    )

    with pytest.raises(ConfigError, match="synthetic stops"):
        broker._order_from_intent(intent)


def test_ibkr_order_sets_configured_account_when_available() -> None:
    broker = IBKRBroker(AppConfig(broker=BrokerConfig(account="DU123456")))
    intent = CryptoOrderFactory.limit_entry(
        event_id="test",
        symbol="BTC/USD",
        quantity=0.01,
        limit_price=50_000.0,
    )

    order = broker._order_from_intent(intent)

    assert order.account == "DU123456"
