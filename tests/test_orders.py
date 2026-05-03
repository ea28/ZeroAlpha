from zeroalpha.domain import OrderType, Side
from zeroalpha.execution.orders import CryptoOrderFactory


def test_bracket_entry_attaches_take_profit_and_stop_loss_for_long() -> None:
    parent, take_profit, stop_loss = CryptoOrderFactory.bracket_entry(
        "event-1",
        "BTC",
        0.1,
        100.0,
        105.0,
        97.0,
        side=Side.BUY,
    )

    assert parent.side == Side.BUY
    assert parent.order_type == OrderType.LMT
    assert parent.transmit is False
    assert take_profit.side == Side.SELL
    assert take_profit.limit_price == 105.0
    assert take_profit.parent_internal_order_id == parent.internal_order_id
    assert take_profit.transmit is False
    assert stop_loss.side == Side.SELL
    assert stop_loss.order_type == OrderType.STP
    assert stop_loss.aux_price == 97.0
    assert stop_loss.parent_internal_order_id == parent.internal_order_id
    assert stop_loss.transmit is True


def test_bracket_entry_reverses_exits_for_short() -> None:
    parent, take_profit, stop_loss = CryptoOrderFactory.bracket_entry(
        "event-1",
        "BTC-PERP",
        0.1,
        100.0,
        95.0,
        103.0,
        side=Side.SELL,
    )

    assert parent.side == Side.SELL
    assert take_profit.side == Side.BUY
    assert take_profit.limit_price == 95.0
    assert stop_loss.side == Side.BUY
    assert stop_loss.order_type == OrderType.STP
    assert stop_loss.aux_price == 103.0


def test_bracket_entry_can_use_native_stop_limit_loss() -> None:
    _, _, stop_loss = CryptoOrderFactory.bracket_entry(
        "event-1",
        "BTC",
        0.1,
        100.0,
        105.0,
        97.0,
        side=Side.BUY,
        stop_loss_limit_price=96.5,
    )

    assert stop_loss.side == Side.SELL
    assert stop_loss.order_type == OrderType.STP_LMT
    assert stop_loss.aux_price == 97.0
    assert stop_loss.limit_price == 96.5
    assert stop_loss.transmit is True
