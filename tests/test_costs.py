from zeroalpha.costs import CommissionModel, SlippageModel, estimate_round_trip_cost


def test_commission_minimum_and_rate() -> None:
    model = CommissionModel()
    assert model.commission(10) == 0.10
    assert model.commission(500) == 1.75
    assert round(model.commission(1_000), 2) == 1.80
    assert round(model.round_trip_commission_bps(10_000), 1) == 36.0


def test_round_trip_cost_components() -> None:
    cost = estimate_round_trip_cost(
        10_000,
        spread_bps=10,
        commission_model=CommissionModel(),
        slippage_model=SlippageModel(base_slippage_bps=5, spread_multiplier=0.5),
        safety_margin_bps=10,
    )
    assert cost.commission_bps == 36
    assert cost.spread_bps == 10
    assert cost.slippage_bps == 20
    assert cost.total_bps == 76


def test_futures_per_contract_fee_converts_to_bps() -> None:
    cost = estimate_round_trip_cost(
        10_000,
        spread_bps=0.5,
        commission_model=CommissionModel(),
        slippage_model=SlippageModel(base_slippage_bps=0.25, spread_multiplier=0.5),
        safety_margin_bps=1.0,
        futures_fee_per_contract=2.02,
        futures_contract_multiplier=0.1,
        reference_price=100_000,
    )

    assert round(cost.commission_bps, 2) == 4.04
    assert round(cost.total_bps, 2) == 6.54
