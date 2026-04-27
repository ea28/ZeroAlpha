from zeroalpha.execution.decision import ExpectedValueInputs, gate_trade


def test_ev_gate_approves_positive_cost_adjusted_trade() -> None:
    decision = gate_trade(
        ExpectedValueInputs(
            calibrated_probability=0.7,
            expected_win=0.03,
            expected_loss=0.015,
            total_cost=0.006,
        ),
        minimum_probability=0.6,
        minimum_expected_value=0.0075,
    )
    assert decision.should_trade
    assert round(decision.expected_value, 4) == 0.0105


def test_ev_gate_rejects_low_probability() -> None:
    decision = gate_trade(
        ExpectedValueInputs(
            calibrated_probability=0.55,
            expected_win=0.05,
            expected_loss=0.01,
            total_cost=0.004,
        ),
        minimum_probability=0.6,
        minimum_expected_value=0.0075,
    )
    assert not decision.should_trade
    assert decision.reason == "probability_below_threshold"
