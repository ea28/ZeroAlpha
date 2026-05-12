# Spot Crypto Strategy And Performance

This note describes the current ZeroAlpha BTC/USD spot strategy for IBKR
Gateway/TWS. The strategy is long/flat spot crypto; futures data is allowed as a
causal feature source, not as the execution instrument for this strategy.

This is experimental financial software, not financial advice. Any real-money
deployment needs account-specific validation of data, commissions, order fills,
position reconciliation, and risk caps.

## Current Strategy

The production path is a cost-aware meta-labeling strategy:

- Generate long BTC/USD candidate entries from completed bars.
- Score candidates with calibrated tabular model ensembles.
- Select trades online with open-position capacity accounting.
- Size orders from the artifact's sizing policy.
- Enter with broker-compatible market-buy cash orders.
- Exit with synthetic stop/profit/timed exits from the model's label geometry.

The baseline setup uses active candidate generation for normal production
research and dense generation for broader intraday studies. Candidate families
include breakout continuation, pullback reclaim, squeeze breakout, liquidity
reversal, range mean reversion, and dense bar-level setup families. The model is
not asked to forecast every bar; it is asked to approve or reject pre-defined
trade opportunities after costs and risk are included.

The production preset uses:

- Online target-frequency selection rather than same-day quota ranking.
- `capacity_release_mode=planned`, so a position slot remains reserved until
  the planned vertical barrier during selection.
- One open runner-managed spot position by default.
- IBKR spot fee model: 18 bps per side, `$1.75` minimum, 1% maximum.
- Market-entry backtest modeling for trade-run compatibility.
- Score-bucket sizing with base, mid, and high notionals.
- Adaptive holding horizons so each signal can use a volatility-scaled vertical
  barrier.
- Prediction-market, futures, and cross-asset features only when timestamped at
  or before the scoring timestamp with the configured latency buffer.

When volatility increases, the strategy should become more selective unless the
model also sees higher expected payoff. Adaptive horizons shorten because the
target move can be reached faster. Gross stops and targets can widen when
minimum gross barriers or volatility multipliers are enabled. Predicted
downside, MAE, spread, and early-adverse probability can lower the selection
score. Sizing remains capped by score buckets, liquidity gates, account capital,
and max-order limits, so high volatility without higher edge results in fewer
or smaller trades.

## Strategy Math And Volatility

For completed bars, simple returns are:

```text
r_t = close_t / close_{t-1} - 1
```

Recent volatility is the standard deviation of recent returns. For a candidate
with `horizon_bars` until its vertical barrier:

```text
sigma_bar = stdev(r_{t-L+1}, ..., r_t)
sigma_horizon = sigma_bar * sqrt(horizon_bars)
```

Round-trip cost is a fraction that includes commission, spread, slippage, and
safety margin:

```text
cost = round_trip_cost_bps / 10000
```

For a long spot candidate with entry price `P0`, label geometry is:

```text
upper_price = P0 * (1 + net_profit_target + cost)
lower_price = P0 * (1 + cost - net_stop_loss)
net_return = exit_price / P0 - 1 - cost
```

The positive label is assigned only if the upper barrier is hit first and the
net return clears the configured target. Same-bar upper/lower collisions are
treated conservatively by default.

Dynamic target and stop geometry use the largest applicable constraint:

```text
net_profit_target = max(
  base_net_profit_target,
  minimum_gross_profit - cost,
  profit_volatility_multiplier * sigma_horizon - cost
)

net_stop_loss = max(
  base_net_stop_loss,
  cost + minimum_gross_stop,
  cost + stop_volatility_multiplier * sigma_horizon
)
```

If volatility multipliers are zero, the system still uses base and minimum
gross geometry while volatility enters through features and adaptive horizons.
If multipliers are enabled, rising `sigma_horizon` widens gross targets and
stops so the strategy is not trying to scalp inside normal noise.

Adaptive horizons estimate the time required to reach a configured move:

```text
movement_bps = max(per_bar_vol_bps, 0.5 * recent_intrabar_range_bps)
raw_seconds = target_move_bps / movement_bps * bar_interval_seconds * setup_multiplier
horizon_seconds = clipped_and_rounded(raw_seconds, min_seconds, max_seconds, granularity)
```

As `movement_bps` increases, the vertical barrier gets shorter. Breakout and
momentum families can use shorter multipliers; mean-reversion and range setups
can be allowed more time. This keeps exit timing tied to market speed rather
than a stale fixed bar count.

Expected value starts from calibrated probability:

```text
label_ev = p * net_profit_target - (1 - p) * net_stop_loss
```

With empirical payoff EV, observed candidate-family payoffs replace the generic
target/stop payoff:

```text
empirical_ev = p * average_label_one_return + (1 - p) * average_label_zero_return
```

The production preset ranks approvals with expected utility:

```text
expected_utility = predicted_return + 0.25 * expected_value - 0.25 * predicted_downside
```

Diagnostics also compute capital efficiency:

```text
risk = max(predicted_mae, predicted_downside, 1e-6)
hours = max(predicted_time_to_exit_seconds / 3600, 1 / 3600)
capital_efficiency =
  (predicted_return + 0.25 * expected_value + 0.10 * predicted_mfe)
  / (risk * sqrt(hours))
  - predicted_early_adverse_probability
```

This is why volatility handling is mostly a selection problem: higher movement
only helps if predicted return and payoff improve more than predicted downside,
adverse excursion, spread cost, and time-at-risk.

## Model Stack

The production stack can include logistic regression, HistGradientBoosting,
RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost, and optional
TabICL/TabPFN models. Kronos is available as a feature source, not as the only
decision engine.

Logistic regression is the calibration and sanity-check baseline. Histogram
boosting and tree ensembles capture nonlinear tabular structure without making
the system depend on one external booster. LightGBM, XGBoost, and CatBoost are
the higher-capacity tabular models used when the walk-forward folds show stable
incremental value. Optional TabICL/TabPFN models are treated as opportunistic
specialists and are skipped when unavailable or unstable locally.

The ensemble can average models, select the best model, learn logistic stacking,
or weight models by fold performance. Probabilities are calibrated with sigmoid
or isotonic calibration before trading thresholds are selected. Calibration is
required because the decision layer uses probabilities in expected-value math,
not only in ranking.

Auxiliary estimates can add predicted return, predicted downside, MAE, MFE,
time-to-exit, and early adverse probability. These estimates feed selection
scores so a high-probability trade can still be rejected when its expected
payoff is small, downside is large, or capital would be tied up too long.

HPO is nested inside walk-forward folds. Optimize on net PnL, Sharpe, or Calmar,
then review trade count, drawdown, hit rate, profit factor, cost drag,
candidate-family concentration, fold concentration, and outlier dependence
before promotion.

## Data And Causality

All bars are normalized to completed close time before feature generation.
External data is treated as point-in-time:

- Binance spot archives for BTCUSDT and cross-asset context.
- Binance USD-M futures for open interest, funding, taker flow, and basis.
- Coinbase and Kraken reference feeds for source checks.
- Polymarket and Kalshi BTC directional contracts for prediction-market
  features.
- IBKR quotes, historical bars, market depth, tick-by-tick data, account state,
  executions, and commission reports.

Prediction-market snapshots use timestamped history only. Current liquidity,
open interest, or book totals are not backfilled into historical rows. IBKR
1-second experiments require tick-backed label and execution replay before
promotion.

## Feature Selection

The most important feature families to validate per run are:

- Momentum/trend and realized-volatility regime.
- Fee, target, stop, and horizon geometry.
- Volume, taker-flow, and microstructure.
- Cross-asset and futures context.
- Prediction-market residual edge, term structure, and provider disagreement.

Use grouped permutation importance, SHAP where supported, native importances,
and leave-family ablations to avoid depending on unstable feature groups.
Feature pruning should be evaluated fold-by-fold, not on the final aggregate
only.

## Risk And Sizing

Risk controls include configured notional caps, max open positions, minimum
fee-efficient notional, loss stops, cooldowns, quote freshness checks, spread
checks, runtime max-loss enforcement, kill-switch checks, and artifact contract
validation.

Sizing modes:

- `fixed`: same requested notional for every approved signal.
- `confidence`: scales size above the selected probability threshold.
- `score_bucket`: assigns base/mid/high notionals by model score.
- `liquidity_score_bucket`: also requires liquidity/spread quality before
  increasing size.

The easy preset trains score-bucket artifacts so weak approvals receive smaller
orders and stronger approvals can receive the configured high notional, capped
again by the broker runner's capital and max-order limits.

## Promotion Checklist

Before using an artifact against an account:

- The artifact was trained with `--entry-order-model market`.
- Walk-forward performance is not concentrated in one fold, day, setup family,
  or feature family.
- Commission, spread, slippage, and safety margin are included.
- Volatility behavior is credible: high-volatility trades show enough expected
  payoff to offset wider stops, shorter horizons, and higher adverse movement.
- The selected threshold and sizing policy are saved in the artifact manifest.
- Broker smoke checks pass against the intended account.
- Runtime event and state logs are written to JSONL.
- The max-loss guard, kill switch, flatten-on-exit behavior, and final
  position/PnL reconciliation are verified.

Rejected for production claims:

- Same-day quota ranking.
- `capacity_release_mode=actual` without explicit research gating.
- Native stop orders for IBKR spot crypto.
- Historical rows that use data unavailable at the scoring timestamp.
- Short-horizon strategies whose gross edge does not clear commission, spread,
  slippage, and safety margin.

## CLI

Use the easy presets for normal production research:

```bash
.venv/bin/zeroalpha easy backtest \
  --config configs/live.local.toml \
  --years 3 \
  --capital-usd 10000 \
  --high-notional-usd 5000

.venv/bin/zeroalpha easy train \
  --config configs/live.local.toml \
  --artifact artifacts/models/zeroalpha_spot_btc_prod.joblib \
  --years 3 \
  --capital-usd 10000
```

Run the broker loop after the artifact, account, and risk limits are validated:

```bash
.venv/bin/zeroalpha easy trade \
  --config configs/live.local.toml \
  --account YOUR_IBKR_ACCOUNT \
  --artifact artifacts/models/zeroalpha_spot_btc_prod.joblib \
  --capital-usd 5000 \
  --max-loss-usd 250 \
  --max-order-notional-usd 1000 \
  --confirm ZEROALPHA_LIVE_TRADE_RUN
```

Advanced research is still available through:

```bash
.venv/bin/zeroalpha backtest ml --help
.venv/bin/zeroalpha model train-meta --help
.venv/bin/zeroalpha broker trade-run --help
```
