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

The baseline setup uses:

- Active or dense candidate generation, depending on data granularity.
- Online target-frequency selection.
- `capacity_release_mode=planned`.
- One open runner-managed spot position by default.
- IBKR spot fee model: 18 bps per side, `$1.75` minimum, 1% maximum.
- Adaptive holding horizons so each signal can use a volatility-scaled vertical
  barrier.
- Prediction-market, futures, and cross-asset features only when timestamped at
  or before the scoring timestamp with the configured latency buffer.

## Reproduction

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

## Model Stack

The production stack can include logistic regression, HistGradientBoosting,
RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost, and optional
TabICL/TabPFN models. The ensemble can average models, select the best model,
learn logistic stacking, or weight models by fold performance. Probabilities are
calibrated with sigmoid or isotonic calibration before trading thresholds are
selected.

HPO is nested inside walk-forward folds. Optimize on net PnL, Sharpe, or Calmar,
then review trade count, drawdown, hit rate, profit factor, cost drag,
candidate-family concentration, and outlier dependence before promotion.

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
