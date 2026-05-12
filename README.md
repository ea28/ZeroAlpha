# ZeroAlpha

ZeroAlpha is an IBKR-aware BTC/USD machine-learning trading system. It covers
data ingestion, feature engineering, walk-forward model training, production
artifact creation, broker execution, sizing, synthetic exits, and runtime risk
controls.

This repository is experimental financial software, not financial advice. Use
real money only after you have validated your own data feed, model artifact,
account permissions, execution behavior, commission assumptions, and risk caps.

## Strategy

ZeroAlpha is a two-stage long/flat BTC/USD spot system. It trades spot crypto
through IBKR, uses no leverage in the production path, and treats futures,
prediction-market, cross-asset, and exchange data as causal feature sources
rather than execution instruments.

The production trade lifecycle is:

1. Build completed BTC/USD bars and point-in-time context features.
2. Generate possible long entries from active, dense, or rule-based candidate
   families such as breakouts, pullback reclaims, range mean reversion, and
   liquidity reversals.
3. Label each candidate with a triple-barrier path outcome: profit target, stop
   loss, or vertical time barrier.
4. Train calibrated meta-label models to answer a narrower question than
   direction alone: "is this candidate worth trading after costs, expected
   payoff, downside, and capacity limits?"
5. Select live trades only when probability, expected value, selection score,
   spread, quote freshness, open-position capacity, and risk guards all pass.
6. Enter with broker-compatible market-buy cash orders and exit with synthetic
   stop/profit/timed exits derived from the artifact's label geometry.

The strategy intentionally separates signal discovery from trade approval. A
candidate can be directionally plausible and still rejected because its expected
payoff does not clear commission, spread, slippage, safety margin, or current
capacity. The artifact stores the candidate policy, feature contract, selected
thresholds, horizon policy, execution assumptions, and sizing policy so the
broker runner can reject incompatible artifacts before placing orders.

When volatility increases, the system should not mechanically increase trade
count or notional. Higher realized volatility affects four controls:

- Adaptive horizons shorten because the expected time to hit a target move is
  lower when per-bar movement is larger.
- Gross profit and stop distances can widen when volatility multipliers or
  minimum gross barriers are enabled, so the trade has enough room to clear
  costs and avoid noise-level exits.
- Model features such as realized volatility, downside volatility, range
  expansion, spread, and regime state can lower the calibrated probability or
  selection score when high volatility is historically unfavorable.
- Sizing remains capped by score buckets, liquidity checks, account capital,
  max order notional, max open positions, and loss limits. If volatility rises
  but edge does not rise with it, the correct behavior is fewer or smaller
  trades, not forced quota filling.

For manual research, tune high-volatility behavior in this order: verify the
cost model, widen gross target/stop floors or volatility multipliers if the
gross barriers are inside normal noise, reduce target trade frequency when
selection quality degrades, then adjust sizing caps. Avoid lowering thresholds
just to preserve trade count during volatile regimes.

## Math

Returns are measured as simple close-to-close returns:

```text
r_t = close_t / close_{t-1} - 1
```

Recent per-bar volatility is the population standard deviation of recent
returns. For an event with a vertical holding horizon of `horizon_bars`, the
horizon-scaled volatility proxy is:

```text
sigma_bar = stdev(r_{t-L+1}, ..., r_t)
sigma_horizon = sigma_bar * sqrt(horizon_bars)
```

Round-trip cost is modeled as a fraction that includes commission, spread,
slippage, and safety margin:

```text
cost = round_trip_cost_bps / 10000
```

For a long candidate with entry price `P0`, the triple-barrier label uses net
targets but converts them into executable gross prices:

```text
upper_price = P0 * (1 + net_profit_target + cost)
lower_price = P0 * (1 + cost - net_stop_loss)
net_return = exit_price / P0 - 1 - cost
```

The label is positive only if the upper barrier is hit before the stop or
vertical barrier and the resulting net return clears `net_profit_target`. The
stop must be larger than round-trip cost, otherwise the lower barrier would not
represent real loss protection after fees.

Dynamic label geometry uses the largest applicable target and stop. All bps
inputs are converted to fractions:

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

This means a higher `sigma_horizon` can widen gross barriers when the
volatility multipliers are enabled. In the production preset, minimum gross
profit and stop floors are also used so tiny net targets do not become
untradeable after IBKR spot fees and spread assumptions.

Adaptive horizons estimate how long a setup should need to realize a configured
target move:

```text
movement_bps = max(per_bar_vol_bps, 0.5 * recent_intrabar_range_bps)
raw_seconds = target_move_bps / movement_bps * bar_interval_seconds * setup_multiplier
horizon_seconds = clipped_and_rounded(raw_seconds, min_seconds, max_seconds, granularity)
```

When volatility or intrabar range rises, `movement_bps` rises and the vertical
barrier gets shorter. Breakout and momentum families can also use shorter setup
multipliers, while mean-reversion or range families can allow more time. The
goal is to avoid holding a fast-moving setup for a stale fixed horizon.

Expected value starts with the calibrated probability `p`:

```text
label_ev = p * net_profit_target - (1 - p) * net_stop_loss
```

When empirical payoff EV is enabled, candidate-family calibration replaces the
generic target/stop payoff with observed average positive and negative returns:

```text
empirical_ev = p * average_label_one_return + (1 - p) * average_label_zero_return
```

Selection scores can use probability alone, expected value, predicted return,
or utility-style combinations. The production preset uses `expected_utility`:

```text
expected_utility = predicted_return + 0.25 * expected_value - 0.25 * predicted_downside
```

The backtester also computes a capital-efficiency score for ranking and
diagnostics:

```text
risk = max(predicted_mae, predicted_downside, 1e-6)
hours = max(predicted_time_to_exit_seconds / 3600, 1 / 3600)
capital_efficiency =
  (predicted_return + 0.25 * expected_value + 0.10 * predicted_mfe)
  / (risk * sqrt(hours))
  - predicted_early_adverse_probability
```

These formulas make volatility actionable: larger downside, larger adverse
excursion, longer time-to-exit, or higher early-adverse probability lowers the
ranking unless predicted return and expected value improve enough to compensate.

## ML Models

The ML layer is a meta-labeling stack, not a standalone price forecaster. The
models receive candidate rows that already represent plausible setups, then
estimate whether each setup is worth trading. Training uses walk-forward folds
so each test slice is later than its training/calibration data.

Implemented model families:

- Logistic regression: calibrated linear baseline for sanity checks, feature
  direction checks, and fast smoke tests.
- HistGradientBoosting: scikit-learn histogram boosting for nonlinear tabular
  structure without external dependencies.
- RandomForest and ExtraTrees: bagged tree ensembles that tolerate noisy
  features and nonlinear interactions; ExtraTrees is often a strong low-touch
  baseline.
- LightGBM, XGBoost, and CatBoost: high-capacity gradient-boosted tree families
  with different regularization, categorical handling, and bias/variance
  behavior.
- TabICL and TabPFN: optional foundation-style tabular classifiers, skipped
  when unavailable or unstable in the local environment.
- Kronos features: optional time-series embeddings used as additional features,
  not as the sole trading decision engine.

The ensemble can average models, choose the best fold-local model, learn a
logistic stacker, or weight models by fold performance. Probabilities are
calibrated with sigmoid or isotonic calibration before thresholds are selected,
because raw classifier scores are not reliable enough for EV gating.

The model report includes more than classification accuracy. It tracks net PnL,
Sharpe, Calmar, max drawdown, hit rate, profit factor, rejected signals,
commission, spread cost, slippage, safety margin, fold-level calibration,
candidate-family performance, and feature diagnostics. Optional auxiliary heads
estimate return, downside, MAE, MFE, time-to-exit, and early adverse movement so
selection can penalize high-risk or capital-inefficient approvals.

HPO runs inside each walk-forward fold, not on the final test data. The main
promotion question is whether a policy remains stable across folds, regimes,
candidate families, feature groups, and execution assumptions. A model that wins
only because of one day, one setup family, one feature source, or one unusually
large trade should not be promoted.

## Quick Start

Install with the project extras you need:

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]" -c constraints-py314.txt
```

`constraints-py314.txt` is a pinned verification constraints file for CPython
3.14. It is optional, but useful when you want the same dependency versions used
by local validation.

Create an untracked local config from `configs/live.example.toml`, then set the
IBKR host, port, account, notional caps, and loss limits for the account you
intend to use. Do not commit account numbers or secrets.

Basic checks:

```bash
.venv/bin/zeroalpha config check --config configs/live.local.toml
.venv/bin/zeroalpha data health-check
.venv/bin/zeroalpha model smoke --models logistic,histgb,extratrees,lightgbm,catboost,xgboost
```

## Data

Research and execution can use:

- Binance spot archives for BTCUSDT history and cross-asset context.
- Binance USD-M futures for open interest, funding, taker flow, and basis.
- Coinbase and Kraken public data for reference candles and source health.
- Polymarket and Kalshi BTC directional markets for causal prediction-market
  features.
- IBKR Gateway/TWS for account state, quotes, executions, commissions,
  historical bars, market depth, and tick-by-tick streams where permissions
  allow.

All features are built point-in-time. OHLCV bars are normalized to completed bar
close time. External feeds can be latency-buffered with
`--external-feature-latency-seconds`. The strict 1-second path requires
tick-backed IBKR label and execution bars plus a live tick-by-tick stream before
scoring.

## Features

Feature families include:

| Family | Examples |
| --- | --- |
| Momentum and trend | Rolling returns, EMA/SMA gaps, breakout slope |
| Technical state | RSI, MACD-style EMA spread, Bollinger z, ATR/range shock |
| Volume and order flow | Quote volume, VWAP distance, volume surprise, taker imbalance |
| Microstructure | Spread, microprice, top-book imbalance, depth imbalance |
| Cross-asset context | ETH/SOL relative strength, ETHBTC, spot/futures basis |
| Futures context | Open interest, funding, taker flow, basis, CME MBT/MET context |
| Prediction markets | Directional mids, deltas, liquidity weights, term structure, provider disagreement |
| IBKR runtime | Bid/ask, quote age, spread, account snapshots, position and PnL state |

Feature selection is available at two levels. Use
`--feature-include-groups`/`--feature-exclude-groups` for family-level ablation,
and `--feature-exclude-patterns` for exact pattern pruning. Reports can include
fold-local permutation importance, SHAP importance for supported models, and
native tree/linear importances.

## Hyperparameter Tuning

`--hpo` runs tuning inside each walk-forward fold so the test slice remains out
of sample. Profiles control search breadth:

- `standard`: balanced search for normal development.
- `wide`: broader model-family exploration.
- `deep`: more intensive tuning for final candidates.
- `quota` and `capacity`: search spaces biased toward target-frequency and
  capital-efficiency experiments.

Optimization can target Sharpe, net PnL, or Calmar. Reports also track trade
count, drawdown, hit rate, profit factor, commission, slippage, spread cost,
model diagnostics, rejected signals, and candidate-family performance. Avoid
selecting a model on a single headline metric; require consistency across folds,
regimes, feature groups, and execution assumptions.

## Risk And Sizing

Risk controls are layered:

- Configured account equity, max notional, max open positions, and minimum
  fee-efficient notional.
- Daily, weekly, rolling-drawdown, consecutive-loss, and cooldown stops.
- Runtime max-loss guard for broker execution.
- Kill switch file checked inside the trade loop.
- Quote freshness and spread checks.
- Artifact contract checks that reject incompatible model/execution policies.
- Feature-contract checks that reject live samples with missing trained feature
  coverage.

Trade size positioning supports fixed, confidence-scaled, score-bucket, and
liquidity-score-bucket sizing. The production preset trains score-bucket
artifacts with base, mid, and high notionals tied to selection score. At
runtime, order size is capped by available capital, configured max order
notional, account-mode notional caps, open-position capacity, and the artifact
sizing policy.

## Execution

`broker trade-run` connects to IBKR, qualifies BTC/USD crypto, loads the
production artifact, verifies the data contract, builds fresh completed-bar
features, scores candidates, submits market buy cash orders for approved
signals, monitors synthetic exits, refreshes account state, writes runtime JSONL
logs, and flattens runner-opened positions on exit when configured.

For real-money operation, use a local live config with the live IBKR port,
explicit account, `enable_live_trading = true`, and
`live_confirmation = "ZEROALPHA_LIVE"`. The broker command also requires the
live confirmation string:

```bash
--confirm ZEROALPHA_LIVE_TRADE_RUN
```

Sandbox validation should use the matching non-live IBKR session and the
corresponding confirmation string:

```bash
--confirm IBKR_PAPER_TRADE_RUN
```

## Verification

Before promoting a model or running against an account:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
.venv/bin/zeroalpha data health-check
.venv/bin/zeroalpha model smoke --models logistic,histgb,extratrees,lightgbm,catboost,xgboost
git diff --check
```

For broker verification, run read-only connectivity checks first:

```bash
.venv/bin/zeroalpha broker smoke \
  --config configs/live.local.toml \
  --account YOUR_IBKR_ACCOUNT \
  --read-only
```

## Repository Layout

```text
configs/                  local config templates
src/zeroalpha/backtest/   candidate and model backtests
src/zeroalpha/broker/     IBKR Gateway/TWS adapter and quote recorder
src/zeroalpha/candidates/ event generation
src/zeroalpha/data/       data clients and quality checks
src/zeroalpha/execution/  order intents and synthetic stops
src/zeroalpha/features/   engineered features
src/zeroalpha/labels/     triple-barrier labels
src/zeroalpha/models/     dataset, ensemble, artifact, interpretability, HPO
src/zeroalpha/monitoring/ runtime event streaming
src/zeroalpha/risk/       risk primitives
tests/                    unit and integration-style tests
```

Generated `data/`, `artifacts/`, logs, caches, local configs, and virtual
environments should stay out of git.

## CLI

The full CLI remains available for research, while the common production
workflow has presets under `zeroalpha easy`.

Backtest the production preset:

```bash
.venv/bin/zeroalpha easy backtest \
  --config configs/live.local.toml \
  --years 3 \
  --capital-usd 10000 \
  --high-notional-usd 5000
```

Train a trade-run compatible model artifact:

```bash
.venv/bin/zeroalpha easy train \
  --config configs/live.local.toml \
  --artifact artifacts/models/zeroalpha_spot_btc_prod.joblib \
  --years 3 \
  --capital-usd 10000
```

Run the broker execution loop:

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

The preset train/backtest commands expand to the existing advanced commands
with market-entry execution modeling, score-bucket sizing, online
capacity-aware selection, adaptive horizons, candidate-type thresholds,
empirical payoff EV, walk-forward HPO, and permutation importance enabled. Add
`--no-hpo` for faster iteration and `--shap` for SHAP diagnostics.

Use the advanced commands when you need custom data, labels, model families,
feature ablations, or execution assumptions:

```bash
.venv/bin/zeroalpha backtest ml --help
.venv/bin/zeroalpha model train-meta --help
.venv/bin/zeroalpha model signal-audit --help
.venv/bin/zeroalpha broker trade-run --help
```

Strict 1-second research remains available through
`zeroalpha.models.ibkr_experiments` and `--strict-live-valid-1s`. A promoted run
must use tick-backed 1-second label/execution replay, online capacity-aware
selection, serialized strategy contracts, and sufficient multi-day coverage.
