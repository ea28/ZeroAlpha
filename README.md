# ZeroAlpha

ZeroAlpha is an IBKR-aware BTC/USD machine-learning trading system. It covers
data ingestion, feature engineering, walk-forward model training, production
artifact creation, broker execution, sizing, synthetic exits, and runtime risk
controls.

This repository is experimental financial software, not financial advice. Use
real money only after you have validated your own data feed, model artifact,
account permissions, execution behavior, commission assumptions, and risk caps.

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

## Easier CLI

The full CLI remains available for research, but the common production workflow
now has presets under `zeroalpha easy`.

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

## Strategy

ZeroAlpha is a two-stage long/flat BTC/USD spot system:

1. Candidate generation finds possible long entries from completed bars.
2. Meta-labeling scores each candidate for whether it is worth trading after
   spread, slippage, commission, expected payoff, and risk constraints.

Candidate modes range from conservative rule setups to dense and active
intraday candidate generation. The production preset uses active candidate
generation, online target-frequency selection, one configured open position by
default, market-entry backtest modeling, and synthetic runtime exits. The
artifact stores the candidate policy, horizon policy, sizing policy, feature
contract, and selected threshold so execution can reject incompatible artifacts.

Labels use triple-barrier outcomes: a profit barrier, stop barrier, and vertical
time barrier. Adaptive horizons can shorten or lengthen the vertical barrier
using recent volatility and setup metadata. When `broker trade-run` uses
`--use-model-exit-geometry`, runtime synthetic stops and profit targets use the
artifact's gross label geometry rather than arbitrary fixed exits.

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

## Models

Implemented model families:

- Logistic regression: calibrated linear baseline that is fast, stable, and
  useful for sanity checks.
- HistGradientBoosting: scikit-learn histogram boosting for nonlinear tabular
  structure without external dependencies.
- RandomForest and ExtraTrees: bagged tree ensembles that handle nonlinear
  interactions and noisy features; ExtraTrees is often a strong low-maintenance
  baseline.
- LightGBM, XGBoost, and CatBoost: gradient-boosted tree families with separate
  bias/variance and regularization behavior. They are the main high-capacity
  tabular models.
- TabICL and TabPFN: optional foundation-style tabular classifiers. They are
  skipped when not installed or not stable in the local environment.
- Kronos features: optional time-series embeddings used as features, not as the
  sole trading decision engine.

The ensemble can average models, select the best model, learn logistic stacking,
or weight models by fold performance. Probabilities are calibrated with sigmoid
or isotonic calibration. Trading decisions use calibrated probabilities,
empirical payoff estimates, predicted return/downside estimates, and selection
score thresholds.

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
liquidity-score-bucket sizing. The easy preset trains score-bucket artifacts
with base, mid, and high notionals tied to selection score. At runtime, order
size is capped by available capital, configured max order notional, account-mode
notional caps, open-position capacity, and the artifact sizing policy.

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

## Advanced Commands

The easy presets are wrappers around the full command surface. Use the advanced
commands when you need custom data, labels, model families, feature ablations,
or execution assumptions.

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
