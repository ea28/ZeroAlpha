# ZeroAlpha

ZeroAlpha is a proprietary, paper-first BTC/USD spot crypto trading system for
IBKR Gateway/TWS. It combines cost-aware walk-forward ML research with a gated
IBKR execution runtime.

This repository is not financial advice. Paper trading can be operated from
this branch after the checks below pass. Live trading is present only as a
live-gated path and cannot be enabled accidentally: it requires a live IBKR
Gateway/TWS login, live API port, explicit account, live config flags, hard
confirmation strings, streaming logs, and notional/loss caps.

No open-source license is granted.

## Production Safety Statement

Main branch execution target:

- Instrument: IBKR spot BTC/USD crypto, `CRYPTO` / `BTC` / `USD`.
- Venues: `PAXOS`, then `ZEROHASH` fallback when configured.
- Mode: long/flat spot only.
- Orders: IBKR market and limit orders only.
- Stops: synthetic stop monitor only. Native `STP` and `STP LMT` are blocked for
  IBKR spot crypto because IBKR documents crypto support as market/limit only.
- Positioning: one open BTC spot position in the autonomous runner.
- Account: every mutating broker command requires explicit `broker.account` or
  `--account`, and the account must appear in TWS `managedAccounts()`.
- Live mode: requires `mode = "live"`, port `4001` or `7496`,
  `enable_live_trading = true`, `live_confirmation = "ZEROALPHA_LIVE"`,
  `--confirm ZEROALPHA_LIVE_TRADE_RUN`, explicit account, max loss, and max
  notional caps.

Paper vs live depends on the IBKR Gateway/TWS session you are signed into. The
app-level gates are an additional guardrail on top of that.

## Install

Tested with CPython `3.14.4`.

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]"
```

For a smaller local install:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

`constraints-py314.txt` records the known-good local verification environment.

## Configuration

Start with `configs/paper.example.toml` and set:

```toml
[runtime]
mode = "paper"

[broker]
host = "127.0.0.1"
port = 4002       # Gateway paper. TWS paper is 7497.
account = "YOUR_PAPER_ACCOUNT"
read_only = true  # order commands connect with read_only=False after confirmation
```

For live-gated operation, use a separate local config with live port `4001` or
`7496`, explicit live account, and:

```toml
[runtime]
mode = "live"
enable_live_trading = true
live_confirmation = "ZEROALPHA_LIVE"
```

Do not commit real account numbers or local secrets.

## Strategy

ZeroAlpha uses a two-stage meta-labeling strategy:

1. Generate candidate long BTC/USD spot trades from completed bars.
2. Score candidates with a calibrated tabular ensemble trained to predict
   whether the candidate is net-profitable after IBKR-style commission, spread,
   slippage, safety margin, and risk constraints.

The production-like baseline keeps the 6h `100/80` style setup, online target
frequency, planned capacity release, EV/probability selection, and one open spot
position. `capacity_release_mode=actual` is research-only and requires
`--research-gate` because it frees capacity using future backtest exit
knowledge.

## Data Sources

Research feeds:

- Binance spot archives: primary BTCUSDT history and cross-asset context such as
  ETHUSDT, SOLUSDT, and ETHBTC.
- Binance USD-M futures: optional BTC/ETH/SOL futures bars, open interest,
  taker buy/sell flow, funding, and basis context.
- Coinbase Exchange: optional BTC-USD reference candles and public order-book
  surfaces.
- Kraken: source health and independent OHLC validation.
- Polymarket CLOB v2: BTC up/down discovery plus `/books`, `/midpoints`,
  `/spreads`, `/prices-history`, and batch price history from
  `https://clob.polymarket.com`.
- Kalshi Trade API v2: BTC directional 5m/15m markets when available,
  orderbooks, trades, candlesticks, and separated ladder/threshold series.
- IBKR Gateway/TWS: execution truth, account/portfolio/PnL, commissions,
  executions, top-of-book quotes, market depth, historical bars, and
  tick-by-tick data where permissions allow. IBKR historical bars support
  `1 secs`; IBKR's built-in real-time bars are only 5-second bars, so
  trade-run verifies a `1 secs` historical bootstrap and then maintains live
  1-second model bars from persistent top-of-book plus tick-by-tick updates.
  For `AGGTRADES`/`TRADES` parity, the live tick stream must be `Last` or
  `AllLast`; `BidAsk` is only top-of-book quote data. One second is treated as
  the minimum data/exit granularity, not as a fixed holding period. Use
  `--adaptive-horizon` to assign each signal its own volatility-scaled vertical
  barrier from 1 second up to the configured cap.

Prediction-market durations attempted by default:

```text
5m, 15m, 30m, 45m, 1h, 2h, 4h, 24h
```

Unsupported or unavailable provider durations are recorded in coverage reports.
Historical Polymarket price-history snapshots use timestamped prices only; live
market totals such as current liquidity/open interest are not backfilled into
past snapshots.

All OHLCV research bars are normalized to completed bar close time before feature
generation.

## Features And Signals

Core feature families:

| Family | Examples |
| --- | --- |
| Momentum/trend | rolling returns, EMA/SMA gaps, breakout slope |
| Technicals | RSI, MACD-style EMA spread, Bollinger z, ATR/range shock |
| Volume/order flow | quote volume, VWAP distance, volume surprise, taker imbalance |
| Microstructure | spread, microprice, top-book imbalance, depth imbalance |
| Cross-asset | ETH/SOL relative strength, ETHBTC, spot/futures basis |
| Futures context | Binance UM OI, taker flow, funding, basis, IBKR MBT context |
| Prediction markets | PM deltas, term structure, side-aligned probability, liquidity weights, platform disagreement |
| IBKR runtime | live bid/ask, quote age, spread, position/PnL snapshots |

Interpretability:

- `--permutation-importance` computes fold-local grouped permutation importance.
- `--shap-importance` computes SHAP for supported fitted base models.
- Native tree/linear importances are reported where available.
- `net_pnl` permutation importance is labeled `threshold_only_net_pnl` unless a
  full strategy policy replay is used.

## Models

Implemented model families:

- Logistic regression
- HistGradientBoosting
- RandomForest
- ExtraTrees
- LightGBM
- XGBoost
- CatBoost
- TabICL / TabPFN when installed and stable
- Average, best, weighted, or logistic stacker
- Sigmoid or isotonic probability calibration

HPO is nested inside walk-forward folds with `--hpo`. The objective can be
`sharpe`, `net_pnl`, or `calmar`; current research optimizes PnL first, Sharpe
second, and trade count third.

Target-frequency settings are an upper turnover objective, not permission to
fill a quota with losing calibration. By default the selector uses
`expected_utility`, so probability, empirical payoff, predicted return, and
downside all have to agree enough for a trade. Use
`--allow-negative-ev-frequency-probe` only for explicit diagnostic runs.
For aggressive intraday runs, prefer `--adaptive-selection-score-floor`; the
learned floor is saved into production artifacts so IBKR trade-run applies the
same weak-signal filter that the walk-forward backtest used.
For high-fee short-horizon spot crypto, keep the default
`minimum_expected_value` at `0.0` unless intentionally testing a stricter edge
floor; a static floor larger than the net target can make every signal
untradable.

When `broker trade-run` uses `--use-model-exit-geometry`, synthetic exits use
the model's gross price barriers (`gross_profit_move` and
`gross_stop_distance`), not the net after-cost label targets. This keeps live
paper exits aligned with the triple-barrier labels and ML backtest replay.

## Command Reference

Local checks:

```bash
.venv/bin/python -m zeroalpha.cli config check --config configs/paper.example.toml
.venv/bin/python -m zeroalpha.cli data health-check --cache-dir /tmp/zeroalpha-source-health
.venv/bin/python -m zeroalpha.cli model smoke --models logistic,histgb,extratrees,lightgbm,catboost,xgboost
.venv/bin/python -m zeroalpha.cli model kronos-status
.venv/bin/python -m zeroalpha.cli db init --path .zeroalpha/zeroalpha.sqlite
.venv/bin/python -m zeroalpha.cli kill-switch enable --config configs/paper.example.toml
.venv/bin/python -m zeroalpha.cli kill-switch disable --config configs/paper.example.toml
```

Data acquisition:

```bash
.venv/bin/python -m zeroalpha.cli data binance-url --symbol BTCUSDT --interval 1h --month 2026-04
.venv/bin/python -m zeroalpha.cli broker historical-bars --config configs/paper.example.toml --account YOUR_PAPER_ACCOUNT --duration "2 D" --bar-size "1 min" --what-to-show AGGTRADES --output data/raw/ibkr/historical_btcusd_1m.jsonl
.venv/bin/python -m zeroalpha.cli broker record-quotes --config configs/paper.example.toml --account YOUR_PAPER_ACCOUNT --duration-seconds 600 --interval-seconds 5 --output data/raw/ibkr/quotes_btcusd.jsonl
```

Research:

```bash
.venv/bin/python -m zeroalpha.cli backtest candidate --config configs/paper.example.toml --years 3 --interval 1h

.venv/bin/python -m zeroalpha.cli backtest ml \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,histgb,extratrees,lightgbm,catboost,xgboost \
  --stacker weighted \
  --hpo \
  --adaptive-threshold \
  --candidate-type-thresholds \
  --empirical-payoff-ev \
  --selection-score expected_utility \
  --target-frequency-mode online \
  --capacity-release-mode planned \
  --adaptive-horizon \
  --min-holding-seconds 1 \
  --adaptive-horizon-target-move-bps 50 \
  --dynamic-exit-overlay \
  --dynamic-exit-checkpoints-seconds 5,15,30,60,120,300 \
  --permutation-importance \
  --shap-importance \
  --output artifacts/backtests/ml_btcusdt_1h.json

.venv/bin/python -m zeroalpha.cli model signal-audit \
  --config configs/paper.example.toml \
  --interval 15m \
  --candidate-mode active \
  --target-trades-per-day 4 \
  --target-frequency-mode online \
  --selection-score expected_utility \
  --models logistic,histgb,extratrees,lightgbm,catboost,xgboost \
  --permutation-importance \
  --shap-importance \
  --output artifacts/models/signal_audit_btcusdt_15m.json

.venv/bin/python -m zeroalpha.cli model sweep-labels \
  --config configs/paper.example.toml \
  --net-profit-targets 0.01,0.015,0.02 \
  --net-stop-losses 0.008,0.012,0.016 \
  --max-holding-hours-values 4,6,8 \
  --models logistic,histgb,extratrees,lightgbm
```

For 1-second IBKR spot research, keep execution modeled as spot crypto. Futures
bars such as CME MBT can be passed through `--context-bars-jsonl` as predictors,
but do not switch `--instrument-model futures` unless intentionally researching a
separate futures-execution strategy.

Local IBKR 1-second replay experiment suite:

```bash
.venv/bin/python scripts/ibkr_1s_experiment_suite.py --dry-run --limit 10 --python .venv/bin/python

.venv/bin/python scripts/ibkr_1s_experiment_suite.py --run --python .venv/bin/python \
  --name champion_repro_extratrees_return_first_1250x8 \
  --name hpo_extratrees_quota_t6 \
  --name data_context_full_ibkr_bidask_futures \
  --name candidate_stride8_fold225_75_112 \
  --name dynamic_exit_balanced_seconds
```

The suite writes a manifest plus ranked summaries under
`artifacts/backtests/ibkr_1s_experiments_20260504/next_experiment_suite/`.
All experiments keep spot crypto execution, the `$10K` aggregate spend cap,
IBKR-style `0.18%` per-side commission with a `$1.75` minimum, and futures data
only as optional features.

Strict live-valid `$10K` BTC spot research:

```bash
.venv/bin/python -m zeroalpha.models.ibkr_experiments \
  --output-dir artifacts/backtests/live_valid_strict_10k \
  --category live_valid_strict_10k \
  --limit 10 \
  --dry-run

.venv/bin/python -m zeroalpha.models.ibkr_experiments \
  --output-dir artifacts/backtests/live_valid_strict_10k \
  --category live_valid_strict_10k \
  --run \
  --force
```

This strict suite is intentionally data-first and fail-closed. A promoted run
must use BTC/USD spot crypto execution, market-entry fill modeling, a `$10K`
aggregate exposure cap, `0.18%` per-side commission with the `$1.75` minimum,
tick-backed IBKR `1 secs` label/execution replay, online cash-aware selection,
and serialized execution/sizing/horizon/setup/data contracts. The readiness gate
requires at least seven calendar days of tick-backed 1-second replay and at least
two walk-forward folds; fourteen or more days is preferred before comparing
against the current strict live-valid baseline.

The current repository includes only the short May 2026 1-second replay window
used for smoke tests. With `--strict-live-valid-1s`, that six-hour window is
expected to reject promotion with `window_too_short` and
`too_few_walk_forward_folds`. That is a safety feature, not a failed strategy
result. Collect or ingest multi-day IBKR tick-by-tick-backed bars before trusting
the strict 1-second HPO matrix.

The strict model path now supports explicit setup-family specialists
(`mean_reversion_exhaustion`, `momentum_continuation`,
`liquidity_vacuum_breakout`, and `chop_no_trade`), microstructure features,
multi-head ranking targets for net return/MAE/MFE/time-to-exit/early adverse
move, `capital_efficiency` selection, dynamic exits from 5 seconds through 1
hour, and a hard gross-edge-over-cost floor. Futures, Binance, and prediction
market feeds may be used only as causal features; execution remains long-only
IBKR BTC spot.

Train and save a production scoring artifact:

```bash
.venv/bin/python -m zeroalpha.cli model train-meta \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,histgb,extratrees,lightgbm,catboost,xgboost \
  --stacker weighted \
  --hpo \
  --target-trades-per-day 3 \
  --target-frequency-mode online \
  --selection-score expected_utility \
  --capacity-release-mode planned \
  --permutation-importance \
  --shap-importance \
  --output artifacts/models/meta_label_walk_forward_btcusdt_1h.json \
  --save-artifact artifacts/models/zeroalpha_spot_btc_prod.joblib
```

Paper broker checks:

```bash
.venv/bin/python -m zeroalpha.cli broker smoke \
  --config configs/paper.example.toml \
  --account YOUR_PAPER_ACCOUNT \
  --read-only

.venv/bin/python -m zeroalpha.cli broker order-test \
  --config configs/paper.example.toml \
  --account YOUR_PAPER_ACCOUNT \
  --notional 100 \
  --offset-bps 20 \
  --confirm PAPER_ORDER_TEST

.venv/bin/python -m zeroalpha.cli broker paper-test \
  --config configs/paper.example.toml \
  --account YOUR_PAPER_ACCOUNT \
  --duration-seconds 600 \
  --interval-seconds 30 \
  --max-cash-usd 10000 \
  --max-loss-usd 1000 \
  --submit-order \
  --order-notional 100 \
  --confirm IBKR_PAPER_TEST

.venv/bin/python -m zeroalpha.cli broker round-trip-test \
  --config configs/paper.example.toml \
  --account YOUR_PAPER_ACCOUNT \
  --notional 100 \
  --hold-seconds 10 \
  --synthetic-stop-loss-bps 100 \
  --max-cash-usd 10000 \
  --max-loss-usd 1000 \
  --confirm IBKR_ROUND_TRIP_TEST
```

Autonomous paper trading example requested for this release:

```bash
.venv/bin/python -m zeroalpha.cli broker trade-run \
  --config configs/paper.example.toml \
  --account YOUR_PAPER_ACCOUNT \
  --model-artifact artifacts/models/zeroalpha_spot_btc_prod.joblib \
  --capital-usd 5000 \
  --max-loss-usd 250 \
  --max-order-notional-usd 5000 \
  --duration-seconds 600 \
  --signal-interval 1 \
  --live-data-mode streaming \
  --require-live-1s-data \
  --tick-by-tick-type Last \
  --history-duration "1800 S" \
  --history-bar-size "1 secs" \
  --history-what-to-show AGGTRADES \
  --max-signal-bar-age-seconds 2.5 \
  --account-refresh-interval-seconds 30 \
  --adaptive-horizon \
  --min-holding-seconds 1 \
  --adaptive-horizon-target-move-bps 50 \
  --use-model-exit-geometry \
  --decision-threshold 0.005 \
  --stream-format json \
  --event-log data/raw/ibkr/trade_run_events.jsonl \
  --state-log data/raw/ibkr/trade_run_state.jsonl \
  --confirm IBKR_PAPER_TRADE_RUN
```

Live-gated example. This will only work with a live config, live Gateway/TWS
session, live port, explicit account, and both live confirmation strings:

```bash
.venv/bin/python -m zeroalpha.cli broker trade-run \
  --config configs/live.local.toml \
  --account YOUR_LIVE_ACCOUNT \
  --model-artifact artifacts/models/zeroalpha_spot_btc_prod.joblib \
  --capital-usd 5000 \
  --max-loss-usd 250 \
  --max-order-notional-usd 1000 \
  --duration-seconds 600 \
  --signal-interval 1 \
  --live-data-mode streaming \
  --require-live-1s-data \
  --tick-by-tick-type Last \
  --history-duration "1800 S" \
  --history-bar-size "1 secs" \
  --history-what-to-show AGGTRADES \
  --max-signal-bar-age-seconds 2.5 \
  --account-refresh-interval-seconds 30 \
  --adaptive-horizon \
  --min-holding-seconds 1 \
  --adaptive-horizon-target-move-bps 50 \
  --use-model-exit-geometry \
  --stream-format json \
  --event-log data/raw/ibkr/live_trade_run_events.jsonl \
  --state-log data/raw/ibkr/live_trade_run_state.jsonl \
  --confirm ZEROALPHA_LIVE_TRADE_RUN
```

## Operational Checklist

Before paper trade-run:

- Gateway/TWS is signed into paper trading.
- API socket is enabled.
- Config mode is `paper`.
- Port is `4002` for Gateway paper or `7497` for TWS paper.
- `broker.account` or `--account` is set and appears in `managedAccounts()`.
- `ruff`, `pytest`, `pip check`, `data health-check`, `model smoke`, and
  `broker smoke` pass.
- A fresh model artifact and manifest exist.
- `kill-switch` is disabled before starting and can be enabled to stop the bot.
- `--max-loss-usd`, `--capital-usd`, and `--max-order-notional-usd` are set.
- The run emits `market.one_second_data_verified` with `ok=true`,
  `bar_size_seconds=1`, and recent gaps near 1.0 second before any model score.
- `--max-missing-model-feature-fraction` is left at its fail-closed default of
  `0`, unless the run is explicitly a diagnostic partial-feature experiment.
- For aggressive paper experiments with the current production artifact, set an
  explicit `--decision-threshold`; leave it at `0` to use the artifact threshold.
- Runtime events are streamed and written to JSONL.

Before live-gated trade-run:

- Paper run has already passed with TWS execution, commission, and PnL
  reconciliation.
- Gateway/TWS is intentionally signed into live.
- Config mode is `live`, port is `4001` or `7496`, and live config flags are set.
- Live notional is smaller than paper verification size.
- You accept that this is still experimental trading software.

## Verification

```bash
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python -m zeroalpha.cli data health-check
.venv/bin/python -m zeroalpha.cli model smoke
.venv/bin/python -m zeroalpha.cli broker smoke --config configs/paper.example.toml --account YOUR_PAPER_ACCOUNT --read-only
git diff --check
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

Generated `data/`, `artifacts/`, logs, caches, and virtual environments are not
source artifacts and should stay out of git.

## License

Private proprietary code. `LicenseRef-Proprietary`. No open-source license is
granted.
