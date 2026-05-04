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
  executions, top-of-book quotes, historical bars, market depth, and
  tick-by-tick snapshots where permissions allow.

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
  --target-frequency-mode online \
  --capacity-release-mode planned \
  --permutation-importance \
  --shap-importance \
  --output artifacts/backtests/ml_btcusdt_1h.json

.venv/bin/python -m zeroalpha.cli model signal-audit \
  --config configs/paper.example.toml \
  --interval 15m \
  --candidate-mode active \
  --target-trades-per-day 4 \
  --target-frequency-mode online \
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

Train and save a production scoring artifact:

```bash
.venv/bin/python -m zeroalpha.cli model train-meta \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,histgb,extratrees,lightgbm,catboost,xgboost \
  --stacker weighted \
  --hpo \
  --target-frequency-mode online \
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
  --signal-interval 60 \
  --history-what-to-show AGGTRADES \
  --max-signal-bar-age-seconds 300 \
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
  --signal-interval 60 \
  --history-what-to-show AGGTRADES \
  --max-signal-bar-age-seconds 300 \
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
