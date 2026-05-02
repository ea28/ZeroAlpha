# ZeroAlpha

ZeroAlpha is a IBKR BTC/USD trading research and execution
system. It is designed around one conservative standard: a strategy must pass
cost-aware walk-forward research, model-driven backtesting, and IBKR paper
operational checks before it is allowed anywhere near live capital.

The current production target is BTC/USD spot crypto through IBKR Gateway in
paper mode, long/flat only. Short research is blocked for spot crypto because
the current IBKR crypto order layer is not a shortable futures or margin
instrument model.

This is trading software, not financial advice. Live trading remains disabled
by default.

## Strategy

ZeroAlpha uses a two-layer meta-labeling strategy:

1. Candidate generators create possible BTC trades from price action, volatility,
   liquidity, and cross-crypto context.
2. A calibrated ML ensemble decides whether each candidate has positive expected
   value after commissions, spread, slippage, missed-fill risk, and risk limits.

The default trade lifecycle is:

```text
Binance/Coinbase/Kraken research data
        -> data quality gates
        -> candidate events
        -> triple-barrier net-of-cost labels
        -> causal feature table
        -> purged walk-forward model training
        -> calibrated probability and EV gate
        -> model-driven backtest with live-style risk rules
        -> IBKR paper smoke/order tests
```

The first executable instrument is IBKR spot BTC/USD. Normal entry and exit are
limit-order-first. Urgent exits are modeled as market IOC. Stops, targets, and
time exits are synthetic in the bot and backtester.

## Models

The implemented model stack is:

- Logistic regression baseline
- LightGBM
- CatBoost
- XGBoost
- TabICL, when installed and stable
- TabPFN, when installed and stable
- Low-capacity stacker over out-of-fold predictions
- Final probability calibrator

The model target is:

```text
P(candidate trade is net-profitable after full estimated costs)
```

It is not a next-candle direction classifier.

Fold-local hyperparameter tuning is available for the GBDT models with
`--hpo`. Foundation models are bounded and run through isolated workers so a
TabICL or TabPFN failure does not hang the full validation job.

Kronos support is included as a feature generator, not as a direct trader:

- `proxy` mode is available immediately and creates causal K-line embedding-like
  features from lagged OHLCV windows.
- `official` mode expects the Kronos repository/model import path to provide
  `model.Kronos` and `model.KronosTokenizer`, plus compatible checkpoint access.
- `auto` mode uses official Kronos when available and falls back to proxy mode.

Use `zeroalpha model kronos-status` before enabling official Kronos features.

## Data Feeds

ZeroAlpha keeps public no-key feeds in the default repo:

- IBKR Gateway/TWS API: execution truth for contract qualification, live bid/ask,
  account state, order status, executions, commissions, rejects, and quote
  recording.
- Binance public archive: primary deep-history research feed for BTCUSDT plus
  cross-crypto context such as ETHUSDT, SOLUSDT, and ETHBTC.
- Coinbase Exchange REST candles: optional BTC-USD reference feed. It is off by
  default in training/backtest commands and can be enabled explicitly.
- Kraken public OHLC endpoint/parser: independent source health and validation
  support.
- Polymarket/Kalshi BTC prediction-market signals: optional research features
  from Polymarket Gamma discovery plus production CLOB v2 market-data endpoints,
  and Kalshi Trade API v2 public market data. These are off by default and can
  be enabled for recent retraining windows with `--prediction-market-signals`.

API-key feeds such as CoinGlass, paid institutional feeds, and macro/on-chain
providers are intentionally not part of the default working path. They can be
added later behind explicit connectors and licensing checks.

## Repository Layout

```text
configs/                  paper IBKR example config
src/zeroalpha/backtest/   candidate and model-driven backtests
src/zeroalpha/broker/     IBKR Gateway adapter, pacing, quote recorder
src/zeroalpha/candidates/ event generation
src/zeroalpha/data/       public data clients and data quality gates
src/zeroalpha/execution/  order intents and decision logic
src/zeroalpha/features/   engineered and Kronos-compatible features
src/zeroalpha/labels/     triple-barrier labeler
src/zeroalpha/models/     dataset builder, ensemble, HPO, sweeps
src/zeroalpha/risk/       paper/live risk checks
src/zeroalpha/validation/ purged walk-forward splitting
tests/                    unit and integration-style tests
```

Generated `data/`, `artifacts/`, logs, caches, and virtual environments are
ignored by git.

## Install

The repo is tested with 64-bit CPython 3.14.4. On this machine:

```bash
/opt/homebrew/bin/python3.14 --version
/opt/homebrew/bin/python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[broker,data,ml,kronos,dev]"
```

For a lighter development install without heavy model packages:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Latest-version optional dependency floors are kept in `pyproject.toml`.

## Configure IBKR Paper

Use `configs/paper.example.toml` as the starting point. The safe defaults are:

- runtime mode: `paper`
- IB Gateway paper port: `4002`
- TWS paper port, if using TWS instead: `7497`
- account equity assumption: `$10,000`
- BTC/USD crypto exchange candidates: `PAXOS`, `ZEROHASH`
- max open positions: `1`
- spot crypto instrument model: `long/flat`

IB Gateway must already be authenticated. In Gateway/TWS, enable socket clients
and disable read-only API only when you are intentionally testing paper orders.

## Verification Commands

Run local tests and lint:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

Check public data sources:

```bash
.venv/bin/python -m zeroalpha.cli data health-check \
  --cache-dir /tmp/zeroalpha-source-health
```

Check model imports:

```bash
.venv/bin/python -m zeroalpha.cli model smoke \
  --models logistic,lightgbm,catboost,xgboost,tabicl,tabpfn
```

Check Kronos availability:

```bash
.venv/bin/python -m zeroalpha.cli model kronos-status
```

Check IBKR paper connectivity without placing orders:

```bash
.venv/bin/python -m zeroalpha.cli broker smoke \
  --config configs/paper.example.toml \
  --read-only
```

Submit and cancel a tiny paper limit order only after confirming the config is
paper mode and the port is a paper port:

```bash
.venv/bin/python -m zeroalpha.cli broker order-test \
  --config configs/paper.example.toml \
  --notional 100 \
  --offset-bps 20 \
  --wait-seconds 2 \
  --cancel-wait-seconds 2 \
  --confirm PAPER_ORDER_TEST
```

The order-test command refuses non-paper mode, refuses live ports, and requires
the explicit confirmation string.

## Research Commands

Train the full meta-label ensemble:

```bash
.venv/bin/python -m zeroalpha.cli model train-meta \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,lightgbm,catboost,xgboost,tabicl,tabpfn \
  --stacker average \
  --hpo
```

Run the model-driven strategy backtest:

```bash
.venv/bin/python -m zeroalpha.cli backtest ml \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,lightgbm,catboost,xgboost \
  --stacker weighted \
  --hpo \
  --adaptive-threshold \
  --candidate-type-thresholds \
  --empirical-payoff-ev \
  --confidence-scaled-sizing \
  --output artifacts/backtests/ml_btcusdt_1h.json
```

Research-only short-side evaluation is available, but it remains gated and does
not change the production BTC/USD spot long/flat safety model:

```bash
.venv/bin/python -m zeroalpha.cli backtest ml \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h \
  --models logistic,lightgbm,catboost,xgboost \
  --stacker weighted \
  --hpo \
  --adaptive-threshold \
  --side-mode long_short \
  --allow-spot-short-research \
  --research-gate \
  --allow-research-short-backtest \
  --candidate-type-thresholds \
  --empirical-payoff-ev \
  --confidence-scaled-sizing \
  --target-trades-per-day 1 \
  --output artifacts/backtests/ml_btcusdt_1h_research.json
```

Enable Coinbase reference candles explicitly when the interval is supported:

```bash
--coinbase-reference-products BTC-USD
```

Enable Binance USD-M futures reference candles for spot/perp basis and futures
flow context:

```bash
--binance-um-futures-reference-symbols BTCUSDT,ETHUSDT,SOLUSDT
```

Enable BTC prediction-market signals from Polymarket and Kalshi:

```bash
--prediction-market-signals \
--prediction-market-durations 5m,15m,30m,1h,2h,4h,24h \
--prediction-market-lookback-days 14
```

For dense intraday research where the model must take a few trades per day, use
quota ranking instead of the strict probability/EV gate:

```bash
--candidate-mode dense \
--interval 15m \
--max-holding-hours 2 \
--target-trades-per-day 3 \
--target-frequency-mode quota \
--selection-score probability \
--research-gate \
--allow-negative-ev-frequency-probe \
--max-open-positions 4
```

Polymarket discovery attempts every requested duration but currently active BTC
short-form markets are mainly 5m, 15m, 1h, and 4h. Kalshi contributes the exact
15-minute BTC up/down series and hourly BTC ladder signals when available. The
model report records per-venue coverage and skipped durations.

Enable Kronos proxy features:

```bash
--kronos-features --kronos-mode proxy
```

Use the rule-only candidate backtest only as a diagnostic baseline:

```bash
.venv/bin/python -m zeroalpha.cli backtest candidate \
  --config configs/paper.example.toml \
  --years 3 \
  --interval 1h
```

## Backtesting Guarantees And Limits

Implemented:

- timestamp-based holding horizons for 1m, 1h, and 4h bars
- triple-barrier labels with conservative same-bar handling
- per-notional IBKR-style commission model
- spread, slippage, and safety-margin attribution
- conservative bar-level limit-fill simulation
- missed-fill reporting
- exit replay from the actual simulated fill timestamp and price
- max one BTC position
- fee-efficient notional rejection
- equity-based sizing
- daily, weekly, drawdown, and cooldown controls
- spot-short rejection for the spot crypto instrument model
- calibrated probability and EV gates

Not implemented for production claims:

- sub-minute or 1-second order-book replay
- queue position, partial queue depletion, and venue-specific maker/taker fill
  modeling
- latency-sensitive cancel/replace replay
- autonomous paper trading daemon
- live trading promotion workflow

Any sub-minute or many-trades-per-day result should be treated as invalid until
tick or L2 replay exists.

## Trading Gate

Do not start autonomous paper trading until a model-driven backtest:

- beats no-trade, buy-and-hold reference, simple rule baselines, and logistic
  regression after full costs
- survives 2x cost stress
- has acceptable calibration by fold and candidate type
- has enough trades to be meaningful
- does not rely on impossible fills
- keeps drawdown inside paper limits
- passes IBKR restart, reconnect, order reject, cancel, and reconciliation tests
- is regenerated from clean `data/` and `artifacts/` directories

Do not start live trading until paper and shadow-live operation prove that IBKR
spreads, fill behavior, rejects, quote age, and reconciliation match the
backtest assumptions.

## Housekeeping

Generated research data and reports are intentionally ignored by git:

```bash
rm -rf data artifacts logs .zeroalpha .pytest_cache .ruff_cache
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
```

The repository should remain reproducible from source, config, and the commands
above. Do not commit virtual environments, downloaded candles, backtest JSON
artifacts, broker logs, or local kill-switch state.

## License

MIT. See `LICENSE`.
