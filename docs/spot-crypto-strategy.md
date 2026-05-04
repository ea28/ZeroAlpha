# Spot Crypto Strategy And Performance

This is the main-branch strategy note for ZeroAlpha. It covers BTC/USD spot
crypto through IBKR Gateway/TWS, long/flat only. Crypto futures remain a
research instrument and feature source unless promoted in a separate branch.

This is not financial advice and does not approve live trading.

## Current Recommendation

The current main-branch recommendation is paper-only BTC/USD spot execution with
live trading gated behind explicit live config and confirmation strings.

Recommended research baseline:

- 6h-style holding horizon with 100/80 bps gross target/stop family.
- Online target-frequency selection.
- `capacity_release_mode=planned`.
- One open spot position.
- IBKR spot fee model: 18 bps per side, minimum `$1.75` per order, max 1% of
  trade value.
- Synthetic stops only for IBKR spot crypto.
- Prediction-market and futures features are allowed only when timestamped
  causally at or before the scoring timestamp with the configured latency buffer.

Best saved research rows to beat:

| Variant | Trades | Net PnL | Sharpe | Note |
| --- | ---: | ---: | ---: | --- |
| no-PM bucket sizing | 23 | `$724.49` | `22.58` | Best PnL; PM kept as shadow/gated feed |
| PM-enabled bucket sizing | 20 | `$584.77` | `24.64` | Best Polymarket/Kalshi-enabled row |
| `$10k x3` compromise | varied | lower than leader | strong | Best capital/trade-count compromise |
| `$5k x6` turnover leader | higher trades | lower PnL | >20 target | Trade-count benchmark |

Promotion threshold for new experiments: beat `$724.49` no-PM net PnL or
materially increase trade count while keeping Sharpe above `20`, without relying
on future-looking ranking, tiny samples, optimistic fees, or missing latency
buffers.

## Current Blockers

Live trading is still blocked until:

- Longer out-of-sample paper data confirms the edge.
- IBKR quote/bar replay survives restart and reconnect tests.
- Paper `broker trade-run` proves order, fill, commission, PnL, and final
  position reconciliation over longer sessions.
- Continuous Polymarket/Kalshi/IBKR order-book snapshots are recorded before
  sub-minute exits are trusted.
- Stop behavior remains synthetic for spot crypto; native `STP`/`STP LMT` must
  remain blocked.

## Reproduction

Train and save the production scoring artifact:

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

Run the main backtest:

```bash
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
  --sizing-mode score_bucket \
  --sizing-base-notional 5000 \
  --sizing-mid-notional 10000 \
  --sizing-high-notional 15000 \
  --permutation-importance \
  --shap-importance \
  --output artifacts/backtests/spot_retest.json
```

Enable PM/futures/cross-asset features:

```bash
--context-symbols ETHUSDT,SOLUSDT,ETHBTC \
--binance-um-futures-reference-symbols BTCUSDT,ETHUSDT,SOLUSDT \
--binance-um-derivatives-metrics \
--binance-um-taker-flow \
--binance-um-basis \
--prediction-market-signals \
--prediction-market-durations 5m,15m,30m,45m,1h,2h,4h,24h
```

IBKR paper runner:

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
  --confirm IBKR_PAPER_TRADE_RUN
```

## Instrument And Execution

IBKR spot BTC/USD:

- `secType = CRYPTO`
- `symbol = BTC`
- `currency = USD`
- primary exchange candidate: `PAXOS`
- fallback exchange candidate: `ZEROHASH`

IBKR documents crypto orders as market and limit only. For this reason,
ZeroAlpha blocks native `STP` and `STP LMT` for spot crypto and implements stop
losses as a synthetic quote monitor that submits a bounded market sell after the
trigger.

Every order-submitting path now requires:

- explicit account configured or passed with `--account`
- account present in `managedAccounts()`
- non-read-only API connection
- runtime mode allowing orders
- notional under paper/live cap
- quote/reference price for quantity market exits
- sell quantity no greater than current BTC position

Emergency cleanup cancels open orders, attempts bounded liquidation only for
runner-created exposure, records final position/PnL state, and emits critical
runtime events.

## Cost Model

IBKR lowest-volume spot crypto schedule:

- `0.18% * Trade Value` per side.
- Minimum commission: `$1.75` per order.
- Maximum commission: `1%` of trade value.

At `$10,000`, commission is about `$18` per side, or 36 bps round trip before
spread, slippage, and safety margin. This is why the earlier futures strategy
did not transfer cleanly to spot: the futures round-trip cost was materially
lower, while spot needs a larger edge just to break even.

## Data And Causality

All bars are normalized to completed close time before feature generation.

Prediction markets are used as leading indicators only when the snapshot
timestamp is causal. Polymarket CLOB v2 price-history snapshots use timestamped
history fields only; current market totals are not backfilled into the past.
Kalshi true directional 5m/15m markets are separated from ladder/threshold
markets, which are exposed under `ladder_*` duration features.

IBKR data available through this branch:

- L1 bid/ask snapshots.
- Historical `AGGTRADES` PAXOS crypto bars, plus `MIDPOINT`, `BID`, `ASK`,
  `BID_ASK`, and other bar types where Gateway permissions allow.
- Market depth snapshots.
- Tick-by-tick snapshots.
- Account summary, portfolio, positions, daily/realized/unrealized PnL.
- TWS executions and commission reports.

## Feature Selection And Interpretability

The current interpretability stack is fold-safe:

- grouped permutation importance on test folds only
- native tree/linear importances
- SHAP importance for supported fitted base estimators
- model-family diagnostics and leave-family comparisons in reports

The most useful families in the profitable spot experiments have been:

- core momentum/trend and realized-volatility regime features
- fee/target/stop geometry features
- volume and taker-flow features when available
- futures context as a leading risk-on/risk-off signal
- prediction-market residual/term-structure features as a shadow/gated feed

PM-enabled runs showed higher Sharpe but lower PnL in the saved champion rows.
The likely causes are stricter gating, lower trade count, sparse PM coverage,
and noisy PM features acting as a risk filter rather than a pure alpha source.

## Why Futures Looked Better

The crypto-futures branch had three structural advantages:

- Lower effective round-trip cost relative to the target move.
- Cleaner short/hedge semantics.
- Better fit between the target horizon and futures liquidity/flow features.

Spot crypto at IBKR pays 36 bps round-trip commission at normal research sizes,
so shorter holding periods and more frequent trading quickly become cost
dominated unless order-book/taker-flow features add a much larger edge.

## Next Experiments

High-priority experiments:

- Longer paper `broker trade-run` sessions with TWS commission/PnL
  reconciliation.
- Continuous IBKR PAXOS L1/L2/tick-by-tick capture.
- Continuous Polymarket/Kalshi order-book capture for 5m/15m/1h/4h BTC markets.
- Feature-pruned reruns using fold-local permutation/SHAP summaries.
- Model family ablations: CatBoost-only, LightGBM-only, ExtraTrees-only,
  HistGB-only, and weighted ensemble.
- More futures-leading features: ETH spot/futures, BTC/ETH basis, OI velocity,
  taker imbalance, funding changes, and liquidation pressure.
- Dynamic exit overlay that keeps the 6h label but exits early only when
  expected remaining value turns negative after fees.

Rejected for production claims:

- hard 1-second exits without tick/L2 replay
- same-day quota ranking
- `capacity_release_mode=actual` without `--research-gate`
- native spot crypto stops
- PM market totals backfilled into historical snapshots

## Appendix: Older Research History

Earlier May 2026 experiments found:

- Very short 5m/sub-hour spot holding tests traded more often but lost too much
  edge after spot commission.
- PM-enabled rows improved Sharpe but often reduced PnL and trade count.
- Wider HPO sometimes overfit and degraded the result.
- IBKR PAXOS top-of-book spread was tiny in samples, but commission dominated.
- Futures context helped explain why futures performed better, but using futures
  as a spot feature needs strict timestamp alignment.

The durable lesson is that spot needs fewer, better trades unless continuous
microstructure data proves a genuine high-turnover edge.
