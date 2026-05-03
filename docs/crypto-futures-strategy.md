# Crypto Futures Strategy Write-Up

This branch documents the BTC crypto-futures research line. It is separate from
the conservative IBKR BTC/USD spot paper-trading target described in the main
README. Nothing in this branch should be read as live-trading approval or
financial advice. The current results are research backtests over a short recent
window and need substantially more out-of-sample data before production use.

## Strategy Summary

The best-performing strategy is a BTCUSDT crypto-futures / perpetual-style
research strategy, not a spot BTC strategy.

The strategy is still built around the existing ZeroAlpha meta-labeling stack:

1. Generate candidate BTC trades from price-action, volatility, support/reclaim,
   trend-continuation, and dense baseline setups.
2. Label each candidate with a net-of-cost triple-barrier outcome.
3. Train calibrated tree ensembles in purged walk-forward splits.
4. Rank candidate events by predicted success probability.
5. Select a daily quota of the strongest signals subject to spacing and risk
   constraints.
6. Replay the selected events through the model-driven futures backtester with
   stop-loss, target, time-exit, fees, spread, and slippage assumptions.

The current champion is long-only. Long/short research was tested, but the short
side was not profitable on the available sample. The execution and order-intent
code can represent stop-loss exits for shorts, but the current model should not
be promoted to live shorting based on these results.

## Data Used

Primary price data:

- `BTCUSDT` 15-minute bars from the Binance public archive.
- Cross-crypto context bars where available, including large-cap crypto pairs
  used by the existing feature pipeline.
- Optional Coinbase/Kraken data remains available for source health checks and
  independent validation, but the champion run uses Binance as the primary bar
  source.

Prediction-market data:

- Polymarket BTC up/down markets through the production CLOB v2 market-data
  path.
- Kalshi BTC up/down market data through the Kalshi Trade API v2.
- Durations requested in the research runs: `5m`, `15m`, `30m`, `1h`, `2h`,
  `4h`, and `24h`.

In the cached research window, prediction-market coverage was useful but sparse:

- Kalshi: 1,742 usable snapshots, concentrated in 15-minute markets.
- Polymarket: 26 usable snapshots across 15-minute, 1-hour, and 4-hour markets.
- Polymarket/Kalshi 5-minute coverage was too thin to support a reliable 5-minute
  strategy in this backtest.

IBKR quote data:

- The branch adds optional ingestion for IBKR quote-recorder JSONL records.
- Features include top-of-book bid/ask, spread, spread in bps, bid size, ask
  size, and simple quote imbalance.
- No useful IBKR quote-record file was available for the champion backtest, so
  these features are implemented but not responsible for the current result.

## Feature And Model Design

The research did not move to a deep neural model. For this dataset shape, the
strongest evidence still favors calibrated boosted trees: the samples are
tabular, noisy, and relatively small. The branch therefore improves the existing
tree ensemble rather than replacing it with an LSTM/Transformer.

Champion model stack:

- `histgb`
- `lightgbm`
- `xgboost`
- sigmoid probability calibration
- weighted stacker
- fold-local wide hyperparameter tuning
- `--hpo-trials 12`

The wide HPO profile adds more regularized candidates for the boosted trees:

- lower learning rates with more estimators
- shallower trees and smaller leaf counts
- larger minimum leaf/sample constraints
- feature and row subsampling
- stronger L1/L2 regularization

An experimental `quota` HPO profile was added and tested. It directly optimizes
the target-frequency selection objective, but it underperformed the standard
wide profile on this sample. It remains opt-in rather than replacing the default
HPO path.

## Signal Selection

The champion selects signals by probability, not by predicted return or expected
utility. The best tested selection settings were:

- interval: `15m`
- target frequency: `4.5` trades/day
- target-frequency mode: `quota`
- selection score: `probability`
- minimum signal spacing: `0.75` hours
- maximum open futures positions: `4`

This selection mechanism intentionally takes more trades than a high absolute
probability cutoff would take. It ranks the day's candidates and chooses the best
available events, while spacing prevents clustered duplicate entries.

Selection-time setup-family filters were also implemented. The best "more
trades" variant excludes weaker setup families at selection time and uses a
tighter stop geometry, but the Sharpe ratio is lower than the champion.

## Stops, Targets, And Risk

Stop losses are part of the research path. The backtest labels and trade replay
use a net-of-cost barrier model with explicit target, stop, time exit, spread,
fees, slippage, and safety-margin assumptions.

Champion risk geometry:

- net profit target: `0.001`
- net stop loss: `0.001`
- minimum gross profit: `35` bps
- minimum gross stop: `30` bps
- max holding time: `2` hours
- assumed spread: `4` bps
- base slippage: `1` bps
- safety margin: `2` bps
- tier fee rate: `0.0004`

The more-active variant that traded 15 times used a tighter gross stop:

- minimum gross profit: `35` bps
- minimum gross stop: `25` bps
- target frequency: `5` trades/day

## Why Capital Deployment Improved

The earlier strategy did not deploy more capital because risk sizing was doing
what it was supposed to do. With dynamic stop distances around 45-50 bps after
costs, `risk_per_trade=0.0075` naturally capped a $10,000 paper account near
roughly $15,000 notional.

For futures research, the equity cap was relaxed while stop-loss risk sizing was
kept. Raising the futures risk budget allowed more notional:

- `risk_per_trade=0.010` supports about `$20,000` notional.
- `risk_per_trade=0.0125` supports about `$25,000` notional.
- `risk_per_trade=0.015` supports about `$30,000` notional.

This is why the best profit improvement came from risk-budget tuning rather than
forcing lower signal thresholds.

## Backtest Window

Champion research window:

- symbol: `BTCUSDT`
- interval: `15m`
- start: `2026-04-27T00:00:00+00:00`
- end: `2026-05-01T00:00:00+00:00`
- instrument model: `futures`
- starting equity: `$10,000`

This is a very short window. The Sharpe ratios below are useful for comparing
variants inside the same experiment, but they should not be treated as stable
long-run estimates.

## Performance

Best Sharpe/profit variant:

| Run | Trades | Avg Notional | Net PnL | Return | Sharpe | Max DD | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `champion_risk0.010_notional20000` | 13 | `$20,000` | `$314.26` | `3.14%` | `31.00` | `0.45%` | `92.31%` |
| `champion_risk0.0125_notional25000` | 13 | `$25,000` | `$392.83` | `3.93%` | `31.02` | `0.57%` | `92.31%` |
| `champion_risk0.015_notional30000` | 13 | `$30,000` | `$471.40` | `4.71%` | `31.05` | `0.68%` | `92.31%` |

The current branch recommendation is the `$30,000` notional futures research
variant because it materially improves PnL without harming the tested Sharpe or
hit rate in this window.

Best "more trades" variant:

| Run | Trades | Avg Notional | Net PnL | Return | Sharpe | Max DD | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sel_g35s25_tpd5_risk0.010_notional20000` | 15 | `$20,000` | `$330.00` | `3.30%` | `19.10` | `0.90%` | `93.33%` |
| `sel_g35s25_tpd5_risk0.0125_notional25000` | 11 | `$25,000` | `$262.50` | `2.63%` | `10.36` | `1.13%` | `90.91%` |
| `sel_g35s25_tpd5_risk0.015_notional30000` | 11 | `$30,000` | `$315.00` | `3.15%` | `10.38` | `1.35%` | `90.91%` |

The more-active variant takes more trades, but its risk-adjusted performance is
worse. The better conclusion is not "trade as often as possible"; it is "use a
quota to avoid starvation, but keep selection quality high."

Other tested paths:

- 5-minute strategy variants were negative with the current sparse
  prediction-market and quote history.
- Long/short variants were negative because the available short candidates did
  not have reliable edge.
- Predicted-return and expected-utility ranking underperformed probability
  ranking.
- Adding more model types did not beat the compact `histgb,lightgbm,xgboost`
  boosted-tree stack.
- The direct quota-HPO profile underperformed the normal wide HPO profile.

## Reproduction Command

The current champion can be reproduced with:

```bash
.venv/bin/python -m zeroalpha.cli backtest ml \
  --config configs/paper.example.toml \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2026-04-27T00:00:00+00:00 \
  --end 2026-05-01T00:00:00+00:00 \
  --instrument-model futures \
  --assumed-spread-bps 4 \
  --tier-rate 0.0004 \
  --base-slippage-bps 1 \
  --safety-margin-bps 2 \
  --candidate-mode dense \
  --side-mode long \
  --min-history-bars 96 \
  --max-holding-hours 2 \
  --net-profit-target 0.001 \
  --net-stop-loss 0.001 \
  --volatility-lookback-bars 96 \
  --minimum-gross-profit-bps 35 \
  --minimum-gross-stop-bps 30 \
  --models histgb,lightgbm,xgboost \
  --calibration-method sigmoid \
  --stacker weighted \
  --hpo \
  --hpo-profile wide \
  --hpo-trials 12 \
  --target-trades-per-day 4.5 \
  --target-frequency-mode quota \
  --selection-score probability \
  --min-signal-spacing-hours 0.75 \
  --research-gate \
  --allow-negative-ev-frequency-probe \
  --notional 30000 \
  --paper-max-notional 30000 \
  --risk-per-trade 0.015 \
  --max-open-positions 4 \
  --prediction-market-signals \
  --prediction-market-durations 5m,15m,30m,1h,2h,4h,24h \
  --prediction-market-max-markets 300 \
  --prediction-market-fidelity-minutes 5 \
  --output artifacts/backtests/crypto_futures_champion.json
```

## Current Recommendation

Use the champion as the futures research baseline:

- `15m` BTCUSDT futures
- long-only
- dense candidates
- `histgb,lightgbm,xgboost`
- sigmoid calibration
- weighted stacker
- wide HPO with 12 trials
- probability-ranked quota selection
- 4.5 trades/day target
- 0.75-hour spacing
- 35 bps gross target / 30 bps gross stop
- `$30,000` research notional
- `1.5%` futures risk budget

Use the 35/25 stop geometry only if the priority is slightly more activity at
the cost of lower Sharpe.

## Limitations And Next Work

The main limitation is data depth. Four days is not enough to trust the Sharpe,
especially with sparse prediction-market data. The next serious improvement is
not another threshold tweak; it is better market microstructure history.

Recommended next steps:

- Record Polymarket CLOB v2 BTC up/down books continuously across all durations.
- Record Kalshi BTC up/down orderbooks, trades, and candles continuously.
- Record IBKR top-of-book and, where available, market-depth data during the same
  windows.
- Re-run the 15-minute futures strategy on at least 30-90 calendar days.
- Re-test 5-minute horizons only after continuous 5-minute Polymarket/Kalshi
  coverage exists.
- Keep shorting disabled until short candidates show positive out-of-sample edge
  with stop-loss replay.
- Add a live/paper guardrail that rejects futures sizing unless stop distance,
  max loss, and max daily loss are all explicitly available.

