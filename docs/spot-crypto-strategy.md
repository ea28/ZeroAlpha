# Spot Crypto Strategy Write-Up

This is the main-branch strategy note for ZeroAlpha. The main branch targets
IBKR BTC/USD spot crypto in paper mode, long/flat only. Crypto-futures research
is intentionally kept off this branch as a separate research line.

This document is not financial advice and does not approve live trading. The
current spot results are weak after full IBKR crypto costs, so the branch should
be treated as a safer execution baseline and research harness rather than a
profitable production strategy.

## Instrument And Execution Model

The main branch trades spot BTC/USD through IBKR-supported crypto venues:

- contract security type: `CRYPTO`
- symbol: `BTC`
- currency: `USD`
- exchange candidates: `PAXOS`, `ZEROHASH`
- instrument model: `spot_crypto`
- side mode: long-only
- maximum open positions: `1`
- shorting: disabled

Spot crypto is not modeled as a margin or futures instrument. The backtester caps
notional by account equity and spot risk limits, and the live/paper safety checks
continue to reject spot short exposure.

## IBKR Commission Model

The cost model uses the published IBKR crypto schedule:

- `0.18% * Trade Value` for monthly trade value up to `$100,000`
- minimum commission: `$1.75` per order
- maximum commission: `1%` of trade value

For a `$10,000` spot BTC trade, this is about `$18` per side, or 36 bps round
trip before spread, slippage, and safety margin. With the research assumptions
used below, the all-in round-trip cost is about 48 bps:

- round-trip commission: 36 bps
- assumed spread: 4 bps
- slippage model: 6 bps
- safety margin: 2 bps

This cost floor is why the futures strategy did not transfer directly to spot.
The futures-style 35 bps target is not viable for IBKR spot BTC because it is
below estimated round-trip cost.

## Stop-Loss Handling

The branch now supports native IBKR-style attached stop-loss orders in the order
intent layer:

- parent entry: `LMT`
- take-profit child: `LMT`
- stop-loss child: `STP`
- optional stop-limit loss child: `STP LMT`
- parent and take-profit transmit flags are false
- final stop child transmit flag is true

This follows the TWS API bracket-order pattern where child orders are attached
with `parentId`, held until the parent fills, and transmitted atomically by the
last child order. The stop-limit option exists for research/paper experiments,
but a plain stop remains the safer default when exit certainty matters more than
limit-price control.

## Holding Horizon

The strategy now supports second-level holding horizons through
`--max-holding-seconds`. There is no artificial minimum beyond a positive value,
so a one-second horizon can be represented.

Important limitation: a one-second horizon is only meaningful with one-second,
tick, or order-book replay data. If the backtest uses 5-minute or 15-minute bars,
sub-bar exits cannot be proven. In that case the horizon support is plumbing for
future tick/L2 tests, not evidence that one-second live trading works.

## Data Used

Primary research data:

- Binance `BTCUSDT` spot bars.
- Cross-crypto context: `ETHUSDT`, `SOLUSDT`, and `ETHBTC`.
- Optional Binance USD-M futures reference bars for basis and futures-context
  features.
- Optional Polymarket CLOB v2 and Kalshi BTC up/down prediction-market features
  when cached data exists for the test window.
- Optional IBKR spot quote-recorder JSONL for bid/ask, spread, top-of-book size,
  and quote imbalance features.
- Optional IBKR futures quote-recorder JSONL for futures top-of-book and
  spot/futures basis-style features.

The current tested windows did not include live IBKR spot or IBKR futures quote
records. Those features are implemented but not responsible for the results
below.

## Model

The main spot model keeps the calibrated tree ensemble:

- `histgb`
- `lightgbm`
- `xgboost`
- sigmoid calibration
- weighted stacker
- optional wide HPO profile

Research on this branch did not justify moving to deep neural models. The
dataset is tabular, relatively small, noisy, and cost-dominated. Calibrated
boosted trees are still the most appropriate baseline.

## Tested Spot Results

All results below use the IBKR spot commission schedule, spot long-only
execution, max one open position, and `$10,000` starting equity.

Short four-day window with Polymarket/Kalshi features:

| Run | Trades | Net PnL | Return | Sharpe | Max DD | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h8_g120s90_tpd2` | 2 | `$125.61` | `1.26%` | `19.10` | `0.00%` | `100%` |
| `h8_g150s120_tpd2` | 2 | `$37.20` | `0.37%` | `19.10` | `0.16%` | `50%` |
| `h2_g120s90_tpd2` | 3 | `$4.27` | `0.04%` | `1.64` | `0.23%` | `66.7%` |

These are not robust enough to promote. The high Sharpe is a tiny-sample
artifact.

Longer April 2026 spot screen:

| Run | Trades | Net PnL | Return | Sharpe | Max DD | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `month_h8_g150s120_tpd0p5` | 6 | `$10.70` | `0.11%` | `0.42` | `1.48%` | `50%` |
| `month_h8_g150s120_tpd0p5_expected_utility` | 6 | `$8.23` | `0.08%` | `0.32` | `1.01%` | `66.7%` |
| `month_h8_g120s90_tpd0p5` | 6 | `-$89.99` | `-0.90%` | `-2.52` | `2.50%` | `50%` |
| `month_h8_g150s120_tpd0p5_wide_hpo` | 6 | `-$388.69` | `-3.89%` | `-11.78` | `4.20%` | `16.7%` |

The best longer-window spot result is only slightly positive before considering
operational risk. Wide HPO overfit and made the result worse.

5-minute / sub-hour tests:

| Run | Trades | Net PnL | Return | Sharpe | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5m_s900_g120s90_tpd4` | 3 | `-$106.91` | `-1.07%` | `-19.10` | `0%` |
| `5m_s1800_g120s90_tpd4` | 3 | `-$106.05` | `-1.06%` | `-37.47` | `0%` |
| `5m_s3600_g120s90_tpd4` | 3 | `-$151.42` | `-1.51%` | `-39.53` | `33.3%` |

The shorter holding tests confirm the intuition: more frequent spot trading is
not currently viable under IBKR's 18 bps per-side commission without much better
microstructure alpha.

## Current Main-Branch Recommendation

Do not promote the current spot model to live trading.

For continued paper research, the least-bad tested baseline is:

- interval: `15m`
- side: long-only
- horizon: `8h`
- gross target: `150` bps
- gross stop: `120` bps
- target frequency: `0.5` trades/day
- selection score: `probability` or `expected_value`
- models: `histgb,lightgbm,xgboost`
- calibration: `sigmoid`
- stacker: `weighted`
- HPO: off by default for this spot baseline

This baseline barely cleared costs over the April screen. The correct next move
is data improvement, not more aggressive thresholds.

## Reproduction Command

```bash
.venv/bin/python -m zeroalpha.cli backtest ml \
  --config configs/paper.example.toml \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2026-04-02T00:00:00+00:00 \
  --end 2026-05-01T00:00:00+00:00 \
  --instrument-model spot_crypto \
  --assumed-spread-bps 4 \
  --tier-rate 0.0018 \
  --minimum-commission 1.75 \
  --maximum-commission-rate 0.01 \
  --base-slippage-bps 1 \
  --safety-margin-bps 2 \
  --candidate-mode dense \
  --side-mode long \
  --min-history-bars 672 \
  --max-holding-hours 8 \
  --net-profit-target 0.009 \
  --net-stop-loss 0.016 \
  --volatility-lookback-bars 192 \
  --minimum-gross-profit-bps 150 \
  --minimum-gross-stop-bps 120 \
  --models histgb,lightgbm,xgboost \
  --calibration-method sigmoid \
  --stacker weighted \
  --target-trades-per-day 0.5 \
  --target-frequency-mode quota \
  --selection-score probability \
  --min-signal-spacing-hours 2.0 \
  --research-gate \
  --allow-negative-ev-frequency-probe \
  --notional 10000 \
  --paper-max-notional 10000 \
  --risk-per-trade 0.012 \
  --max-open-positions 1 \
  --output artifacts/backtests/spot_main_research/main_spot_baseline.json
```

## Next Work

The most important improvements are:

- Record IBKR spot quotes continuously during market hours and replay actual
  spreads/fills instead of assuming static spread.
- Record IBKR CME crypto futures or other futures-reference quotes as a separate
  feature stream when the account has permissions.
- Record Polymarket CLOB v2 and Kalshi orderbooks continuously, especially 5m
  and 15m BTC markets.
- Move sub-minute testing to tick or L2 replay before trusting one-second
  holding logic.
- Add paper-order smoke tests for native attached `STP` and `STP LMT` exits.
- Keep max one open spot position and long-only mode until real paper fills show
  that backtest assumptions are conservative.

