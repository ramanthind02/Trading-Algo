# Spread vs Market — NDX 2026 realism lane

- Window: `2026-02-24T00:00:00+00:00` .. `2026-03-03T00:00:00+00:00` (UTC)
- Lane: `NautilusPnLEngine`, `INTRADAY_OPEN_TO_CLOSE`, constant +1 long target
- Limit policies measured **pure-passive** (no CROSS_AFTER) to isolate
  maker spread capture vs fill risk.
- Spread = **liquidity-signed half-spread** (keyed on Nautilus
  `OrderFilled.liquidity_side`): **+ = captured** (MAKER), **- = paid**
  (TAKER). Robust to the MT5 bar-vs-quote price skew.

| Policy | Sessions | Entry fills | Rejects | Fill rate | MAKER | TAKER | Avg spread (px) | Avg spread (bps) | Total log-ret | Δ vs MARKET |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MARKET_ON_OPEN | 5 | 5 | 0 | 100.00% | 0 | 5 | -1.1200 | -0.450 | +0.019146 | +0.000000 |
| LIMIT_AT_TOUCH | 5 | 5 | 0 | 100.00% | 5 | 0 | +0.8200 | +0.329 | +0.024470 | +0.005324 |
| LIMIT_IMPROVE(1) | 5 | 5 | 0 | 100.00% | 5 | 0 | +0.8200 | +0.329 | +0.024470 | +0.005324 |
| LIMIT_IMPROVE(2) | 5 | 5 | 0 | 100.00% | 5 | 0 | +0.8200 | +0.329 | +0.024470 | +0.005324 |

## Reading the table

- **MARKET_ON_OPEN** crosses the spread: every entry is a TAKER fill with a
  *negative* signed spread (pays ~half-spread vs mid). Fill rate is 100%.
- **LIMIT_AT_TOUCH** rests at the near touch: fills are MAKER with a
  *non-negative* signed spread (captures ~half-spread) but only when the
  market trades to the touch — so fill rate < 100% (the rest is fill risk).
- **LIMIT_IMPROVE(n)** rests inside the spread: higher fill rate than
  at-touch, less spread captured per fill.
- The economics are directionally sane iff LIMIT_AT_TOUCH's avg signed
  spread `>= 0 >=` MARKET_ON_OPEN's (maker captures, taker pays).