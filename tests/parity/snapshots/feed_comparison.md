# Feed comparison: futures vs cfd

Snapshots: `baseline_futures_vectorized` vs `cfd_vectorized`. Similarity gate for the futures->CFD migration (deliberate numbers change).


## Phase: test

| metric | futures | cfd | delta |
|---|---:|---:|---:|
| strategy.sharpe | 1.0136 | 1.7797 | +0.7661 |
| strategy.total_return | 0.1299 | 0.1952 | +0.0653 |
| strategy.max_drawdown | -0.0465 | -0.0380 | +0.0084 |
| strategy.calmar | 0.8003 | 1.3971 | +0.5968 |
| strategy.sortino | 1.4354 | 2.6350 | +1.1995 |

**Combined daily-return correlation (futures vs cfd)**: `0.8940` over 843 overlapping days (futures n=843, cfd n=868).

| ensemble | futures Sharpe | cfd Sharpe | delta |
|---|---:|---:|---:|
| algomatic_momentum_signal_long_defaults_long | 0.182 | 0.329 | +0.147 |
| buy_hold_long | 5.728 | 7.413 | +1.685 |
| calendar_ensemble_long | 0.359 | 0.466 | +0.107 |
| double7s_long | 0.323 | 0.786 | +0.463 |
| mr_indices_long | 0.317 | 0.483 | +0.166 |
| mr_indices_long_long | 0.752 | 1.229 | +0.477 |
| rebalancing_es_tlt_long | 0.988 | 1.051 | +0.063 |
| rebalancing_es_tlt_long_short | 0.792 | 0.661 | -0.131 |
| regime_lrsi_signal_long_short | 0.665 | 0.788 | +0.124 |
| robust_trend_breakout_cl_long_short | -0.451 | 0.367 | +0.818 |
| robust_trend_breakout_gc_long | 1.053 | 1.785 | +0.732 |
| sma_regime_long_short_long_short | 0.313 | 0.968 | +0.655 |