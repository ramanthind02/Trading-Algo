# Min-step parameter perturbation

Axis-aligned ±min_step neighbours around the robustness-selected combination.

| Metric | Value |
|---|---:|
| Peak | 3.0000 |
| Median (neighbours) | 2.5000 |
| P10 | 2.5000 |
| P90 | 2.5000 |
| Optimism bias | +0.5000 |
| Stability ratio | 0.8333 |
| Floor | 2.0000 |
| Result | **PASS** |

Passes: median metric 2.50 clears the floor of 2.00 but shows optimism bias (+0.50, 17% of peak) or stability 0.83. Monitor parameter sensitivity before going live.

## Neighbour runs

| Role | Param | Direction | Metric |
|---|---|---|---:|
| neighbor | exit_bars | down | 2.5000 |
| neighbor | exit_bars | up | 2.5000 |
| neighbor | momentum_lookback | down | 2.5000 |
| neighbor | momentum_lookback | up | 2.5000 |
| neighbor | rsi_max | down | 2.5000 |
| neighbor | rsi_max | up | 2.5000 |
| neighbor | rsi_period | down | 2.5000 |
| neighbor | rsi_period | up | 2.5000 |

Total neighbour evaluations: 8