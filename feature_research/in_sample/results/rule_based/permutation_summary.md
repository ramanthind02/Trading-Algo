# In-Sample Permutation Summary

**Feature:** seasonalindiceseof_signal_D_smaPeriod_200  |  **Type:** rule_based

## Funnel (vector shuffle → pipeline permutation)

| Stage | Tested | Passed |
|-------|--------|--------|
| Stage 1 (vector shuffle) | 1 | 0 |
| Stage 2 (pipeline/candle) | 0 | 0 |

**Computational savings (Stage 1 gate):** 50.0%

## Per-combo metrics and p-values

| param_combo | S1 metric | S1 p-val | S1 pass | S2 metric | S2 p-val | S2 pass |
|-------------|-----------|----------|--------|-----------|----------|--------|
|  | 1.4380 | 0.3465 | ✗ | — | — | — |

Full data: `permutation_summary.csv`
