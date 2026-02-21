# Enhanced Top-K Ensemble Selection (Smoothed-Only)

> **Status:** Active library specification
> **Date:** 2026-02-20

## Overview
Top-k walkforward selection now uses a single ranking driver:
- **Smoothed objective** (after trade-frequency filtering).

Diversification-aware correlation penalties are removed for now.

## Inputs
- `qualifying_params`
- `training_data`, `training_target`
- `smoothed_objectives`
- Config:
  - `top_k`
  - `trade_freq_min`

## Algorithm
1. Evaluate each parameter combo signal on fold training data.
2. Compute `trade_frequency` and hard-filter combos below `trade_freq_min`.
3. Rank survivors by `smoothed_objective` descending (deterministic tie-break on `param_label` ascending).
4. Select top `k` labels.

## Output Fields
Enhanced fold outputs include:
- `trade_frequency`
- `selected_in_top_k`

`selected_by_diversity` and diversity/correlation scoring are removed.

## Reporting Contract
- `selected_params_detailed.csv` includes selected ensemble-member rows per fold.
- When enhanced selection is enabled, row selection uses `selected_in_top_k=True`.
- For continuous features, `bin_count` is treated as a parameter dimension and included in `param_label`.
