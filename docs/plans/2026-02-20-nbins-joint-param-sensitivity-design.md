# Continuous Binning Param Sensitivity — Joint `n_bins` Design

**Date:** 2026-02-20
**Status:** Approved

## Problem

`feature_research/continuous_binning/param_sensitivity.ipynb` currently fixes `N_BINS` to a single value (`config.binning_params.bin_counts[0]`). This prevents joint ranking across feature parameters and bin counts.

## Goal

Treat `n_bins` as an optimizable parameter in the same sensitivity grid and generate one unified report/ranking.

## Design

1. Replace scalar `N_BINS` with list `N_BINS_GRID` (defaulting to `config.binning_params.bin_counts`).
2. Keep feature extraction at bias-parameter combo granularity (cache-efficient).
3. In metric-grid cell, evaluate every `(bias_param_combo, n_bins)` pair.
4. Add `n_bins` to each output row and include it in report `param_names`.
5. Keep plotting/report API unchanged (single call to `generate_parameter_sensitivity_report`).

## Invariants

- No changes to control-file schemas or pipeline artifacts.
- Metric definition and stability logic remain unchanged.
- Existing behavior is preserved when `N_BINS_GRID` has one value.

## Out of Scope

- Rule-based notebook changes.
- New plotting primitives.
- Base model API changes.
