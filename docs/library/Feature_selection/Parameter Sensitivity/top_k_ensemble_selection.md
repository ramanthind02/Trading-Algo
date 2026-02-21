# Enhanced Top-K Ensemble Selection (Current)

> **Status:** Active library specification
> **Date:** 2026-02-20
> **Extends:** `param_selection_rule.md` rank/cap stage

## Overview
The enhanced top-k selector now optimizes two objectives only:
- **Stability/quality** via fold-smoothed objective.
- **Diversity** via correlation-aware greedy selection.

`Robustness` block-CV scoring was removed for rule-based/parameterized signals because there is no model refit phase in this flow and the extra term added noise without useful discrimination.

## Inputs
- `qualifying_params`: params that already passed upstream quality/stability gates.
- `training_data`, `training_target`: fold training slice.
- `smoothed_objectives`: per-param smoothed metric values.
- Config knobs:
  - `top_k`
  - `trade_freq_min`
  - `diversity_weight` (default `0.20`)
  - `quality_exponent` (default `2.0`)

## Algorithm
1. Compute signal series for each parameter combo on training data.
2. Compute trade frequency and hard-filter `trade_freq < trade_freq_min`.
3. Normalize surviving `smoothed_objective` values to `[0, 1]`.
4. Compute quality score:
   - `quality(P) = normalized_smoothed(P) ** quality_exponent`
   - Exponent > 1 increases separation so stronger smoothed performers dominate.
5. Compute Pearson signal correlation matrix across surviving params.
6. Greedy selection:
   - Select highest quality first.
   - For each remaining candidate:
     - `adjusted(P) = quality(P) * (1 - diversity_weight * max_abs_corr(P, selected))`
   - Add highest adjusted candidate until `top_k` or exhaustion.

## Output Fields
Enhanced fold outputs expose:
- `trade_frequency`
- `quality_score`
- `selected_by_diversity`

`robustness_score` is intentionally removed.

## Reporting Contract
- `selected_params_detailed.csv` should represent selected ensemble members per fold.
- When enhanced selection is enabled, rows are keyed by `selected_by_diversity=True` (not only the single rank-1 feature).
- For continuous features, `bin_count` is treated as a parameter dimension and must appear in `param_label` for exact combo traceability.

## Pre-Committed Defaults
| Config key | Default | Purpose |
|---|---:|---|
| `trade_freq_min` | `0.05` | remove low-activity combos |
| `diversity_weight` | `0.20` | correlation penalty strength |
| `quality_exponent` | `2.0` | emphasize high smoothed quality |
| `top_k` | `3` | ensemble size cap |

## Rationale
- Keeps selection focused on stable smoothed performance.
- Retains diversification without letting correlation penalty overrule quality.
- Produces auditable per-fold selected-combo reporting aligned with ensemble behavior.
