# Binning Architecture Hard Cutover Design

**Date:** 2026-02-13  
**Status:** Approved  
**Scope:** Binning/base-model layer hard cutover with schema break and global naming cutover

## Objective

Refactor the binning model layer to match the new feature-validation architecture:

- Continuous features use a region-aware quantile pipeline with adjusted-Sharpe position multipliers.
- Rule-based features use per-level (-1, 0, +1) statistics with adjusted-Sharpe multipliers.
- Base-model outputs are continuous multipliers, not legacy best-bin binary flags.
- Fitted-state persistence is schema-breaking and explicit for region/multiplier semantics.
- Naming is cut over globally to `continuous_binning` and `rule_based`.

## Decisions (Approved)

- Cutover mode: **hard cutover**
- Persistence migration: **schema-breaking now**
- Scope: **binning layer only** (validator package implemented separately)
- Refactor style: **in-place rewrite**
- Naming: normalize to `continuous_binning` and `rule_based`, applied **everywhere** (files/classes/config/schema/docs/tests)
- Process exception: write docs without commit in this workspace due unrelated active changes

## Runtime Architecture

### 1) In-place class rewrite

Rewrite existing class responsibilities in place:

- `BinningModelBase` (`feature_selection/base_models/base_model.py`)
- `ContinuousBinningModel` (renamed from quantile class)
- `RuleBasedModel` (renamed from rule-based class)

### 2) Fit pipeline behavior

`fit(feature_data, target_data, normalization_data=None)`:

1. Align indexes and drop invalid rows.
2. Build bins/levels:
   - `continuous_binning`: quantile bins with robust duplicate handling.
   - `rule_based`: levels `-1`, `0`, `+1`.
3. Compute per-bin/per-level stats:
   - `count`, `mean_return`, `volatility`, `sharpe`, `adjusted_sharpe`, `t_stat`, range metadata.
4. Detect contiguous significant regions (continuous only) using pre-specified `t_threshold` and `min_region_width`.
5. Apply direction-aware metric threshold filter (`long`, `short`, `long_short`).
6. Precompute position multipliers per bin/level using clipped adjusted Sharpe.

### 3) Predict pipeline behavior

`predict(feature_data, strategy, normalization_data=None, scaled=False)`:

1. Map feature values into fitted bins/levels.
2. Emit continuous position multipliers from fitted lookup tables.
3. Emit `0` for inactive bins/levels.
4. Apply optional post-scaling path when `scaled=True`.

### 4) Removed semantics

- Remove best-bin binary selection (`best_long_bin_`, `best_short_bin_`).
- Remove rule-model passthrough output behavior.
- Remove lazy-compatibility assumptions based on old fitted keys.

## Persistence: Schema-Breaking Fitted State

Per fitted base model, persist explicit v2 fields:

- `model_version`: `"binning_v2"`
- `model_type`: `"continuous_binning"` or `"rule_based"`
- `bin_edges`
- `bin_stats`
- `significant_regions`
- `active_bins_by_strategy`
- `position_multipliers_by_strategy`
- `fit_config`

Hard-removed fitted keys:

- `thresholds`
- `best_long_bin`
- `best_short_bin`

Old fitted artifacts are intentionally non-loadable; loaders fail with clear migration errors.

## Global Naming Cutover

### Files

- `feature_selection/base_models/quantile_binning.py` -> `feature_selection/base_models/continuous_binning.py`
- `feature_selection/base_models/rule_based_binning.py` -> `feature_selection/base_models/rule_based.py`

### Classes

- `QuantileBinningModel` -> `ContinuousBinningModel`
- `RuleBasedBinningModel` -> `RuleBasedModel`

### Config/Schema IDs

- `model_type: "continuous_binning"`
- `model_type: "rule_based"`

### Call sites

All imports, constructors, model factories, and persistence restoration paths are updated to new names only.

## Data Flow Contract

1. Bias node output per param variant -> per-model fit against shared vol-scaled target.
2. Each model independently computes and stores v2 bin/region state.
3. Ensemble/base-model layer averages member outputs (simple mean).
4. Permutation and walkforward layers consume continuous multipliers from unchanged fit/predict contract.

## Error Handling

Fail fast, explicit errors:

- Missing/misaligned target index.
- Continuous binning collapse below usable minimum.
- Rule-based unexpected domain values (unless explicitly allowed by config).
- Predict called before fit.
- Attempt to load old schema.

No silent fallback to legacy best-bin behavior.

## Testing Strategy

1. Unit tests:
   - Sharpe/adjusted Sharpe/t-stat calculations.
   - Region detection + min-width logic.
   - Direction-aware threshold filtering.
   - Multiplier clipping for long/short/long_short.
2. Persistence tests:
   - v2 schema save/load round-trip.
   - old-schema rejection path.
3. Integration tests:
   - ensemble fit/predict through renamed models and identifiers.
4. Initial targeted commands:
   - `source venv/bin/activate && pytest tests/test_ensemble_base_models.py -v`
   - `source venv/bin/activate && pytest tests/test_vault_system.py -v`

## Out of Scope

- Implementing new `feature_selection/validators/` package in this change.
- Backward compatibility shims for old names or old fitted schema.
- Strategy weighting changes (ensemble remains simple average).
