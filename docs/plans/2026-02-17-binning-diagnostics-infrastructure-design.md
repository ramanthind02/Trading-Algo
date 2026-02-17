# Binning Diagnostics Infrastructure Design

**Date:** 2026-02-17  
**Task:** `docs/kanban/to-do/feature_validator/binning/T005_binning_diagnostics_infrastructure.md`

## 1. Objective
Implement T005 binning diagnostics infrastructure with minimal-regression integration into the current `ContinuousBinningModel`/`BinningModelBase` contracts, while keeping existing model defaults unchanged for researcher experimentation.

## 2. Scope and Boundaries
In scope:
- New diagnostics module for binning success validation and region metadata extraction.
- Shape classification (`tail` vs `hump`) and feature coverage calculation.
- Integration exports under `feature_selection/validators/binning`.
- API docs update in canonical feature-selection API page.
- Focused model hardening only where diagnostics contract requires stricter invariants.

Out of scope:
- Plotting (T007), advanced shape/adjacency breakdown (T006), and full report assembly (T008).
- Threshold policy changes and default parameter tuning.

## 3. Chosen Approach
Adopt a diagnostics-first wrapper on top of existing fitted model state:
- Consume `bin_stats_`, `significant_regions_`, and `bin_edges_` from fitted models.
- Keep `ContinuousBinningModel` behavioral defaults as-is.
- Add narrow hardening checks to prevent invalid fitted-state structures from producing silent wrong diagnostics.

Rationale:
- Lowest blast radius across current base-model tests.
- Directly unlocks T006/T007/T008 dependencies.
- Avoids unnecessary refactor churn in `BinningModelBase`.

## 4. Architecture
Add package:
- `feature_selection/validators/binning/__init__.py`
- `feature_selection/validators/binning/diagnostics.py`

Core dataclasses:
- `BinningSuccessCriteria(metric_threshold, t_threshold, min_region_width)` (`frozen=True`)
- `RegionMetadata(start_bin, end_bin, bins, mean_sharpe, mean_t_stat, sample_count, feature_range)` (`frozen=True`)

Core functions:
- `validate_binning_success(model, criteria) -> bool`
- `extract_region_metadata(model) -> list[RegionMetadata]`
- `detect_region_shape(region, n_bins) -> Literal["tail", "hump"]`
- `calculate_coverage(regions, feature_data) -> float`

## 5. Data Flow
1. Validate fitted model contract (`is_fitted_`, required attrs present, consistent structures).
2. Extract region rows by mapping each region's bins into `bin_stats_`.
3. Aggregate per-region metrics:
   - `mean_sharpe`: mean of bin `sharpe`
   - `mean_t_stat`: mean absolute `t_stat`
   - `sample_count`: sum of bin `count`
   - `feature_range`: min/max over bin `feature_min`/`feature_max`
4. Validate success against criteria:
   - width constraint
   - metric threshold constraint per bin
   - t-stat threshold per bin
5. Compute coverage from union of `feature_range` intervals over non-null feature observations.

## 6. Validation Rules
- Success means at least one region passes all criteria.
- Region width: `len(region.bins) >= min_region_width`.
- Threshold compliance: all bins in region must satisfy metric and t-stat constraints.
- Shape detection:
  - `tail`: touches extreme bin (`start_bin == 0` or `end_bin == n_bins - 1`)
  - `hump`: otherwise
- Coverage always clamped to `[0.0, 100.0]`.

## 7. Error Handling
- Raise `ValueError` for unfitted or structurally invalid model state.
- `detect_region_shape` rejects invalid `n_bins` and inconsistent region bounds.
- `calculate_coverage` returns `0.0` for empty/NaN-only feature inputs.

## 8. Testing Strategy (TDD)
Add integration tests under `tests/integration/feature_validator/binning/`:
- success/failure criteria checks
- metadata extraction correctness
- tail/hump detection
- deterministic coverage computation

Add targeted model test(s) only if hardening touches `base_model.py`.

## 9. Documentation Impact
Update `docs/api/feature_selection.md` validator-stage section to include binning diagnostics API entrypoints and dataclasses, since `docs/api/feature_selection/validators.md` does not currently exist.

## 10. Risks and Mitigations
- Risk: malformed `significant_regions_` from future model changes.
  - Mitigation: strict structural validation and explicit exceptions.
- Risk: overlap double-counting in coverage.
  - Mitigation: interval-union logic before counting values.
- Risk: tests coupled to non-deterministic datasets.
  - Mitigation: synthetic deterministic fixtures.

## 11. Acceptance Mapping
This design directly maps to T005 interfaces, invariants, and acceptance tests, and is intentionally staged so T006/T007/T008 can consume stable diagnostics contracts without rework.
