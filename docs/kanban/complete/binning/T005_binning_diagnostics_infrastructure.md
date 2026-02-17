# T005 — Binning Diagnostics Infrastructure

## Goal
Build core infrastructure for binning success validation, including success criteria verification (region detection, width requirements, threshold compliance) and region metadata extraction from fitted ContinuousBinningModel instances.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 133-179: Phase 2 specification)
- `feature_selection/base_models/base_model.py` — BinningModelBase interface with fitted state
- `feature_selection/base_models/continuous_binning.py` — ContinuousBinningModel implementation
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards

## Scope
In scope:
- BinningSuccessCriteria dataclass: encapsulates metric_threshold, t_threshold, min_region_width
- Success validation logic: verify fitted model meets criteria (at least one valid region exists)
- Region metadata extraction: parse significant_regions_, bin_stats_ from fitted model
- Shape detection interface: classify regions as "tail" (touches extreme bin) vs "hump" (surrounded by neutral bins)
- Coverage calculation: percentage of feature distribution covered by tradeable regions

Out of scope:
- Visualization (T007: binning diagnostic plots)
- Full report generation (T008: BinningDiagnosticsReport dataclass)
- Parameter sensitivity analysis (separate phase)

## Interfaces (must match)
- Add: `feature_selection/validators/binning/diagnostics.py`
  - `BinningSuccessCriteria` dataclass (frozen, with metric_threshold, t_threshold, min_region_width)
  - `validate_binning_success(model: BinningModelBase, criteria: BinningSuccessCriteria) -> bool`
    - Returns True if model.significant_regions_ contains at least one valid region
    - Validates region width >= criteria.min_region_width
    - Validates bins in region exceed metric_threshold and |t_stat| >= t_threshold
  - `extract_region_metadata(model: BinningModelBase) -> List[RegionMetadata]`
    - Returns list of RegionMetadata dataclasses (frozen) with:
      - start_bin, end_bin, bins, mean_sharpe, mean_t_stat, sample_count, feature_range
  - `detect_region_shape(region: RegionMetadata, n_bins: int) -> Literal["tail", "hump"]`
    - "tail": region.start_bin == 0 or region.end_bin == n_bins - 1
    - "hump": surrounded by neutral bins (not touching extremes)
  - `calculate_coverage(regions: List[RegionMetadata], feature_data: pd.Series) -> float`
    - Returns percentage [0.0, 100.0] of feature values falling in tradeable regions
    - Uses feature_range from RegionMetadata to determine inclusion

- Modify: `feature_selection/validators/binning/__init__.py` — export new functions and dataclasses

## Data Contracts
- RegionMetadata (frozen dataclass):
  - start_bin: int — first bin in region
  - end_bin: int — last bin in region
  - bins: List[int] — all bins in region (consecutive)
  - mean_sharpe: float — average Sharpe across region bins
  - mean_t_stat: float — average |t_stat| across region bins
  - sample_count: int — total observations in region
  - feature_range: Tuple[float, float] — (min, max) feature values covered

- Input: fitted ContinuousBinningModel with:
  - bin_stats_: Dict[int, Dict[str, float]] — per-bin statistics
  - significant_regions_: List[Dict[str, object]] — detected regions
  - bin_edges_: List[float] — bin boundaries

## Dependencies
- feature_selection/base_models/base_model.py (BinningModelBase)
- feature_selection/base_models/continuous_binning.py (ContinuousBinningModel)
- dataclasses (frozen=True)
- typing (List, Tuple, Literal)
- pandas (for coverage calculation)

## Invariants / Constraints
- Deterministic: same fitted model + criteria => same validation result
- Coverage must be in [0.0, 100.0] range
- Region width validation: len(region.bins) >= min_region_width
- Threshold compliance: all bins in valid region must exceed both metric_threshold and t_threshold
- Shape detection: mutually exclusive (tail XOR hump)

## Acceptance tests

**Unit tests** (location: `tests/validators/binning/`):
- `test_validate_binning_success_valid_regions()` — synthetic BinningModelBase stub with 2+ consecutive bins exceeding thresholds; assert returns True
- `test_validate_binning_success_no_regions()` — stub with empty significant_regions_; assert returns False
- `test_validate_binning_success_isolated_spikes()` — stub with single-bin spikes only (width < min_region_width); assert returns False
- `test_extract_region_metadata_structure()` — synthetic bin_stats_ and significant_regions_ with known values; assert RegionMetadata fields match expected values
- `test_detect_region_shape_tail_low()` — region starting at bin 0; assert shape == "tail"
- `test_detect_region_shape_tail_high()` — region ending at bin n_bins-1; assert shape == "tail"
- `test_detect_region_shape_hump()` — region surrounded by neutral bins; assert shape == "hump"
- `test_calculate_coverage_known_range()` — synthetic feature_data Series with known distribution; handcrafted feature_range; assert coverage percentage matches arithmetic expectation
- `test_calculate_coverage_zero()` — feature_range outside all feature values; assert coverage == 0.0
- `test_calculate_coverage_full()` — feature_range spans all feature values; assert coverage == 100.0

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- Default config: RSI lookback 5, Ticker.ES, TimeFrame.D, date range 2020-01-01 to 2023-12-31
- Customizable: accepts bias_module, param_name, param_value, ticker, timeframe as parameters for researcher exploration
- Verifies: validate_binning_success returns bool, extract_region_metadata returns non-empty list with correct fields, detect_region_shape returns "tail" or "hump", calculate_coverage returns float in [0.0, 100.0]
- Cache policy: use existing cache (USE_CACHE=True); skip with message "Run CacheManager.populate_cache() first" if cache missing
- Researcher manual verification:
  - Inspect terminal output for per-bin Sharpe ratios and t-stats
  - Confirm region metadata (start_bin, end_bin, mean_sharpe, feature_range) looks plausible for RSI
  - Confirm coverage percentage is in a sensible range (e.g., 20-50% for typical RSI)

## Definition of done
- [ ] Unit tests added under `tests/validators/binning/`
- [ ] Integration test covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- [ ] BinningSuccessCriteria, RegionMetadata dataclasses implemented in `feature_selection/validators/binning/diagnostics.py`
- [ ] Success validation, metadata extraction, shape detection, coverage functions implemented
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/validators/binning/ -q` passes (unit tests)
- [ ] `pytest tests/integration/feature_validator/test_binning_diagnostics.py -q` passes (integration test)

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=5, default from spec)
- Shape detection enables researcher to distinguish tail effects (monotonic extremes) from hump patterns (non-monotonic optimal regions)
- Coverage metric helps identify features with narrow tradeable zones (potential overfitting) vs broad applicability
