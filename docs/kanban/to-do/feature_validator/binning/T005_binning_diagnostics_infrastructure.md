# T005 — Binning Diagnostics Infrastructure

## Goal
Build core infrastructure for binning success validation, including success criteria verification (region detection, width requirements, threshold compliance) and region metadata extraction from fitted ContinuousBinningModel instances.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 133-179: Phase 2 specification)
- `feature_selection/base_models/base_model.py` — BinningModelBase interface with fitted state
- `feature_selection/base_models/continuous_binning.py` — ContinuousBinningModel implementation

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
1. `pytest tests/integration/feature_validator/binning/test_success_criteria.py::test_validate_binning_success_with_valid_regions -q` — fitted RSI model with 2+ consecutive bins exceeding thresholds passes validation
2. `pytest tests/integration/feature_validator/binning/test_success_criteria.py::test_validate_binning_success_no_regions -q` — model with only isolated single-bin spikes fails validation
3. `pytest tests/integration/feature_validator/binning/test_region_metadata.py::test_extract_region_metadata_rsi -q` — extract RegionMetadata from fitted RSI model, verify mean_sharpe and feature_range accuracy
4. `pytest tests/integration/feature_validator/binning/test_shape_detection.py::test_detect_tail_vs_hump -q` — classify region at bin 0-2 as "tail", region at bin 5-7 (n_bins=15) as "hump"
5. `pytest tests/integration/feature_validator/binning/test_coverage.py::test_calculate_coverage_rsi -q` — RSI with lookback=14, TimeFrame.D, verify coverage matches expected percentage

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/binning/`
- [ ] BinningSuccessCriteria, RegionMetadata dataclasses implemented in `feature_selection/validators/binning/diagnostics.py`
- [ ] Success validation, metadata extraction, shape detection, coverage functions implemented
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/integration/feature_validator/binning/ -q` passes

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=15, default from spec)
- Shape detection enables researcher to distinguish tail effects (monotonic extremes) from hump patterns (non-monotonic optimal regions)
- Coverage metric helps identify features with narrow tradeable zones (potential overfitting) vs broad applicability
