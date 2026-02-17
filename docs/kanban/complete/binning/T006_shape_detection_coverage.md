# T006 — Shape Detection and Coverage Analysis

## Goal
Implement detailed shape classification (tail vs hump patterns) and comprehensive coverage analysis to guide researcher interpretation of binning diagnostics, distinguishing monotonic tail effects from non-monotonic optimal regions and identifying narrow vs broad tradeable zones.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 165-167: shape detection and coverage specification)
- `feature_selection/validators/binning/diagnostics.py` — T005 infrastructure (RegionMetadata, detect_region_shape)
- `feature_selection/base_models/base_model.py` — bin_stats_ structure
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards

## Scope
In scope:
- Enhanced shape detection: extend basic tail/hump to include directional classification (long_tail, short_tail, long_hump, short_hump)
- Multi-region shape summary: aggregate shape patterns across all detected regions
- Coverage breakdown: per-region coverage percentages and cumulative totals
- Region adjacency analysis: detect gaps between regions, identify isolated vs connected patterns
- Feature distribution diagnostics: quantile boundaries, bin occupancy counts

Out of scope:
- Visualization (T007: binning diagnostic plots)
- Report generation (T008: BinningDiagnosticsReport)
- Threshold tuning recommendations (researcher responsibility)

## Interfaces (must match)
- Add: `feature_selection/validators/binning/shape_analysis.py`
  - `ShapeClassification` dataclass (frozen):
    - shape_type: Literal["long_tail", "short_tail", "long_hump", "short_hump"]
    - is_monotonic: bool — True for tails, False for humps
    - touches_extreme: bool — True if region includes bin 0 or n_bins-1
    - direction: Literal["long", "short"] — based on mean_sharpe sign
  - `classify_region_shape(region: RegionMetadata, n_bins: int) -> ShapeClassification`
    - Extends T005 detect_region_shape with directional classification
    - Long: mean_sharpe > 0, Short: mean_sharpe < 0
    - Tail: touches extreme (start_bin == 0 or end_bin == n_bins - 1)
    - Hump: surrounded by neutral bins
  - `analyze_multi_region_shapes(regions: List[RegionMetadata], n_bins: int) -> Dict[str, int]`
    - Returns count of each shape_type (e.g., {"long_tail": 1, "short_hump": 2})
  - `calculate_region_coverage_breakdown(regions: List[RegionMetadata], feature_data: pd.Series) -> List[RegionCoverage]`
    - Returns list of RegionCoverage dataclasses with per-region and cumulative coverage
  - `detect_region_adjacency(regions: List[RegionMetadata]) -> AdjacencyAnalysis`
    - Identifies gaps between regions, classifies as isolated vs connected
    - Returns AdjacencyAnalysis dataclass with gap sizes and connectivity metrics

- Add: Supporting dataclasses in `feature_selection/validators/binning/shape_analysis.py`
  - `RegionCoverage` (frozen): region_id, individual_coverage_pct, cumulative_coverage_pct
  - `AdjacencyAnalysis` (frozen): gap_sizes (List[int]), is_connected (bool), isolation_score (float)

- Modify: `feature_selection/validators/binning/__init__.py` — export new functions and dataclasses

## Data Contracts
- ShapeClassification (frozen dataclass):
  - shape_type: Literal["long_tail", "short_tail", "long_hump", "short_hump"]
  - is_monotonic: bool
  - touches_extreme: bool
  - direction: Literal["long", "short"]

- RegionCoverage (frozen dataclass):
  - region_id: int — index in regions list
  - individual_coverage_pct: float — [0.0, 100.0]
  - cumulative_coverage_pct: float — [0.0, 100.0]

- AdjacencyAnalysis (frozen dataclass):
  - gap_sizes: List[int] — bin counts between consecutive regions
  - is_connected: bool — True if all gaps <= 2 bins
  - isolation_score: float — mean gap size / n_bins (higher = more isolated)

## Dependencies
- feature_selection/validators/binning/diagnostics.py (RegionMetadata from T005)
- feature_selection/base_models/base_model.py (BinningModelBase)
- dataclasses (frozen=True)
- typing (List, Dict, Literal)
- pandas (for coverage calculations)
- numpy (for quantile and statistical operations)

## Invariants / Constraints
- Deterministic: same regions + feature_data => same shape classifications and coverage
- Coverage sum: individual_coverage_pct values must sum to <= 100.0 (non-overlapping regions)
- Cumulative coverage: monotonically increasing across regions
- Gap sizes: non-negative integers (gap_size >= 0)
- Isolation score: [0.0, 1.0] range (normalized by n_bins)
- Shape classification: mutually exclusive categories

## Acceptance tests

**Unit tests** (location: `tests/validators/binning/`):
- `test_classify_long_tail()` — synthetic RegionMetadata at bins 0-2 with positive mean_sharpe; assert shape_type == "long_tail", touches_extreme == True, is_monotonic == True
- `test_classify_short_tail()` — synthetic RegionMetadata at bins 13-14 (n_bins=15) with negative mean_sharpe; assert shape_type == "short_tail"
- `test_classify_long_hump()` — synthetic RegionMetadata at bins 5-7 with positive mean_sharpe, not touching extremes; assert shape_type == "long_hump", touches_extreme == False
- `test_classify_short_hump()` — synthetic RegionMetadata at bins 5-7 with negative mean_sharpe; assert shape_type == "short_hump"
- `test_analyze_multi_region_shapes_counts()` — list of 3 synthetic regions with known types; assert returned dict counts are exact
- `test_calculate_region_coverage_breakdown_order()` — synthetic regions with known feature_ranges and feature_data; assert cumulative_coverage_pct is monotonically increasing
- `test_calculate_region_coverage_breakdown_sum()` — assert sum of individual_coverage_pct values equals total (non-overlapping regions)
- `test_detect_region_adjacency_gaps()` — synthetic regions with known bin gaps; assert gap_sizes list matches expected
- `test_detect_region_adjacency_connected()` — all gaps <= 2 bins; assert is_connected == True
- `test_detect_region_adjacency_isolated()` — large gaps between regions; assert is_connected == False, isolation_score > 0

**Note:** T006 is primarily a pure logic/algorithm task. The shape classification and adjacency algorithms can be fully verified with synthetic RegionMetadata inputs, so unit tests are the primary test vehicle.

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- Default config: RSI lookback 5, Ticker.ES, TimeFrame.D, date range 2020-01-01 to 2023-12-31
- Customizable: accepts bias_module, param_name, param_value, ticker, timeframe as parameters for researcher exploration
- Verifies: classify_region_shape returns valid ShapeClassification, coverage breakdown sums correctly, adjacency analysis identifies real gaps/connections from live data
- Cache policy: use existing cache (USE_CACHE=True); skip with message "Run CacheManager.populate_cache() first" if cache missing
- Researcher manual verification:
  - Inspect terminal output for per-region shape_type labels (expect "long_tail" or "short_tail" for RSI extremes)
  - Check cumulative coverage percentage looks plausible
  - Verify isolation_score reflects whether RSI regions are contiguous or fragmented

## Definition of done
- [ ] Unit tests added under `tests/validators/binning/`
- [ ] Integration test covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- [ ] ShapeClassification, RegionCoverage, AdjacencyAnalysis dataclasses implemented
- [ ] Shape classification, multi-region analysis, coverage breakdown, adjacency detection functions implemented
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/validators/binning/ -q` passes (unit tests)
- [ ] `pytest tests/integration/feature_validator/test_binning_diagnostics.py -q` passes (integration test)

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=15)
- Long tail (low RSI bins): typical oversold pattern with positive Sharpe
- Short tail (high RSI bins): typical overbought pattern with negative Sharpe
- Hump patterns: non-monotonic optimal regions indicating mean-reversion zones
- Adjacency analysis helps distinguish fragmented (noisy) vs cohesive (robust) feature structure
- Coverage breakdown enables comparison of tradeable zone breadth across parameter combinations
