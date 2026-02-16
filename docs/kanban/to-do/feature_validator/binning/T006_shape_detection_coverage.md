# T006 — Shape Detection and Coverage Analysis

## Goal
Implement detailed shape classification (tail vs hump patterns) and comprehensive coverage analysis to guide researcher interpretation of binning diagnostics, distinguishing monotonic tail effects from non-monotonic optimal regions and identifying narrow vs broad tradeable zones.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 165-167: shape detection and coverage specification)
- `feature_selection/validators/binning/diagnostics.py` — T005 infrastructure (RegionMetadata, detect_region_shape)
- `feature_selection/base_models/base_model.py` — bin_stats_ structure

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
1. `pytest tests/integration/feature_validator/binning/test_shape_classification.py::test_classify_long_tail -q` — RSI region at bins 0-2 with positive Sharpe classified as "long_tail"
2. `pytest tests/integration/feature_validator/binning/test_shape_classification.py::test_classify_short_hump -q` — region at bins 5-7 with negative Sharpe classified as "short_hump"
3. `pytest tests/integration/feature_validator/binning/test_multi_region_shapes.py::test_analyze_shapes_rsi -q` — RSI with multiple regions, verify shape count dictionary accuracy
4. `pytest tests/integration/feature_validator/binning/test_coverage_breakdown.py::test_region_coverage_rsi -q` — calculate per-region and cumulative coverage for RSI lookback=14
5. `pytest tests/integration/feature_validator/binning/test_adjacency.py::test_detect_gaps -q` — identify gaps between regions, verify isolation score calculation
6. `pytest tests/integration/feature_validator/binning/test_adjacency.py::test_connected_regions -q` — regions with small gaps (<=2 bins) marked as connected

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/binning/`
- [ ] ShapeClassification, RegionCoverage, AdjacencyAnalysis dataclasses implemented
- [ ] Shape classification, multi-region analysis, coverage breakdown, adjacency detection functions implemented
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/integration/feature_validator/binning/test_shape* tests/integration/feature_validator/binning/test_coverage* tests/integration/feature_validator/binning/test_adjacency* -q` passes

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=15)
- Long tail (low RSI bins): typical oversold pattern with positive Sharpe
- Short tail (high RSI bins): typical overbought pattern with negative Sharpe
- Hump patterns: non-monotonic optimal regions indicating mean-reversion zones
- Adjacency analysis helps distinguish fragmented (noisy) vs cohesive (robust) feature structure
- Coverage breakdown enables comparison of tradeable zone breadth across parameter combinations
