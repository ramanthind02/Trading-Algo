# T008 — Binning Report Generation

## Goal
Create BinningDiagnosticsReport dataclass and generation logic to aggregate all binning diagnostics (success criteria, region metadata, shape classifications, coverage analysis, diagnostic plots) into a structured report with pass/fail verdict for researcher-in-the-loop validation decisions.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` (lines 159-179: BinningDiagnosticsReport specification)
- `feature_selection/validators/binning/diagnostics.py` — T005 infrastructure
- `feature_selection/validators/binning/shape_analysis.py` — T006 shape and coverage
- `feature_selection/validators/binning/plots.py` — T007 diagnostic plots
- `feature_selection/validators/reports/base.py` — base report structure
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards

## Scope
In scope:
- BinningDiagnosticsReport dataclass: comprehensive container for all binning diagnostics
- Report generation function: orchestrates validation, metadata extraction, shape analysis, plotting
- Pass/fail verdict logic: based on success criteria from T005
- Failure mode detection: classify failure types (no regions, isolated spikes, insufficient edge)
- Report serialization: save to JSON for persistence and reproduction
- Report display: pretty-print summary to console

Out of scope:
- Interactive report viewers (static reports only)
- Automated threshold tuning (researcher pre-specifies thresholds)
- Cross-parameter comparison (handled in parameter sensitivity phase)
- Ensemble formation recommendations (researcher decision after reviewing reports)

## Interfaces (must match)
- Add: `feature_selection/validators/binning/report.py`
  - `BinningDiagnosticsReport` dataclass (frozen):
    - feature_column: str
    - parameter_combo: Dict[str, object] — e.g., {"lookback": 14}
    - success_verdict: bool — overall pass/fail
    - failure_mode: Optional[Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]]
    - criteria: BinningSuccessCriteria
    - regions: List[RegionMetadata]
    - shape_summary: Dict[str, int] — from T006 analyze_multi_region_shapes
    - coverage_breakdown: List[RegionCoverage] — from T006
    - total_coverage_pct: float
    - adjacency_analysis: AdjacencyAnalysis — from T006
    - diagnostic_plots: Dict[str, Figure] — keys: "heatmap", "boundaries", "multiplier_curve", "panel"
    - timestamp: str — ISO format generation time
  - `generate_binning_report(model: BinningModelBase, feature_data: pd.Series, criteria: BinningSuccessCriteria, strategy: str = "long") -> BinningDiagnosticsReport`
    - Orchestrates all diagnostic steps (validate, extract, classify, plot)
    - Returns comprehensive report
  - `detect_failure_mode(model: BinningModelBase, criteria: BinningSuccessCriteria) -> Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]`
    - "no_regions": significant_regions_ is empty
    - "isolated_spikes": regions exist but all have width < min_region_width
    - "insufficient_edge": regions exist but none exceed metric_threshold
    - "none": success (at least one valid region)
  - `save_report(report: BinningDiagnosticsReport, output_dir: str) -> str`
    - Saves report to JSON (metadata only, plots saved separately as PNG)
    - Returns path to saved report file
  - `display_report_summary(report: BinningDiagnosticsReport) -> None`
    - Pretty-prints key metrics to console (verdict, regions, coverage, failure mode)

- Modify: `feature_selection/validators/binning/__init__.py` — export report dataclass and functions
- Modify: `feature_selection/validators/reports/__init__.py` — add BinningDiagnosticsReport to report types

## Data Contracts
- BinningDiagnosticsReport (frozen dataclass):
  - feature_column: str
  - parameter_combo: Dict[str, object]
  - success_verdict: bool
  - failure_mode: Optional[Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]]
  - criteria: BinningSuccessCriteria
  - regions: List[RegionMetadata]
  - shape_summary: Dict[str, int]
  - coverage_breakdown: List[RegionCoverage]
  - total_coverage_pct: float
  - adjacency_analysis: AdjacencyAnalysis
  - diagnostic_plots: Dict[str, Figure]
  - timestamp: str

- Report serialization (JSON):
  - All fields except diagnostic_plots (plots saved as separate PNG files)
  - Plot paths stored in JSON (e.g., "heatmap": "plots/heatmap_rsi_D_lookback_14.png")

## Dependencies
- feature_selection/validators/binning/diagnostics.py (T005: validation, metadata)
- feature_selection/validators/binning/shape_analysis.py (T006: shape, coverage, adjacency)
- feature_selection/validators/binning/plots.py (T007: plot generation)
- feature_selection/base_models/base_model.py (BinningModelBase)
- feature_selection/validators/reports/base.py (report base structure)
- dataclasses (frozen=True)
- typing (Dict, List, Literal, Optional)
- json (serialization)
- pathlib (file path handling)
- datetime (timestamp generation)

## Invariants / Constraints
- Deterministic: same model + feature_data + criteria => identical report (except timestamp)
- Success verdict: True if and only if at least one valid region exists
- Failure mode: mutually exclusive categories (only one active at a time)
- Coverage sum: total_coverage_pct = sum of individual_coverage_pct across all regions
- Plot keys: must include "heatmap", "boundaries", "multiplier_curve", "panel"
- JSON serialization: all dataclass fields except Figure objects (plots saved separately)
- Timestamp format: ISO 8601 (YYYY-MM-DDTHH:MM:SS)

## Acceptance tests

**Unit tests** (location: `tests/validators/binning/`):
- `test_generate_report_success_verdict()` — synthetic BinningModelBase stub with valid regions; assert BinningDiagnosticsReport.success_verdict == True and failure_mode == "none"
- `test_generate_report_no_regions_verdict()` — stub with empty significant_regions_; assert success_verdict == False and failure_mode == "no_regions"
- `test_detect_failure_mode_isolated_spikes()` — stub with only single-bin spikes (width < min_region_width); assert failure_mode == "isolated_spikes"
- `test_detect_failure_mode_insufficient_edge()` — stub with regions present but all below metric_threshold; assert failure_mode == "insufficient_edge"
- `test_detect_failure_mode_none()` — stub with at least one valid region; assert failure_mode == "none"
- `test_report_dataclass_fields()` — construct BinningDiagnosticsReport with synthetic inputs; assert all required fields are present and types are correct
- `test_save_report_creates_json()` — call save_report with tempfile.TemporaryDirectory; assert JSON file exists at expected path
- `test_save_report_json_fields()` — load saved JSON; assert all non-Figure fields are present and values match original report
- `test_display_report_summary_no_exception()` — call display_report_summary with synthetic report; assert no exceptions raised and key terms ("verdict", "coverage", "regions") appear in stdout
- `test_report_parameter_combo_stored()` — generate report with {"lookback": 14}; assert parameter_combo in JSON matches input

**Integration tests:**
- Covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- Default config: RSI lookback 5, Ticker.ES, TimeFrame.D, date range 2020-01-01 to 2023-12-31
- Customizable: accepts bias_module, param_name, param_value, ticker, timeframe as parameters for researcher exploration
- Verifies: generate_binning_report returns BinningDiagnosticsReport with all fields populated, save_report creates JSON in temp directory, display_report_summary prints readable terminal output
- Cache policy: use existing cache (USE_CACHE=True); skip with message "Run CacheManager.populate_cache() first" if cache missing
- Researcher manual verification:
  - Terminal output from display_report_summary should show: feature name, success verdict, failure mode, number of regions, total coverage %, shape summary counts
  - Open saved JSON file and confirm all metadata fields are present and human-readable
  - Confirm plot PNG files are saved alongside JSON in the output directory
  - If SAVE_INTEGRATION_OUTPUTS env var set, report files are copied to tests/integration/outputs/ for persistent review

## Definition of done
- [ ] Unit tests added under `tests/validators/binning/`
- [ ] Integration test covered by `tests/integration/feature_validator/test_binning_diagnostics.py::test_binning_full_pipeline()`
- [ ] BinningDiagnosticsReport dataclass implemented in `feature_selection/validators/binning/report.py`
- [ ] Report generation, failure mode detection, save/load, display functions implemented
- [ ] JSON serialization preserves all metadata (plots saved separately)
- [ ] Docs updated under `docs/api/feature_selection/validators.md`
- [ ] `pytest tests/validators/binning/ -q` passes (unit tests)
- [ ] `pytest tests/integration/feature_validator/test_binning_diagnostics.py -q` passes (integration test)

## Notes
- Test with RSI continuous feature: lookback=[2,3,4,5,6,7,8,9,10], TimeFrame.D
- Use quantile binning (n_bins=15)
- Report generation flow:
  1. Validate success using T005 validate_binning_success
  2. Extract regions using T005 extract_region_metadata
  3. Classify shapes using T006 classify_region_shape
  4. Calculate coverage using T006 calculate_region_coverage_breakdown
  5. Analyze adjacency using T006 detect_region_adjacency
  6. Generate plots using T007 plot functions
  7. Detect failure mode using detect_failure_mode
  8. Assemble BinningDiagnosticsReport
- Researcher workflow:
  1. Generate report per parameter combination
  2. Review diagnostic plots (heatmap shows significance, boundaries show coverage)
  3. Check success_verdict and failure_mode
  4. Compare reports across parameter combinations
  5. Select parameter combos with success_verdict=True for further validation phases
- Failure modes guide researcher action:
  - "no_regions": feature may lack structure, consider different feature or thresholds
  - "isolated_spikes": likely noise, increase min_region_width or reject feature
  - "insufficient_edge": regions exist but too weak, adjust metric_threshold or reject
- Save directory structure: `output_dir/{feature_column}/binning_report_{param_hash}.json`
