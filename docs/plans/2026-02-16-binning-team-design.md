# Binning Diagnostics Agent Team Design

> **Date:** 2026-02-16
> **Purpose:** Parallel agent team implementation of Feature Validator binning tasks (T006, T007, T008)
> **Approach:** Mock-Driven Parallel Development with Integration Phase
> **Status:** Design approved, ready for implementation

---

## Overview

Implement the remaining 3 binning diagnostics tasks using a team of 3 specialist agents working in parallel. T005 (Binning Diagnostics Infrastructure) is already complete. This design covers T006 (Shape Detection), T007 (Diagnostic Plots), and T008 (Report Generation).

**Key design decisions:**
- **Parallel development:** All 3 agents work simultaneously for maximum speed
- **Mock-driven:** Agents use mocks for cross-task dependencies during development
- **Clear interfaces:** Contract-first approach with well-defined dataclasses and function signatures
- **Integration phase:** Team lead wires up real dependencies and validates full pipeline after parallel development
- **Manual verification:** Integration test produces terminal output and plots for researcher inspection

---

## Team Structure

### Agent 1: `shape-analyzer` (T006)

**Responsibility:** Enhanced shape detection and coverage analysis

**Deliverables:**
- `feature_selection/validators/binning/shape_analysis.py`
- Dataclasses:
  - `ShapeClassification` - directional shape classification (long_tail, short_tail, long_hump, short_hump)
  - `RegionCoverage` - per-region and cumulative coverage percentages
  - `AdjacencyAnalysis` - gap detection and connectivity metrics
- Functions:
  - `classify_region_shape(region, n_bins)` - classify single region shape
  - `analyze_multi_region_shapes(regions, n_bins)` - aggregate shape counts
  - `calculate_region_coverage_breakdown(regions, feature_data)` - per-region coverage
  - `detect_region_adjacency(regions)` - gap sizes and isolation score
- Unit tests: `tests/validators/binning/test_shape_analysis.py`

**Dependencies:** Uses existing `RegionMetadata` from T005

**Task spec:** `docs/kanban/to-do/feature_validator/binning/T006_shape_detection_coverage.md`

---

### Agent 2: `plot-generator` (T007)

**Responsibility:** Visualization functions for binning diagnostics

**Deliverables:**
- `feature_selection/validators/binning/plots.py`
- Functions:
  - `plot_bin_heatmap(model, metric, figsize, cmap)` - color-coded bin performance
  - `plot_region_boundaries(model, feature_data, regions, figsize)` - histogram with region overlays
  - `plot_position_multiplier_curve(model, strategy, figsize)` - step function showing position scaling
  - `create_diagnostic_panel(model, feature_data, regions, strategy, figsize)` - 3-panel combined view
- Unit tests: `tests/validators/binning/test_plots.py`

**Dependencies:**
- Uses existing `BinningModelBase`, `RegionMetadata` from T005
- Uses `ShapeClassification` from T006 (mocked during development)

**Task spec:** `docs/kanban/to-do/feature_validator/binning/T007_binning_diagnostic_plots.md`

---

### Agent 3: `report-builder` (T008)

**Responsibility:** Comprehensive report generation and orchestration

**Deliverables:**
- `feature_selection/validators/binning/report.py`
- Dataclass:
  - `BinningDiagnosticsReport` - comprehensive container for all diagnostics
- Functions:
  - `generate_binning_report(model, feature_data, criteria, strategy)` - orchestrate full report
  - `detect_failure_mode(model, criteria)` - classify failure types
  - `save_report(report, output_dir)` - JSON serialization
  - `display_report_summary(report)` - pretty-print terminal output
- Unit tests: `tests/validators/binning/test_report.py`

**Dependencies:**
- Uses T005 diagnostics (`validate_binning_success`, `extract_region_metadata`)
- Uses T006 shape analysis (mocked during development)
- Uses T007 plots (mocked during development)

**Task spec:** `docs/kanban/to-do/feature_validator/binning/T008_binning_report_generation.md`

---

## Interface Contracts

### Contract 1: Shape Analysis (T006)

```python
from dataclasses import dataclass
from typing import Literal
import pandas as pd

@dataclass(frozen=True)
class ShapeClassification:
    shape_type: Literal["long_tail", "short_tail", "long_hump", "short_hump"]
    is_monotonic: bool
    touches_extreme: bool
    direction: Literal["long", "short"]

@dataclass(frozen=True)
class RegionCoverage:
    region_id: int
    individual_coverage_pct: float  # [0.0, 100.0]
    cumulative_coverage_pct: float  # [0.0, 100.0]

@dataclass(frozen=True)
class AdjacencyAnalysis:
    gap_sizes: list[int]
    is_connected: bool
    isolation_score: float  # [0.0, 1.0]

def classify_region_shape(region: RegionMetadata, n_bins: int) -> ShapeClassification:
    """Classify region shape with directional info."""
    ...

def analyze_multi_region_shapes(
    regions: list[RegionMetadata], n_bins: int
) -> dict[str, int]:
    """Return count of each shape_type."""
    ...

def calculate_region_coverage_breakdown(
    regions: list[RegionMetadata], feature_data: pd.Series
) -> list[RegionCoverage]:
    """Calculate per-region and cumulative coverage percentages."""
    ...

def detect_region_adjacency(regions: list[RegionMetadata]) -> AdjacencyAnalysis:
    """Analyze gaps between regions and compute connectivity metrics."""
    ...
```

---

### Contract 2: Plots (T007)

```python
from matplotlib.figure import Figure
import pandas as pd

def plot_bin_heatmap(
    model: BinningModelBase,
    metric: str = "sharpe",
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "RdYlGn"
) -> Figure:
    """Generate horizontal bar chart heatmap color-coded by metric."""
    ...

def plot_region_boundaries(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    figsize: tuple[int, int] = (10, 6)
) -> Figure:
    """Histogram of feature_data with shaded tradeable regions."""
    ...

def plot_position_multiplier_curve(
    model: BinningModelBase,
    strategy: str = "long",
    figsize: tuple[int, int] = (10, 6)
) -> Figure:
    """Step function showing position multipliers across feature range."""
    ...

def create_diagnostic_panel(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    strategy: str = "long",
    figsize: tuple[int, int] = (18, 12)
) -> Figure:
    """Combined 3-panel figure (heatmap, boundaries, multiplier curve)."""
    ...
```

---

### Contract 3: Report (T008)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class BinningDiagnosticsReport:
    feature_column: str
    parameter_combo: dict[str, object]
    success_verdict: bool
    failure_mode: Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"] | None
    criteria: BinningSuccessCriteria
    regions: list[RegionMetadata]
    shape_summary: dict[str, int]
    coverage_breakdown: list[RegionCoverage]
    total_coverage_pct: float
    adjacency_analysis: AdjacencyAnalysis
    diagnostic_plots: dict[str, Figure]
    timestamp: str

def generate_binning_report(
    model: BinningModelBase,
    feature_data: pd.Series,
    criteria: BinningSuccessCriteria,
    strategy: str = "long"
) -> BinningDiagnosticsReport:
    """Orchestrate all diagnostic steps and return comprehensive report."""
    ...

def detect_failure_mode(
    model: BinningModelBase, criteria: BinningSuccessCriteria
) -> Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]:
    """Classify binning failure type."""
    ...

def save_report(report: BinningDiagnosticsReport, output_dir: str) -> str:
    """Save report to JSON (metadata) and PNG (plots)."""
    ...

def display_report_summary(report: BinningDiagnosticsReport) -> None:
    """Pretty-print key metrics to console."""
    ...
```

---

## Coordination Strategy

### Phase 1: Parallel Development

**Timeline:**
- All 3 agents spawn simultaneously
- Each agent works independently on implementation + unit tests
- Agents create temporary mocks for dependencies they don't own
- No inter-agent communication during Phase 1

**Mock Strategy:**

**`plot-generator` mocks:**
```python
# In tests/validators/binning/test_plots.py
def _mock_shape_classification() -> ShapeClassification:
    return ShapeClassification(
        shape_type="long_tail",
        is_monotonic=True,
        touches_extreme=True,
        direction="long"
    )
```

**`report-builder` mocks:**
```python
# In tests/validators/binning/test_report.py
def _mock_shape_summary() -> dict[str, int]:
    return {"long_tail": 1, "short_tail": 1}

def _mock_coverage_breakdown() -> list[RegionCoverage]:
    return [
        RegionCoverage(region_id=0, individual_coverage_pct=20.0, cumulative_coverage_pct=20.0),
        RegionCoverage(region_id=1, individual_coverage_pct=15.0, cumulative_coverage_pct=35.0),
    ]

def _mock_adjacency_analysis() -> AdjacencyAnalysis:
    return AdjacencyAnalysis(gap_sizes=[5, 3], is_connected=False, isolation_score=0.25)

def _mock_diagnostic_plots() -> dict[str, Figure]:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    return {"heatmap": fig, "boundaries": fig, "multiplier_curve": fig, "panel": fig}
```

**Communication:**
- Agents send completion message to team lead when Phase 1 done
- Team lead tracks: 3/3 agents must finish before Phase 2

---

### Phase 2: Integration

**Team lead orchestrates after all 3 agents complete Phase 1:**

1. **Remove mocks** - Delete all temporary mock functions from test files
2. **Wire up dependencies** - Update imports to use real T006/T007 code
3. **Create integration test** - `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
4. **Run integration test** - Validate full pipeline T005 → T006 → T007 → T008
5. **Update exports** - Add new functions/dataclasses to `__init__.py` files
6. **Verify manual inspection** - Check terminal output and saved plots

**Integration test spec:**
- **Location:** `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
- **Data:** RSI lookback=5, tickers=[ES, NQ, YM, RTY], date range 2000-2024
- **Cache policy:** Use existing cache, skip if missing
- **Output directory:** `tests/integration/outputs/binning/rsi_lookback_5/` (gitignored)
- **Artifacts:**
  - `binning_heatmap.png`
  - `region_boundaries.png`
  - `multiplier_curve.png`
  - `diagnostic_panel.png`
  - `report.json`

**Success criteria:**
- Integration test passes
- Terminal output shows readable summary
- Plots are saved and visually correct
- No import errors, type hints pass mypy

---

## Testing Strategy

### Unit Tests (Per Agent)

**`shape-analyzer` (8 tests minimum):**
- `test_classify_long_tail()` - bins 0-2, positive Sharpe → "long_tail"
- `test_classify_short_tail()` - bins 13-14, negative Sharpe → "short_tail"
- `test_classify_long_hump()` - bins 5-7, positive Sharpe → "long_hump"
- `test_classify_short_hump()` - bins 5-7, negative Sharpe → "short_hump"
- `test_analyze_multi_region_shapes_counts()` - verify shape counts
- `test_calculate_region_coverage_breakdown()` - verify coverage sums
- `test_detect_region_adjacency_gaps()` - verify gap calculations
- `test_detect_region_adjacency_connected()` - verify connectivity flag

**`plot-generator` (6 tests minimum):**
- `test_plot_bin_heatmap_returns_figure()` - returns matplotlib Figure
- `test_plot_bin_heatmap_metric_options()` - test all metrics ("sharpe", "t_stat", etc.)
- `test_plot_region_boundaries_returns_figure()` - returns Figure with regions
- `test_plot_position_multiplier_curve_returns_figure()` - returns step plot
- `test_create_diagnostic_panel_subplot_count()` - verify 3 subplots
- `test_save_plots_to_file()` - verify PNG creation

**`report-builder` (6 tests minimum):**
- `test_generate_report_success_verdict()` - valid regions → success=True
- `test_generate_report_no_regions_verdict()` - empty regions → failure_mode="no_regions"
- `test_detect_failure_mode_isolated_spikes()` - single-bin spikes → "isolated_spikes"
- `test_detect_failure_mode_insufficient_edge()` - weak metrics → "insufficient_edge"
- `test_save_report_creates_json()` - JSON file created
- `test_display_report_summary()` - terminal output readable

---

### Integration Test (Team Lead)

**Test:** `test_binning_full_pipeline_integration()`

**Steps:**
1. Load cached RSI features (lookback=5)
2. Fit `ContinuousBinningModel` (n_bins=15, strategy="long")
3. Run T005: `validate_binning_success()`, `extract_region_metadata()`
4. Run T006: `classify_region_shape()`, `calculate_region_coverage_breakdown()`, `detect_region_adjacency()`
5. Run T007: `create_diagnostic_panel()` + individual plots
6. Run T008: `generate_binning_report()`, `save_report()`, `display_report_summary()`
7. Save all artifacts to `tests/integration/outputs/binning/rsi_lookback_5/`
8. Print terminal summary for manual verification

**Terminal output format:**
```
========================================================================
BINNING DIAGNOSTICS INTEGRATION TEST
========================================================================
Feature: rsi_signal_D_lookback_5
Tickers: [ES, NQ, YM, RTY]
Date range: 2000-01-01 -> 2024-12-31
Samples: XXXXX observations

Binning Results:
✓ Success verdict: True
✓ Failure mode: none
✓ Regions detected: 2

Region 0 (long_tail):
  - Bins: [0, 1, 2]
  - Mean Sharpe: 0.68
  - Mean t-stat: 3.2
  - Coverage: 18.5%

Region 1 (short_tail):
  - Bins: [13, 14]
  - Mean Sharpe: 0.52
  - Mean t-stat: 2.4
  - Coverage: 20.0%

Shape Summary:
  - long_tail: 1
  - short_tail: 1

Total Coverage: 38.5%
Adjacency: connected=False, isolation_score=0.42

Plots saved to: tests/integration/outputs/binning/rsi_lookback_5/
  ✓ binning_heatmap.png
  ✓ region_boundaries.png
  ✓ multiplier_curve.png
  ✓ diagnostic_panel.png

Report saved to: tests/integration/outputs/binning/rsi_lookback_5/report.json
========================================================================
```

---

## Acceptance Criteria

**Per-agent deliverables:**
- All unit tests pass (pytest)
- Code follows project conventions (frozen dataclasses, type hints, functional style)
- Mocks are clearly marked as temporary
- Functions match interface contracts exactly

**Integration deliverables:**
- Integration test passes
- Terminal output is readable and informative
- All 4 plots are saved correctly
- JSON report contains all metadata fields
- Plots are visually correct (manual inspection by researcher)
- Documentation updated in `docs/api/feature_selection/validators.md`

**Final validation:**
- `pytest tests/validators/binning/ -q` passes (all unit tests)
- `pytest tests/integration/feature_validator/binning/ -q` passes (integration test)
- `mypy feature_selection/validators/binning/ --strict` passes (type checking)
- Researcher confirms plots and terminal output look correct

---

## References

**Task specifications:**
- T005: `docs/kanban/to-do/feature_validator/binning/T005_binning_diagnostics_infrastructure.md` (complete)
- T006: `docs/kanban/to-do/feature_validator/binning/T006_shape_detection_coverage.md`
- T007: `docs/kanban/to-do/feature_validator/binning/T007_binning_diagnostic_plots.md`
- T008: `docs/kanban/to-do/feature_validator/binning/T008_binning_report_generation.md`

**Design documents:**
- Feature Validator spec: `docs/library/Feature_selection/feature_validator.md`
- Integration testing spec: `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md`
- Continuous binning spec: `docs/library/Feature_selection/Features/Continuous_binning.md`

**Existing code:**
- T005 implementation: `feature_selection/validators/binning/diagnostics.py`
- T005 integration test: `tests/integration/feature_validator/binning/test_binning_diagnostics_pipeline_integration.py`
- Base model interface: `feature_selection/base_models/base_model.py`
- Continuous binning model: `feature_selection/base_models/continuous_binning.py`

---

**End of design document.**
