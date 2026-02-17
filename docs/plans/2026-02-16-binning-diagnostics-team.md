# Binning Diagnostics Agent Team Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Feature Validator binning tasks T006-T008 using 3 parallel specialist agents with mock-driven development and integration phase.

**Architecture:** Three specialist agents (shape-analyzer, plot-generator, report-builder) work in parallel using mocks for cross-dependencies. Team lead coordinates integration phase after all agents complete, wiring up real dependencies and creating comprehensive integration test.

**Tech Stack:** Python 3.11+, pytest, matplotlib, pandas, dataclasses, typing

**Design Doc:** `docs/plans/2026-02-16-binning-team-design.md`

---

## Phase 1: Setup and Interface Contracts

### Task 1: Create Interface Contract Files

**Goal:** Define interface contracts that all agents will implement

**Files:**
- Create: `feature_selection/validators/binning/shape_analysis.py` (stub)
- Create: `feature_selection/validators/binning/plots.py` (stub)
- Create: `feature_selection/validators/binning/report.py` (stub)

**Step 1: Create shape_analysis.py with interface stubs**

```python
"""Enhanced shape detection and coverage analysis for binning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from feature_selection.validators.binning.diagnostics import RegionMetadata


@dataclass(frozen=True)
class ShapeClassification:
    """Directional shape classification for a binning region."""

    shape_type: Literal["long_tail", "short_tail", "long_hump", "short_hump"]
    is_monotonic: bool
    touches_extreme: bool
    direction: Literal["long", "short"]


@dataclass(frozen=True)
class RegionCoverage:
    """Per-region and cumulative coverage percentages."""

    region_id: int
    individual_coverage_pct: float  # [0.0, 100.0]
    cumulative_coverage_pct: float  # [0.0, 100.0]


@dataclass(frozen=True)
class AdjacencyAnalysis:
    """Gap detection and connectivity metrics between regions."""

    gap_sizes: list[int]
    is_connected: bool
    isolation_score: float  # [0.0, 1.0]


def classify_region_shape(region: RegionMetadata, n_bins: int) -> ShapeClassification:
    """Classify region shape with directional info.

    Args:
        region: Region metadata from fitted binning model
        n_bins: Total number of bins in the model

    Returns:
        Shape classification with type, monotonicity, extremes, direction
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def analyze_multi_region_shapes(
    regions: list[RegionMetadata], n_bins: int
) -> dict[str, int]:
    """Return count of each shape_type across all regions.

    Args:
        regions: List of region metadata
        n_bins: Total number of bins

    Returns:
        Dictionary mapping shape_type to count
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def calculate_region_coverage_breakdown(
    regions: list[RegionMetadata], feature_data: pd.Series
) -> list[RegionCoverage]:
    """Calculate per-region and cumulative coverage percentages.

    Args:
        regions: List of region metadata
        feature_data: Original feature values

    Returns:
        List of RegionCoverage with individual and cumulative percentages
    """
    raise NotImplementedError("Agent shape-analyzer will implement")


def detect_region_adjacency(regions: list[RegionMetadata]) -> AdjacencyAnalysis:
    """Analyze gaps between regions and compute connectivity metrics.

    Args:
        regions: List of region metadata (assumed sorted by start_bin)

    Returns:
        Adjacency analysis with gap sizes, connectivity flag, isolation score
    """
    raise NotImplementedError("Agent shape-analyzer will implement")
```

**Step 2: Create plots.py with interface stubs**

```python
"""Visualization functions for binning diagnostics."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validators.binning.diagnostics import RegionMetadata


def plot_bin_heatmap(
    model: BinningModelBase,
    metric: str = "sharpe",
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "RdYlGn",
) -> Figure:
    """Generate horizontal bar chart heatmap color-coded by metric.

    Args:
        model: Fitted binning model
        metric: Metric to visualize ("sharpe", "t_stat", "sample_count", "mean_return")
        figsize: Figure size in inches
        cmap: Matplotlib colormap name

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def plot_region_boundaries(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Histogram of feature_data with shaded tradeable regions.

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        regions: List of detected regions
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def plot_position_multiplier_curve(
    model: BinningModelBase,
    strategy: str = "long",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """Step function showing position multipliers across feature range.

    Args:
        model: Fitted binning model
        strategy: Trading strategy ("long" or "short")
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    raise NotImplementedError("Agent plot-generator will implement")


def create_diagnostic_panel(
    model: BinningModelBase,
    feature_data: pd.Series,
    regions: list[RegionMetadata],
    strategy: str = "long",
    figsize: tuple[int, int] = (18, 12),
) -> Figure:
    """Combined 3-panel figure (heatmap, boundaries, multiplier curve).

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        regions: List of detected regions
        strategy: Trading strategy ("long" or "short")
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure with 3 subplots
    """
    raise NotImplementedError("Agent plot-generator will implement")
```

**Step 3: Create report.py with interface stubs**

```python
"""Comprehensive report generation for binning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
)
from feature_selection.validators.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
)


@dataclass(frozen=True)
class BinningDiagnosticsReport:
    """Comprehensive container for all binning diagnostics."""

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
    strategy: str = "long",
) -> BinningDiagnosticsReport:
    """Orchestrate all diagnostic steps and return comprehensive report.

    Args:
        model: Fitted binning model
        feature_data: Original feature values
        criteria: Success criteria for validation
        strategy: Trading strategy ("long" or "short")

    Returns:
        Comprehensive diagnostics report
    """
    raise NotImplementedError("Agent report-builder will implement")


def detect_failure_mode(
    model: BinningModelBase, criteria: BinningSuccessCriteria
) -> Literal["no_regions", "isolated_spikes", "insufficient_edge", "none"]:
    """Classify binning failure type.

    Args:
        model: Fitted binning model
        criteria: Success criteria

    Returns:
        Failure mode classification
    """
    raise NotImplementedError("Agent report-builder will implement")


def save_report(report: BinningDiagnosticsReport, output_dir: str) -> str:
    """Save report to JSON (metadata) and PNG (plots).

    Args:
        report: Binning diagnostics report
        output_dir: Directory to save report and plots

    Returns:
        Path to saved JSON report file
    """
    raise NotImplementedError("Agent report-builder will implement")


def display_report_summary(report: BinningDiagnosticsReport) -> None:
    """Pretty-print key metrics to console.

    Args:
        report: Binning diagnostics report
    """
    raise NotImplementedError("Agent report-builder will implement")
```

**Step 4: Commit interface stubs**

```bash
git add feature_selection/validators/binning/shape_analysis.py
git add feature_selection/validators/binning/plots.py
git add feature_selection/validators/binning/report.py
git commit -m "feat: add interface stubs for binning diagnostics (T006-T008)

Define interface contracts for shape analysis, plotting, and reporting.
Agents will implement these during parallel development phase.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Create Output Directory Structure

**Goal:** Set up gitignored output directory for integration test artifacts

**Files:**
- Create: `tests/integration/outputs/.gitignore`
- Create: `tests/integration/outputs/binning/.gitkeep`

**Step 1: Create .gitignore for outputs directory**

```bash
mkdir -p tests/integration/outputs/binning
```

**Step 2: Create .gitignore file**

```
# Ignore all integration test outputs
*

# But keep the .gitignore file itself
!.gitignore

# And keep directory structure markers
!**/.gitkeep
```

**Step 3: Create .gitkeep to preserve directory**

```bash
touch tests/integration/outputs/binning/.gitkeep
```

**Step 4: Commit directory structure**

```bash
git add tests/integration/outputs/.gitignore
git add tests/integration/outputs/binning/.gitkeep
git commit -m "chore: add integration test output directory structure

Create gitignored directory for integration test plots and reports.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Spawn Parallel Agents

### Task 3: Spawn Agent 1 (shape-analyzer)

**Goal:** Launch shape-analyzer agent to implement T006

**Step 1: Create team with TeamCreate**

Use TeamCreate tool:
```json
{
  "team_name": "binning-diagnostics",
  "description": "Feature Validator binning tasks T006-T008 implementation",
  "agent_type": "team-lead"
}
```

**Step 2: Spawn shape-analyzer agent**

Use Task tool to spawn agent:
```json
{
  "subagent_type": "general-purpose",
  "team_name": "binning-diagnostics",
  "name": "shape-analyzer",
  "description": "Implement T006 shape analysis",
  "prompt": "You are the shape-analyzer agent responsible for implementing T006 (Shape Detection and Coverage Analysis).

**Your task:** Implement the 4 functions in feature_selection/validators/binning/shape_analysis.py according to the interface contracts and task specification.

**Task spec:** docs/kanban/to-do/feature_validator/binning/T006_shape_detection_coverage.md

**Interface contracts:** Already defined in shape_analysis.py - replace NotImplementedError with real implementations

**Key deliverables:**
1. Implement classify_region_shape() - directional shape classification
2. Implement analyze_multi_region_shapes() - aggregate shape counts
3. Implement calculate_region_coverage_breakdown() - per-region coverage
4. Implement detect_region_adjacency() - gap detection and connectivity
5. Write comprehensive unit tests in tests/validators/binning/test_shape_analysis.py
6. Update exports in feature_selection/validators/binning/__init__.py

**Dependencies:** Use existing RegionMetadata from T005 (already implemented)

**Testing approach:**
- Use synthetic RegionMetadata fixtures with known values
- Test all 4 shape types: long_tail, short_tail, long_hump, short_hump
- Test edge cases: empty regions, single region, overlapping ranges
- Verify coverage calculations sum correctly
- Verify adjacency gap calculations

**Acceptance criteria:**
- All unit tests pass (minimum 8 tests)
- Code follows project conventions (frozen dataclasses, type hints, functional style)
- Functions match interface contracts exactly
- No dependencies on T007 or T008 (those are handled by other agents)

**Use @superpowers:test-driven-development for implementation.**"
}
```

---

### Task 4: Spawn Agent 2 (plot-generator)

**Goal:** Launch plot-generator agent to implement T007

**Step 1: Spawn plot-generator agent**

Use Task tool to spawn agent:
```json
{
  "subagent_type": "general-purpose",
  "team_name": "binning-diagnostics",
  "name": "plot-generator",
  "description": "Implement T007 plotting functions",
  "prompt": "You are the plot-generator agent responsible for implementing T007 (Binning Diagnostic Plots).

**Your task:** Implement the 4 plotting functions in feature_selection/validators/binning/plots.py according to the interface contracts and task specification.

**Task spec:** docs/kanban/to-do/feature_validator/binning/T007_binning_diagnostic_plots.md

**Interface contracts:** Already defined in plots.py - replace NotImplementedError with real implementations

**Key deliverables:**
1. Implement plot_bin_heatmap() - color-coded bin performance
2. Implement plot_region_boundaries() - histogram with region overlays
3. Implement plot_position_multiplier_curve() - step function plot
4. Implement create_diagnostic_panel() - 3-panel combined view
5. Write comprehensive unit tests in tests/validators/binning/test_plots.py
6. Update exports in feature_selection/validators/binning/__init__.py

**Dependencies:**
- Use existing BinningModelBase, RegionMetadata from T005 (already implemented)
- T006 (ShapeClassification) is being implemented by another agent in parallel
- For unit tests, create mock ShapeClassification data as needed

**Mock strategy for unit tests:**
```python
def _mock_shape_classification() -> ShapeClassification:
    return ShapeClassification(
        shape_type=\"long_tail\",
        is_monotonic=True,
        touches_extreme=True,
        direction=\"long\"
    )
```

**Testing approach:**
- Use synthetic BinningModelBase stubs with known bin_stats_
- Test that each function returns matplotlib Figure objects
- Test metric options (sharpe, t_stat, sample_count, mean_return)
- Test that plots can be saved to tempfile
- Verify subplot counts and structure
- Don't worry about visual correctness (manual verification happens in integration test)

**Acceptance criteria:**
- All unit tests pass (minimum 6 tests)
- Code follows project conventions (type hints, clear function signatures)
- Functions match interface contracts exactly
- All plots use matplotlib (no interactive plotly/bokeh)
- Mocks are clearly marked as temporary

**Use @superpowers:test-driven-development for implementation.**"
}
```

---

### Task 5: Spawn Agent 3 (report-builder)

**Goal:** Launch report-builder agent to implement T008

**Step 1: Spawn report-builder agent**

Use Task tool to spawn agent:
```json
{
  "subagent_type": "general-purpose",
  "team_name": "binning-diagnostics",
  "name": "report-builder",
  "description": "Implement T008 report generation",
  "prompt": "You are the report-builder agent responsible for implementing T008 (Binning Report Generation).

**Your task:** Implement the report generation functions in feature_selection/validators/binning/report.py according to the interface contracts and task specification.

**Task spec:** docs/kanban/to-do/feature_validator/binning/T008_binning_report_generation.md

**Interface contracts:** Already defined in report.py - replace NotImplementedError with real implementations

**Key deliverables:**
1. Implement generate_binning_report() - orchestrate full diagnostic pipeline
2. Implement detect_failure_mode() - classify failure types
3. Implement save_report() - JSON serialization + plot saving
4. Implement display_report_summary() - terminal output
5. Write comprehensive unit tests in tests/validators/binning/test_report.py
6. Update exports in feature_selection/validators/binning/__init__.py

**Dependencies:**
- Use T005 functions: validate_binning_success, extract_region_metadata (already implemented)
- T006 (shape analysis) and T007 (plots) are being implemented by other agents in parallel
- For unit tests, create mocks for T006/T007 outputs

**Mock strategy for unit tests:**
```python
def _mock_shape_summary() -> dict[str, int]:
    return {\"long_tail\": 1, \"short_tail\": 1}

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
    return {\"heatmap\": fig, \"boundaries\": fig, \"multiplier_curve\": fig, \"panel\": fig}
```

**Testing approach:**
- Use synthetic BinningModelBase stubs with known state
- Test all failure modes: no_regions, isolated_spikes, insufficient_edge, none
- Test JSON serialization (exclude Figure objects, save plots separately)
- Test terminal output formatting
- Verify timestamp is ISO format
- Verify all report fields are populated

**Acceptance criteria:**
- All unit tests pass (minimum 6 tests)
- Code follows project conventions (frozen dataclasses, type hints, functional style)
- Functions match interface contracts exactly
- Mocks are clearly marked as temporary
- Report orchestration uses real T005 functions, mocked T006/T007 for unit tests

**Use @superpowers:test-driven-development for implementation.**"
}
```

---

### Task 6: Monitor Agent Progress

**Goal:** Track agent completion and handle any blocked states

**Step 1: Wait for agent completion messages**

Agents will send completion messages when Phase 1 is done. Expected messages:
- "shape-analyzer: T006 implementation complete, all unit tests passing"
- "plot-generator: T007 implementation complete, all unit tests passing"
- "report-builder: T008 implementation complete, all unit tests passing"

**Step 2: If any agent is blocked, provide assistance**

Check for:
- Import errors (missing dependencies)
- Interface mismatches (dataclass field mismatches)
- Test failures (incorrect mock structure)

**Step 3: Verify all 3 agents completed**

Once all 3 completion messages received, proceed to Phase 3.

---

## Phase 3: Integration

### Task 7: Remove Mocks and Wire Real Dependencies

**Goal:** Replace all temporary mocks with real cross-module dependencies

**Files:**
- Modify: `tests/validators/binning/test_plots.py` (remove mocks)
- Modify: `tests/validators/binning/test_report.py` (remove mocks)

**Step 1: Check plot-generator mocks**

```bash
grep -n "_mock_" tests/validators/binning/test_plots.py
```

If mocks exist for ShapeClassification, remove them and import real version:
```python
# Remove this:
# def _mock_shape_classification() -> ShapeClassification: ...

# Add this:
from feature_selection.validators.binning.shape_analysis import ShapeClassification
```

**Step 2: Check report-builder mocks**

```bash
grep -n "_mock_" tests/validators/binning/test_report.py
```

Remove all mock functions and import real implementations:
```python
# Remove mocks, add real imports:
from feature_selection.validators.binning.shape_analysis import (
    classify_region_shape,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    detect_region_adjacency,
)
from feature_selection.validators.binning.plots import (
    plot_bin_heatmap,
    plot_region_boundaries,
    plot_position_multiplier_curve,
    create_diagnostic_panel,
)
```

**Step 3: Update report.py to use real T006/T007**

Verify `feature_selection/validators/binning/report.py` imports real functions:
```python
from feature_selection.validators.binning.shape_analysis import (
    classify_region_shape,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    detect_region_adjacency,
)
from feature_selection.validators.binning.plots import create_diagnostic_panel
```

**Step 4: Run all unit tests to verify integration**

```bash
pytest tests/validators/binning/ -v
```

Expected: All tests pass (no mock-related failures)

**Step 5: Commit mock removal**

```bash
git add tests/validators/binning/test_plots.py
git add tests/validators/binning/test_report.py
git add feature_selection/validators/binning/report.py
git commit -m "refactor: remove mocks and wire real dependencies

Replace temporary mocks with real T006/T007 implementations.
Integration phase: all modules now use cross-module dependencies.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Update Module Exports

**Goal:** Add new functions and dataclasses to __init__.py exports

**Files:**
- Modify: `feature_selection/validators/binning/__init__.py`
- Modify: `feature_selection/validators/__init__.py`

**Step 1: Update binning/__init__.py**

```python
"""Binning diagnostics for continuous feature validation."""

from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
    calculate_coverage,
    detect_region_shape,
    extract_region_metadata,
    validate_binning_success,
)
from feature_selection.validators.binning.plots import (
    create_diagnostic_panel,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
)
from feature_selection.validators.binning.report import (
    BinningDiagnosticsReport,
    detect_failure_mode,
    display_report_summary,
    generate_binning_report,
    save_report,
)
from feature_selection.validators.binning.shape_analysis import (
    AdjacencyAnalysis,
    RegionCoverage,
    ShapeClassification,
    analyze_multi_region_shapes,
    calculate_region_coverage_breakdown,
    classify_region_shape,
    detect_region_adjacency,
)

__all__ = [
    # T005 (existing)
    "BinningSuccessCriteria",
    "RegionMetadata",
    "validate_binning_success",
    "extract_region_metadata",
    "detect_region_shape",
    "calculate_coverage",
    # T006 (new)
    "ShapeClassification",
    "RegionCoverage",
    "AdjacencyAnalysis",
    "classify_region_shape",
    "analyze_multi_region_shapes",
    "calculate_region_coverage_breakdown",
    "detect_region_adjacency",
    # T007 (new)
    "plot_bin_heatmap",
    "plot_region_boundaries",
    "plot_position_multiplier_curve",
    "create_diagnostic_panel",
    # T008 (new)
    "BinningDiagnosticsReport",
    "generate_binning_report",
    "detect_failure_mode",
    "save_report",
    "display_report_summary",
]
```

**Step 2: Update validators/__init__.py**

Add new exports to top-level validators module:
```python
"""Feature validator public interfaces."""

from feature_selection.validators.binning import (
    AdjacencyAnalysis,
    BinningDiagnosticsReport,
    BinningSuccessCriteria,
    RegionCoverage,
    RegionMetadata,
    ShapeClassification,
    analyze_multi_region_shapes,
    calculate_coverage,
    calculate_region_coverage_breakdown,
    classify_region_shape,
    create_diagnostic_panel,
    detect_failure_mode,
    detect_region_adjacency,
    detect_region_shape,
    display_report_summary,
    extract_region_metadata,
    generate_binning_report,
    plot_bin_heatmap,
    plot_position_multiplier_curve,
    plot_region_boundaries,
    save_report,
    validate_binning_success,
)

__all__ = [
    # T005
    "BinningSuccessCriteria",
    "RegionMetadata",
    "validate_binning_success",
    "extract_region_metadata",
    "detect_region_shape",
    "calculate_coverage",
    # T006
    "ShapeClassification",
    "RegionCoverage",
    "AdjacencyAnalysis",
    "classify_region_shape",
    "analyze_multi_region_shapes",
    "calculate_region_coverage_breakdown",
    "detect_region_adjacency",
    # T007
    "plot_bin_heatmap",
    "plot_region_boundaries",
    "plot_position_multiplier_curve",
    "create_diagnostic_panel",
    # T008
    "BinningDiagnosticsReport",
    "generate_binning_report",
    "detect_failure_mode",
    "save_report",
    "display_report_summary",
]
```

**Step 3: Verify imports**

```bash
python -c "from feature_selection.validators.binning import (
    ShapeClassification, RegionCoverage, AdjacencyAnalysis,
    classify_region_shape, analyze_multi_region_shapes,
    calculate_region_coverage_breakdown, detect_region_adjacency,
    plot_bin_heatmap, plot_region_boundaries, plot_position_multiplier_curve,
    create_diagnostic_panel, BinningDiagnosticsReport, generate_binning_report,
    detect_failure_mode, save_report, display_report_summary
); print('All imports successful')"
```

Expected output: "All imports successful"

**Step 4: Commit export updates**

```bash
git add feature_selection/validators/binning/__init__.py
git add feature_selection/validators/__init__.py
git commit -m "feat: export new binning diagnostics APIs

Add T006-T008 functions and dataclasses to public API.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Create Comprehensive Integration Test

**Goal:** Test full pipeline T005 → T006 → T007 → T008 with real data

**Files:**
- Create: `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`

**Step 1: Write integration test**

```python
"""Comprehensive integration test for binning diagnostics T005-T008."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.validators.binning import (
    BinningSuccessCriteria,
    generate_binning_report,
    display_report_summary,
    save_report,
)
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2024, 12, 31)
USE_CACHE = True
POPULATE_CACHE = True

ENSEMBLE_TICKERS = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
BIAS_SPEC = {
    "module_name": "rsi",
    "timeframes": [TimeFrame.D],
    "params": {"lookback": [5]},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _maybe_populate_cache(project_root: Path) -> None:
    if not POPULATE_CACHE:
        return

    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")

    manager = CacheManager(candle_dir=str(candle_dir))

    params = BIAS_SPEC.get("params", {})
    keys = list(params.keys())
    values = [value if isinstance(value, list) else [value] for value in params.values()]
    param_combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]

    all_specs = [
        {
            "module_name": BIAS_SPEC["module_name"],
            "params": combo,
            "timeframes": BIAS_SPEC.get("timeframes", [TimeFrame.D]),
        }
        for combo in param_combos
    ]

    result = manager.populate_cache(
        bias_node_specs=all_specs,
        tickers=ENSEMBLE_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        show_progress=False,
        overwrite_existing=False,
    )
    assert result["failed"] == 0, f"Cache population failures: {result}"


def _extract_rsi_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=BIAS_SPEC,
        ticker=ENSEMBLE_TICKERS,
        start=START_DATE,
        end=END_DATE,
        use_millisecond_offset=True,
        target_col="log_return",
        use_cache=USE_CACHE,
    )
    return features_df, targets_df


@pytest.mark.integration
def test_binning_full_pipeline_integration() -> None:
    """Comprehensive test of T005-T008 binning diagnostics pipeline.

    Tests full workflow:
    1. Extract RSI features from cache
    2. Fit ContinuousBinningModel
    3. Generate comprehensive diagnostics report (T005-T008)
    4. Save plots and JSON report to outputs directory
    5. Display terminal summary for manual verification

    Manual verification:
    - Check terminal output for readable summary
    - Inspect plots in tests/integration/outputs/binning/rsi_lookback_5/
    - Verify JSON report contains all metadata
    """
    project_root = _project_root()
    _maybe_populate_cache(project_root)

    # Extract features
    features_df, targets_df = _extract_rsi_features()
    feature_col = "rsi_signal_D_lookback_5"
    assert feature_col in features_df.columns

    feature_series = features_df[feature_col].copy()
    feature_series.name = feature_col
    target_series = targets_df.loc[feature_series.index, "log_return"]

    print("\n" + "=" * 72)
    print("BINNING DIAGNOSTICS INTEGRATION TEST")
    print("=" * 72)
    print(f"Feature: {feature_col}")
    print(f"Tickers: {[ticker.name for ticker in ENSEMBLE_TICKERS]}")
    print(f"Date range: {START_DATE.date()} -> {END_DATE.date()}")
    print(f"Samples: {len(feature_series)} observations")

    # Fit binning model
    model = ContinuousBinningModel(
        n_bins=15,
        selection_metric="sharpe",
        strategy="long",
        metric_threshold=0.0,
        t_threshold=0.5,
        min_region_width=1,
    )
    model.fit(feature_series, target_series)

    # Generate comprehensive report (T005-T008)
    criteria = BinningSuccessCriteria(
        metric_threshold=0.0, t_threshold=0.5, min_region_width=1
    )
    report = generate_binning_report(
        model=model, feature_data=feature_series, criteria=criteria, strategy="long"
    )

    # Verify report structure
    assert isinstance(report.success_verdict, bool)
    assert report.failure_mode in ["no_regions", "isolated_spikes", "insufficient_edge", "none"]
    assert isinstance(report.shape_summary, dict)
    assert isinstance(report.coverage_breakdown, list)
    assert 0.0 <= report.total_coverage_pct <= 100.0
    assert "heatmap" in report.diagnostic_plots
    assert "boundaries" in report.diagnostic_plots
    assert "multiplier_curve" in report.diagnostic_plots
    assert "panel" in report.diagnostic_plots

    # Save to outputs directory
    output_dir = project_root / "tests" / "integration" / "outputs" / "binning" / "rsi_lookback_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_report(report, str(output_dir))
    assert Path(report_path).exists()

    # Verify plots were saved
    plot_files = ["binning_heatmap.png", "region_boundaries.png",
                  "multiplier_curve.png", "diagnostic_panel.png"]
    for plot_file in plot_files:
        plot_path = output_dir / plot_file
        assert plot_path.exists(), f"Missing plot: {plot_file}"

    # Display terminal summary
    display_report_summary(report)

    print(f"\nPlots saved to: {output_dir}")
    for plot_file in plot_files:
        print(f"  ✓ {plot_file}")
    print(f"\nReport saved to: {report_path}")
    print("=" * 72)

    # Verify JSON structure
    with open(report_path) as f:
        report_json = json.load(f)

    assert "feature_column" in report_json
    assert "success_verdict" in report_json
    assert "failure_mode" in report_json
    assert "shape_summary" in report_json
    assert "total_coverage_pct" in report_json
    assert report_json["feature_column"] == feature_col


if __name__ == "__main__":
    # Allows researchers to run this file directly
    test_binning_full_pipeline_integration()
```

**Step 2: Run integration test**

```bash
pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py -v -s
```

Expected output:
- Test passes
- Terminal shows readable summary with regions, shapes, coverage
- Plots saved to tests/integration/outputs/binning/rsi_lookback_5/
- JSON report saved

**Step 3: Manual verification**

1. Check terminal output:
   - Success verdict (True/False)
   - Failure mode classification
   - Region count and shapes
   - Coverage percentage

2. Inspect plots:
   ```bash
   open tests/integration/outputs/binning/rsi_lookback_5/diagnostic_panel.png
   ```
   Verify:
   - Heatmap shows color-coded bins
   - Region boundaries overlay on histogram
   - Multiplier curve shows step function

3. Inspect JSON:
   ```bash
   cat tests/integration/outputs/binning/rsi_lookback_5/report.json | jq .
   ```
   Verify all metadata fields present

**Step 4: Commit integration test**

```bash
git add tests/integration/feature_validator/binning/test_binning_full_pipeline.py
git commit -m "test: add comprehensive binning diagnostics integration test

Tests full T005-T008 pipeline with RSI lookback 5.
Saves plots and JSON report for manual verification.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Run Full Test Suite

**Goal:** Verify all unit and integration tests pass

**Step 1: Run all binning unit tests**

```bash
pytest tests/validators/binning/ -v
```

Expected: All tests pass (20+ tests across 4 modules)

**Step 2: Run integration test**

```bash
pytest tests/integration/feature_validator/binning/ -v
```

Expected: Both integration tests pass (old T005 test + new full pipeline test)

**Step 3: Type check with mypy**

```bash
mypy feature_selection/validators/binning/ --strict
```

Expected: No type errors

**Step 4: Verify no regressions**

```bash
pytest tests/validators/ -v
```

Expected: All validator tests pass (including existing EDA, permutation tests)

---

### Task 11: Update Documentation

**Goal:** Document new APIs in feature selection docs

**Files:**
- Modify: `docs/api/feature_selection.md`

**Step 1: Add binning diagnostics section**

Add to docs/api/feature_selection.md:

```markdown
## Binning Diagnostics (T005-T008)

### Shape Analysis (T006)

**ShapeClassification**
- Directional shape classification: long_tail, short_tail, long_hump, short_hump
- Fields: shape_type, is_monotonic, touches_extreme, direction

**classify_region_shape(region, n_bins)**
- Classify single region shape with directional info
- Returns: ShapeClassification

**analyze_multi_region_shapes(regions, n_bins)**
- Aggregate shape type counts across all regions
- Returns: dict mapping shape_type to count

**RegionCoverage**
- Per-region coverage statistics
- Fields: region_id, individual_coverage_pct, cumulative_coverage_pct

**calculate_region_coverage_breakdown(regions, feature_data)**
- Calculate coverage percentages for each region
- Returns: list of RegionCoverage

**AdjacencyAnalysis**
- Gap detection and connectivity metrics
- Fields: gap_sizes, is_connected, isolation_score

**detect_region_adjacency(regions)**
- Analyze gaps between regions
- Returns: AdjacencyAnalysis

### Diagnostic Plots (T007)

**plot_bin_heatmap(model, metric, figsize, cmap)**
- Horizontal bar chart heatmap color-coded by metric
- Metrics: "sharpe", "t_stat", "sample_count", "mean_return"
- Returns: matplotlib Figure

**plot_region_boundaries(model, feature_data, regions, figsize)**
- Histogram with shaded tradeable regions
- Returns: matplotlib Figure

**plot_position_multiplier_curve(model, strategy, figsize)**
- Step function showing position scaling
- Returns: matplotlib Figure

**create_diagnostic_panel(model, feature_data, regions, strategy, figsize)**
- Combined 3-panel view (heatmap + boundaries + curve)
- Returns: matplotlib Figure with 3 subplots

### Report Generation (T008)

**BinningDiagnosticsReport**
- Comprehensive diagnostics container
- Fields: feature_column, parameter_combo, success_verdict, failure_mode,
  criteria, regions, shape_summary, coverage_breakdown, total_coverage_pct,
  adjacency_analysis, diagnostic_plots, timestamp

**generate_binning_report(model, feature_data, criteria, strategy)**
- Orchestrate full diagnostic pipeline (T005-T007)
- Returns: BinningDiagnosticsReport

**detect_failure_mode(model, criteria)**
- Classify binning failure type
- Returns: "no_regions" | "isolated_spikes" | "insufficient_edge" | "none"

**save_report(report, output_dir)**
- Save JSON metadata + PNG plots
- Returns: path to JSON file

**display_report_summary(report)**
- Pretty-print terminal summary
- Returns: None (prints to stdout)
```

**Step 2: Commit documentation**

```bash
git add docs/api/feature_selection.md
git commit -m "docs: document binning diagnostics APIs (T006-T008)

Add API documentation for shape analysis, plotting, and reporting.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 12: Team Cleanup

**Goal:** Gracefully shutdown agent team and clean up resources

**Step 1: Send shutdown requests to agents**

Use SendMessage tool for each agent:
```json
{
  "type": "shutdown_request",
  "recipient": "shape-analyzer",
  "content": "T006 implementation complete and integrated. Thank you!"
}
```

Repeat for "plot-generator" and "report-builder".

**Step 2: Wait for shutdown confirmations**

Agents will respond with shutdown approval.

**Step 3: Delete team**

Use TeamDelete tool:
```json
{}
```

---

## Acceptance Criteria

**Code quality:**
- [ ] All unit tests pass (20+ tests across T006-T008)
- [ ] Integration test passes and produces readable output
- [ ] No mypy errors with --strict
- [ ] Code follows project conventions (frozen dataclasses, type hints, functional style)

**Functionality:**
- [ ] Shape classification works for all 4 types (long_tail, short_tail, long_hump, short_hump)
- [ ] Coverage calculations sum correctly
- [ ] Adjacency analysis detects gaps
- [ ] All 4 plot functions produce valid matplotlib Figures
- [ ] Report generation orchestrates full pipeline
- [ ] JSON serialization works (excluding Figure objects)
- [ ] Terminal output is readable

**Manual verification:**
- [ ] Terminal summary shows success verdict, regions, shapes, coverage
- [ ] Plots saved to tests/integration/outputs/binning/rsi_lookback_5/
- [ ] Heatmap shows color-coded bins (green=high Sharpe, red=low)
- [ ] Region boundaries overlay correctly on histogram
- [ ] Multiplier curve shows step function
- [ ] JSON report contains all metadata

**Documentation:**
- [ ] API docs updated in docs/api/feature_selection.md
- [ ] All new functions and dataclasses documented

---

## References

- Design doc: `docs/plans/2026-02-16-binning-team-design.md`
- Task specs: `docs/kanban/to-do/feature_validator/binning/T006-T008*.md`
- Integration spec: `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md`
- Feature validator spec: `docs/library/Feature_selection/feature_validator.md`

---

**End of implementation plan.**
