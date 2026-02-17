# Binning Diagnostics Infrastructure (T005) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add binning diagnostics infrastructure (success criteria validation, region metadata extraction, shape detection, and coverage) plus tests/docs updates, while keeping current `ContinuousBinningModel` defaults unchanged.

**Architecture:** Introduce a new `feature_selection/validators/binning` package with pure diagnostics utilities that consume fitted `BinningModelBase` state (`bin_stats_`, `significant_regions_`, `bin_edges_`). Keep model behavior stable and only add minimal hardening needed for diagnostics contract correctness. Validate behavior through deterministic integration tests.

**Tech Stack:** Python 3, `dataclasses`, `typing`, `pandas`, `pytest`.

---

### Task 1: Create Binning Validators Package and RED Tests

**Files:**
- Create: `feature_selection/validators/binning/__init__.py`
- Create: `feature_selection/validators/binning/diagnostics.py`
- Create: `tests/integration/feature_validator/binning/test_success_criteria.py`
- Create: `tests/integration/feature_validator/binning/test_region_metadata.py`
- Create: `tests/integration/feature_validator/binning/test_shape_detection.py`
- Create: `tests/integration/feature_validator/binning/test_coverage.py`

**Step 1: Write failing tests for T005 interfaces**

```python
from dataclasses import FrozenInstanceError

from feature_selection.validators.binning.diagnostics import (
    BinningSuccessCriteria,
    RegionMetadata,
    calculate_coverage,
    detect_region_shape,
    extract_region_metadata,
    validate_binning_success,
)


def test_success_criteria_dataclass_is_frozen() -> None:
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
    try:
        criteria.metric_threshold = 0.0
    except FrozenInstanceError:
        assert True
```

**Step 2: Run tests to verify RED**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_success_criteria.py -q
```
Expected: `ModuleNotFoundError` or import/function-not-found failures.

**Step 3: Add skeletal module and exports (minimal stubs)**

```python
# diagnostics.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BinningSuccessCriteria:
    metric_threshold: float
    t_threshold: float
    min_region_width: int
```

**Step 4: Re-run tests and confirm still RED for unimplemented behaviors**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_success_criteria.py -q
```
Expected: assertion failures for unimplemented functions.

**Step 5: Commit scaffold + RED tests**

```bash
git add feature_selection/validators/binning tests/integration/feature_validator/binning
git commit -m "test: add red tests for binning diagnostics interfaces"
```

### Task 2: Implement Region Metadata Extraction and Shape Detection (GREEN)

**Files:**
- Modify: `feature_selection/validators/binning/diagnostics.py`
- Modify: `feature_selection/validators/binning/__init__.py`
- Modify: `tests/integration/feature_validator/binning/test_region_metadata.py`
- Modify: `tests/integration/feature_validator/binning/test_shape_detection.py`

**Step 1: Write/complete failing behavior tests for metadata extraction and shape detection**

```python
def test_extract_region_metadata_aggregates_bin_stats() -> None:
    model = _build_fitted_model_fixture()
    regions = extract_region_metadata(model)
    assert regions[0].sample_count == 120
    assert regions[0].feature_range == (10.0, 39.9)


def test_detect_tail() -> None:
    region = RegionMetadata(
        start_bin=0,
        end_bin=2,
        bins=[0, 1, 2],
        mean_sharpe=0.9,
        mean_t_stat=2.3,
        sample_count=90,
        feature_range=(0.0, 20.0),
    )
    assert detect_region_shape(region, n_bins=15) == "tail"
```

**Step 2: Run targeted tests to verify RED**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_region_metadata.py tests/integration/feature_validator/binning/test_shape_detection.py -q
```
Expected: failing assertions.

**Step 3: Implement extraction + shape logic minimally to satisfy tests**

```python
def detect_region_shape(region: RegionMetadata, n_bins: int) -> Literal["tail", "hump"]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if region.start_bin < 0 or region.end_bin < region.start_bin:
        raise ValueError("Invalid region bin bounds")
    return "tail" if (region.start_bin == 0 or region.end_bin == n_bins - 1) else "hump"
```

**Step 4: Run tests to verify GREEN**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_region_metadata.py tests/integration/feature_validator/binning/test_shape_detection.py -q
```
Expected: all pass.

**Step 5: Commit metadata + shape implementation**

```bash
git add feature_selection/validators/binning tests/integration/feature_validator/binning
git commit -m "feat: add region metadata extraction and shape detection"
```

### Task 3: Implement Success Validation and Coverage (GREEN)

**Files:**
- Modify: `feature_selection/validators/binning/diagnostics.py`
- Modify: `tests/integration/feature_validator/binning/test_success_criteria.py`
- Modify: `tests/integration/feature_validator/binning/test_coverage.py`

**Step 1: Write/complete failing tests for validation and coverage rules**

```python
def test_validate_binning_success_with_valid_regions() -> None:
    model = _build_valid_model_fixture()
    criteria = BinningSuccessCriteria(metric_threshold=0.5, t_threshold=2.0, min_region_width=2)
    assert validate_binning_success(model, criteria) is True


def test_calculate_coverage_union_ranges() -> None:
    feature = pd.Series([5, 15, 25, 35, 45, 55])
    regions = [
        RegionMetadata(..., feature_range=(10.0, 30.0)),
        RegionMetadata(..., feature_range=(25.0, 50.0)),
    ]
    assert calculate_coverage(regions, feature) == 66.66666666666666
```

**Step 2: Run tests to verify RED**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_success_criteria.py tests/integration/feature_validator/binning/test_coverage.py -q
```
Expected: failing assertions.

**Step 3: Implement validation + coverage functions minimally**

```python
def calculate_coverage(regions: list[RegionMetadata], feature_data: pd.Series) -> float:
    clean = feature_data.dropna()
    if clean.empty or not regions:
        return 0.0
    covered = pd.Series(False, index=clean.index)
    for region in regions:
        low, high = region.feature_range
        covered = covered | ((clean >= low) & (clean <= high))
    return float(max(0.0, min(100.0, (covered.sum() / len(clean)) * 100.0)))
```

**Step 4: Run tests to verify GREEN**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/test_success_criteria.py tests/integration/feature_validator/binning/test_coverage.py -q
```
Expected: all pass.

**Step 5: Commit validation + coverage implementation**

```bash
git add feature_selection/validators/binning tests/integration/feature_validator/binning
git commit -m "feat: add binning success validation and coverage metrics"
```

### Task 4: API Docs Update and Full T005 Verification

**Files:**
- Modify: `docs/api/feature_selection.md`
- Modify: `feature_selection/validators/__init__.py`
- Modify: `feature_selection/validators/binning/__init__.py`

**Step 1: Add public API docs entry for diagnostics contracts**

```markdown
### Binning diagnostics entrypoints
Public symbols:
- `BinningSuccessCriteria`
- `RegionMetadata`
- `validate_binning_success(...)`
- `extract_region_metadata(...)`
- `detect_region_shape(...)`
- `calculate_coverage(...)`
```

**Step 2: Run targeted integration suite required by T005**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/ -q
```
Expected: all tests pass.

**Step 3: Run nearby regression tests for continuous binning behavior**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/base_models/test_continuous_regions.py tests/base_models/test_continuous_binning_multipliers.py tests/base_models/test_binning_base_stats.py -q
```
Expected: all tests pass.

**Step 4: Final verification command bundle**

Run:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate && pytest tests/integration/feature_validator/binning/ tests/base_models/test_continuous_regions.py tests/base_models/test_continuous_binning_multipliers.py tests/base_models/test_binning_base_stats.py -q
```
Expected: all pass with clean output.

**Step 5: Commit docs + exports + final state**

```bash
git add docs/api/feature_selection.md feature_selection/validators/__init__.py feature_selection/validators/binning/__init__.py
git commit -m "docs: add binning diagnostics API and export surface"
```
