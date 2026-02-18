# Continuous Binning Analysis Smoke Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a continuous-binning analysis pipeline with per-parameter outputs and refactor the binning integration test into a dry-run smoke check.

**Architecture:** Reuse the existing continuous-binning config/data loader to drive a new binning analysis pipeline that generates `BinningDiagnosticsReport` outputs. Provide a thin script entrypoint and a dry-run path used by the integration test to avoid data dependencies.

**Tech Stack:** Python 3.12, pytest, pandas, matplotlib (Agg), feature_selection validators.

---

### Task 1: Add binning analysis config fields

**Files:**
- Modify: `feature_research/continuous_binning/config.py`

**Step 1: Write the failing test**

Create a new unit test ensuring the config exposes binning analysis settings.

```python
from feature_research.continuous_binning.config import load_config


def test_binning_analysis_config_fields() -> None:
    config = load_config()
    assert config.binning_params.n_bins > 0
    assert config.binning_params.metric_threshold >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_config.py::test_binning_analysis_config_fields -v`
Expected: FAIL (missing binning_params attribute)

**Step 3: Write minimal implementation**

Extend `ResearchConfig` with a new frozen dataclass for binning settings.

```python
@dataclass(frozen=True)
class BinningAnalysisConfig:
    n_bins: int
    selection_metric: str
    strategy: str
    metric_threshold: float
    t_threshold: float
    min_region_width: int
    max_regions: int
    direction_filter: str
```

Attach it to `ResearchConfig` as `binning_params` and populate defaults in `load_config()`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_config.py::test_binning_analysis_config_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
```

---

### Task 2: Implement binning analysis pipeline

**Files:**
- Create: `feature_research/continuous_binning/binning_analysis.py`
- Modify: `feature_research/continuous_binning/data_loader.py`

**Step 1: Write the failing test**

Add a unit test to validate that dry-run mode creates output directories without data access.

```python
from pathlib import Path

from feature_research.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.continuous_binning.config import load_config


def test_binning_analysis_dry_run(tmp_path: Path) -> None:
    config = load_config()
    output_dir = tmp_path / "results" / config.bias_spec["module_name"]
    results = run_binning_analysis_pipeline(config, output_dir, dry_run=True)
    assert output_dir.exists()
    assert results == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_pipeline.py::test_binning_analysis_dry_run -v`
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

Implement the pipeline to:
- Expand bias specs via `expand_bias_specs`.
- Create per-combo directories with `param_combo_label`.
- If `dry_run`, return empty mapping without extraction.
- Else, load data, fit `ContinuousBinningModel`, call `generate_binning_report`, save report to the combo directory, and collect paths.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_pipeline.py::test_binning_analysis_dry_run -v`
Expected: PASS

**Step 5: Commit**

```bash
```

---

### Task 3: Add binning analysis entrypoint

**Files:**
- Create: `feature_research/continuous_binning/run_binning_analysis.py`

**Step 1: Write the failing test**

Add a compile check for the new entrypoint.

```python
import py_compile


def test_binning_analysis_entrypoint_compiles() -> None:
    py_compile.compile("feature_research/continuous_binning/run_binning_analysis.py", doraise=True)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_entrypoint.py::test_binning_analysis_entrypoint_compiles -v`
Expected: FAIL (file missing)

**Step 3: Write minimal implementation**

Create the script to load config and call `run_binning_analysis_pipeline` with `config.reports_dir`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_entrypoint.py::test_binning_analysis_entrypoint_compiles -v`
Expected: PASS

**Step 5: Commit**

```bash
```

---

### Task 4: Refactor integration test into smoke check

**Files:**
- Modify: `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`

**Step 1: Write the failing test**

Replace the existing test to call the dry-run pipeline and assert output dir creation.

```python
from pathlib import Path

from feature_research.continuous_binning.binning_analysis import run_binning_analysis_pipeline
from feature_research.continuous_binning.config import load_config


def test_binning_full_pipeline_integration_smoke(tmp_path: Path) -> None:
    config = load_config()
    output_dir = tmp_path / "results" / config.bias_spec["module_name"]
    results = run_binning_analysis_pipeline(config, output_dir, dry_run=True)
    assert output_dir.exists()
    assert results == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py::test_binning_full_pipeline_integration_smoke -v`
Expected: FAIL (name mismatch or old test still present)

**Step 3: Write minimal implementation**

Remove the old integration test body and replace with the smoke test above.

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py::test_binning_full_pipeline_integration_smoke -v`
Expected: PASS

**Step 5: Commit**

```bash
```

---

### Task 5: Update API docs

**Files:**
- Modify: `docs/api/data_pipeline.md`

**Step 1: Write the failing test**

Add a doc check that asserts the new entrypoint is mentioned.

```python
from pathlib import Path


def test_data_pipeline_docs_mentions_binning_analysis() -> None:
    content = Path("docs/api/data_pipeline.md").read_text(encoding="utf-8")
    assert "run_binning_analysis.py" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/docs/test_data_pipeline_docs.py::test_data_pipeline_docs_mentions_binning_analysis -v`
Expected: FAIL (string missing)

**Step 3: Write minimal implementation**

Add a short subsection under EDA/Research entrypoints describing `feature_research/continuous_binning/run_binning_analysis.py`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/docs/test_data_pipeline_docs.py::test_data_pipeline_docs_mentions_binning_analysis -v`
Expected: PASS

**Step 5: Commit**

```bash
```

---

### Task 6: Run targeted verification

**Files:**
- Test: `tests/integration/feature_validator/binning/test_binning_full_pipeline.py`
- Test: `tests/unit-tests/feature_validator/test_binning_analysis_pipeline.py`
- Test: `tests/unit-tests/feature_validator/test_binning_analysis_entrypoint.py`

**Step 1: Run unit tests**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_pipeline.py -v`
Expected: PASS

**Step 2: Run entrypoint compile test**

Run: `pytest tests/unit-tests/feature_validator/test_binning_analysis_entrypoint.py -v`
Expected: PASS

**Step 3: Run integration smoke test**

Run: `pytest tests/integration/feature_validator/binning/test_binning_full_pipeline.py -v`
Expected: PASS

**Step 4: Commit verification note**

```bash
```

---

## Notes
- Baseline full pytest suite currently fails due to missing external dependencies; this plan uses only targeted tests.
- Ensure all new code keeps ASCII-only characters.
