# Continuous EDA Research Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a config-driven continuous-feature EDA research pipeline in `feature_research/continuous_binning/` that a researcher can run end-to-end, with integration tests that import the same codepath to detect regressions.

**Architecture:** `config.py` holds all researcher-editable settings as a `ResearchConfig` dataclass. `data_loader.py` wraps cache population and feature extraction. `pipeline.py` exports `run_continuous_eda_pipeline(config, output_dir)` — the reusable core that calls the existing `run_eda_for_continuous_feature` + `save_eda_report` for each param combo. `run_eda.py` is a thin `__main__` entry point. Integration tests import `run_continuous_eda_pipeline` directly, so any regression in the research script breaks the tests.

**Tech Stack:** `pandas`, `pathlib`, existing `feature_selection.eda.*`, `feature_extraction.feature_extractor.extract_features_for_bias_node`, `utils.cache_manager.CacheManager`, `utils.enums.{Ticker, TimeFrame}`, `pytest`, `matplotlib` (Agg backend for tests)

---

## Context: Key existing functions

- `extract_features_for_bias_node(bias_spec, ticker, start, end, use_cache, target_col)` → `(features_df, targets_df)` — `features_df` has one column per param combo (named e.g. `rsi_D_lookback_5`) plus a `ticker` column
- `run_eda_for_continuous_feature(feature, target, timestamps, metadata, config)` → `ContinuousEDAReport`
- `save_eda_report(report, output_dir, overwrite)` → `Path` — creates `output_dir / feature_name / param_hash /`
- `EDAMetadata(feature_name, param_combo, timeframe, ticker, timestamp)`
- `EDAConfig(n_bins, rolling_window, objective_fn, max_lag, bootstrap_iterations, random_seed)`
- `CacheManager(candle_dir=str(candle_dir)).populate_cache(bias_node_specs, tickers, start_date, end_date, show_progress, overwrite_existing)`

Venv activation (required before any Python command): `source /home/raman/repos/Trading-Algo/venv/bin/activate`

---

## Task 1: `config.py` and `__init__.py`

**Files:**
- Create: `feature_research/continuous_binning/__init__.py`
- Create: `feature_research/continuous_binning/config.py`
- Create: `tests/feature_research/__init__.py`
- Create: `tests/feature_research/test_config.py`

**Step 1: Write the failing test**

```python
# tests/feature_research/test_config.py
from datetime import datetime
from pathlib import Path

import pytest

from feature_research.continuous_binning.config import ResearchConfig, load_config
from utils.enums import Ticker, TimeFrame


def test_load_config_returns_research_config():
    config = load_config()
    assert isinstance(config, ResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert config.start == datetime(2000, 1, 1)
    assert config.end == datetime(2024, 12, 31)
    assert config.bias_spec["module_name"] == "rsi"
    assert isinstance(config.bias_spec["params"]["lookback"], list)
    assert config.target_col == "log_return"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_reports_dir_includes_module_name():
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    assert "continuous_binning" in str(config.reports_dir)
```

**Step 2: Run test to verify it fails**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/feature_research/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'feature_research'`

**Step 3: Create `__init__.py` files and `config.py`**

```python
# feature_research/__init__.py
# (empty)
```

```python
# feature_research/continuous_binning/__init__.py
from feature_research.continuous_binning.config import ResearchConfig, load_config
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline

__all__ = ["ResearchConfig", "load_config", "run_continuous_eda_pipeline"]
```

```python
# feature_research/continuous_binning/config.py
"""Researcher-editable configuration for the continuous binning EDA pipeline.

Edit the values in load_config() to customise tickers, date range, and bias node specs.
All other scripts import from here — change once, apply everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from utils.enums import Ticker, TimeFrame

# ---------------------------------------------------------------------------
# Root of the feature_research tree — resolved at import time so scripts
# work regardless of the working directory they are launched from.
# ---------------------------------------------------------------------------
_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_CB_DIR = _FEATURE_RESEARCH_DIR / "continuous_binning"


@dataclass(frozen=True)
class ResearchConfig:
    """All researcher-editable settings for continuous-binning EDA.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict
        Bias-node specification.  ``params`` values may be lists for grid search.
        Example::

            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
            }
    target_col : str
        Column from ``targets_df`` to use as the prediction target.
        Options: ``"log_return"``, ``"log_return_atr"``, ``"log_return_ewsd"``.
    strategy : str
        Passed to the EDA summary label only.
        Options: ``"long"``, ``"short"``, ``"long-short"``.
    use_cache : bool
        Whether to use pre-computed caches for feature extraction.
    populate_cache : bool
        If True, run ``CacheManager.populate_cache()`` before extraction.
    reports_dir : Path
        Root output directory for EDA reports.
        Default: ``feature_research/continuous_binning/results/{module_name}/``
    """

    tickers: list[Ticker]
    start: datetime
    end: datetime
    bias_spec: dict
    target_col: str
    strategy: str
    use_cache: bool
    populate_cache: bool
    reports_dir: Path


def load_config() -> ResearchConfig:
    """Return the default research configuration.

    **Edit this function** to customise tickers, dates, and bias node specs.
    All pipeline scripts import from here.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,   # E-Mini S&P 500
        Ticker.NQ,   # E-Mini Nasdaq-100
        Ticker.YM,   # E-Mini Dow Jones
        Ticker.RTY,  # E-Mini Russell 2000
    ]

    start = datetime(2000, 1, 1)
    end = datetime(2024, 12, 31)

    bias_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    }

    target_col = "log_return"
    strategy = "long-short"

    # Caching
    use_cache = True
    populate_cache = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    module_name = bias_spec["module_name"]
    reports_dir = _CB_DIR / "results" / module_name

    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        bias_spec=bias_spec,
        target_col=target_col,
        strategy=strategy,
        use_cache=use_cache,
        populate_cache=populate_cache,
        reports_dir=reports_dir,
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/feature_research/test_config.py -v
```

Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add feature_research/__init__.py \
        feature_research/continuous_binning/__init__.py \
        feature_research/continuous_binning/config.py \
        tests/feature_research/__init__.py \
        tests/feature_research/test_config.py
git commit -m "feat: add ResearchConfig and load_config for continuous binning EDA"
```

---

## Task 2: `data_loader.py`

**Files:**
- Create: `feature_research/continuous_binning/data_loader.py`
- Modify: `tests/feature_research/test_config.py` → add data_loader unit tests (or create `tests/feature_research/test_data_loader.py`)

**Step 1: Write the failing tests**

```python
# tests/feature_research/test_data_loader.py
from itertools import product

import pytest

from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    param_combo_label,
)
from utils.enums import TimeFrame


def test_expand_bias_specs_single_param():
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 2
    assert result[0]["params"] == {"lookback": 2}
    assert result[1]["params"] == {"lookback": 3}
    assert all(r["module_name"] == "rsi" for r in result)


def test_expand_bias_specs_grid():
    spec = {
        "module_name": "cmma",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [20, 50], "atr_length": [14, 21]},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 4  # 2x2 grid


def test_expand_bias_specs_scalar_params():
    """Scalar param values (not lists) should be treated as single-element lists."""
    spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": 5},
    }
    result = expand_bias_specs(spec)
    assert len(result) == 1
    assert result[0]["params"] == {"lookback": 5}


def test_param_combo_label_single():
    assert param_combo_label({"lookback": 5}) == "lookback_5"


def test_param_combo_label_multi():
    label = param_combo_label({"lookback": 20, "atr_length": 14})
    # deterministic: sorted keys
    assert label == "atr_length_14__lookback_20"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/feature_research/test_data_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'feature_research.continuous_binning.data_loader'`

**Step 3: Implement `data_loader.py`**

```python
# feature_research/continuous_binning/data_loader.py
"""Data loading and cache management helpers for the continuous binning research pipeline."""
from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from feature_research.continuous_binning.config import ResearchConfig

from feature_extraction.feature_extractor import extract_features_for_bias_node
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame


def expand_bias_specs(bias_spec: dict) -> list[dict]:
    """Expand a bias_spec with list-valued params into one spec per param combo."""
    params = bias_spec.get("params", {})
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]
    return [
        {
            "module_name": bias_spec["module_name"],
            "params": combo,
            "timeframes": bias_spec.get("timeframes", [TimeFrame.D]),
        }
        for combo in combos
    ]


def param_combo_label(combo: dict) -> str:
    """Return a human-readable folder name for a param combo dict.

    Examples
    --------
    >>> param_combo_label({"lookback": 5})
    "lookback_5"
    >>> param_combo_label({"lookback": 20, "atr_length": 14})
    "atr_length_14__lookback_20"
    """
    parts = [f"{k}_{v}" for k, v in sorted(combo.items())]
    return "__".join(parts)


def populate_cache_if_needed(config: "ResearchConfig") -> None:
    """Populate the feature cache if config.populate_cache is True.

    Safe to call even if cache already exists — ``overwrite_existing=False``
    means only missing entries are computed.
    """
    if not config.populate_cache:
        return

    project_root = Path(__file__).resolve().parents[3]
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        print(f"[data_loader] WARNING: candle directory not found at {candle_dir}. Skipping cache population.")
        return

    manager = CacheManager(candle_dir=str(candle_dir))
    expanded = expand_bias_specs(config.bias_spec)
    summary = manager.populate_cache(
        bias_node_specs=expanded,
        tickers=config.tickers,
        start_date=config.start,
        end_date=config.end,
        show_progress=True,
        overwrite_existing=False,
    )
    print(f"[data_loader] Cache populated: {summary}")


def load_features_for_combo(
    single_combo_spec: dict,
    config: "ResearchConfig",
) -> tuple[pd.Series, pd.Series, str] | None:
    """Extract feature + target Series for a single param combo across all config tickers.

    Returns
    -------
    (feature, target, feature_col) or None if extraction fails / returns empty data.

    The returned Series are aligned (same index, NaNs dropped) and concatenated
    across all tickers in ``config.tickers``.
    """
    try:
        features_df, targets_df = extract_features_for_bias_node(
            bias_spec=single_combo_spec,
            ticker=config.tickers,
            start=config.start,
            end=config.end,
            use_millisecond_offset=True,
            target_col=config.target_col,
            use_cache=config.use_cache,
        )
    except Exception as exc:
        print(f"[data_loader] Feature extraction failed for {single_combo_spec['params']}: {exc}")
        return None

    if features_df is None or features_df.empty:
        print(f"[data_loader] Empty features for {single_combo_spec['params']}. Skipping.")
        return None

    feature_cols = [c for c in features_df.columns if c != "ticker"]
    if not feature_cols:
        return None
    feature_col = feature_cols[0]

    target_col_name = config.target_col if config.target_col in targets_df.columns else (
        [c for c in targets_df.columns if c != "ticker"][0]
    )

    aligned = pd.DataFrame(
        {"feature": features_df[feature_col], "target": targets_df[target_col_name]}
    ).dropna()

    if aligned.empty:
        return None

    return aligned["feature"], aligned["target"], feature_col
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/feature_research/test_data_loader.py -v
```

Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add feature_research/continuous_binning/data_loader.py \
        tests/feature_research/test_data_loader.py
git commit -m "feat: add data_loader with expand_bias_specs and cache helpers"
```

---

## Task 3: `pipeline.py`

**Files:**
- Create: `feature_research/continuous_binning/pipeline.py`

No isolated unit test here — the integration test in Task 5 is the test for this module.
Write the implementation now; verify via integration test in Task 5.

**Step 1: Implement `pipeline.py`**

```python
# feature_research/continuous_binning/pipeline.py
"""Core EDA pipeline for continuous binning research.

Entry point for tests and scripts alike — import ``run_continuous_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.continuous_binning.config import ResearchConfig

from feature_research.continuous_binning.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import run_eda_for_continuous_feature, save_eda_report
from utils.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict, fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def run_continuous_eda_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full continuous-feature EDA pipeline for every param combo in config.

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run ``run_eda_for_continuous_feature`` to produce a ``ContinuousEDAReport``.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : ResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports.  Created if it does not exist.
        Each param combo writes to ``output_dir / param_label /``.

    Returns
    -------
    dict[str, Path]
        Mapping of ``param_label`` → saved report path for each successful combo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)
    tf = _normalize_timeframe(config.bias_spec)

    results: dict[str, Path] = {}

    print(f"\n{'='*64}")
    print(f"Continuous EDA Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} → {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP — no data")
            continue

        feature, target, feature_col = data
        timestamps = pd.DatetimeIndex(feature.index)

        rolling_window = max(20, min(252, len(feature) // 4))

        metadata = EDAMetadata(
            feature_name=feature_col,
            param_combo=combo,
            timeframe=tf,
            ticker=config.tickers[0],
            timestamp=datetime.now(),
        )
        eda_config = EDAConfig(n_bins=15, rolling_window=rolling_window)

        report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        results[label] = saved_path

        pearson = report.common_stats.correlation_analysis.pearson
        tau = report.continuous_stats.monotonicity_test.kendall_tau
        trend = report.continuous_stats.decile_analysis.overall_trend
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  pearson={pearson:+.3f}  "
            f"tau={tau:+.3f}  trend={trend}  {viable}  warnings={warnings}"
        )

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded → {output_dir}\n")
    return results
```

**Step 2: Commit (before running tests — tests come in Task 5)**

```bash
git add feature_research/continuous_binning/pipeline.py
git commit -m "feat: implement run_continuous_eda_pipeline"
```

---

## Task 4: `run_eda.py`

**Files:**
- Create: `feature_research/continuous_binning/run_eda.py`

**Step 1: Implement the entry point**

```python
# feature_research/continuous_binning/run_eda.py
"""Entry point for the continuous binning EDA research pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_eda.py

Results are written to feature_research/continuous_binning/results/{module_name}/.
Edit feature_research/continuous_binning/config.py to change tickers, dates, or bias specs.
"""
from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline

if __name__ == "__main__":
    config = load_config()
    results = run_continuous_eda_pipeline(config, config.reports_dir)
    print(f"EDA complete. {len(results)} param combos written to {config.reports_dir}")
```

**Step 2: Commit**

```bash
git add feature_research/continuous_binning/run_eda.py
git commit -m "feat: add run_eda.py entry point for continuous binning EDA"
```

---

## Task 5: Integration test — `test_continuous_eda_pipeline.py`

**Files:**
- Create: `tests/integration/feature_validator/test_continuous_eda_pipeline.py`

This test imports `run_continuous_eda_pipeline` directly — the same function the researcher runs. If the research script regresses, this test breaks.

**Step 1: Write the failing test**

```python
# tests/integration/feature_validator/test_continuous_eda_pipeline.py
"""Integration test for the continuous binning EDA research pipeline.

Imports run_continuous_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: RSI lookback=5, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from feature_research.continuous_binning.config import ResearchConfig, load_config
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline
from utils.enums import Ticker, TimeFrame


def _minimal_config(tmp_output: Path) -> ResearchConfig:
    """Single-combo RSI-5 config for fast integration smoke test."""
    return ResearchConfig(
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2023, 12, 31),
        bias_spec={
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 5},
        },
        target_col="log_return",
        strategy="long-short",
        use_cache=True,
        populate_cache=True,
        reports_dir=tmp_output,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _skip_if_no_data() -> None:
    candle_dir = _project_root() / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")


@pytest.mark.integration
def test_continuous_eda_pipeline_smoke(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    lookback: int = 5,
) -> None:
    """Smoke test: single RSI param combo, ES daily, 2020-2023.

    Verifies:
    - Pipeline runs without exception
    - Exactly one result entry returned
    - Output directory contains expected JSON + plot files

    All key config inputs are exposed as parameters so researchers can call
    this directly with custom values for interactive validation.
    """
    _skip_if_no_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookback},
            },
            target_col="log_return",
            strategy="long-short",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_continuous_eda_pipeline(config, Path(tmpdir))

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    label = f"lookback_{lookback}"
    assert label in results, f"Expected key '{label}' in results, got {list(results.keys())}"

    report_path = results[label]
    assert report_path.exists(), f"Report path does not exist: {report_path}"
    assert (report_path / "metadata.json").exists()
    assert (report_path / "common_stats.json").exists()
    assert (report_path / "feature_stats.json").exists()
    assert (report_path / "diagnostics.json").exists()
    plots_dir = report_path / "plots"
    assert plots_dir.is_dir()
    expected_plots = [
        "time_series_fig.png",
        "rolling_corr_fig.png",
        "rolling_obj_fig.png",
        "decile_plot_fig.png",
        "histogram_fig.png",
        "qq_plot_fig.png",
        "kde_fig.png",
    ]
    for plot_file in expected_plots:
        assert (plots_dir / plot_file).exists(), f"Missing plot: {plot_file}"


@pytest.mark.integration
def test_continuous_eda_pipeline_multi_combo(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    lookbacks: list[int] | None = None,
) -> None:
    """Multi-combo smoke test: RSI lookbacks [5, 10], ES daily, 2020-2023.

    Verifies:
    - Pipeline produces one result per param combo
    - Separate output directories created for each combo

    Exposed as parameters for researcher-driven exploration.
    """
    _skip_if_no_data()
    lookbacks = lookbacks or [5, 10]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": lookbacks},
            },
            target_col="log_return",
            strategy="long-short",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_continuous_eda_pipeline(config, Path(tmpdir))

    assert len(results) == len(lookbacks), (
        f"Expected {len(lookbacks)} results, got {len(results)}"
    )
    for lb in lookbacks:
        assert f"lookback_{lb}" in results
```

**Step 2: Run test to verify current state**

```bash
pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py -v -m integration
```

Expected: Tests either PASS (if data available) or SKIP with "Missing persisted candle directory" — both are correct outcomes.
If you see an ImportError or AttributeError instead, fix the source of the error before continuing.

**Step 3: Verify the pipeline function works from the command line (manual smoke)**

If `data/ohlc_data/` exists:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
python -c "
from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline
import tempfile; from pathlib import Path
config = load_config()
with tempfile.TemporaryDirectory() as t:
    r = run_continuous_eda_pipeline(config, Path(t))
    print('Results:', list(r.keys()))
"
```

Expected: prints combo labels and `Results: ['lookback_2', 'lookback_3', ...]`

**Step 4: Commit**

```bash
git add tests/integration/feature_validator/test_continuous_eda_pipeline.py
git commit -m "test: add integration test for continuous EDA research pipeline"
```

---

## Task 6: Refactor `test_eda_pipeline.py`

**Files:**
- Modify: `tests/integration/feature_validator/test_eda_pipeline.py`

**Step 1: Remove `test_common_eda_continuous`**

The `test_common_eda_continuous` test in `test_eda_pipeline.py` is now superseded by `test_continuous_eda_pipeline.py` (Task 5). Remove it to avoid duplication.

Keep `test_rule_based_eda` — it will be superseded later when the rule-based research pipeline is built.

Open `tests/integration/feature_validator/test_eda_pipeline.py` and delete the entire `test_common_eda_continuous` function (lines 175–249 approximately). Also remove any imports that are only used by that function if they become unused.

**Step 2: Verify remaining tests still pass**

```bash
pytest tests/integration/feature_validator/test_eda_pipeline.py -v -m integration
```

Expected: `test_rule_based_eda` PASSES or SKIPS (if no data). No `test_common_eda_continuous` in output.

**Step 3: Run the full integration test suite for the feature validator**

```bash
pytest tests/integration/feature_validator/ -v -m integration
```

Expected: all tests PASS or SKIP. No failures.

**Step 4: Commit**

```bash
git add tests/integration/feature_validator/test_eda_pipeline.py
git commit -m "refactor: remove test_common_eda_continuous (superseded by test_continuous_eda_pipeline)"
```

---

## Final Verification

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate

# Unit tests
pytest tests/feature_research/ -v

# Integration tests (PASS or SKIP depending on data availability)
pytest tests/integration/feature_validator/ -v -m integration

# End-to-end script (only if data/ohlc_data/ exists)
python feature_research/continuous_binning/run_eda.py
```

Expected output directory after end-to-end run:
```
feature_research/continuous_binning/results/rsi/
├── lookback_2/
│   ├── {feature_col}/
│   │   └── {8-char-hash}/
│   │       ├── metadata.json
│   │       ├── common_stats.json
│   │       ├── feature_stats.json
│   │       ├── diagnostics.json
│   │       └── plots/
│   │           ├── time_series_fig.png  ... (7 plots)
├── lookback_3/
└── ...  (through lookback_10)
```

> **Note on directory nesting:** `save_eda_report` appends `{feature_name}/{param_hash}` to the provided output dir. The pipeline passes `output_dir / param_label` (e.g., `results/rsi/lookback_5`), so the final path is `results/rsi/lookback_5/{feature_col}/{8-char-hash}/`. The `feature_col` and hash subdirs are created automatically. The researcher navigates by `lookback_N` folder.
