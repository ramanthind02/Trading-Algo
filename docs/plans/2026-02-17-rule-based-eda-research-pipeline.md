# Rule-Based EDA Research Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a config-driven research pipeline for rule-based feature EDA that mirrors `feature_research/continuous_binning/` and outputs per-param-combo reports to disk.

**Architecture:** `feature_research/rule_based/` is a self-contained package with its own `config.py`, `data_loader.py`, `pipeline.py`, and a thin `run_eda.py` entry point. Integration tests import `run_rule_based_eda_pipeline` directly so the researcher script and tests share the same codepath. The old `feature_research/rule-based/` directory (hyphen, empty, not git-tracked) is removed and replaced.

**Tech Stack:** Python 3.11, pandas, matplotlib (Agg backend), pytest, `feature_selection.eda.eda_reporter.run_eda_for_rule_based_feature`, `feature_extraction.feature_extractor.extract_features_for_bias_node`, `utils.cache_manager.CacheManager`

---

## Context for the implementer

This mirrors `feature_research/continuous_binning/` almost exactly. Key differences:

| Aspect | continuous_binning | rule_based |
|--------|-------------------|------------|
| EDA function | `run_eda_for_continuous_feature` | `run_eda_for_rule_based_feature` |
| Report type | `ContinuousEDAReport` | `RuleBasedEDAReport` |
| EDAConfig | `EDAConfig(n_bins=15, rolling_window=...)` | `EDAConfig(rolling_window=..., bootstrap_iterations=500)` |
| Default module | `rsi` | `rsi_signal` |
| Default params | `lookback: [2..10]` | `rsi_period: [2, 3, 5, 7]` + fixed oversold/overbought/exit params |
| Summary line | pearson / tau / trend | per-level Sharpe for levels -1, 0, 1 |
| Plot count | 7 (3 common + 4 continuous) | 5 (3 common + 2 rule-based) |
| Rule-based plots | — | `level_plot_fig.png`, `transition_heatmap_fig.png` |

**Key existing files to read before implementing:**
- `feature_research/continuous_binning/config.py` — exact model for Task 1
- `feature_research/continuous_binning/data_loader.py` — exact model for Task 2
- `feature_research/continuous_binning/pipeline.py` — exact model for Task 3
- `feature_research/continuous_binning/__init__.py` — exact model for Task 1
- `tests/feature_research/test_config.py` — exact model for Task 1 unit tests
- `tests/feature_research/test_data_loader.py` — exact model for Task 2 unit tests
- `tests/integration/feature_validator/test_continuous_eda_pipeline.py` — exact model for Task 5
- `tests/integration/feature_validator/test_eda_pipeline.py` — file to refactor in Task 6

**Venv:** Always activate before running any Python command:
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
```

---

## Task 1: `config.py` + `__init__.py` + remove stale directory

**Files:**
- Remove: `feature_research/rule-based/` (empty directory, not git-tracked)
- Create: `feature_research/rule_based/__init__.py`
- Create: `feature_research/rule_based/config.py`
- Create: `tests/feature_research/test_rule_based_config.py`

**Step 1: Remove the old hyphenated directory**

```bash
rmdir feature_research/rule-based
```

Expected: directory removed (it's empty).

**Step 2: Write the failing tests**

Create `tests/feature_research/test_rule_based_config.py`:

```python
from datetime import datetime
from pathlib import Path

from feature_research.rule_based.config import RuleBasedResearchConfig, load_config
from utils.enums import Ticker


def test_load_config_returns_rule_based_research_config():
    config = load_config()
    assert isinstance(config, RuleBasedResearchConfig)


def test_load_config_defaults():
    config = load_config()
    assert Ticker.ES in config.tickers
    assert Ticker.NQ in config.tickers
    assert config.start == datetime(2020, 1, 1)
    assert config.end == datetime(2024, 12, 31)
    assert config.bias_spec["module_name"] == "rsi_signal"
    assert isinstance(config.bias_spec["params"]["rsi_period"], list)
    assert config.target_col == "log_return"
    assert config.use_cache is True
    assert config.populate_cache is True


def test_reports_dir_includes_module_name():
    config = load_config()
    assert config.bias_spec["module_name"] in str(config.reports_dir)
    assert "rule_based" in str(config.reports_dir)
```

**Step 3: Run the tests — expect FAIL**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/feature_research/test_rule_based_config.py -v
```

Expected: `ModuleNotFoundError` (module not yet created).

**Step 4: Create `feature_research/rule_based/__init__.py`**

```python
from feature_research.rule_based.config import RuleBasedResearchConfig, load_config
from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline

__all__ = ["RuleBasedResearchConfig", "load_config", "run_rule_based_eda_pipeline"]
```

Note: this will import `pipeline.py` which doesn't exist yet. Create a stub pipeline temporarily if needed, or create the `__init__.py` after Task 3.

**Step 4b: Create `feature_research/rule_based/config.py`**

```python
"""Researcher-editable configuration for the rule-based EDA pipeline.

Edit the values in load_config() to customise tickers, date range, and bias node specs.
All other scripts import from here — change once, apply everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_RB_DIR = _FEATURE_RESEARCH_DIR / "rule_based"


@dataclass(frozen=True)
class RuleBasedResearchConfig:
    """All researcher-editable settings for rule-based feature EDA.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict[str, Any]
        Bias-node specification. ``params`` values may be lists for grid search.
        Example::

            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": [2, 3, 5, 7],
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
    target_col : str
        Column from ``targets_df`` to use as the prediction target.
    strategy : str
        Passed to the EDA summary label only.
    use_cache : bool
        Whether to use pre-computed caches for feature extraction.
    populate_cache : bool
        If True, run ``CacheManager.populate_cache()`` before extraction.
    reports_dir : Path
        Root output directory for EDA reports.
        Default: ``feature_research/rule_based/results/{module_name}/``
    """

    tickers: list[Ticker]
    start: datetime
    end: datetime
    bias_spec: dict[str, Any]
    target_col: str
    strategy: str
    use_cache: bool
    populate_cache: bool
    reports_dir: Path


def load_config() -> RuleBasedResearchConfig:
    """Return the default research configuration.

    **Edit this function** to customise tickers, dates, and bias node specs.
    All pipeline scripts import from here.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,  # E-Mini S&P 500
        Ticker.NQ,  # E-Mini Nasdaq-100
    ]

    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)

    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {
            "rsi_period": [2, 3, 5, 7],
            "oversold": 25.0,
            "overbought": 65.0,
            "strategy_mode": "long",
            "exit_policy": "threshold_or_bars",
            "exit_bars": 5,
        },
    }

    target_col = "log_return"
    strategy = "long"

    use_cache = True
    populate_cache = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    module_name = bias_spec["module_name"]
    reports_dir = _RB_DIR / "results" / module_name

    return RuleBasedResearchConfig(
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

**Step 5: Create a stub `pipeline.py` so `__init__.py` imports without error**

Create `feature_research/rule_based/pipeline.py` with just the function signature:

```python
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_research.rule_based.config import RuleBasedResearchConfig


def run_rule_based_eda_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    raise NotImplementedError
```

**Step 6: Run the tests — expect PASS**

```bash
pytest tests/feature_research/test_rule_based_config.py -v
```

Expected: all 3 tests PASS.

**Step 7: Commit**

```bash
git add feature_research/rule_based/ tests/feature_research/test_rule_based_config.py
git commit -m "feat: add RuleBasedResearchConfig and load_config"
```

---

## Task 2: `data_loader.py`

**Files:**
- Create: `feature_research/rule_based/data_loader.py`
- Create: `tests/feature_research/test_rule_based_data_loader.py`

**Step 1: Write the failing tests**

Create `tests/feature_research/test_rule_based_data_loader.py`:

```python
from utils.enums import TimeFrame
from feature_research.rule_based.data_loader import expand_bias_specs, param_combo_label


def test_expand_bias_specs_single_param():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": [2, 3]},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 2
    assert result[0]["params"] == {"rsi_period": 2}
    assert result[1]["params"] == {"rsi_period": 3}
    assert all(r["module_name"] == "rsi_signal" for r in result)


def test_expand_bias_specs_grid():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": [2, 3], "oversold": [20.0, 25.0]},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 4  # 2 × 2


def test_expand_bias_specs_scalar_params():
    bias_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": {"rsi_period": 2, "oversold": 25.0},
    }
    result = expand_bias_specs(bias_spec)
    assert len(result) == 1
    assert result[0]["params"] == {"rsi_period": 2, "oversold": 25.0}


def test_param_combo_label_single():
    assert param_combo_label({"rsi_period": 2}) == "rsi_period_2"


def test_param_combo_label_multi():
    label = param_combo_label({"rsi_period": 2, "oversold": 25.0})
    assert label == "oversold_25.0__rsi_period_2"  # sorted alphabetically
```

**Step 2: Run the tests — expect FAIL**

```bash
pytest tests/feature_research/test_rule_based_data_loader.py -v
```

Expected: `ImportError` (data_loader not created yet).

**Step 3: Create `feature_research/rule_based/data_loader.py`**

This is a direct copy of `feature_research/continuous_binning/data_loader.py` with one change: the `TYPE_CHECKING` import points to `RuleBasedResearchConfig`.

```python
"""Data loading and cache management helpers for the rule-based EDA research pipeline."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from feature_research.rule_based.config import RuleBasedResearchConfig

from feature_extraction.feature_extractor import extract_features_for_bias_node
from utils.cache_manager import CacheManager
from utils.enums import TimeFrame


def expand_bias_specs(bias_spec: dict[str, Any]) -> list[dict[str, Any]]:
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


def param_combo_label(combo: dict[str, Any]) -> str:
    """Return a human-readable folder name for a param combo dict.

    Examples
    --------
    >>> param_combo_label({"rsi_period": 2})
    'rsi_period_2'
    >>> param_combo_label({"rsi_period": 2, "oversold": 25.0})
    'oversold_25.0__rsi_period_2'
    """
    parts = [f"{k}_{v}" for k, v in sorted(combo.items())]
    return "__".join(parts)


def populate_cache_if_needed(config: "RuleBasedResearchConfig") -> None:
    """Populate the feature cache if config.populate_cache is True.

    Safe to call even if cache already exists — ``overwrite_existing=False``
    means only missing entries are computed.
    """
    if not config.populate_cache:
        return

    project_root = Path(__file__).resolve().parents[2]
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        print(
            f"[data_loader] WARNING: candle directory not found at {candle_dir}. "
            "Skipping cache population."
        )
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
    single_combo_spec: dict[str, Any],
    config: "RuleBasedResearchConfig",
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

    target_col_name = (
        config.target_col
        if config.target_col in targets_df.columns
        else [c for c in targets_df.columns if c != "ticker"][0]
    )

    aligned = pd.DataFrame(
        {"feature": features_df[feature_col], "target": targets_df[target_col_name]}
    ).dropna()

    if aligned.empty:
        return None

    return aligned["feature"], aligned["target"], feature_col
```

**Step 4: Run the tests — expect PASS**

```bash
pytest tests/feature_research/test_rule_based_data_loader.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add feature_research/rule_based/data_loader.py tests/feature_research/test_rule_based_data_loader.py
git commit -m "feat: add rule_based data_loader helpers"
```

---

## Task 3: `pipeline.py`

**Files:**
- Modify: `feature_research/rule_based/pipeline.py` (replace stub from Task 1)

**Step 1: Replace the stub with the full implementation**

```python
# feature_research/rule_based/pipeline.py
"""Core EDA pipeline for rule-based feature research.

Entry point for tests and scripts alike — import ``run_rule_based_eda_pipeline``
rather than duplicating this logic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (safe for scripts and tests)

if TYPE_CHECKING:
    from feature_research.rule_based.config import RuleBasedResearchConfig

from feature_research.rule_based.data_loader import (
    expand_bias_specs,
    load_features_for_combo,
    param_combo_label,
    populate_cache_if_needed,
)
from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import run_eda_for_rule_based_feature, save_eda_report
from utils.enums import TimeFrame


def _normalize_timeframe(bias_spec: dict[str, Any], fallback: TimeFrame = TimeFrame.D) -> TimeFrame:
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def run_rule_based_eda_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    """Run the full rule-based feature EDA pipeline for every param combo in config.

    For each param combo:
    1. Extract feature + target data (cache-backed).
    2. Build ``EDAMetadata`` and ``EDAConfig``.
    3. Run ``run_eda_for_rule_based_feature`` to produce a ``RuleBasedEDAReport``.
    4. Save the report under ``output_dir / param_label /``.
    5. Print a one-line summary.

    Parameters
    ----------
    config : RuleBasedResearchConfig
        Researcher-defined settings (tickers, dates, bias_spec, cache flags).
    output_dir : Path
        Root directory for output reports. Created if it does not exist.
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
    print(f"Rule-Based EDA Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Target  : {config.target_col}  |  Strategy: {config.strategy}")
    print(f"Combos  : {len(expanded)}")
    print(f"Output  : {output_dir}")
    print(f"{'='*64}\n")

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
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
        eda_config = EDAConfig(rolling_window=rolling_window, bootstrap_iterations=500)

        report = run_eda_for_rule_based_feature(feature, target, timestamps, metadata, eda_config)

        combo_output_dir = output_dir / label
        combo_output_dir.mkdir(parents=True, exist_ok=True)

        saved_path = save_eda_report(report=report, output_dir=combo_output_dir, overwrite=True)
        results[label] = saved_path

        stats_by_level = report.rule_stats.per_level_stats.stats_by_level
        level_parts = "  ".join(
            f"L[{lvl}]: sharpe={stats_by_level[lvl].sharpe:+.2f}"
            if lvl in stats_by_level
            else f"L[{lvl}]: n/a"
            for lvl in [-1, 0, 1]
        )
        viable = "VIABLE" if report.diagnostics.is_viable else f"FLAGS({len(report.diagnostics.red_flags)})"
        warnings_count = len(report.diagnostics.warnings)

        print(
            f"  [{label}] n={len(feature):,}  {level_parts}  {viable}  warnings={warnings_count}"
        )

    print(f"\nDone. {len(results)}/{len(expanded)} combos succeeded -> {output_dir}\n")
    return results
```

**Step 2: Smoke-test the import (not a real test, just sanity)**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
python -c "from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add feature_research/rule_based/pipeline.py
git commit -m "feat: implement run_rule_based_eda_pipeline"
```

---

## Task 4: `run_eda.py`

**Files:**
- Create: `feature_research/rule_based/run_eda.py`

**Step 1: Create the entry point**

```python
"""Entry point for the rule-based feature EDA research pipeline.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/rule_based/run_eda.py

Results are written to feature_research/rule_based/results/{module_name}/.
Edit feature_research/rule_based/config.py to change tickers, dates, or bias specs.
"""
from feature_research.rule_based.config import load_config
from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline

if __name__ == "__main__":
    config = load_config()
    results = run_rule_based_eda_pipeline(config, config.reports_dir)
    print(f"EDA complete. {len(results)} param combos written to {config.reports_dir}")
```

**Step 2: Verify import works**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
python -c "import feature_research.rule_based.run_eda; print('OK')"
```

Expected: `OK` (the `if __name__ == "__main__"` guard prevents execution on import).

**Step 3: Commit**

```bash
git add feature_research/rule_based/run_eda.py
git commit -m "feat: add run_eda.py entry point for rule-based EDA"
```

---

## Task 5: Integration test

**Files:**
- Create: `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`

**Step 1: Write the integration test**

Create `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`:

```python
# tests/integration/feature_validator/test_rule_based_eda_pipeline.py
"""Integration tests for the rule-based EDA research pipeline.

Imports run_rule_based_eda_pipeline directly from feature_research so any
regression in the researcher's script is immediately caught here.

Default config: rsi_signal rsi_period=2, Ticker.ES, 2020-2023.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from feature_research.rule_based.config import RuleBasedResearchConfig
from feature_research.rule_based.pipeline import run_rule_based_eda_pipeline
from utils.enums import Ticker, TimeFrame


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _skip_if_no_data() -> None:
    candle_dir = _project_root() / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")


@pytest.mark.integration
def test_rule_based_eda_pipeline_smoke(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_period: int = 2,
) -> None:
    """Smoke test: single rsi_signal param combo, ES daily, 2020-2023.

    Verifies:
    - Pipeline runs without exception
    - Exactly one result entry returned
    - Output directory contains expected JSON + plot files

    All key config inputs are exposed as parameters so researchers can call
    this directly with custom values for interactive validation.
    """
    _skip_if_no_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RuleBasedResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": rsi_period,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_rule_based_eda_pipeline(config, Path(tmpdir))

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        report_path = list(results.values())[0]
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
            "level_plot_fig.png",
            "transition_heatmap_fig.png",
        ]
        for plot_file in expected_plots:
            assert (plots_dir / plot_file).exists(), f"Missing plot: {plot_file}"


@pytest.mark.integration
def test_rule_based_eda_pipeline_multi_combo(
    tickers: list[Ticker] | None = None,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2023, 12, 31),
    rsi_periods: list[int] | None = None,
) -> None:
    """Multi-combo smoke test: rsi_period=[2, 3], ES daily, 2020-2023.

    Verifies:
    - Pipeline produces one result per param combo
    - Separate output directories created for each combo

    Exposed as parameters for researcher-driven exploration.
    """
    _skip_if_no_data()
    rsi_periods = rsi_periods or [2, 3]

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RuleBasedResearchConfig(
            tickers=tickers or [Ticker.ES],
            start=start,
            end=end,
            bias_spec={
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": rsi_periods,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            target_col="log_return",
            strategy="long",
            use_cache=True,
            populate_cache=True,
            reports_dir=Path(tmpdir),
        )
        results = run_rule_based_eda_pipeline(config, Path(tmpdir))

        assert len(results) == len(rsi_periods), (
            f"Expected {len(rsi_periods)} results, got {len(results)}"
        )
```

**Step 2: Run the integration tests**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/integration/feature_validator/test_rule_based_eda_pipeline.py -v
```

Expected: Both tests PASS (or SKIP if `data/ohlc_data/` is missing — skip is the correct behavior).

**Step 3: Commit**

```bash
git add tests/integration/feature_validator/test_rule_based_eda_pipeline.py
git commit -m "test: add integration test for rule-based EDA research pipeline"
```

---

## Task 6: Remove superseded test

**Files:**
- Delete: `tests/integration/feature_validator/test_eda_pipeline.py`

`test_eda_pipeline.py` previously contained both `test_common_eda_continuous` (already removed) and `test_rule_based_eda`. Now that `test_rule_based_eda_pipeline.py` supersedes `test_rule_based_eda`, the file is empty of useful tests and should be deleted.

**Step 1: Verify `test_eda_pipeline.py` only contains `test_rule_based_eda`**

Read `tests/integration/feature_validator/test_eda_pipeline.py` and confirm `test_rule_based_eda` is the only test function. (It should be — `test_common_eda_continuous` was removed in a prior commit.)

**Step 2: Delete the file**

```bash
git rm tests/integration/feature_validator/test_eda_pipeline.py
```

**Step 3: Run the full test suite to confirm nothing is broken**

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/ -v --ignore=tests/integration
```

Then run integration tests:

```bash
pytest tests/integration/ -v
```

Expected: All tests pass (integration tests skip gracefully if no data).

**Step 4: Commit**

```bash
git commit -m "refactor: remove test_rule_based_eda (superseded by test_rule_based_eda_pipeline)"
```

---

## Final verification

After all tasks:

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate

# All unit tests pass
pytest tests/feature_research/ -v

# Integration tests pass or skip
pytest tests/integration/feature_validator/ -v

# Import smoke test
python -c "from feature_research.rule_based import RuleBasedResearchConfig, load_config, run_rule_based_eda_pipeline; print('imports OK')"
```
