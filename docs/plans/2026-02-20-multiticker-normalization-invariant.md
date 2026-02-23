# Multi-Ticker Normalization Invariant Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add defensive guardrails that prevent raw `log_return` from being used as the binning target when multiple tickers are in the research config. A researcher who accidentally sets `target_col = "log_return"` with 4 tickers should get a clear error — not silently wrong results.

**Architecture:** Two enforcement layers:
1. **Config validation** (`ResearchConfig.__post_init__`): raise if `target_col == "log_return"` and `len(tickers) > 1`.
2. **Data loader validation** (`load_features_for_combo`): raise if the returned target has anomalously high cross-ticker variance, indicating raw (not normalized) returns.

Layer 1 is the primary guard. Layer 2 is a belt-and-suspenders check for code paths that bypass `ResearchConfig`.

**Tech Stack:** `feature_research/in_sample/continuous_binning/config.py`, `feature_research/in_sample/continuous_binning/data_loader.py`, `tests/feature_research/test_config.py`

---

### Context: Why Raw Returns Break Multi-Ticker Binning

When `ContinuousBinningModel.fit(feature, target)` receives a concatenated target across ES, NQ, YM, RTY with raw `log_return`, the return magnitudes are driven by each ticker's absolute price level rather than its risk. ES might have daily log returns around ±0.01 (1%) while in some periods NQ reaches ±0.03 (3%). The bin quantiles learned from this mixture will be dominated by NQ's larger moves. The model will then assign the "high return" bin to ES observations that are merely moderate NQ returns, producing systematically biased signals.

With `log_return_ewsd` or `log_return_atr`, each return is divided by that ticker's vol estimate, making them dimensionless and cross-ticker comparable.

---

### Task 1: Add config validation test

**Files:**
- Test: `tests/feature_research/test_config.py` (new or existing)

**Step 1: Write the failing test**

```python
"""Tests for ResearchConfig validation."""
import pytest
from datetime import datetime
from pathlib import Path
from feature_research.in_sample.continuous_binning.config import (
    ResearchConfig,
    BinningAnalysisConfig,
    PermutationSuiteConfig,
)
from feature_research.walkforward.config import WalkforwardResearchConfig
from utils.core.enums import Ticker, TimeFrame


def _make_config(tickers, target_col):
    """Build a minimal valid ResearchConfig."""
    return ResearchConfig(
        tickers=tickers,
        start=datetime(2000, 1, 1),
        end=datetime(2023, 12, 31),
        bias_spec={"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 5}},
        target_col=target_col,
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=Path("/tmp/test_reports"),
    )


def test_raw_log_return_with_multiple_tickers_raises():
    """Config must reject raw log_return with multiple tickers."""
    with pytest.raises(ValueError, match="log_return.*multiple tickers"):
        _make_config(
            tickers=[Ticker.ES, Ticker.NQ],
            target_col="log_return",
        )


def test_raw_log_return_with_single_ticker_allowed():
    """Single ticker with raw log_return is allowed (no normalization needed)."""
    config = _make_config(tickers=[Ticker.ES], target_col="log_return")
    assert config.target_col == "log_return"


def test_raw_return_with_multiple_tickers_raises():
    """raw_return with multiple tickers is also invalid — same problem."""
    with pytest.raises(ValueError, match="raw_return.*multiple tickers"):
        _make_config(
            tickers=[Ticker.ES, Ticker.NQ, Ticker.YM],
            target_col="raw_return",
        )


def test_log_return_ewsd_with_multiple_tickers_allowed():
    """Vol-normalized target with multiple tickers is valid."""
    config = _make_config(
        tickers=[Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY],
        target_col="log_return_ewsd",
    )
    assert config.target_col == "log_return_ewsd"


def test_log_return_atr_with_multiple_tickers_allowed():
    """ATR-normalized target with multiple tickers is valid."""
    config = _make_config(
        tickers=[Ticker.ES, Ticker.NQ],
        target_col="log_return_atr",
    )
    assert config.target_col == "log_return_atr"
```

**Step 2: Run to verify they fail**

```bash
pytest tests/feature_research/test_config.py -v
```

Expected: all tests FAIL (no validation exists yet in `ResearchConfig`).

---

### Task 2: Add `__post_init__` validation to `ResearchConfig`

**Files:**
- Modify: `feature_research/in_sample/continuous_binning/config.py`

**Step 1: Add `__post_init__`**

`ResearchConfig` is a `@dataclass(frozen=True)`. Add validation:

```python
RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})

@dataclass(frozen=True)
class ResearchConfig:
    # ... existing fields ...

    def __post_init__(self) -> None:
        if self.target_col in RAW_TARGET_COLS and len(self.tickers) > 1:
            raise ValueError(
                f"target_col='{self.target_col}' uses raw (unnormalized) returns but "
                f"{len(self.tickers)} tickers are configured. "
                "Raw returns cannot be compared across tickers with different volatility. "
                "Use 'log_return_ewsd' or 'log_return_atr' for multi-ticker research."
            )
```

Note: `RAW_TARGET_COLS` is defined at module level, not inside the class, to avoid frozen-dataclass issues.

**Step 2: Run tests**

```bash
pytest tests/feature_research/test_config.py -v
```

Expected: all tests PASS.

**Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v --ignore=tests/integration
```

Expected: all unit tests pass. If any test creates a multi-ticker config with `log_return`, update it to use `log_return_atr` or `log_return_ewsd`.

**Step 4: Commit**

```bash
git add feature_research/in_sample/continuous_binning/config.py \
        tests/feature_research/test_config.py
git commit -m "feat(config): validate vol-normalized target for multi-ticker research configs"
```

---

### Task 3: Add belt-and-suspenders check in `load_features_for_combo`

This catches code paths that bypass `ResearchConfig` entirely (e.g., tests or scripts that call `load_features_for_combo` directly).

**Files:**
- Modify: `feature_research/in_sample/continuous_binning/data_loader.py`

**Step 1: Add a cross-ticker std-ratio check**

After the aligned `target_series` is assembled (post-`dropna`), add a diagnostic check. The idea: if `target_col` contains "return" but not "atr" or "ewsd", and there are multiple tickers present in the data, warn (or raise) that normalization may be missing.

```python
# In load_features_for_combo, after building target_series:
_UNNORMALIZED_RETURN_COLS = {"log_return", "raw_return"}

if config.target_col in _UNNORMALIZED_RETURN_COLS:
    # Check if multiple tickers contributed to the target
    unique_tickers = features_df["ticker"].nunique() if "ticker" in features_df.columns else 1
    if unique_tickers > 1:
        raise ValueError(
            f"load_features_for_combo received target_col='{config.target_col}' "
            f"with {unique_tickers} tickers. "
            "Raw return targets must not be mixed across tickers. "
            "Use 'log_return_ewsd' or 'log_return_atr'."
        )
```

**Note:** This check runs even if `ResearchConfig` validation was bypassed. It's a pure defensive guard.

**Step 2: Write test**

```python
def test_load_features_raises_on_raw_return_multiticker(monkeypatch):
    """Data loader raises if raw return is used with multiple tickers."""
    from feature_research.in_sample.continuous_binning import data_loader
    import pandas as pd
    from utils.core.enums import Ticker, TimeFrame

    # Build a minimal multi-ticker features_df stub
    features_df = pd.DataFrame({
        "ticker": ["ES", "NQ"],
        "rsi_signal_D_lookback_5": [45.0, 55.0],
    })
    targets_df = pd.DataFrame({
        "log_return": [0.001, 0.002],
        "ticker": ["ES", "NQ"],
    })

    class FakeConfig:
        target_col = "log_return"
        tickers = [Ticker.ES, Ticker.NQ]
        start = None
        end = None
        use_cache = False

    # Monkeypatch extract_features_for_bias_node to return our stubs
    monkeypatch.setattr(
        "feature_research.in_sample.continuous_binning.data_loader.extract_features_for_bias_node",
        lambda **_: (features_df, targets_df),
    )

    with pytest.raises(ValueError, match="Raw return targets must not be mixed"):
        data_loader.load_features_for_combo(
            single_combo_spec={"module_name": "rsi", "params": {"lookback": 5}, "timeframes": [TimeFrame.D]},
            config=FakeConfig(),
        )
```

**Step 3: Run test**

```bash
pytest tests/feature_research/test_config.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add feature_research/in_sample/continuous_binning/data_loader.py \
        tests/feature_research/test_config.py
git commit -m "feat(data_loader): guard against raw returns with multi-ticker datasets"
```

---

### Task 4: Add a warning to `ContinuousBinningModel.fit` docstring

Not a code change — document the invariant so future contributors see it immediately when reading the model.

**Files:**
- Modify: `feature_selection/base_models/continuous_binning.py`

In the `fit` method docstring, add:

```python
"""
...
Parameters
----------
feature_data : pd.Series
    ...
target : pd.Series
    The prediction target. IMPORTANT: when this model is fit on data from
    multiple tickers, target MUST be volatility-normalized (e.g., log_return / EWSD
    or log_return / ATR). Using raw log returns across tickers with different
    volatility will bias the bin quantiles toward the most volatile ticker.
    Use 'log_return_ewsd' or 'log_return_atr' targets from the feature extractor.
...
"""
```

**Commit:**

```bash
git add feature_selection/base_models/continuous_binning.py
git commit -m "docs(binning): document vol-normalization invariant for multi-ticker targets"
```

---

### Summary of Changes

| Layer | File | Guard |
|---|---|---|
| Config | `feature_research/in_sample/continuous_binning/config.py` | `__post_init__` raises if raw target + >1 ticker |
| Data loader | `feature_research/in_sample/continuous_binning/data_loader.py` | Runtime check in `load_features_for_combo` |
| Model doc | `feature_selection/base_models/continuous_binning.py` | Docstring documents the invariant |

### Error Messages (Exact)

When violated, the researcher sees:

**Config-level:**
```
ValueError: target_col='log_return' uses raw (unnormalized) returns but 4 tickers are configured.
Raw returns cannot be compared across tickers with different volatility.
Use 'log_return_ewsd' or 'log_return_atr' for multi-ticker research.
```

**Data-loader-level:**
```
ValueError: load_features_for_combo received target_col='log_return' with 4 tickers.
Raw return targets must not be mixed across tickers.
Use 'log_return_ewsd' or 'log_return_atr'.
```

These are clear, actionable, and tell the researcher exactly what to change.
