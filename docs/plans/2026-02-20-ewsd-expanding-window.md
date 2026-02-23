# EWSD Expanding Window Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `EWSDNode` to (1) use an expanding window for the long-run component so it never outputs NaN or a hardcoded prior; (2) align its `long_run_window` default with `fast_volatility.py` (2520 days, not 252); (3) output valid estimates from the second observation onwards; (4) ensure the EWSD values fed into `log_return_ewsd` targets are never NaN for rows with sufficient data.

**Architecture:** Three separate issues in `nodes/ewsd.py`:
1. `deque(maxlen=252)` caps the long-run window at 1 year. Should be `deque(maxlen=2520)` (10 years). Both Carver's methodology and `fast_volatility.py` use 2520.
2. `if len(self.returns_history) >= 20:` — uses hardcoded prior `sigma_long = 0.01` for first 20 observations. Should output a real estimate (from the short-run EWMA only, or expanding std) from observation 2.
3. The blending logic is fine but relies on a stable `sigma_long` initialization. Consistency check needed.

The fix must not change the blending formula or annualization factor (multiply by 16 = sqrt(256)). It also must not introduce NaN in `ewsd_annual_pct` output.

**Tech Stack:** `nodes/ewsd.py`, `utils/fast_volatility.py` (for reference), `tests/` for EWSDNode.

---

### Task 1: Add tests that document the current broken behavior

**Files:**
- Test: `tests/nodes/test_ewsd.py` (new or existing)

**Step 1: Write failing tests**

```python
"""Tests for EWSDNode expanding window and no-NaN behavior."""
import numpy as np
import pytest
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes.ewsd import EWSDNode


def _make_candle(ticker, tf, close, open_=None):
    """Helper to build a minimal Candle."""
    return Candle(
        ticker=ticker,
        timeframe=tf,
        open=open_ or close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1000,
        datetime=None,  # not used in compute_candle
    )


def test_ewsd_outputs_valid_value_from_second_candle():
    """Node must output a valid (non-NaN, non-hardcoded 1%) estimate from candle 2."""
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    closes = [4000.0, 4010.0, 4005.0]  # only 3 candles

    results = []
    for close in closes:
        candle = _make_candle(Ticker.ES, TimeFrame.D, close)
        out = node._compute_candle(candle)
        results.append(out[0])  # ewsd_daily_pct

    # Candle 1: should be nonzero default
    assert results[0] > 0, "First candle must have a non-zero default"
    # Candle 2: must not return exactly 1.0 (the hardcoded 0.01 sigma_long * 100)
    # It should reflect the actual return from candle 1 to candle 2
    # Return = (4010-4000)/4000 = 0.0025
    # EWMA variance = (0.0025)^2 = 6.25e-6
    # sigma_short = sqrt(6.25e-6) = 0.0025
    # Result should be near 0.0025, not 0.01
    assert abs(results[1] / 100.0 - 0.0025) < 0.001, \
        f"Second candle sigma should reflect actual return, got {results[1]/100.0:.6f}"


def test_ewsd_long_run_window_is_2520():
    """long_run_window default must be 2520 (10 years), not 252 (1 year)."""
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    assert node.long_run_window == 2520, \
        f"long_run_window should be 2520, got {node.long_run_window}"
    assert node.returns_history.maxlen == 2520, \
        f"deque maxlen should be 2520, got {node.returns_history.maxlen}"


def test_ewsd_never_outputs_nan():
    """Output must be finite for any sequence of valid prices."""
    np.random.seed(99)
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    price = 4000.0
    for _ in range(100):
        price *= (1 + np.random.normal(0, 0.01))
        candle = _make_candle(Ticker.ES, TimeFrame.D, price)
        out = node._compute_candle(candle)
        assert np.isfinite(out[0]), f"ewsd_daily_pct is NaN/Inf at price={price}"
        assert np.isfinite(out[1]), f"ewsd_annual_pct is NaN/Inf at price={price}"
        assert out[0] > 0, "ewsd_daily_pct must be positive"


def test_ewsd_expands_before_2520():
    """
    Long-run std must grow with expanding window before 2520 observations.
    After 30 candles, sigma_long should reflect actual 30-day std, not just the prior.
    """
    np.random.seed(7)
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    price = 4000.0
    daily_vol = 0.02  # 2% daily vol

    for _ in range(50):
        price *= (1 + np.random.normal(0, daily_vol))
        candle = _make_candle(Ticker.ES, TimeFrame.D, price)
        out = node._compute_candle(candle)

    # After 50 observations with 2% daily vol, sigma_long should be ~2%
    # (within a factor of 2 given small sample)
    sigma_long_pct = node.sigma_long * 100  # stored as decimal, output as pct
    # Actually sigma_long is stored as a decimal, check directly
    assert 0.005 < node.sigma_long < 0.08, \
        f"sigma_long after 50 obs of 2% daily vol = {node.sigma_long:.4f}, expected ~0.02"


def test_ewsd_matches_fast_volatility_blending():
    """
    After warm-up, EWSDNode blending (70/30) must match fast_volatility.py logic.
    Both use lambda=0.06061, 70% short-run EWMA, 30% long-run std.
    """
    import numpy as np
    from utils.compute.fast_volatility import compute_ewsd_annualized_from_closes

    np.random.seed(3)
    closes = 4000.0 * np.cumprod(1 + np.random.normal(0, 0.01, 500))

    # fast_volatility gives a single final value over all closes
    expected_annual = compute_ewsd_annualized_from_closes(closes, long_run_window=2520)

    # EWSDNode gives per-candle values; the last one should match closely
    node = EWSDNode(ticker=Ticker.ES, tf=TimeFrame.D)
    for close in closes:
        candle = _make_candle(Ticker.ES, TimeFrame.D, close)
        out = node._compute_candle(candle)

    node_annual_pct = out[1]  # ewsd_annual_pct
    node_annual = node_annual_pct / 100.0

    # Allow 20% relative error (small differences from different EWMA initialization)
    rel_err = abs(node_annual - expected_annual) / expected_annual
    assert rel_err < 0.20, \
        f"EWSDNode ({node_annual:.4f}) deviates >20% from fast_volatility ({expected_annual:.4f})"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/nodes/test_ewsd.py -v
```

Expected: `test_ewsd_long_run_window_is_2520` FAILS (current default is 252). `test_ewsd_outputs_valid_value_from_second_candle` likely FAILS. Others may pass or fail.

---

### Task 2: Fix `EWSDNode` — expand the window and remove the hardcoded prior

**Files:**
- Modify: `nodes/ewsd.py`

**Step 1: Change `long_run_window` default from 252 to 2520**

```python
# Before:
long_run_window: int = 252,      # 1 year for long-run estimate

# After:
long_run_window: int = 2520,     # 10 years of daily data (Carver's methodology)
```

**Step 2: Remove the `>= 20` guard and use an expanding estimate instead**

```python
# Before (lines 156-162):
if len(self.returns_history) >= 20:  # Need minimum data
    arr = np.array(self.returns_history, dtype=np.float64)
    n_ret = len(arr)
    if CYTHON_NODES_AVAILABLE and compute_stddev_sample_fast is not None:
        self.sigma_long = compute_stddev_sample_fast(arr, n_ret)
    else:
        self.sigma_long = np.std(self.returns_history, ddof=1)

# After:
# Expanding window: compute sigma_long from whatever data is available (>= 2 obs).
# If only 1 observation, fall back to the short-run estimate — no hardcoded prior.
n_obs = len(self.returns_history)
if n_obs >= 2:
    arr = np.array(self.returns_history, dtype=np.float64)
    if CYTHON_NODES_AVAILABLE and compute_stddev_sample_fast is not None:
        self.sigma_long = compute_stddev_sample_fast(arr, len(arr))
    else:
        self.sigma_long = float(np.std(arr, ddof=1))
elif n_obs == 1:
    # Only one return so far — use it directly as our vol estimate
    self.sigma_long = abs(float(self.returns_history[0]))
# If n_obs == 0 (first candle after the initial one), sigma_long keeps
# its prior value from initialization. We now initialize it to something
# sensible (see Step 3).
```

**Step 3: Replace the hardcoded initial prior with a reasonable default**

The first candle sets `sigma_long = 0.01` (hardcoded 1% daily). This is used until `n_obs >= 2`. After the fix, it's used only for the very first return observation (n_obs == 0 after the first return). Keep 1% as a reasonable initial guess but document it clearly:

```python
# Initial estimates (only used for the very first observation before any real data)
self.sigma_long: float = 0.01  # 1% daily vol prior — replaced after first return seen
```

This is acceptable: it's only used for 1 observation.

**Step 4: Fix the annualization comment**

The current code says:
```python
# Multiply by 16 (sqrt(256)) to annualize daily volatility
```
`sqrt(252) ≈ 15.87`, `sqrt(256) = 16.0`. Carver uses 16 as a round number. Keep the factor 16 but correct the comment:
```python
# Annualize: multiply by 16 (Carver's rounded sqrt(256) ≈ sqrt(252) approximation)
```

**Step 5: Run failing tests**

```bash
pytest tests/nodes/test_ewsd.py -v
```

Expected: all tests PASS.

**Step 6: Run full test suite to check for regressions**

```bash
pytest tests/ -v --ignore=tests/integration
```

Expected: all unit tests pass.

**Step 7: Commit**

```bash
git add nodes/ewsd.py tests/nodes/test_ewsd.py
git commit -m "fix(ewsd): use expanding window (2520d), remove hardcoded prior, emit valid vol from obs 2"
```

---

### Task 3: Verify `log_return_ewsd` targets no longer drop excess rows due to early NaN

The feature extractor shifts EWSD by -1 before normalizing returns. If early EWSD values are NaN (or the initial prior is too different from real values), rows will be dropped unnecessarily.

**Step 1: Write a verification test**

```python
def test_ewsd_targets_no_excessive_nan_drop():
    """
    When EWSD is fixed to have no NaN after obs 2, log_return_ewsd should
    have the same number of valid rows as log_return_atr.
    """
    import os
    if not os.path.exists("data/ohlc_data"):
        pytest.skip("Cache not available")

    from feature_research.continuous_binning.config import load_config
    from feature_research.continuous_binning.data_loader import load_features_for_combo, expand_bias_specs

    config = load_config()
    import dataclasses
    config_ewsd = dataclasses.replace(config, target_col="log_return_ewsd")
    config_atr = dataclasses.replace(config, target_col="log_return_atr")

    single_spec = expand_bias_specs(config.bias_spec)[0]

    data_ewsd = load_features_for_combo(single_spec, config_ewsd)
    data_atr = load_features_for_combo(single_spec, config_atr)

    assert data_ewsd is not None
    assert data_atr is not None

    _, target_ewsd, _ = data_ewsd
    _, target_atr, _ = data_atr

    # EWSD targets should not drop significantly more rows than ATR targets
    n_ewsd = len(target_ewsd)
    n_atr = len(target_atr)
    ratio = n_ewsd / n_atr
    assert ratio > 0.95, \
        f"EWSD dropped {n_atr - n_ewsd} more rows than ATR ({ratio:.2%} coverage)"
```

**Step 2: Run test**

```bash
pytest tests/nodes/test_ewsd.py::test_ewsd_targets_no_excessive_nan_drop -v
```

Expected: PASS.

**Step 3: Commit if passing**

```bash
git commit -m "test(ewsd): verify log_return_ewsd row coverage matches log_return_atr"
```

---

### Summary of Changes

| File | Change |
|---|---|
| `nodes/ewsd.py` | `long_run_window` default: `252` → `2520` |
| `nodes/ewsd.py` | `deque(maxlen=...)` from 252 to 2520 |
| `nodes/ewsd.py` | Remove `>= 20` guard, use expanding `std` from obs 2 |
| `nodes/ewsd.py` | Update comment: annualization factor |
| `tests/nodes/test_ewsd.py` | 5 new tests covering NaN, window, consistency |

### Consistency Note

`utils/fast_volatility.py` uses `long_run_window=2520` (already correct). After this fix, `EWSDNode` and `fast_volatility.compute_ewsd_annualized_from_closes` will use the same window. The slight difference in their outputs is expected (online vs batch computation, EWMA initialization).
