# Enhanced Top-K Ensemble Selection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the naive smoothed-rank top-k selection in the walkforward runner with a three-objective algorithm (stability + robustness + diversity) that filters by trade frequency, scores by cross-block consistency, and selects via greedy diversity-aware ranking.

**Architecture:** A new pure-function module `feature_research/walkforward/top_k_selection.py` holds all scoring and selection logic. `_build_fold_scores()` in `runner.py` calls into this module when `config.use_enhanced_selection = True`. The legacy path is preserved as the default. All new logic is unit-tested in isolation before integration.

**Tech Stack:** Python 3.9+, `pandas`, `numpy`. No new dependencies.

**Spec:** [`docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`](../library/Feature_selection/Parameter%20Sensitivity/top_k_ensemble_selection.md)

---

## Read First

Before writing any code, read these files:

- `feature_research/walkforward/runner.py` — the integration target; understand `_build_fold_scores()` and `FoldScoreRow`
- `feature_research/walkforward/config.py` — `WalkforwardResearchConfig` dataclass
- `utils/grid_smoothing.py` — `add_smoothed_objective()` — produces the smoothed_objective values this algorithm consumes
- `tests/feature_research/walkforward/test_runner.py` — existing test patterns to follow

---

## Task 1: Create `top_k_selection.py` with `compute_trade_frequency`

**Files:**
- Create: `feature_research/walkforward/top_k_selection.py`
- Test: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** Trade frequency is the fraction of bars in a signal series where the signal is non-zero. A signal series is a `pd.Series` of float values (positions or returns scaled by position) aligned to the training window's DatetimeIndex.

**Step 1: Write the failing test**

```python
# tests/feature_research/walkforward/test_top_k_selection.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from feature_research.walkforward.top_k_selection import compute_trade_frequency


def _make_signal(values: list[float]) -> pd.Series:
    idx = pd.date_range("2000-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_trade_frequency_all_nonzero():
    signal = _make_signal([1.0, -1.0, 1.0, 1.0])
    assert compute_trade_frequency(signal) == pytest.approx(1.0)


def test_trade_frequency_half():
    signal = _make_signal([1.0, 0.0, 1.0, 0.0])
    assert compute_trade_frequency(signal) == pytest.approx(0.5)


def test_trade_frequency_all_zero():
    signal = _make_signal([0.0, 0.0, 0.0])
    assert compute_trade_frequency(signal) == pytest.approx(0.0)


def test_trade_frequency_empty_series():
    signal = pd.Series([], dtype=float)
    assert compute_trade_frequency(signal) == pytest.approx(0.0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
# feature_research/walkforward/top_k_selection.py
"""
Enhanced top-k ensemble selection: three-objective scoring and greedy diversity selection.

Spec: docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trade_frequency(signal: pd.Series) -> float:
    """Return fraction of bars where signal is non-zero.

    Parameters
    ----------
    signal : pd.Series
        Position or return series for a single param combo over the training window.

    Returns
    -------
    float
        Fraction of bars in [0, 1] where abs(signal) > 0.
    """
    if signal.empty:
        return 0.0
    return float((signal != 0).sum() / len(signal))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py::test_trade_frequency_all_nonzero tests/feature_research/walkforward/test_top_k_selection.py::test_trade_frequency_half tests/feature_research/walkforward/test_top_k_selection.py::test_trade_frequency_all_zero tests/feature_research/walkforward/test_top_k_selection.py::test_trade_frequency_empty_series -v
```
Expected: 4 PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add compute_trade_frequency to top_k_selection"
```

---

## Task 2: Add `compute_robustness_score`

**Files:**
- Modify: `feature_research/walkforward/top_k_selection.py`
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** Given a list of per-block objective metric values, return `mean / (1 + cv)`. The coefficient of variation (cv) is `std / |mean|`. If mean ≤ 0, return 0. If all blocks have identical values, cv = 0 and robustness = mean.

**Step 1: Write the failing tests**

```python
# Append to tests/feature_research/walkforward/test_top_k_selection.py
from feature_research.walkforward.top_k_selection import compute_robustness_score


def test_robustness_score_perfectly_consistent():
    # cv = 0 → score = mean
    assert compute_robustness_score([0.6, 0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.6)


def test_robustness_score_high_variance_penalised():
    # same mean, higher variance → lower score
    low_var = compute_robustness_score([0.6, 0.58, 0.62, 0.59, 0.61])
    high_var = compute_robustness_score([0.6, 0.10, 1.10, 0.20, 1.00])
    assert low_var > high_var


def test_robustness_score_zero_mean_returns_zero():
    assert compute_robustness_score([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_robustness_score_negative_mean_returns_zero():
    assert compute_robustness_score([-0.2, -0.3, -0.1]) == pytest.approx(0.0)


def test_robustness_score_single_block():
    # cv undefined for n=1, should not raise; returns mean
    assert compute_robustness_score([0.5]) == pytest.approx(0.5)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "robustness" -v
```
Expected: `ImportError` for `compute_robustness_score`.

**Step 3: Implement**

```python
# Add to feature_research/walkforward/top_k_selection.py

def compute_robustness_score(block_metrics: list[float]) -> float:
    """Compute robustness score from per-block objective metrics.

    Score = mean_block / (1 + cv_block), where cv = std / |mean|.
    Returns 0 if mean <= 0 or the input list is empty.

    Parameters
    ----------
    block_metrics : list[float]
        Objective metric (e.g. Sharpe) computed independently on each block.
    """
    if not block_metrics:
        return 0.0
    arr = np.array(block_metrics, dtype=float)
    mean_val = float(np.mean(arr))
    if mean_val <= 0.0:
        return 0.0
    if len(arr) == 1:
        return mean_val
    std_val = float(np.std(arr, ddof=1))
    cv = std_val / abs(mean_val)
    return mean_val / (1.0 + cv)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "robustness" -v
```
Expected: 5 PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add compute_robustness_score"
```

---

## Task 3: Add `compute_signal_correlation_matrix`

**Files:**
- Modify: `feature_research/walkforward/top_k_selection.py`
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** Takes a `dict[str, pd.Series]` mapping param label → signal series (same index, same length). Returns a symmetric `pd.DataFrame` of Pearson correlations with labels as both index and columns.

**Step 1: Write the failing tests**

```python
from feature_research.walkforward.top_k_selection import compute_signal_correlation_matrix


def _make_signals(matrix: dict[str, list[float]]) -> dict[str, pd.Series]:
    idx = pd.date_range("2000-01-01", periods=len(next(iter(matrix.values()))), freq="B")
    return {label: pd.Series(vals, index=idx) for label, vals in matrix.items()}


def test_corr_matrix_identical_signals():
    signals = _make_signals({"A": [1.0, -1.0, 1.0, 1.0], "B": [1.0, -1.0, 1.0, 1.0]})
    corr = compute_signal_correlation_matrix(signals)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_corr_matrix_opposite_signals():
    signals = _make_signals({"A": [1.0, -1.0, 1.0], "B": [-1.0, 1.0, -1.0]})
    corr = compute_signal_correlation_matrix(signals)
    assert corr.loc["A", "B"] == pytest.approx(-1.0)


def test_corr_matrix_diagonal_is_one():
    signals = _make_signals({"A": [0.5, 0.3, -0.2], "B": [0.1, 0.8, 0.4]})
    corr = compute_signal_correlation_matrix(signals)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_corr_matrix_is_symmetric():
    signals = _make_signals({"A": [1.0, 0.5, -0.2], "B": [0.3, 0.8, 0.1], "C": [-0.1, 0.2, 0.9]})
    corr = compute_signal_correlation_matrix(signals)
    pd.testing.assert_frame_equal(corr, corr.T)


def test_corr_matrix_single_param():
    signals = _make_signals({"A": [1.0, 0.5, -0.2]})
    corr = compute_signal_correlation_matrix(signals)
    assert corr.shape == (1, 1)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "corr_matrix" -v
```

**Step 3: Implement**

```python
# Add to feature_research/walkforward/top_k_selection.py

def compute_signal_correlation_matrix(
    signal_series: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation matrix of training signal series.

    Parameters
    ----------
    signal_series : dict[str, pd.Series]
        Mapping from param label to its training-window signal/return series.
        All series must share the same DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix indexed and columned by param labels.
    """
    labels = list(signal_series)
    if not labels:
        return pd.DataFrame()
    df = pd.DataFrame({label: signal_series[label] for label in labels})
    return df.corr(method="pearson")
```

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "corr_matrix" -v
```
Expected: 5 PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add compute_signal_correlation_matrix"
```

---

## Task 4: Add `compute_quality_scores`

**Files:**
- Modify: `feature_research/walkforward/top_k_selection.py`
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** Normalises stability (smoothed objective) and robustness scores to [0, 1] within the surviving param set, then combines with pre-committed weights. Returns a `dict[str, float]` of quality scores.

**Step 1: Write the failing tests**

```python
from feature_research.walkforward.top_k_selection import compute_quality_scores


def test_quality_scores_normalised_to_unit_interval():
    smoothed = {"A": 0.8, "B": 0.6, "C": 0.4}
    robustness = {"A": 0.5, "B": 0.7, "C": 0.3}
    scores = compute_quality_scores(smoothed, robustness, weight_stability=0.5, weight_robustness=0.5)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_quality_scores_best_param_scores_one():
    # When one param dominates both dimensions it should score 1.0
    smoothed = {"A": 1.0, "B": 0.5}
    robustness = {"A": 1.0, "B": 0.5}
    scores = compute_quality_scores(smoothed, robustness, weight_stability=0.5, weight_robustness=0.5)
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)


def test_quality_scores_all_equal_inputs():
    # All identical → all score 0 (after normalisation)
    smoothed = {"A": 0.6, "B": 0.6}
    robustness = {"A": 0.4, "B": 0.4}
    scores = compute_quality_scores(smoothed, robustness, weight_stability=0.5, weight_robustness=0.5)
    assert scores["A"] == pytest.approx(0.0)
    assert scores["B"] == pytest.approx(0.0)


def test_quality_scores_weights_applied():
    # Stability-only weighting: robustness should not affect ranking
    smoothed = {"A": 1.0, "B": 0.0}
    robustness = {"A": 0.0, "B": 1.0}
    scores_stab = compute_quality_scores(smoothed, robustness, weight_stability=1.0, weight_robustness=0.0)
    assert scores_stab["A"] > scores_stab["B"]
    scores_rob = compute_quality_scores(smoothed, robustness, weight_stability=0.0, weight_robustness=1.0)
    assert scores_rob["B"] > scores_rob["A"]
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "quality_score" -v
```

**Step 3: Implement**

```python
# Add to feature_research/walkforward/top_k_selection.py

def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a dict of floats to [0, 1]. Returns all-zeros if range = 0."""
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return {k: 0.0 for k in values}
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}


def compute_quality_scores(
    smoothed_objectives: dict[str, float],
    robustness_scores: dict[str, float],
    weight_stability: float = 0.5,
    weight_robustness: float = 0.5,
) -> dict[str, float]:
    """Compute composite quality score from stability and robustness components.

    Both inputs are normalised independently to [0, 1] before combining.

    Parameters
    ----------
    smoothed_objectives : dict[str, float]
        Neighbour-smoothed objective per param label (stability proxy).
    robustness_scores : dict[str, float]
        Block-CV robustness score per param label.
    weight_stability, weight_robustness : float
        Weights summing to 1.0.

    Returns
    -------
    dict[str, float]
        Quality score per param label, in [0, 1].
    """
    labels = list(smoothed_objectives)
    stab_norm = _normalize(smoothed_objectives)
    rob_norm = _normalize(robustness_scores)
    return {
        label: weight_stability * stab_norm.get(label, 0.0) + weight_robustness * rob_norm.get(label, 0.0)
        for label in labels
    }
```

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "quality_score" -v
```
Expected: 4 PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add compute_quality_scores"
```

---

## Task 5: Add `greedy_diversity_select`

**Files:**
- Modify: `feature_research/walkforward/top_k_selection.py`
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** Greedy selection that picks the highest-quality param first, then iteratively picks the candidate with the best `quality * (1 - diversity_weight * max_abs_corr_with_selected)`. Returns up to k labels in selection order.

**Step 1: Write the failing tests**

```python
from feature_research.walkforward.top_k_selection import greedy_diversity_select


def _make_corr(labels: list[str], values: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Build a symmetric correlation DataFrame from upper-triangle values."""
    df = pd.DataFrame(1.0, index=labels, columns=labels)
    for (a, b), v in values.items():
        df.loc[a, b] = v
        df.loc[b, a] = v
    return df


def test_greedy_select_returns_k_items():
    quality = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}
    corr = _make_corr(["A", "B", "C", "D"], {("A","B"): 0.3, ("A","C"): 0.2, ("A","D"): 0.1,
                                               ("B","C"): 0.3, ("B","D"): 0.2, ("C","D"): 0.3})
    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.4)
    assert len(selected) == 3


def test_greedy_select_first_is_highest_quality():
    quality = {"A": 0.9, "B": 0.8, "C": 0.7}
    corr = _make_corr(["A", "B", "C"], {("A","B"): 0.5, ("A","C"): 0.5, ("B","C"): 0.5})
    selected = greedy_diversity_select(quality, corr, k=2, diversity_weight=0.4)
    assert selected[0] == "A"


def test_greedy_select_high_corr_penalised():
    # B is nearly identical to A (corr=0.99), C is independent (corr=0.0).
    # Even though B has higher quality than C, C should be preferred second.
    quality = {"A": 1.0, "B": 0.9, "C": 0.5}
    corr = _make_corr(["A", "B", "C"], {("A","B"): 0.99, ("A","C"): 0.0, ("B","C"): 0.0})
    selected = greedy_diversity_select(quality, corr, k=2, diversity_weight=0.4)
    assert selected[0] == "A"
    assert selected[1] == "C"  # C preferred over B despite lower quality


def test_greedy_select_k_larger_than_candidates():
    quality = {"A": 0.9, "B": 0.8}
    corr = _make_corr(["A", "B"], {("A","B"): 0.3})
    selected = greedy_diversity_select(quality, corr, k=5, diversity_weight=0.4)
    assert len(selected) == 2  # capped at available candidates


def test_greedy_select_no_diversity_weight_equals_quality_rank():
    quality = {"A": 0.9, "B": 0.8, "C": 0.7}
    corr = _make_corr(["A", "B", "C"], {("A","B"): 1.0, ("A","C"): 1.0, ("B","C"): 1.0})
    # With diversity_weight=0, all corr penalties are zero → pure quality ranking
    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.0)
    assert selected == ["A", "B", "C"]
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "greedy_select" -v
```

**Step 3: Implement**

```python
# Add to feature_research/walkforward/top_k_selection.py

def greedy_diversity_select(
    quality_scores: dict[str, float],
    corr_matrix: pd.DataFrame,
    k: int,
    diversity_weight: float = 0.40,
) -> list[str]:
    """Select up to k param labels using greedy max-marginal diversity.

    The first selection is the highest-quality param (unconditional).
    Each subsequent selection maximises: quality * (1 - diversity_weight * max_abs_corr).

    Parameters
    ----------
    quality_scores : dict[str, float]
        Composite quality score per param label.
    corr_matrix : pd.DataFrame
        Symmetric Pearson correlation matrix indexed by param label.
    k : int
        Target ensemble size.
    diversity_weight : float
        Penalty weight on correlation (0 = no diversity enforcement).

    Returns
    -------
    list[str]
        Selected param labels in selection order (first = highest priority).
    """
    if not quality_scores:
        return []

    remaining = sorted(quality_scores, key=quality_scores.__getitem__, reverse=True)
    selected: list[str] = [remaining.pop(0)]

    while len(selected) < k and remaining:
        best_score = float("-inf")
        best_p: str | None = None

        for p in remaining:
            if p in corr_matrix.index and any(s in corr_matrix.columns for s in selected):
                max_corr = max(
                    abs(corr_matrix.loc[p, s])
                    for s in selected
                    if s in corr_matrix.columns
                )
            else:
                max_corr = 0.0
            adjusted = quality_scores[p] * (1.0 - diversity_weight * max_corr)
            if adjusted > best_score:
                best_score = adjusted
                best_p = p

        if best_p is None:
            break
        selected.append(best_p)
        remaining.remove(best_p)

    return selected
```

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "greedy_select" -v
```
Expected: 5 PASS.

**Step 5: Run all top_k_selection tests**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -v
```
Expected: All PASS.

**Step 6: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add greedy_diversity_select"
```

---

## Task 6: Add config fields to `WalkforwardResearchConfig`

**Files:**
- Modify: `feature_research/walkforward/config.py`
- Modify: `tests/feature_research/walkforward/test_config.py`

**Context:** Add new optional fields with defaults. The `use_enhanced_selection` flag is `False` by default so all existing callers continue working unchanged. Validation rules: `trade_freq_min` in [0, 1], `n_robustness_blocks` ≥ 3, `diversity_weight` in [0, 1], weights must be ≥ 0.

**Step 1: Write the failing tests**

```python
# Append to tests/feature_research/walkforward/test_config.py
from datetime import datetime
from feature_research.walkforward.config import WalkforwardResearchConfig


def test_config_enhanced_selection_defaults():
    cfg = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2015, 1, 1),
    )
    assert cfg.use_enhanced_selection is False
    assert cfg.trade_freq_min == pytest.approx(0.05)
    assert cfg.n_robustness_blocks == 5
    assert cfg.diversity_weight == pytest.approx(0.40)
    assert cfg.weight_stability == pytest.approx(0.50)
    assert cfg.weight_robustness == pytest.approx(0.50)


def test_config_rejects_invalid_trade_freq_min():
    with pytest.raises(ValueError, match="trade_freq_min"):
        WalkforwardResearchConfig(
            train_start=datetime(2000, 1, 1),
            train_end=datetime(2015, 1, 1),
            trade_freq_min=1.5,
        )


def test_config_rejects_too_few_robustness_blocks():
    with pytest.raises(ValueError, match="n_robustness_blocks"):
        WalkforwardResearchConfig(
            train_start=datetime(2000, 1, 1),
            train_end=datetime(2015, 1, 1),
            n_robustness_blocks=2,
        )


def test_config_rejects_invalid_diversity_weight():
    with pytest.raises(ValueError, match="diversity_weight"):
        WalkforwardResearchConfig(
            train_start=datetime(2000, 1, 1),
            train_end=datetime(2015, 1, 1),
            diversity_weight=1.5,
        )
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_config.py -k "enhanced" -v
```

**Step 3: Add fields to `WalkforwardResearchConfig`**

In `feature_research/walkforward/config.py`, add these fields to the `WalkforwardResearchConfig` dataclass (after `output_root`):

```python
    use_enhanced_selection: bool = False
    trade_freq_min: float = 0.05
    n_robustness_blocks: int = 5
    diversity_weight: float = 0.40
    weight_stability: float = 0.50
    weight_robustness: float = 0.50
```

Add these validation checks to `__post_init__`:

```python
        if not (0.0 <= self.trade_freq_min <= 1.0):
            raise ValueError("trade_freq_min must be in [0, 1]")
        if self.n_robustness_blocks < 3:
            raise ValueError("n_robustness_blocks must be >= 3")
        if not (0.0 <= self.diversity_weight <= 1.0):
            raise ValueError("diversity_weight must be in [0, 1]")
        if self.weight_stability < 0.0 or self.weight_robustness < 0.0:
            raise ValueError("weight_stability and weight_robustness must be >= 0")
```

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_config.py -v
```
Expected: All PASS (new tests + existing unchanged).

**Step 5: Commit**

```bash
git add feature_research/walkforward/config.py tests/feature_research/walkforward/test_config.py
git commit -m "feat(walkforward): add enhanced selection config fields"
```

---

## Task 7: Add `run_enhanced_selection` orchestrator to `top_k_selection.py`

**Files:**
- Modify: `feature_research/walkforward/top_k_selection.py`
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Context:** This function ties together all the components from Tasks 1–5 into a single call that `_build_fold_scores` will invoke. It takes:
- `training_data`: training-window candles DataFrame
- `training_target`: training-window target Series
- `param_grid`: list of param dicts
- `evaluate_param_combo`: the same Callable as in the runner
- `smoothed_objectives`: dict from grid smoothing (already computed upstream)
- `objective_metric`: Callable[[pd.Series], float]
- `config`: `WalkforwardResearchConfig`

It returns a `EnhancedSelectionResult` dataclass.

**Step 1: Write the failing test**

```python
# Append to tests/feature_research/walkforward/test_top_k_selection.py
from feature_research.walkforward.top_k_selection import run_enhanced_selection, EnhancedSelectionResult
from feature_research.walkforward.config import WalkforwardResearchConfig
from datetime import datetime
import numpy as np


def _dummy_evaluate(candles, target, params):
    """Deterministic dummy: returns a signal series based on lookback param."""
    n = len(target)
    lookback = params.get("lookback", 1)
    rng = np.random.default_rng(seed=int(lookback))
    vals = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.2, 0.5])
    return pd.Series(vals, index=target.index) * 0.01  # small returns


def _make_training_data(n: int = 500):
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    candles = pd.DataFrame({"close": np.random.default_rng(0).normal(100, 1, n)}, index=idx)
    target = pd.Series(np.random.default_rng(1).normal(0, 0.01, n), index=idx)
    return candles, target


def test_run_enhanced_selection_returns_result():
    candles, target = _make_training_data()
    param_grid = [{"lookback": i} for i in range(3, 10)]
    smoothed = {f"lookback={i}": 0.5 + i * 0.01 for i in range(3, 10)}
    cfg = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1), train_end=datetime(2010, 1, 1),
        use_enhanced_selection=True,
    )

    def objective(series):
        if series.std() == 0:
            return 0.0
        return float(series.mean() / series.std())

    result = run_enhanced_selection(
        training_data=candles,
        training_target=target,
        param_grid=param_grid,
        evaluate_param_combo=_dummy_evaluate,
        smoothed_objectives=smoothed,
        objective_metric=objective,
        config=cfg,
    )
    assert isinstance(result, EnhancedSelectionResult)
    assert len(result.selected_labels) <= cfg.top_k
    assert all(label in smoothed for label in result.selected_labels)


def test_run_enhanced_selection_respects_trade_freq_min():
    # All params produce zero signal → all filtered by trade_freq
    candles, target = _make_training_data(200)
    param_grid = [{"lookback": 3}]
    smoothed = {"lookback=3": 0.5}
    cfg = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1), train_end=datetime(2005, 1, 1),
        use_enhanced_selection=True, trade_freq_min=0.99,  # almost impossible to pass
    )

    def zero_signal(candles, target, params):
        return pd.Series(0.0, index=target.index)

    def objective(s):
        return 0.0

    result = run_enhanced_selection(
        training_data=candles, training_target=target, param_grid=param_grid,
        evaluate_param_combo=zero_signal, smoothed_objectives=smoothed,
        objective_metric=objective, config=cfg,
    )
    assert result.selected_labels == []
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -k "run_enhanced" -v
```

**Step 3: Implement `EnhancedSelectionResult` and `run_enhanced_selection`**

```python
# Add to feature_research/walkforward/top_k_selection.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

# (add these imports at top of file if not already present)
# import numpy as np
# import pandas as pd


@dataclass(frozen=True)
class EnhancedSelectionResult:
    """Outputs from the enhanced top-k selection procedure for one fold."""
    selected_labels: list[str]
    trade_frequencies: dict[str, float]
    robustness_scores: dict[str, float]
    quality_scores: dict[str, float]
    corr_matrix: pd.DataFrame


def run_enhanced_selection(
    training_data: pd.DataFrame,
    training_target: pd.Series,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series],
    smoothed_objectives: dict[str, float],
    objective_metric: Callable[[pd.Series], float],
    config: object,  # WalkforwardResearchConfig (avoid circular import with TYPE_CHECKING)
) -> EnhancedSelectionResult:
    """Run the full enhanced top-k selection pipeline for one training fold.

    Spec: docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md
    """
    from feature_research.walkforward.runner import _canonical_param_label  # local import to avoid circular

    # --- Step 1: Evaluate all params on full training window, get signal series ---
    param_label_map = {_canonical_param_label(p): p for p in param_grid}
    signal_series: dict[str, pd.Series] = {}
    for label, params in param_label_map.items():
        signal_series[label] = evaluate_param_combo(training_data, training_target, params)

    # --- Step 2: Hard filter — trade frequency ---
    trade_frequencies = {label: compute_trade_frequency(s) for label, s in signal_series.items()}
    surviving = [
        label for label in param_label_map
        if trade_frequencies[label] >= config.trade_freq_min
    ]

    if not surviving:
        return EnhancedSelectionResult(
            selected_labels=[], trade_frequencies=trade_frequencies,
            robustness_scores={}, quality_scores={}, corr_matrix=pd.DataFrame(),
        )

    # --- Step 3: Robustness scores (N-block CV on training data) ---
    n = len(training_data)
    block_size = max(1, n // config.n_robustness_blocks)
    block_indices = [
        training_data.index[i * block_size: (i + 1) * block_size]
        for i in range(config.n_robustness_blocks)
    ]

    robustness_scores: dict[str, float] = {}
    for label in surviving:
        params = param_label_map[label]
        block_metrics = []
        for block_idx in block_indices:
            if len(block_idx) == 0:
                continue
            block_candles = training_data.loc[block_idx]
            block_target = training_target.loc[block_idx]
            block_signal = evaluate_param_combo(block_candles, block_target, params)
            block_metrics.append(float(objective_metric(block_signal)))
        robustness_scores[label] = compute_robustness_score(block_metrics)

    # Filter out params with negative mean block performance
    surviving = [label for label in surviving if robustness_scores[label] > 0.0]
    if not surviving:
        return EnhancedSelectionResult(
            selected_labels=[], trade_frequencies=trade_frequencies,
            robustness_scores=robustness_scores, quality_scores={}, corr_matrix=pd.DataFrame(),
        )

    # --- Step 4: Composite quality score ---
    surviving_smoothed = {label: smoothed_objectives.get(label, 0.0) for label in surviving}
    surviving_robustness = {label: robustness_scores[label] for label in surviving}
    quality_scores = compute_quality_scores(
        surviving_smoothed, surviving_robustness,
        weight_stability=config.weight_stability,
        weight_robustness=config.weight_robustness,
    )

    # --- Step 5: Signal correlation matrix ---
    surviving_signals = {label: signal_series[label] for label in surviving}
    corr_matrix = compute_signal_correlation_matrix(surviving_signals)

    # --- Step 6: Greedy diversity selection ---
    selected = greedy_diversity_select(
        quality_scores=quality_scores,
        corr_matrix=corr_matrix,
        k=config.top_k,
        diversity_weight=config.diversity_weight,
    )

    return EnhancedSelectionResult(
        selected_labels=selected,
        trade_frequencies=trade_frequencies,
        robustness_scores=robustness_scores,
        quality_scores=quality_scores,
        corr_matrix=corr_matrix,
    )
```

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/top_k_selection.py tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "feat(walkforward): add EnhancedSelectionResult and run_enhanced_selection"
```

---

## Task 8: Integrate into `_build_fold_scores` in `runner.py`

**Files:**
- Modify: `feature_research/walkforward/runner.py`
- Modify: `tests/feature_research/walkforward/test_runner.py`

**Context:** When `config.use_enhanced_selection` is True, call `run_enhanced_selection` on the training data and use its `selected_labels` as `top_k_features`. The test-fold scoring (raw and smoothed objectives) continues unchanged — enhanced selection only changes which params are picked as `top_k_features`.

Also add new columns to `fold_scores_df`:
- `trade_frequency` (float, NaN if enhanced selection not used)
- `robustness_score` (float, NaN if enhanced selection not used)
- `quality_score` (float, NaN if enhanced selection not used)
- `selected_by_diversity` (bool)

**Step 1: Write the failing test**

```python
# Append to tests/feature_research/walkforward/test_runner.py
from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.runner import run_walkforward_research
from datetime import datetime
import numpy as np
import pandas as pd


def _make_candles_target(n: int = 600):
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    candles = pd.DataFrame({"close": np.ones(n) * 100.0}, index=idx)
    rng = np.random.default_rng(42)
    target = pd.Series(rng.normal(0, 0.01, n), index=idx)
    return candles, target


def _dummy_evaluate(candles, target, params):
    lookback = params.get("lookback", 5)
    rng = np.random.default_rng(int(lookback))
    return pd.Series(rng.choice([-0.01, 0.0, 0.01], size=len(target), p=[0.3, 0.2, 0.5]), index=target.index)


def test_enhanced_selection_produces_top_k_features_column():
    candles, target = _make_candles_target()
    param_grid = [{"lookback": i} for i in range(3, 10)]
    cfg = WalkforwardResearchConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2004, 1, 1),
        num_steps=3,
        top_k=3,
        use_enhanced_selection=True,
    )
    report = run_walkforward_research(
        candles_df=candles, target=target, feature_type="continuous",
        module_name="rsi", config=cfg, param_grid=param_grid,
        evaluate_param_combo=_dummy_evaluate,
    )
    assert "top_k_features" in report.selection_summary_df.columns
    assert "trade_frequency" in report.fold_scores_df.columns
    assert "robustness_score" in report.fold_scores_df.columns
    assert "quality_score" in report.fold_scores_df.columns
    assert "selected_by_diversity" in report.fold_scores_df.columns
```

**Step 2: Run to verify failure**

```bash
pytest tests/feature_research/walkforward/test_runner.py -k "enhanced_selection" -v
```
Expected: FAIL — missing columns.

**Step 3: Modify `_build_fold_scores` in `runner.py`**

Replace the section starting at line `top_k_features = ranked_df["param_label"].head(top_k).tolist()` with:

```python
    # --- Enhanced top-k selection (optional) ---
    from feature_research.walkforward.top_k_selection import run_enhanced_selection  # local to avoid circular

    if getattr(config, "use_enhanced_selection", False):
        train_mask = cast(pd.Series, fold_row["_train_mask"])
        train_candles = candles_df.loc[train_mask]
        train_target = target.loc[train_mask]
        enh_result = run_enhanced_selection(
            training_data=train_candles,
            training_target=train_target,
            param_grid=param_grid,
            evaluate_param_combo=evaluate_param_combo,
            smoothed_objectives={
                row["param_label"]: row["smoothed_objective"]
                for _, row in smoothed_df.iterrows()
            },
            objective_metric=objective_metric,
            config=config,
        )
        top_k_features = enh_result.selected_labels

        # Attach enhanced scores to fold_scores_df rows
        fold_scores_df = fold_scores_df.copy()
        fold_scores_df["trade_frequency"] = fold_scores_df["param_label"].map(
            enh_result.trade_frequencies
        )
        fold_scores_df["robustness_score"] = fold_scores_df["param_label"].map(
            enh_result.robustness_scores
        )
        fold_scores_df["quality_score"] = fold_scores_df["param_label"].map(
            enh_result.quality_scores
        )
        fold_scores_df["selected_by_diversity"] = fold_scores_df["param_label"].isin(
            enh_result.selected_labels
        )
    else:
        top_k_features = ranked_df["param_label"].head(top_k).tolist()
        fold_scores_df["trade_frequency"] = float("nan")
        fold_scores_df["robustness_score"] = float("nan")
        fold_scores_df["quality_score"] = float("nan")
        fold_scores_df["selected_by_diversity"] = False
```

**Important:** The function signature must also receive `config` and `param_grid` and `candles_df`/`target`. Check the current signature of `_build_fold_scores` and the call site in `run_walkforward_research` — you will need to thread these through. Currently `_build_fold_scores` receives `param_grid` already. Add `config: WalkforwardResearchConfig` as a parameter and update the call site accordingly.

**Step 4: Run to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_runner.py -v
```
Expected: All PASS.

**Step 5: Run the full walkforward test suite**

```bash
pytest tests/feature_research/walkforward/ -v
```
Expected: All PASS.

**Step 6: Commit**

```bash
git add feature_research/walkforward/runner.py tests/feature_research/walkforward/test_runner.py
git commit -m "feat(walkforward): integrate enhanced selection into _build_fold_scores"
```

---

## Task 9: End-to-End Verification

**Files:**
- Modify: `tests/feature_research/walkforward/test_top_k_selection.py`

**Purpose:** Verify that the enhanced selection actually behaves differently from naive top-k in a controlled scenario where two params are nearly identical (should only appear once together in the enhanced result).

**Step 1: Write the test**

```python
def test_enhanced_selection_avoids_redundant_highly_correlated_pair():
    """When params B and C are perfectly correlated (corr=1.0), the greedy
    algorithm should pick one of them but not both if a less-correlated alternative D exists."""
    quality = {"A": 1.0, "B": 0.9, "C": 0.89, "D": 0.5}
    # B and C are nearly identical to each other
    corr = _make_corr(
        ["A", "B", "C", "D"],
        {("A","B"): 0.2, ("A","C"): 0.2, ("A","D"): 0.1,
         ("B","C"): 0.99, ("B","D"): 0.1, ("C","D"): 0.1},
    )
    selected = greedy_diversity_select(quality, corr, k=3, diversity_weight=0.4)
    # A and B should be selected; C (near-duplicate of B) should lose to D
    assert "A" in selected
    assert "B" in selected
    assert "C" not in selected  # B crowds out C
    assert "D" in selected
```

**Step 2: Run**

```bash
pytest tests/feature_research/walkforward/test_top_k_selection.py::test_enhanced_selection_avoids_redundant_highly_correlated_pair -v
```
Expected: PASS.

**Step 3: Run full test suite**

```bash
pytest tests/feature_research/walkforward/ tests/unit-tests/utils/test_grid_smoothing.py -v
```
Expected: All PASS, no regressions.

**Step 4: Commit**

```bash
git add tests/feature_research/walkforward/test_top_k_selection.py
git commit -m "test(walkforward): add end-to-end diversity verification test"
```

---

## Done

After Task 9, the enhanced selection is:
- Fully unit-tested in isolation (Tasks 1–5)
- Config-validated (Task 6)
- Orchestrated into a single callable (Task 7)
- Integrated into the runner behind an opt-in flag (Task 8)
- Verified to behave differently from naive top-k in a meaningful scenario (Task 9)

**To activate in research notebooks:** Set `use_enhanced_selection=True` in `WalkforwardResearchConfig` and add the new config fields (`trade_freq_min`, `n_robustness_blocks`, `diversity_weight`, etc.) as needed.
