# Walkforward Correctness Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the data leakage bug where `evaluate_param_combo` returns returns from a model fitted on the full IS period instead of per-fold training data.

**Architecture:** The current `_build_bin_count_specific_returns` in `pipeline.py` fits `ContinuousBinningModel` on the full IS feature/target data upfront, then `_build_walkforward_evaluator` wraps a closure over those pre-computed returns. When the runner calls `evaluate_param_combo(fold_candles, fold_target, params)`, the function ignores the fold boundary and just slices the pre-computed series. The fix: make the evaluator refit the model on the training portion of the fold each time it is called. The feature series and target series (pre-computed from the full IS period via the cache) are still computed once globally — that is fine because EWSD, ATR, and RSI are all causal (value at time T depends only on data up to T). Only the binning model's quantile edges must be learned from training data.

**Tech Stack:** Python, pandas, `ContinuousBinningModel` (`feature_selection/base_models/continuous_binning.py`), `pipeline.py`, `runner.py`

---

### Context: What Was Already Fixed

`runner.py` was already updated to track `raw_objective` (IS/train metric) and `oos_objective` (OOS/test metric) separately per fold. The `score_param` closure already splits the returned series at `train_index` / `test_index`. This is correct scaffolding — but only meaningful once the evaluator properly fits on training data only.

### Context: The Exact Bug Location

[feature_research/in_sample/continuous_binning/pipeline.py:92-117](../../feature_research/in_sample/continuous_binning/pipeline.py#L92-L117)

`_build_bin_count_specific_returns` fits `ContinuousBinningModel` on `feature` and `target` which span the entire IS period (e.g., 2000–2023). `_build_walkforward_evaluator` returns a closure:

```python
def evaluate_param_combo(fold_candles, _fold_target, params):
    returns = combo_returns[_combo_key(params)]       # full IS returns
    fold_returns = returns.reindex(fold_candles.index)  # just slicing
    return fold_returns
```

For fold 3 (train 2006–2015, test 2015–2016), the model generating `returns` saw 2015–2016 data during training. The OOS score is therefore not OOS.

---

### Task 1: Add failing tests that prove the bug exists

**Files:**
- Modify: `tests/feature_research/walkforward/test_runner.py`

**Step 1: Write the failing test**

```python
def test_evaluate_param_combo_does_not_use_future_data():
    """
    Prove the evaluator refits on training data only.
    A model fit on IS-only data should produce different bin edges
    than one fit on IS+OOS data. We verify by checking that the
    evaluator called on a fold returns a model whose bin edges
    match training-only data.
    """
    import numpy as np
    import pandas as pd
    from feature_research.in_sample.continuous_binning.config import load_config
    from feature_research.in_sample.continuous_binning.pipeline import _build_fold_aware_evaluator

    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=200, freq="B")
    feature = pd.Series(np.random.uniform(0, 100, 200), index=dates, name="rsi_signal_D_lookback_5")
    target = pd.Series(np.random.normal(0, 1, 200), index=dates, name="log_return_atr")

    # Split: train = first 150, test = last 50
    train_feature = feature.iloc[:150]
    train_target = target.iloc[:150]
    test_feature = feature.iloc[150:]

    # Build the fold-aware evaluator
    evaluator = _build_fold_aware_evaluator(
        feature=feature,
        target=target,
        config=None,  # pass minimal config (see implementation)
        bin_count=5,
    )

    # Simulate a fold call: pass combined candles+target
    combined_df = pd.DataFrame({"close": target.values}, index=dates)
    params = {"lookback": 5, "bin_count": 5}
    returned_returns = evaluator(combined_df, target, params)

    # Train-period returns must come from a model fit only on train data
    # Verify: model fit on full data gives different edges than train-only
    from feature_selection.base_models.continuous_binning import ContinuousBinningModel
    full_model = ContinuousBinningModel(n_bins=5, bin_counts=[5])
    full_model.fit(feature, target)
    full_edges = full_model.bin_edges_

    train_model = ContinuousBinningModel(n_bins=5, bin_counts=[5])
    train_model.fit(train_feature, train_target)
    train_edges = train_model.bin_edges_

    # Edges should differ because quantiles differ between full vs train-only data
    assert not np.allclose(full_edges, train_edges, atol=1e-6), \
        "Train-only and full-IS bin edges are identical — test data has no effect"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/ramanthind/Desktop/Enigma/Trading-Algo
pytest tests/feature_research/walkforward/test_runner.py::test_evaluate_param_combo_does_not_use_future_data -v
```

Expected: `ImportError` (function doesn't exist yet) or `AttributeError`.

---

### Task 2: Replace `_build_walkforward_evaluator` with a fold-aware evaluator

**Files:**
- Modify: `feature_research/in_sample/continuous_binning/pipeline.py`

**Context:** The new evaluator must:
1. Accept a pre-computed `feature` and `target` series (full IS period, read-only — still computed from cache once)
2. When called for a fold, receive `fold_candles` whose index covers `[train_start, test_end]`
3. Identify the train portion by slicing `feature`/`target` to `train_index` — use the runner's train/test split that is already embedded in the fold candles index
4. Refit `ContinuousBinningModel` on train-only feature/target
5. Predict on full fold feature range, return the signal × target returns

The runner passes `fold_candles` whose index is `train_index ∪ test_index`. The evaluator knows the boundary because `runner.py` exposes `train_index` and `test_index` to `score_param`. However, the `evaluate_param_combo` signature is `(fold_candles, fold_target, params) -> pd.Series`. The runner already splits by index after the call.

The simplest correct fix: **pass the full fold feature+target to the evaluator, and tell it where the train boundary is via a side channel**. The cleanest approach is to embed the train boundary into `fold_candles` as a special attribute, or to change the evaluator signature.

**Chosen approach:** Change the evaluator so it receives combined fold data (train+test) as before, but the evaluator itself detects the train boundary. Since the feature/target series are pre-computed for the full IS period, the evaluator can use `fold_candles.index` to identify which rows belong to train vs test. The runner already knows `train_index` and `test_index` from fold_row masks. **Pass `train_end_boundary` as an extra closure variable** by capturing it differently.

**Revised approach (simplest, no signature change):** Change from pre-computing returns once per param combo to computing per-fold. The closure should capture `feature` and `target` (the full series) and refit per call using a train cutoff. The train/test boundary is inferred by using the runner's existing `train_end` which is stored on `fold_row`. But the runner doesn't pass `train_end` into `evaluate_param_combo`.

**Final chosen approach:** Change `evaluate_param_combo` signature to accept `train_end: pd.Timestamp` as a kwarg. Update runner to pass it. This is a 3-line change to the runner and a full rewrite of the evaluator.

Wait — the runner calls `evaluate_param_combo(fold_candles, fold_target, params)` and `score_param` handles train/test splitting AFTER the call. To avoid changing the runner's public API, embed the train index into `fold_candles` using its index metadata (train rows come before test rows chronologically).

**Cleanest solution that requires minimal changes:**

The evaluator closure captures `feature` and `target` (full IS). It receives `fold_candles` (train+test combined). It needs to know the train/test split. Since the runner passes `fold_candles = candles_df.loc[combined_mask]` and the rows are sorted chronologically (train comes before test), we need the boundary. The runner stores `train_end` in `fold_row`.

**Pass `train_end` through `fold_target.attrs`** (pandas Series metadata dict):

In `runner.py` `_build_fold_scores`, before calling `evaluate_param_combo`, annotate the fold_target:
```python
fold_target = target.loc[combined_mask].copy()
fold_target.attrs["train_end"] = fold_row["train_end"]  # pd.Timestamp
```

The evaluator reads `train_end = fold_target.attrs.get("train_end")` and uses it to split.

**Step 1: Update `_build_fold_scores` in `runner.py` to pass train_end via attrs**

```python
# In _build_fold_scores, before score_param calls:
fold_target = target.loc[combined_mask].copy()
fold_target.attrs["train_end"] = fold_row["train_end"]  # pd.Timestamp boundary
```

**Step 2: Replace `_build_walkforward_evaluator` and `_build_bin_count_specific_returns` in `pipeline.py`**

Remove `_build_bin_count_specific_returns` (the pre-computing function).
Remove `_build_walkforward_evaluator` (the pre-compute-based closure).
Replace with `_build_fold_aware_evaluator`:

```python
def _build_fold_aware_evaluator(
    feature_by_combo: dict[tuple[tuple[str, object], ...], pd.Series],
    target_by_combo: dict[tuple[tuple[str, object], ...], pd.Series],
    config: "ResearchConfig",
) -> Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]:
    """Build an evaluator that refits ContinuousBinningModel on the training fold only.

    feature_by_combo and target_by_combo are pre-computed for the full IS period
    (this is fine: EWSD/ATR/RSI are all causal — value at T depends only on data up to T).
    The BINNING MODEL is refit from scratch on training data per fold call.
    """
    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        key = _combo_key(params)
        feature = feature_by_combo[key]
        target = target_by_combo[key]

        # Identify train/test boundary from the fold_target metadata
        train_end: pd.Timestamp | None = fold_target.attrs.get("train_end")
        if train_end is None:
            raise ValueError(
                "fold_target.attrs['train_end'] not set. "
                "Ensure runner.py passes train_end via fold_target.attrs."
            )

        # Align full-IS feature/target to the fold's date range
        fold_index = _normalize_datetime_index(fold_candles.index)
        feature_fold = feature.reindex(fold_index).dropna()
        target_fold = target.reindex(fold_index).dropna()

        aligned = pd.DataFrame(
            {"feature": feature_fold, "target": target_fold}
        ).dropna()

        if aligned.empty:
            return pd.Series(dtype=float)

        # Split into train (fit model) and test (evaluate)
        train_mask = aligned.index <= train_end
        train_df = aligned[train_mask]

        if len(train_df) < 10:
            # Insufficient training data for this fold
            return pd.Series(dtype=float)

        # Refit ContinuousBinningModel on TRAINING DATA ONLY
        bin_count = int(params.get("bin_count", config.binning_params.bin_counts[0]))
        model = ContinuousBinningModel(
            n_bins=bin_count,
            bin_counts=[bin_count],
            selection_metric=config.binning_params.selection_metric,
            strategy=config.binning_params.strategy,
            metric_threshold=config.binning_params.metric_threshold,
            t_threshold=config.binning_params.t_threshold,
            shrinkage_k=config.binning_params.shrinkage_k,
            long_clip_min=config.binning_params.long_clip_min,
            long_clip_max=config.binning_params.long_clip_max,
            short_clip_min=config.binning_params.short_clip_min,
            short_clip_max=config.binning_params.short_clip_max,
            use_coverage_bonus=config.binning_params.use_coverage_bonus,
            coverage_bonus_per_10pct=config.binning_params.coverage_bonus_per_10pct,
            max_coverage_bonus=config.binning_params.max_coverage_bonus,
        )
        model.fit(train_df["feature"], train_df["target"])

        # Predict on FULL fold (runner splits IS/OOS after this call)
        signal = model.predict(aligned["feature"], strategy=config.binning_params.strategy)
        returns = _normalize_series_datetime_index(signal.mul(aligned["target"]))
        return returns

    return evaluate_param_combo
```

**Step 3: Update `run_continuous_eda_pipeline` and `run_continuous_walkforward_pipeline` to use the new evaluator**

In both pipeline functions, replace the combo_returns pre-computation block:

```python
# REMOVE (old approach):
combo_returns: dict[...] = {}
for single_spec in expanded:
    ...
    combo_returns[_combo_key(combo_params)] = _build_bin_count_specific_returns(...)

evaluator = _build_walkforward_evaluator(combo_returns)

# ADD (new approach):
feature_by_combo: dict[...] = {}
target_by_combo: dict[...] = {}
for single_spec in expanded:
    ...
    feature_by_combo[_combo_key(combo_params)] = feature  # full IS feature series
    target_by_combo[_combo_key(combo_params)] = target    # full IS target series

evaluator = _build_fold_aware_evaluator(feature_by_combo, target_by_combo, config)
```

Note: `bin_count` is now a param in `params` dict (already present from `_expand_params_with_bin_count`), so the evaluator reads it from `params`. The `successful_param_grid` (passed to walkforward runner) already contains `bin_count` as a key.

**Step 4: Update `runner.py` to annotate `fold_target` with `train_end`**

In `_build_fold_scores`, add before the `score_param` calls:

```python
fold_target = target.loc[combined_mask].copy()
fold_target.attrs["train_end"] = fold_row["train_end"]
```

**Step 5: Run existing tests**

```bash
pytest tests/feature_research/ -v
```

Expected: all existing tests pass (same interface, same return type).

**Step 6: Run new test from Task 1**

```bash
pytest tests/feature_research/walkforward/test_runner.py::test_evaluate_param_combo_does_not_use_future_data -v
```

Expected: PASS.

**Step 7: Verify enhanced_selection still works**

`run_enhanced_selection` in `top_k_selection.py` calls `evaluate_param_combo(training_data, training_target, params)`. After this fix, `training_target` won't have `train_end` in `.attrs`, so the evaluator will raise. Fix: when called from `run_enhanced_selection` (train-only context), set `train_end = training_target.index.max()` so the entire input is treated as train data.

In `runner.py` `_build_fold_scores`:
```python
train_target = target.loc[train_mask].copy()
train_target.attrs["train_end"] = fold_row["train_end"]  # all input is "train"
```

The enhanced_selection evaluator will then fit on full training_data (correct).

**Step 8: Commit**

```bash
git add feature_research/in_sample/continuous_binning/pipeline.py \
        feature_research/walkforward/runner.py \
        tests/feature_research/walkforward/test_runner.py
git commit -m "fix(walkforward): refit binning model per fold to eliminate IS data leakage"
```

---

### Task 3: Verify OOS metrics are now meaningfully different from IS metrics

**Step 1: Run walkforward on a small dataset and inspect output**

```python
# In a notebook or quick script:
from feature_research.in_sample.continuous_binning.config import load_config
from feature_research.in_sample.continuous_binning.pipeline import run_continuous_walkforward_pipeline
from pathlib import Path
import dataclasses

config = load_config()
wf = dataclasses.replace(config.walkforward, enabled=True, num_steps=3)
config = dataclasses.replace(config, walkforward=wf)
report = run_continuous_walkforward_pipeline(config, Path("/tmp/wf_test"))

# raw_objective = IS (train fold) metric
# oos_objective = OOS (test fold) metric
# Expect: raw_objective > oos_objective (IS overfits; OOS is lower)
print(report.fold_scores_df[["param_label", "raw_objective", "oos_objective"]].head(20))
```

Expected: `raw_objective` systematically higher than `oos_objective` across params (sign of proper IS vs OOS split). Previously both would be similarly inflated.

**Step 2: Commit if output looks correct**

```bash
git commit -m "test(walkforward): verify IS/OOS divergence after correctness fix"
```

---

### Correctness Invariant After This Fix

The following must hold after every fold evaluation:
- Model bin edges are computed using only rows where `index <= train_end`
- The returned returns series covers both train and test rows (runner handles splitting)
- `oos_objective` is a genuine out-of-sample metric
- `raw_objective` (IS) will typically be higher — this IS/OOS gap is the overfitting signal you want to track
