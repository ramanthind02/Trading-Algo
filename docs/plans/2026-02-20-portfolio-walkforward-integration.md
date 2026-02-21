# Portfolio Walkforward Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a second evaluation stage to the walkforward pipeline that uses the existing Portfolio class to produce production-identical aggregate OOS metrics and visualizations after parameter selection.

**Architecture:** Two-stage design.
- **Stage 1 (Param Selection)** — unchanged: per-fold lightweight `ContinuousBinningModel` refit + OOS scoring → top-k param selection (implemented in `2026-02-20-walkforward-correctness-fix.md`).
- **Stage 2 (Portfolio Simulation)** — new: once top-k params are identified from Stage 1, build a `DiversifiedEnsemble` containing those `ContinuousBinningModel` instances, wrap it in a `Portfolio`, fit on the training fold, and evaluate on the test fold. This produces position fractions that are multiplied against realized vol-normalized returns, giving the "what would the Portfolio have made" metric. Stage 2 uses the same candles data already loaded; no new data fetching.

The key insight: Stage 1 finds stable params cheaply; Stage 2 evaluates their combined portfolio performance via the production class. This keeps Stage 1 fast (no DiversifiedEnsemble overhead) while making Stage 2 production-identical.

**Prerequisite:** `2026-02-20-walkforward-correctness-fix.md` must be implemented first.

**Tech Stack:** `ensemble/portfolio.py`, `ensemble/diversified_ensemble.py`, `feature_selection/base_models/continuous_binning.py`, `feature_research/continuous_binning/pipeline.py`, `feature_research/walkforward/runner.py`

---

### Context: Why Not Use Portfolio for Stage 1 Scoring?

`DiversifiedEnsemble.fit_from_candles` expects a candles DataFrame with `datetime, open, high, low, close, volume, ticker, timeframe` columns and extracts features from raw candles using the bias node cache. Running this for every param combo × every fold would be expensive and tightly couples research to the production candle format. Stage 1's `ContinuousBinningModel.fit` is fast (~100ms vs ~1–2s for DiversifiedEnsemble). Keep Stage 1 lightweight.

### Context: Portfolio's Expected Candle Format

`Portfolio.fit_from_candles` calls `DiversifiedEnsemble.fit_from_candles` which expects:
```
DataFrame columns: datetime, open, high, low, close, volume, ticker, timeframe
```
The `load_candles_for_config` in `data_loader.py` already loads this format but may not include `volume` and `timeframe` columns. The Portfolio filters by `timeframe == self.trading_timeframe`. Need to ensure candles have these columns before passing to Portfolio.

### Context: DiversifiedEnsemble with Dynamic Base Models

`DiversifiedEnsemble` is designed for control-file-driven production use. But `base_models` is a plain dict — we can add `ContinuousBinningModel` instances directly without going through control files for research purposes. We need to:
1. Set `model.feature_column` to the expected feature column name (e.g., `rsi_signal_D_lookback_5`)
2. Add to `ensemble.base_models[model_name] = model`
3. Set `ensemble.required_columns` to the list of feature columns
4. Call `ensemble.fit_from_candles(train_candles, target_data)` — this will extract features from candles and call `model.fit(feature_series, target_series)`

---

### Task 1: Understand and test DiversifiedEnsemble with programmatic base models

**Files:**
- Test: `tests/feature_research/walkforward/test_portfolio_integration.py` (new)

**Step 1: Write an integration smoke test**

```python
"""Verify ContinuousBinningModel can be injected into DiversifiedEnsemble programmatically."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import Portfolio
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from utils.enums import TimeFrame, Ticker


def _make_mock_candles(tickers, n=300):
    """Build minimal candles DataFrame in Portfolio format."""
    frames = []
    for i, ticker in enumerate(tickers):
        dates = pd.date_range("2005-01-01", periods=n, freq="B")
        np.random.seed(i)
        close = 4000 + np.cumsum(np.random.randn(n) * 10)
        open_ = close * (1 + np.random.randn(n) * 0.001)
        frames.append(pd.DataFrame({
            "datetime": dates,
            "open": open_,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1000.0,
            "ticker": ticker.name,
            "timeframe": TimeFrame.D,
        }))
    return pd.concat(frames, ignore_index=True)


def test_diversified_ensemble_accepts_programmatic_base_models():
    """DiversifiedEnsemble can be built with ContinuousBinningModel instances."""
    tickers = [Ticker.ES]
    candles = _make_mock_candles(tickers, n=300)

    model = ContinuousBinningModel(n_bins=5, bin_counts=[5])
    # The feature column name follows the standard naming convention
    model.feature_column = "rsi_signal_D_lookback_5"

    ensemble = DiversifiedEnsemble(target_volatility=0.15)
    ensemble.base_models["rsi_signal_D_lookback_5_long"] = model
    ensemble.required_columns = ["rsi_signal_D_lookback_5"]

    # fit_from_candles should extract RSI-5 from candles and call model.fit()
    # (requires cache to be populated for this test — use candles_override)
    # This test verifies the plumbing, not full numeric correctness
    with pytest.raises((ValueError, KeyError)):
        # Expected to fail without cache — confirms the plumbing reaches model.fit()
        ensemble.fit_from_candles(candles, target_data=pd.Series(dtype=float))
```

**Step 2: Run to understand failure mode**

```bash
pytest tests/feature_research/walkforward/test_portfolio_integration.py -v
```

Expected: may succeed or fail with a specific error. This tells us exactly what additional setup is needed to inject models programmatically.

---

### Task 2: Create `ResearchPortfolioEvaluator` in the walkforward module

This is a new module that wraps Portfolio/DiversifiedEnsemble creation for research use. It accepts pre-computed feature series (from the cache pipeline), not raw candles, and bridges to the Portfolio API.

**Files:**
- Create: `feature_research/walkforward/portfolio_evaluator.py`

**Design:**

```python
"""
Portfolio-based evaluation for walkforward research.

After Stage 1 selects top-k params per fold, Stage 2 creates a Portfolio
containing those params and evaluates its aggregate OOS performance.

Key design decision: features are extracted from candles using the standard
extract_features_for_bias_node pipeline, so the DiversifiedEnsemble receives
candles in the expected format. The Portfolio then handles all vol normalization,
weighting, and position sizing identically to live trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import Portfolio
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from utils.enums import TimeFrame


@dataclass(frozen=True)
class FoldPortfolioResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    oos_portfolio_sharpe: float
    oos_portfolio_returns: pd.Series
    n_params_selected: int


def build_research_portfolio(
    selected_params: list[dict[str, Any]],
    binning_config: Any,  # BinningAnalysisConfig
    tickers: list[Any],   # list[Ticker]
    trading_timeframe: TimeFrame = TimeFrame.D,
    target_volatility: float = 0.15,
) -> Portfolio:
    """
    Build a Portfolio with one DiversifiedEnsemble containing one
    ContinuousBinningModel per selected param combo.

    Parameters
    ----------
    selected_params : list[dict]
        Top-k param dicts from Stage 1 selection. Each dict contains
        bias node params (e.g., {"lookback": 5, "bin_count": 8}).
    binning_config : BinningAnalysisConfig
        Binning hyperparameters from ResearchConfig.
    tickers : list[Ticker]
        Tickers for this research run.
    trading_timeframe : TimeFrame
        Portfolio trading timeframe.
    target_volatility : float
        Target annualized volatility (τ in Carver's formula, e.g., 0.15 = 15%).

    Returns
    -------
    Portfolio (unfitted)
    """
    models: dict[str, ContinuousBinningModel] = {}
    required_columns: list[str] = []

    for params in selected_params:
        lookback = params.get("lookback")
        bin_count = params.get("bin_count", binning_config.bin_counts[0])
        # Feature column name follows: {module}_{feature}_{timeframe}_lookback_{v}
        # This must match what extract_features_for_bias_node generates
        feature_col = f"rsi_signal_D_lookback_{lookback}"  # TODO: generalize to other modules

        model = ContinuousBinningModel(
            n_bins=int(bin_count),
            bin_counts=[int(bin_count)],
            selection_metric=binning_config.selection_metric,
            strategy=binning_config.strategy,
            metric_threshold=binning_config.metric_threshold,
            t_threshold=binning_config.t_threshold,
            shrinkage_k=binning_config.shrinkage_k,
            long_clip_min=binning_config.long_clip_min,
            long_clip_max=binning_config.long_clip_max,
            short_clip_min=binning_config.short_clip_min,
            short_clip_max=binning_config.short_clip_max,
            use_coverage_bonus=binning_config.use_coverage_bonus,
        )
        model.feature_column = feature_col

        model_key = f"{feature_col}_{binning_config.strategy}"
        models[model_key] = model
        required_columns.append(feature_col)

    ensemble = DiversifiedEnsemble(target_volatility=target_volatility)
    ensemble.base_models = models
    ensemble.required_columns = required_columns
    ensemble.base_tf = trading_timeframe

    portfolio = Portfolio(
        ensembles=[ensemble],
        trading_timeframe=trading_timeframe,
        target_volatility=target_volatility,
    )
    return portfolio
```

**Step 3: Write test**

```python
def test_build_research_portfolio_creates_portfolio_with_correct_models():
    from feature_research.walkforward.portfolio_evaluator import build_research_portfolio
    from feature_research.continuous_binning.config import BinningAnalysisConfig
    from utils.enums import TimeFrame, Ticker

    binning_config = BinningAnalysisConfig(bin_counts=[5], strategy="long")
    params = [{"lookback": 5, "bin_count": 5}, {"lookback": 7, "bin_count": 5}]
    portfolio = build_research_portfolio(
        selected_params=params,
        binning_config=binning_config,
        tickers=[Ticker.ES],
        trading_timeframe=TimeFrame.D,
    )

    assert len(portfolio.ensembles) == 1
    ensemble = portfolio.ensembles[0]
    assert len(ensemble.base_models) == 2
    assert "rsi_signal_D_lookback_5_long" in ensemble.base_models
    assert "rsi_signal_D_lookback_7_long" in ensemble.base_models
    assert not portfolio.is_fitted_
```

**Step 4: Run test**

```bash
pytest tests/feature_research/walkforward/test_portfolio_integration.py::test_build_research_portfolio_creates_portfolio_with_correct_models -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add feature_research/walkforward/portfolio_evaluator.py \
        tests/feature_research/walkforward/test_portfolio_integration.py
git commit -m "feat(walkforward): add ResearchPortfolioEvaluator scaffold"
```

---

### Task 3: Add per-fold portfolio evaluation function

**Files:**
- Modify: `feature_research/walkforward/portfolio_evaluator.py`

Add `evaluate_fold_portfolio`:

```python
def evaluate_fold_portfolio(
    train_candles: pd.DataFrame,
    test_candles: pd.DataFrame,
    selected_params: list[dict[str, Any]],
    target_series: pd.Series,    # vol-normalized returns for full IS period
    binning_config: Any,
    tickers: list[Any],
    trading_timeframe: TimeFrame = TimeFrame.D,
    target_volatility: float = 0.15,
    objective_metric: Callable[[pd.Series], float] | None = None,
) -> FoldPortfolioResult:
    """
    Fit a Portfolio on the training fold and evaluate on the test fold.

    The Portfolio uses DiversifiedEnsemble with one ContinuousBinningModel
    per selected param. Features are extracted from candles (cache-backed).
    Vol normalization, WeightLayer (FDM), and IDM are applied identically
    to live trading.

    Parameters
    ----------
    train_candles : pd.DataFrame
        Raw candles for training fold. Must have columns:
        datetime, open, high, low, close, volume, ticker, timeframe.
    test_candles : pd.DataFrame
        Raw candles for test fold (same schema).
    selected_params : list[dict]
        Params selected by Stage 1 for this fold.
    target_series : pd.Series
        Vol-normalized forward returns (full IS period) for realized return computation.
    binning_config : BinningAnalysisConfig
        Binning hyperparameters.
    tickers : list[Ticker]
        Research universe.
    trading_timeframe : TimeFrame
        Portfolio trading timeframe.
    target_volatility : float
        Target annual volatility.
    objective_metric : callable, optional
        Function from pd.Series -> float for scoring OOS returns.
        Defaults to Sharpe.

    Returns
    -------
    FoldPortfolioResult
    """
    from feature_research.walkforward.metrics import resolve_objective_metric

    if objective_metric is None:
        objective_metric = resolve_objective_metric("sharpe")

    portfolio = build_research_portfolio(
        selected_params=selected_params,
        binning_config=binning_config,
        tickers=tickers,
        trading_timeframe=trading_timeframe,
        target_volatility=target_volatility,
    )

    # Fit Portfolio on training candles
    # DiversifiedEnsemble extracts features internally from candles via cache
    train_target = target_series.reindex(train_candles.index).dropna()
    portfolio.fit_from_candles(train_candles, target_data=train_target)

    # Predict position fractions on test candles
    predictions_df = portfolio.predict_from_candles(test_candles)
    # predictions_df columns: ticker, forecast_score, position_fraction

    # Compute OOS returns: position_fraction × realized_vol_normalized_return
    test_target = target_series.reindex(test_candles.index).dropna()
    # Average position fraction across tickers per timestamp (equal weight)
    if "position_fraction" in predictions_df.columns:
        avg_position = (
            predictions_df.groupby(predictions_df.index)["position_fraction"].mean()
        )
    else:
        avg_position = pd.Series(0.0, index=test_target.index)

    aligned = pd.DataFrame(
        {"position": avg_position, "target": test_target}
    ).dropna()

    oos_returns = (aligned["position"] * aligned["target"]).rename("portfolio_returns")
    oos_sharpe = float(objective_metric(oos_returns[oos_returns != 0]))

    train_ts = pd.Timestamp(train_candles["datetime"].min())
    train_te = pd.Timestamp(train_candles["datetime"].max())
    test_ts = pd.Timestamp(test_candles["datetime"].min())
    test_te = pd.Timestamp(test_candles["datetime"].max())

    return FoldPortfolioResult(
        fold_id=-1,  # caller sets this
        train_start=train_ts,
        train_end=train_te,
        test_start=test_ts,
        test_end=test_te,
        oos_portfolio_sharpe=oos_sharpe,
        oos_portfolio_returns=oos_returns,
        n_params_selected=len(selected_params),
    )
```

**Step 6: Write test for `evaluate_fold_portfolio`**

```python
def test_evaluate_fold_portfolio_returns_valid_sharpe():
    """End-to-end: Portfolio fits on train, predicts on test, returns a finite Sharpe."""
    # This test requires cache to be available — mark as integration
    # Skip if cache not available
    import os
    if not os.path.exists("data/ohlc_data"):
        pytest.skip("Cache data not available")

    from feature_research.walkforward.portfolio_evaluator import evaluate_fold_portfolio
    from feature_research.continuous_binning.config import load_config, BinningAnalysisConfig
    from feature_research.continuous_binning.data_loader import load_candles_for_config

    config = load_config()
    candles = load_candles_for_config(config)

    # Ensure required portfolio columns exist
    if "volume" not in candles.columns:
        candles["volume"] = 0.0
    if "timeframe" not in candles.columns:
        from utils.enums import TimeFrame
        candles["timeframe"] = TimeFrame.D

    # Split candles into train (before 2016) and test (2016-2018)
    train_candles = candles[candles["datetime"] < "2016-01-01"]
    test_candles = candles[
        (candles["datetime"] >= "2016-01-01") &
        (candles["datetime"] < "2018-01-01")
    ]

    # Load target series (vol-normalized returns)
    from feature_research.continuous_binning.data_loader import load_features_for_combo, expand_bias_specs
    single_spec = expand_bias_specs(config.bias_spec)[0]
    data = load_features_for_combo(single_spec, config)
    assert data is not None
    _, target, _ = data

    selected_params = [{"lookback": 5, "bin_count": 5}, {"lookback": 7, "bin_count": 5}]
    result = evaluate_fold_portfolio(
        train_candles=train_candles,
        test_candles=test_candles,
        selected_params=selected_params,
        target_series=target,
        binning_config=config.binning_params,
        tickers=config.tickers,
    )

    assert isinstance(result.oos_portfolio_sharpe, float)
    assert not pd.isna(result.oos_portfolio_sharpe)
    assert not result.oos_portfolio_returns.empty
```

**Step 7: Run test**

```bash
pytest tests/feature_research/walkforward/test_portfolio_integration.py::test_evaluate_fold_portfolio_returns_valid_sharpe -v
```

Expected: PASS (with cache available).

**Step 8: Commit**

```bash
git add feature_research/walkforward/portfolio_evaluator.py \
        tests/feature_research/walkforward/test_portfolio_integration.py
git commit -m "feat(walkforward): add per-fold portfolio evaluation with production Portfolio class"
```

---

### Task 4: Wire portfolio evaluation into `WalkforwardRunReport`

After `run_walkforward_research` completes, add a portfolio simulation pass using the selected top-k params per fold.

**Files:**
- Modify: `feature_research/walkforward/runner.py`
- Modify: `feature_research/walkforward/runner.py` — `WalkforwardRunReport` dataclass

**Step 1: Add `portfolio_results_df` to `WalkforwardRunReport`**

```python
@dataclass(frozen=True)
class WalkforwardRunReport:
    folds_df: pd.DataFrame
    fold_scores_df: pd.DataFrame
    selection_summary_df: pd.DataFrame
    portfolio_results_df: pd.DataFrame  # new: per-fold Portfolio OOS metrics
```

**Step 2: Add `run_portfolio_simulation` function to `runner.py`**

```python
def run_portfolio_simulation(
    candles_df: pd.DataFrame,
    target: pd.Series,
    fold_rows: list[dict],
    selection_summary_df: pd.DataFrame,
    config: "WalkforwardResearchConfig",
    research_config: Any,  # ResearchConfig (avoid circular import via TYPE_CHECKING)
) -> pd.DataFrame:
    """
    Stage 2: For each fold, build Portfolio from selected top-k params,
    fit on train, evaluate on test.
    Reads top_k_features from selection_summary_df (set in Stage 1).
    """
    import json
    from feature_research.walkforward.portfolio_evaluator import evaluate_fold_portfolio

    rows: list[dict] = []
    for fold_row in fold_rows:
        fold_id = int(fold_row["fold_id"])
        summary_row = selection_summary_df[
            selection_summary_df["fold_id"] == fold_id
        ]
        if summary_row.empty:
            continue

        top_k_json = summary_row["top_k_features"].iloc[0]
        top_k_labels: list[str] = json.loads(top_k_json)

        if not top_k_labels:
            rows.append({
                "fold_id": fold_id,
                "oos_portfolio_sharpe": float("nan"),
                "n_params_selected": 0,
            })
            continue

        # Parse param labels back to param dicts
        selected_params = [
            dict(kv.split("=") for kv in label.split("|"))
            for label in top_k_labels
        ]
        # Convert numeric values back
        parsed_params: list[dict] = []
        for params in selected_params:
            parsed: dict[str, object] = {}
            for k, v in params.items():
                try:
                    parsed[k] = int(v)
                except ValueError:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            parsed_params.append(parsed)

        train_mask = fold_row["_train_mask"]
        test_mask = fold_row["_test_mask"]
        train_candles = candles_df.loc[train_mask].copy()
        test_candles = candles_df.loc[test_mask].copy()

        # Ensure portfolio candle format
        if "volume" not in train_candles.columns:
            train_candles["volume"] = 0.0
            test_candles["volume"] = 0.0
        if "timeframe" not in train_candles.columns:
            train_candles["timeframe"] = research_config.trading_timeframe
            test_candles["timeframe"] = research_config.trading_timeframe

        try:
            result = evaluate_fold_portfolio(
                train_candles=train_candles,
                test_candles=test_candles,
                selected_params=parsed_params,
                target_series=target,
                binning_config=research_config.binning_params,
                tickers=research_config.tickers,
            )
            rows.append({
                "fold_id": fold_id,
                "oos_portfolio_sharpe": result.oos_portfolio_sharpe,
                "n_params_selected": result.n_params_selected,
            })
        except Exception as exc:
            rows.append({
                "fold_id": fold_id,
                "oos_portfolio_sharpe": float("nan"),
                "n_params_selected": len(parsed_params),
                "error": str(exc),
            })

    return pd.DataFrame(rows)
```

**Step 3: Call `run_portfolio_simulation` from `run_walkforward_research` if `research_config` is provided**

Add optional `research_config` parameter to `run_walkforward_research`:
```python
def run_walkforward_research(
    ...,
    research_config: Any | None = None,  # ResearchConfig, optional
) -> WalkforwardRunReport:
```

After building `selection_summary_df`:
```python
if research_config is not None:
    portfolio_results_df = run_portfolio_simulation(
        candles_df=candles_df,
        target=target,
        fold_rows=fold_rows,
        selection_summary_df=selection_summary_df,
        config=config,
        research_config=research_config,
    )
else:
    portfolio_results_df = pd.DataFrame(
        columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected"]
    )
```

**Step 4: Update `run_continuous_walkforward_pipeline` to pass `research_config`**

```python
walkforward_report = run_walkforward_research(
    ...,
    research_config=config,  # pass full ResearchConfig
)
```

**Step 5: Run tests**

```bash
pytest tests/feature_research/ -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add feature_research/walkforward/runner.py \
        feature_research/walkforward/portfolio_evaluator.py
git commit -m "feat(walkforward): wire Portfolio simulation into WalkforwardRunReport"
```

---

### Task 5: Expose portfolio Sharpe in summary artifacts and output

**Files:**
- Modify: `feature_research/walkforward/io.py`

Add portfolio Sharpe to the walkforward summary markdown/CSV already written by `write_walkforward_artifacts`. Read `report.portfolio_results_df` and append a section showing:
- Mean OOS Portfolio Sharpe across folds
- Per-fold OOS Portfolio Sharpe
- N params selected per fold

This allows you to open `shared_results/continuous/rsi/walkforward/summary.md` and immediately see the Portfolio-equivalent performance.

**Step 1: Update `write_walkforward_artifacts` to include portfolio results**

In `io.py`, in the summary file writing section, append:

```python
if not report.portfolio_results_df.empty and "oos_portfolio_sharpe" in report.portfolio_results_df.columns:
    valid = report.portfolio_results_df["oos_portfolio_sharpe"].dropna()
    mean_sharpe = valid.mean()
    summary_lines.append(f"\n## Portfolio Simulation (Stage 2)\n")
    summary_lines.append(f"Mean OOS Portfolio Sharpe: {mean_sharpe:.3f}\n")
    summary_lines.append(report.portfolio_results_df.to_markdown(index=False))
```

**Step 2: Update tests to handle new `portfolio_results_df` field**

In `tests/feature_research/walkforward/test_io.py`, update any report construction to include the new field:
```python
report = WalkforwardRunReport(
    folds_df=...,
    fold_scores_df=...,
    selection_summary_df=...,
    portfolio_results_df=pd.DataFrame(
        columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected"]
    ),
)
```

**Step 3: Run tests**

```bash
pytest tests/feature_research/ -v
```

Expected: all tests pass.

**Step 4: Commit**

```bash
git add feature_research/walkforward/io.py \
        tests/feature_research/walkforward/test_io.py
git commit -m "feat(walkforward): include Portfolio Sharpe in walkforward artifacts"
```

---

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Portfolio in Stage 1 scoring? | No | Too slow per fold × param combo |
| Portfolio in Stage 2 aggregate? | Yes | Production-identical, uses existing metrics infra |
| Feature column naming | Hard-coded for RSI | Generalize later when more modules added |
| DiversifiedEnsemble injection | Programmatic base_models dict | Avoids control file overhead for research |
| Candle format | Add volume+timeframe cols if missing | Backwards-compatible |
| Error handling | Store NaN + error message | Never crash the whole run on a single fold |

### Generalization Note for Other Modules

The `build_research_portfolio` function currently hard-codes `rsi_signal_D_lookback_{n}` as the feature column name. When adding EWMAC, ATR, or other modules, this function needs to be extended. The standard column naming convention from CLAUDE.md is:
```
{module}_{feature}_{timeframe}_{param}_{value}
```
Use `utils.helpers.parse_feature_column_name` / `format_feature_column_name` to generate these dynamically from the bias spec.
