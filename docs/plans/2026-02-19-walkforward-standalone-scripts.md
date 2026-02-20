# Walkforward Standalone Scripts — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add standalone walkforward scripts for both `continuous_binning` and `rule_based` feature types that run walkforward-only analysis (no EDA) with the enhanced selection algorithm enabled by default.

**Architecture:** Each feature type gets a new `run_continuous_walkforward_pipeline` / `run_rule_based_walkforward_pipeline` function in its existing `pipeline.py`, mirroring how EDA functions are structured. New `run_walkforward.py` entry-point scripts call these functions with `use_enhanced_selection=True`. The `io.py` writer is updated to conditionally include enhanced selection columns in `fold_scores.csv`.

**Tech Stack:** Python 3.9+, `pandas`, `matplotlib`. No new dependencies.

**Prerequisite:** `docs/plans/2026-02-19-top-k-ensemble-selection.md` Tasks 1–9 must be complete. This plan assumes `WalkforwardResearchConfig` already has `use_enhanced_selection`, `trade_freq_min`, `n_robustness_blocks`, `diversity_weight`, `weight_stability`, `weight_robustness` fields (added by Task 6 of that plan) and that `_build_fold_scores` in `runner.py` emits `trade_frequency`, `robustness_score`, `quality_score`, `selected_by_diversity` columns in `fold_scores_df` when `use_enhanced_selection=True` (added by Task 8 of that plan).

**Spec:** [`docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`](../library/Feature_selection/Parameter%20Sensitivity/top_k_ensemble_selection.md)

---

## Read First

Before writing any code, read these files:

- `feature_research/walkforward/io.py` — `write_walkforward_artifacts` hard-codes fold_scores columns; this is the integration target for Task 1
- `feature_research/continuous_binning/pipeline.py` — the EDA pipeline; Task 2 adds a sibling function following the same data-loading pattern
- `feature_research/rule_based/pipeline.py` — same for rule-based; Task 3 follows identical structure
- `tests/feature_research/walkforward/test_io.py` — existing io tests; Task 1 appends to this file
- `tests/feature_research/test_continuous_pipeline_walkforward.py` — existing pipeline tests; Task 2 appends here
- `tests/feature_research/test_rule_based_pipeline_walkforward.py` — existing pipeline tests; Task 3 appends here

---

## Task 1: Update `io.py` to write enhanced selection columns

**Files:**
- Modify: `feature_research/walkforward/io.py`
- Modify: `tests/feature_research/walkforward/test_io.py`

**Context:** `write_walkforward_artifacts` currently selects exactly 6 columns from `fold_scores_df` when writing to CSV. After enhanced selection (Task 8 of the first plan), `fold_scores_df` optionally contains `trade_frequency`, `robustness_score`, `quality_score`, `selected_by_diversity`. These must be written when present, and omitted when absent (legacy non-enhanced runs).

**Step 1: Write the failing tests**

Append to `tests/feature_research/walkforward/test_io.py`:

```python
def _build_enhanced_report() -> WalkforwardRunReport:
    """WalkforwardRunReport with enhanced selection columns in fold_scores_df."""
    folds_df = pd.DataFrame(
        {
            "fold_id": [0],
            "train_start": [pd.Timestamp("2020-01-01")],
            "train_end": [pd.Timestamp("2020-01-31")],
            "test_start": [pd.Timestamp("2020-02-01")],
            "test_end": [pd.Timestamp("2020-02-29")],
            "train_samples": [31],
            "test_samples": [29],
        }
    )
    fold_scores_df = pd.DataFrame(
        {
            "fold_id": [0, 0],
            "param_label": ["x=1", "x=2"],
            "raw_objective": [0.4, 0.6],
            "smoothed_objective": [0.45, 0.65],
            "rank": [2, 1],
            "selected_feature": [False, True],
            "trade_frequency": [0.6, 0.7],
            "robustness_score": [0.5, 0.8],
            "quality_score": [0.55, 0.75],
            "selected_by_diversity": [False, True],
        }
    )
    selection_summary_df = pd.DataFrame(
        {
            "fold_id": [0],
            "selected_feature": ["x=2"],
            "selected_raw_objective": [0.6],
            "selected_smoothed_objective": [0.65],
            "top_k_features": ['["x=2","x=1"]'],
        }
    )
    return WalkforwardRunReport(
        folds_df=folds_df,
        fold_scores_df=fold_scores_df,
        selection_summary_df=selection_summary_df,
    )


def test_write_walkforward_artifacts_includes_enhanced_columns_when_present(
    tmp_path: Path,
) -> None:
    report = _build_enhanced_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    written = pd.read_csv(paths.fold_scores_csv)
    assert written.columns.tolist() == [
        "fold_id",
        "param_label",
        "raw_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
        "trade_frequency",
        "robustness_score",
        "quality_score",
        "selected_by_diversity",
    ]


def test_write_walkforward_artifacts_legacy_report_omits_enhanced_columns(
    tmp_path: Path,
) -> None:
    # _build_report() returns a report WITHOUT enhanced columns (legacy path)
    report = _build_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    written = pd.read_csv(paths.fold_scores_csv)
    assert "trade_frequency" not in written.columns
    assert "robustness_score" not in written.columns
    assert "quality_score" not in written.columns
    assert "selected_by_diversity" not in written.columns
    # Base columns still present
    assert written.columns.tolist() == [
        "fold_id",
        "param_label",
        "raw_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
    ]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/feature_research/walkforward/test_io.py -k "enhanced_columns or legacy_report" -v
```
Expected: 2 FAIL (the io.py still writes exactly 6 columns, so the enhanced-columns test fails; the legacy test may pass by accident but verify).

**Step 3: Modify `io.py`**

In `feature_research/walkforward/io.py`, replace the `fold_scores_df` column selection block (currently lines 78–86):

```python
# OLD
    report.fold_scores_df[
        [
            "fold_id",
            "param_label",
            "raw_objective",
            "smoothed_objective",
            "rank",
            "selected_feature",
        ]
    ].to_csv(paths.fold_scores_csv, index=False, lineterminator="\n")
```

with:

```python
# NEW
    _FOLD_SCORE_BASE_COLS = [
        "fold_id",
        "param_label",
        "raw_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
    ]
    _FOLD_SCORE_ENHANCED_COLS = [
        "trade_frequency",
        "robustness_score",
        "quality_score",
        "selected_by_diversity",
    ]
    fold_score_cols = _FOLD_SCORE_BASE_COLS + [
        c for c in _FOLD_SCORE_ENHANCED_COLS if c in report.fold_scores_df.columns
    ]
    report.fold_scores_df[fold_score_cols].to_csv(
        paths.fold_scores_csv, index=False, lineterminator="\n"
    )
```

**Step 4: Run all io tests to verify they pass**

```bash
pytest tests/feature_research/walkforward/test_io.py -v
```
Expected: All PASS (existing tests + 2 new tests).

**Step 5: Commit**

```bash
git add feature_research/walkforward/io.py tests/feature_research/walkforward/test_io.py
git commit -m "feat(walkforward/io): write enhanced selection columns to fold_scores.csv when present"
```

---

## Task 2: Add `run_continuous_walkforward_pipeline` to `continuous_binning/pipeline.py`

**Files:**
- Modify: `feature_research/continuous_binning/pipeline.py`
- Modify: `tests/feature_research/test_continuous_pipeline_walkforward.py`

**Context:** New function that skips EDA entirely. It loads feature data for every param combo, builds the same lookup-based `evaluate_param_combo` closure used by the EDA pipeline, then calls `run_walkforward_research` and writes artifacts. No `run_eda_for_continuous_feature` or `save_eda_report` calls.

**Step 1: Write the failing test**

Append to `tests/feature_research/test_continuous_pipeline_walkforward.py`:

```python
def test_run_continuous_walkforward_pipeline_returns_report_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.continuous_binning.pipeline import run_continuous_walkforward_pipeline

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 3}},
        ],
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )

    report = run_continuous_walkforward_pipeline(config, tmp_path / "wf_out")

    assert isinstance(report, WalkforwardRunReport)
    walkforward_dir = (
        config.walkforward.output_root
        / "continuous"
        / config.bias_spec["module_name"]
        / "walkforward"
    )
    assert walkforward_dir.exists()
    assert (walkforward_dir / "selection_summary.csv").exists()
    assert (walkforward_dir / "fold_scores.csv").exists()
    assert (walkforward_dir / "folds.csv").exists()


def test_run_continuous_walkforward_pipeline_raises_if_no_combos_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.continuous_binning.pipeline import run_continuous_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
        ],
    )
    # Simulate every combo failing to load
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.load_features_for_combo",
        lambda single_spec, _config: None,
    )

    with pytest.raises(ValueError, match="No param combos loaded successfully"):
        run_continuous_walkforward_pipeline(config, tmp_path / "wf_out")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/feature_research/test_continuous_pipeline_walkforward.py -k "run_continuous_walkforward" -v
```
Expected: `ImportError` — `run_continuous_walkforward_pipeline` does not exist yet.

**Step 3: Implement**

Add the following function to `feature_research/continuous_binning/pipeline.py`, after `run_continuous_eda_pipeline` (before `_build_fold_structure`):

```python
def run_continuous_walkforward_pipeline(
    config: "ResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    """Run walkforward research only (no EDA) for continuous features.

    Loads feature data for all param combos, builds the return-series evaluator,
    runs walkforward research with optional enhanced selection, and writes artifacts
    to ``config.walkforward.output_root``.

    Parameters
    ----------
    config : ResearchConfig
        Research settings. Set ``config.walkforward.use_enhanced_selection = True``
        to activate the three-objective enhanced selection algorithm.
    output_dir : Path
        Created if it does not exist. Not used for artifact output — artifacts
        are written to ``config.walkforward.output_root`` via
        ``write_walkforward_artifacts``.

    Returns
    -------
    WalkforwardRunReport
        Folds, fold scores (with enhanced columns if enabled), and selection summary.

    Raises
    ------
    ValueError
        If no param combos load successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)

    print(f"\n{'='*64}")
    print(f"Continuous Walkforward Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"Enhanced selection: {getattr(config.walkforward, 'use_enhanced_selection', False)}")
    print(f"{'='*64}\n")

    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature = paired["feature"]
        target = paired["target"]
        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)
        print(f"  [{label}] loaded n={len(feature):,}")

    if not successful_param_grid or reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)

    walkforward_report = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type="continuous",
        module_name=str(config.bias_spec["module_name"]),
        config=config.walkforward,
        param_grid=successful_param_grid,
        evaluate_param_combo=_build_walkforward_evaluator(combo_returns),
    )
    stability_figure, _ = plot_selection_stability(
        selection_summary_df=walkforward_report.selection_summary_df,
        top_k=config.walkforward.top_k,
    )
    timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
    write_walkforward_artifacts(
        report=walkforward_report,
        walkforward_stability_figure=stability_figure,
        fold_timeline_figure=timeline_figure,
        feature_type="continuous",
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)

    print(
        f"\nDone. {len(successful_param_grid)}/{len(expanded)} combos "
        f"-> walkforward artifacts written to {config.walkforward.output_root}\n"
    )
    return walkforward_report
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/feature_research/test_continuous_pipeline_walkforward.py -v
```
Expected: All PASS (existing tests + 2 new tests).

**Step 5: Commit**

```bash
git add feature_research/continuous_binning/pipeline.py tests/feature_research/test_continuous_pipeline_walkforward.py
git commit -m "feat(continuous): add run_continuous_walkforward_pipeline"
```

---

## Task 3: Add `run_rule_based_walkforward_pipeline` to `rule_based/pipeline.py`

**Files:**
- Modify: `feature_research/rule_based/pipeline.py`
- Modify: `tests/feature_research/test_rule_based_pipeline_walkforward.py`

**Context:** Identical pattern to Task 2 but for the rule-based pipeline. The only differences are the `feature_type="rule_based"` string and the config type.

**Step 1: Write the failing tests**

Append to `tests/feature_research/test_rule_based_pipeline_walkforward.py`:

```python
def test_run_rule_based_walkforward_pipeline_returns_report_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.rule_based.pipeline import run_rule_based_walkforward_pipeline

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 3,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
        ],
    )
    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )

    report = run_rule_based_walkforward_pipeline(config, tmp_path / "wf_out")

    assert isinstance(report, WalkforwardRunReport)
    walkforward_dir = (
        config.walkforward.output_root
        / "rule_based"
        / config.bias_spec["module_name"]
        / "walkforward"
    )
    assert walkforward_dir.exists()
    assert (walkforward_dir / "selection_summary.csv").exists()
    assert (walkforward_dir / "fold_scores.csv").exists()
    assert (walkforward_dir / "folds.csv").exists()


def test_run_rule_based_walkforward_pipeline_raises_if_no_combos_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.rule_based.pipeline import run_rule_based_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2, "oversold": 25.0, "overbought": 65.0,
                    "strategy_mode": "long", "exit_policy": "threshold_or_bars", "exit_bars": 5,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.rule_based.pipeline.load_features_for_combo",
        lambda single_spec, _config: None,
    )

    with pytest.raises(ValueError, match="No param combos loaded successfully"):
        run_rule_based_walkforward_pipeline(config, tmp_path / "wf_out")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/feature_research/test_rule_based_pipeline_walkforward.py -k "run_rule_based_walkforward" -v
```
Expected: `ImportError`.

**Step 3: Implement**

Add the following function to `feature_research/rule_based/pipeline.py`, after `run_rule_based_eda_pipeline` (before `_build_fold_structure`):

```python
def run_rule_based_walkforward_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> "WalkforwardRunReport":
    """Run walkforward research only (no EDA) for rule-based features.

    Loads feature data for all param combos, builds the return-series evaluator,
    runs walkforward research with optional enhanced selection, and writes artifacts
    to ``config.walkforward.output_root``.

    Parameters
    ----------
    config : RuleBasedResearchConfig
        Research settings. Set ``config.walkforward.use_enhanced_selection = True``
        to activate the three-objective enhanced selection algorithm.
    output_dir : Path
        Created if it does not exist. Not used for artifact output — artifacts
        are written to ``config.walkforward.output_root`` via
        ``write_walkforward_artifacts``.

    Returns
    -------
    WalkforwardRunReport
        Folds, fold scores (with enhanced columns if enabled), and selection summary.

    Raises
    ------
    ValueError
        If no param combos load successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    populate_cache_if_needed(config)

    expanded = expand_bias_specs(config.bias_spec)

    print(f"\n{'='*64}")
    print(f"Rule-Based Walkforward Pipeline: {config.bias_spec['module_name'].upper()}")
    print(f"Tickers : {[t.name for t in config.tickers]}")
    print(f"Period  : {config.start.date()} -> {config.end.date()}")
    print(f"Combos  : {len(expanded)}")
    print(f"Enhanced selection: {getattr(config.walkforward, 'use_enhanced_selection', False)}")
    print(f"{'='*64}\n")

    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None

    for single_spec in expanded:
        combo = single_spec["params"]
        label = param_combo_label(combo)

        data = load_features_for_combo(single_spec, config)
        if data is None:
            print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature = paired["feature"]
        target = paired["target"]
        combo_returns[_combo_key(combo)] = _normalize_series_datetime_index(feature.mul(target))
        successful_param_grid.append(dict(combo))
        if reference_index is None:
            reference_index = _normalize_datetime_index(target.index)
        print(f"  [{label}] loaded n={len(feature):,}")

    if not successful_param_grid or reference_index is None:
        raise ValueError("No param combos loaded successfully; check cache and bias_spec.")

    reference_target = pd.Series(0.0, index=reference_index, name="walkforward_target")
    reference_candles = pd.DataFrame({"close": reference_target}, index=reference_index)

    walkforward_report = run_walkforward_research(
        candles_df=reference_candles,
        target=reference_target,
        feature_type="rule_based",
        module_name=str(config.bias_spec["module_name"]),
        config=config.walkforward,
        param_grid=successful_param_grid,
        evaluate_param_combo=_build_walkforward_evaluator(combo_returns),
    )
    stability_figure, _ = plot_selection_stability(
        selection_summary_df=walkforward_report.selection_summary_df,
        top_k=config.walkforward.top_k,
    )
    timeline_figure, _ = plot_fold_timeline(folds_df=walkforward_report.folds_df)
    write_walkforward_artifacts(
        report=walkforward_report,
        walkforward_stability_figure=stability_figure,
        fold_timeline_figure=timeline_figure,
        feature_type="rule_based",
        module_name=str(config.bias_spec["module_name"]),
        root_dir=config.walkforward.output_root,
    )
    plt.close(stability_figure)
    plt.close(timeline_figure)

    print(
        f"\nDone. {len(successful_param_grid)}/{len(expanded)} combos "
        f"-> walkforward artifacts written to {config.walkforward.output_root}\n"
    )
    return walkforward_report
```

**Step 4: Run all rule-based pipeline tests to verify they pass**

```bash
pytest tests/feature_research/test_rule_based_pipeline_walkforward.py -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add feature_research/rule_based/pipeline.py tests/feature_research/test_rule_based_pipeline_walkforward.py
git commit -m "feat(rule_based): add run_rule_based_walkforward_pipeline"
```

---

## Task 4: Create `continuous_binning/run_walkforward.py`

**Files:**
- Create: `feature_research/continuous_binning/run_walkforward.py`

**Context:** Entry-point script for continuous-feature walkforward. It loads the researcher's config, overrides `walkforward.enabled=True` and `walkforward.use_enhanced_selection=True` using `dataclasses.replace` (safe on frozen dataclasses), then delegates to `run_continuous_walkforward_pipeline`. Researchers run this directly as a script — no test needed (it is a thin CLI wrapper with no logic of its own).

**Step 1: Create the file**

```python
# feature_research/continuous_binning/run_walkforward.py
"""Entry point for the continuous binning walkforward research pipeline.

Runs walkforward-only analysis (no EDA) with enhanced selection enabled.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_walkforward.py

Artifacts are written to the ``output_root`` defined in ``config.walkforward``
(default: ``feature_research/shared_results/continuous/{module_name}/walkforward/``).

To customise tickers, dates, bias_spec, or walkforward parameters, edit
``feature_research/continuous_binning/config.py``.

To disable enhanced selection, set ``use_enhanced_selection=False`` in the
``dataclasses.replace`` call below, or override ``load_config()`` directly.
"""
import dataclasses
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.pipeline import run_continuous_walkforward_pipeline

if __name__ == "__main__":
    config = load_config()

    # Enable walkforward and enhanced selection regardless of what load_config() returns.
    walkforward = dataclasses.replace(
        config.walkforward,
        enabled=True,
        use_enhanced_selection=True,
    )
    config = dataclasses.replace(config, walkforward=walkforward)

    report = run_continuous_walkforward_pipeline(config, config.reports_dir / "walkforward")
    n_folds = len(report.folds_df)
    print(
        f"Walkforward complete. {n_folds} folds. "
        f"Artifacts written to {config.walkforward.output_root}"
    )
```

**Step 2: Verify the script is importable**

```bash
python -c "import feature_research.continuous_binning.run_walkforward"
```
Expected: No output (script-level code only runs under `__main__`).

**Step 3: Commit**

```bash
git add feature_research/continuous_binning/run_walkforward.py
git commit -m "feat(continuous): add run_walkforward.py entry-point script with enhanced selection"
```

---

## Task 5: Create `rule_based/run_walkforward.py`

**Files:**
- Create: `feature_research/rule_based/run_walkforward.py`

**Context:** Identical pattern to Task 4 but for rule-based features.

**Step 1: Create the file**

```python
# feature_research/rule_based/run_walkforward.py
"""Entry point for the rule-based feature walkforward research pipeline.

Runs walkforward-only analysis (no EDA) with enhanced selection enabled.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/rule_based/run_walkforward.py

Artifacts are written to the ``output_root`` defined in ``config.walkforward``
(default: ``feature_research/shared_results/rule_based/{module_name}/walkforward/``).

To customise tickers, dates, bias_spec, or walkforward parameters, edit
``feature_research/rule_based/config.py``.

To disable enhanced selection, set ``use_enhanced_selection=False`` in the
``dataclasses.replace`` call below, or override ``load_config()`` directly.
"""
import dataclasses
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.rule_based.config import load_config
from feature_research.rule_based.pipeline import run_rule_based_walkforward_pipeline

if __name__ == "__main__":
    config = load_config()

    # Enable walkforward and enhanced selection regardless of what load_config() returns.
    walkforward = dataclasses.replace(
        config.walkforward,
        enabled=True,
        use_enhanced_selection=True,
    )
    config = dataclasses.replace(config, walkforward=walkforward)

    report = run_rule_based_walkforward_pipeline(config, config.reports_dir / "walkforward")
    n_folds = len(report.folds_df)
    print(
        f"Walkforward complete. {n_folds} folds. "
        f"Artifacts written to {config.walkforward.output_root}"
    )
```

**Step 2: Verify the script is importable**

```bash
python -c "import feature_research.rule_based.run_walkforward"
```
Expected: No output.

**Step 3: Commit**

```bash
git add feature_research/rule_based/run_walkforward.py
git commit -m "feat(rule_based): add run_walkforward.py entry-point script with enhanced selection"
```

---

## Task 6: Full regression test

**Purpose:** Verify all walkforward tests pass together and no existing tests regressed.

**Step 1: Run the full walkforward + feature_research test suite**

```bash
pytest tests/feature_research/ tests/feature_research/walkforward/ tests/unit-tests/utils/test_grid_smoothing.py -v
```
Expected: All PASS, no regressions.

**Step 2: Commit (if any fixes were needed)**

Only commit if you made additional fixes during the regression run.

---

## Done

After Task 6, the enhanced walkforward is:

- **Articulately persisted**: `fold_scores.csv` automatically includes `trade_frequency`, `robustness_score`, `quality_score`, `selected_by_diversity` when enhanced selection is active (Task 1)
- **Runnable independently**: `run_continuous_walkforward_pipeline` and `run_rule_based_walkforward_pipeline` skip EDA and run only walkforward analysis (Tasks 2–3)
- **Script-accessible**: `python feature_research/continuous_binning/run_walkforward.py` and `python feature_research/rule_based/run_walkforward.py` work out of the box with enhanced selection on (Tasks 4–5)

**To run walkforward research with enhanced selection:**

```bash
# Continuous features
python feature_research/continuous_binning/run_walkforward.py

# Rule-based features
python feature_research/rule_based/run_walkforward.py
```

**To enable walkforward within an existing EDA run**, set in `config.py`:
```python
walkforward = WalkforwardResearchConfig(
    train_start=...,
    train_end=...,
    enabled=True,
    use_enhanced_selection=True,  # activates three-objective selection
    top_k=5,
    # ... other fields
)
```
