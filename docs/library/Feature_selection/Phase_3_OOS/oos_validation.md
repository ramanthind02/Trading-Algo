# Validation & OOS Phases

> [!note] Status: Library reference — Phases 3 & 4 of [[Feature_selection/pipeline]]  
> Last updated: 2026-04-07

---

## Overview

After IS EDA and permutation (Phases 1–2), a single researcher-selected param combo is handed off to two sequential single-fold evaluation phases:

| Phase | Split | Train window | Hold-out window | Entrypoint |
|-------|-------|-------------|----------------|------------|
| **3 — Validation** | 2019-01 → 2022-12 | 2000-01 → 2018-12 | 2019-01 → 2022-12 | `python -m feature_research validation` |
| **4 — OOS** | 2023-01 → 2025-09 | 2000-01 → 2022-12 | 2023-01 → 2025-09 | `python -m feature_research oos` |

The OOS train window **includes the validation period** (2019–2022). When both `validation_window` and `oos_window` are set in config, `_effective_oos_window()` extends the OOS train start to `validation_window.train_start` automatically.

Dates are config-driven via `ResearchConfig.validation_window` and `ResearchConfig.oos_window` in `feature_research/config.py`.

---

## Param combo handoff: `eval_bias_spec`

Both phases evaluate a **single fixed combo** rather than the full EDA grid. The researcher pins this combo after IS review by setting `eval_bias_spec` in `InSamplePhaseDefaultsConfig`:

```python
# feature_research/config.py  (inside load_config)
_defaults = InSampleDefaultsCatalog.default_for(timeframe)
in_sample_defaults = replace(
    _defaults,
    continuous=replace(
        _defaults.continuous,
        eval_bias_spec={
            "module_name": "cyclical_rsi",
            "timeframes": [timeframe],
            "params": {
                "long_period": [100],   # ← single values from IS selection
                "rsi_period":  [2],
                "short_period": [4],
            },
        },
    ),
)
```

`ResearchConfig.eval_bias_spec` falls back to the full `bias_spec` (EDA grid) when not set — safe to leave unset during early IS exploration. Once set, every downstream phase (validation, OOS, permutation scripts) uses this spec exclusively.

---

## Portfolio simulation for equity curves

Equity curves in both phases are generated using the Portfolio class rather than the IS EDA metric (`signal × EWSD-target`). This gives realistic simulated P&L:

### Signal → position fraction

```
signal (0 or 1, one row per bar per ticker from bias node)
    │
    ▼  TFPortfolio(ensembles=[])
    │  .fit(calculate_returns_from_candles(train_candles))
    │    → IDM = √(1 / (mean_corr + 0.01)),  capped at 2.5
    │    → instrument_weight = 1 / N_tickers  (equal weight)
    ▼
  TFPortfolio.predict(combined_forecasts)   [forecast_score = signal]
    │  position_fraction = signal × instrument_weight × IDM
    ▼
  calculate_strategy_returns_from_positions(positions_df, candles)
    │  • position at bar t applied to return of bar t+1  (1-bar lookahead-free shift)
    │  • daily_return_t = position_fraction_{t−1} × log(close_t / close_{t−1})
    │  • returns summed across tickers per date
    ▼
  cumsum per ticker  →  equity_curve_*.csv
```

**Key**: `TFPortfolio` is used purely for IDM and instrument weights — no base models, no DiversifiedEnsemble. The signal is passed directly as the `forecast_score`. This gives position sizing consistent with how the combo would be sized in production (single-signal Carver allocation with diversification).

### Training candles for IDM

The IDM fit uses candles **strictly before the holdout window start** (`extended_start → holdout_start − 1 day`). This prevents information leakage from the holdout period into the IDM estimate.

| Phase | IDM training candles | Holdout candles |
|-------|---------------------|----------------|
| Validation | 2000-01 → 2018-12 | 2019-01 → 2022-12 |
| OOS | 2000-01 → 2022-12 | 2023-01 → 2025-09 |

### Implementation

`feature_research.research_table_exports`:

- `_build_portfolio_positions_df(signal, ticker, train_candles, timeframe)` — fits `TFPortfolio`, calls `predict()`, returns `{ticker, datetime, position_fraction}` DataFrame
- `_portfolio_equity_frames_for_selection(...)` — computes per-ticker `calculate_strategy_returns_from_positions`, formats equity curve rows
- `write_walkforward_equity_powerbi_csvs(..., portfolio_candles=candles)` — orchestrates holdout and extended slices; falls back to `signal × target` when `portfolio_candles=None`

Called from `feature_research.pipelines._shared._run_evaluation_pipeline` after `run_walkforward_research`.

---

## Validation phase (Phase 3)

### What runs

`python -m feature_research validation`  
→ `feature_research.validation.run_validation.main`  
→ `run_validation_pipeline(config)` → `_run_evaluation_pipeline(phase="validation", config)`

1. Loads cache-backed signal and target for `eval_bias_spec` combo over full date range
2. Builds a single fold row: train = `[validation_window.train_start, train_end]`, test = `[test_start, test_end]`
3. Runs `run_walkforward_research` → `WalkforwardRunReport` (fold scores, selection summary, tearsheets)
4. Writes Power BI equity CSVs via `write_walkforward_equity_powerbi_csvs`

### Power BI artifacts

Written to **`output_root / powerbi / validation /`** (fixed path; `module_name` and internal `feature_type` are **not** in the folder name so Power BI data sources stay valid when you change feature).

| File | Contents |
|------|----------|
| `equity_curve_validation_only.csv` | Portfolio simulation returns, **validation window only** (2019–2022). Columns: `datetime`, `param_combo_label`, `feature_name`, `ticker`, `strategy_return`, `cumulative_strategy_return`, `fold_id`, **`rolling_sharpe_annualized`** (rolling mean/std × √`bars_per_year`, default **126** daily bars), **`rolling_sharpe_window_bars`** (the window used). |
| `equity_curve_train_and_validation.csv` | Same schema, **train + validation combined** (2000–2022); rolling Sharpe is computed **within** each ticker/fold/combo after sorting by date (so the validation-only file does not use pre-validation bars in the rolling window). |

Both files have one row per trading bar per ticker. Import both into Power BI and use `fold_id` to filter (single fold = fold_id 0).

### Tearsheets

HTML reports stay under the feature-specific run folder:  
`output_root / signed_signal / <module_name> / validation / tearsheets /`

- `validation_ensemble_tearsheet.html` — validation period QuantStats report  
- `train_ensemble_tearsheet.html` — training period QuantStats report  
- `train_and_validation_ensemble_tearsheet.html` — combined period  

---

## OOS phase (Phase 4)

### What runs

`python -m feature_research oos`  
→ `feature_research.oos.run_oos.main`  
→ `run_oos_pipeline(config)` → `_run_evaluation_pipeline(phase="oos", config)`

Structurally identical to validation; differences:

- Window: `_effective_oos_window(config)` — train start = `validation_window.train_start` when validation is configured (so OOS train covers 2000–2022)
- Test period: `oos_window.test_start → test_end` (2023–2025)

### Power BI artifacts

Written to **`output_root / powerbi / oos /`** (same stable-path rule as validation).

| File | Contents |
|------|----------|
| `equity_curve_oos_test_only.csv` | Portfolio simulation returns, **OOS test window only** (2023–2025); includes the same **rolling Sharpe** columns as validation exports. |
| `equity_curve_train_val_and_test.csv` | Same schema, **full period** (2000–2025); rolling window uses only bars present in this file (test-only file does not borrow train/val history for the roll). |

### Power BI usage

Load all four equity CSVs (2 validation + 2 OOS) in a single dataset. Relate on `param_combo_label` + `feature_name`. Use a slicer on a `phase` column (add as a calculated column or combine with a `phase` literal before import) to compare:

- Validation only vs. train+val
- OOS only vs. full history

This lets you visually check whether the hold-out performance is consistent with in-sample build-up.

---

## Permutation scripts (optional)

After visual review, a vector-shuffle permutation null can be run on the validation or OOS window:

```
python -m feature_research validation-permutation [--nreps N --seed S]
python -m feature_research oos-permutation        [--nreps N --seed S]
```

These use the same `eval_bias_spec` combo and the same window definitions as the main validation/OOS pipelines. Results are written alongside the equity CSVs.

---

## Configuration reference

All window definitions live in `feature_research/config.py` → `load_config()`:

```python
validation_window = OOSWindowConfig(
    train_start=datetime(2000, 1, 1),
    train_end=datetime(2018, 12, 31),
    test_start=datetime(2019, 1, 1),
    test_end=datetime(2022, 12, 31),
)

oos_window = OOSWindowConfig(
    train_start=datetime(2000, 1, 1),
    train_end=datetime(2022, 12, 31),
    test_start=datetime(2023, 1, 1),
    test_end=datetime(2025, 9, 18),
)
```

`OOSWindowConfig` enforces `train_end < test_start < test_end` at construction.

---

## Relationship to IS equity curve

| Phase | Equity curve source | Return metric | Per-ticker? |
|-------|--------------------|--------------------|-------------|
| IS EDA (`equity_curve.csv`) | `signal × log_return_ewsd` (EWSD-normalised) | Signal quality metric (not P&L) | Yes |
| Validation / OOS | `position_fraction × actual_log_return` via `TFPortfolio` | Realistic % P&L | Yes |

The IS equity curve is optimised for param screening (vol-normalised, comparable across instruments). The validation/OOS curves are optimised for realistic assessment of the selected combo's profitability.

---

> [!info] See also
> - [[Feature_selection/pipeline]] — full pipeline overview (Phases 0–5)
> - [[Feature_selection/Phase_2_WF/param_stability]] — selection rule applied at graduation
> - [[Feature_selection/Phase_2_WF/walkforward]] — multi-fold walk-forward (reference)
> - [[Ensemble/portfolio]] — TFPortfolio, IDM, instrument weights
