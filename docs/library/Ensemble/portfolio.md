# Portfolio & Forecast Pipeline Architecture

> **Scope:** Full pipeline from BaseModels to tradeable contracts (Robert Carver methodology).

> [!note] Cache-native orchestration
> `TFPortfolio` and `GlobalPortfolio` now provide cache-native adapters (`fit_from_cache`, `predict_from_cache`) driven by `PortfolioCacheQuery`. Legacy candle-frame entrypoints remain as temporary compatibility shims.

---

## Pipeline

The pipeline has two levels. Within each timeframe, the per-TF path produces model streams.
Those streams are then adapter-encoded and combined cross-ticker/timeframe by the existing
`WeightLayer` inside `GlobalPortfolio` before final position sizing.

```
BaseModels → DiversifiedEnsemble → WeightLayer → TFPortfolio (D) ──┐
  (0/1)        (per-model           (combined       (position        │
  signals)      forecasts)           forecast        fractions       ├→ GlobalPortfolio(adapter + WeightLayer) → PositionSizer
                                     + FDM)          + IDM)          │   (cross-stream        (top-level        (contract
                                                  TFPortfolio (W) ──┤    diversification)     orchestrator)     quantities)
                                                  TFPortfolio (M) ──┘
```

`Portfolio` is a backward-compatible alias for `TFPortfolio`. For single-timeframe use the two are interchangeable.

---

## Key Formulas

### DiversifiedEnsemble — vol-scaled forecast per model
```
F = (τ / σ) × X      (then clipped at 2.0)
```
- `τ` = target volatility (`target_volatility_`, default 0.15)
- `σ` = instrument blended (EWSD) volatility, floored at `1e-8`
- `X` = signal (0 when inactive; carries the signed direction when active)

The implementation in [`ensemble/diversified_ensemble.py`](../../../ensemble/diversified_ensemble.py)
uses **direct Carver vol-targeting with `h_i = 1`** (the `√h_i` exposure-fraction term has been
dropped). The per-bar "forecast if active" is `τ / σ`, capped at `2.0`, then multiplied by the
signal. Exposure fractions (`model_exposure_fractions_`, `1/n_bins`) are still tracked as fitted
metadata but no longer divide the forecast.

### WeightLayer — Forecast Diversification Multiplier
```
FDM = min(√(1 / (mean_corr + 0.01)), fdm_max)      (default fdm_max = 2.0)
```
Applied after combining per-model forecasts into a single `forecast_score`, which is then
clipped to `[-2.0, 2.0]`.

### Portfolio — Instrument Diversification Multiplier
```
IDM = min(√(1 / (mean_corr + 0.01)), 2.5)
position = forecast × instrument_weight × IDM
```
`mean_corr` here is across instrument return correlations (not forecast correlations).

### PositionSizer — contracts
```
contracts = (position_fraction × capital) / (price × multiplier × fx_rate)
```

---

## Multiplier Comparison

| Multiplier | Layer | Cap | Correlation source |
|---|---|---|---|
| FDM | WeightLayer (per-TF) | **2.0** | Forecast value correlations (model-level, within one timeframe) |
| IDM | TFPortfolio | **2.5** | Instrument return correlations (instrument-level, within one timeframe) |
| cross-stream FDM | GlobalPortfolio-internal WeightLayer | **2.0** | Encoded stream correlations across tickers/timeframes/strategies |

All three use the same functional form: `√(1 / (mean_corr + ε))` with `ε = 0.01` (shared helper
`_correlation_multiplier_from_corr_matrix` in `ensemble/weight_layer.py`). The higher cap on IDM
reflects that instrument-level diversification can be greater than forecast-level diversification.

---

## File Structure

```
ensemble/
├── diversified_ensemble.py        # Owns base models; generates per-model vol-scaled forecasts
├── weight_layer.py                # Combines forecasts; applies FDM (factory + ClusteredWeightLayer)
├── weight_hierarchy.py            # Nested-tree parsing / equal-split weights for hierarchy modes
├── portfolio.py                   # Public re-exports: TFPortfolio, GlobalPortfolio, Portfolio,
│                                  #   PortfolioWorld, PortfolioCacheQuery, helpers
├── portfolio_impl/
│   ├── tf_portfolio.py            # TFPortfolio (per-TF IDM); Portfolio = backward-compatible alias
│   ├── global_portfolio_impl.py   # GlobalPortfolio (top-level orchestrator + cross-TF WeightLayer)
│   ├── portfolio_cache.py         # PortfolioCacheQuery (cache-native request dataclass)
│   ├── portfolio_returns.py       # calculate_idm_from_returns, return matrices
│   ├── portfolio_global_streams.py# build_daily_grid, align_forecast_vectors_to_daily_grid, ...
│   └── global_weight_layer_adapter.py  # encode/decode synthetic __GLOBAL__ streams
└── __init__.py                    # exports DiversifiedEnsemble, GlobalPortfolio, Portfolio,
                                   #   PortfolioWorld, TFPortfolio, WeightLayer, BaseWeightLayer,
                                   #   ClusteredWeightLayer

execution/
├── position_sizer.py              # PositionSizer, ContractSpec, Position, RoundingMethod
└── __init__.py
```

`Portfolio` is exported from both `ensemble` and `ensemble.portfolio` as a backward-compatible
alias for `TFPortfolio` (defined in `ensemble/portfolio_impl/tf_portfolio.py`). `PortfolioCacheQuery`
is re-exported from `ensemble.portfolio` (defined in `ensemble/portfolio_impl/portfolio_cache.py`).

---

## Control File Format (JSON)

Ensembles are configured and persisted via JSON control files validated by
`validate_control_file` in [`ensemble/ensemble_utils.py`](../../../ensemble/ensemble_utils.py).
The `metadata.is_fit` flag tracks training state. Every base-model entry must use the
**signed-signal** contract (legacy binning model types are rejected — see [base model](base_model.md)).

```json
{
  "metadata": { "is_fit": true, "base_tf": "D" },
  "base_models": [
    {
      "name": "rsi_signal_D_lookback_14_long",
      "model_type": "signed_signal",
      "feature_column": "rsi_signal_D_lookback_14",
      "strategy": "long",
      "bias_node_spec": { "module_name": "rsi", "timeframes": ["D"], "params": { "lookback": 14 } }
    }
  ],
  "fitted_ensemble": {
    "weights": {},
    "exposure_fractions": {},
    "feature_names": [],
    "target_volatility": 0.15,
    "unique_tickers": [],
    "instrument_weights": {},
    "n_tickers": 0
  }
}
```

Required base-model keys: `name`, `model_type` (`"signed_signal"`), `feature_column`, `strategy`,
`bias_node_spec`. When `metadata.is_fit` is `true`, `fitted_ensemble` is required and must contain
`weights`, `exposure_fractions`, `feature_names`, `target_volatility`, `unique_tickers`,
`instrument_weights`, and `n_tickers`. When `is_fit` is `false`, `fitted_ensemble` must be absent.

Control files live in the [vault](../Vault/vault.md). `is_fit: false` → the ensemble must be
trained before prediction.

---

## Usage Examples

### Single-timeframe (unchanged)

```python
from ensemble import DiversifiedEnsemble, WeightLayer, Portfolio
from execution import PositionSizer, ContractSpec

# 1. DiversifiedEnsemble: per-model vol-scaled forecasts
ensemble = DiversifiedEnsemble(control_file_path="config.json")
ensemble.fit(X, ticker, volatility, y)
forecasts = ensemble.predict(X, ticker, volatility)
# → DataFrame["ticker", "model_name", "forecast", "signal"]

# 2. WeightLayer: combine + apply FDM
wl = WeightLayer(fdm_max=2.0)
wl.fit([forecasts], signals_df)
combined = wl.combine([forecasts])
# → DataFrame["ticker", "forecast_score"]

# 3. Portfolio (= TFPortfolio): instrument weights + IDM
portfolio = Portfolio(max_position_pct=2.0, idm_max=2.5)
portfolio.fit(instrument_returns)
positions = portfolio.predict(combined)
# → DataFrame["ticker", "forecast_score", "position_fraction"]

# 4. PositionSizer: convert to contracts
specs = {"ES": ContractSpec(ticker="ES", price=4800, multiplier=50)}
sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
contracts = sizer.calculate_positions(positions)
# → DataFrame with "contracts", "notional_value", "notional_pct"
```

### Multi-timeframe (two-level architecture)

```python
from ensemble import TFPortfolio, GlobalPortfolio, WeightLayer
from ensemble.portfolio import PortfolioCacheQuery
from utils.enums import TimeFrame

# 1. Build one TFPortfolio per timeframe
tf_daily = TFPortfolio(trading_timeframe=TimeFrame.D, ...)
tf_weekly = TFPortfolio(trading_timeframe=TimeFrame.W, ...)

# 2. Wrap in GlobalPortfolio with WeightLayer (adapter-driven global combine)
global_p = GlobalPortfolio(
    tf_portfolios=[tf_daily, tf_weekly],
    weight_layer=WeightLayer(weight_method="equal_signal", fdm_max=2.0),
)

# 3. Fit from central cache query
query = PortfolioCacheQuery(
    tickers=("ES", "NQ"),
    start=train_start,
    end=train_end,
    timeframes=(TimeFrame.D, TimeFrame.W),
)
global_p.fit_from_cache(query, instrument_returns=returns_df)

# 4. Predict from central cache query
positions = global_p.predict_from_cache(query)
# → DataFrame["ticker", "datetime", "forecast_score", "position_fraction"]
```

---

## Portfolio research — portfolio addition gate

This section describes the local portfolio-fit workflow for deciding whether to **add a new vault ensemble (candidate)** to an existing portfolio configuration.

> [!important]
> Canonical workflow authority lives in `docs/SaaS/robustness_tests/portfolio_addition.md`. Use that document for the conceptual gate definition and contamination rules.

> [!note]
> Some local code, config, and CLI names still use the older term `inclusion`. Treat that as migration-era compatibility terminology for the **portfolio addition** phase.

The local workflow reuses the same **train / validation / test** windows as `portfolio_research.config.load_config()` and the same global portfolio scoring path as `run_single_phase_for_prop_firm` (validation = fit on train, score on validation; test = fit on train+validation, score on test).

**Principle:** The target workflow is `exploration -> validation -> portfolio_addition`. For local compatibility tooling, the portfolio-addition checks still appear under `inclusion`-named config and CLI surfaces.

| Step | What it checks | Default rule (tunable in config) |
|------|----------------|----------------------------------|
| **Gate 1 — diversification** | For each **same-`base_tf` peer** ensemble: **Pearson** and **Spearman** correlation of **ensemble-level** validation forecast streams (mean over intersected tickers per peer). | Pass only if **every** peer has finite values with Pearson `< corr_max` and Spearman `< spearman_corr_max` (defaults **0.7**). No same-TF peers → vacuous pass. |
| **Gate 2 — standalone performance** | Candidate **alone** in `ensemble_dirs`; validation **Sharpe / Sortino / Calmar** on combined returns. | Pass if Sharpe > `sharpe_min` (default **0.3**). |
| **Standalone vs each baseline** | One row per baseline ensemble + candidate: Sharpe / Sortino / Calmar on **train**, **validation**, and **train+validation** (single-ensemble portfolio each). | Report only (see CSV). |
| **Uplift** | Full portfolio **without** vs **with** candidate on **train**, **validation**, and **concatenated train+validation**; Sharpe / Sortino / Calmar each window. | Pass if **each** window satisfies `Sharpe_with > Sharpe_without - uplift_slack` (default slack **0.05**). |
| **Optional test confirmation** | Opt-in: standalone candidate **test** Sharpe vs **validation** Sharpe ratio. | Pass if `Sharpe_test / Sharpe_val > test_sharpe_ratio_min` (default **0.5**). |

> **Correlation estimators:** Gate 1 uses **Pearson** and **Spearman** on aligned **validation** `forecast_score` series per ticker. The live **WeightLayer** FDM uses **Ledoit–Wolf** on **standardized in-sample** pivots at fit time — same economic object (forecasts), different estimator and window. See [weight layer](weight_layer.md) for production FDM.

### Configuration

- **`PortfolioInclusionConfig`** in `feature_research/config.py`: compatibility-era config for the portfolio-addition gate. It holds thresholds (`corr_max`, `spearman_corr_max`, `sharpe_min`, `uplift_slack`, `test_sharpe_ratio_min`), `output_subdir` (default `inclusion`), and optional default candidate path/key.
- **`ResearchConfig.portfolio_inclusion`**: current compatibility container for those defaults while the package migrates toward explicit `portfolio_addition` naming.
- **Baseline portfolio** (tickers, train/validation/test windows, `ensemble_dirs`, weight layer, etc.) still comes from **`portfolio_research.config.load_config()`** — the CLI loads both configs.

### CLI

From the repo root (venv Python), pass a **repo-relative** path to the candidate ensemble directory (same style as `ensemble_dirs` values):

```powershell
.\.venv\Scripts\python.exe -m feature_research.run_inclusion_gates
```

With ``portfolio_inclusion.candidate_repo_relative_path`` set in ``feature_research.config.load_config()``, no CLI arguments are required. Optional overrides: ``--candidate-path``, ``--candidate-key``, ``--emit-tearsheets`` / ``--no-emit-tearsheets``, ``--no-preflight``.

The command name is expected to change as the migration finishes; until then, interpret it as the local entrypoint for the portfolio-addition phase.

### Artifacts

Under `output_root` / `portfolio_inclusion.output_subdir`: `inclusion_<candidate_key>_summary.csv`, `_corr_by_peer_and_ticker.csv` (Pearson + Spearman per peer–ticker), `_standalone_by_ensemble.csv`, `_uplift_by_window.csv`.

### Code entrypoints

- `feature_research/inclusion_gates.py` — compatibility implementation of the portfolio-addition decision (`run_inclusion_decision`, `pearson_corr_candidate_vs_each_peer`, `write_inclusion_reports`, …)
- `feature_research/run_inclusion_gates.py` — compatibility CLI

**See also:** [Cache user guide](../Cache/user_guide.md) (portfolio workflow and preflight), [Vault user guide](../Vault/user_guide.md) (ensemble layout).

---

---

## Futures Contract Simulation (`portfolio_research.futures_sim`)

An optional parallel simulation path that converts `position_fraction` signals to **integer futures contracts** and produces tearsheets and diagnostics alongside the standard log-return tearsheets.  Activated by setting `futures_sim.enabled = True` in `portfolio_research.config.load_config()`.

### Purpose

The standard research pipeline works in fractional-return space (`position_fraction × instrument_return`, summed across tickers).  That is the *most accurate* continuous backtest, but it does not model the rounding that occurs when trading real micro-futures contracts.  The sim layer answers:

- How many contracts would we have held each bar, and what was the actual dollar PnL?
- How large is the rounding-induced tracking error vs the fractional baseline?
- Were there days where margin requirements exceeded account capital?

### Configuration

All parameters live in `PortfolioResearchConfig.futures_sim` (`FuturesSimConfig`):

```python
from portfolio_research.config import (
    FuturesSimConfig, FuturesInstrumentSpec, LeverageMode
)

futures_sim = FuturesSimConfig(
    enabled=True,                      # flip to activate
    account_capital=100_000.0,         # USD starting size
    leverage_mode=LeverageMode.FINITE, # or INFINITE to skip margin checks
    emit_diagnostics_csv=True,
    emit_tracking_error_csv=True,
    instrument_specs={
        "NQ": FuturesInstrumentSpec(
            multiplier=2.0,            # MNQ: $2/point
            margin_long=3_653.0,       # maintenance margin per contract
            margin_short=3_576.0,
            product_code="MNQ",
        ),
        "ES": FuturesInstrumentSpec(
            multiplier=5.0,            # MES: $5/point
            margin_long=2_413.0,
            margin_short=2_265.0,
            product_code="MES",
        ),
        "GC": FuturesInstrumentSpec(
            multiplier=10.0,           # MGC: $10/oz (10 troy oz)
            margin_long=2_817.0,
            margin_short=2_817.0,
            product_code="MGC",
        ),
    },
)
```

`LeverageMode.FINITE` flags any bar where `|contracts| × margin_per_contract > account_capital`.  `LeverageMode.INFINITE` skips the check (useful for large-capital sensitivity analysis).

### Sizing formula

```
contract_value = price × multiplier
contracts      = round(position_fraction × account_capital / contract_value)
```

Rounding is `round()` (banker's rounding to nearest integer).  The sign of `contracts` tracks direction (positive = long, negative = short).

### PnL convention

Both legs use **simple returns** so the comparison is dollar-for-dollar:

| Path | Formula |
|---|---|
| Discrete | `contracts × (next_close − close) × multiplier` |
| Fractional | `position_fraction × account_capital × simple_return` |

Daily % returns for tearsheets = `daily_USD_PnL / account_capital`, so all QuantStats metrics (Sharpe, drawdown, etc.) are directly comparable to the standard portfolio tearsheets.

### Outputs

All written to `output_root / {phase} / futures_sim /`:

| File | Content |
|---|---|
| `{phase}_diagnostics.csv` | Per-bar per-ticker: position_fraction, price, contracts, notional, margin_required, margin_available, leverage_breach, discrete_pnl, fractional_pnl |
| `{phase}_tracking_error.csv` | Per-date aggregate: discrete_total_pnl, fractional_total_pnl, discrete_pct_return, fractional_pct_return, daily_tracking_error_usd, daily_tracking_error_pct, cumulative_te_usd, annualised_te_vol_usd, any_leverage_breach |
| `{phase}_tracking_error_summary.html` | Self-contained HTML: portfolio-level summary table, cumulative TE sparkline chart, per-ticker breakdown, interpretation guide |
| `{phase}_discrete_tearsheet.html` | Standard QuantStats HTML tearsheet driven by `discrete_pct_return` (strategy) vs `fractional_pct_return` (baseline); identical format to regular portfolio tearsheets |

### Interpreting tracking error

- **Tracking error** arises purely from integer rounding of contracts.  A positive cumulative TE means discrete contracts outperformed the fractional path on net; negative means rounding cost performance.
- **Annualised TE vol** = `std(daily_TE_USD) × √252`.  Divide by annualised USD PnL vol to see rounding error as a fraction of strategy risk.
- **Margin breach days** indicate the account size is too small for the implied position size.  Fix by increasing `account_capital` or reducing `max_position_pct` in the main config.

### Code entrypoints

- `portfolio_research/futures_sim.py` — `run_futures_sim()` (main function), `_simulate_ticker_bars()`, `_emit_discrete_tearsheet()`, `_emit_tracking_error_summary()`
- `portfolio_research/config.py` — `FuturesSimConfig`, `FuturesInstrumentSpec`, `LeverageMode`
- `portfolio_research/pipelines/portfolio_test.py` — hook in `_evaluate_phase()` after `combined_positions` is clipped, before standard return calculation

---

## Key Design Decisions

- **Vector output from Ensemble:** one row per (sample, model) → WeightLayer applies weights externally
- **FDM/IDM separation:** FDM at model-level (WeightLayer), IDM at instrument-level (Portfolio) — distinct diversification benefits
- **Immutable dataclasses:** `ContractSpec` and `Position` are frozen for safety
- **Control files:** single JSON holds both config and fitted state; `is_fit` flag gates prediction

---

**See also:** [weight layer](weight_layer.md), [base model](base_model.md), [vault](../Vault/vault.md), [Cache architecture](../Cache/architecture.md), [Cache user guide](../Cache/user_guide.md) (portfolio workflow; portfolio-addition summary cross-linked there)

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
