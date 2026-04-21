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
F_i = (τ / (σ × √h_i)) × X_i
```
- `τ` = target volatility (e.g. 0.20)
- `σ` = instrument blended volatility
- `h_i` = exposure fraction (`1 / n_bins`)
- `X_i` = binary signal (0 or 1)

### WeightLayer — Forecast Diversification Multiplier
```
FDM = min(√(1 / (mean_corr + 0.01)), 2.0)
```
Applied after combining per-model forecasts into a single `forecast_score`.

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

All three use the same functional form: `√(1 / (mean_corr + ε))`. The higher cap on IDM reflects that instrument-level diversification can be greater than forecast-level diversification.

---

## File Structure

```
ensemble/
├── diversified_ensemble.py  # Owns base models; generates per-model vol-scaled forecasts
├── weight_layer.py          # Per-TF: combines forecasts; applies intra-TF FDM
├── portfolio.py             # TFPortfolio (per-TF IDM), GlobalPortfolio (top-level orchestrator)
│                            # Portfolio = TFPortfolio (backward-compatible alias)
└── __init__.py

execution/
├── position_sizer.py        # Converts position fractions → contract quantities
└── __init__.py
```

---

## Control File Format (JSON)

Ensembles are configured and persisted via JSON control files. The `is_fit` flag tracks training state.

```json
{
  "metadata": { "is_fit": true, "base_tf": "D" },
  "base_models": [
    {
      "name": "rsi_signal_D_lookback_14_long",
      "model_type": "QuantileBinningModel",
      "feature_column": "rsi_signal_D_lookback_14",
      "strategy": "long",
      "constructor_params": { "n_bins": 10 }
    }
  ],
  "fitted_base_models": { "...": "..." },
  "fitted_ensemble": {
    "weights": {},
    "model_exposure_fractions": {},
    "feature_names": []
  }
}
```

Control files live in [[vault]]. `is_fit: false` → ensemble must be trained before prediction.

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

## Portfolio research — inclusion gates

Research-phase workflow for deciding whether to **add a new vault ensemble (candidate)** to an existing portfolio configuration. It reuses the same **train / validation / test** windows as `portfolio_research.config.load_config()` and the same global portfolio scoring path as `run_single_phase_for_prop_firm` (validation = fit on train, score on validation; test = fit on train+validation, score on test).

**Principle:** Gate on **validation data only**; treat the **test** window as a one-shot sanity check after you already accept the candidate on validation.

| Step | What it checks | Default rule (tunable in config) |
|------|----------------|----------------------------------|
| **Gate 1 — diversification** | For each **same-`base_tf` peer** ensemble: **Pearson** and **Spearman** correlation of **ensemble-level** validation forecast streams (mean over intersected tickers per peer). | Pass only if **every** peer has finite values with Pearson `< corr_max` and Spearman `< spearman_corr_max` (defaults **0.7**). No same-TF peers → vacuous pass. |
| **Gate 2 — standalone performance** | Candidate **alone** in `ensemble_dirs`; validation **Sharpe / Sortino / Calmar** on combined returns. | Pass if Sharpe > `sharpe_min` (default **0.3**). |
| **Standalone vs each baseline** | One row per baseline ensemble + candidate: Sharpe / Sortino / Calmar on **train**, **validation**, and **train+validation** (single-ensemble portfolio each). | Report only (see CSV). |
| **Uplift** | Full portfolio **without** vs **with** candidate on **train**, **validation**, and **concatenated train+validation**; Sharpe / Sortino / Calmar each window. | Pass if **each** window satisfies `Sharpe_with > Sharpe_without - uplift_slack` (default slack **0.05**). |
| **Optional test confirmation** | Opt-in: standalone candidate **test** Sharpe vs **validation** Sharpe ratio. | Pass if `Sharpe_test / Sharpe_val > test_sharpe_ratio_min` (default **0.5**). |

> **Correlation estimators:** Gate 1 uses **Pearson** and **Spearman** on aligned **validation** `forecast_score` series per ticker. The live **WeightLayer** FDM uses **Ledoit–Wolf** on **standardized in-sample** pivots at fit time — same economic object (forecasts), different estimator and window. See [[weight_layer]] for production FDM.

### Configuration

- **`PortfolioInclusionConfig`** in `feature_research/config.py`: thresholds (`corr_max`, `spearman_corr_max`, `sharpe_min`, `uplift_slack`, `test_sharpe_ratio_min`), `output_subdir` (default `inclusion`), optional default candidate path/key.
- **`ResearchConfig.portfolio_inclusion`**: holds defaults; `feature_research.config.load_config()` returns a default `PortfolioInclusionConfig()`.
- **Baseline portfolio** (tickers, train/validation/test windows, `ensemble_dirs`, weight layer, etc.) still comes from **`portfolio_research.config.load_config()`** — the CLI loads both configs.

### CLI

From the repo root (venv Python), pass a **repo-relative** path to the candidate ensemble directory (same style as `ensemble_dirs` values):

```powershell
.\.venv\Scripts\python.exe -m feature_research.run_inclusion_gates
```

With ``portfolio_inclusion.candidate_repo_relative_path`` set in ``feature_research.config.load_config()``, no CLI arguments are required. Optional overrides: ``--candidate-path``, ``--candidate-key``, ``--emit-tearsheets`` / ``--no-emit-tearsheets``, ``--no-preflight``.

### Artifacts

Under `output_root` / `portfolio_inclusion.output_subdir`: `inclusion_<candidate_key>_summary.csv`, `_corr_by_peer_and_ticker.csv` (Pearson + Spearman per peer–ticker), `_standalone_by_ensemble.csv`, `_uplift_by_window.csv`.

### Code entrypoints

- `feature_research/inclusion_gates.py` — `run_inclusion_decision`, `pearson_corr_candidate_vs_each_peer`, `write_inclusion_reports`, …
- `feature_research/run_inclusion_gates.py` — CLI

**See also:** [[Cache/user_guide]] (portfolio workflow and preflight), [[Vault/user_guide]] (ensemble layout).

---

## Key Design Decisions

- **Vector output from Ensemble:** one row per (sample, model) → WeightLayer applies weights externally
- **FDM/IDM separation:** FDM at model-level (WeightLayer), IDM at instrument-level (Portfolio) — distinct diversification benefits
- **Immutable dataclasses:** `ContractSpec` and `Position` are frozen for safety
- **Control files:** single JSON holds both config and fitted state; `is_fit` flag gates prediction

---

**See also:** [[weight_layer]], [[base_model]], [[vault]], [[Cache/architecture]], [[Cache/user_guide]] (portfolio workflow; inclusion gates summary cross-linked there)
