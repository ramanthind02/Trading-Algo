# Portfolio & Forecast Pipeline Architecture

> **Scope:** Full pipeline from BaseModels to tradeable contracts (Robert Carver methodology).

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
from utils.enums import TimeFrame

# 1. Build one TFPortfolio per timeframe
tf_daily = TFPortfolio(trading_timeframe=TimeFrame.D, ...)
tf_weekly = TFPortfolio(trading_timeframe=TimeFrame.W, ...)

# 2. Wrap in GlobalPortfolio with WeightLayer (adapter-driven global combine)
global_p = GlobalPortfolio(
    tf_portfolios=[tf_daily, tf_weekly],
    weight_layer=WeightLayer(weight_method="hrp_classic", fdm_max=2.0),
)

# 3. Fit: supply candles and instrument returns per timeframe
global_p.fit(
    candles_per_tf={TimeFrame.D: daily_candles, TimeFrame.W: weekly_candles},
    instrument_returns=returns_df,
)

# 4. Predict: returns the standard output schema
positions = global_p.predict(
    candles_per_tf={TimeFrame.D: daily_candles, TimeFrame.W: weekly_candles}
)
# → DataFrame["ticker", "datetime", "forecast_score", "position_fraction"]
```

---

## Key Design Decisions

- **Vector output from Ensemble:** one row per (sample, model) → WeightLayer applies weights externally
- **FDM/IDM separation:** FDM at model-level (WeightLayer), IDM at instrument-level (Portfolio) — distinct diversification benefits
- **Immutable dataclasses:** `ContractSpec` and `Position` are frozen for safety
- **Control files:** single JSON holds both config and fitted state; `is_fit` flag gates prediction

---

**See also:** [[weight_layer]], [[base_model]], [[vault]]
