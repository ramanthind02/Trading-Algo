# Portfolio & Forecast Pipeline Architecture

> **Scope:** Full pipeline from BaseModels to tradeable contracts (Robert Carver methodology).

---

## Pipeline

```
BaseModels → DiversifiedEnsemble → WeightLayer → Portfolio → PositionSizer
  (0/1)        (per-model           (combined       (position    (contract
  signals)      forecasts)           forecast        fractions)   quantities)
                                     + FDM)          + IDM)
```

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
| FDM | WeightLayer | **2.0** | Forecast value correlations (model-level) |
| IDM | Portfolio | **2.5** | Instrument return correlations (instrument-level) |

Both use the same functional form: `√(1 / (mean_corr + ε))` — higher cap on IDM reflects that instrument-level diversification can be greater.

---

## File Structure

```
ensemble/
├── diversified_ensemble.py  # Owns base models; generates per-model vol-scaled forecasts
├── weight_layer.py          # Combines forecasts; applies FDM
├── portfolio.py             # Applies instrument weights + IDM
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

## Usage Example

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

# 3. Portfolio: instrument weights + IDM
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

---

## Key Design Decisions

- **Vector output from Ensemble:** one row per (sample, model) → WeightLayer applies weights externally
- **FDM/IDM separation:** FDM at model-level (WeightLayer), IDM at instrument-level (Portfolio) — distinct diversification benefits
- **Immutable dataclasses:** `ContractSpec` and `Position` are frozen for safety
- **Control files:** single JSON holds both config and fitted state; `is_fit` flag gates prediction

---

**See also:** [[weight_layer]], [[base_model]], [[vault]]
