# Forecast Pipeline Architecture

## Overview

The forecast pipeline converts trading signals into position sizes using Robert Carver's systematic trading framework. The architecture separates concerns into distinct layers:

```
Base Models → Ensemble → WeightLayer → Portfolio → PositionSizer → Contracts
```

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Base Models │────▶│  Ensemble   │────▶│ WeightLayer │────▶│  Portfolio  │────▶│PositionSizer│
│             │     │             │     │             │     │             │     │             │
│ Binary      │     │ Per-model   │     │ Combined    │     │ Position    │     │ Contract    │
│ signals     │     │ forecasts   │     │ forecast    │     │ fractions   │     │ quantities  │
│ (0/1)       │     │ (DataFrame) │     │ + FDM       │     │ + IDM       │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Key Classes

### 1. DiversifiedEnsemble (`ensemble/diversified_ensemble.py`)

**Purpose**: Owns base models and generates per-model volatility-scaled forecasts.

**Input**: Raw feature data, ticker, volatility
**Output**: `DataFrame['ticker', 'model_name', 'forecast', 'signal']`

**Key Formula**:
```
F_i = (τ / (σ × √h_i)) × X_i
```
- `τ` = target volatility (e.g., 0.20)
- `σ` = instrument's blended volatility
- `h_i` = exposure fraction (1/n_bins)
- `X_i` = binary signal (0 or 1)

**Key Attributes**:
- `base_models`: Dict of base model instances
- `model_exposure_fractions_`: h_i values per model
- `weights_`: Diversification weights (for legacy use)

---

### 2. WeightLayer (`ensemble/weight_layer.py`)

**Purpose**: Combines forecasts from multiple models using inverse correlation weights and applies FDM.

**Input**: List of forecast DataFrames from Ensemble(s)
**Output**: `DataFrame['ticker', 'forecast_score']` (FDM-scaled)

**Key Formula**:
```
FDM = min(√(1 / (mean_corr + 0.01)), 2.0)
```

**Components**:
- `InverseCorrelationWeighter`: Calculates weights where less-correlated models get higher weight
  ```
  d_i = 1 / (1 + avg_abs_corr_i)
  w_i = d_i / Σd_j
  ```

**Key Attributes**:
- `weights_`: Model weights (sum to 1.0)
- `fdm_`: Forecast Diversification Multiplier
- `mean_forecast_correlation_`: For diagnostics

---

### 3. Portfolio (`ensemble/portfolio.py`)

**Purpose**: Applies instrument weighting and IDM to combined forecasts.

**Input**: Combined forecasts from WeightLayer
**Output**: `DataFrame['ticker', 'forecast_score', 'position_fraction']`

**Key Formula**:
```
IDM = min(√(1 / (mean_corr + 0.01)), 2.5)
position = forecast × instrument_weight × IDM
```

**Key Attributes**:
- `idm_`: Instrument Diversification Multiplier
- `instrument_weights`: Custom or equal weights per instrument
- `max_position_pct`: Optional position cap

---

### 4. PositionSizer (`execution/position_sizer.py`)

**Purpose**: Converts position fractions to tradeable contract quantities.

**Input**: Position fractions from Portfolio
**Output**: `DataFrame` with contracts, notional values

**Key Formula**:
```
contracts = (position_fraction × capital) / (price × multiplier × fx_rate)
```

**Components**:
- `ContractSpec`: Dataclass holding price, multiplier, fx_rate per instrument
- `RoundingMethod`: ROUND, FLOOR, or CEILING

**Key Attributes**:
- `capital`: Account size
- `contract_specs`: Dict of ContractSpec per ticker

---

## Multiplier Summary

| Multiplier | Source | Cap | Purpose |
|------------|--------|-----|---------|
| FDM | WeightLayer | 2.0 | Diversification benefit from combining models |
| IDM | Portfolio | 2.5 | Diversification benefit from multiple instruments |

Both use the same formula: `√(1 / (mean_corr + ε))` but on different correlations:
- FDM: forecast value correlations
- IDM: instrument return correlations

---

## File Structure

```
ensemble/
├── diversified_ensemble.py  # Owns base models, generates per-model forecasts
├── weight_layer.py          # Combines forecasts with FDM
├── portfolio.py             # Applies IDM and instrument weights
└── __init__.py

execution/
├── position_sizer.py        # Converts to contracts
└── __init__.py
```

---

## Usage Example

```python
from ensemble import DiversifiedEnsemble, WeightLayer, Portfolio
from execution import PositionSizer, ContractSpec

# 1. Ensemble generates per-model forecasts
ensemble = DiversifiedEnsemble(control_file_path='config.json')
ensemble.fit(X, ticker, volatility, y)
forecasts = ensemble.predict(X, ticker, volatility)
# Returns: DataFrame['ticker', 'model_name', 'forecast', 'signal']

# 2. WeightLayer combines with FDM
weight_layer = WeightLayer(fdm_max=2.0)
weight_layer.fit([forecasts], signals_df)
combined = weight_layer.combine([forecasts])
# Returns: DataFrame['ticker', 'forecast_score']

# 3. Portfolio applies IDM
portfolio = Portfolio(max_position_pct=2.0, idm_max=2.5)
portfolio.fit(instrument_returns)
positions = portfolio.predict(combined)
# Returns: DataFrame['ticker', 'forecast_score', 'position_fraction']

# 4. PositionSizer converts to contracts
specs = {'ES': ContractSpec(ticker='ES', price=4800, multiplier=50)}
sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
contracts = sizer.calculate_positions(positions)
# Returns: DataFrame with contracts, notional_value, notional_pct
```

---

## Control File Format

Ensembles are configured via JSON control files:

```json
{
  "metadata": {
    "is_fit": true,
    "base_tf": "D"
  },
  "base_models": [
    {
      "name": "rsi_signal_D_lookback_14_long",
      "model_type": "QuantileBinningModel",
      "feature_column": "rsi_signal_D_lookback_14",
      "strategy": "long",
      "constructor_params": {"n_bins": 10}
    }
  ],
  "fitted_base_models": {...},
  "fitted_ensemble": {
    "weights": {...},
    "model_exposure_fractions": {...},
    "feature_names": [...]
  }
}
```

---

## Key Design Decisions

1. **Vector output from Ensemble**: Returns one row per (sample, model) instead of scalar. This allows WeightLayer to apply weights externally.

2. **Separation of FDM/IDM**: FDM lives in WeightLayer (model-level), IDM lives in Portfolio (instrument-level).

3. **Immutable dataclasses**: `ContractSpec` and `Position` are frozen for safety.

4. **Control files**: Single JSON file contains both config and fitted state, controlled by `is_fit` flag.
