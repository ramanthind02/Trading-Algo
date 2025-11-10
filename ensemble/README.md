# TradingEnsemble Class

A trading-specific ensemble class that combines multiple binary strategy signals using correlation coefficients as weights and applies risk-adjusted position sizing based on volatility and exposure fractions.

## Overview

The `TradingEnsemble` class is designed for combining trading strategies where:
- Features are binary (0 or 1) indicating whether each strategy is in a trade
- Target is continuous log returns 
- Weights are correlation coefficients normalized to sum to 1
- Final prediction includes risk scaling and exposure adjustment using the formula: `r/v * Σ(w_i * X_i / √h_i)`

## Key Parameters

- `r`: Target annual portfolio risk (default: 0.15 = 15%)
- `w_i`: Weight of each feature (correlation coefficient)
- `h_i`: Exposure fraction (fraction of time each strategy is in market)
- `v`: Instrument's annualized volatility (passed as 'annualized_volatility' column)

## Installation

The class is part of the `ensemble` module:

```python
from ensemble import TradingEnsemble
```

## Quick Start

```python
import pandas as pd
import numpy as np
from ensemble import TradingEnsemble

# Create sample data
X = pd.DataFrame({
    'strategy_1': [0, 1, 0, 1, 1],  # Binary strategy signals
    'strategy_2': [1, 0, 1, 0, 1],
    'strategy_3': [0, 0, 1, 1, 0],
    'annualized_volatility': [0.15, 0.18, 0.12, 0.20, 0.16]  # Required!
})
y = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])  # Log returns

# Fit the ensemble
ensemble = TradingEnsemble(r=0.20)  # 20% target risk
ensemble.fit(X, y)

# Make predictions
predictions = ensemble.predict(X)
print(f"Position sizes: {predictions}")

# Get feature importance
print(f"Weights: {ensemble.get_feature_importance()}")
```

## Advanced Usage

### Save and Load Configuration

```python
# Fit and save configuration
ensemble = TradingEnsemble(r=0.15)
ensemble.fit(X_train, y_train)
ensemble.save_config("my_ensemble_config.json")

# Load in production (no need to refit)
production_ensemble = TradingEnsemble(config_path="my_ensemble_config.json")
predictions = production_ensemble.predict(X_new)
```

### Get Detailed Statistics

```python
# Feature importance (weights)
importance = ensemble.get_feature_importance()

# Exposure statistics
exposure_stats = ensemble.get_exposure_stats()
for feature, stats in exposure_stats.items():
    print(f"{feature}: weight={stats['weight']:.3f}, exposure={stats['exposure_fraction']:.3f}")
```

## Input Requirements

### Feature Matrix (X)
- **Type**: `pandas.DataFrame`
- **Required column**: `'annualized_volatility'` (continuous, positive values)
- **Feature columns**: All other columns must be binary (0 or 1) for fit()
- **Shape**: (n_samples, n_features + 1)

### Target (y)  
- **Type**: `pandas.Series` or `numpy.ndarray`
- **Values**: Continuous (typically log returns)
- **Shape**: (n_samples,)

## Output

The `predict()` method returns position sizes calculated as:
```
position_size = (r / volatility) * Σ(w_i * signal_i / √h_i)
```

Where:
- `r`: Target annual risk
- `volatility`: Annualized volatility for each sample
- `w_i`: Weight of feature i (normalized correlation coefficient)
- `signal_i`: Binary signal from feature i (0 or 1)
- `h_i`: Exposure fraction of feature i

## Configuration File Format

The ensemble saves/loads configurations in JSON format:

```json
{
  "metadata": {
    "created_at": "2025-11-09T01:10:00",
    "model_type": "TradingEnsemble",
    "num_features": 3,
    "target_risk": 0.15
  },
  "weights": {
    "strategy_1": 0.4,
    "strategy_2": 0.35,
    "strategy_3": 0.25
  },
  "exposure_fractions": {
    "strategy_1": 0.3,
    "strategy_2": 0.4,
    "strategy_3": 0.25
  },
  "feature_names": ["strategy_1", "strategy_2", "strategy_3"],
  "parameters": {
    "r": 0.15
  }
}
```

## Error Handling

The class provides comprehensive validation:

- **Missing volatility column**: Raises `ValueError` if 'annualized_volatility' not found
- **Non-binary features**: Raises `ValueError` if features contain values other than 0/1 during fit
- **Invalid volatility**: Raises `ValueError` for non-positive volatility values
- **Unfitted model**: Raises `ValueError` when calling predict() before fit()
- **Missing features**: Raises `ValueError` if predict() data missing required features

## Examples

### Example 1: Basic Trading Ensemble

```python
from ensemble import TradingEnsemble
import pandas as pd
import numpy as np

# Sample trading data
np.random.seed(42)
n_samples = 1000

data = {
    'momentum_signal': np.random.binomial(1, 0.3, n_samples),
    'mean_reversion_signal': np.random.binomial(1, 0.2, n_samples), 
    'breakout_signal': np.random.binomial(1, 0.25, n_samples),
    'annualized_volatility': np.random.lognormal(np.log(0.15), 0.2, n_samples).clip(0.05, 0.4)
}

X = pd.DataFrame(data)
# Create correlated returns
y = (0.01 * X['momentum_signal'] + 
     0.008 * X['mean_reversion_signal'] +
     0.012 * X['breakout_signal'] +
     np.random.normal(0, 0.02, n_samples))

# Fit ensemble with 18% target risk
ensemble = TradingEnsemble(r=0.18)
ensemble.fit(X, y)

print(f"Fitted ensemble: {ensemble}")
print(f"Feature weights: {ensemble.get_feature_importance()}")

# Generate position sizes for new data
new_signals = pd.DataFrame({
    'momentum_signal': [1, 0, 1],
    'mean_reversion_signal': [0, 1, 1], 
    'breakout_signal': [1, 0, 0],
    'annualized_volatility': [0.15, 0.20, 0.12]
})

positions = ensemble.predict(new_signals)
print(f"Position sizes: {positions}")
```

### Example 2: Production Workflow

```python
# Training phase - fit and save
training_ensemble = TradingEnsemble(r=0.15, save_path="production_ensemble.json")
training_ensemble.fit(X_train, y_train)
config_path = training_ensemble.save_config()
print(f"Model saved to: {config_path}")

# Production phase - load and predict
production_ensemble = TradingEnsemble(config_path=config_path)
daily_positions = production_ensemble.predict(live_data)
```

## Mathematical Details

### Weight Calculation
1. Calculate correlation coefficient between each feature and target: `corr_i = corr(X_i, y)`
2. Take absolute values: `abs_corr_i = |corr_i|`
3. Normalize to sum to 1: `w_i = abs_corr_i / Σabs_corr_j`

### Exposure Calculation  
For each feature i: `h_i = (number of 1s) / (total samples)`

### Position Sizing Formula
```
position_size = (r / v) * Σ(w_i * X_i / √h_i)
```

This formula:
- Scales by target risk `r` and inverse volatility `1/v` for risk management
- Weights each signal by its correlation strength `w_i`
- Adjusts for time-in-market exposure using `1/√h_i` (strategies with lower exposure get higher weight when active)

## Integration with Existing Codebase

The `TradingEnsemble` follows the same sklearn-style interface as other models in the `feature_selection` module:
- `fit(X, y)` for training
- `predict(X)` for inference  
- Comprehensive input validation
- Configuration save/load functionality
- Detailed documentation and error messages

This makes it compatible with existing workflows and easy to integrate into the trading infrastructure.
