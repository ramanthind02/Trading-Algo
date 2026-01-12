# Implementation Specifications

> **🔨 This Document**: Concrete build tasks, method signatures, unit tests, and step-by-step instructions  
> **📖 For Context**: See [forecast_specs.md](./forecast_specs.md) for design rationale, formulas, and examples

**Version**: 3.0.0  
**Date**: 2025-01-09  
**Status**: Ready for Implementation

**Purpose**: This document contains the concrete, actionable implementation tasks for building the forecast generation and position sizing system.

**Major Changes (v3.0.0)**:
- **FDM & IDM SEPARATION**: Split single DM into FDM (Forecast Diversification Multiplier) at Weight layer and IDM (Instrument Diversification Multiplier) at Portfolio layer
- **FDM**: Calculated from forecast value correlations, applied at Weight layer after forecast combination
- **IDM**: Calculated from instrument return correlations, applied at Portfolio layer after instrument weighting
- Follows Carver's proven framework with separate multipliers for forecast-level and instrument-level diversification

**Major Changes (v2.0.0)**:
- **MAJOR RESTRUCTURE**: Ensembles now perform risk management per base model (volatility scaling)
- **NEW**: Weight Layer for combining all base model forecasts using weight vector
- **SIMPLIFIED**: Portfolio layer becomes minimal (mostly pass-through)
- This solves signal dilution when ensembles have different numbers of base models

---

## Table of Contents

1. [Phase 1: Refactor Ensemble (Risk Management)](#phase-1-refactor-ensemble-risk-management)
2. [Phase 2: Create Weight Layer](#phase-2-create-weight-layer)
3. [Phase 3: Simplify Portfolio](#phase-3-simplify-portfolio)
4. [Phase 4: Create Volatility Utilities](#phase-4-create-volatility-utilities)
5. [Phase 5: Create Execution Layer](#phase-5-create-execution-layer)
6. [Phase 6: Integration & Testing](#phase-6-integration--testing)

---

## Phase 1: Refactor Ensemble (Risk Management)

**Goal**: Refactor ensemble to perform risk management per base model. Calculate volatility-adjusted forecast for each base model (assuming signal=1), output vector of forecasts (one per base model) instead of combined forecast.

**Estimated Effort**: 4-6 hours

**⚠️ BREAKING CHANGE**: This phase changes the `predict()` return structure to a vector (one row per base model) instead of a single combined forecast. All calling code must be updated.

### Files to Modify

- `ensemble/diversified_ensemble.py`
- `tests/test_diversified_ensemble.py`

### Current State Analysis

**What to Remove**:
- ❌ Signal combination logic (moves to Weight layer)
- ❌ Inverse correlation weight calculation (moves to Weight layer)
- ❌ Combined forecast calculation

**What to Keep**:
- ✅ Base model signal generation
- ✅ Exposure fraction tracking per model
- ✅ Model name tracking

**What to Add**:
- ✅ `volatility` parameter in `predict()`
- ✅ `target_volatility` parameter
- ✅ Volatility scaling per base model
- ✅ Vector output (one row per base model)
- ✅ Model name in output DataFrame

### Method Signature Changes

#### Before:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: pd.Series,
    normalization_data: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """Returns: Array of combined forecasts"""
```

#### After:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: Union[pd.Series, Dict[str, float]],  # Blended volatility per instrument
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'model_name', 'forecast', 'signal'])
    
    Where:
    - ticker: Instrument identifier
    - model_name: Base model identifier
    - forecast: Volatility-adjusted forecast (0 if signal inactive)
      This already incorporates exposure adjustment via sqrt(h_i)
    - signal: Binary signal {0, 1}
    
    Note: One row per base model (vector output, not combined)
    Note: Exposure fraction h_i is used internally but not included in output
    """
```

### Implementation Steps

1. **Add volatility and target_volatility parameters**:
   ```python
   def __init__(
       self,
       target_volatility: float = 0.20,  # NEW
       # ... other existing parameters
   ):
       self.target_volatility = target_volatility
   ```

2. **Calculate volatility-adjusted forecast per base model**:
   ```python
   def predict(self, X, ticker, volatility, normalization_data=None):
       # Get binary signals from all base models
       binary_signals = self._get_binary_signals(X, ticker, normalization_data)
       
       # Convert volatility to dict if needed
       vol_dict = volatility if isinstance(volatility, dict) else volatility.to_dict()
       
       # For each sample and each base model, calculate forecast
       results = []
       for idx, (tick, row_signals) in enumerate(zip(ticker, binary_signals.iterrows())):
           ticker_val = tick if isinstance(tick, str) else tick
           instrument_vol = vol_dict.get(ticker_val)
           
           if instrument_vol is None:
               raise ValueError(f"Missing volatility for ticker: {ticker_val}")
           
           # For each base model
           for model_name, signal in row_signals[1].items():
               base_model = self.base_models[model_name]
               h_i = self.model_exposure_fractions_[model_name]
               
               # Calculate volatility-adjusted forecast (assuming signal=1)
               forecast_if_active = (
                   self.target_volatility / 
                   (instrument_vol * np.sqrt(h_i))
               )
               
               # Apply signal: 0 if inactive, forecast_if_active if active
               forecast = forecast_if_active if signal == 1 else 0.0
               
               results.append({
                   'ticker': ticker_val,
                   'model_name': model_name,
                   'forecast': forecast,
                   'signal': int(signal)
               })
       
       return pd.DataFrame(results)
   ```

3. **Store model exposure fractions during fit**:
   ```python
   # In fit(), store exposure fraction for each base model
   # Note: base_models is a dict[str, BaseModel] in current implementation
   self.model_exposure_fractions_ = {
       model_name: 1.0 / base_model.n_bins 
       for model_name, base_model in self.base_models.items()
   }
   ```

4. **Remove signal combination logic**:
   - Remove inverse correlation weight calculation (moves to Weight layer)
   - Remove weighted sum of signals
   - Keep only per-model forecast calculation

### Unit Tests

Create tests in `tests/test_diversified_ensemble.py`:

```python
def test_vector_output_structure():
    """Test that output is vector (one row per base model)"""
    ensemble = DiversifiedEnsemble(target_volatility=0.20, models=[...])
    ensemble.fit(X_train, y_train, ticker_train, volatility_train)
    
    volatility = pd.Series({'TEST': 0.20})
    predictions = ensemble.predict(X_test, ticker_test, volatility)
    
    # Check it's a DataFrame
    assert isinstance(predictions, pd.DataFrame)
    
    # Check required columns
    assert 'ticker' in predictions.columns
    assert 'model_name' in predictions.columns
    assert 'forecast' in predictions.columns
    assert 'signal' in predictions.columns
    
    # Check vector structure: should have one row per base model per sample
    n_samples = len(X_test)
    n_models = len(ensemble.base_models)
    assert len(predictions) == n_samples * n_models

def test_volatility_scaling_per_model():
    """Test that volatility scaling is applied per base model"""
    ensemble = DiversifiedEnsemble(target_volatility=0.20, models=[...])
    ensemble.fit(X_train, y_train, ticker_train, volatility_train)
    
    # Test with matching volatility (should give forecast = 1.0 / sqrt(h))
    volatility = pd.Series({'TEST': 0.20})
    predictions = ensemble.predict(X_test, ticker_test, volatility)
    
    # For a model with h=0.1, forecast_if_active = 0.20 / (0.20 * sqrt(0.1)) ≈ 3.16
    model_forecasts = predictions[predictions['model_name'] == 'model_0']
    active_forecasts = model_forecasts[model_forecasts['signal'] == 1]['forecast']
    
    if len(active_forecasts) > 0:
        expected = 0.20 / (0.20 * np.sqrt(0.1))
        assert np.isclose(active_forecasts.iloc[0], expected, rtol=0.01)

def test_inactive_signals_zero_forecast():
    """Test that inactive signals have forecast = 0"""
    ensemble = DiversifiedEnsemble(target_volatility=0.20, models=[...])
    ensemble.fit(X_train, y_train, ticker_train, volatility_train)
    
    volatility = pd.Series({'TEST': 0.20})
    predictions = ensemble.predict(X_test, ticker_test, volatility)
    
    # All inactive signals should have forecast = 0
    inactive = predictions[predictions['signal'] == 0]
    assert np.allclose(inactive['forecast'].values, 0.0)

def test_exposure_fraction_per_model():
    """Test that exposure fraction is stored per model"""
    ensemble = DiversifiedEnsemble(target_volatility=0.20, models=[...])
    ensemble.fit(X_train, y_train, ticker_train, volatility_train)
    
    # Check that model_exposure_fractions_ is populated
    assert hasattr(ensemble, 'model_exposure_fractions_')
    assert len(ensemble.model_exposure_fractions_) == len(ensemble.base_models)
    
    # Check that exposure fractions match n_bins
    for model_name, h_i in ensemble.model_exposure_fractions_.items():
        n_bins = ensemble.base_models[model_name].n_bins
        expected_h = 1.0 / n_bins
        assert np.isclose(h_i, expected_h)
```

### Migration Notes

**Breaking Changes**:
1. `predict()` now returns `pd.DataFrame` instead of `np.ndarray`
2. Output structure changed from single array to DataFrame with 3 columns
3. `volatility` parameter removed from `predict()`

**Required Updates to Calling Code**:
```python
# OLD (before Phase 1):
predictions = ensemble.predict(X, ticker, volatility)
# predictions is np.ndarray

# NEW (after Phase 1):
predictions = ensemble.predict(X, ticker, volatility)
# predictions is pd.DataFrame with columns: ['ticker', 'model_name', 'forecast', 'signal']

# Access values:
forecasts = predictions['forecast'].values
tickers = predictions['ticker'].values
model_names = predictions['model_name'].values
```

---

## Phase 2: Create Weight Layer

**Goal**: Create a new Weight layer that combines forecasts from all base models across all ensembles using a weight vector (inverse correlation method) and applies FDM (Forecast Diversification Multiplier).

**Estimated Effort**: 5-7 hours

### Files to Create

- `ensemble/weight_layer.py`
- `ensemble/inverse_correlation_weighter.py`
- `tests/test_weight_layer.py`

### WeightLayer Class

```python
from typing import List, Protocol
import pandas as pd
import numpy as np

class Weighter(Protocol):
    """Protocol for weight calculation methods"""
    def fit(self, signals: pd.DataFrame) -> None: ...
    def get_weights(self) -> pd.Series: ...

class WeightLayer:
    """Combine forecasts from all base models using weight vector and apply FDM"""
    
    def __init__(
        self, 
        weight_method: str = 'inverse_correlation',
        fdm_max: float = 2.0
    ):
        """
        Parameters
        ----------
        weight_method : str, default='inverse_correlation'
            Method for calculating weights. Options:
            - 'inverse_correlation': Use inverse correlation weights
            - 'linear': Use linear model (future)
            - 'ml': Use ML model (future)
        fdm_max : float, default=2.0
            Maximum FDM value (Carver's recommendation)
        """
        self.weight_method = weight_method
        self.fdm_max = fdm_max
        if weight_method == 'inverse_correlation':
            self.weighter = InverseCorrelationWeighter()
        else:
            raise ValueError(f"Unknown weight method: {weight_method}")
        self.fdm_ = None  # Will be calculated during fit()
    
    def fit(
        self,
        forecast_vectors: List[pd.DataFrame],
        signals: pd.DataFrame
    ) -> 'WeightLayer':
        """
        Fit weights and FDM from training data.
        
        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles (training data)
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
            Used for both weight calculation (via signals) and FDM calculation (via forecast values)
        signals : pd.DataFrame
            Binary signals from all base models (for inverse correlation weight calculation)
            Columns: model names, rows: samples
        
        Returns
        -------
        self
        """
        # Fit the weighter (uses signals for correlation)
        self.weighter.fit(signals)
        
        # Calculate FDM from forecast value correlations
        self.fdm_ = self._calculate_fdm(forecast_vectors)
        
        return self
    
    def _calculate_fdm(self, forecast_vectors: List[pd.DataFrame]) -> float:
        """
        Calculate Forecast Diversification Multiplier from forecast value correlations.
        
        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles (training data)
        
        Returns
        -------
        float
            FDM value (capped at fdm_max)
        """
        # Concatenate all forecasts
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        # Pivot: columns = model_name, rows = (ticker, sample), values = forecast
        # Need to create a unique index for each sample
        forecast_matrix = all_forecasts.pivot_table(
            index=['ticker'],  # Group by ticker for now
            columns='model_name',
            values='forecast',
            aggfunc='mean'  # If multiple samples per ticker, take mean
        )
        
        # Calculate correlation matrix of forecast values
        corr_matrix = forecast_matrix.corr().abs()
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack()
        
        # Calculate mean correlation
        mean_corr = correlations.mean() if len(correlations) > 0 else 0.5
        
        # Floor negative correlations at zero (Carver's recommendation)
        mean_corr = max(mean_corr, 0.0)
        
        # Calculate FDM: sqrt(1 / (mean_corr + epsilon))
        epsilon = 0.01  # Small value to avoid division by zero
        fdm = np.sqrt(1.0 / (mean_corr + epsilon))
        
        # Cap at fdm_max
        fdm = min(fdm, self.fdm_max)
        
        return fdm
    
    def combine(
        self,
        forecast_vectors: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine forecasts from all base models and apply FDM.
        
        Parameters
        ----------
        forecast_vectors : list[pd.DataFrame]
            List of forecast vectors from all ensembles.
            Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
        
        Returns
        -------
        pd.DataFrame with columns: ['ticker', 'forecast_score']
            forecast_score is already FDM-scaled
        """
        if self.fdm_ is None:
            raise ValueError("WeightLayer must be fitted before calling combine()")
        
        # Concatenate all forecast vectors
        all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
        
        # Get weights
        weights = self.weighter.get_weights()
        
        # Group by ticker and calculate weighted sum
        results = []
        for ticker, group in all_forecasts.groupby('ticker'):
            # Weighted sum of forecasts
            forecast_weighted = (
                group['forecast'] * group['model_name'].map(weights)
            ).sum()
            
            # Apply FDM
            forecast_score = forecast_weighted * self.fdm_
            
            results.append({
                'ticker': ticker,
                'forecast_score': forecast_score
            })
        
        return pd.DataFrame(results)
```

### InverseCorrelationWeighter Class

```python
import pandas as pd
import numpy as np

class InverseCorrelationWeighter:
    """Calculate weights based on inverse correlation"""
    
    def fit(self, signals: pd.DataFrame) -> None:
        """
        Fit weights from binary signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Binary signals from all base models.
            Columns: model names, rows: samples
        """
        # Calculate correlation matrix
        corr_matrix = signals.corr().abs()
        
        # For each model, calculate average correlation with others
        avg_correlations = {}
        for model_name in signals.columns:
            other_models = [m for m in signals.columns if m != model_name]
            avg_corr = corr_matrix.loc[model_name, other_models].mean()
            avg_correlations[model_name] = avg_corr
        
        # Convert to diversification scores
        diversification_scores = {
            model: 1.0 / (1.0 + avg_corr)
            for model, avg_corr in avg_correlations.items()
        }
        
        # Normalize to sum to 1.0
        total = sum(diversification_scores.values())
        self.weights_ = pd.Series({
            model: score / total
            for model, score in diversification_scores.items()
        })
    
    def get_weights(self) -> pd.Series:
        """Get fitted weights"""
        return self.weights_
```

### Unit Tests

```python
def test_weight_layer_combines_all_models():
    """Test that Weight layer combines forecasts from all base models"""
    # Create forecast vectors from two ensembles
    ensemble1_forecasts = pd.DataFrame({
        'ticker': ['TEST', 'TEST'],
        'model_name': ['model_1', 'model_2'],
        'forecast': [1.0, 0.5],
        'signal': [1, 1]
    })
    
    ensemble2_forecasts = pd.DataFrame({
        'ticker': ['TEST'],
        'model_name': ['model_3'],
        'forecast': [0.8],
        'signal': [1]
    })
    
    # Create signals for fitting
    signals = pd.DataFrame({
        'model_1': [1, 0, 1],
        'model_2': [0, 1, 1],
        'model_3': [1, 1, 0]
    })
    
    weight_layer = WeightLayer()
    weight_layer.fit([ensemble1_forecasts, ensemble2_forecasts], signals)
    
    combined = weight_layer.combine([ensemble1_forecasts, ensemble2_forecasts])
    
    # Should have one row per unique ticker
    assert len(combined) == 1
    assert 'forecast_score' in combined.columns

def test_inverse_correlation_weights():
    """Test that inverse correlation weights are calculated correctly"""
    # Create signals with known correlations
    signals = pd.DataFrame({
        'model_1': [1, 0, 1, 0, 1],
        'model_2': [1, 0, 1, 0, 1],  # High correlation with model_1
        'model_3': [0, 1, 0, 1, 0]   # Low correlation with others
    })
    
    weighter = InverseCorrelationWeighter()
    weighter.fit(signals)
    weights = weighter.get_weights()
    
    # Weights should sum to 1.0
    assert np.isclose(weights.sum(), 1.0)
    
    # model_3 should have higher weight (lower correlation)
    assert weights['model_3'] > weights['model_1']
    assert weights['model_3'] > weights['model_2']

def test_fdm_calculation():
    """Test that FDM is calculated from forecast value correlations"""
    # Create forecast vectors with known structure
    forecast_vectors = [
        pd.DataFrame({
            'ticker': ['TEST'] * 3,
            'model_name': ['model_1', 'model_2', 'model_3'],
            'forecast': [1.0, 0.8, 0.6],
            'signal': [1, 1, 1]
        })
    ]
    
    signals = pd.DataFrame({
        'model_1': [1, 0, 1],
        'model_2': [0, 1, 1],
        'model_3': [1, 1, 0]
    })
    
    weight_layer = WeightLayer(fdm_max=2.0)
    weight_layer.fit(forecast_vectors, signals)
    
    # FDM should be calculated and stored
    assert weight_layer.fdm_ is not None
    assert 1.0 <= weight_layer.fdm_ <= 2.0

def test_fdm_application():
    """Test that FDM is applied during combine()"""
    forecast_vectors = [
        pd.DataFrame({
            'ticker': ['TEST'],
            'model_name': ['model_1'],
            'forecast': [1.0],
            'signal': [1]
        })
    ]
    
    signals = pd.DataFrame({
        'model_1': [1, 0, 1]
    })
    
    weight_layer = WeightLayer(fdm_max=2.0)
    weight_layer.fit(forecast_vectors, signals)
    
    # Manually set FDM for testing
    weight_layer.fdm_ = 1.5
    
    combined = weight_layer.combine(forecast_vectors)
    
    # Forecast should be scaled by FDM
    # If weighted sum = 1.0, then with FDM=1.5, result should be 1.5
    assert combined['forecast_score'].iloc[0] == 1.0 * 1.5

def test_fdm_capping():
    """Test that FDM is capped at fdm_max"""
    # Create forecasts that would give very high FDM
    forecast_vectors = [
        pd.DataFrame({
            'ticker': ['TEST'] * 10,
            'model_name': [f'model_{i}' for i in range(10)],
            'forecast': np.random.rand(10),
            'signal': [1] * 10
        })
    ]
    
    signals = pd.DataFrame({
        f'model_{i}': np.random.randint(0, 2, 100)
        for i in range(10)
    })
    
    weight_layer = WeightLayer(fdm_max=2.0)
    weight_layer.fit(forecast_vectors, signals)
    
    # FDM should be capped at 2.0
    assert weight_layer.fdm_ <= 2.0
```

---

## Phase 3: Simplify Portfolio

**Goal**: Simplify Portfolio to minimal layer that handles instrument weighting, IDM (Instrument Diversification Multiplier) application, and optional position capping. Most risk management moved to Ensemble layer, signal combination and FDM moved to Weight layer.

**Estimated Effort**: 3-4 hours

**Required Imports**:
```python
from typing import Union, Dict, Optional, List
import pandas as pd
import numpy as np
```

### Files to Modify

- `ensemble/portfolio.py`
- `tests/test_portfolio.py`

### New Parameters

Add to `__init__`:

```python
def __init__(
    self,
    weight_layer: WeightLayer,  # NEW: Weight layer for combining forecasts
    trading_timeframe: TimeFrame,  # Explicit trading timeframe
    max_position_pct: Optional[float] = None,  # Optional position cap
    instrument_weights: Optional[dict[str, float]] = None,  # Optional instrument weights
    idm_max: float = 2.5  # Maximum IDM value (Carver's recommendation)
):
    """
    Create a Portfolio for a SINGLE trading timeframe.
    
    Parameters
    ----------
    weight_layer : WeightLayer
        Weight layer that combines forecasts from all ensembles
    trading_timeframe : TimeFrame
        The timeframe this portfolio trades on (e.g., TimeFrame.D for daily)
    max_position_pct : float, optional
        Maximum position size per instrument. If None, no capping.
    instrument_weights : dict[str, float], optional
        Weight for each instrument. If None, equal weight.
        Weights should sum to 1.0
    idm_max : float, default=2.5
        Maximum IDM value (capped to prevent excessive leverage)
    """
    self.weight_layer = weight_layer
    self.trading_timeframe = trading_timeframe
    self.max_position_pct = max_position_pct
    self.instrument_weights = instrument_weights
    self.idm_max = idm_max
    self.idm_ = None  # Will be calculated during fit()
```

### Method Signature Changes

#### Before:
```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    timeframe: TimeFrame,
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Returns: DataFrame(['ticker', 'timeframe', 'forecast'])"""
```

#### After:
```python
def fit(
    self,
    instrument_returns: pd.DataFrame  # Historical returns for IDM calculation
) -> 'Portfolio':
    """
    Fit IDM from historical instrument returns.
    
    Parameters
    ----------
    instrument_returns : pd.DataFrame
        Historical returns for all instruments.
        Columns: instrument tickers, rows: time periods
    
    Returns
    -------
    self
    """
    self.idm_ = self._calculate_idm(instrument_returns)
    return self

def _calculate_idm(self, instrument_returns: pd.DataFrame) -> float:
    """
    Calculate Instrument Diversification Multiplier from return correlations.
    
    Parameters
    ----------
    instrument_returns : pd.DataFrame
        Historical returns for all instruments.
        Columns: instrument tickers, rows: time periods
    
    Returns
    -------
    float
        IDM value (capped at idm_max)
    """
    # Calculate correlation matrix of instrument returns
    corr_matrix = instrument_returns.corr().abs()
    
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix.where(mask).stack()
    
    # Calculate mean correlation
    mean_corr = correlations.mean() if len(correlations) > 0 else 0.5
    
    # Floor negative correlations at zero (Carver's recommendation)
    mean_corr = max(mean_corr, 0.0)
    
    # Calculate IDM: sqrt(1 / (mean_corr + epsilon))
    epsilon = 0.01  # Small value to avoid division by zero
    idm = np.sqrt(1.0 / (mean_corr + epsilon))
    
    # Cap at idm_max
    idm = min(idm, self.idm_max)
    
    return idm

def predict(
    self,
    combined_forecasts: pd.DataFrame  # From Weight layer
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'position_fraction'])
    
    Parameters
    ----------
    combined_forecasts : pd.DataFrame
        Combined forecasts from Weight layer.
        Columns: ['ticker', 'forecast_score']
        (Already FDM-scaled)
    
    Returns
    -------
    pd.DataFrame with columns:
        - ticker: Instrument identifier
        - forecast_score: Combined forecast from Weight layer (passed through)
        - position_fraction: Position fraction after instrument weighting, IDM, and capping
    """
    if self.idm_ is None:
        raise ValueError("Portfolio must be fitted before calling predict()")
```

### Implementation Steps

#### Step 1: Apply Instrument Weight

```python
def _apply_instrument_weights(
    self,
    combined_forecasts: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply instrument weights to combined forecasts.
    
    Parameters
    ----------
    combined_forecasts : pd.DataFrame
        Combined forecasts from Weight layer.
        Columns: ['ticker', 'forecast_score']
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_score', 'position_weighted']
    """
    df = combined_forecasts.copy()
    
    # Apply instrument weights
    if self.instrument_weights is None:
        # Equal weight per unique instrument
        unique_instruments = df['ticker'].nunique()
        instrument_weight = 1.0 / unique_instruments if unique_instruments > 0 else 0.0
        df['instrument_weight'] = instrument_weight
    else:
        df['instrument_weight'] = df['ticker'].map(self.instrument_weights)
    
    # Weight the forecast
    df['position_weighted'] = df['forecast_score'] * df['instrument_weight']
    
    return df
```

#### Step 2: Apply Instrument Diversification Multiplier (IDM)

```python
def _apply_idm(
    self,
    positions: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply IDM to account for portfolio-level diversification.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Columns: ['ticker', 'forecast_score', 'position_weighted']
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_score', 'position_idm']
    """
    df = positions.copy()
    df['position_idm'] = df['position_weighted'] * self.idm_
    return df
```

#### Step 3: Apply Position Cap (Optional)

```python
def _apply_position_cap(
    self,
    positions: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply position cap if specified.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Columns: ['ticker', 'forecast_score', 'position_idm']
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_score', 'position_fraction']
    """
    df = positions.copy()
    
    if self.max_position_pct is not None:
        df['position_fraction'] = df['position_idm'].clip(upper=self.max_position_pct)
    else:
        df['position_fraction'] = df['position_idm']
    
    return df[['ticker', 'forecast_score', 'position_fraction']]
```

#### Step 3: Update predict() method

```python
def predict(
    self,
    combined_forecasts: pd.DataFrame  # From Weight layer
) -> pd.DataFrame:
    """
    Main prediction method - applies instrument weighting and optional capping.
    
    Parameters
    ----------
    combined_forecasts : pd.DataFrame
        Combined forecasts from Weight layer.
        Columns: ['ticker', 'forecast_score']
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_score', 'position_fraction']
    """
    # Validate input
    if not isinstance(combined_forecasts, pd.DataFrame):
        raise ValueError(f"combined_forecasts must be pd.DataFrame, got {type(combined_forecasts)}")
    
    required_cols = ['ticker', 'forecast_score']
    missing_cols = set(required_cols) - set(combined_forecasts.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Apply instrument weights
    weighted = self._apply_instrument_weights(combined_forecasts)
    
    # Apply IDM
    idm_scaled = self._apply_idm(weighted)
    
    # Apply position cap (if specified)
    result = self._apply_position_cap(idm_scaled)
    
    return result
```

### Unit Tests (Buy/Hold Scenarios)

Create tests in `tests/test_portfolio.py`:

```python
def test_instrument_weighting():
    """Test that instrument weights are applied correctly"""
    weight_layer = WeightLayer()
    weight_layer.fit(...)  # Fit with training data
    
    portfolio = Portfolio(
        weight_layer=weight_layer,
        trading_timeframe=TimeFrame.DAILY,
        max_position_pct=None,  # No capping
        instrument_weights=None  # Equal weight
    )
    
    # Combined forecast from Weight layer
    combined_forecasts = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0]
    })
    
    result = portfolio.predict(combined_forecasts)
    
    # With 1 instrument, equal weight = 1.0, so position = 1.0 * 1.0 = 1.0
    assert np.isclose(result['position_fraction'].iloc[0], 1.0)

def test_multiple_instruments_equal_weight():
    """Test equal weighting across multiple instruments"""
    weight_layer = WeightLayer()
    weight_layer.fit(...)
    
    portfolio = Portfolio(
        weight_layer=weight_layer,
        trading_timeframe=TimeFrame.DAILY,
        max_position_pct=None,
        instrument_weights=None  # Equal weight
    )
    
    # Two instruments with same forecast
    combined_forecasts = pd.DataFrame({
        'ticker': ['TEST1', 'TEST2'],
        'forecast_score': [1.0, 1.0]
    })
    
    result = portfolio.predict(combined_forecasts)
    
    # Each should get 0.5 weight (1/2 instruments)
    # Position = 1.0 * 0.5 = 0.5
    assert len(result) == 2
    assert np.allclose(result['position_fraction'].values, [0.5, 0.5])

def test_custom_instrument_weights():
    """Test custom instrument weights"""
    weight_layer = WeightLayer()
    weight_layer.fit(...)
    
    portfolio = Portfolio(
        weight_layer=weight_layer,
        trading_timeframe=TimeFrame.DAILY,
        max_position_pct=None,
        instrument_weights={'TEST1': 0.7, 'TEST2': 0.3}
    )
    
    combined_forecasts = pd.DataFrame({
        'ticker': ['TEST1', 'TEST2'],
        'forecast_score': [1.0, 1.0]
    })
    
    result = portfolio.predict(combined_forecasts)
    
    # TEST1: 1.0 * 0.7 = 0.7
    # TEST2: 1.0 * 0.3 = 0.3
    assert np.isclose(result[result['ticker'] == 'TEST1']['position_fraction'].iloc[0], 0.7)
    assert np.isclose(result[result['ticker'] == 'TEST2']['position_fraction'].iloc[0], 0.3)

def test_idm_application():
    """Test that IDM is applied after instrument weighting"""
    weight_layer = WeightLayer()
    weight_layer.fit(...)
    
    # Create returns for IDM calculation
    instrument_returns = pd.DataFrame({
        'TEST1': np.random.normal(0, 0.01, 1000),
        'TEST2': np.random.normal(0, 0.01, 1000)
    })
    
    portfolio = Portfolio(
        weight_layer=weight_layer,
        trading_timeframe=TimeFrame.DAILY,
        max_position_pct=None,
        idm_max=2.5
    )
    
    # Fit to calculate IDM
    portfolio.fit(instrument_returns)
    
    # Manually set IDM for testing
    portfolio.idm_ = 2.0
    
    combined_forecasts = pd.DataFrame({
        'ticker': ['TEST1'],
        'forecast_score': [1.0]
    })
    
    result = portfolio.predict(combined_forecasts)
    
    # With single instrument, weight=1.0, forecast=1.0, IDM=2.0
    # Position = 1.0 * 1.0 * 2.0 = 2.0
    assert np.isclose(result['position_fraction'].iloc[0], 2.0)

def test_position_capping():
    """Test that position is capped at max_position_pct"""
    weight_layer = WeightLayer()
    weight_layer.fit(...)
    
    instrument_returns = pd.DataFrame({
        'TEST': np.random.normal(0, 0.01, 1000)
    })
    
    portfolio = Portfolio(
        weight_layer=weight_layer,
        trading_timeframe=TimeFrame.DAILY,
        max_position_pct=2.0,  # Cap at 2.0
        idm_max=2.5
    )
    
    portfolio.fit(instrument_returns)
    portfolio.idm_ = 3.0  # High IDM that would exceed cap
    
    # Forecast that would give 4.0 uncapped (with IDM=3.0)
    combined_forecasts = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.33]  # 1.33 * 1.0 * 3.0 = 4.0
    })
    
    result = portfolio.predict(combined_forecasts)
    
    # Should be capped at 2.0 (assuming single instrument, weight=1.0)
    assert np.isclose(result['position_fraction'].iloc[0], 2.0)
```

---

## Phase 3: Create Execution Layer

**Goal**: Create a new execution layer that converts position fractions to tradeable contracts.

**Estimated Effort**: 3-4 hours

### Files to Create

- `execution/__init__.py`
- `execution/position_sizer.py`
- `tests/test_position_sizer.py`

### Data Classes

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ContractSpec:
    """Specification for a futures contract"""
    ticker: str
    price: float          # Current market price
    multiplier: float     # Points to dollars (e.g., $20 for NQ)
    fx_rate: float = 1.0  # FX conversion (default USD/USD = 1.0)
    min_tick: float = 0.25  # Minimum price increment

@dataclass(frozen=True)
class Position:
    """Calculated position for an instrument"""
    ticker: str
    forecast_score: float
    position_fraction: float
    target_dollars: float
    contracts: int
    notional_value: float
    notional_pct: float
```

### PositionSizer Class

```python
from enum import Enum
from typing import Dict
import pandas as pd
import numpy as np

class RoundingMethod(Enum):
    ROUND = 'round'
    FLOOR = 'floor'
    CEILING = 'ceiling'

class PositionSizer:
    """Convert position fractions to tradeable contracts"""
    
    def __init__(
        self,
        capital: float,
        contract_specs: Dict[str, ContractSpec],
        rounding_method: RoundingMethod = RoundingMethod.ROUND
    ):
        """
        Parameters
        ----------
        capital : float
            Account capital in base currency (USD)
        contract_specs : dict[str, ContractSpec]
            Contract specifications per ticker
        rounding_method : RoundingMethod
            How to round fractional contracts
        """
        self.capital = capital
        self.contract_specs = contract_specs
        self.rounding_method = rounding_method
    
    def calculate_positions(
        self,
        position_fractions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert position fractions to contracts.
        
        Parameters
        ----------
        position_fractions : pd.DataFrame
            Columns: ['ticker', 'forecast_score', 'position_fraction']
        
        Returns
        -------
        pd.DataFrame with columns:
            - ticker
            - forecast_score
            - position_fraction
            - target_dollars
            - contracts
            - notional_value
            - notional_pct
        """
        results = []
        
        for _, row in position_fractions.iterrows():
            ticker = row['ticker']
            forecast_score = row['forecast_score']
            position_fraction = row['position_fraction']
            
            # Get contract spec
            spec = self.contract_specs.get(ticker)
            if spec is None:
                raise ValueError(f"No contract spec for {ticker}")
            
            # Calculate target dollar allocation
            target_dollars = position_fraction * self.capital
            
            # Calculate contract value
            contract_value = spec.price * spec.multiplier * spec.fx_rate
            
            # Calculate raw contracts
            contracts_raw = target_dollars / contract_value
            
            # Round contracts
            contracts = self._round_contracts(contracts_raw)
            
            # Calculate actual notional
            notional_value = contracts * contract_value
            notional_pct = notional_value / self.capital if self.capital > 0 else 0.0
            
            results.append({
                'ticker': ticker,
                'forecast_score': forecast_score,
                'position_fraction': position_fraction,
                'target_dollars': target_dollars,
                'contracts': contracts,
                'notional_value': notional_value,
                'notional_pct': notional_pct
            })
        
        return pd.DataFrame(results)
    
    def _round_contracts(self, contracts_raw: float) -> int:
        """Round fractional contracts based on rounding method"""
        if self.rounding_method == RoundingMethod.ROUND:
            return int(round(contracts_raw))
        elif self.rounding_method == RoundingMethod.FLOOR:
            return int(np.floor(contracts_raw))
        elif self.rounding_method == RoundingMethod.CEILING:
            return int(np.ceil(contracts_raw))
        else:
            raise ValueError(f"Unknown rounding method: {self.rounding_method}")
```

### Unit Tests

```python
def test_contract_calculation():
    """Test basic contract calculation"""
    specs = {
        'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20, fx_rate=1.0)
    }
    sizer = PositionSizer(capital=1_000_000, contract_specs=specs)
    
    positions = pd.DataFrame({
        'ticker': ['NQ'],
        'forecast_score': [0.5],
        'position_fraction': [0.6]
    })
    
    result = sizer.calculate_positions(positions)
    
    # Target: $600,000
    # Contract value: 16,000 * 20 = $320,000
    # Raw contracts: 600,000 / 320,000 = 1.875
    # Rounded: 2
    # Notional: 2 * 320,000 = $640,000
    assert result['contracts'].iloc[0] == 2
    assert result['notional_value'].iloc[0] == 640_000
    assert np.isclose(result['notional_pct'].iloc[0], 0.64)

def test_floor_rounding():
    """Test floor rounding method"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(
        capital=100_000,
        contract_specs=specs,
        rounding_method=RoundingMethod.FLOOR
    )
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'position_fraction': [0.28]  # Would give 2.8 contracts
    })
    
    result = sizer.calculate_positions(positions)
    
    # Floor(2.8) = 2
    assert result['contracts'].iloc[0] == 2

def test_ceiling_rounding():
    """Test ceiling rounding method"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(
        capital=100_000,
        contract_specs=specs,
        rounding_method=RoundingMethod.CEILING
    )
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [1.0],
        'position_fraction': [0.22]  # Would give 2.2 contracts
    })
    
    result = sizer.calculate_positions(positions)
    
    # Ceiling(2.2) = 3
    assert result['contracts'].iloc[0] == 3

def test_zero_contracts():
    """Test handling of zero contracts (position too small)"""
    specs = {'TEST': ContractSpec(ticker='TEST', price=10000, multiplier=100)}
    sizer = PositionSizer(capital=100_000, contract_specs=specs)
    
    positions = pd.DataFrame({
        'ticker': ['TEST'],
        'forecast_score': [0.1],
        'position_fraction': [0.001]  # Very small position
    })
    
    result = sizer.calculate_positions(positions)
    
    # Should round to 0
    assert result['contracts'].iloc[0] == 0
    assert result['notional_value'].iloc[0] == 0.0
```

---

## Phase 4: Create Volatility Utilities

**Goal**: Create utilities for calculating blended volatility (70% EWMA-32 + 30% 10-year average).

**Estimated Effort**: 3-4 hours

### Files to Create

- `utils/volatility.py`
- `tests/test_volatility.py`

### BlendedVolatility Class

```python
import pandas as pd
import numpy as np
from typing import Optional

class BlendedVolatility:
    """
    Calculate Carver's blended volatility estimate.
    
    Formula: 0.70 * EWMA-32 + 0.30 * 10-year average
    """
    
    def __init__(
        self,
        short_span: int = 32,
        long_window: int = 2520,  # ~10 years of daily data
        short_weight: float = 0.70,
        annualization_factor: float = 252  # Trading days per year
    ):
        """
        Parameters
        ----------
        short_span : int, default=32
            Span for EWMA (short-run estimate)
        long_window : int, default=2520
            Window for rolling average (long-run estimate)
            Default is ~10 years of daily data
        short_weight : float, default=0.70
            Weight for short-run estimate (Carver uses 0.70)
        annualization_factor : float, default=252
            Factor to annualize daily volatility
        """
        self.short_span = short_span
        self.long_window = long_window
        self.short_weight = short_weight
        self.long_weight = 1.0 - short_weight
        self.annualization_factor = annualization_factor
    
    def calculate(
        self,
        returns: pd.Series,
        min_periods_short: Optional[int] = None,
        min_periods_long: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate blended volatility from returns.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns (not annualized)
        min_periods_short : int, optional
            Minimum periods for EWMA. Defaults to short_span
        min_periods_long : int, optional
            Minimum periods for rolling average. Defaults to long_window
        
        Returns
        -------
        pd.Series
            Blended annualized volatility
        """
        if min_periods_short is None:
            min_periods_short = self.short_span
        if min_periods_long is None:
            min_periods_long = self.long_window
        
        # Calculate short-run estimate (EWMA)
        short_vol = returns.ewm(
            span=self.short_span,
            min_periods=min_periods_short
        ).std()
        
        # Calculate long-run estimate (rolling average)
        long_vol = returns.rolling(
            window=self.long_window,
            min_periods=min_periods_long
        ).std()
        
        # Blend
        blended_vol = (
            self.short_weight * short_vol +
            self.long_weight * long_vol
        )
        
        # Annualize
        annualized_vol = blended_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def calculate_by_ticker(
        self,
        returns_df: pd.DataFrame,
        ticker_col: str = 'ticker',
        return_col: str = 'returns',
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate blended volatility for multiple tickers.
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            DataFrame with returns data
        ticker_col : str, default='ticker'
            Column name for ticker symbols
        return_col : str, default='returns'
            Column name for returns
        date_col : str, optional
            Column name for dates (for sorting)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [ticker_col, date_col (if provided), 'blended_volatility']
        """
        if date_col:
            returns_df = returns_df.sort_values([ticker_col, date_col])
        
        results = []
        
        for ticker, group in returns_df.groupby(ticker_col):
            returns_series = group[return_col]
            blended_vol = self.calculate(returns_series)
            
            result_df = pd.DataFrame({
                ticker_col: ticker,
                'blended_volatility': blended_vol
            })
            
            if date_col:
                result_df[date_col] = group[date_col].values
            
            results.append(result_df)
        
        return pd.concat(results, ignore_index=True)
```

### Unit Tests

```python
def test_blended_volatility_basic():
    """Test basic blended volatility calculation"""
    # Create synthetic returns with known volatility
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    calculator = BlendedVolatility()
    blended_vol = calculator.calculate(returns)
    
    # Should be annualized and non-null after sufficient periods
    assert not blended_vol.iloc[-1] == np.nan
    assert blended_vol.iloc[-1] > 0
    # Annual vol should be roughly 0.01 * sqrt(252) ≈ 0.159
    assert 0.10 < blended_vol.iloc[-1] < 0.25

def test_blended_volatility_weights():
    """Test that weights are applied correctly"""
    # Create returns where short-run vol differs from long-run vol
    returns = pd.Series(np.random.normal(0, 0.01, 3000))
    # Add recent spike in volatility
    returns.iloc[-100:] = np.random.normal(0, 0.03, 100)
    
    calculator = BlendedVolatility()
    blended_vol = calculator.calculate(returns)
    
    # Short-run should react faster to spike
    # Long-run should be more stable
    # Blend should be in between
    assert blended_vol.iloc[-1] > blended_vol.iloc[-200]

def test_blended_volatility_by_ticker():
    """Test multi-ticker calculation"""
    # Create returns for multiple tickers
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    data = []
    
    for ticker in ['NQ', 'ES', 'GC']:
        returns = np.random.normal(0, 0.01, 500)
        for date, ret in zip(dates, returns):
            data.append({
                'date': date,
                'ticker': ticker,
                'returns': ret
            })
    
    df = pd.DataFrame(data)
    
    calculator = BlendedVolatility()
    result = calculator.calculate_by_ticker(
        df,
        ticker_col='ticker',
        return_col='returns',
        date_col='date'
    )
    
    # Should have volatility for all tickers
    assert set(result['ticker'].unique()) == {'NQ', 'ES', 'GC'}
    assert 'blended_volatility' in result.columns
    
    # Check one ticker's last volatility
    nq_vol = result[result['ticker'] == 'NQ']['blended_volatility'].iloc[-1]
    assert 0.05 < nq_vol < 0.30
```

---

## Phase 5: Integration & Testing

**Goal**: Wire everything together and create end-to-end tests.

**Estimated Effort**: 4-6 hours

### Files to Create

- `scripts/demo_position_sizing.py`
- `tests/test_e2e_position_sizing.py`

### Demo Script

```python
"""
Demo script showing end-to-end position sizing workflow.
"""
import pandas as pd
import numpy as np
from ensemble.portfolio import Portfolio
from ensemble.diversified_ensemble import DiversifiedEnsemble
from execution.position_sizer import PositionSizer, ContractSpec
from utils.volatility import BlendedVolatility
from feature_selection.base_models.base_model import BaseModel
from data_processing.time_frame import TimeFrame

# Step 1: Load data
print("Loading data...")
# (Assume data loading code here)

# Step 2: Calculate blended volatility
print("Calculating blended volatility...")
vol_calculator = BlendedVolatility()
volatility_df = vol_calculator.calculate_by_ticker(
    returns_df,
    ticker_col='ticker',
    return_col='returns',
    date_col='date'
)

# Get latest volatility for each ticker
latest_volatility = (
    volatility_df
    .groupby('ticker')
    .last()
    ['blended_volatility']
)

# Step 3: Create ensembles
print("Creating ensembles...")
# Mean reversion ensemble
mr_models = [
    BaseModel(n_bins=10, strategy='long') for _ in range(10)
]
mr_ensemble = DiversifiedEnsemble(models=mr_models)
mr_ensemble.fit(X_train_mr, y_train_mr, ticker_train_mr)

# Momentum ensemble
mom_models = [
    BaseModel(n_bins=10, strategy='long') for _ in range(10)
]
mom_ensemble = DiversifiedEnsemble(models=mom_models)
mom_ensemble.fit(X_train_mom, y_train_mom, ticker_train_mom)

# Step 4: Create portfolio
print("Creating portfolio...")
portfolio = Portfolio(
    ensembles=[mr_ensemble, mom_ensemble],
    trading_timeframe=TimeFrame.DAILY,
    target_volatility=0.20,  # 20% annual target
    dm=1.0,  # Start conservative, fit later
    max_position_pct=2.0
)

# Step 5: Generate position fractions
print("Generating position fractions...")
position_fractions = portfolio.predict(
    X_test,
    ticker_test,
    latest_volatility,
    TimeFrame.DAILY
)

print("\nPosition Fractions:")
print(position_fractions)

# Step 6: Convert to contracts
print("\nConverting to contracts...")
contract_specs = {
    'NQ': ContractSpec(ticker='NQ', price=16000, multiplier=20),
    'ES': ContractSpec(ticker='ES', price=4800, multiplier=50),
    'YM': ContractSpec(ticker='YM', price=38000, multiplier=5),
    'GC': ContractSpec(ticker='GC', price=2000, multiplier=100),
    'CL': ContractSpec(ticker='CL', price=75, multiplier=1000)
}

sizer = PositionSizer(
    capital=1_000_000,
    contract_specs=contract_specs
)

positions = sizer.calculate_positions(position_fractions)

print("\nFinal Positions:")
print(positions)

# Step 7: Summary statistics
print("\n=== Portfolio Summary ===")
print(f"Total Target Allocation: ${positions['target_dollars'].sum():,.0f}")
print(f"Total Notional Value: ${positions['notional_value'].sum():,.0f}")
print(f"Total Notional %: {positions['notional_pct'].sum():.1%}")
print(f"Number of Instruments: {len(positions)}")
print(f"Number of Contracts: {positions['contracts'].sum()}")
```

### End-to-End Tests

```python
def test_e2e_position_sizing():
    """Test complete workflow from signals to contracts"""
    # Setup
    models = [BaseModel(n_bins=10) for _ in range(5)]
    ensemble = DiversifiedEnsemble(models=models)
    ensemble.fit(X_train, y_train, ticker_train)
    
    portfolio = Portfolio(
        ensembles=[ensemble],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0,
        max_position_pct=2.0
    )
    
    volatility = pd.Series({'TEST': 0.20})
    
    # Get position fractions
    positions = portfolio.predict(X_test, ticker_test, volatility)
    
    # Convert to contracts
    specs = {'TEST': ContractSpec(ticker='TEST', price=100, multiplier=100)}
    sizer = PositionSizer(capital=100_000, contract_specs=specs)
    contracts = sizer.calculate_positions(positions)
    
    # Verify complete pipeline
    assert 'forecast_score' in positions.columns
    assert 'position_fraction' in positions.columns
    assert 'contracts' in contracts.columns
    assert 'notional_value' in contracts.columns
    
    # Position fraction should be in reasonable range
    assert 0 <= positions['position_fraction'].iloc[0] <= 2.0
    
    # Contracts should be non-negative integer
    assert contracts['contracts'].iloc[0] >= 0
    assert isinstance(contracts['contracts'].iloc[0], (int, np.integer))

def test_e2e_multiple_ensembles():
    """Test workflow with multiple ensembles"""
    # Create two ensembles
    ensemble1 = DiversifiedEnsemble(models=[BaseModel(n_bins=10) for _ in range(5)])
    ensemble2 = DiversifiedEnsemble(models=[BaseModel(n_bins=10) for _ in range(5)])
    
    ensemble1.fit(X_train, y_train, ticker_train)
    ensemble2.fit(X_train, y_train, ticker_train)
    
    # Portfolio with both ensembles
    portfolio = Portfolio(
        ensembles=[ensemble1, ensemble2],
        trading_timeframe=TimeFrame.DAILY,
        target_volatility=0.20,
        dm=1.0
    )
    
    volatility = pd.Series({'TEST': 0.20})
    positions = portfolio.predict(X_test, ticker_test, volatility)
    
    # Forecast should be average of two ensembles
    # (exact value depends on model predictions, just check it's computed)
    assert not positions['forecast_score'].isnull().any()
```

---

## Phase 6: FDM/IDM Calculation Utilities (Optional)

**Goal**: Create utility functions for calculating FDM and IDM from correlations. These are now built into WeightLayer and Portfolio, but utilities can be useful for validation and analysis.

**Estimated Effort**: 1-2 hours (optional)

### Files to Create

- `utils/diversification.py`
- `tests/test_diversification.py`

### FDM Calculator

```python
import pandas as pd
import numpy as np
from typing import List

def calculate_fdm(
    forecast_vectors: List[pd.DataFrame],
    fdm_max: float = 2.0
) -> float:
    """
    Calculate Forecast Diversification Multiplier from forecast value correlations.
    
    Parameters
    ----------
    forecast_vectors : list[pd.DataFrame]
        List of forecast vectors from all ensembles
    fdm_max : float, default=2.0
        Maximum FDM value
    
    Returns
    -------
    float
        FDM value (capped at fdm_max)
    """
    # Same implementation as WeightLayer._calculate_fdm()
    # ... (see Phase 2 implementation)
```

### IDM Calculator

```python
def calculate_idm(
    instrument_returns: pd.DataFrame,
    idm_max: float = 2.5
) -> float:
    """
    Calculate Instrument Diversification Multiplier from return correlations.
    
    Parameters
    ----------
    instrument_returns : pd.DataFrame
        Historical returns for all instruments
    idm_max : float, default=2.5
        Maximum IDM value
    
    Returns
    -------
    float
        IDM value (capped at idm_max)
    """
    # Same implementation as Portfolio._calculate_idm()
    # ... (see Phase 3 implementation)
    
    def fit_with_validation(
        self,
        returns: pd.Series,
        annualization_factor: float = 252,
        min_dm: float = 0.5,
        max_dm: float = 5.0
    ) -> dict:
        """
        Fit DM with validation and bounds checking.
        
        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns
        annualization_factor : float
            Factor to annualize volatility
        min_dm : float, default=0.5
            Minimum allowed DM
        max_dm : float, default=5.0
            Maximum allowed DM
        
        Returns
        -------
        dict with keys:
            - dm_optimal: Fitted DM value
            - dm_bounded: DM after applying bounds
            - realized_vol: Realized volatility
            - target_vol: Target volatility
            - ratio: target / realized
        """
        # Calculate realized volatility
        realized_vol = returns.std() * np.sqrt(annualization_factor)
        
        # Calculate optimal DM
        dm_optimal = self.target_volatility / realized_vol if realized_vol > 0 else 1.0
        
        # Apply bounds
        dm_bounded = np.clip(dm_optimal, min_dm, max_dm)
        
        return {
            'dm_optimal': dm_optimal,
            'dm_bounded': dm_bounded,
            'realized_vol': realized_vol,
            'target_vol': self.target_volatility,
            'ratio': dm_optimal
        }
```

### Unit Tests

```python
def test_fdm_calculation():
    """Test FDM calculation from forecast correlations"""
    # Create forecast vectors with known correlations
    forecast_vectors = [
        pd.DataFrame({
            'ticker': ['TEST'] * 3,
            'model_name': ['model_1', 'model_2', 'model_3'],
            'forecast': [1.0, 0.8, 0.6],
            'signal': [1, 1, 1]
        })
    ]
    
    fdm = calculate_fdm(forecast_vectors, fdm_max=2.0)
    
    # FDM should be between 1.0 and 2.0
    assert 1.0 <= fdm <= 2.0

def test_idm_calculation():
    """Test IDM calculation from return correlations"""
    # Create returns with known correlations
    np.random.seed(42)
    returns = pd.DataFrame({
        'NQ': np.random.normal(0, 0.01, 1000),
        'ES': np.random.normal(0, 0.01, 1000),
        'GC': np.random.normal(0, 0.01, 1000)
    })
    
    idm = calculate_idm(returns, idm_max=2.5)
    
    # IDM should be between 1.0 and 2.5
    assert 1.0 <= idm <= 2.5

def test_fdm_capping():
    """Test that FDM is capped at fdm_max"""
    # Create forecasts that would give very high FDM
    # (uncorrelated forecasts)
    forecast_vectors = [
        pd.DataFrame({
            'ticker': ['TEST'] * 10,
            'model_name': [f'model_{i}' for i in range(10)],
            'forecast': np.random.rand(10),
            'signal': [1] * 10
        })
    ]
    
    fdm = calculate_fdm(forecast_vectors, fdm_max=2.0)
    
    # Should be capped at 2.0
    assert fdm <= 2.0

def test_idm_capping():
    """Test that IDM is capped at idm_max"""
    # Create returns that would give very high IDM
    # (uncorrelated instruments)
    returns = pd.DataFrame({
        f'INST_{i}': np.random.normal(0, 0.01, 1000)
        for i in range(20)
    })
    
    idm = calculate_idm(returns, idm_max=2.5)
    
    # Should be capped at 2.5
    assert idm <= 2.5
```

---

## Summary Checklist

### Phase 1: Refactor Ensemble (Risk Management) ⚠️
- [ ] Add target_volatility parameter
- [ ] Add volatility parameter to predict()
- [ ] Implement volatility scaling per base model
- [ ] Change output to vector (one row per base model)
- [ ] Add model_name to output
- [ ] Remove signal combination logic
- [ ] Update unit tests for vector output

### Phase 2: Create Weight Layer 🆕
- [ ] Create WeightLayer class
- [ ] Create InverseCorrelationWeighter class
- [ ] Implement fit() method for weight calculation and FDM calculation
- [ ] Implement _calculate_fdm() method (forecast value correlations)
- [ ] Implement combine() method for forecast combination with FDM application
- [ ] Add unit tests for weight calculation
- [ ] Add unit tests for FDM calculation
- [ ] Add unit tests for forecast combination with FDM

### Phase 3: Simplify Portfolio ⚠️
- [ ] Remove volatility scaling logic (moved to Ensemble)
- [ ] Add fit() method for IDM calculation from instrument returns
- [ ] Add _calculate_idm() method (instrument return correlations)
- [ ] Add _apply_idm() method to apply IDM after instrument weighting
- [ ] Simplify to instrument weighting, IDM application, and optional capping
- [ ] Update to accept combined forecasts from Weight layer (already FDM-scaled)
- [ ] Update unit tests for IDM calculation and application

### Phase 4: Create Volatility Utilities 🆕
- [ ] Create BlendedVolatility class
- [ ] Implement EWMA-32 calculation
- [ ] Implement 10-year rolling average
- [ ] Add multi-ticker support
- [ ] Add unit tests

### Phase 5: Create Execution Layer 🆕
- [ ] Create ContractSpec dataclass
- [ ] Create Position dataclass
- [ ] Create PositionSizer class
- [ ] Implement contract rounding
- [ ] Add unit tests

### Phase 6: Integration & Testing ✅
- [ ] Create demo script
- [ ] Create end-to-end tests
- [ ] Validate complete pipeline
- [ ] Test multiple ensembles with different numbers of base models
- [ ] Verify signal dilution is solved
- [ ] Document usage examples

---

## Total Estimated Effort

- **Phase 1**: 4-6 hours (Refactor Ensemble for risk management)
- **Phase 2**: 5-7 hours (Create Weight Layer with FDM)
- **Phase 3**: 3-4 hours (Simplify Portfolio with IDM)
- **Phase 4**: 3-4 hours (Create Volatility Utilities)
- **Phase 5**: 3-4 hours (Create Execution Layer)
- **Phase 6**: 4-6 hours (Integration & Testing)
- **Phase 7**: 1-2 hours (FDM/IDM Utilities - optional)

**Total**: 23-33 hours (~3-4 days of focused work)

---

## Corrections Applied (2025-01-09)

This section documents corrections made to align implementation specs with forecast specs and current codebase structure.

### Issue 1: Model Reference Inconsistency (Fixed)
**Problem**: Spec used `self.models` (list) but codebase uses `self.base_models` (dict).  
**Solution**: Updated to use dict-based structure:
```python
self.model_exposure_fractions_ = {
    model_name: 1.0 / base_model.n_bins 
    for model_name, base_model in self.base_models.items()
}
```

### Issue 2: Exposure Calculation Array Assumptions (Fixed)
**Problem**: Code assumed numpy array operations on signals, but actual structure is DataFrame with model names as columns.  
**Solution**: Updated to iterate over DataFrame rows and columns explicitly.

### Issue 3: Volatility Parameter Type Clarity (Fixed)
**Problem**: Ambiguous whether volatility parameter was indexed by sample or ticker.  
**Solution**: 
- Clarified documentation: `Union[pd.Series, Dict[str, float]]`
- If Series: must have ticker symbols as index
- If dict: maps ticker → volatility value
- Added validation and conversion logic in Portfolio.predict()

### Issue 4: Instrument Weight Calculation Bug (Fixed)
**Problem**: Used `len(df)` (number of rows) instead of number of unique instruments.  
**Solution**: Changed to `df['ticker'].nunique()` for correct equal weighting.

### Issue 5: Breaking Change - Return Type (Documented)
**Problem**: Ensemble.predict() currently returns `np.ndarray`, spec changes to `pd.DataFrame`.  
**Solution**: 
- Documented as BREAKING CHANGE in Phase 1
- Added migration notes
- Added output structure validation tests
- Updated Portfolio to expect and validate DataFrame input

### Issue 6: Missing Type Imports (Fixed)
**Problem**: Union type used without import statement.  
**Solution**: Added required imports to Phase 2 specification.

### Additional Improvements
1. **Validation**: Added input validation for volatility parameter in Portfolio
2. **Error Messages**: Added descriptive error messages for ensemble output validation
3. **Edge Cases**: Added handling for empty ensemble list in averaging
4. **Documentation**: Clarified that ensembles now return DataFrames, not arrays
5. **Consistency**: Used `as_index=False` in groupby for cleaner code

### Migration Checklist
- [ ] Phase 1: Change ensemble output to DataFrame (BREAKING)
- [ ] Phase 1: Remove volatility parameter from ensemble.predict()
- [ ] Phase 1: Update all tests expecting numpy array to expect DataFrame
- [ ] Phase 2: Add volatility parameter to portfolio.predict()
- [ ] Phase 2: Update portfolio to handle DataFrame from ensembles
- [ ] Phase 2: Add Union type imports
- [ ] Integration: Update all calling code to use new signatures

---

**End of Implementation Specifications**

