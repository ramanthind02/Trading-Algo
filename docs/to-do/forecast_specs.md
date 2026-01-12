# Forecast Generation & Position Sizing Architecture

> **📖 This Document**: Context, design principles, formulas, and examples  
> **🔨 For Implementation**: See [implementation_specs.md](./implementation_specs.md)

**Version**: 6.0.0  
**Date**: 2025-01-09  
**Status**: Specification (Context & Design)  
**Implementation**: See `implementation_specs.md`

**Major Changes**:
- **v3.0.0**: Moved all diversification, scaling, and capping to Portfolio layer. Ensembles now only combine signals using inverse correlation weights and return raw forecasts [0, 1].
- **v3.1.0**: Combined FDM and IDM into a single Diversification Multiplier (DM). Simple fitting from past performance: $\text{DM} = \frac{\text{target volatility}}{\text{realized volatility}}$. Trivial fitting approach won't overfit.
- **v3.2.0**: Removed forecast scalar (redundant - multiply by 10 then divide by 10). Raw forecasts in [0, 1] are used directly as multipliers. Removed forecast capping (max raw forecast is 1.0, so no capping needed).
- **v3.3.0**: Updated to use Carver's blended volatility estimate (70% EWMA-32 + 30% 10-year average) instead of simple EWSD-30. Added comprehensive unit testing guide with buy/hold scenarios for easier validation.
- **v4.0.0**: Split into two documents: `forecast_specs.md` (context, design, formulas, examples) and `implementation_specs.md` (concrete build tasks, method signatures, unit tests). Better organization for understanding vs. building.
- **v5.0.0**: **MAJOR RESTRUCTURE**: Ensembles now perform risk management per base model (volatility scaling). New Weight layer combines all base model forecasts using weight vector (inverse correlation). Portfolio layer becomes minimal (mostly pass-through). This solves signal dilution when ensembles have different numbers of base models.
- **v6.0.0**: **FDM & IDM SEPARATION**: Split single DM into FDM (Forecast Diversification Multiplier) at Weight layer and IDM (Instrument Diversification Multiplier) at Portfolio layer, following Carver's approach. FDM based on forecast value correlations, IDM based on instrument return correlations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer 1: Base Models](#layer-1-base-models)
3. [Layer 2: Ensemble](#layer-2-ensemble)
4. [Layer 3: Weight Layer](#layer-3-weight-layer)
5. [Layer 4: Portfolio](#layer-4-portfolio)
6. [Layer 5: Execution](#layer-5-execution)
7. [Complete Workflow Example](#complete-workflow-example)
8. [Practical Guide: Tuning Diversification Multiplier (DM)](#practical-guide-tuning-diversification-multiplier-dm)
9. [Unit Testing Guide: Buy/Hold Scenarios](#unit-testing-guide-buyhold-scenarios)

---

## Architecture Overview

Our system follows a **functional, layered architecture** where each layer has a single responsibility and passes data forward without side effects. This follows Robert Carver's systematic trading framework with adaptations for our binary signal generation approach.

```
┌─────────────────────────────────────────────────────────────┐
│                     TRADING SYSTEM LAYERS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Base Models                                        │
│  ├─ Input:  Raw features (EWMAC, RSI, Momentum, etc.)      │
│  └─ Output: Binary signals {0, 1}                           │
│                                                              │
│  Layer 2: Ensemble (Risk Management per Model)               │
│  ├─ Input:  Binary signals from base models                 │
│  ├─ Process: Volatility scaling per base model             │
│  │           (assumes signal=1, uses model's avg exposure)  │
│  └─ Output: Vector of forecasts (one per base model)        │
│                                                              │
│  Layer 3: Weight Layer (Signal Combination)                 │
│  ├─ Input:  Forecast vectors from all ensembles             │
│  ├─ Process: Combine using weight vector                   │
│  │           Apply FDM (forecast value correlations)       │
│  └─ Output: Combined forecast per instrument (FDM-scaled)   │
│                                                              │
│  Layer 4: Portfolio (Minimal - Pass-through)                │
│  ├─ Input:  Combined forecasts (FDM-scaled)                │
│  ├─ Process: Instrument weighting, IDM, capping (optional) │
│  └─ Output: Position fractions (% of capital)               │
│                                                              │
│  Layer 5: Execution                                          │
│  ├─ Input:  Position fractions, contract specs              │
│  └─ Output: Number of contracts to trade                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```


### Design Principles

1. **Separation of Concerns**: Each layer handles one aspect of the trading decision
2. **Functional Core**: Pure functions with no side effects where possible
3. **Immutable Data**: Use frozen dataclasses and return new objects
4. **Type Safety**: Full type hints on all interfaces
5. **Testability**: Each layer can be tested independently
6. **Ensemble as Risk Manager**: Each ensemble performs risk management on its base models
7. **Weight Layer for Combination**: All base model forecasts are combined at weight layer using pluggable methods

### Key Architectural Decision: Risk Management at Ensemble Level

**Why move risk management to Ensemble layer?**

1. **Solves Signal Dilution Problem**
   - **Problem**: If Ensemble A has 10 base models and Ensemble B has 2 base models, averaging ensemble forecasts causes signal dilution
   - **Example**: All A active (10/10) + half B active (1/2) → average = 0.75, even though 11/12 models are active
   - **Solution**: Don't combine at ensemble level. Instead, output vector of forecasts (one per base model) and combine all base models at weight layer

2. **Cleaner separation of concerns**
   - **Ensemble**: "How much risk per base model?" (volatility scaling per model)
   - **Weight Layer**: "How to combine all base models?" (inverse correlation, linear model, etc.)
   - **Portfolio**: "How to allocate across instruments?" (instrument weighting, capping)
   
3. **Pluggable combination methods**
   - Weight layer can use different methods: inverse correlation, linear models, ML models
   - Easy to experiment with different combination strategies
   - Ensembles just organize similar signals, don't combine them

4. **Per-model risk management**
   - Each base model gets its own volatility-adjusted forecast
   - Uses model's average exposure fraction for proper scaling
   - More granular control over risk allocation

### Forecast Diversification Multiplier (FDM)

**Purpose**: The Forecast Diversification Multiplier (FDM) is a **scaling factor** applied at the Weight layer that increases combined forecast strength to account for diversification benefits when multiple base model forecasts are combined.

**The Problem Without It**:
- When multiple base model forecasts are combined, the resulting combined forecast is typically weaker than if all models agreed perfectly
- Example: 10 base models with average correlation 0.3 → combined forecast is "muffled" by diversification
- Without FDM, we under-utilize the signal strength of our base models

**The Solution**:
- **FDM** accounts for diversification across:
  - Base models within ensembles
  - Base models across different ensembles
  - Low correlation between forecast values (not just signals)

**Calculation Method** (Carver's Approach):
- Based on **correlation matrix of forecast values** (not binary signals)
- Calculated during `fit()` from training data
- Formula: $\text{FDM} = \sqrt{\frac{1}{\bar{\rho} + \epsilon}}$ where $\bar{\rho}$ is mean forecast correlation
- **Capped at 2.0** to prevent excessive scaling

**Typical Values** (from Carver):
- 2 trading rules: ~1.02 FDM
- 6 trading rules: ~1.27 FDM
- 30 trading rules: ~1.81 FDM

**Key Insight**: FDM is **global** (same for all instruments) because it's about how forecasts combine, not instrument-specific properties.

### Instrument Diversification Multiplier (IDM)

**Purpose**: The Instrument Diversification Multiplier (IDM) is a **scaling factor** applied at the Portfolio layer that increases position sizes to account for portfolio-level diversification benefits across multiple instruments.

**The Problem Without It**:
- A diversified portfolio of many instruments has lower realized volatility than the sum of individual instrument risks
- Example: 10 uncorrelated instruments → portfolio volatility much lower than if perfectly correlated
- Without IDM, we'd miss our target portfolio volatility (e.g., target 20%, realize only 12%)

**The Solution**:
- **IDM** accounts for diversification across:
  - Different instruments (NQ, ES, GC, CL)
  - Different asset classes (equities, commodities, FX)
  - Low correlation between instrument returns

**Calculation Method** (Carver's Approach):
- Based on **correlation matrix of instrument returns** (not forecasts)
- Calculated during `fit()` from historical returns
- Formula: $\text{IDM} = \sqrt{\frac{1}{\bar{\rho} + \epsilon}}$ where $\bar{\rho}$ is mean return correlation
- **Capped at 2.5** to prevent excessive leverage

**Typical Values** (from Carver):
- 2 instruments: ~1.20 IDM
- 10 instruments: ~2.20 IDM
- 30+ instruments: ~2.50 IDM (capped)

**Key Insight**: IDM is **global** (applied to all instruments) because it's a portfolio-level property, not instrument-specific.

### Why Two Separate Multipliers?

**Separation of Concerns**:
- **FDM**: Handles forecast-level diversification (how signals combine)
- **IDM**: Handles instrument-level diversification (how instruments combine)

**Different Data Sources**:
- **FDM**: Uses forecast value correlations (from Weight layer training)
- **IDM**: Uses instrument return correlations (from Portfolio historical data)

**Different Caps**:
- **FDM**: Capped at 2.0 (forecast strength restoration)
- **IDM**: Capped at 2.5 (portfolio leverage control)

**Mathematical Correctness**: While multiplication is commutative, separating FDM and IDM makes the logic clearer and aligns with Carver's proven framework.

### Weighting Strategy

**Ensemble Level** (Portfolio Layer):
- **Equal weight** across all ensembles
- Ensembles represent different trading styles/regimes (e.g., Mean Reversion, Momentum)
- Instruments can appear in **multiple ensembles**
- Forecasts are **averaged** across ensembles (not summed) to equate risk

**Instrument Level** (Portfolio Layer):
- **Current**: Equal weight per instrument
- **Future**: Carver-style handcrafting (top-down: asset class → group → instrument)
- Risk-agnostic weights (volatility scaling handles risk adjustment)

### Blended Volatility Estimate

**Purpose**: Robert Carver uses a **blended volatility estimate** to balance short-term volatility clustering with long-term mean reversion. This provides more robust risk forecasting and prevents dangerous over-leveraging during quiet market periods.

**Formula**:

$$\sigma_{\text{blended}} = 0.70 \times \sigma_{\text{short}} + 0.30 \times \sigma_{\text{long}}$$

Where:
- $\sigma_{\text{short}}$: **Short-run estimate** = Exponentially Weighted Standard Deviation (EWSD) with **32-day span**
- $\sigma_{\text{long}}$: **Long-run estimate** = **10-year rolling average**

**Why This Matters**:

1. **Improved Risk Forecasting**: Volatility is "sticky" but mean-reverting. The blend accounts for both short-term clustering and long-term reversion.

2. **Prevents Over-Leveraging**: A purely short-term estimate can drop to dangerously low levels during quiet periods, signaling huge position sizes just before market crises. The long-term component acts as a safeguard.

3. **Reduces Trading Costs**: The stable 10-year component dampens day-to-day changes, leading to fewer position adjustments and lower transaction costs.

4. **Application**: Used in both forecast generation (for risk normalization) and position sizing (inverse relationship with position size).

**Implementation**:
- Calculate EWMA-32 for short-run estimate
- Calculate 10-year rolling average for long-run estimate
- Blend with 70/30 weighting
- Update daily (short-run changes daily, long-run changes slowly)

**Note**: This replaces simple EWSD-30 or other single-method volatility estimates. The blended approach is critical for robust risk management.

---

## Layer 1: Base Models

### Responsibility

**Generate binary trading signals** based on feature values and binning strategies.

### Current Implementation

✅ **Status**: Already implemented correctly in `feature_selection/base_models/base_model.py`

### Input

- `feature_data`: Raw feature values (e.g., EWMAC_32_64_D)
- `target_data`: Returns (for training only)
- `normalization_data`: Optional volatility data (for training only)

### Output

- Binary signal: $X_i \in \{0, 1\}$
  - $X_i = 1$: Signal is active (trade)
  - $X_i = 0$: Signal is inactive (no trade)

### Algorithm

```
1. Fit Phase:
   a. Bin raw features into n_bins using quantile/tree binning
   b. Calculate Sortino ratio for each bin
   c. Select best_long_bin and best_short_bin
   d. Store thresholds for prediction

2. Predict Phase:
   a. Assign input to bin based on thresholds
   b. Return 1 if bin == best_bin, else 0
```

### Key Parameters

- `n_bins`: Number of bins (typically 10, means 10% exposure)
- `selection_metric`: 'sortino' or 'mean'
- `strategy`: 'long' or 'short'

### Notes

- Base models know **when** to trade, not **how much**
- Binary output is intentional - sizing happens at higher layers
- Each model typically has exposure fraction $h_i \approx \frac{1}{n\_bins}$

---

## Layer 2: Ensemble

### Responsibility

**Perform risk management on each individual base model** and output a vector of volatility-adjusted forecasts (one per base model). Ensembles organize similar signals together but do NOT combine them.

### Current Implementation

⚠️ **Status**: Needs major refactoring - now performs risk management per base model instead of combining signals

### Input

From predict():
- `X`: DataFrame with raw features
- `ticker`: Instrument ticker symbols
- `volatility`: Blended volatility per instrument (annualized, 70% EWMA-32 + 30% 10-year average)
- `normalization_data`: Optional normalization data for base models

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `model_name`: Base model identifier (one row per base model)
- `forecast`: Volatility-adjusted forecast for this base model (0 if signal inactive)
- `signal`: Binary signal from base model {0, 1}

**Key Change**: Output is now a **vector** (one row per base model) instead of a single combined forecast.

**Note**: Exposure fraction $h_i$ is used internally to calculate the forecast but is not included in the output (it's already incorporated into the forecast value).

### Formula

#### Step 1: Get Binary Signals from Base Models

For each base model $i$ at time $t$:
- $X_{i,t} \in \{0, 1\}$: Binary signal from base model $i$

#### Step 2: Calculate Volatility-Adjusted Forecast Per Model

For each base model $i$, calculate the forecast assuming signal=1:

$$F_{i,\text{vol-adjusted}} = \frac{\tau}{\sigma_{\text{blended}} \times \sqrt{h_i}}$$

Where:
- $\tau$: Target annual portfolio volatility (e.g., 0.20 = 20%)
- $\sigma_{\text{blended}}$: Instrument's blended annualized volatility (70% EWMA-32 + 30% 10-year average)
- $h_i$: Exposure fraction for model $i$ (fraction of time in market, typically $\approx \frac{1}{n\_bins}$)

**Intuition**: This calculates the position size we would take if this model's signal were active (signal=1). The $\sqrt{h_i}$ adjustment accounts for sparse signals having lower realized volatility.

#### Step 3: Apply Signal

For each base model $i$:

$$F_i = \begin{cases}
F_{i,\text{vol-adjusted}} & \text{if } X_{i,t} = 1 \\
0 & \text{if } X_{i,t} = 0
\end{cases}$$

**Key Design Decision**: We calculate the forecast assuming signal=1, then multiply by the actual signal. This ensures inactive models contribute 0, while active models contribute their full volatility-adjusted forecast.

#### Step 4: Return Vector of Forecasts

Output one row per base model with:
- `forecast`: $F_i$ (volatility-adjusted forecast, 0 if signal inactive)
- `signal`: $X_{i,t}$ (binary signal)

**Note**: Exposure fraction $h_i$ is used internally in Step 2 to calculate the forecast but is not included in the output. The forecast already incorporates the exposure adjustment via $\sqrt{h_i}$.

**Why Vector Output**: This allows the Weight layer to combine all base models from all ensembles using a weight vector, avoiding signal dilution when ensembles have different numbers of base models.

### Key Parameters

- `target_volatility` ($\tau$): Annual portfolio volatility target (default: 0.20)
- **Note**: Volatility calculation happens in Ensemble layer (not Portfolio)

### Implementation Notes

#### What Changes from Current Code

**Add to Ensemble**:
- ✅ `volatility` parameter in predict()
- ✅ `target_volatility` parameter
- ✅ Volatility scaling per base model
- ✅ Vector output (one row per base model)
- ✅ Model name tracking in output

**Remove from Ensemble**:
- ❌ Signal combination (moves to Weight layer)
- ❌ Inverse correlation weights (moves to Weight layer)
- ❌ Combined forecast calculation

**Keep in Ensemble**:
- ✅ Base model signal generation
- ✅ Exposure fraction tracking per model

#### Method Signatures

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

### Notes

- Ensemble performs **risk management per base model**, not signal combination
- Output is a **vector** (one forecast per base model), not a single combined forecast
- Volatility calculation happens here (not in Portfolio)
- Ensembles organize similar signals but don't combine them
- All base models from all ensembles are combined later at Weight layer
- This design solves signal dilution when ensembles have different numbers of base models

---

## Layer 3: Weight Layer

### Responsibility

**Combine forecasts from all base models across all ensembles** using a weight vector. This is where signal combination happens.

### Current Implementation

🆕 **Status**: New component to be created

### Input

From combine():
- `forecast_vectors`: List of DataFrames from all ensembles
  - Each DataFrame has columns: `['ticker', 'model_name', 'forecast', 'signal']`
  - One row per base model

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `forecast_score`: Combined forecast (weighted sum of all base model forecasts)

### Formula

#### Step 1: Collect All Base Model Forecasts

For each instrument, collect forecasts from **all base models across all ensembles**:

$$\mathbf{F} = [F_1, F_2, \ldots, F_N]$$

Where $N$ is the total number of base models across all ensembles.

#### Step 2: Calculate Weight Vector

Weights are based on **inverse correlation** between base models (sophisticated approach that automatically adapts to signal correlations):

$$w_i = \frac{d_i}{\sum_{j=1}^{N} d_j}$$

Where diversification score:

$$d_i = \frac{1}{1 + \bar{\rho}_i}$$

And $\bar{\rho}_i$ is the average absolute correlation between base model $i$ and all other base models.

**Calculation method**:
1. Build correlation matrix of binary signals from all base models over training period
2. For each base model $i$, calculate average absolute correlation with all other base models
3. Convert to diversification score (inverse relationship)
4. Normalize to sum to 1.0

**Key Insight**: This inverse correlation weighting **accounts for diversification** across all base models, regardless of which ensemble they belong to. Models with lower correlation to others get higher weights.

**Note**: This approach is calculated during `fit()` and stored for use in `combine()`.

#### Step 3: Combine Forecasts

For each instrument at time $t$:

$$F_{\text{weighted}} = \sum_{i=1}^{N} F_{i,t} \cdot w_i$$

Where:
- $F_{i,t}$: Volatility-adjusted forecast from base model $i$ (from Ensemble layer, already incorporates exposure adjustment)
- $w_i$: Weight for base model $i$ (from inverse correlation)
- $\sum_{i=1}^{N} w_i = 1$: Weights sum to 1

**Note**: No need to track exposure fraction - it's already incorporated into each $F_{i,t}$ via the $\sqrt{h_i}$ adjustment in the Ensemble layer.

#### Step 4: Apply Forecast Diversification Multiplier (FDM)

After combining forecasts, apply FDM to restore forecast strength:

$$F_{\text{combined}} = F_{\text{weighted}} \times \text{FDM}$$

Where:
- $\text{FDM}$: Forecast Diversification Multiplier (calculated from forecast value correlations during `fit()`)
- FDM is **global** (same for all instruments) because it's about how forecasts combine
- **Capped at 2.0** to prevent excessive scaling

**Calculation of FDM** (during `fit()`):
1. Extract forecast values for all base models from training data
2. Build correlation matrix of forecast values (not binary signals)
3. Calculate mean correlation: $\bar{\rho} = \frac{1}{N(N-1)/2} \sum_{i<j} |\rho_{i,j}|$
4. Calculate FDM: $\text{FDM} = \sqrt{\frac{1}{\bar{\rho} + \epsilon}}$ (with small $\epsilon$ to avoid division by zero)
5. Cap at 2.0: $\text{FDM} = \min(\text{FDM}, 2.0)$
6. Floor negative correlations at zero (Carver's recommendation)

**Intuition**: When forecasts are uncorrelated ($\bar{\rho} \approx 0$), FDM is high (~$\sqrt{1/0.01} \approx 10$), but capped at 2.0. When forecasts are highly correlated ($\bar{\rho} \approx 1$), FDM is low (~$\sqrt{1/1} = 1.0$).

### Key Parameters

- `weight_method`: Method for calculating weights (default: 'inverse_correlation')
  - **'inverse_correlation'**: Use inverse correlation weights (current implementation)
  - **'linear'**: Use linear model to learn weights (future)
  - **'ml'**: Use ML model to learn weights (future)
- `fdm_max`: Maximum FDM value (default: 2.0, following Carver's recommendation)

### Implementation Notes

#### Pluggable Weight Methods

The Weight layer is designed to be **pluggable** - you can swap different combination methods:

```python
class WeightLayer:
    def __init__(self, weight_method: str = 'inverse_correlation'):
        self.weight_method = weight_method
        if weight_method == 'inverse_correlation':
            self.weighter = InverseCorrelationWeighter()
        elif weight_method == 'linear':
            self.weighter = LinearWeighter()
        elif weight_method == 'ml':
            self.weighter = MLWeighter()
        else:
            raise ValueError(f"Unknown weight method: {weight_method}")
```

#### Method Signatures

```python
def fit(
    self,
    forecast_vectors: List[pd.DataFrame],
    signals: pd.DataFrame  # Binary signals from all base models (for weight calculation)
) -> 'WeightLayer':
    """
    Fit weights and FDM from training data.
    
    Parameters
    ----------
    forecast_vectors : list[pd.DataFrame]
        List of forecast vectors from all ensembles (training data)
        Used for both weight calculation and FDM calculation
    signals : pd.DataFrame
        Binary signals from all base models (for inverse correlation weight calculation)
        Columns: model names, rows: samples
    
    Returns
    -------
    self
    """

def combine(
    self,
    forecast_vectors: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine forecasts from all base models.
    
    Parameters
    ----------
    forecast_vectors : list[pd.DataFrame]
        List of forecast vectors from all ensembles.
        Each DataFrame has columns: ['ticker', 'model_name', 'forecast', 'signal']
    
    Returns
    -------
    pd.DataFrame with columns: ['ticker', 'forecast_score']
    """
```

### Notes

- Weight layer combines **all base models from all ensembles** (not just within one ensemble)
- This solves signal dilution: if Ensemble A has 10 models and Ensemble B has 2 models, all 12 models are combined with equal consideration
- Weights are calculated based on correlations between **all base models**, not just within ensembles
- Pluggable design allows experimentation with different combination methods
- Inverse correlation is the default method, but linear models and ML models can be added later

---

## Layer 4: Portfolio

### Responsibility

**Minimal layer** that handles instrument weighting and optional position capping. Most risk management happens at Ensemble layer, and signal combination happens at Weight layer.

**Key Design**: Each Portfolio manages ONE trading timeframe only. For multi-timeframe trading, use multiple Portfolio instances.

### Current Implementation

⚠️ **Status**: Needs simplification (currently does too much - most logic moves to Ensemble and Weight layers)

### Input

From predict():
- `combined_forecasts`: DataFrame from Weight layer with columns `['ticker', 'forecast_score']`

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `forecast_score`: Combined forecast from Weight layer (passed through)
- `position_fraction`: Position fraction after instrument weighting and capping

### Architecture: Single-Timeframe Portfolio Design ⭐ NEW

**Key Principle**: Each Portfolio manages ONE trading timeframe only.

**Why Single-Timeframe?**
- **Features can be multi-timeframe** (e.g., weekly RSI used in daily strategy)
- **Trading is single-timeframe** (you trade daily OR weekly, not both simultaneously in same portfolio)
- **Cleaner separation**: Each Portfolio knows when it trades (its trading timeframe)
- **Real-world alignment**: Separate trading accounts = separate portfolios

**Multi-Timeframe Trading**: Use multiple Portfolio instances
- Daily Portfolio: `Portfolio(ensembles=[...], trading_timeframe=TimeFrame.D)`
- Weekly Portfolio: `Portfolio(ensembles=[...], trading_timeframe=TimeFrame.W)`
- MLManager routes candles to appropriate portfolio based on timeframe

### Architecture: Ensemble vs Instrument Weighting

**Key Principle**: Ensembles are **equal-weighted** within a Portfolio. Instruments can appear in **multiple ensembles** and are weighted at the portfolio level.

#### Why Instruments Appear in Multiple Ensembles

Different ensembles capture different market regimes or trading styles:

- **Example 1**: "Mean Reversion" ensemble contains only stock indices (e.g., RSI-2 works well for mean reversion on indices)
- **Example 2**: "Momentum" ensemble contains all asset classes (momentum works better when combining asset classes)

In this case, stock indices appear in **both** ensembles, allowing them to benefit from both mean reversion and momentum signals.

#### Ensemble Weighting Strategy

**Equal weight across ensembles** within a Portfolio to equate risk:
- Each ensemble gets equal representation in the final forecast
- This ensures no single ensemble dominates the portfolio
- Risk is equalized across different trading styles/regimes

#### Instrument Weighting Strategy

**Current Implementation**: Equal weight per instrument
- Each instrument gets equal weight: $w_{\text{instrument}} = \frac{1}{N_{\text{instruments}}}$

**Future Implementation** (Carver-style handcrafting):
- Top-down allocation by asset class → group → instrument
- Example: Equities (33%) → Indices (16.7%) → NQ (5.6%), ES (5.6%), YM (5.6%)
- Risk-agnostic weights (volatility scaling handles risk adjustment)

### Formula

#### Step 1: Use Combined Forecast from Weight Layer

The Weight layer has already combined all base model forecasts and applied FDM. We use this directly:

$$f_{\text{combined}} = F_{\text{combined}}$$

Where $F_{\text{combined}}$ is the combined forecast from Weight layer (already volatility-adjusted, weighted, and FDM-scaled).

#### Step 2: Apply Instrument Weight

$$\text{position}_{\text{weighted}} = f_{\text{combined}} \times w_{\text{instrument}}$$

Where $w_{\text{instrument}}$ is the allocation weight for this instrument.

**Current Implementation - Equal Weight Per Instrument**:

$$w_{\text{instrument}} = \frac{1}{N_{\text{instruments}}}$$

Where $N_{\text{instruments}}$ is the total number of unique instruments.

**Future Implementation - Carver-Style Handcrafting** (top-down allocation):

1. **Asset Class Level**: Divide 100% equally among asset classes
   - Equities: 33.3%
   - Commodities: 33.3%
   - FX: 33.3%

2. **Group Level**: Within each asset class, divide equally among groups
   - Equities → Indices: 16.7%, Sectors: 16.7%
   - Commodities → Energies: 11.1%, Metals: 11.1%, Agricultural: 11.1%

3. **Instrument Level**: Within each group, divide equally among instruments
   - Indices → NQ: 5.6%, ES: 5.6%, YM: 5.6%

**Note**: These weights are **risk-agnostic**. Volatility scaling already happened at Ensemble layer.

#### Step 3: Apply Instrument Diversification Multiplier (IDM)

After instrument weighting, apply IDM to account for portfolio-level diversification:

$$\text{position}_{\text{idm}} = \text{position}_{\text{weighted}} \times \text{IDM}$$

Where:
- $\text{IDM}$: Instrument Diversification Multiplier (calculated from instrument return correlations during `fit()`)
- IDM is **global** (applied to all instruments) because it's a portfolio-level property
- **Capped at 2.5** to prevent excessive leverage

**Calculation of IDM** (during `fit()`):
1. Get historical returns for all instruments in the portfolio
2. Build correlation matrix of instrument returns
3. Calculate mean correlation: $\bar{\rho} = \frac{1}{N(N-1)/2} \sum_{i<j} |\rho_{i,j}|$
4. Calculate IDM: $\text{IDM} = \sqrt{\frac{1}{\bar{\rho} + \epsilon}}$ (with small $\epsilon$ to avoid division by zero)
5. Cap at 2.5: $\text{IDM} = \min(\text{IDM}, 2.5)$
6. Floor negative correlations at zero (Carver's recommendation)

**Intuition**: When instruments are uncorrelated ($\bar{\rho} \approx 0$), IDM is high, allowing more leverage safely. When instruments are highly correlated ($\bar{\rho} \approx 1$), IDM is low (~1.0), indicating less diversification benefit.

#### Step 4: Cap Position (Optional)

$$\text{position}_{\text{final}} = \min(\text{position}_{\text{idm}}, \text{position}_{\text{max}})$$

Where $\text{position}_{\text{max}}$ is the maximum position size (e.g., 2.0 = 200% of average risk).

### Complete Formula

$$\boxed{
\text{position\_fraction} = \min\left(
  F_{\text{combined}} \times w_{\text{instrument}} \times \text{IDM}, 
  \text{position}_{\text{max}}
\right)
}$$

Where:
- $F_{\text{combined}}$: Combined forecast from Weight layer (already volatility-adjusted, weighted, and FDM-scaled)
- $w_{\text{instrument}}$: Instrument allocation weight
- $\text{IDM}$: Instrument Diversification Multiplier (portfolio-level)
- $\text{position}_{\text{max}}$: Maximum position size cap

### Key Parameters

- `max_position_pct`: Maximum position per instrument (default: 2.0, optional)
- `instrument_weights`: Dict mapping ticker → weight (default: equal per instrument)
  - **Current**: Equal weight: $w_{\text{instrument}} = \frac{1}{N_{\text{instruments}}}$
  - **Future**: Carver-style handcrafting by asset class → group → instrument
- `idm_max`: Maximum IDM value (default: 2.5, following Carver's recommendation)

**Note**: Risk management is distributed across layers:
- **Ensemble layer**: Volatility scaling per base model
- **Weight layer**: Forecast combination and FDM application
- **Portfolio layer**: Instrument weighting, IDM application, and optional capping

### Implementation Notes

#### What Changes from Current Code

**Add to Portfolio**:
- ✅ `target_volatility` parameter (default: 0.20)
- ✅ `dm` parameter (diversification multiplier, default: 1.0 initially, fit from backtests)
- ✅ `max_position_pct` parameter (default: 2.0)
- ✅ Volatility scaling logic (with √h adjustment)
- ✅ DM application (multiply position by DM after volatility scaling)
- ✅ Position capping

**Keep in Portfolio**:
- ✅ Forecast averaging across ensembles
- ✅ DataFrame output structure

**Remove from Portfolio**:
- ❌ Ensemble aggregation by timeframe (Portfolio now single-timeframe)
- ❌ `timeframe` parameter from predict() (Portfolio knows its trading timeframe)

#### Constructor

```python
def __init__(
    self,
    weight_layer: WeightLayer,  # NEW: Weight layer for combining forecasts
    trading_timeframe: TimeFrame,  # Explicit trading timeframe
    max_position_pct: Optional[float] = None,  # Optional position cap
    instrument_weights: Optional[Dict[str, float]] = None,  # Optional instrument weights
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
    instrument_weights : Dict[str, float], optional
        Weight for each instrument. If None, equal weight.
    idm_max : float, default=2.5
        Maximum IDM value (capped to prevent excessive leverage)
    """
```

#### Method Signatures

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

def predict(
    self,
    combined_forecasts: pd.DataFrame  # From Weight layer
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'position_fraction'])
    
    Parameters
    ----------
    combined_forecasts : pd.DataFrame
        Combined forecasts from Weight layer with columns:
        ['ticker', 'forecast_score']
        (Already FDM-scaled)
    """
```

### Notes

- **Minimal responsibility**: Portfolio handles instrument allocation, IDM application, and optional capping
- **Single-timeframe design**: Each Portfolio manages ONE trading timeframe only
  - For multi-timeframe trading, use multiple Portfolio instances
- **Instrument weighting**: Applies allocation weights across instruments (equal weight by default)
- **IDM application**: Applies Instrument Diversification Multiplier to account for portfolio-level diversification
- **Optional capping**: Can cap positions at max_position_pct if specified
- **No volatility scaling**: Volatility scaling happens at Ensemble layer
- **No FDM application**: FDM is applied at Weight layer
- **Clear separation**: 
  - Ensemble: Risk management per base model (volatility scaling)
  - Weight Layer: Signal combination and FDM application
  - Portfolio: Instrument allocation, IDM application, and capping

---

## Layer 5: Execution

### Responsibility

**Convert position fractions to tradeable contracts** based on contract specifications and capital.

### Current Implementation

🆕 **Status**: New component to be created

### Input

From calculate_positions():
- `position_fractions`: DataFrame with ticker and position_fraction
- `capital`: Account capital (in USD)
- `contract_specs`: Dictionary of contract specifications per ticker

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `forecast_score`: Original forecast score
- `position_fraction`: Target position fraction
- `contracts`: Number of contracts to trade (integer)
- `notional_value`: Actual notional value of position
- `target_dollars`: Target dollar allocation
- `notional_pct`: Notional value as % of capital

### Formula

#### Contract Value

For each instrument:

$$V_{\text{contract}} = P_{\text{instrument}} \times M_{\text{futures}} \times R_{\text{FX}}$$

Where:
- $P_{\text{instrument}}$: Current market price
- $M_{\text{futures}}$: Futures multiplier (e.g., $20/point for NQ)
- $R_{\text{FX}}$: FX conversion rate (e.g., GBP/USD = 1.27)

#### Target Dollar Allocation

$$D_{\text{target}} = \text{position\_fraction} \times \text{Capital}$$

#### Number of Contracts (Raw)

$$N_{\text{raw}} = \frac{D_{\text{target}}}{V_{\text{contract}}}$$

#### Number of Contracts (Rounded)

Apply rounding method:

$$N_{\text{final}} = 
\begin{cases}
\lfloor N_{\text{raw}} \rfloor & \text{if rounding='floor'} \\
\lceil N_{\text{raw}} \rceil & \text{if rounding='ceiling'} \\
\text{round}(N_{\text{raw}}) & \text{if rounding='round'}
\end{cases}$$

#### Actual Notional Value

$$V_{\text{notional}} = N_{\text{final}} \times V_{\text{contract}}$$

### Complete Formula (Carver-Style)

This is the formula from your image, now in context:

$$\boxed{
N_i = f_{i,t} \times \frac{\tau \times \text{Capital} \times w_i \times \text{DM}}{\text{Price}_i \times \text{Multiplier}_i \times \sigma\%_i \times \text{FX}_i}
}$$

**Our implementation separates this into**:
- Forecast generation: $f_{i,t}$ (volatility-adjusted per base model)
- Forecast combination: $F_{\text{combined}} = \left(\sum F_i \cdot w_i\right) \times \text{FDM}$ (at Weight layer)
- Position fraction: $F_{\text{combined}} \times w_{\text{instrument}} \times \text{IDM}$ (at Portfolio layer)
- Contract sizing: $\frac{\text{position\_fraction} \times \text{Capital}}{\text{Price}_i \times \text{Multiplier}_i \times \text{FX}_i}$

### Key Parameters

- `capital`: Account capital in base currency
- `rounding_method`: 'round', 'floor', or 'ceiling'
- `contract_specs`: Per-instrument specifications

### Contract Specification Structure

```python
@dataclass(frozen=True)
class ContractSpec:
    ticker: str
    price: float          # Current market price
    multiplier: float     # Points to dollars (e.g., $20 for NQ)
    fx_rate: float = 1.0  # FX conversion (default USD/USD = 1.0)
    min_tick: float = 0.25  # Minimum price increment
```

### Implementation Notes

#### New Module Structure

Create `execution/position_sizer.py`:
- `ContractSpec` dataclass
- `Position` dataclass
- `PositionSizer` class

#### Rounding Strategy Considerations

- **'round'**: Most balanced, recommended for live trading
- **'floor'**: Conservative, never exceeds target
- **'ceiling'**: Aggressive, always takes minimum position

#### Handling Zero Contracts

If $N_{\text{final}} = 0$:
- Log the missed trade
- Track slippage from target
- Consider capital requirements vs instrument choice

### Notes

- Separates **what to trade** (Portfolio) from **how to execute** (Sizer)
- Contract specs can be updated independently
- Makes backtesting vs live trading transparent (same logic, different specs)

---

## Complete Workflow Example

### Scenario Setup

**Portfolio Configuration**:
- 2 Ensembles: Mean Reversion, Momentum (equal weighted)
- Instruments:
  - **Mean Reversion ensemble**: NQ, ES, YM (stock indices only)
  - **Momentum ensemble**: NQ, ES, GC, CL (all asset classes)
  - **Note**: NQ and ES appear in **both** ensembles
- Each ensemble has 10 base models with $n_{\text{bins}} = 10$ (so $h_i \approx 0.1$)

**Parameters**:
- Capital: $1,000,000 
- Target Volatility ($\tau$): 20% annual
- FDM: 1.5 (calculated from forecast correlations, capped at 2.0)
- IDM: 2.0 (calculated from instrument return correlations, capped at 2.5)
- Instrument weighting: Equal (5 unique instruments: NQ, ES, YM, GC, CL)

**Note**: This example uses FDM=1.5 and IDM=2.0 to demonstrate the full workflow. In initial implementation, these are calculated from correlation matrices during `fit()`.

**Market Conditions** (for this example):
- NQ: Strong mean reversion signal, weak momentum
- ES: Strong mean reversion signal, moderate momentum
- YM: Moderate mean reversion signal, no momentum (not in Momentum ensemble)
- GC: No mean reversion (not in Mean Reversion ensemble), strong momentum
- CL: No mean reversion (not in Mean Reversion ensemble), moderate momentum

### Step-by-Step Calculation for All Instruments

We'll calculate positions for all 5 instruments, showing how the system handles instruments in single vs multiple ensembles.

---

#### Instrument 1: NQ (Nasdaq-100 E-mini)

**Appears in**: Mean Reversion + Momentum ensembles

##### Layer 1: Base Models → Binary Signals

**Mean Reversion Ensemble** (10 base models):
```
signals_MR = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0]  # 4/10 active
```

**Momentum Ensemble** (10 base models):
```
signals_MOM = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]  # 2/10 active
```

##### Layer 2: Ensemble → Forecast Vectors

**Mean Reversion Ensemble** (10 base models, 4 active):
- Each active model gets volatility-adjusted forecast: $F_i = \frac{0.20}{0.25 \times \sqrt{0.1}} \approx 2.53$
- **Ensemble output**: Vector of 10 forecasts (4 active with value 2.53, 6 inactive with value 0)

**Momentum Ensemble** (10 base models, 2 active):
- Each active model gets volatility-adjusted forecast: $F_i = \frac{0.20}{0.25 \times \sqrt{0.1}} \approx 2.53$
- **Ensemble output**: Vector of 10 forecasts (2 active with value 2.53, 8 inactive with value 0)

##### Layer 3: Weight Layer → Combined Forecast

**Step 1: Collect all base model forecasts** (20 total: 10 from MR + 10 from MOM)

**Step 2: Apply inverse correlation weights**:
- Weights calculated from signal correlations during `fit()`
- Weighted sum: $F_{\text{weighted}} = \sum_{i=1}^{20} F_i \cdot w_i \approx 1.52$ (assuming equal weights for simplicity)

**Step 3: Apply FDM**:
- FDM = 1.5 (calculated from forecast value correlations, capped at 2.0)
- $F_{\text{combined}} = 1.52 \times 1.5 = 2.28$

##### Layer 4: Portfolio → Position Fraction

**Step 1: Use combined forecast from Weight layer**:
$$f_{\text{combined}} = 2.28$$

**Step 2: Apply instrument weight** (5 unique instruments, equal weight):
$$w_{\text{NQ}} = \frac{1}{5} = 0.20$$
$$\text{position}_{\text{weighted}} = 2.28 \times 0.20 = 0.456$$

**Step 3: Apply IDM**:
- IDM = 2.0 (calculated from instrument return correlations, capped at 2.5)
- $\text{position}_{\text{idm}} = 0.456 \times 2.0 = 0.912$

**Step 4: Cap position** (max = 2.0, already below):
$$\text{position}_{\text{final}} = 0.912$$

##### Layer 4: Execution → Number of Contracts

**NQ contract specs**:
- Price: $16,000
- Multiplier: $20/point
- FX rate: 1.0

**Contract value**:
$$V_{\text{contract}} = 16{,}000 \times 20 \times 1.0 = \$320{,}000$$

**Target allocation**:
$$D_{\text{target}} = 0.912 \times 1{,}000{,}000 = \$912{,}000$$

**Raw contracts**:
$$N_{\text{raw}} = \frac{912{,}000}{320{,}000} = 2.85$$

**Rounded contracts**:
$$N_{\text{final}} = \text{round}(2.85) = 3$$

**Actual notional**:
$$V_{\text{notional}} = 3 \times 320{,}000 = \$960{,}000$$

**Result**: **3 contracts** (notional: $960k, 96% of capital)

---

#### Instrument 2: ES (S&P 500 E-mini)

**Appears in**: Mean Reversion + Momentum ensembles

##### Layer 2: Ensemble → Forecast Score

**Mean Reversion Ensemble**:
- 5/10 models active
- Raw forecast: $F_{\text{raw,MR}} = 0.58$
- **Ensemble output**: $F_{\text{MR}} = 0.58$

**Momentum Ensemble**:
- 3/10 models active
- Raw forecast: $F_{\text{raw,MOM}} = 0.35$
- **Ensemble output**: $F_{\text{MOM}} = 0.35$

##### Layer 3: Portfolio → Position Fraction

**Average forecast**:
$$\bar{F}_{\text{ES}} = \frac{0.58 + 0.35}{2} = 0.465$$

**Use raw forecast directly**:
$$f_{\text{norm}} = \bar{F}_{\text{ES}} = 0.465$$

**ES parameters**:
- Blended volatility: $\sigma_{\text{blended,ES}} = 0.22$ (22% annualized)
- Average exposure: $\bar{h} = 0.1$

**Position calculation**:
$$\text{position}_{\text{final}} = \min\left(0.465 \times \frac{0.20}{0.22 \times \sqrt{0.1}} \times 3.0 \times 0.20, 2.0\right) = \min(0.58, 2.0) = 0.58$$

##### Layer 4: Execution → Number of Contracts

**ES contract specs**:
- Price: $4,800
- Multiplier: $50/point
- FX rate: 1.0

**Contract value**: $V_{\text{contract}} = 4{,}800 \times 50 = \$240{,}000$

**Target allocation**: $D_{\text{target}} = 0.58 \times 1{,}000{,}000 = \$580{,}000$

**Contracts**: $N = \text{round}(\frac{580{,}000}{240{,}000}) = \text{round}(2.42) = 2$

**Notional**: $V_{\text{notional}} = 2 \times 240{,}000 = \$480{,}000$

**Result**: **2 contracts** (notional: $480k, 48% of capital)

---

#### Instrument 3: YM (Dow Jones E-mini)

**Appears in**: Mean Reversion only (not in Momentum ensemble)

##### Layer 2: Ensemble → Forecast Score

**Mean Reversion Ensemble**:
- 3/10 models active
- Raw forecast: $F_{\text{raw,MR}} = 0.32$
- **Ensemble output**: $F_{\text{MR}} = 0.32$

**Momentum Ensemble**: N/A (YM not in this ensemble)

##### Layer 3: Portfolio → Position Fraction

**Average forecast** (only 1 ensemble):
$$\bar{F}_{\text{YM}} = 0.32$$

**Use raw forecast directly**:
$$f_{\text{norm}} = \bar{F}_{\text{YM}} = 0.32$$

**YM parameters**:
- Blended volatility: $\sigma_{\text{blended,YM}} = 0.20$ (20% annualized)
- Average exposure: $\bar{h} = 0.1$

**Position calculation**:
$$\text{position}_{\text{final}} = \min\left(0.32 \times \frac{0.20}{0.20 \times \sqrt{0.1}} \times 3.0 \times 0.20, 2.0\right) = \min(0.48, 2.0) = 0.48$$

##### Layer 4: Execution → Number of Contracts

**YM contract specs**:
- Price: $38,000
- Multiplier: $5/point
- FX rate: 1.0

**Contract value**: $V_{\text{contract}} = 38{,}000 \times 5 = \$190{,}000$

**Target allocation**: $D_{\text{target}} = 0.48 \times 1{,}000{,}000 = \$480{,}000$

**Contracts**: $N = \text{round}(\frac{480{,}000}{190{,}000}) = \text{round}(2.53) = 3$

**Notional**: $V_{\text{notional}} = 3 \times 190{,}000 = \$570{,}000$

**Result**: **3 contracts** (notional: $570k, 57% of capital)

---

#### Instrument 4: GC (Gold)

**Appears in**: Momentum only (not in Mean Reversion ensemble)

##### Layer 2: Ensemble → Forecast Score

**Momentum Ensemble**:
- 6/10 models active (strong momentum signal)
- Raw forecast: $F_{\text{raw,MOM}} = 0.65$
- **Ensemble output**: $F_{\text{MOM}} = 0.65$

**Mean Reversion Ensemble**: N/A (GC not in this ensemble)

##### Layer 3: Portfolio → Position Fraction

**Average forecast** (only 1 ensemble):
$$\bar{F}_{\text{GC}} = 0.65$$

**Use raw forecast directly**:
$$f_{\text{norm}} = \bar{F}_{\text{GC}} = 0.65$$

**GC parameters**:
- Blended volatility: $\sigma_{\text{blended,GC}} = 0.18$ (18% annualized)
- Average exposure: $\bar{h} = 0.1$

**Position calculation**:
$$\text{position}_{\text{final}} = \min\left(0.65 \times \frac{0.20}{0.18 \times \sqrt{0.1}} \times 3.0 \times 0.20, 2.0\right) = \min(0.77, 2.0) = 0.77$$

##### Layer 4: Execution → Number of Contracts

**GC contract specs**:
- Price: $2,000/oz
- Multiplier: $100/oz
- FX rate: 1.0

**Contract value**: $V_{\text{contract}} = 2{,}000 \times 100 = \$200{,}000$

**Target allocation**: $D_{\text{target}} = 0.77 \times 1{,}000{,}000 = \$770{,}000$

**Contracts**: $N = \text{round}(\frac{770{,}000}{200{,}000}) = \text{round}(3.85) = 4$

**Notional**: $V_{\text{notional}} = 4 \times 200{,}000 = \$800{,}000$

**Result**: **4 contracts** (notional: $800k, 80% of capital)

---

#### Instrument 5: CL (Crude Oil)

**Appears in**: Momentum only (not in Mean Reversion ensemble)

##### Layer 2: Ensemble → Forecast Score

**Momentum Ensemble**:
- 4/10 models active
- Raw forecast: $F_{\text{raw,MOM}} = 0.42$
- **Ensemble output**: $F_{\text{MOM}} = 0.42$

##### Layer 3: Portfolio → Position Fraction

**Average forecast**: $\bar{F}_{\text{CL}} = 0.42$

**Use raw forecast directly**:
$$f_{\text{norm}} = \bar{F}_{\text{CL}} = 0.42$$

**CL parameters**:
- Blended volatility: $\sigma_{\text{blended,CL}} = 0.30$ (30% annualized - high volatility)
- Average exposure: $\bar{h} = 0.1$

**Position calculation**:
$$\text{position}_{\text{final}} = \min\left(0.42 \times \frac{0.20}{0.30 \times \sqrt{0.1}} \times 3.0 \times 0.20, 2.0\right) = \min(0.42, 2.0) = 0.42$$

##### Layer 4: Execution → Number of Contracts

**CL contract specs**:
- Price: $75/barrel
- Multiplier: $1,000/barrel
- FX rate: 1.0

**Contract value**: $V_{\text{contract}} = 75 \times 1{,}000 = \$75{,}000$

**Target allocation**: $D_{\text{target}} = 0.42 \times 1{,}000{,}000 = \$420{,}000$

**Contracts**: $N = \text{round}(\frac{420{,}000}{75{,}000}) = \text{round}(5.6) = 6$

**Notional**: $V_{\text{notional}} = 6 \times 75{,}000 = \$450{,}000$

**Result**: **6 contracts** (notional: $450k, 45% of capital)

### Summary for All Instruments

| Ticker | Ensembles | Forecast | Position % | Target $ | Contracts | Notional $ | Notional % | Notes |
|--------|-----------|----------|-----------|----------|-----------|------------|------------|-------|
| NQ | MR + MOM | 0.395 | 60% | $600k | 2 | $640k | 64% | Appears in 2 ensembles, averaged |
| ES | MR + MOM | 0.465 | 58% | $580k | 2 | $480k | 48% | Appears in 2 ensembles, averaged |
| YM | MR only | 0.32 | 48% | $480k | 3 | $570k | 57% | Single ensemble, lower forecast |
| GC | MOM only | 0.65 | 77% | $770k | 4 | $800k | 80% | Strong momentum, single ensemble |
| CL | MOM only | 0.42 | 42% | $420k | 6 | $450k | 45% | High volatility reduces position |

**Key Observations**:

1. **Ensemble Averaging**: NQ and ES have lower final forecasts because they average signals from both ensembles. This prevents over-weighting.

2. **Single Ensemble Instruments**: YM, GC, and CL only appear in one ensemble, so their forecast equals that ensemble's output (no averaging).

3. **Volatility Impact**: CL has a moderate forecast (0.42) but lower position % (42%) due to high volatility (30% vs others at 18-25%).

4. **Contract Sizing**: All instruments now trade non-zero contracts with $1M capital.

5. **Notional vs Target**: Actual notional values differ from target allocations due to contract rounding:
   - NQ: Target $600k → Notional $640k (rounding up from 1.875 contracts)
   - ES: Target $580k → Notional $480k (rounding down from 2.42 contracts)
   - YM: Target $480k → Notional $570k (rounding up from 2.53 contracts)
   - GC: Target $770k → Notional $800k (rounding up from 3.85 contracts)
   - CL: Target $420k → Notional $450k (rounding up from 5.6 contracts)

6. **Total Portfolio Exposure**: 
   - Total notional: $2,940,000 (294% of capital)
   - This is normal for futures trading where notional can exceed capital due to margin requirements
   - Actual risk is controlled by volatility scaling, not notional value

7. **Diversification**: The system successfully combines:
   - Different asset classes (equities, commodities)
   - Different trading styles (mean reversion, momentum)
   - Different volatility profiles (18% to 30%)
   - All while maintaining equal risk weighting through volatility scaling

### Portfolio-Level Analysis

**Risk Allocation by Instrument**:
- Each instrument's position % represents its contribution to portfolio risk
- Total position %: $0.60 + 0.58 + 0.48 + 0.77 + 0.42 = 2.85$ (285% of capital)
- This exceeds 100% because we're using leverage (futures contracts)
- Actual portfolio risk is controlled by the target volatility ($\tau = 20\%$)

**Ensemble Contribution**:
- **Mean Reversion Ensemble** contributes to: NQ, ES, YM
- **Momentum Ensemble** contributes to: NQ, ES, GC, CL
- Equal weighting ensures neither ensemble dominates

**Volatility Scaling in Action**:
- CL has highest volatility (30%) → smallest position % (42%) despite moderate forecast
- GC has lowest volatility (18%) → largest position % (77%) despite strong forecast
- This demonstrates how volatility scaling equalizes risk across instruments

**Contract Rounding Impact**:
- Rounding creates small deviations from target allocations
- Total target: $2,850,000
- Total notional: $2,940,000
- Difference: $90,000 (9% over-allocation)
- This is acceptable and can be managed through rebalancing

---

## Practical Guide: Calculating FDM and IDM

### Forecast Diversification Multiplier (FDM)

FDM is calculated from forecast value correlations during Weight layer `fit()`:

1. **Extract forecast values** from training data for all base models
2. **Build correlation matrix** of forecast values (not binary signals)
3. **Calculate mean correlation**: $\bar{\rho} = \frac{1}{N(N-1)/2} \sum_{i<j} |\rho_{i,j}|$
4. **Calculate FDM**: $\text{FDM} = \sqrt{\frac{1}{\bar{\rho} + 0.01}}$ (with small epsilon)
5. **Cap at 2.0**: $\text{FDM} = \min(\text{FDM}, 2.0)$
6. **Floor negative correlations at zero** (Carver's recommendation)

**Typical Values**:
- 2 base models: ~1.02 FDM
- 6 base models: ~1.27 FDM
- 30 base models: ~1.81 FDM

### Instrument Diversification Multiplier (IDM)

IDM is calculated from instrument return correlations during Portfolio `fit()`:

1. **Get historical returns** for all instruments in the portfolio
2. **Build correlation matrix** of instrument returns
3. **Calculate mean correlation**: $\bar{\rho} = \frac{1}{N(N-1)/2} \sum_{i<j} |\rho_{i,j}|$
4. **Calculate IDM**: $\text{IDM} = \sqrt{\frac{1}{\bar{\rho} + 0.01}}$ (with small epsilon)
5. **Cap at 2.5**: $\text{IDM} = \min(\text{IDM}, 2.5)$
6. **Floor negative correlations at zero** (Carver's recommendation)

**Typical Values**:
- 2 instruments: ~1.20 IDM
- 10 instruments: ~2.20 IDM
- 30+ instruments: ~2.50 IDM (capped)

### Alternative: Simple Fitting from Past Performance (Legacy)

For initial validation, you can still use the simple fitting approach:

**Note**: This approach is simpler but less principled than correlation-based FDM/IDM. Recommended for initial validation only.

---

## Unit Testing Guide: Buy/Hold Scenarios

### Why Buy/Hold Testing?

Buy/hold scenarios provide **simple, verifiable test cases** where:
- Forecast = 1.0 (maximum signal strength)
- Exposure fraction = 1.0 (always in market)
- Expected behavior is clear and easy to validate

This makes it much easier to test the position sizing logic without complex signal generation.

### Test Scenarios

#### Scenario 1: Perfect Buy/Hold (Baseline)

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$ (maximum signal)
- Exposure: $\bar{h} = 1.0$ (always in market)
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- FDM: $1.0$ (no forecast diversification scaling)
- IDM: $1.0$ (no instrument diversification scaling)
- Instrument weight: $w = 1.0$ (single instrument)

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{1.0}} \times 1.0 \times 1.0 \times 1.0 = 1.0$$

**Verification**: Position fraction = 1.0 (100% of capital allocated)

**Why this works**: When target volatility equals instrument volatility and we're always in the market, we should allocate 100% of capital.

#### Scenario 2: High Volatility Instrument

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.40$ (40% - twice the target)
- FDM: $1.0$
- IDM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.40 \times \sqrt{1.0}} \times 1.0 \times 1.0 \times 1.0 = 0.5$$

**Verification**: Position fraction = 0.5 (50% of capital allocated)

**Why this works**: High volatility instrument requires smaller position to maintain target risk.

#### Scenario 3: Low Volatility Instrument

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.10$ (10% - half the target)
- FDM: $1.0$
- IDM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.10 \times \sqrt{1.0}} \times 1.0 \times 1.0 \times 1.0 = 2.0$$

**Verification**: Position fraction = 2.0 (200% of capital - leverage)

**Why this works**: Low volatility instrument allows larger position (with leverage) to maintain target risk.

#### Scenario 4: Forecast Diversification Multiplier (FDM) Effect

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$ (after combination)
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- FDM: $1.5$ (forecast diversification scaling)
- IDM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{1.0}} \times 1.5 \times 1.0 \times 1.0 = 1.5$$

**Verification**: Position fraction = 1.5 (150% of capital - scaled by FDM)

**Why this works**: FDM increases forecast strength to account for diversification when combining multiple base model forecasts.

#### Scenario 4b: Instrument Diversification Multiplier (IDM) Effect

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$ (after FDM scaling)
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- FDM: $1.0$
- IDM: $2.0$ (instrument diversification scaling)
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{1.0}} \times 1.0 \times 2.0 \times 1.0 = 2.0$$

**Verification**: Position fraction = 2.0 (200% of capital - scaled by IDM)

**Why this works**: IDM increases position size to account for portfolio-level diversification benefits.

#### Scenario 5: Sparse Signals (Low Exposure)

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$ (maximum signal when active)
- Exposure: $\bar{h} = 0.1$ (only in market 10% of time)
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- FDM: $1.0$
- IDM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{0.1}} \times 1.0 \times 1.0 \times 1.0 = 1.0 \times \frac{0.20}{0.0632} = 3.16$$

**Verification**: Position fraction = 3.16 (316% of capital - leverage)

**Why this works**: Sparse signals have lower realized volatility ($\sqrt{0.1} \approx 0.316$), so we can take larger positions when active.

#### Scenario 6: Position Capping

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.05$ (5% - very low)
- FDM: $1.0$
- IDM: $1.0$
- Instrument weight: $w = 1.0$
- Max position: $2.0$ (200% cap)

**Uncapped Position**:
$$\text{position}_{\text{uncapped}} = 1.0 \times \frac{0.20}{0.05 \times \sqrt{1.0}} \times 1.0 \times 1.0 \times 1.0 = 4.0$$

**Expected Position** (after capping):
$$\text{position} = \min(4.0, 2.0) = 2.0$$

**Verification**: Position fraction = 2.0 (capped at maximum)

**Why this works**: Position capping prevents excessive leverage even when volatility is very low.

### Test Implementation

**Unit Test Structure**:

```python
def test_buy_hold_baseline():
    """Test perfect buy/hold scenario with matching volatilities"""
    forecast = 1.0
    exposure = 1.0
    target_vol = 0.20
    instrument_vol = 0.20
    dm = 1.0
    instrument_weight = 1.0
    
    position = calculate_position(
        forecast=forecast,
        exposure=exposure,
        target_vol=target_vol,
        instrument_vol=instrument_vol,
        fdm=fdm,
        idm=idm,
        instrument_weight=instrument_weight
    )
    
    assert position == 1.0, f"Expected 1.0, got {position}"

def test_buy_hold_high_volatility():
    """Test buy/hold with high volatility instrument"""
    # ... similar structure
    assert position == 0.5

def test_buy_hold_low_volatility():
    """Test buy/hold with low volatility instrument"""
    # ... similar structure
    assert position == 2.0

# ... additional test cases
```

### Benefits of Buy/Hold Testing

1. **Simple Verification**: Easy to calculate expected values by hand
2. **Clear Intent**: Tests focus on position sizing logic, not signal generation
3. **Edge Case Coverage**: Tests extreme scenarios (high/low volatility, capping)
4. **Regression Prevention**: Catches bugs in formula implementation
5. **Documentation**: Tests serve as executable documentation of expected behavior

### Integration with Full System Tests

Buy/hold tests validate the **core position sizing formula**. Full system tests (with actual signals) validate:
- Signal generation and combination
- Ensemble averaging
- End-to-end workflow
- Contract rounding
- Notional value calculations

---

## Implementation

**For concrete implementation tasks, method signatures, unit tests, and step-by-step build instructions, see:**

📋 **[implementation_specs.md](./implementation_specs.md)**

The implementation document contains:
- Phase-by-phase refactoring tasks
- Method signatures (before/after)
- Unit test templates with buy/hold scenarios
- Code examples for all components
- Estimated effort per phase

**Summary**: ~20-28 hours of implementation work across 6 phases:
1. Refactor Ensemble (2-3 hours)
2. Enhance Portfolio (6-8 hours)
3. Create Execution Layer (3-4 hours)
4. Create Volatility Utilities (3-4 hours)
5. Integration & Testing (4-6 hours)
6. DM Fitting Utility (2-3 hours)

---

## References

1. **Robert Carver** - *Systematic Trading* (2015)
2. **Robert Carver** - *Leveraged Trading* (2019)
3. **Robert Carver** - Blog: https://qoppac.blogspot.com/
4. Our codebase:
   - `feature_selection/base_models/base_model.py`
   - `ensemble/diversified_ensemble.py`
   - `ensemble/portfolio.py`

---

## Appendix: Mathematical Symbols

| Symbol | Name | Meaning | Units |
|--------|------|---------|-------|
| $X_i$ | Signal | Binary signal from base model $i$ | {0, 1} |
| $w_i$ | Feature weight | Diversification weight for model $i$ | dimensionless |
| $h_i$ | Exposure fraction | Fraction of time model $i$ is in market | [0, 1] |
| $F$ | Forecast | Forecast score | dimensionless |
| $\tau$ | Target volatility | Annual portfolio volatility target | annualized % |
| $\sigma_{\text{blended}}$ | Instrument volatility | Blended annualized volatility (70% EWMA-32 + 30% 10-year) | annualized % |
| $\text{DM}$ | Diversification Multiplier | Accounts for strategy and instrument diversification | dimensionless |
| $N$ | Contracts | Number of contracts to trade | integer |
| $P$ | Price | Current market price | currency |
| $M$ | Multiplier | Futures contract multiplier | currency/point |
| $R$ | FX rate | Foreign exchange rate | dimensionless |

---

**End of Specification Document**
