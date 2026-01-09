# Forecast Generation & Position Sizing Architecture

> **📖 This Document**: Context, design principles, formulas, and examples  
> **🔨 For Implementation**: See [implementation_specs.md](./implementation_specs.md)

**Version**: 4.0.0  
**Date**: 2025-01-08  
**Status**: Specification (Context & Design)  
**Implementation**: See `implementation_specs.md`

**Major Changes**:
- **v3.0.0**: Moved all diversification, scaling, and capping to Portfolio layer. Ensembles now only combine signals using inverse correlation weights and return raw forecasts [0, 1].
- **v3.1.0**: Combined FDM and IDM into a single Diversification Multiplier (DM). Simple fitting from past performance: $\text{DM} = \frac{\text{target volatility}}{\text{realized volatility}}$. Trivial fitting approach won't overfit.
- **v3.2.0**: Removed forecast scalar (redundant - multiply by 10 then divide by 10). Raw forecasts in [0, 1] are used directly as multipliers. Removed forecast capping (max raw forecast is 1.0, so no capping needed).
- **v3.3.0**: Updated to use Carver's blended volatility estimate (70% EWMA-32 + 30% 10-year average) instead of simple EWSD-30. Added comprehensive unit testing guide with buy/hold scenarios for easier validation.
- **v4.0.0**: Split into two documents: `forecast_specs.md` (context, design, formulas, examples) and `implementation_specs.md` (concrete build tasks, method signatures, unit tests). Better organization for understanding vs. building.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer 1: Base Models](#layer-1-base-models)
3. [Layer 2: Ensemble](#layer-2-ensemble)
4. [Layer 3: Portfolio](#layer-3-portfolio)
5. [Layer 4: Execution](#layer-4-execution)
6. [Complete Workflow Example](#complete-workflow-example)
7. [Practical Guide: Tuning Diversification Multiplier (DM)](#practical-guide-tuning-diversification-multiplier-dm)
8. [Unit Testing Guide: Buy/Hold Scenarios](#unit-testing-guide-buyhold-scenarios)

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
│  Layer 2: Ensemble                                           │
│  ├─ Input:  Binary signals from base models                 │
│  └─ Output: Raw forecast scores [0, 1], exposure fraction   │
│                                                              │
│  Layer 3: Portfolio (Risk Management Hub)                    │
│  ├─ Input:  Raw forecasts, volatility (blended)            │
│  ├─ Process: Vol-scale, DM                                 │
│  └─ Output: Position fractions (% of capital)               │
│                                                              │
│  Layer 4: Execution                                          │
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
6. **Portfolio as Risk Hub**: All standardization, scaling, and diversification happens at portfolio level
   - **Why**: Ensembles already use inverse correlation weights for base model diversification
   - **Benefit**: Cleaner separation - ensembles combine signals, portfolio manages risk
   - **Simplicity**: Single point of control for all risk parameters (DM, caps)

### Key Architectural Decision: Diversification at Portfolio Level

**Why move diversification and capping to Portfolio?**

1. **Inverse correlation weights already handle base model diversification**
   - Ensemble layer calculates sophisticated weights based on signal correlations
   - These weights naturally boost diversification benefits
   - Adding diversification scaling at ensemble level would be redundant

2. **Cleaner separation of concerns**
   - **Ensemble**: "Which signals should I combine?"
   - **Portfolio**: "How much risk should I take?"
   
3. **Single point of control**
   - All risk parameters (DM, position_cap) managed in one place
   - Easier to tune and backtest
   - Clearer for ensemble averaging (raw [0,1] range is natural to average)

4. **Matches Carver's philosophy**
   - Carver applies diversification multipliers after combining forecasts, not during signal generation
   - Our portfolio layer is analogous to Carver's position sizing system

### Diversification Multiplier (DM)

**Purpose**: The Diversification Multiplier (DM) is a **scaling factor** that increases position sizes to account for diversification benefits, allowing us to hit our target portfolio volatility.

**The Problem Without It**:
- If we treat all strategies and instruments as if they were perfectly correlated, we under-allocate capital
- Example: 10 uncorrelated strategies across 5 instruments have much lower portfolio volatility than if they were perfectly correlated
- Without scaling up, we'd miss our target volatility (e.g., target 20%, realize only 10%)

**The Solution**:
- **DM** accounts for diversification across:
  - Base models within ensembles (inverse correlation weights already help)
  - Multiple ensembles (different trading styles)
  - Different instruments (NQ, ES, GC, CL)
  - Different asset classes (equities, commodities, FX)
  - Low correlation between strategies and instruments

**Why a Single Multiplier?**
- **Simpler**: One parameter instead of two (FDM and IDM)
- **Same goal**: Both strategy and instrument diversification reduce portfolio volatility
- **Easier to tune**: Fit directly from past performance
- **Less overfitting risk**: Single parameter is less prone to overfitting than multiple correlated parameters

**Implementation Strategy** (Simple Fitting from Past Performance):

1. **Phase 1: Start Conservative** (Initial)
   - DM = 1.0 (no scaling)
   - Run backtests and measure realized volatility
   - Likely outcome: Under-allocation (realized vol < target vol)

2. **Phase 2: Simple Fitting** (Recommended)
   - Calculate optimal DM from backtest results:
     $$\text{DM}_{\text{optimal}} = \frac{\text{target volatility}}{\text{realized volatility}}$$
   - This is a trivial form of fitting that won't overfit
   - Example: Target 20%, Realized 12% → DM = 1.67
   - Re-run backtests with fitted DM to validate

3. **Phase 3: Refinement** (Optional)
   - Monitor realized volatility over time
   - Adjust DM if portfolio composition changes significantly
   - Can use rolling window to adapt to changing market conditions

**Key Insight**: This is a **simple tuning parameter** that can be directly estimated from past performance. The fitting is trivial (one division) and won't overfit because it's just scaling to match a target.

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
- $\sigma_{\text{short}}$: **Short-run estimate** = Exponentially Weighted Moving Average (EWMA) with **32-day span**
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

**Combine base model signals** into a standardized forecast score that represents signal strength.

### Current Implementation

⚠️ **Status**: Needs refactoring (currently does volatility scaling, which should be at Portfolio layer)

### Input

From predict():
- `X`: DataFrame with raw features
- `ticker`: Instrument ticker symbols
- `normalization_data`: Optional normalization data for base models

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `forecast_score`: Raw combined forecast (unitless, avg ≈ $\bar{h}$, range [0, 1])
- `exposure_fraction`: Average exposure fraction $\bar{h}$

### Formula

#### Step 1: Combine Binary Signals

For each instrument at time $t$:

$$F_{\text{raw}} = \sum_{i=1}^{N_{\text{models}}} X_{i,t} \cdot w_i$$

Where:
- $X_{i,t} \in \{0, 1\}$: Binary signal from base model $i$ at time $t$
- $w_i$: Diversification weight for model $i$
- $\sum_{i=1}^{N} w_i = 1$: Weights sum to 1

#### Step 2: Calculate Diversification Weights

Weights are based on **inverse correlation** between models (sophisticated approach that automatically adapts to signal correlations):

$$w_i = \frac{d_i}{\sum_{j=1}^{N} d_j}$$

Where diversification score:

$$d_i = \frac{1}{1 + \bar{\rho}_i}$$

And $\bar{\rho}_i$ is the average absolute correlation between model $i$ and all other models.

**Calculation method**:
1. Build correlation matrix of binary signals $X_i$ over training period
2. For each model $i$, calculate average absolute correlation with all other models
3. Convert to diversification score (inverse relationship)
4. Normalize to sum to 1.0

**Key Insight**: This inverse correlation weighting **already accounts for diversification** at the base model level. Models with lower correlation to others get higher weights, naturally boosting diversification benefits.

**Note**: This approach is calculated during `fit()` and stored for use in `predict()`.

#### Step 3: Calculate Average Exposure

**Note**: This step is **our addition** to Carver's framework. Carver doesn't explicitly track exposure fraction during forecast generation, but we calculate it here for use in volatility scaling at the Portfolio layer.

For active signals only:

$$\bar{h}_t = \frac{1}{N_{\text{active}}} \sum_{i: X_{i,t}=1} h_i$$

Where $h_i$ is the exposure fraction for model $i$ (fraction of time in market).

**Why we track this**: When we later apply volatility scaling at the Portfolio layer, we need to account for the fact that sparse signals (low $h_i$) have lower realized volatility, requiring position size adjustment via $\sqrt{h}$.

#### Step 4: Return Forecast Score (No Scaling, No Capping)

$$F_{\text{ensemble}} = F_{\text{raw}}$$

**Key Design Decision**: We do **NOT** apply any scaling, diversification multiplier, or capping at the ensemble layer. The raw forecast is passed directly to the Portfolio layer.

**Why**:
- **Weights sum to 1.0**: Since $\sum w_i = 1$, the raw forecast $F_{\text{raw}} \in [0, 1]$ with average ≈ $\bar{h}$ (exposure fraction)
- **Diversification already handled**: Inverse correlation weights already account for base model diversification
- **Cleaner separation**: Portfolio layer handles all scaling and diversification multipliers
- **Ensemble averaging**: Averaging forecasts across ensembles at portfolio level works naturally with raw [0,1] range
- **No scaling needed**: Raw forecasts can be used directly as multipliers for position sizing

### Key Parameters

- `target_volatility`: **REMOVED** (moves to Portfolio layer)
- `dm`: **REMOVED** (moves to Portfolio layer)
- **Note**: No scaling or capping needed - raw forecasts in [0, 1] are used directly

### Implementation Notes

#### What Changes from Current Code

**Remove from Ensemble**:
- ❌ Volatility parameter in predict()
- ❌ Division by instrument volatility
- ❌ Multiplication by instrument weights
- ❌ Target volatility usage
- ❌ Diversification multiplier calculation and application
- **Note**: No scaling or capping needed - raw forecasts in [0, 1] are used directly

**Keep in Ensemble**:
- ✅ Inverse correlation weight calculation during `fit()`
- ✅ Binary signal combination using weights
- ✅ Exposure fraction tracking

**Add to Ensemble**:
- ✅ Return raw forecast (no scaling) with exposure fraction

#### Method Signatures

```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'exposure_fraction'])
    
    Where:
    - forecast_score: Raw combined signal in [0, 1] range
    - exposure_fraction: Average h across active models
    """
```

### Notes

- Ensemble knows **which signals to combine** but not **portfolio risk targets** or **scaling factors**
- Forecast scores are raw weighted sums in [0, 1] range (since weights sum to 1)
- Inverse correlation weighting handles base model diversification
- All standardization, scaling, and diversification multipliers happen at Portfolio layer
- Multiple ensembles can be averaged naturally at Portfolio layer

---

## Layer 3: Portfolio

### Responsibility

**Convert forecast scores to position fractions** by applying volatility scaling and portfolio-level risk management.

**Key Design**: Each Portfolio manages ONE trading timeframe only. For multi-timeframe trading, use multiple Portfolio instances.

### Current Implementation

⚠️ **Status**: Needs enhancement (currently just averages predictions without volatility scaling)

### Input

From predict():
- `X`: Feature matrix
- `ticker`: Instrument tickers
- `volatility`: Blended volatility per instrument (annualized, 70% EWMA-32 + 30% 10-year average)
- `normalization_data`: For ensemble base models

**Note**: `timeframe` parameter is removed - Portfolio now knows its trading timeframe via constructor.

### Output

DataFrame with columns:
- `ticker`: Instrument identifier
- `forecast_score`: Average forecast across ensembles
- `position_fraction`: Fraction of capital to allocate (% as decimal)

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

#### Step 1: Average Forecasts Across Ensembles (Equal Weight Ensembles)

For each instrument, average forecasts from **all ensembles that contain it**:

$$\bar{F}_{\text{instrument}} = \frac{1}{N_{\text{ensembles,instrument}}} \sum_{j \in \text{ensembles with instrument}} F_{j,\text{instrument}}$$

Where $N_{\text{ensembles,instrument}}$ is the number of ensembles that trade this specific instrument.

**Example**: If NQ appears in "Mean Reversion" and "Momentum" ensembles:
- Forecast from Mean Reversion: $F_{\text{MR}} = 0.4$ (raw, no scaling)
- Forecast from Momentum: $F_{\text{MOM}} = 0.3$ (raw, no scaling)
- Average: $\bar{F}_{\text{NQ}} = \frac{0.4 + 0.3}{2} = 0.35$

**Key Insight**: This averaging **equates risk across ensembles**. An instrument appearing in 2 ensembles gets the average of both forecasts, not the sum. This prevents instruments in multiple ensembles from being over-weighted.

#### Relationship to Carver's Framework

**Carver's Approach**:
- Each instrument has its own set of strategies (trading rules)
- Strategies are excluded per-instrument based on cost thresholds
- Forecast weights are recalculated per-instrument after exclusions

**Our Adaptation**:
- Ensembles (like Carver's "trading styles") are equal-weighted
- Instruments can appear in multiple ensembles (unlike Carver's per-instrument strategy sets)
- Forecast averaging across ensembles equates risk (similar to Carver's equal strategy weighting)
- Instrument weights are applied at portfolio level (like Carver's handcrafting)

**Key Difference**: In Carver's system, an instrument only trades strategies that are cost-effective for it. In our system, an instrument can appear in multiple ensembles, and we average the forecasts to prevent over-weighting.

#### Step 2: Use Raw Forecast Directly

Since raw forecasts are already in [0, 1] range and we need a multiplier for position sizing, we can use the raw forecast directly:

$$f_{\text{norm}} = \bar{F}_{\text{instrument}}$$

**Why no scaling needed**: 
- Raw forecasts are in [0, 1] range (weights sum to 1, signals are binary)
- We need a multiplier where 1.0 = maximum strength signal
- The raw forecast already serves this purpose - no need to scale by 10 then divide by 10

**Note**: This is simpler than Carver's approach where he scales to 10 then normalizes. We skip the redundant scaling step.

#### Step 3: Calculate Volatility-Adjusted Position

Apply the **core position sizing formula**:

$$\text{position}_{\text{vol-adjusted}} = f_{\text{norm}} \times \frac{\tau}{\sigma_{\text{blended}} \times \sqrt{\bar{h}}}$$

Where:
- $f_{\text{norm}}$: Normalized forecast strength
- $\tau$: Target annual portfolio volatility (e.g., 0.20 = 20%)
- $\sigma_{\text{blended}}$: Instrument's blended annualized volatility (70% EWMA-32 + 30% 10-year average)
- $\sqrt{\bar{h}}$: Square root of average exposure fraction

**Intuition**: The $\sqrt{h}$ adjustment accounts for the fact that if you're only in the market 10% of the time ($h = 0.1$), your realized volatility is $\sqrt{0.1} \approx 0.316 \times$ the full-time volatility.

**Why Blended Volatility**: Using Carver's blended estimate (70% short-run, 30% long-run) prevents over-leveraging during quiet periods and provides more robust risk forecasting. See [Blended Volatility Estimate](#blended-volatility-estimate) section for details.

#### Step 4: Apply Diversification Multiplier (DM)

$$\text{position}_{\text{diversified}} = \text{position}_{\text{vol-adjusted}} \times \text{DM}$$

**Purpose**: DM is a **scaling factor** that increases position sizes to account for diversification benefits across:
- Base models within ensembles (inverse correlation weights already help)
- Multiple ensembles (different trading styles)
- Different instruments (NQ, ES, GC, CL)
- Different asset classes (equities, commodities, FX)
- Low correlation between strategies and instruments

**Why we need this**: Without DM, we would be treating all strategies and instruments as if they were perfectly correlated, leading to under-allocation. Since our strategies and instruments are **not perfectly correlated**, we can take **larger positions** while maintaining the same risk level. This scaling factor allows us to **hit our target portfolio volatility**.

**Simple Fitting from Past Performance** (Recommended):

The optimal DM can be directly estimated from backtest results:

$$\text{DM}_{\text{optimal}} = \frac{\text{target volatility}}{\text{realized volatility}}$$

**Process**:
1. Run backtest with DM = 1.0 (no scaling)
2. Measure realized portfolio volatility
3. Calculate: $\text{DM} = \frac{0.20}{\text{realized\_vol}}$ (if target is 20%)
4. Re-run backtest with fitted DM to validate

**Why this works**:
- **Trivial fitting**: Just one division, no complex optimization
- **Won't overfit**: Single parameter, directly related to target metric
- **Intuitive**: If realized vol is half of target, double the positions (DM = 2.0)
- **Robust**: Works well in practice and is easy to understand

**Initial Implementation**:

$$\text{DM} = 1.0 \quad \text{(no scaling, conservative)}$$

Start with DM = 1.0, then fit from backtest results.

**Typical values** (for reference):
- Single strategy, single instrument: DM = 1.0
- Multiple strategies, single instrument: DM ≈ 1.2-1.8
- Multiple strategies, multiple instruments: DM ≈ 1.5-2.5
- Highly diversified portfolio: DM ≈ 2.0-3.0

**Note**: This is a **simple tuning parameter** that can be directly estimated from past performance. The fitting is trivial and won't overfit.

#### Step 5: Apply Instrument Weight

$$\text{position}_{\text{weighted}} = \text{position}_{\text{diversified}} \times w_{\text{instrument}}$$

Where $w_{\text{instrument}}$ is the allocation weight for this instrument.

**Current Implementation - Equal Weight Per Instrument**:

$$w_{\text{instrument}} = \frac{1}{N_{\text{instruments}}}$$

Where $N_{\text{instruments}}$ is the total number of unique instruments across all ensembles.

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

**Note**: These weights are **risk-agnostic**. Volatility scaling (Step 3) handles risk adjustment, so a low-volatility bond and high-volatility crypto can receive the same weight.

#### Step 6: Cap Position

$$\text{position}_{\text{final}} = \min(\text{position}_{\text{weighted}}, \text{position}_{\text{max}})$$

Where $\text{position}_{\text{max}}$ is the maximum position size (e.g., 2.0 = 200% of average risk).

### Complete Formula

Combining all steps:

$$\boxed{
\text{position\_fraction} = \min\left(
  \bar{F} \times \frac{\tau}{\sigma_{\text{blended}} \times \sqrt{\bar{h}}} \times \text{DM} \times w_{\text{instr}}, 
  \text{position}_{\text{max}}
\right)
}$$

**Simplified notation**:

$$\boxed{
\text{position\_fraction} = \min\left(
  f_{\text{norm}} \times \frac{\tau}{\sigma_{\text{blended}} \times \sqrt{\bar{h}}} \times \text{DM} \times w_{\text{instr}}, 
  \text{position}_{\text{max}}
\right)
}$$

Where:
- $f_{\text{norm}} = \bar{F}$ (raw forecast in [0, 1] range, no scaling needed)
- $\sigma_{\text{blended}} = 0.70 \times \sigma_{\text{EWMA-32}} + 0.30 \times \sigma_{\text{10-year}}$ (Carver's blended volatility estimate)

### Key Parameters

- `target_volatility` ($\tau$): Annual portfolio volatility target (default: 0.20)
- `dm`: **Diversification Multiplier** (default: 1.0 initially, fit from backtests)
  - Start conservative at 1.0 (no scaling)
  - Fit from past performance: $\text{DM} = \frac{\text{target volatility}}{\text{realized volatility}}$
  - Typical range: 1.0 - 2.5 depending on portfolio diversification
  - Simple fitting approach won't overfit
- `max_position_pct`: Maximum position per instrument (default: 2.0)
- `instrument_weights`: Dict mapping ticker → weight (default: equal per instrument)
  - **Current**: Equal weight: $w_{\text{instrument}} = \frac{1}{N_{\text{instruments}}}$
  - **Future**: Carver-style handcrafting by asset class → group → instrument

**Note**: No forecast scalar needed - raw forecasts are in [0, 1] and can be used directly as multipliers for position sizing.

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
    ensembles: List[Ensemble],  # Changed from Dict[TimeFrame, List[Ensemble]]
    trading_timeframe: TimeFrame,  # NEW: Explicit trading timeframe
    target_volatility: float = 0.20,
    dm: float = 1.0,
    max_position_pct: float = 2.0,
    instrument_weights: Optional[Dict[str, float]] = None
):
    """
    Create a Portfolio for a SINGLE trading timeframe.
    
    Parameters
    ----------
    ensembles : List[Ensemble]
        List of ensemble models (no longer organized by timeframe)
    trading_timeframe : TimeFrame
        The timeframe this portfolio trades on (e.g., TimeFrame.D for daily)
    target_volatility : float
        Target annual portfolio volatility
    dm : float
        Diversification Multiplier
    max_position_pct : float
        Maximum position size per instrument
    instrument_weights : Dict[str, float], optional
        Weight for each instrument. If None, equal weight.
    """
```

#### Method Signatures

```python
def predict(
    self,
    X: pd.DataFrame,
    ticker: pd.Series,
    volatility: pd.Series,  # Blended volatility per instrument (70% EWMA-32 + 30% 10-year)
    normalization_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Returns: DataFrame(['ticker', 'forecast_score', 'position_fraction'])
    
    Note: timeframe parameter removed - Portfolio knows its trading timeframe
    """
```

### Notes

- **Single-timeframe design**: Each Portfolio manages ONE trading timeframe only ⭐ NEW
  - For multi-timeframe trading, use multiple Portfolio instances
  - MLManager routes candles to appropriate portfolio based on timeframe
- Portfolio is the **central hub** for all scaling and diversification
- Uses raw forecasts directly (no scaling needed - they're already in [0, 1] range)
- Applies **DM** (diversification multiplier, default 1.0) to scale for all diversification benefits
  - Accounts for both strategy and instrument diversification
  - Simple fitting: $\text{DM} = \frac{\text{target volatility}}{\text{realized volatility}}$
  - Trivial fitting approach won't overfit
- Uses blended volatility estimate (70% EWMA-32, 30% 10-year average) for robust risk management
- Position fractions can exceed 1.0 (e.g., 1.5 = 150% of capital allocated)
- Capping at position level only (e.g., max_position_pct = 2.0)
- **Ensembles are equal-weighted** (not instruments)
- **Instruments can appear in multiple ensembles** - forecasts are averaged to equate risk
- **Instrument weighting**: Currently equal per instrument, future: Carver-style handcrafting by asset class
- **Clear separation**: Ensemble handles signal combination, Portfolio handles risk management
- **Simple tuning**: DM is a single parameter that can be directly estimated from past performance

---

## Layer 4: Execution

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
- Forecast generation: $f_{i,t}$ (raw forecast in [0, 1], used directly)
- Position fraction: $\frac{\tau \times w_i \times \text{DM}}{\sigma\%_i \times \sqrt{h_i}}$
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
- DM: 3.0 
- Instrument weighting: Equal (5 unique instruments: NQ, ES, YM, GC, CL)

**Note**: This example uses DM=3.0 to demonstrate the full workflow. In initial implementation, start with DM=1.0, then fit from backtest results using $\text{DM} = \frac{\text{target volatility}}{\text{realized volatility}}$. Raw forecasts are used directly (no scaling needed).

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

##### Layer 2: Ensemble → Forecast Score

**Mean Reversion Ensemble**:
- Diversification weights: $w = [0.12, 0.08, 0.15, 0.10, 0.09, 0.11, 0.08, 0.13, 0.07, 0.07]$
- Raw forecast: $F_{\text{raw,MR}} = (1 \times 0.12) + (1 \times 0.15) + (1 \times 0.10) + (1 \times 0.13) = 0.50$
- Average exposure: $\bar{h}_{\text{MR}} = \frac{4 \times 0.1}{4} = 0.1$
- **Ensemble output**: $F_{\text{MR}} = 0.50$ (raw, no scaling)

**Momentum Ensemble**:
- Diversification weights: $w = [0.10, 0.08, 0.12, 0.15, 0.09, 0.11, 0.14, 0.08, 0.07, 0.06]$
- Raw forecast: $F_{\text{raw,MOM}} = (1 \times 0.15) + (1 \times 0.14) = 0.29$
- Average exposure: $\bar{h}_{\text{MOM}} = \frac{2 \times 0.1}{2} = 0.1$
- **Ensemble output**: $F_{\text{MOM}} = 0.29$ (raw, no scaling)

##### Layer 3: Portfolio → Position Fraction

**Step 1: Average across ensembles**:
$$\bar{F}_{\text{NQ}} = \frac{F_{\text{MR}} + F_{\text{MOM}}}{2} = \frac{0.50 + 0.29}{2} = 0.395$$

**Use raw forecast directly**:
$$f_{\text{norm}} = \bar{F}_{\text{NQ}} = 0.395$$

**NQ parameters**:
- Blended volatility: $\sigma_{\text{blended,NQ}} = 0.25$ (25% annualized)
  - Short-run (EWMA-32): $\sigma_{\text{short}} = 0.24$ (24%)
  - Long-run (10-year): $\sigma_{\text{long}} = 0.27$ (27%)
  - Blended: $0.70 \times 0.24 + 0.30 \times 0.27 = 0.25$
- Average exposure: $\bar{h} = 0.1$ (average of both ensembles)

**Step 2: Volatility-adjusted position**:
$$\text{position}_{\text{vol}} = 0.395 \times \frac{0.20}{0.25 \times \sqrt{0.1}} = 0.395 \times 2.53 = 1.00$$

**Step 3: Apply DM**:
$$\text{position}_{\text{div}} = 1.00 \times 3.0 = 3.00$$

**Step 4: Apply instrument weight** (5 unique instruments, equal weight):
$$w_{\text{NQ}} = \frac{1}{5} = 0.20$$
$$\text{position}_{\text{weighted}} = 3.00 \times 0.20 = 0.60$$

**Step 5: Cap position** (max = 2.0, already below):
$$\text{position}_{\text{final}} = 0.60$$

##### Layer 4: Execution → Number of Contracts

**NQ contract specs**:
- Price: $16,000
- Multiplier: $20/point
- FX rate: 1.0

**Contract value**:
$$V_{\text{contract}} = 16{,}000 \times 20 \times 1.0 = \$320{,}000$$

**Target allocation**:
$$D_{\text{target}} = 0.60 \times 1{,}000{,}000 = \$600{,}000$$

**Raw contracts**:
$$N_{\text{raw}} = \frac{600{,}000}{320{,}000} = 1.875$$

**Rounded contracts**:
$$N_{\text{final}} = \text{round}(1.875) = 2$$

**Actual notional**:
$$V_{\text{notional}} = 2 \times 320{,}000 = \$640{,}000$$

**Result**: **2 contracts** (notional: $640k, 64% of capital)

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

## Practical Guide: Tuning Diversification Multiplier (DM)

### Simple Fitting from Past Performance

The DM can be directly estimated from backtest results using a trivial fitting approach that won't overfit.

#### Step 1: Initial Backtest (Baseline)

1. **Set DM to 1.0** (no scaling):
   ```python
   portfolio = Portfolio(
       ensembles=[...],
       trading_timeframe=TimeFrame.D,
       target_volatility=0.20,
       dm=1.0,  # No diversification scaling
       max_position_pct=2.0
   )
   ```

2. **Run backtest** and measure:
   - Target volatility: 20% annual
   - Realized volatility: ? (likely lower, e.g., 10-15%)
   - Example: Realized = 12%

#### Step 2: Simple Fitting

3. **Calculate optimal DM**:
   ```python
   dm_optimal = target_volatility / realized_volatility
   # Example: 0.20 / 0.12 = 1.67
   ```

4. **Update portfolio** with fitted DM:
   ```python
   portfolio = Portfolio(
       ensembles=[...],
       trading_timeframe=TimeFrame.D,
       target_volatility=0.20,
       dm=1.67,  # Fitted from backtest
       max_position_pct=2.0
   )
   ```

5. **Re-run backtest** to validate:
   - Should now realize volatility close to target (e.g., 19-21%)
   - If still off, fine-tune slightly

#### Step 3: Validation and Monitoring

6. **Check results**:
   - Realized volatility should be within ±2% of target
   - Monitor over time - DM may need adjustment if portfolio composition changes

7. **Red flags**:
   - **Too high**: Realized vol >> target vol (over-leveraged), large drawdowns
   - **Too low**: Realized vol << target vol (under-utilized), very small positions

### Why This Works

**Trivial Fitting**:
- Single parameter, single division: $\text{DM} = \frac{\tau}{\sigma_{\text{realized}}}$
- No complex optimization or multiple parameters
- Directly related to target metric (volatility)

**Won't Overfit**:
- One parameter is much less prone to overfitting than multiple correlated parameters
- The relationship is linear and intuitive
- Easy to validate on out-of-sample data

**Robust**:
- Works well in practice
- Easy to understand and explain
- Can be updated periodically as portfolio evolves

### Example Tuning Log

```
Iteration 0 (Baseline):
  DM = 1.0
  Target Vol: 20%, Realized Vol: 12%
  → Under-allocated by 40%

Iteration 1 (Fitted):
  DM = 0.20 / 0.12 = 1.67
  Target Vol: 20%, Realized Vol: 19.5%
  → Good! Within 2% of target

Final value: DM = 1.67
```

### Optional: Rolling Window Adaptation

For production systems, you can update DM periodically using a rolling window:

```python
# Every quarter, refit DM from last 6 months of data
recent_returns = get_recent_returns(window='6M')
realized_vol = calculate_volatility(recent_returns)
dm_updated = target_volatility / realized_vol
```

This allows DM to adapt to changing market conditions while remaining simple and robust.

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
- DM: $1.0$ (no diversification scaling)
- Instrument weight: $w = 1.0$ (single instrument)

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{1.0}} \times 1.0 \times 1.0 = 1.0$$

**Verification**: Position fraction = 1.0 (100% of capital allocated)

**Why this works**: When target volatility equals instrument volatility and we're always in the market, we should allocate 100% of capital.

#### Scenario 2: High Volatility Instrument

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.40$ (40% - twice the target)
- DM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.40 \times \sqrt{1.0}} \times 1.0 \times 1.0 = 0.5$$

**Verification**: Position fraction = 0.5 (50% of capital allocated)

**Why this works**: High volatility instrument requires smaller position to maintain target risk.

#### Scenario 3: Low Volatility Instrument

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.10$ (10% - half the target)
- DM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.10 \times \sqrt{1.0}} \times 1.0 \times 1.0 = 2.0$$

**Verification**: Position fraction = 2.0 (200% of capital - leverage)

**Why this works**: Low volatility instrument allows larger position (with leverage) to maintain target risk.

#### Scenario 4: Diversification Multiplier Effect

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- DM: $2.0$ (diversification scaling)
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{1.0}} \times 2.0 \times 1.0 = 2.0$$

**Verification**: Position fraction = 2.0 (200% of capital - scaled by DM)

**Why this works**: DM increases position size to account for diversification benefits.

#### Scenario 5: Sparse Signals (Low Exposure)

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$ (maximum signal when active)
- Exposure: $\bar{h} = 0.1$ (only in market 10% of time)
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.20$ (20%)
- DM: $1.0$
- Instrument weight: $w = 1.0$

**Expected Position**:
$$\text{position} = 1.0 \times \frac{0.20}{0.20 \times \sqrt{0.1}} \times 1.0 \times 1.0 = 1.0 \times \frac{0.20}{0.0632} = 3.16$$

**Verification**: Position fraction = 3.16 (316% of capital - leverage)

**Why this works**: Sparse signals have lower realized volatility ($\sqrt{0.1} \approx 0.316$), so we can take larger positions when active.

#### Scenario 6: Position Capping

**Setup**:
- Forecast: $f_{\text{norm}} = 1.0$
- Exposure: $\bar{h} = 1.0$
- Target volatility: $\tau = 0.20$ (20%)
- Instrument volatility: $\sigma_{\text{blended}} = 0.05$ (5% - very low)
- DM: $1.0$
- Instrument weight: $w = 1.0$
- Max position: $2.0$ (200% cap)

**Uncapped Position**:
$$\text{position}_{\text{uncapped}} = 1.0 \times \frac{0.20}{0.05 \times \sqrt{1.0}} \times 1.0 \times 1.0 = 4.0$$

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
        dm=dm,
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
