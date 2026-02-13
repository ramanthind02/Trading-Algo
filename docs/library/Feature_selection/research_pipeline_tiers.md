# Tiered Research Pipeline — From Discovery to Production

**Purpose:** This document defines a **tiered research pipeline** for feature development, from initial hypothesis-driven discovery through enhancement and deployment. It formalizes when and how to use different research methodologies (hypothesis testing, data mining, machine learning) at each stage of feature maturity.

**Core principle:** Theory → Data, not Data → Theory. Start with plausible mechanisms, test rigorously, deploy conservatively, enhance carefully.

---

## **The Fundamental Challenge: Past Performance ≠ Future Performance**

**ALL empirical feature selection is biased toward features that performed well on training data.**

This is tautologically true, regardless of methodology:
- **Hypothesis-driven testing:** You test features with historical edge, hope they persist
- **Data mining:** You search for features with historical edge, hope they're real
- **Machine learning:** You fit coefficients to maximize historical edge, hope they generalize

**The ONLY way to trust future performance:**
1. **Strong theoretical mechanism** (behavioral bias, risk premium, market structure)
2. **Long stable track record** (20+ years, multiple regimes)
3. **Parameter stability** (small changes don't break it)
4. **Production validation** (works live, not just in backtests)

**Therefore:**
- **Hypothesis-driven testing** is required for ALL new features (Tier 1)
- **Data mining** is acceptable ONLY for variations of proven features (Tier 2)
- **Machine learning** is acceptable ONLY on proven features as inputs (Tier 3)
- **Production monitoring** is the ultimate test (Tier 4)

---

## **Overview: Four-Tier Pipeline**

| Tier | Purpose | Input | Output | Methodology |
|------|---------|-------|--------|-------------|
| **1. Discovery** | Find new robust features | Hypothesis + theory | 2-5 features/year | Hypothesis-driven permutation testing |
| **2. Enhancement** | Improve proven features | Live-tested features | 0-2 enhancements/year | Data mining on variations (with FDR) |
| **3. ML Ensemble** | Capture non-linear interactions | 5+ proven features | 1 ensemble model (optional) | Regularized ML on proven inputs |
| **4. Monitoring** | Continuous validation | All deployed features | Retire 10-20%/year | Production tracking |

**Key:** Each tier has different standards, because the risk and prior strength differ.

---

## **Tier 1: Hypothesis-Driven Discovery (NEW FEATURES)**

**Purpose:** Find genuinely new sources of alpha with strong theoretical grounding.

**This is the ONLY tier that produces new features.** All other tiers build on features discovered here.

### **When to Use**

✅ **Use Tier 1 for:**
- Testing a new trading hypothesis (mean reversion, momentum, breakout, seasonality, etc.)
- Exploring a new market phenomenon (documented in academic literature or practitioner research)
- Designing features based on economic/behavioral theory

❌ **Do NOT use Tier 1 for:**
- "Let me try 100 random indicator combinations and see what works"
- Features selected purely on past performance (no theory)
- Variations of existing features (use Tier 2)

### **Process**

```
Step 1: Hypothesis Formation (Theory → Data)
  Researcher identifies plausible mechanism:
  - Mean reversion: Behavioral overreaction + correction
  - Momentum: Underreaction to news, trend persistence
  - Breakout: Supply/demand imbalance, continuation patterns
  - Value: Risk premium, long-term mean reversion
  - Seasonality: Weather, harvest cycles, tax effects

  NOT: "This indicator looked good in my backtest"

Step 2: Feature Design
  Design feature to capture mechanism:
  - Choose appropriate indicator (RSI for mean reversion, etc.)
  - Parametrize BROADLY (e.g. RSI lookback 2-20, not just 14)
  - Avoid cherry-picking (test family of params, not single "optimal")

Step 3: Permutation Testing (Current Framework)
  Run full permutation testing pipeline:
  - Use ALL walkforward data for permutation testing (max power)
  - Stages 1-5: vector shuffle, pipeline perm, ensemble perm, walkforward perm
  - Fixed ensemble (parameter averaging via neighbor selection)
  - Pre-specified thresholds (no data snooping)

  See: validation_framework_philosophy.md, in-sample_pt.md

Step 4: Hold-Out Validation
  - Reserve most recent data (20-30% of total)
  - NEVER touched during permutation testing
  - Final gate: does feature work on truly unseen data?
  - Pass criteria: Sharpe > 0, no obvious pathologies

Step 5: Production Deployment
  - Deploy with FIXED parameters (no adaptation)
  - Start with small allocation (1-5% of portfolio)
  - Monitor for 6-12 months (does it work as expected?)
  - If Sharpe matches expectations → increase allocation
  - If Sharpe < 0 for 6+ months → retire (don't re-tune)
```

### **Expected Outcomes**

| Metric | Value | Notes |
|--------|-------|-------|
| **Features tested/year** | 5-20 | Hand-selected, strong priors |
| **Pass permutation tests** | 20-40% | With hypothesis-driven selection |
| **Pass hold-out** | 60-80% | Of those passing permutation |
| **Work in production** | 60-80% | Of those passing hold-out |
| **Net deployed/year** | 2-5 | High quality, theory-backed |

**Key:** Low volume, high quality. Each feature has 10-30% chance of making it to production.

### **Quality Standards**

Before testing a feature in Tier 1, it must satisfy:

**1. Economic/Behavioral Rationale**
- [ ] Clear hypothesis about WHY this feature should predict returns
- [ ] Grounded in economics, behavioral finance, or market structure
- [ ] Examples: mean reversion (overreaction), momentum (underreaction), value (risk premium)
- [ ] NOT: "This pattern showed up in my exploratory data analysis"

**2. Prior Research or Strong Theory**
- [ ] Documented in academic literature (papers, books), OR
- [ ] Novel hypothesis with rigorous theoretical justification, OR
- [ ] Practitioner research with economic reasoning
- [ ] NOT: "I tried 50 things and this worked"

**3. Broad Parametrization**
- [ ] Test a range of parameters (e.g. lookback 2-20)
- [ ] Don't cherry-pick (e.g. "RSI 14 works, so test only RSI 14")
- [ ] Ensemble selection will find stable region

**4. Multiple Testing Awareness**
- [ ] Track how many features tested this year
- [ ] If >20/year, consider stricter alpha or FDR correction
- [ ] Maintain feature testing log (see Tier 4)

### **Examples**

✅ **Good Tier 1 Features:**
- RSI mean reversion (DeBondt & Thaler overreaction hypothesis)
- Momentum following breakouts (Jegadeesh & Titman trend persistence)
- Commodity seasonality (supply/demand cycles, documented patterns)
- Cross-sectional value (Fama-French risk premium)

❌ **Bad Tier 1 Features:**
- "MACD + RSI + Bollinger combination that backtested well" (no theory)
- "This pattern in Bitcoin looked good" (data mining)
- "RSI but with volume filter" (this is Tier 2, not Tier 1 — it's a variation)

### **Reference**

Full technical specifications:
- [Validation Framework Philosophy](validation_framework_philosophy.md)
- [In-Sample Permutation Testing](permutation_testing/in-sample_pt.md)
- [Base Model (Ensemble)](base_models/base_model.md)
- [Continuous Binning](feature_types/Continuous_binning.md)

---

## **Tier 2: Variation Testing (ENHANCEMENTS OF PROVEN FEATURES)**

**Purpose:** Find small improvements to features that are ALREADY deployed and working in production.

**Key difference from Tier 1:** You're not discovering new features; you're enhancing existing ones that have PROVEN themselves live.

### **When to Use**

✅ **Use Tier 2 for:**
- Feature has been live for 1-2+ years
- Consistent positive Sharpe (e.g. 0.8-1.5)
- You want to test variations (filters, regimes, combinations)

❌ **Do NOT use Tier 2 for:**
- Features that only worked in backtest (not live)
- Features that are currently failing in production
- Completely new features (use Tier 1)

### **Process**

```
Step 1: Identify Base Feature
  Base feature requirements:
  ✅ Deployed in production for 1-2+ years
  ✅ Sharpe > 0.8 consistently
  ✅ Mechanism is understood and stable
  ✅ Passed full Tier 1 validation

  Example: RSI mean reversion, Sharpe = 1.2 (live for 2 years)

Step 2: Generate Variations
  Test small enhancements (ONE at a time):
  - Volume filter: only trade when volume > average
  - Volatility regime: only trade in high/low vol
  - Time-of-day filter: only trade morning/afternoon
  - Sector filter: only in certain sectors
  - Combine with other proven features: RSI + Momentum

  NOT: Test 100 random filter combinations

Step 3: Modified Permutation Testing
  More conservative than Tier 1:

  Option A (stricter): Use only FIRST FOLD for permutation testing
    - Reserve folds 2-5 as true OOS
    - Test variations on first fold only
    - Validate on folds 2-5 (unseen data)
    - More conservative (lower power, but prevents overfitting)

  Option B (current framework): Use all walkforward
    - Same as Tier 1 (use all folds for permutation)
    - Acceptable if testing <10 variations
    - Apply FDR correction if testing >10 variations

  Recommendation: Use Option A if testing >20 variations

Step 4: Performance Comparison
  Variation must BEAT baseline:
  - Baseline = original feature (e.g. RSI, Sharpe = 1.2)
  - Variation must achieve Sharpe > 1.4 (substantial improvement)
  - If Sharpe 1.2 → 1.25: not worth the complexity
  - If Sharpe 1.2 → 1.5: deploy

Step 5: Hold-Out Validation
  Same as Tier 1:
  - Test on hold-out data
  - Compare to baseline (does enhancement persist OOS?)
  - If enhancement disappears, it was spurious

Step 6: Production Deployment
  Deploy as REPLACEMENT for baseline OR as ADDITION:
  - Replacement: "RSI with volume filter" replaces "RSI"
  - Addition: "RSI + Momentum combo" is a new feature alongside RSI and Momentum
  - Monitor: does enhancement degrade to baseline over time?
  - If yes: enhancement was spurious, revert to baseline
```

### **Expected Outcomes**

| Metric | Value | Notes |
|--------|-------|-------|
| **Variations tested/year** | 10-50 per base feature | Only for proven features |
| **Pass permutation tests** | 5-10% | Most variations don't help |
| **Beat baseline on hold-out** | 30-50% | Of those passing permutation |
| **Work in production** | 50-70% | Of those beating baseline |
| **Net deployed/year** | 1-2 | Rare to find real improvements |

**Key:** Most variations fail. Real improvements are rare.

### **Quality Standards**

**1. Base Feature Must Be Proven**
- [ ] Live for 1-2+ years (not just backtested)
- [ ] Sharpe > 0.8 consistently
- [ ] Passed full Tier 1 validation (permutation tests + hold-out)

**2. Variation Must Be Small and Interpretable**
- [ ] ONE change at a time (not "RSI + volume + sector + time-of-day")
- [ ] Clear reasoning (why should this filter help?)
- [ ] Mechanism still makes sense (not destroying the original logic)

**3. Multiple Testing Control**
- [ ] If testing >10 variations, apply FDR correction (Benjamini-Hochberg)
- [ ] Track: how many variations tested per base feature
- [ ] Expected false positives = N_variations × α

**4. Substantial Improvement Required**
- [ ] Sharpe improvement > 0.2 (e.g. 1.2 → 1.4)
- [ ] If improvement < 0.1, not worth the added complexity

### **Examples**

✅ **Good Tier 2 Variations:**
- Base: RSI mean reversion (Sharpe = 1.2, live 2 years)
  - Variation: RSI + volume filter (only trade when volume > avg)
  - Rationale: High volume confirms strength of reversal

- Base: Momentum (Sharpe = 0.9, live 3 years)
  - Variation: Momentum + volatility regime (only in low vol)
  - Rationale: Momentum works better in stable markets

❌ **Bad Tier 2 Variations:**
- Base: RSI (only backtested, not live) → can't use Tier 2
- Variation: RSI + 10 random filters combined → too complex, no clear reasoning
- "Let me try 100 filter combinations on RSI" → data mining without strong priors

---

## **Tier 3: ML Ensemble (NON-LINEAR INTERACTIONS OF PROVEN FEATURES)**

**Purpose:** Learn optimal combinations and regime-dependent weightings of proven features.

**Key principle:** ML on PROVEN inputs (gold), not random features (garbage).

### **When to Use**

✅ **Use Tier 3 when:**
- You have 5-10+ features deployed and working in production
- Features are uncorrelated or weakly correlated (0.3-0.7 correlation)
- You want to capture non-linear interactions (e.g. "RSI works in high vol, momentum works in low vol")

❌ **Do NOT use Tier 3 for:**
- <5 deployed features (not enough inputs)
- Highly correlated features (0.9+ correlation — just pick one)
- Features that only worked in backtest (not proven live)

### **Process**

```
Step 1: Collect Proven Features
  Requirements:
  ✅ ALL inputs must have passed Tier 1
  ✅ ALL inputs must be live and working (Sharpe > 0.5)
  ✅ Minimum 5 features (preferably 10+)

  Example inputs:
  - RSI mean reversion (Sharpe 1.2, live 2 years)
  - Momentum (Sharpe 0.9, live 3 years)
  - Breakout (Sharpe 1.0, live 1.5 years)
  - Value (Sharpe 1.1, live 5 years)
  - Seasonality (Sharpe 0.8, live 2 years)

Step 2: Add Regime Indicators (Optional)
  Context features (not alpha features):
  - Volatility level (VIX, realized vol)
  - Market regime (trending vs mean-reverting)
  - Sector indicators
  - Macro indicators (rates, credit spreads)

  These help ML learn WHEN each feature works best

Step 3: ML Model Selection
  Prefer interpretable, regularized models:

  Option A: Linear (simplest, most interpretable)
    - LASSO (L1 regularization) → sparse weights
    - Ridge (L2 regularization) → stable weights
    - Elastic Net (L1 + L2) → balance
    - Cross-validation for regularization strength

  Option B: Ensemble (non-linear, more complex)
    - Random Forest → feature importance
    - Gradient Boosting → non-linear interactions
    - Still regularized (max depth, min samples, etc.)

  Option C: Simple Rules (most interpretable)
    - Decision tree (max depth 3-5)
    - If-then rules (e.g. "if VIX > 20, use RSI; else use momentum")
    - Manually constructed based on domain knowledge

  Recommendation: Start with Option A (linear) or C (rules)

Step 4: Train/Test Split
  Conservative approach (similar to Tier 2 Option A):
  - Train on first fold ONLY
  - Validate on folds 2-5 (true OOS)
  - Hold-out: final validation

  Or: Use cross-validation within first fold, then validate on folds 2-5

Step 5: Baseline Comparison
  ML ensemble must BEAT simple average:
  - Baseline: equal-weighted average of all features
  - ML ensemble: optimized weights/combinations
  - Improvement threshold: Sharpe increase > 0.3 (e.g. 1.3 → 1.6)
  - If improvement < 0.2: not worth the complexity

Step 6: Interpretability Check
  Understand WHY the ML works:
  - Which features get highest weights?
  - When does it switch between features (regimes)?
  - Does the logic make sense (or is it spurious)?
  - Can you explain it in plain English?

Step 7: Production Deployment
  Deploy with fallback:
  - Primary: ML ensemble
  - Fallback: simple equal-weighted average of features
  - Monitor: if ML Sharpe drops below simple average for 6+ months → revert to fallback
  - Never re-train in production (fixed weights)
```

### **Expected Outcomes**

| Metric | Value | Notes |
|--------|-------|-------|
| **Models tested/year** | 1-3 | Rare; requires 5-10+ proven features |
| **Beat simple average in-sample** | 50-70% | Not all ML helps |
| **Beat simple average on hold-out** | 30-50% | Many overfit |
| **Work in production** | 50-70% | Of those beating on hold-out |
| **Improvement magnitude** | 20-40% Sharpe increase | When successful |

**Key:** High risk, high reward. Only attempt when you have solid inputs.

### **Quality Standards**

**1. ALL Inputs Are Proven**
- [ ] Every feature passed Tier 1 validation
- [ ] Every feature has 1+ years live performance
- [ ] Every feature has Sharpe > 0.5 in production
- [ ] NOT: "I'll throw in some random indicators to see if they help"

**2. Limited Dimensionality**
- [ ] 5-20 input features (not 100+)
- [ ] If >20 features, apply feature selection FIRST (e.g. LASSO)
- [ ] Avoid curse of dimensionality

**3. Regularization Required**
- [ ] Cross-validation for hyperparameters
- [ ] Regularization strength chosen on validation set (not test set)
- [ ] Conservative: prefer simpler models over complex

**4. Interpretability Required**
- [ ] Can you explain why the ML works?
- [ ] Do the weights/rules make sense?
- [ ] If black box with no clear logic → reject (likely spurious)

**5. Substantial Improvement Required**
- [ ] Must beat simple average by >0.3 Sharpe
- [ ] If barely better: not worth the risk and complexity

### **Examples**

✅ **Good Tier 3 Use Cases:**
- Input: RSI, Momentum, Value, Breakout, Seasonality (all live 2+ years, Sharpe 0.8-1.2)
- ML task: Learn when to use RSI (high vol) vs Momentum (low vol)
- Result: Sharpe 1.3 (simple avg) → 1.7 (ML ensemble)

❌ **Bad Tier 3 Use Cases:**
- Input: 100 random indicators (not proven)
- ML task: Find which ones work
- Result: Overfitting, no generalization

---

## **Tier 4: Production Monitoring (CONTINUOUS VALIDATION)**

**Purpose:** Continuously validate all deployed features; retire broken ones.

**Key principle:** Production is the ONLY true OOS test.

### **Process**

```
Step 1: Track All Deployed Features
  For each feature in production:
  - Daily PnL
  - Rolling Sharpe (6-month and 12-month windows)
  - Drawdown (current and max)
  - Correlation with other features
  - Transaction costs (slippage, commissions)

Step 2: Automated Monitoring
  Red flags (investigate immediately):
  - Sharpe < 0 for 3+ months
  - Sharpe < -0.5 for 1+ month (severe degradation)
  - Drawdown > 2x expected (from backtest)
  - Sudden correlation spike with other features (0.9+)
  - Execution issues (slippage >> expected)

Step 3: Retirement Criteria
  Tier 1 (Core features): Conservative retirement
    - Sharpe < 0 for 6+ months → investigate deeply
    - Understand why it broke (regime change? market structure?)
    - Retire only if fundamentally broken

  Tier 2 (Enhancements): Aggressive retirement
    - Sharpe < baseline for 3+ months → retire immediately
    - Revert to baseline feature (e.g. RSI without filter)

  Tier 3 (ML Ensemble): Moderate retirement
    - Sharpe < simple average for 6+ months → revert to fallback (simple average)
    - Do NOT re-train (if it's broken, it's broken)

Step 4: Annual Review
  Every 12 months, review all features:
  - What % of graduated features are still working?
    - Target: 60-80% (if <50%, tests too lax; if >90%, tests too strict)
  - What was the empirical false discovery rate?
    - Compare to expected FDR (α = 0.1 → expect 10% false positives)
  - Are certain types of features failing more than others?
    - Momentum dying? Mean reversion dying? Adjust research focus

Step 5: Feature Testing Log
  Maintain permanent record:

  {
    "feature_name": "RSI_mean_reversion",
    "tier": 1,
    "date_tested": "2023-01-15",
    "hypothesis": "Oversold RSI predicts rebounds (overreaction)",
    "permutation_result": "PASS (Sharpe 1.5, p=0.02)",
    "holdout_result": "PASS (Sharpe 1.3)",
    "production_deployed": "2023-06-01",
    "production_sharpe_6mo": 1.2,
    "production_sharpe_12mo": 1.1,
    "status": "ACTIVE",
    "notes": "Stable performance, as expected"
  }

  Track EVERY feature (pass or fail) to understand true success rates
```

### **Expected Attrition**

| Source | Attrition/Year | Reason |
|--------|----------------|--------|
| **Tier 1 (Core)** | 5-10% | Rare; genuine regime changes |
| **Tier 2 (Enhanced)** | 20-30% | Enhancements often spurious |
| **Tier 3 (ML)** | 30-50% | ML is fragile; reverts to baseline |

**Net:** Expect to lose 10-20% of features per year. This is NORMAL and HEALTHY.

### **Red Flags (System-Level)**

If you see these patterns, your testing is too lax:
- **>30% attrition/year:** Tests are not filtering enough
- **Features failing immediately (<6 months):** Hold-out wasn't truly OOS
- **All features from one research period failing:** Overfitting to a specific regime

Adjust testing standards (lower α, stricter thresholds, longer hold-out).

---

## **Summary: When to Use Each Tier**

| Scenario | Tier | Why |
|----------|------|-----|
| **New hypothesis** (mean reversion, momentum, etc.) | **Tier 1** | Only way to discover genuinely new features |
| **Enhance live feature** (add filter, combine features) | **Tier 2** | Building on proven base; data mining acceptable |
| **Combine 5-10 live features** (regime-dependent weights) | **Tier 3** | ML on proven inputs; non-linear interactions |
| **Monitor deployed features** (track performance, retire failures) | **Tier 4** | Production is ultimate test |

**Do NOT:**
- ❌ Data mine from scratch (Tier 1 requires hypothesis, not search)
- ❌ Use ML on untested features (Tier 3 requires proven inputs)
- ❌ Keep trading broken features (Tier 4 requires discipline to retire)

---

## **Philosophy: Quality Over Quantity**

**Tier 1 produces 2-5 features per year** (not 100).

**Why this is correct:**
- Each feature has strong theory (mean reversion, momentum, etc.)
- Each feature passes rigorous permutation tests (5 stages)
- Each feature works on hold-out (true OOS)
- Each feature works in production (1+ years)

**Expected lifetime value:**
- 5 features/year × 5 years = 25 total features tested
- 60% pass all tests = 15 deployed
- 70% work in production = 10-11 stable features after 5 years

**This is ENOUGH.**
- 10 robust, uncorrelated features → portfolio Sharpe 1.5-2.0
- 100 mediocre, overfitted features → portfolio Sharpe 0.5-1.0 (degrades over time)

**Quality beats quantity.** Always.

---

## **References**

**Core framework:**
- [Validation Framework Philosophy](validation_framework_philosophy.md) — Why hypothesis-driven testing is required
- [In-Sample Permutation Testing](permutation_testing/in-sample_pt.md) — Technical specification for Tier 1

**Related:**
- [Base Model (Ensemble)](base_models/base_model.md) — Parameter averaging, fixed ensembles
- [Continuous Binning](feature_types/Continuous_binning.md) — Pre-specified thresholds, region detection
- [Rule-Based Features](feature_types/rule_based.md) — Discrete signal handling
