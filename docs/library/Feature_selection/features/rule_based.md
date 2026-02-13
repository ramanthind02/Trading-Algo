# Rule-Based Features (Spec)

Rule-based features use the **rule-based binning model**. The feature is already discrete: a vector of **-1, 0, 1** (short / flat / long) from a rule (e.g. breakout, threshold). No quantile or threshold binning is applied — only **Sharpe-based position scaling** as in the continuous binning pipeline.

---

## Inputs

| Input | Description |
|-------|-------------|
| **Feature** | Integer-like vector of **-1, 0, 1** (signed rule output). |
| **Target** | **Vol-scaled returns** (e.g. return / EWSD or return / ATR per observation), for unitless comparison across markets when training on multiple assets. |

---

## Fit Behavior

- Treat the three levels **-1, 0, 1** as three “bins.”
- For each level, compute on the **vol-scaled target** (returns where feature = that level):
  - `mean_return`, `volatility`, **Sharpe**, **Adjusted Sharpe** = `Sharpe * sqrt(N) / (sqrt(N) + k)`.
- Store per-level stats for use in predict. No binning or region detection — only the scaling step.

---

## Position Scaling (Same as Continuous Binning)

Apply the same logic as in [Continuous_binning](Continuous_binning.md) Step 6:

- **Position multiplier** (for non-zero levels):  
  - For level **+1 (long):** output `clip(1 + adjusted_Sharpe_long, 0.5, 2.0)`.  
  - For level **-1 (short):** Invert the logic — output **negative**: `-clip(1 + adjusted_Sharpe_short, 0.5, 2.0)` (range [-2.0, -0.5]).  
  - For level **0:** output **0** (flat).

  In **long_short** mode (both +1 and -1 used), output can range from **-2 to 2**. In **short**-only mode, output is in [-2, -0.5].

Target must be vol-scaled so that Sharpe (and adjusted Sharpe) are comparable across assets and markets.

---

## Predict

- Map each observation’s rule value (-1, 0, 1) to the corresponding position multiplier from fit.
- If `scaled=True`, apply volatility scaling (e.g. EWSD/ATR) and clipping as in the existing pipeline.

---

## Summary

- **Input:** Rule output **-1, 0, 1**; **target:** vol-scaled returns.  
- **Model:** Rule-based binning model; fit computes per-level Sharpe and adjusted Sharpe; predict returns **continuous** position multipliers via `clip(1 + adjusted_Sharpe, 0.5, 2.0)` for ±1 and 0 for flat.  
- **Reference:** Scaling details (adjusted Sharpe, clip range, shrinkage k) — see [Continuous_binning](Continuous_binning.md).
