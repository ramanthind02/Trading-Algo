# Rule-Based Features

> [!summary] What it is
> Feature is already discrete: a vector of **-1 / 0 / +1** from a rule (e.g. breakout, threshold crossing). No quantile binning — only Sharpe-based position scaling applied to the three fixed levels.

Inputs: rule output vector + vol-scaled target (see [[base_feature]]).

---

## Inputs

| Input | Description |
|-------|-------------|
| **Feature** | Integer vector of -1, 0, +1 (short / flat / long) |
| **Target** | Vol-scaled returns (e.g. `return / EWSD`) — unitless, cross-asset comparable |

---

## Fit Behavior

Treat the three levels as three fixed "bins." For each level, compute on vol-scaled target:

- `mean_return`, `volatility`
- `sharpe` = `mean_return / volatility`
- **Adjusted Sharpe** = `Sharpe × sqrt(N) / (sqrt(N) + k)`

Store per-level stats. No binning, region detection, or merging — only the scaling step.

---

## Position Scaling (Same Formula as [[continuous_binning]])

| Level | Formula | Output range |
|-------|---------|-------------|
| **+1 (long)** | `clip(1 + adjusted_Sharpe_long, 0.5, 2.0)` | [0.5, 2.0] |
| **-1 (short)** | `-clip(1 + adjusted_Sharpe_short, 0.5, 2.0)` | [-2.0, -0.5] |
| **0 (flat)** | `0.0` | 0 |

- `long_short` mode (both ±1 active): output range is **[-2, 2]**
- `short`-only mode: output range is [-2, -0.5]

---

## Predict

- Map each observation's rule value (-1, 0, +1) to the corresponding multiplier from fit
- If `scaled=True`: apply volatility scaling (EWSD/ATR) and clipping per existing pipeline

---

> [!info] See also
> - [[base_feature]] — shared target definition, Sharpe formula, and position multiplier spec
> - [[continuous_binning]] — adjusted Sharpe shrinkage details and clip range reference
