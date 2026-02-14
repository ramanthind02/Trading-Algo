# Binning — Continuous Features (Spec & Info Sheet)

Binning turns continuous features into **trading zones** that capture **tail effects** and **humps** in the feature distribution where excess returns appear. This doc specifies a systematic, statistically defensible pipeline with a simple user API and a clear fit/predict interface.

---

## Purpose

- **Goal:** Use continuous features to exploit **tails** (e.g. RSI &lt; 22) and **humps** (e.g. RSI 45–60) where expected return reliably differs from zero.
- **Design principle:** Answer one question only — **where does expected return reliably differ from zero?**  
  Not: “where is Sharpe highest” or “where does the backtest look best.” Those invite selection bias.
- **Output:** For each bin (or merged region), output a **continuous value** (position multiplier), not just binary in/out. Risk scaling is explicit and bounded.

---

## User-Facing API (Simple)

The user configures:

| Parameter | Description |
|-----------|-------------|
| **`n_bins`** | Number of quantile bins (e.g. **15**). Fixed globally; do not optimize per feature. |
| **Trade direction** | `long`, `short`, or `long_short`. The model can operate in any of these modes; the mode only changes **which bins are kept** in the threshold filter (Step 5). |
| **Selection metric** | Objective used to rank bins: e.g. **mean return**, **Sharpe**, **Sortino**, or other metric. |
| **Metric threshold** | **Pre-specified** threshold (set before seeing any data). Bins **below** this (for long) or **above** this (for short) are eliminated. Only bins that clear the threshold are kept. See Safe Defaults for recommended values. |

The model receives:

- **Feature vector** (continuous).
- **Target vector** (e.g. forward returns). **Must be vol-scaled** (e.g. return / EWSD or return / ATR per observation) so that units are comparable across different markets and assets when training on multiple symbols in aggregate.

Binning is always on **raw** feature values; normalization/vol scaling is applied only to **signals** (e.g. in `predict`), consistent with the existing `BinningModelBase` design.

---

## Interface: fit / predict

The binning model follows the same pattern as the existing base:

- **`fit(feature_data, target_data, normalization_data=None)`**  
  - Quantile-bin the feature.  
  - Compute per-bin stats (mean return, volatility, optional t-stat, and the chosen selection metric).  
  - Optionally detect contiguous significant regions and merge into trading zones.  
  - Apply threshold by trade direction and store thresholds / regions / scaling parameters.

- **`predict(feature_data, strategy='long'|'short'|'long_short', normalization_data=None, scaled=False)`**  
  - Assign observations to bins (or regions) using fitted thresholds.  
  - Output **continuous** values per bin (e.g. position multiplier).  
  - If `scaled=True`, apply volatility scaling (e.g. EWSD/ATR) and clipping as in the existing pipeline.

Existing code reference: `BinningModelBase` and `QuantileBinningModel` in `feature_selection/base_models/base_model.py` and `quantile_binning.py` — same fit/predict contract; this spec extends with regions, t-stats, thresholding, and continuous risk scaling.

---

## Pipeline (Fully Automated)

### Step 1 — Fixed Quantile Binning

- Use **one** global bin count (e.g. **15 bins**). Do **not** vary by feature.
- Rationale: Varying bin counts is hidden hyperparameter search; consistency reduces false discoveries.
- Implementation: Quantile binning (equal sample count per bin), with the same fallbacks as `QuantileBinningModel` when duplicates collapse bins.

### Step 2 — Per-Bin Statistics

For each bin compute:

- `mean_return`
- `volatility` (e.g. std of returns in bin)
- **Sharpe** (e.g. mean_return / vol, optionally annualized). Unitless and **comparable across features and bins**. Requires the **target to be vol-scaled** (e.g. return / vol per row) so that comparison is valid across different markets when training on multiple assets.
- **Adjusted Sharpe** (shrinkage by sample size):  
  `adjusted_Sharpe = Sharpe * sqrt(N) / (sqrt(N) + k)`  
  Bins with more samples are trusted more; small-N bins are shrunk toward zero.
- **t-stat:** `t = mean_return / (vol / sqrt(N))`  
  With equal-N quantile bins, ranking by t-stat is a stable way to find where mean return is reliably non-zero.

Store the user-chosen **selection metric** (mean return, Sharpe, Sortino, etc.) per bin for thresholding and region detection. **Adjusted Sharpe** is used for the continuous position multiplier (Step 6).

### Step 3 — Contiguous Significant Regions

- Scan bins in order and group **adjacent** bins that satisfy:
  - `|t| > t_threshold` (e.g. **1.75** exploratory, **2.0** stricter).
- Enforce a **minimum region width** (e.g. **2 bins**, preferably 3). Discard isolated single-bin spikes.
- Result: “Trading zones” (e.g. “RSI &lt; 22 → long zone”, “RSI 45–60 → short zone”). We trade **regions of statistical structure**, not raw bins — more stable.

### Step 4 — Merge Regions into Trading Zones

- Represent each contiguous region by its feature range (min/max of the bins in that region).
- For prediction: map feature value → zone → continuous score (see Step 6).

### Step 5 — Threshold Filter (Direction-Aware)

The **mode** (long, short, or long_short) only affects this filtering step:

- **Long:** Drop bins (or regions) whose selection metric is **below** the user threshold. Keep only bins good for long.
- **Short:** Drop bins (or regions) whose selection metric is **above** the user threshold (we want “worst” return bins for shorting).
- **Long_short:** Apply direction-aware filtering for both sides; keep bins that pass the long threshold and bins that pass the short threshold (each with its own criterion).

Only bins/regions that pass the filter for the chosen mode are used in the final signal.

### Step 6 — Continuous Risk Scaling (Adjusted Sharpe)

Map edge to a **position multiplier** (continuous output per bin/region) using **adjusted Sharpe**:

- **Position multiplier:**  
  - **Long:** `clip(1 + adjusted_Sharpe, 0.5, 2.0)` (output in [0.5, 2.0]).  
  - **Short:** Invert the logic — output **negative** multipliers: `-clip(1 + adjusted_Sharpe, 0.5, 2.0)` (output in [-2.0, -0.5]).  
  - **Long_short:** Both apply; model output can range from **-2 to 2** (long bins → positive, short bins → negative).

  No per-feature rescaling.

- **Why Sharpe (and adjusted Sharpe) here:**  
  - **Unitless:** Already normalized by conditional volatility.  
  - **Comparable across features:** A Sharpe of 1.5 in one feature is directly comparable to 1.5 in another; relative edge strength is preserved when combining multiple features.  
  - **Signal vs noise:** High mean but very noisy → lower Sharpe; strong, stable mean → higher Sharpe.  


Do **not** allow unbounded sizing; clipping is risk control. Baseline exposure = 1; max tilt = 2.0. In long_short mode the **output range is -2 to 2**.

---

## Shape Detection (Automatic)

Do **not** manually choose “tail vs hump.” Let the data decide:

- **Tail:** Detected region touches an **extreme** bin (leftmost or rightmost).
- **Hump:** Detected region is **surrounded** by neutral (non-significant) bins.

No extra user input required.

---

## Multiple Regions

- **Both tails:** e.g. left tail positive, right tail negative → trade both (long one, short the other as per strategy).
- **Single hump:** e.g. middle hump positive → one long zone.
- **Reject non-contiguous “structure”:** e.g. bin 3 and bin 7 strong, nothing in between → treat as noise and reject (min region width and contiguity rules handle this).

---

## Threshold and t-Stat: Do Not Optimize

- **t-stat threshold:** Pick one (e.g. 2.0) and keep it global. Do not tune per feature.
- **Selection metric threshold:** **Pre-specify** one value (e.g. Sharpe > 0.5 or mean return > 0.1) BEFORE seeing any data. Apply the same threshold to all features and to both original and permuted data (for permutation testing). Do not optimize or adjust per feature.

**Critical for permutation testing:** The threshold must be set before looking at in-sample results. If you choose the threshold based on observed data (e.g. "use the 80th percentile of bin Sharpes"), you introduce data snooping and invalidate permutation tests.

Consistency beats clever tuning.

---

## Optional Upgrades

### False discovery control (FDR)

- Convert bin t-stats to p-values; apply **Benjamini–Hochberg** (or similar) to control expected false discoveries.
- Preferable to ad-hoc Sharpe/mean thresholds when testing many features.

### Bootstrap robustness

- Resample data (e.g. 200–500 times), recompute regions each time.
- Keep only zones that appear **consistently** across bootstrap runs. Fragile edges drop out quickly.

---

## Safe Defaults (Production-Like)

Use one set of defaults globally; do not tune per feature. **All thresholds must be pre-specified before seeing data.**

| Parameter | Default | Notes |
|-----------|---------|--------|
| **bins** | 15 | Fixed quantile binning |
| **t threshold** | 2.0 | \|t\| &gt; 2 for "significant" bin |
| **Min region width** | 2 bins | Prefer 3 to drop more noise |
| **Shrinkage (k)** | 20 | For adjusted_Sharpe: Sharpe × sqrt(N)/(sqrt(N)+k) |
| **Metric threshold** | **Sharpe > 0.5** or **mean > 0.1** | Pre-specified. Sharpe > 0.5 is conservative; Sharpe > 0.3 is exploratory. For mean return, depends on timeframe (e.g. daily: > 0.05, weekly: > 0.1). |
| **Selection metric** | Sharpe or mean return | Sharpe preferred (accounts for volatility). Mean return is simpler but ignores noise. |
| **Clip exposure** | [0.5, 2.0] long; [-2, -0.5] short | Long_short output range: **-2 to 2** |

**Recommended starting point:**
- Selection metric: **Sharpe**
- Metric threshold: **0.5** (conservative) or **0.3** (exploratory for in-sample screening before permutation tests)
- t threshold: **2.0**
- Min region width: **3 bins**


## Summary

- **User sets:** n_bins, trade direction, selection metric, metric threshold.  
- **Model does:** Quantile binning → per-bin stats (Sharpe, adjusted Sharpe, t-stat) → contiguous regions → direction-aware threshold filter → position_multiplier = clip(1 + adjusted_Sharpe, 0.5, 2.0); fit/predict as in current base.  
- **Philosophy:** Do not optimize signals; build **filters that reject unstable structure**. Use adjusted Sharpe for **universally comparable** position scaling across bins and features; no per-feature rescaling.

Pipeline in one line: **Detect → shrink → normalize → test → survive** — not “search → hope → overfit.”
