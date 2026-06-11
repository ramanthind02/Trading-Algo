# Agent Handoff — 2026-06-07

## Context

Branch: `mt5_data`. Two unrelated tasks were completed this session.

---

## Task 1: EDA Parallelization (COMPLETE)

**Problem**: Exploration runs were slow (~10 s/combo × up to 25 combos ≈ 4 min total).  
**Root cause**: The EDA bootstrap loop in `research/feature/pipelines/in_sample.py` ran sequentially despite `ResearchConfig.n_jobs=8`.

**Fix applied**:

1. **`lib/core/research_feed.py`** — `set_research_feed()` now stamps `RESEARCH_DATA_FEED` and `RESEARCH_FUTURES_TICKERS` env vars so loky worker processes inherit the correct feed (in-memory dict is not inherited by loky).

2. **`lib/compute/daily_ewsd_volatility.py`** — `set_ewsd_blend_weights()` / `reset_ewsd_blend_weights()` now stamp `EWSD_BLEND_WEIGHTS` env var; `_default_ewsd_config()` reads it as fallback so workers see the correct EWSD blend.

3. **`research/feature/pipelines/in_sample.py`** — New module-level function `_run_eda_for_single_combo(...)` (must be at module level for loky pickling). The EDA for-loop was replaced with `joblib.Parallel(n_jobs=min(n_jobs, len(eda_tasks)), backend="loky")` when `n_jobs > 1`. Falls back to sequential if `n_jobs=1` or only one task.

**Status**: Syntax-checked, not yet run end-to-end with a real spec. User should test by running any exploration spec with the frontend.

---

## Task 2: CFD vs Futures Metric Discrepancy (DIAGNOSED — no bug)

**User concern**: Running `double7s_rerun` spec on the Darwinex CFD feed showed Sharpe 0.39, NW t-stat 1.52, DSR 89.1% — much lower than the futures run (Sharpe 0.68, t-stat 3.63, DSR 99.8%). The equity curve appeared to "start at 2010" rather than the training window start. User suspected the equity curve was showing the validation window.

**Confirmed findings** (via direct CSV inspection):

- `equity_curve.csv` correctly ends at **2018-12-28** (within training window 2000-01-01 to 2018-12-31)
- CFD data for ES/NQ from Darwinex starts ~2007. After 200-bar MA warmup (~10 months), first active date is ~mid-2008.
- 2008-2009: bear market, price below 200-MA → double7s never fires → equity curve is flat → visual appearance of "starting at 2010"
- Effective CFD history: ~10.5 years (2008-2018) vs ~19 years for futures (2000-2018)
- The 2000-2007 bull market (best conditions for double7s) is entirely absent from CFD history
- This explains the full metric gap — **no code bug**

**Current `equity_curve.csv`** (overwritten by subsequent futures run):
- ES: 4782 rows, 2000-01-04 → 2018-12-28, final_cumret=0.62, n_active=1280
- NQ: 4782 rows, 2000-01-04 → 2018-12-28, final_cumret=0.71, n_active=1283

---

## Pending / Open Items

- **Verify EDA parallelism end-to-end**: Run an exploration spec through the frontend and confirm the `[n_jobs=8]` message appears in the run log and the run completes faster. Watch for any pickling errors from joblib (would surface as a traceback in the run output).
- **User decision on CFD vs futures for double7s research**: The strategy looks much better on futures (19 years, includes 2000-2007 bull market). If user is trading CFDs, the CFD Sharpe (~0.39) is the realistic expectation. User should decide which feed to use for vault promotion.

---

## Key Files Modified This Session

| File | Change |
|------|--------|
| `lib/core/research_feed.py` | Stamps env vars on `set_research_feed` for loky worker inheritance |
| `lib/compute/daily_ewsd_volatility.py` | Stamps `EWSD_BLEND_WEIGHTS` env var for loky worker inheritance |
| `research/feature/pipelines/in_sample.py` | EDA loop parallelized via joblib; new `_run_eda_for_single_combo` module-level function |

---

## Architecture Reminder

- `ResearchConfig.n_jobs: int = 8` — already the default; controls both EDA parallelism and (if enabled) permutation tests
- `PermutationResearchConfig.n_jobs_combos` — separate, only for permutation tests (disabled by default)
- Process-global state (`research_feed._state`, `daily_ewsd_volatility._BLEND_OVERRIDE`) must be propagated via env vars to reach loky workers
- The frontend runs exploration via `POST /api/runs` → `research/feature/pipelines/in_sample.py`
