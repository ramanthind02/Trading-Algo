# Performance metrics library

## 1. Purpose

This document defines how performance and risk metrics (e.g. Sharpe, Sortino, drawdown statistics) are computed and exposed across research jobs, portfolios, and product API surfaces.

The authoritative implementation is the **external package `quantfoundry_core`** (installed dependency, imported as `quantfoundry_core.metrics` / `quantfoundry_core.robustness`). Scalar Sharpe/Sortino helpers live in `features.validation.objective_metrics` (themselves backed by `quantfoundry_core.metrics`); a few local equity/risk helpers live in `lib.metrics`. The former top-level `metrics/` compatibility shell has been **deleted**. See §7–§8.

Scope:

- **Scalar / tabular metric outputs** are the product-correctness contract — no dependence on HTML tearsheets or plotting.
- **One canonical numerical contract** so research code, workers, and API responses stay aligned.

Related documents:

- `docs/SaaS/zone_manager.md` — UTC time boundaries and zone slicing; metrics run on series produced from zone-filtered bars.
- `docs/SaaS/data_flow.md` — where research and evaluation phases produce return streams.
- `docs/SaaS/technical_design.md` — platform components.

Implementation detail (QuantStats-aligned metric table adapter): §7.

---

## 2. Canonical input: simple per-period returns

**Default convention:** metrics consume **simple (non-compounded) per-bar returns**

\[
r_t = \frac{W_t - W_{t-1}}{W_{t-1}}
\]

or the equivalent from an equity or mark-to-market level series, after fees and costs if the user requested a **net** series.

Properties:

- **Stable interface:** every KPI function takes the same kind of object: an ordered sequence of returns with a known **periods per year** for annualization (see §5).
- **Separation from sizing:** position sizing, contract counts, and dollar P\&L are simulated **upstream**. The metrics layer receives **returns**, not raw dollars, so Sharpe/Sortino/DD stay comparable across capital bases.

### 2.1 Named return conventions (API and Core)

To avoid silent mixing of semantics, any public contract should allow (or fix) an explicit enum, for example:

| Convention | Meaning |
|--------------|--------|
| `SIMPLE_PER_PERIOD` | Default. Simple return per bar. |
| `LOG_PER_PERIOD` | \(\ln(1 + r_t)\) per bar — optional for specialized stats; not the default for ratio KPIs unless documented. |

Gross vs net is orthogonal: **`PnlBasis.GROSS`** vs **`PnlBasis.NET_AFTER_COSTS`** metadata on how the return series was built.

---

## 3. Time indexing: UTC

All series passed into the metrics pipeline use timestamps interpreted in **UTC**, consistent with `zone_manager.md`.

- Daily, hourly, or other bar sizes are allowed; **bar size does not imply a specific annualization factor** — see §5.
- **Session semantics** (e.g. exchange calendar vs UTC calendar days) are defined where bars are built (Core / worker), not inside each metric formula.

---

## 4. Boundary: dollars and equity → returns

Metrics do **not** take dollar P\&L or contract ladders as primary inputs.

A dedicated **conversion layer** (Core or worker helper) produces `SIMPLE_PER_PERIOD` returns:

- From **equity curve** \(W_t\): e.g. `pct_change`-style increments with explicit NA policy aligned to the bar grid.
- From **PnL increments** \(\Delta W_t\) and lagged equity: \(r_t = \Delta W_t / W_{t-1}\).

That layer documents:

- whether costs are included;
- how missing bars are filled or omitted;
- initial capital anchor for the first period.

After conversion, downstream code is **numpy-only vectors** plus metadata (`periods_per_year`, convention enums).

---

## 5. Annualization and intraday bars

QuantFoundry must **not** hard-code “252 trading days” inside every metric without an explicit parameter.

Every annualized KPI accepts (or derives from a tagged `TimeFrame`) **`periods_per_year`**:

| Example bar type | Typical `periods_per_year` source |
|------------------|-----------------------------------|
| Daily (trading calendar) | Config or exchange calendar helper (often ~252 US equities). |
| Hourly / 4H / 1H | Product-defined: e.g. `TimeFrame.bars_per_year` in research code, or worker config for the asset class. |

**Intraday is first-class:** the same formulas apply to any evenly or irregularly spaced series as long as **returns are per bar** and **annualization is explicit**. Third-party libraries that assume “daily only” must not be the source of truth for SaaS metrics.

---

## 6. Simple vs compounded (wealth aggregation)

**Why keep simple returns as default for ratios**

- Sharpe, Sortino, and volatility are standardly defined from **per-period simple returns** (or sometimes log returns) with a clear annualization multiplier.
- Policy matches common research practice in the existing `Trading-Algo` stack (e.g. non-compounded tearsheet mode for Carver-style workflows).

**When compounded objects matter**

- **Cumulative wealth** is naturally multiplicative: \(W_T / W_0 = \prod_t (1 + r_t)\).
- That is a **derived series** for reporting or diagnostics, not a replacement for the default return convention.

**Conversions (small, explicit helpers)**

- Simple → log (per period): `log1p(r)`.
- Log → simple: `expm1(l)`.
- Simple returns → wealth index (normalize \(W_0 = 1\)): `cumprod(1 + r)`.

Do not mix conventions inside a single formula without naming it in code and docs.

---

## 7. QuantStats-aligned table (`QuantFoundry-Core`)

The metrics **row vocabulary** from QuantStats `reports.metrics(..., display=False)` (basic vs full, compounded vs simple summed returns, optional benchmark columns) defines the baseline table for dashboards and APIs: Sharpe, Probabilistic Sharpe, Omega, VaR rows, streak stats, horizon returns (`MTD`, `3M`, …), drawdown summaries, ulcer/serenity, and—with a benchmark—R², information ratio, Treynor, and formatted Greek rows.

**Stable façade — the authoritative metrics surface:**

- Module: **`quantfoundry_core.metrics`** (external package, verified present at commit a07b6bf).
- Entrypoints:
  - **`compute_aligned_performance_metrics(...)` → `AlignedMetricsReport`** for full/basic tables.
  - **`compute_scalar_metric(...)`** with **`MetricName`** for hot-path single-metric loops.
  - **`compute_rolling_sharpe(...)`** / **`compute_monthly_returns_heatmap(...)`** for time-series chart payloads.
- Artifact shape: prefer **`AlignedMetricsReport.to_json_dict()`** for worker/API persistence; materialize DataFrames only at display/export boundaries.
- Contract helpers: **`ReportMode`**, **`ReturnsCompounding`**, **`ReturnsValidationError`**, **`MetricsSchemaError`**, and **`returns_fingerprint(...)`** (all exported from `quantfoundry_core.metrics`).
- **Timezone:** UTC-aware indexes are stripped to UTC-naive before the metrics pipeline (mixed tz-aware alignment is brittle otherwise).
- **Usage rule:** application code must call the public `quantfoundry_core.metrics` surface and must not import private QuantStats internals directly.

**In-repo consumers (verified):**

- `analysis/plotting/graphing/quantstats_reports.py` imports `AlignedMetricsReport`, `ReportMode`, `ReturnsCompounding`, `compute_aligned_performance_metrics` directly from `quantfoundry_core.metrics`.
- `research/feature/research_table_exports.py` imports `ReturnsValidationError`, `compute_rolling_sharpe`.
- `features/validation/objective_metrics.py` imports `compute_scalar_metric`, `MetricName`, `ReturnsCompounding`, `ReturnsValidationError` and wraps them as the `metric_sharpe`/`metric_sortino`/`metric_calmar`/… scalar helpers (with local edge-case fallbacks).

**Catalog**

Static tuples **`BASIC_ROWS_STRATEGY_ONLY`** plus **`EXTRA_FULL_ROWS`** (both exported by `quantfoundry_core.metrics`) approximate the programmatic row titles for **`compounded=False`**. QuantStats substitutes **“Cumulative Return”** for **“Total Return”** when `compounded=True`; treat returned keys as authoritative.

---

## 8. Local `analysis.metrics` helpers

There is **no** repo-local barrel that re-exports performance formulas — the old top-level `metrics/` package (including its `metrics/__init__.py` shell and `metrics/performance/` scalar-class wrappers `SharpeRatio`/`SortinoRatio`) has been **deleted**. Performance scalars come from `features.validation.objective_metrics` (`metric_sharpe`, `metric_sortino`, …, backed by `quantfoundry_core.metrics`).

The only local helpers are the small risk/equity utilities re-exported by `analysis/metrics/__init__.py`:

- `cumulative_returns`, `equity_peak` (from `analysis/metrics/equity.py`), and
- `drawdown_series`, `max_drawdown` (from `analysis/metrics/drawdown.py`).

(The former `equity_curve` alias was dropped.) Plotting/tearsheet code under `analysis/plotting/graphing/` consumes `quantfoundry_core.metrics` directly.

Layering for any pure-NumPy/pandas worker variant must still preserve **`AlignedMetricsReport`** keys:

```
normalized returns + periods_per_year
       →
  primitives (Sharpe/Sortino/DD/… keyed like QuantStats rows)
       →
  immutable report dict / DataFrame
```

---

## 9. Open items (for future revisions)

- Exact **JSON schema** for metric bundles returned by QuantFoundry-API (names, rounding, nullable fields for short series).
- **Minimum sample size** policies (when to return `null` vs `0` vs error).
- **Risk-free rate** handling per currency and per bar frequency.
- **Benchmark-relative** metrics (beta, information ratio): second return series plus alignment rules.

When those are fixed, add a subsection here and mirror field lists in `technical_design.md` if exposed on HTTP APIs.

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
