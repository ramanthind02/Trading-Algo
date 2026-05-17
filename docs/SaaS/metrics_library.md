# Performance metrics library

## 1. Purpose

This document defines how **QuantFoundry** computes and exposes **raw performance and risk metrics** (e.g. Sharpe, Sortino, drawdown statistics) across research jobs, portfolios, and future API surfaces.

Scope for the MVP narrative here:

- **Scalar / tabular metric outputs only** — no dependence on HTML tearsheets or plotting for product correctness.
- **One canonical numerical contract** so workers, Core, and API responses stay aligned.

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

## 7. QuantStats-aligned table (`quant-foundry-core`)

The metrics **row vocabulary** from QuantStats `reports.metrics(..., display=False)` (basic vs full, compounded vs simple summed returns, optional benchmark columns) defines the baseline table for dashboards and APIs: Sharpe, Probabilistic Sharpe, Omega, VaR rows, streak stats, horizon returns (`MTD`, `3M`, …), drawdown summaries, ulcer/serenity, and—with a benchmark—R², information ratio, Treynor, and formatted Greek rows.

**Stable façade:**

- Module: **`quantfoundry_core.metrics`**.
- Entrypoint: **`compute_aligned_performance_metrics(...)` → `AlignedMetricsReport`** (`to_plain_dict()`, `to_dataframe()`).
- **Implementation:** an **in-repo** Apache-2.0-derived slice under `quantfoundry_core.metrics.vendor_qs` (no PyPI **`quantstats`** dependency). **`tabulate`** is a normal dependency for display-mode printing only.
- **Timezone:** UTC-aware indexes are stripped to UTC-naive before the metrics pipeline (mixed tz-aware alignment is brittle otherwise).
- **Parity:** `quant-foundry-core/tests/test_metrics_golden.py` asserts equality against **`tests/fixtures/quantstats_metrics_golden.json`** (generated once from QuantStats upstream for locked seeds).

**Catalog**

Static tuple **`BASIC_ROWS_STRATEGY_ONLY`** plus **`EXTRA_FULL_ROWS`** approximate the programmatic row titles for **`compounded=False`**. QuantStats substitutes **“Cumulative Return”** for **“Total Return”** when `compounded=True`; treat returned keys as authoritative.

---

## 8. NumPy-first engine (future / optional footprint)

Workers or stripped environments may swap the adapter body for pure NumPy/pandas helpers while preserving **`AlignedMetricsReport`** keys. Rough layering stays:

```
normalized returns + periods_per_year
       →
  primitives (Sharpe/Sortino/DD/… keyed like QuantStats rows)
       →
  immutable report dict / DataFrame
```

`Trading-Algo/metrics/` can seed primitives; conformance = same parity harness.

---

## 9. Open items (for future revisions)

- Exact **JSON schema** for metric bundles returned by QuantFoundry-API (names, rounding, nullable fields for short series).
- **Minimum sample size** policies (when to return `null` vs `0` vs error).
- **Risk-free rate** handling per currency and per bar frequency.
- **Benchmark-relative** metrics (beta, information ratio): second return series plus alignment rules.

When those are fixed, add a subsection here and mirror field lists in `technical_design.md` if exposed on HTTP APIs.
