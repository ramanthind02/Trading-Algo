# Strategy engineering — from a brief to a parsimonious spec

How the agent turns a strategy *idea* into a well-designed `StrategySpec`. This is the
design-judgment layer **upstream** of [[Strategy_research/strategy_spec]] (the formal object) and
[[Strategy_research/research_report]] (how results come back). The whole pipeline:

```
brief (markdown) → [this doc: classify · study · design parsimoniously] → StrategySpec → research → memo
```

> [!important]
> **The first defense against overfitting is the design, not the validation.** Favor strategies
> with **few parameters**, **simple constructs**, and a **clear economic effect captured robustly**.
> If a simple moving average captures the effect, do **not** reach for a ten-variable MA
> approximation. Validation (the report lenses) is the back-line check; parsimony is the front line.

---

## 1. The brief (intake)

A strategy run starts from a **markdown brief** the user hands the agent — the context for the
strategy being worked on. It typically carries: the idea, the **economic rationale**, a category
hint, any reference (paper, blog, prior result), the target instruments, and constraints. The
agent ingests this as its starting context and drives the design from it.

The brief is prose and informal; the agent's job is to convert it into a disciplined design and
then a `StrategySpec`. If the brief is thin, the agent fills gaps from the taxonomy and from what
already works (§3) — and states the assumptions it made.

---

## 2. Strategy taxonomy

Most strategies fit a category, and each category maps to an existing **node folder** and a **vault
sleeve** — so the agent always has real house-style examples to study. (Folders confirmed under
`nodes/`; canonical category map in `nodes/_taxonomy.py`; sleeves in
`ensemble.vault.constants.VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES`.)

| Category | Economic effect | Simplest construct | Node folder | Vault sleeve(s) | Example nodes |
|----------|-----------------|--------------------|-------------|-----------------|---------------|
| **Trend following** | persistent drift / positive autocorrelation | EWMA crossover; SMA regime | `nodes/momentum/`, `nodes/regime/sma/` | `trend_following`, `silver_trend`, `momentum`, `momentum_gc` | `ewmac`, `sma_regime_signal`, `supertrend_cross` |
| **Mean reversion** | overextension reverts | RSI / %B threshold; z-score | `nodes/mean_reversion/` | `mean_reversion_indices`, `crude_oil_mr`, `silver_mr` | `rsi_signal`, `percent_b_signal`, `scaled_z_mr`, `double7s`, `williamsr_signal` |
| **Breakout** | range expansion / new extremes | Donchian channel; close > prior | `nodes/breakout/` | `breakout`, `gc_breakout`, `cl_breakout` | `donchian_long_only`, `close_breakout`, `robust_trend_breakout` |
| **Calendar / seasonal** | recurring date effects | day-of-week / month / holiday gate | `nodes/seasonal/` | `seasonal` | `turnaround_tuesday`, `seasonal_indices_eof`, `pre_holiday_equity`, `fomc_drift` |
| **Flow effects** | predictable rebalancing flows | scheduled flow signal | `nodes/pairs/rebalancing_flow.py` | — | `rebalancing_flow` |
| **Pairs / spread** | relative value of two legs | spread z-score | `nodes/pairs/spread.py` | `es_tlt` | `spread` |
| **Buy & hold** | long-run risk premium | constant long | `nodes/buy_hold/` | `buy_hold` | core |

Classifying the brief into a category tells the agent **which folder to study, which sleeve it
would join, and which simple construct is the default starting point.**

---

## 3. Study what already works (before designing)

The agent surveys existing strategies first — for inspiration, house style, and to avoid
redundancy:

- **Is there already a node** that does this (or most of it)? Reuse or extend before inventing.
- **What is the house pattern** for this category? (e.g. mean-reversion nodes emit a long entry on
  an oversold threshold with a bars-or-threshold exit; trend nodes are EWMA/SMA based.) Match it.
- **What is already in the sleeve**, and how would this correlate with it? A near-duplicate of an
  existing winner adds little — the diversification lens (research_report § F) will catch it, but
  it is cheaper to notice at design time.
- **What is working well** in the vault — use winners as templates, not as things to clone.

Node implementation mechanics live in [[bias_nodes/creating_nodes]]; this step is about *design
intent*, not code.

---

## 4. Design principles — parsimony

The core discipline. Every one of these directly serves the overfit defense and the report lenses
(A simplicity, B plateau, C overfit).

1. **Fewest parameters that capture the effect.** A parameter is a degree of freedom and a chance
   to overfit. The house favors very few: `close_breakout` has **zero** parameters;
   `sma_regime_signal` has **one** (often fixed at 252); `ewmac` has **two** (fast/slow span);
   `turnaround_tuesday` is a calendar rule with essentially none. Aim for **1–3 swept parameters**.
2. **Simplest construct that works.** If an SMA captures the trend, use the SMA — not a complex MA
   approximation that folds in ten other variables. Complexity must *earn its place* with a real,
   explainable improvement, not a backtest bump.
3. **Robust capture over maximal fit.** The goal is to capture the effect *durably across regimes
   and instruments*, not to maximize in-sample Sharpe. A broad plateau beats a sharp peak.
4. **Fix the parameters that do not need optimizing.** The agent should **fix** any parameter it
   judges non-critical — standard RSI thresholds (30/70), a conventional exit horizon, a 252-day
   trend filter — and **state why**. Every fixed parameter removes a grid dimension and a
   degree of freedom. Only sweep what genuinely needs exploring.
5. **Small search space.** Fewer swept params and shorter value lists → a small grid. The ≤300-combo
   cap is a ceiling, not a target — most good designs are **well under ~50 combos**. A large grid
   is a design smell.
6. **Every parameter needs an economic "why."** If you cannot say what a parameter *means* (not
   what value backtests best), it should be fixed or removed. Parameters without meaning are pure
   overfit surface.

---

## 5. From brief to spec — the workflow

1. **Ingest** the brief; restate the idea, the economic rationale, and the target instruments.
2. **Classify** the category (§2) → node folder, sleeve, default construct.
3. **Study** existing nodes in that folder + the sleeve's current members (§3); note reuse,
   house pattern, and likely correlations.
4. **Design** the simplest construct that captures the effect (§4.2). Prefer reuse/extension of an
   existing node over a new one.
5. **Choose the minimal grid**: 1–3 swept parameters, each economically motivated; **fix the rest
   with stated justification** (§4.4). Confirm the grid is small (§4.5).
6. **Write the `StrategySpec`** ([[Strategy_research/strategy_spec]]) — universe, holding, vol
   scaling, execution, sleeve target.
7. **Research** it and report ([[Strategy_research/research_report]]).

---

## 6. Anti-patterns

- **Kitchen-sink indicators** — stacking many conditions to lift the backtest. Each condition is a
  fitted degree of freedom.
- **Optimizing every parameter** — sweeping params that have a sensible conventional value. Fix
  them.
- **Complexity where simple works** — the ten-variable MA where an SMA suffices.
- **Parameters with no economic meaning** — knobs that exist only to be tuned.
- **Redundant strategies** — a near-clone of an existing sleeve member; adds search/compute and
  little diversification.
- **A large grid** — many params × many values. A small, deliberate grid is the design goal.

---

> _Authored 2026-06-06. Taxonomy grounded against the `nodes/` folder structure
> (`mean_reversion/`, `breakout/`, `momentum/`, `seasonal/`, `pairs/`, `buy_hold/`),
> `nodes/_taxonomy.py`, and the vault sleeves in `ensemble.vault.constants`._

> _Verified against current code via CodeGraph on 2026-06-07._
