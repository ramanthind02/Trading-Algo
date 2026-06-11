# Strategy Spec — the source of truth for a researched strategy

This document defines `StrategySpec`: the single, flat, declarative object that fully
describes one strategy to research, evaluate, and (if approved) store in the vault.

It is the contract an **agent** writes against (or the Spec Builder UI edits directly). The
agent/UI never edits the sprawling `research/feature/config.py` or
`research/portfolio/config.py`; it emits one `StrategySpec`, and a thin **adapter** translates
that spec into the existing pipeline configs and the Nautilus execution engine. The spec is
intentionally small and self-validating so an agent (or a human) can construct a complete,
correct strategy definition without internalizing the whole research stack.

The canonical entry point is **`research/specs/`** (round-trippable JSON shared by agent and
frontend). The Python API is `research.spec.{strategy_spec, serialization, adapter}`; the
adapter entrypoints called by the frontend are `to_feature_config` and `apply_vol_scaling`.

> [!note]
> Pipeline this rides on:
> `Candles → Bias nodes → DiversifiedEnsemble → WeightLayer → Portfolio → PositionSizer`,
> with execution simulated by `NautilusPnLEngine` (`research/portfolio/pnl/nautilus_engine.py`).

---

## 1. Philosophy — metrics guide, the human decides

There are **no automatic accept/reject gates** in this spec. The pipeline computes the full
metric panel (Sharpe / Sortino / Calmar, max drawdown, PSR, permutation p-values,
walk-forward degradation, turnover, cost drag, per-window breakdown) and the agent
**presents** it. A human approves or rejects every strategy.

Consequences that are baked into the design:

- **No `gates` field.** Metrics are advisory diagnostics, not goalposts.
- **No "optimize until a metric passes" loop.** Hill-climbing toward a threshold is data
  mining by another name. The parameter grid is an **exploration surface** — run once,
  reported in full (parameter-sensitivity, robustness) — not an optimizer target.
- **The test window is never used for selection.** It is scored once, at the end, for the
  approval decision. See [[Portfolio_research/holdout]] and [[SaaS/robustness_tests/portfolio_holdout]].

---

## 2. The `StrategySpec` object

Real `@dataclass(frozen=True)` (matches the codebase convention). The sketch below is the
**target**; the dataclass is implemented to match this doc, not the other way around.

```python
@dataclass(frozen=True)
class StrategySpec:
    # ── identity ──────────────────────────────────────────────────────────
    name: str                       # kebab/snake; becomes the vault ensemble stem
    hypothesis: str                 # one-paragraph economic rationale
    author: str
    created: datetime
    spec_version: str = "1.0"

    # ── universe & timeframe ──────────────────────────────────────────────
    tickers: tuple[Ticker, ...]
    data_feed: DataFeed             # NORGATE_FUTURES | DARWINEX_CFD
    mode: StrategyMode              # DAILY | INTRADAY
    timeframe: TimeFrame            # D/W/M (daily mode) | H1/M15/... (intraday mode)

    # ── windows (defaulted per mode when None) ────────────────────────────
    windows: ResearchWindows | None = None

    # ── signal & direction ───────────────────────────────────────────────
    signal: SignalSpec              # module_name + param_grid (<= 300 combos)
    direction: Direction            # LONG | SHORT | LONG_SHORT

    # ── volatility scaling ────────────────────────────────────────────────
    vol_scaling: VolScaling = VolScaling.BLENDED          # OFF | BLENDED | LONG_ONLY
    vol_scaling_model: VolScalingModel = VolScalingModel.INHERIT_DAILY  # intraday only

    # ── risk / sizing ─────────────────────────────────────────────────────
    risk: RiskSpec = RiskSpec()     # target_vol, forecast_cap, max_position_pct, buffer_fraction

    # ── account ───────────────────────────────────────────────────────────
    account: AccountSpec = AccountSpec()  # capital, multipliers/fx, prop_constraints (informational)

    # ── execution ─────────────────────────────────────────────────────────
    execution: ExecutionSpec = ExecutionSpec()  # entry/exit policy (market|limit), unfilled_limit, holding, fill_feed(derived)

    # ── vault target (used only on approval) ──────────────────────────────
    vault: VaultTarget              # weight_hierarchy_group + ensemble_name
```

### Sub-objects

```python
class StrategyMode(Enum):
    DAILY = "daily"        # maps to TimeFrame D / W / M
    INTRADAY = "intraday"  # H1, M15, ... (intraday timeframes)

class DataFeed(Enum):
    NORGATE_FUTURES = "norgate_futures"   # continuous back-adjusted futures
    DARWINEX_CFD = "darwinex_cfd"         # MT5 CFD feed (faithful % returns)

@dataclass(frozen=True)
class SignalSpec:
    module_name: str                      # bias-node taxonomy key (e.g. "rsi_signal")
    param_grid: dict[str, list]           # list-valued params -> cartesian grid expansion

@dataclass(frozen=True)
class ResearchWindows:
    train: tuple[datetime, datetime]
    validation: tuple[datetime, datetime]
    test: tuple[datetime, datetime]       # LOCKED — never used for selection

class VolScaling(Enum):
    OFF = "off"            # raw all-in/all-out signal, no F = τ/σ scaling
    BLENDED = "blended"    # σ = 0.7·short(32d EWMA) + 0.3·long(2520d) — the default
    LONG_ONLY = "long_only"  # σ = long-run only (blend 0.0 / 1.0), no short-term influence

class VolScalingModel(Enum):
    INHERIT_DAILY = "inherit_daily"     # σ from DAILY returns (default; see §5)
    INTRADAY_CUSTOM = "intraday_custom" # σ from intraday-bar returns (future plug-in)

@dataclass(frozen=True)
class RiskSpec:
    target_vol: float = 0.15      # τ in F = τ/σ
    forecast_cap: float = 2.0     # ± clamp on F
    max_position_pct: float = 3.5 # per-instrument notional cap
    buffer_fraction: float = 0.0  # Carver no-trade band (0 = always rebalance to target)

@dataclass(frozen=True)
class AccountSpec:
    capital: float = 50_000.0
    prop_constraints: PropConstraints | None = None  # informational; not an auto-gate

@dataclass(frozen=True)
class VaultTarget:
    weight_hierarchy_group: str   # one of the 13 sleeves (see §7)
    ensemble_name: str
```

---

## 3. Window defaults per mode

If `windows is None`, the adapter fills these fixed defaults from the canonical configs.
A spec may override them, but the **test window is always locked** out of selection.

| Mode | Train | Validation | Test (locked) |
|------|-------|------------|---------------|
| **DAILY** (D/W/M) | `2000-01-01 → 2018-12-31` | `2019-01-01 → 2022-12-31` | `2023-01-01 → latest` |
| **INTRADAY** (H1/M15/…) | `2018-01-01 → 2022-12-31` | `2023-01-01 → 2024-12-31` | `2025-01-01 → latest` |

Daily defaults match `research/portfolio/config.py` (`train_window` / `validation_window` /
`test_window`, lines ~768-779) and `research/feature/config.py` (`ResearchWindowConfig`).
Note: feature research itself has **no test window** by design — the locked test holdout only
runs in portfolio research and monitoring.

---

## 4. Signal & parameter grid

`signal.module_name` is a bias-node taxonomy key; `signal.param_grid` uses **list-valued
params** that expand to a cartesian grid (same shape the pipeline already consumes as
`bias_spec`). See [[bias_nodes/creating_nodes]] for the node interface (online
`_compute_candle(candle) -> List`, not vectorized).

**Hard rule — the grid must not exceed 300 combinations.** The product of the list lengths is
validated at construction; the dataclass raises `ValueError` if it exceeds 300. This caps both
overfitting surface and compute.

```python
# 4 × 3 × 2 = 24 combos — OK
SignalSpec(
    module_name="rsi_signal",
    param_grid={"rsi_period": [2, 3, 5, 7], "oversold": [20, 25, 30], "exit_bars": [3, 5]},
)
```

---

## 5. Volatility scaling semantics

The daily volatility model is EWSD (`nodes/volatility/ewsd/ewsd.py`): a **70 % short-run +
30 % long-run** blend, **not** a plain average.

- **Short** = 32-day EWMA on squared returns (λ = 0.06061), weight `0.7`
- **Long** = 2520-bar (~10 yr) expanding sample stddev, weight `0.3`
- Annualized ×16 (≈ √252); forecast `F = τ / σ`, clamped to ±`forecast_cap`.

`VolScaling` collapses the two real decisions into one field:

| Value | Scale by τ/σ? | σ estimator |
|-------|---------------|-------------|
| `OFF` | No | — (raw ±1 signal) |
| `BLENDED` | Yes | `0.7·short + 0.3·long` (default) |
| `LONG_ONLY` | Yes | long-run only (blend `0.0 / 1.0`) — removes short-term influence |

### Intraday: `vol_scaling_model`

Two strategy **modes** exist for vol scaling — daily (M/W/D) and intraday — and intraday has
two options:

- **`INHERIT_DAILY` (default, recommended).** Compute σ from the instrument's **daily**
  returns (the EWSD above) and use that σ to size the intraday position.
  **Precise definition:** σ comes from the *daily* return series, **not** from running the
  32-day/2520-bar spans on intraday bars — those spans would silently mean ~5 days / ~1.5 years
  on hourly data. Sizing is holding-period-agnostic (`F = τ/σ` sets instantaneous notional), so
  daily σ is the correct risk unit. For a flat-overnight strategy, daily (close-to-close)
  variance ≥ open-to-close variance, so this slightly **over**-estimates risk → **under**-sizes:
  conservative, the safe direction.
- **`INTRADAY_CUSTOM` (future plug-in).** σ computed on intraday-bar returns with
  intraday-appropriate EWMA spans and annualization. More precise (captures the open/close vol
  smile), more to get wrong. Not yet implemented; declare-only.

---

## 6. Execution model (Engine A)

A `StrategySpec` describes an **Engine A** strategy: a level signal whose target is reached by
**market or passive limit**, with **no per-trade stops**. Fill-sensitive bracket/scalping
strategies are **Engine B** — a separate, deferred state machine that is *not* a bias node and is
out of scope for the spec. The full rationale (and why we do not build the "node with intrabar
stops" middle) is in [[Strategy_research/execution_architecture]].

Execution is simulated by `NautilusPnLEngine` on real bar/quote data, producing honest fill
economics (`FillDiagnostic`: maker/taker, half-spread captured vs paid per fill). The spec selects
policies; it does not invent a cost model.

```python
@dataclass(frozen=True)
class ExecutionSpec:
    entry_policy: OrderPolicy = OrderPolicy.MARKET_ON_OPEN   # urgent: take the signal
    exit_policy:  OrderPolicy = OrderPolicy.MARKET_ON_OPEN   # patient: a limit can scrape spread
    unfilled_limit: UnfilledLimitPolicy = UnfilledLimitPolicy.CROSS_AFTER  # CROSS_AFTER | CARRY
    holding: Holding = Holding.OVERNIGHT   # OVERNIGHT (hold multi-day) | INTRADAY (flat outside session)
    fill_feed: FillFeed = FillFeed.DERIVED   # fine (M1/tick) iff any leg uses a limit, else signal_bar
```

`OrderPolicy` / `Holding` map onto existing enums in
`research/portfolio/pnl/nautilus_engine.py`:

| Spec | Engine enum | Values |
|------|-------------|--------|
| `entry_policy` / `exit_policy` | `ExecutionPolicy` | `MARKET_ON_OPEN`, `LIMIT_AT_TOUCH`, `LIMIT_IMPROVE` |
| `holding` | `ExecutionWindowPolicy` | `OVERNIGHT → CLOSE_TO_CLOSE`, `INTRADAY → INTRADAY_OPEN_TO_CLOSE` |

The engine's third value, `ROLLOVER_FLATTEN_REENTER`, is **not** a spec choice — it is the
portfolio-level **swap-avoidance overlay** (decision of record:
[[Data/feed_and_execution_decision]]), applied to the *net* book position per instrument, not
declared per strategy. See [[Strategy_research/execution_architecture]] (§ swap avoidance).

**Per-leg policy.** Entry is often signal-urgent (cross with a market order so you do not miss the
edge); the exit can be patient (rest a limit to scrape the spread). Today the engine uses one
policy for both legs — splitting it into `entry_policy` / `exit_policy` is a genuine addition.

**`holding`** is the alpha's intent, *not* the realized return convention:

- `OVERNIGHT` — the edge wants multi-day exposure (all daily/weekly strategies). The realized
  return convention (close-to-close vs rollover-bounded open-to-close) is set by the **portfolio
  swap policy**, which on CFDs defaults to rollover-bounded open-to-close per the decision of
  record. The spec does not set it.
- `INTRADAY` — the edge wants session-only exposure, flat outside the session (genuinely intraday
  alphas). No overnight gap at all.

**`unfilled_limit`** — what to do when a passive limit does not fill while the signal still wants
the target. `CROSS_AFTER(window)` works the limit for N minutes then crosses with market
(guarantees the move); `CARRY` leaves the position to the next signal bar (the rollover overlay's
"missed the fill → carry" behavior). The node never learns about the miss — the signal dictates
the target and execution chases it.

**`fill_feed` is derived and validated** from the leg policies:

- both legs `MARKET_ON_OPEN` → `fill_feed = SIGNAL_BAR`. The whole grid may run on the fast
  vectorized bar-spine lane; Nautilus market-on-open is an optional refinement.
- any leg `∈ {LIMIT_*}` → `fill_feed ∈ {BARS_M1, TICKS}` **required**. Every combo runs through
  `NautilusPnLEngine` on the fine feed, or the limit fill rots (assumes fills that never happened).

### When passive limits matter — and the compute implication

Passive limits help only when **spread is a material fraction of the per-trade edge** (wide-spread
CFDs, short holds, non-urgent mean-reversion). The CFD/Nautilus migration lifted test Sharpe
1.01 → 1.78 → 2.27 by modeling fills properly. They are **not free**: the rollover overlay found
market wins in the financing dead-zone — so this is a per-leg, per-strategy choice, never a
default.

Market grids are cheap (vectorized). Limit grids run every combo through Nautilus on M1/tick —
10–100× heavier — so the 300-combo cap bites hardest there. **Recommended pattern:** explore the
*signal* parameters cheaply on the vectorized market lane, then run only the **shortlist** through
the limit/Nautilus path to measure real fill economics. See [[Data/hybrid_tick_backtest]].

---

## 7. Vault target

On approval, the strategy is stored under the nested vault layout
`<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_name>_<direction>/`. The
`weight_hierarchy_group` must be one of the 13 manual sleeves
(`ensemble.vault.constants.VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES`):

```
mean_reversion_indices   buy_hold      es_tlt        seasonal
momentum                 trend_following  momentum_gc  crude_oil_mr
gc_breakout              cl_breakout   breakout      silver_mr   silver_trend
```

See [[Vault/vault]] and [[Vault/user_guide]] for the on-disk layout and save flow.

---

## 8. Adapter contract — how the spec maps onto the existing pipeline

The adapter is a pure translation layer. It never mutates the canonical configs; it builds
fresh config objects from the spec.

| `StrategySpec` field | Maps onto |
|----------------------|-----------|
| `tickers`, `timeframe` | `ResearchConfig.tickers` / `.timeframe`, `PortfolioResearchConfig.tickers` |
| `data_feed` | candle source selection (Norgate vs Darwinex MT5) — see [[Data/feed_and_execution_decision]] |
| `windows` | `ResearchWindowConfig` (feature) + `ResearchWindow` train/val/test (portfolio) |
| `signal.module_name` + `param_grid` | `bias_spec` dict with list-valued params (grid expansion) |
| `direction` | `in_sample_defaults.strategy` + node `strategy_mode` param |
| `vol_scaling` | EWSD blend weights + `DiversifiedEnsemble` τ/σ scaling toggle |
| `vol_scaling_model` | which return series feeds σ (daily vs intraday) |
| `risk.target_vol` / `max_position_pct` | `PortfolioSourceConfig.target_volatility` / `.max_position_pct` |
| `risk.forecast_cap` / `buffer_fraction` | `DiversifiedEnsemble` cap / position buffer |
| `account.capital` | `PositionSizer` capital / futures-sim profile |
| `execution.*` | `NautilusPnLEngine` `ExecutionWindowPolicy` / `ExecutionPolicy` / `CrossAfterPolicy` |
| `vault.*` | `VaultSaveConfig.weight_hierarchy_group` / `.ensemble_name` |

The adapter also **validates** the spec end-to-end: ≤300 combos, windows ordered
(train < validation < test), `weight_hierarchy_group` ∈ the 13 sleeves, and `fill_feed` consistent
with the leg policies (fine feed required when any leg uses a limit).

---

## Built status

- **`StrategySpec` dataclass + adapter** — fully implemented in `research/spec/` (Python) and
  mirrored as TypeScript interfaces in `frontend/web/src/api/types.ts`. Tests in
  `tests/unit-tests/spec/`.
- **Per-leg `entry_policy` / `exit_policy` + `CARRY`** — modelled in `ExecutionSpec` and
  serialized/deserialized; `NautilusPnLEngine` still uses a single policy for both legs (open item).
- **`research/specs/*.json`** — the shared round-trippable JSON contract; `load_spec` / `save_spec`
  in `research/spec/serialization.py` validate on every load.

## Open items

- **Per-leg split in `NautilusPnLEngine`** (today one policy for both legs; see
  [[Strategy_research/execution_architecture]]).
- **Two-bounds reporting** (c2c vs rollover-bounded o2c).
- **`INTRADAY_CUSTOM`** vol-scaling model.
- **Intraday `TimeFrame` enum members** beyond D/W/M (H1 is a member; M15 and finer are not yet).
- **Engine B** (bracket / scalping state machine on partitioned instruments) — deferred; *not* a
  bias node. See [[Strategy_research/execution_architecture]].

> _Authored 2026-06-06. Updated 2026-06-07 to reflect built state._

> _Verified against current code via CodeGraph on 2026-06-07._
