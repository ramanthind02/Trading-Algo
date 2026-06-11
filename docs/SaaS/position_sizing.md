# Position Sizing

## 1. Purpose

This document describes how the platform converts a strategy's raw forecast signal into a tradeable contract quantity. The pipeline has four stages: signal-level volatility scaling, signal combination, portfolio-level sizing, and contract conversion. Each stage has a distinct multiplier that the researcher configures.

**Implementation status.** Stages 1–4 (§3–§6) are implemented in `Trading-Algo`
today and the formulas below match the code. §9 ("Account Simulation Model") is a
**forward-looking spec** — the `AccountState`, `MarginSpec`, margin-call, and
`AccountSimReport` machinery is design intent and is **not yet built**; the
current backtest/return path is the vectorized `research/portfolio/futures_sim.py`
simulation, which does not model margin, cash, or risk-free accrual (and does not
model transaction costs — see `transaction_costs.md`).

Related documents:
- `docs/SaaS/weight_layer.md` — FDM and weight layer methodology
- `docs/SaaS/portfolio_deployment.md` — IDM and portfolio-level fitting
- `docs/SaaS/transaction_costs.md` — cost spec (not yet wired into the return path)
- `docs/SaaS/robustness_tests/monitoring.md` — equity-curve / drawdown monitoring spec
- `docs/library/Ensemble/portfolio.md` — implementation reference for IDM / `TFPortfolio`

---

## 2. Pipeline Overview

```
Base model signal X_i ∈ [-1, 1]
        ↓  DiversifiedEnsemble: volatility scaling × τ
Per-signal forecast F_i ∈ [-2, 2]
        ↓  WeightLayer: signal combination × FDM
Combined forecast per ticker F_combined ∈ [-2, 2]
        ↓  TFPortfolio: instrument weighting × IDM
Position fraction ∈ [-cap, +cap]
        ↓  PositionSizer: capital × (price × multiplier × fx_rate)
Contracts (integer)
```

---

## 3. Stage 1 — Signal-Level Volatility Scaling

**Implemented in:** `ensemble/diversified_ensemble.py`

The DiversifiedEnsemble converts each base model's binary or continuous signal into a volatility-scaled forecast:

$$F_i = \frac{\tau}{\sigma_\text{inst} \times \sqrt{h_i}} \times X_i$$

| Variable | Meaning |
|---|---|
| $X_i$ | Raw base model signal ∈ [-1, 1]; binary strategies output {-1, 0, 1} |
| $\tau$ | Target volatility fraction. Configured as `target_volatility` on `DiversifiedEnsemble` (constructor default `0.15`; examples below use `0.25`). |
| $\sigma_\text{inst}$ | Rolling daily volatility of the instrument's returns |
| $h_i$ | Holding period of this signal in bars |
| $F_i$ | Volatility-scaled forecast for signal $i$ |

**Purpose of this step:** Normalize signals across instruments with different volatilities and signals with different holding periods. A signal on a volatile instrument (high $\sigma$) generates a smaller forecast magnitude than the same signal on a quiet instrument — because a fixed fraction of capital on the volatile instrument already carries more risk.

The denominator $\sigma_\text{inst} \times \sqrt{h_i}$ estimates the expected vol of a position held for $h_i$ bars. Dividing $\tau$ by this gives the position size that achieves the target vol contribution.

---

## 4. Stage 2 — Signal Combination (FDM)

**Implemented in:** `ensemble/weight_layer.py`

The WeightLayer combines all per-signal forecasts for a given ticker into a single combined forecast, then applies the Forecast Diversification Multiplier (FDM):

$$F_\text{combined} = \text{FDM} \times \sum_i w_i F_i$$

$$\text{FDM} = \min\!\left(\sqrt{\frac{1}{\bar{\rho}_\text{signals} + 0.01}},\ 2.0\right)$$

where $\bar{\rho}_\text{signals}$ is the mean pairwise correlation of the signals being combined, and $w_i$ are the weights assigned by the selected weight method.

**Purpose of this step:** Signals within the same ensemble are correlated. Combining them without adjustment would undersize the combined position relative to what a single well-chosen signal would produce. The FDM inflates the combined signal to restore the target volatility contribution, accounting for the imperfect correlation between signals.

FDM is capped at 2.0 (Carver's recommendation) to prevent excessive leverage when signals are very weakly correlated.

---

## 5. Stage 3 — Portfolio-Level Sizing (IDM)

**Implemented in:** `ensemble/portfolio_impl/tf_portfolio.py`

The TFPortfolio applies instrument weights and the Instrument Diversification Multiplier (IDM):

$$\text{position\_fraction} = F_\text{combined} \times w_\text{instrument} \times \text{IDM}$$

$$\text{IDM} = \min\!\left(\sqrt{\frac{1}{\bar{\rho}_\text{instruments} + 0.01}},\ 2.5\right)$$

where $\bar{\rho}_\text{instruments}$ is the mean pairwise correlation of portfolio instrument returns, and $w_\text{instrument}$ is the weight assigned to this instrument in the portfolio.

An optional position cap clips the output:

$$\text{position\_fraction} = \text{clip}(\text{position\_fraction},\ -\text{cap},\ +\text{cap})$$

**Purpose of this step:** The IDM applies the same diversification logic as FDM but at the instrument level. A portfolio of uncorrelated instruments can carry more total risk without exceeding the target volatility — the IDM captures this. Equal-weight instruments with mean correlation 0.3 produce an IDM of approximately 1.7.

IDM is capped at 2.5. The `position_fraction` is dimensionless — it represents the fraction of total capital allocated to this instrument.

---

## 6. Stage 4 — Contract Conversion

**Implemented in:** `execution/position_sizer.py`

The PositionSizer converts the position fraction to an integer contract count:

$$\text{target\_dollars} = \text{position\_fraction} \times \text{capital}$$

$$\text{contract\_value} = \text{price} \times \text{multiplier} \times \text{fx\_rate}$$

$$\text{contracts} = \text{round}\!\left(\frac{\text{target\_dollars}}{\text{contract\_value}}\right)$$

| Variable | Meaning | Example |
|---|---|---|
| capital | Total account capital in base currency | $500,000 |
| price | Current market price of the instrument | 5,400 (ES points) |
| multiplier | Dollar value per point of price movement | $50/point (ES standard) |
| fx_rate | Conversion from instrument currency to base | 1.0 for USD instruments |
| contract_value | Notional value of one contract | $270,000 |

**Rounding methods:** `ROUND` (standard, default), `FLOOR` (conservative — never over-allocate), `CEILING` (aggressive). For a small account where one contract is a significant fraction of capital, rounding error can be large — this is a minimum account size consideration, not a platform bug.

**Granularity consideration:** Micro futures contracts (e.g. MES = $5/point vs ES = $50/point) reduce the minimum position increment tenfold. For accounts below ~$50,000 trading standard ES, micro contracts significantly reduce rounding error. The canonical micro/mini dollar-per-point table lives in `lib/core/futures_micro_specs.py` (`canonical_listed_micro_futures()`); `micro_contract_fractional_and_whole(...)` is the shared sizing helper used by `research/portfolio/futures_sim.py` and the live prop forecast path, and `PositionSizer.from_listed_micro(...)` exposes it for live sizing.

> _Verified against the working tree on 2026-06-10._

---

## 7. End-to-End Example

Assumptions:
- Strategy: EWMAC on ES, binary signal, IS vol = 1.5% daily
- τ = 0.25, h = 20 bars
- Signal X = 1 (long)
- FDM = 1.3 (four signals combined, average correlation 0.45)
- Instrument weight = 0.2 (ES is 20% of portfolio)
- IDM = 1.6 (six instruments, average correlation 0.25)
- Capital = $500,000, ES price = 5,400, ES multiplier = $50

**Stage 1:**
$$F_i = \frac{0.25}{0.015 \times \sqrt{20}} \times 1 = \frac{0.25}{0.067} \approx 3.73 \rightarrow \text{clipped to } 2.0$$

**Stage 2:**
$$F_\text{combined} = 1.3 \times 2.0 = 2.6 \rightarrow \text{clipped to } 2.0$$

**Stage 3:**
$$\text{position\_fraction} = 2.0 \times 0.2 \times 1.6 = 0.64$$

**Stage 4:**
$$\text{target\_dollars} = 0.64 \times 500{,}000 = \$320{,}000$$
$$\text{contract\_value} = 5{,}400 \times 50 = \$270{,}000$$
$$\text{contracts} = \text{round}(320{,}000 / 270{,}000) = \text{round}(1.19) = 1$$

---

## 8. Configuration

The researcher configures the following per deployment:

| Parameter | Where set | Description |
|---|---|---|
| `τ` (target vol) | DiversifiedEnsemble | Annualised vol target as a fraction (e.g. 0.25) |
| `instrument_weights` | TFPortfolio | Per-ticker weights; equal-weight default |
| `IDM` | TFPortfolio (fitted or override) | Fit from IS correlation matrix; capped at 2.5 |
| `FDM` | WeightLayer (fitted) | Fit from IS signal correlations; capped at 2.0 |
| `capital` | PositionSizer | Account capital in base currency |
| `price` | PositionSizer (live) | Updated at signal generation time |
| `multiplier` | PositionSizer | From `ContractSpec`; fixed per instrument |
| `fx_rate` | PositionSizer | Fixed at 1.0 for USD-denominated futures |
| `rounding_method` | PositionSizer | ROUND / FLOOR / CEILING |
| `max_position_pct` | TFPortfolio | Optional cap on position_fraction per instrument |

All of these are stored in the `PortfolioSnapshot` at deployment time. The live signal pipeline uses the snapshot values; prices are the only input that updates bar-by-bar.

---

## 9. Account Simulation Model (forward-looking spec — not yet implemented)

> **Status:** This section is a design spec. The classes below (`AccountState`,
> `MarginSpec`, `AccountSimConfig`, `AccountSimReport`) do **not** exist in the
> current codebase, and the present return path does not simulate margin, cash,
> risk-free accrual, or margin calls. Today, returns are produced by the
> vectorized `research/portfolio/futures_sim.py` (`run_futures_sim`), which sizes
> positions and computes instrument returns but applies no account-level margin
> or cash mechanics and no transaction costs. Treat everything below as intended
> future behaviour.

The future backtest engine is intended to track a full account simulation rather than applying a single capital scaling factor. This section specifies the account state, margin mechanics, cash treatment, and daily settlement procedure.

### 9.1 Account State

At each bar the simulation maintains:

```python
@dataclass
class AccountState:
    equity: float           # total account value = cash + unrealized_pnl
    cash: float             # settled cash (earns risk-free rate)
    margin_posted: float    # initial margin locked as collateral
    unrealized_pnl: float   # mark-to-market P&L on open positions
    open_positions: dict[str, int]   # ticker → current contract count (signed)
    daily_pnl: float        # realised P&L for this bar (from settlement)
```

`equity = cash + unrealized_pnl` is the risk-bearing capacity of the account. `margin_posted` is a subset of `cash` that is locked as exchange collateral — it is still technically cash but unavailable for new positions.

### 9.2 Margin

Each instrument has an associated margin specification:

```python
@dataclass(frozen=True)
class MarginSpec:
    ticker: str
    initial_margin_per_contract: float      # dollars; required to open a position
    maintenance_margin_per_contract: float  # dollars; minimum to keep position open
```

Margin is posted when a new position is opened and released when the position is reduced or closed.

**Available capital for new positions:**

$$\text{available\_capital} = \text{equity} - \text{margin\_posted\_current} - \text{margin\_required\_new}$$

Before executing any rebalance, the platform checks whether available capital is sufficient to post initial margin for the new or expanded positions. If not, the order is scaled down to the largest position size that can be margined, and a capital constraint warning is surfaced.

**Margin call simulation:**

If at any bar `equity < maintenance_margin_posted` (i.e., losses have eroded equity below the maintenance threshold), the simulation records a margin call event. The position is closed at the next bar's open price, transaction costs are applied, and margin is released. The simulation does not attempt to add capital to meet margin calls — margin call events are surfaced as alerts to indicate the strategy requires more capital for the chosen leverage level.

### 9.3 Cash and Risk-Free Rate

The account holds two components: margin collateral and free cash. In practice, futures margin can be posted in T-bills, so the full account balance earns the risk-free rate. The simulation applies this:

$$\text{cash}_{t+1} = \text{cash}_t \times \left(1 + \frac{r_f}{252}\right)$$

where $r_f$ is the annualised risk-free rate. Free cash includes the portion posted as margin (consistent with the T-bill collateral treatment).

**Risk-free rate source:**

| Mode | Behaviour |
|---|---|
| `flat` | Fixed annual rate (default: 0.05 = 5%, researcher-configurable) |
| `historical` | Time-varying rate loaded from a rate series (e.g. 3-month US T-bill); researcher supplies the series |

The risk-free rate contribution is reported separately in the cost/performance report so the researcher can distinguish strategy edge from carry.

### 9.4 Daily Settlement (Futures Mark-to-Market)

Exchange-traded futures are marked to market daily. At the end of each bar:

1. Compute price change: $\Delta p_t = p_t - p_{t-1}$
2. Daily P&L per contract: $\text{pnl}_{t} = \text{contracts}_t \times \Delta p_t \times \text{multiplier} \times \text{fx\_rate}$
3. Cash settles: `cash += pnl_t` — gains flow into free cash; losses are debited from free cash
4. Unrealized P&L resets to zero after each daily settlement (positions are repriced at the settlement price)
5. Transaction costs are applied when positions change:

$$\text{trade\_cost}_t = |\Delta\text{contracts}_t| \times \text{cost\_per\_contract\_per\_side}$$

$$\text{net\_cash\_change}_t = \text{pnl}_t - \text{trade\_cost}_t + \text{rfr\_accrual}_t$$

### 9.5 Minimum Account Size Check

The platform surfaces a minimum capital estimate before the researcher runs a backtest. For each instrument in the portfolio, the minimum capital to trade one contract without a margin call is:

$$\text{min\_capital}_i = \frac{\text{initial\_margin}_i}{\text{position\_fraction\_max}_i}$$

where `position_fraction_max` is the maximum expected position fraction for that instrument (from IS statistics). The portfolio minimum is:

$$\text{min\_capital} = \max_i\!\left(\text{min\_capital}_i\right)$$

If the researcher's configured capital is below this estimate, the platform displays a warning: some bars may produce margin call events or positions scaled to zero. This is not a hard block — the simulation runs regardless and logs all constraint events — but it is a signal that the capital assumption is unrealistic for the chosen strategy.

### 9.6 Configuration

```python
@dataclass(frozen=True)
class AccountSimConfig:
    starting_capital: float                          # e.g. 500_000.0
    risk_free_rate_mode: Literal["flat", "historical"]
    risk_free_rate_flat: float | None                # e.g. 0.05; used if mode = "flat"
    risk_free_rate_series: pd.Series | None          # date → annual rate; used if mode = "historical"
    margin_specs: dict[str, MarginSpec]              # ticker → MarginSpec
    apply_margin_calls: bool                         # default True; set False to run unconstrained
```

### 9.7 AccountSimReport

```python
@dataclass(frozen=True)
class AccountSimReport:
    equity_curve: pd.Series          # daily total equity
    cash_curve: pd.Series            # daily cash balance
    margin_curve: pd.Series          # daily margin posted
    daily_pnl: pd.Series             # daily settled P&L (gross of costs and RFR)
    daily_costs: pd.Series           # daily transaction costs
    daily_rfr: pd.Series             # daily risk-free rate accrual
    net_pnl: pd.Series               # daily_pnl - daily_costs + daily_rfr
    margin_call_events: list[date]   # dates where equity < maintenance margin
    capital_constraint_events: list[date]  # dates where a position was scaled down
    gross_sharpe: float
    net_sharpe: float                # after costs
    total_sharpe: float              # after costs and including RFR
    rfr_contribution_pct: float      # RFR as pct of total return
    max_drawdown: float
    annualised_return: float
```

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
