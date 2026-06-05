# Transaction Costs

> **Status: forward-looking spec — NOT yet implemented.** Transaction costs are
> currently **not modelled** anywhere in the `Trading-Algo` vectorized return
> path. The portfolio simulation `research/portfolio/futures_sim.py`
> (`run_futures_sim`) computes instrument returns and positions but applies **no
> commission, slippage, or spread deduction**; the only "cost" it reports is the
> integer-rounding tracking error vs a fractional-contract baseline. The
> `SlippageConfig`, `CommissionConfig`, `InstrumentCostConfig`, and `CostReport`
> dataclasses below do **not** exist in the codebase. This document is the
> intended design for cost modelling; until it lands, treat all current backtest
> Sharpe/return figures as **gross** (pre-cost).

## 1. Purpose

This document specifies how transaction costs are intended to be modelled in QuantFoundry backtests. Accurate cost modelling is essential for realistic IS and OOS performance estimates — an uncostted backtest systematically overstates live performance, particularly for strategies with high turnover.

Related documents:
- `docs/SaaS/position_sizing.md` — contract quantities that determine cost magnitude (and the account-sim spec these costs would feed)
- `docs/SaaS/metrics_library.md` — gross vs net return conventions

---

## 2. Why Costs Matter

For a futures strategy with:
- Annual turnover of 5 round-trips per instrument
- ES standard contract at 5,400 points × $50 = $270,000 notional
- Slippage of 1 tick ($12.50) per side + commission of $3 per side

Cost per round-trip = ($12.50 + $3.00) × 2 sides = $31 per contract.

On a $500,000 account running 1 ES contract, that is 5 × $31 = $155/year — negligible at ~0.03% per year.

But for a strategy with 50 round-trips per year and a smaller account ($100,000) running the same contract: 50 × $31 = $1,550/year = 1.55% annual drag. At an uncostted Sharpe of 0.8, that drag is material.

The intended rule: **always run robustness tests on net returns**. Gross returns are available for comparison and to separate strategy quality from cost structure, but the Sharpe and t-stat that feed IS tests, DSR, validation, and holdout should be net once cost modelling lands. (Until then, those statistics are computed on gross returns — see the status banner above.)

---

## 3. Cost Components

### 3.1 Spread / Slippage

The bid-ask spread represents the cost of entering and exiting a position at market. For futures, this is expressed in ticks. The standard assumption is that the strategy pays the half-spread on each trade (entering at the ask, exiting at the bid, or vice versa).

**Configuration per instrument:**

```python
@dataclass(frozen=True)
class SlippageConfig:
    slippage_ticks: float    # ticks paid per side (e.g. 0.5 for half a tick)
    tick_size: float         # dollar value per tick (e.g. $12.50 for ES)
```

**Cost per trade per contract:**
$$\text{slippage\_cost} = \text{slippage\_ticks} \times \text{tick\_size}$$

Typical values for liquid US futures:

| Instrument | Tick size | Typical slippage assumption |
|---|---|---|
| ES (S&P 500) | $12.50 | 0.5–1.0 ticks/side |
| NQ (Nasdaq 100) | $5.00 | 0.5–1.0 ticks/side |
| GC (Gold) | $10.00 | 0.5–1.0 ticks/side |
| CL (Crude Oil) | $10.00 | 0.5–1.0 ticks/side |

Illiquid instruments, small-cap instruments, or instruments traded during off-hours warrant higher slippage assumptions.

### 3.2 Commission

Broker commission is a fixed fee per contract per side, independent of the instrument price.

**Configuration per instrument (or as a global default):**

```python
@dataclass(frozen=True)
class CommissionConfig:
    commission_per_contract: float    # dollars per contract per side (e.g. $2.50)
```

Typical values:
- Interactive Brokers (retail): ~$0.85/contract/side for US futures
- Interactive Brokers (tiered): ~$0.25–0.50/contract/side at higher volume
- Third-party introducing brokers: $2–5/contract/side

### 3.3 Total Per-Trade Cost

For each trade (each position change), the cost applied per contract traded is:

$$\text{cost\_per\_contract} = \text{slippage\_cost} + \text{commission\_per\_contract}$$

$$\text{total\_trade\_cost} = |\Delta\text{contracts}| \times \text{cost\_per\_contract}$$

where $|\Delta\text{contracts}|$ is the absolute change in contract count between the previous position and the new position.

---

## 4. Cost Application in Backtesting

At each bar where the position changes ($|\Delta\text{contracts}| > 0$), the platform deducts the trade cost from the bar's P&L:

```
gross_pnl_t  = (price_t - price_{t-1}) × multiplier × contracts_{t-1}
trade_cost_t = |contracts_t - contracts_{t-1}| × cost_per_contract
net_pnl_t    = gross_pnl_t - trade_cost_t
```

The return series used for all metrics:

$$r_t^\text{net} = \frac{\text{net\_pnl}_t}{\text{capital}}$$

---

## 5. Cost Configuration

### 5.1 Per-Instrument Config

Each instrument in the platform has a cost configuration. The researcher can use the platform defaults or override per strategy.

```python
@dataclass(frozen=True)
class InstrumentCostConfig:
    ticker: str
    slippage_ticks: float              # per side
    tick_size: float                   # dollars per tick
    commission_per_contract: float     # per side in dollars
    
    @property
    def cost_per_contract_per_side(self) -> float:
        return (self.slippage_ticks * self.tick_size) + self.commission_per_contract
```

### 5.2 Platform Defaults

The platform ships with cost defaults for each supported instrument based on standard retail broker assumptions. These are conservative (slightly high) — it is better to underestimate live performance than to overestimate it.

Defaults are shown in the backtest UI alongside results. The researcher can override them in the strategy config. Any override is stored in the strategy version metadata.

### 5.3 Zero-Cost Mode

The researcher can run a zero-cost backtest to see gross performance. This is available as an explicit option, not the default. Zero-cost results are labelled **GROSS** in the UI to prevent confusion with net results.

---

## 6. Cost Reporting

Each backtest run produces a cost breakdown alongside the standard performance metrics:

```python
@dataclass(frozen=True)
class CostReport:
    total_cost: float              # total cost over the backtest period in dollars
    annual_cost_drag_pct: float    # annualised cost as % of capital
    n_trades: int                  # number of individual trades (position changes)
    avg_cost_per_trade: float      # average cost per trade in dollars
    cost_config: InstrumentCostConfig  # what was assumed
    gross_sharpe: float            # uncostted Sharpe
    net_sharpe: float              # costted Sharpe
    sharpe_drag: float             # gross_sharpe - net_sharpe
```

The UI shows gross and net Sharpe side-by-side so the researcher can see how much cost drag the strategy carries at its current turnover level. A high `sharpe_drag` relative to `net_sharpe` signals that the strategy is marginally profitable only before costs — a fragile result.

---

## 7. Turnover Awareness

The cost drag of a strategy depends on turnover. A slow trend-following strategy may trade 5 times per year; a mean-reversion strategy may trade 50 times. The platform computes:

$$\text{annual\_turnover} = \frac{\text{total\_trades}}{T_\text{years}}$$

$$\text{break\_even\_sharpe\_drag} = \frac{\text{annual\_cost\_drag\_pct}}{\text{net\_vol}}$$

A strategy where cost drag exceeds 25% of the net Sharpe is warned in the IS robustness summary. This is not a hard gate — it is information. The researcher may accept higher costs on a high-Sharpe strategy or for instruments where no lower-cost alternative exists.

---

## 8. Deferred (Post-MVP)

**Market impact:** For large positions relative to average daily volume, simple slippage underestimates cost — the act of trading moves the market. Market impact models (square-root model or similar) are deferred. For liquid futures in the size ranges typical of retail systematic traders, simple slippage is a reasonable approximation.

**Roll costs:** Continuous futures contracts require periodic rolling from the expiring front month to the next. The roll cost depends on the carry structure (backwardation/contango) and spread at the roll date. Roll cost tracking is deferred; researchers should be aware that backtests using continuous backadjusted data absorb roll costs implicitly in the price series.

**Financing costs:** For leveraged positions exceeding 1.0× capital, financing costs apply. For futures which are inherently leveraged instruments, the carry is embedded in the futures basis rather than an explicit financing charge. Not modelled separately in MVP.

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
