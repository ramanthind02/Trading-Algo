# P&L lanes and the final-validation lane

How a `position_fraction` frame (or a live vault) becomes a return series, the **three lanes** that do it, and the **lookahead-free** gold-standard check to run before promoting a candidate. Companion to [[Strategy_research/execution_architecture]] (which covers *how* a signal reaches a target — market vs passive limit) and [[Data/mt5_timezones]].

> [!important]
> One economic model, **three lanes** with different speed/fidelity trade-offs. They must agree.
> - **Vectorized** — fast pandas. The research workhorse (thousands of sweeps). Frozen parity baseline.
> - **Nautilus realistic** — event-driven `BacktestEngine`, real fills/spread. Opt-in; reconciled against the vectorized lane.
> - **Final-validation** — the *real live strategy* in a `BacktestEngine`, signal generated **on-the-fly**. Lookahead-free by construction. Run once per promotion candidate.
>
> The first two consume a **precomputed** `position_fraction` frame; the third **computes the signal in-loop**, so signal *and* execution share one causal clock.

---

## The three lanes

| Lane | Entry point | Signal source | Cost | Use for |
|---|---|---|---|---|
| **Vectorized** | `ensemble/portfolio_impl/portfolio_tester.py::calculate_strategy_returns_from_positions` (via `research/portfolio/pnl/pnl_engine.py::VectorizedPnLEngine`) | precomputed `positions_df` | ms–s | every research sweep / permutation; the frozen parity baseline |
| **Nautilus realistic** | `research/portfolio/pnl/nautilus_engine.py::NautilusPnLEngine` (`pnl_engine="nautilus"`) | precomputed `positions_df` | minutes | realistic fills/spread study, execution policy comparison |
| **Final-validation** | `research/validation/validation_lane.py::run_validation_backtest` / `scripts/validate_candidate.py` | **the real `VaultForecastEngine`, in-loop** | minutes (+ a one-time cache preflight) | the handful of strategies you actually promote |

You cannot run Nautilus on thousands of research backtests — it's far too heavy. So the vectorized lane exists for **throughput**, and the heavier lanes exist for **fidelity**. The discipline is that the fast lane must continuously *earn trust* by reconciling against the correct-by-construction ones.

---

## The causality contract (read this before touching a lane)

The same model is encoded in more than one place, so its conventions live in **one** module — `ensemble/portfolio_impl/backtest_conventions.py` — and every lane consumes that single definition rather than re-deriving it.

### Convention #1 — no-lookahead holding shift

A `position_fraction` stamped at session `t` is the position **decided** using information through `t`; it is **held over the next session `t+1`**. Equivalently, the position held during session `X` is the one decided at `X-1`.

- Vectorized lane: realizes `pos[t]` on the *next* candle (the `next_datetime` merge).
- Nautilus lane: `backtest_conventions.shift_positions_to_holding` (called by `_targets_by_session_date`).
- Final-validation lane: **automatic** — the event loop computes the signal at `on_bar(t)` (after `t` closed) and the order fills on the next bar, so the shift is an emergent property of the clock, not a hand-applied `.shift(1)`.

> [!warning]
> Omitting this shift is a **one-day lookahead** that silently *inflates Sharpe* (it suppresses realized vol). It was a real bug: the Nautilus adapter held `pos[t]` on day `t`, giving corr 0.90 / vol-ratio 0.79 vs the vectorized lane; with the shift, corr 0.9999 / vol-ratio 0.99. A constant target is shift-*invariant*, so it cannot expose this — **reconcile lanes on a *varying* signal** (`tests/research/test_nautilus_pnl_lane.py::test_nautilus_vs_vectorized_varying_signal_reconciles`).

### Convention #2 — intraday entry/exit prices (de-staled open)

Under the intraday window the entry is the first *tradeable* price of the session and the exit is the session close. The MT5 CFD feed's first M1 bar of a session **opens at the prior close carried across the 00:00–01:00 dead zone** — a stale, untradeable price — and reprices within that minute. So the genuinely tradeable open is that **first bar's close** (`data_platform/providers/mt5/cfd_candles.py::_first_tradeable_open`), and the daily bar's `open` is de-staled to it. Without this, `log(close/open)` is secretly ≈ close-to-close (it banks the overnight gap). See [[Data/mt5_timezones]].

### Nautilus as the correctness oracle

Nautilus is **event-driven**: a strategy physically cannot see a future bar, and the matching engine only fills against data at/after the order. That guarantees execution-time causality. It does **not** police *signal-timestamp alignment* — feeding it a target computed with future info still produces a clairvoyant backtest. So we treat Nautilus as the oracle the fast lane reconciles against, and the final-validation lane (signal *and* execution under one clock) closes the remaining gap the precompute split leaves open.

---

## The final-validation lane

`scripts/validate_candidate.py` runs the **real live** `VaultRebalanceStrategy` (`deployment/live/vault_strategy.py`) through a Nautilus `BacktestEngine` over historical data. The signal is produced in-loop by the real `VaultForecastEngine`, bounded to the simulation clock:

```python
# deployment/live/vault_strategy.py
return self._engine.evaluate(as_of=self.clock.utc_now())
```

`evaluate(as_of=...)` caps the candle window at `as_of` (`forecast_engine.py::_build_query`): a **no-op in live** (wall clock ≥ cache end) but in the backtest it's the sim clock, so the engine only ever sees candles `<= now`. Lookahead-free by construction — and because it runs the *exact* live code, a passing validation is also a **research↔live consistency check**.

> [!note]
> It's the **expensive** lane (the forecast engine refits per decision + a one-time cache preflight). Run it on candidates headed to production, not on research sweeps.

### Run it

```powershell
.\.venv\Scripts\python.exe scripts\validate_candidate.py `
    --vault-root vault --start 2024-01-01 --end 2025-01-01 `
    --tickers ES NQ GC CL SI
```

Output: resolved tickers, decision/fill count, and headline metrics (Sharpe, ann return/vol, total return) from the account equity curve.

### How it's wired

- **Instruments** — MT5-venue Nautilus instruments built from the data-store specs so they match what the live strategy resolves (`{native}.MT5`, native = `brokers.resolve(broker, canonical)`; broker defaults to Darwinex so native == the M1 data-store key, e.g. `NQ`→`NDX`).
- **Quotes** — synthesized from stored M1 closes, **converted to real UTC** (see gotcha below).
- **Signal data** — the `VaultForecastEngine` reads daily candles + bias artifacts from the central cache, **not** the backtest stream. `ensure_central_cache_coverage` preflights it from the canonical loader (research feed = CFD) into the CFD-namespaced cache; the default/live cache is untouched.
- **Equity** — a small daily-sampling actor records account equity → returns → metrics.

### Two gotchas it encodes (and that bit us)

1. **`prediction_daily_max_bars` must be `>` `warmup_min_bars`.** They are both `500` in the live config — a latent **off-by-one**: a ticker whose calendar yields one fewer session in the capped window returns 499 and fails the `>= 500` warmup gate forever. The lane uses `750`. *(Worth fixing in the live config too.)*
2. **Synth-quote timestamps must be real UTC.** The live decision/rollover-deadzone gates are UTC-reckoned (`deployment/live/runtime/broker_clock.py` derives ET + broker-EET via `zoneinfo`). Stored MT5 digits are broker EET/EEST mislabelled UTC; `_to_real_utc` re-interprets them in `brokers.broker_tzinfo(broker)` and converts to UTC (round-trips with `broker_clock`). Without it the decision timer never fires → no fills.

### Robustness

The decision loop now degrades to flat (logs, doesn't crash) if a forecast evaluation raises — a transient cache/data miss must not kill a live node.

---

## When to use which

- **Iterating / sweeping params** → vectorized (fast; the frozen baseline).
- **"Does spread/execution change the conclusion?"** → Nautilus realistic (reconciled to the vectorized lane on a varying signal).
- **"Is this candidate real, end-to-end, with no lookahead, exactly as live would run it?"** → final-validation, once, before [[Vault/user_guide|promotion]].

> _Added 2026-06-07. Grounded against `research/validation/validation_lane.py`, `deployment/live/{vault_strategy,forecast_engine}.py`, `ensemble/portfolio_impl/backtest_conventions.py`, and `research/portfolio/pnl/` via CodeGraph._

> _Verified against current code via CodeGraph on 2026-06-07._
