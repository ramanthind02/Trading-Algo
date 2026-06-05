# QuantFoundry — Product Vision

> **Status: aspirational product vision.** This page describes the intended
> QuantFoundry SaaS product and its differentiators. None of these surfaces are
> built in the current repo — `Trading-Algo` today is a local research workbench
> with a Flask research UI (see `ui_ux.md`). Performance claims (vectorized
> sweeps, 10k-iteration permutation tests, streaming equity curves) are targets,
> not measured current behaviour. The brand/design-system docs referenced below
> are planned and not yet present in `docs/`.

## Core Differentiators

### 1. Performance
Existing platforms run parameter sweeps and permutation tests sequentially. We don't.

- **Vectorized batch computation** — a 100-parameter sweep runs as a single matrix operation, not 100 sequential backtests. Target: under 5 seconds.
- **Permutation tests** — 10,000 iterations in under 30 seconds via vectorized null distributions.
- **Research Cache** — cached runs are visible and interactive, not hidden. Re-running any cached parameter combination is instant.
- **Streaming results** — equity curve appears bar by bar as the backtest runs. No spinners.
- **Strategy compilation at save time** — code is compiled on save, not interpreted on every run.

### 2. UI/UX
Most algo platforms are built by engineers for engineers. QuantFoundry is built for researchers.

- **Progressive disclosure** — simple by default, complex on demand.
- **Zone timeline bar** — always visible, always shows what data you're touching.
- **Keyboard-first** — run, save, compare, switch zones without touching the mouse.
- **Run comparison built-in** — click any two past runs to overlay them. No separate mode.
- **Foundry aesthetic** — dark Ink shell, Paper type, Ember accents; **Space Grotesk** + **JetBrains Mono**; mark is `[Quant|Foundry]` with ember brackets and cursor. (Dedicated brand / design-system docs are planned but not yet present in `docs/`.)

### 3. Fast Lifecycle
Sign up to live signal generation in an afternoon.

- **Strategy templates** — five working strategies on first login. First experience is interactive, not educational.
- **Smart zone defaults** — data auto-split on project creation. Adjust if you want, skip if you don't.
- **Deploy without a broker** — Signal API is live from day one. Paper trade manually while you get comfortable.

## The Competitive Gap

| | QuantConnect | TradingView | QuantFoundry |
|---|---|---|---|
| Performance | Slow | N/A | Fast |
| UI/UX | Functional, ugly | Beautiful, shallow | Beautiful, deep |
| Time to deploy | Days | N/A | Hours |
| Researcher-focused | No | No | Yes |

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
