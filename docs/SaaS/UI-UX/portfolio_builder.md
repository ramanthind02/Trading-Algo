# Portfolio Builder (Compose tab)

**Parent:** [UI/UX README](README.md)  
**Route:** `/portfolios/{id}/compose` (old `/portfolio-builder` redirects here)

> **Navigation:** This is the **Compose tab** inside the portfolio hub — not a top-level nav item. See [portfolios.md](portfolios.md) for the full hub spec.

## Purpose

**Compose** a portfolio from **committed** strategies (weights, weight layer, holdout evaluation, deploy). Does **not** create the portfolio or set the portfolio zone — that is done at portfolio creation.

## Prerequisites

| Requirement | Where |
|-------------|--------|
| Portfolio exists | [Portfolios](portfolios.md) |
| Portfolio zone locked | At portfolio create |
| ≥1 committed strategy | Research step 6 or Strategy Library |

Entry: Portfolios card **Compose**, or Builder with portfolio picker if none selected.

## MVP screens

### Portfolio picker (if no `portfolio_id`)

Same list as [Portfolios](portfolios.md) — select one to compose. No zone editing.

### Composition editor

| Control | Behavior |
|---------|----------|
| Portfolio zone | Read-only banner (dates from portfolio) |
| Strategy selector | Committed versions only |
| Weights | Equal default; manual override |
| Weight layer | Select/lock per [`../weight_layer.md`](../weight_layer.md) before holdout |

### Holdout evaluation

- Uses **locked portfolio zone** only — no zone picker
- Portfolio backtest `scope: portfolio`
- [`../robustness_tests/portfolio_holdout.md`](../robustness_tests/portfolio_holdout.md)

### Commit version / Deploy

Unchanged — creates `PortfolioVersion`, optional [Deployment](deployment.md).

## vs Research vs Portfolios

| Area | Portfolio zone | Strategies |
|------|----------------|------------|
| **Portfolios** | Set once at create | — |
| **Research** | Read-only | Develop + commit drafts |
| **Portfolio Builder** | Read-only | Pick committed + weights |

## MVP vs future

| Feature | MVP | Future |
|---------|-----|--------|
| Compose + deploy | ✅ | Auto weight optimization |

See [mvp_scope.md](mvp_scope.md).
