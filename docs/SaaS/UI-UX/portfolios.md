# Portfolios

**Parent:** [UI/UX README](README.md)  
**Routes:** `/portfolios`, `/portfolios/new`, `/portfolios/{id}`, `/portfolios/{id}/{tab}`

## What a portfolio is

A **portfolio** is the top-level container for everything the user builds. It holds:

- **Universe** — tickers + timeframe
- **Portfolio zone** — holdout dates, fixed at creation
- **Strategies** — all committed strategies researched inside this portfolio
- **Compositions** — weight configurations built from those strategies
- **Deployments** — live running instances with API keys

All strategy research, composition, and deployment is scoped to a portfolio. There is no global strategy library or global deployment list — those views exist inside the portfolio.

## Portfolio list (`/portfolios`)

Entry point after login (or from sidebar). No sidebar clutter — everything branches from here.

```
Portfolios                                          [+ New portfolio]

┌──────────────────────────────────────────────────────────────────┐
│ Prop Futures Core                           ● Running            │
│ ES, NQ, CL · Daily                                               │
│ 4 strategies  ·  1 deployment  ·  Updated 2d ago                 │
│                                    [Open →]                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Equity Momentum                             ○ No deployment       │
│ SPY, QQQ · Daily                                                  │
│ 2 strategies  ·  0 deployments  ·  Updated 1w ago                │
│                                    [Open →]                       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ No portfolios yet
│ Create your first portfolio to start researching strategies.
│                                    [+ New portfolio]              │
└──────────────────────────────────────────────────────────────────┘
```

**Card fields:**

| Field | Notes |
|-------|-------|
| Name | Portfolio name |
| Status pill | `● Running` (ember) / `○ No deployment` (muted) |
| Universe summary | Ticker list + timeframe |
| Strategy count | Committed strategies only |
| Deployment count | Active deployments |
| Last updated | Relative timestamp (JetBrains Mono, muted) |
| Open | One primary CTA per card → portfolio hub |

One action per card. No secondary action clutter. Everything available inside.

## Create portfolio wizard (`/portfolios/new`)

**The only place** the portfolio zone is set. Four steps, linear, no back-and-forth.

### Step A — Identity

| Field | Required |
|-------|----------|
| Name | ✅ |
| Description | Optional |

### Step B — Universe

| Field | MVP |
|-------|-----|
| Tickers | Multi-select (searchable) ✅ |
| Timeframe | Daily only ✅ |

### Step C — Portfolio zone (holdout)

| Element | Spec |
|---------|------|
| Label | **Portfolio zone** |
| Sublabel | "Your final holdout for evaluation. Cannot be changed after creation." |
| Control | Date range picker (inclusive UTC) |
| Timeline preview | `[ available data ][ ←── strategy development ──→ ][ PORTFOLIO ZONE ]` |
| Default | Last 20% of shared history across selected tickers |
| Validation | Non-empty; warn if bar count is thin |

### Step D — Review & confirm

```
Review
  Prop Futures Core
  ES, NQ, CL · Daily
  Portfolio zone: 2022-01-01 → 2024-12-31  (520 bars)

  [ ✓ ] I understand the portfolio zone cannot be changed.
        It applies to every strategy I research here.

                                     [ Create portfolio → ]
```

On submit: persist portfolio → lock zone immediately → redirect to portfolio hub (Overview tab).

## Portfolio hub (`/portfolios/{id}`)

The hub is a **tabbed page**. All portfolio actions live here — no jumping between top-level nav items.

```
← Portfolios

Prop Futures Core                                         ● Running
ES, NQ, CL  ·  Daily  ·  Portfolio zone: 2022-01–2024-12  🔒

[ Overview ]  [ Strategies ]  [ Compose ]  [ Deploy ]
────────────────────────────────────────────────────────
  (tab content below)
```

The portfolio zone, universe, and status are always visible in the header — users never lose context.

---

### Tab: Overview

At-a-glance health. No actions here — just signal and status.

| Section | Content |
|---------|---------|
| Portfolio zone | Date range + bar count + lock icon — read-only |
| Strategy summary | `N committed` · `N in research` — links to Strategies tab |
| Latest signals | Most recent signal per ticker (if deployed) — links to Deploy tab |
| Quick links | [Continue Research →] [View Deployment →] |

Empty state (no strategies yet):
```
Nothing here yet.
Start by researching your first strategy.

[ Open Research → ]
```

---

### Tab: Strategies

All committed strategy versions for this portfolio. Replaces the global Strategy Library.

```
Strategies                                      [ Open Research → ]

┌─────────────────────────────────────────────────────────────────┐
│ RSI Mean Reversion                                              │
│ ES, NQ  ·  Committed 2025-03-14  ·  v3                         │
│ Val SR: 1.24  ·  Test: passed                                   │
│                               [View] [Add to Compose →]         │
└─────────────────────────────────────────────────────────────────┘
```

| Field | Notes |
|-------|-------|
| Strategy name | |
| Universe | Tickers used |
| Committed date + version | JetBrains Mono |
| Val SR / test status | Summary validation metrics |
| View | Detail: version history, metrics, source code |
| Add to Compose | Pre-selects strategy in Compose tab |

Empty state:
```
No committed strategies yet.
Research a strategy and commit it from step 6.

[ Open Research → ]
```

---

### Tab: Compose

Build a portfolio version by combining committed strategies. Replaces Portfolio Builder.

**Prerequisites:** ≥1 committed strategy in this portfolio.

```
Compose                              Portfolio zone: 2022-01–2024-12  🔒

  Strategy weights
  ┌──────────────────────────────────────┬──────────┬──────────────┐
  │ Strategy                             │ Weight   │              │
  ├──────────────────────────────────────┼──────────┼──────────────┤
  │ RSI Mean Reversion v3                │  50%     │ [Remove]     │
  │ EWMAC Momentum v2                    │  50%     │ [Remove]     │
  └──────────────────────────────────────┴──────────┴──────────────┘

  [ + Add strategy ]

  Weight layer    [ equal_signal ▾ ]

  ─────────────────────────────────────
  Holdout evaluation (portfolio zone)
  [ Run evaluation ]

  — or —

  [ Deploy this composition → ]
```

| Control | Behavior |
|---------|----------|
| Strategy selector | Committed versions only; opens strategy picker modal |
| Weights | Equal default; manual override; must sum to 100% |
| Weight layer | Dropdown (MVP: `equal_signal` default) |
| Run evaluation | Backtests on locked portfolio zone |
| Deploy | Creates `PortfolioVersion` → opens Deploy tab |

---

### Tab: Deploy

Live deployments for this portfolio. Replaces the global Deployment page.

```
Deploy

┌─────────────────────────────────────────────────────────────────┐
│ v2  ·  equal_signal  ·  ES NQ CL                ● Running       │
│ Signals as of: 2026-05-18 16:00 UTC                             │
│                                                                  │
│ Ticker   Forecast   Position                                     │
│ ES       +0.42      Long                                         │
│ NQ       -0.11      Flat                                         │
│ CL       +0.78      Long                                         │
│                                                                  │
│ Signal API key: qf_live_•••••••••••        [Manage keys] [Stop] │
└─────────────────────────────────────────────────────────────────┘

[ + New deployment ]
```

**Deployment card fields:**

| Field | MVP |
|-------|-----|
| Version + weight layer + universe | ✅ |
| Status pill (Running / Stopped) | ✅ |
| Latest signals table (as_of + tickers) | ✅ when API returns data |
| API key prefix + Manage keys | ✅ |
| Stop | ✅ — confirm dialog before stopping |

**Signal API (manage keys modal):**

| Action | MVP |
|--------|-----|
| Create key (secret shown once) | ✅ |
| List keys (prefix only) | ✅ |
| Revoke key | ✅ |
| Docs + example snippet | ✅ |

Empty state:
```
No active deployment.
Compose your strategies and deploy when ready.

[ Go to Compose → ]
```

---

## Portfolio zone lock semantics

| State | Zone UI |
|-------|---------|
| Creating | Editable in wizard only |
| Created | Locked everywhere — 🔒 + dates + bar count |
| Research / Compose / Deploy | Read-only banner; no edit control |

If strategies exist: *"Portfolio zone is shared by N strategies. It cannot be changed."*

**API backstop:** even if UI locks at create, the API rejects `project_test_*` changes after `first_training_job_at` — defense in depth.

## MVP vs future

| Feature | MVP | Future |
|---------|-----|--------|
| List + create wizard + lock | ✅ | |
| Hub with 4 tabs | ✅ | |
| Strategy detail + metrics | ✅ | |
| Compose with equal weights | ✅ | |
| Deploy + API keys + signals | ✅ | |
| Clone portfolio with new zone | | ✅ |
| Edit universe after create | | ✅ |
| Auto weight optimization | | ✅ |
| Live performance charts | | ✅ |
| Monitoring suite | | ✅ |

## Relationship to Research

Research is never launched from the sidebar alone — it always has a portfolio context. Entry points:

- Portfolio hub → Overview tab → `[Continue Research →]`
- Portfolio hub → Strategies tab → `[Open Research →]`
- Sidebar Research item (shows active portfolio sub-label; picks up last step)
