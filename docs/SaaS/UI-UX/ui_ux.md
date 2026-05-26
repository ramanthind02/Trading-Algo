# UI/UX Structure (Index)

> **Hub:** [README.md](README.md)

## Core principle

Portfolios are first-class citizens. Strategies, composition, and deployment are all managed **inside** the portfolio — not through separate top-level sections. The app has three navigation items.

## Navigation (3 items)

```
├── Dashboard      — cross-portfolio overview: deployment health, recent signals
├── Portfolios     — create, view, and manage portfolios (the hub for everything)
└── Research       — strategy development pipeline (always portfolio-scoped)
```

Details: [navigation.md](navigation.md)

## Quick links

| Area | Document |
|------|----------|
| Navigation & shell | [navigation.md](navigation.md) |
| **Portfolios hub** | [portfolios.md](portfolios.md) |
| Dashboard | [dashboard/mvp_ui.md](dashboard/mvp_ui.md) |
| Research pipeline | [research_workspace/user_flow.md](research_workspace/user_flow.md) |
| Brand guide | [brand.md](brand.md) · [design_system.md](design_system.md) |
| MVP scope | [mvp_scope.md](mvp_scope.md) |

## What lives where

| User goal | Where |
|-----------|-------|
| Create a portfolio | Portfolios → New portfolio |
| View portfolio health | Portfolios → {portfolio} → Overview tab |
| See committed strategies | Portfolios → {portfolio} → Strategies tab |
| Compose a portfolio version | Portfolios → {portfolio} → Compose tab |
| Deploy and manage API keys | Portfolios → {portfolio} → Deploy tab |
| Develop a new strategy | Research (portfolio-scoped) |
| See cross-portfolio overview | Dashboard |

## Portfolio hub (tabs)

The portfolio detail page is the hub. All portfolio actions live in four tabs:

| Tab | Route | Purpose |
|-----|-------|---------|
| Overview | `/portfolios/{id}` | At-a-glance: zone, strategy count, latest signals |
| Strategies | `/portfolios/{id}/strategies` | Committed strategy catalog for this portfolio |
| Compose | `/portfolios/{id}/compose` | Weight strategies + holdout evaluation + deploy |
| Deploy | `/portfolios/{id}/deploy` | Live deployments, signal tables, API keys |

## Research pipeline (steps 1–6)

Linear flow, always within a portfolio:

| Step | Route | Doc |
|------|-------|-----|
| Pick portfolio | `/research` | [project_selector.md](research_workspace/project_selector.md) |
| 1 Setup | `/research/{id}/setup` | [project_setup.md](research_workspace/project_setup.md) |
| 2 Editor | `/research/{id}/editor` | [strategy_editor.md](research_workspace/strategy_editor.md) |
| 3 Sweep | `/research/{id}/parameter-sweep` | [parameter_sweep.md](research_workspace/parameter_sweep.md) |
| 4 Robustness | `/research/{id}/robustness` | [strategy_robustness.md](research_workspace/strategy_robustness.md) |
| 5 Addition | `/research/{id}/portfolio-addition` | [portfolio_addition.md](research_workspace/portfolio_addition.md) |
| 6 Commit | `/research/{id}/commit` | [commit_strategy.md](research_workspace/commit_strategy.md) |

Details: [research_workspace/user_flow.md](research_workspace/user_flow.md)

## MVP scope

[mvp_scope.md](mvp_scope.md)
