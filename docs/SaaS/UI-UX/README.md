# QuantFoundry UI/UX Documentation

Product navigation, page specifications, shared components, and design tokens for **QuantFoundry-Web**.

## Related documents

| Document | Role |
|----------|------|
| [`../technical_design.md`](../technical_design.md) | API entities, hosting, Signal API |
| [`../data_flow.md`](../data_flow.md) | Research → portfolio → deployment lifecycle |
| [`../zone_manager.md`](../zone_manager.md) | Zone types, UTC boundaries, Core vs API |
| [`../product_vision.md`](../product_vision.md) | Differentiators, aesthetic direction |
| [`../robustness_tests/`](../robustness_tests/) | Statistical specs referenced by research UI |

## Core principle

Portfolios are first-class citizens. Three nav items. Everything else lives inside the portfolio.

```
Dashboard → cross-portfolio overview
Portfolios → the hub: view, strategies, compose, deploy
Research → strategy pipeline (always portfolio-scoped)
```

## Document map

### Platform shell

| Page | Description |
|------|-------------|
| [Navigation](navigation.md) | 3-item sidebar, routing, portfolio hub concept |
| [Brand v1.0](brand.md) | Logo, colors, type, exports |
| [Design system](design_system.md) | App tokens, components, dark UI mapping |
| [MVP scope](mvp_scope.md) | What ships vs what's deferred |
| [Components index](components/README.md) | Shared component IDs and cross-links |

### Portfolios (hub)

| Page | Description |
|------|-------------|
| **[Portfolios](portfolios.md)** | **List, create, and the tabbed hub (Overview / Strategies / Compose / Deploy)** |

The portfolio hub consolidates what were previously three separate areas:

| Hub tab | Was |
|---------|-----|
| Strategies tab | Strategy Library |
| Compose tab | Portfolio Builder |
| Deploy tab | Deployment |

### Dashboard

| Page | Description |
|------|-------------|
| **[Dashboard MVP UI](dashboard/mvp_ui.md)** | **Start here** — cross-portfolio cards |
| [Dashboard overview](dashboard/README.md) | Purpose, principles, layout |
| [Page layout](dashboard/page_layout.md) | Sections A–D, responsive grid |
| [Deployment summary card](dashboard/deployment_summary_card.md) | Portfolio card component |
| [Empty states](dashboard/empty_states.md) | Zero deployments, zero projects |
| [API contract](dashboard/api_contract.md) | `GET /api/deployments`, polling |
| [Dashboard phase 2](dashboard/phase_2.md) | Sparklines, monitoring badges |

### Research Workspace

| Page | Description |
|------|-------------|
| **[User flow](research_workspace/user_flow.md)** | **Start here** — 6-step pipeline, stepper, gates |
| **[Display patterns](research_workspace/display_patterns.md)** | **Design system for research data** — Verdict Cards, charts, number formatting |
| [Research MVP UI](research_workspace/mvp_ui.md) | Ship order per step |
| 0 [Select portfolio](research_workspace/project_selector.md) | Pick portfolio |
| 1 [Research setup](research_workspace/project_setup.md) | Strategy zones; portfolio zone read-only |
| 2 [Strategy editor](research_workspace/strategy_editor.md) | Code + active strategy |
| 3 [Parameter sweep](research_workspace/parameter_sweep.md) | Dynamic params, IS tests, selection |
| 4 [Strategy robustness](research_workspace/strategy_robustness.md) | Validation-zone tests |
| 5 [Portfolio addition](research_workspace/portfolio_addition.md) | Correlation + gate |
| 6 [Commit strategy](research_workspace/commit_strategy.md) | Finalize to portfolio |
| [Backtest runner](research_workspace/backtest_runner.md) | Auxiliary quick run |
| [Zone manager (API)](research_workspace/zone_manager.md) | Field reference for setup |

### Retained specs (now as portfolio tabs)

These remain as detailed specs but are accessed as tabs inside `/portfolios/{id}`:

| Page | Hub tab |
|------|---------|
| [Strategy Library](strategy_library.md) | Strategies tab |
| [Portfolio Builder](portfolio_builder.md) | Compose tab |
| [Deployment](deployment.md) | Deploy tab |

## Entry point

Start with [Navigation](navigation.md) → [Portfolios](portfolios.md) hub spec → [Dashboard MVP UI](dashboard/mvp_ui.md).
