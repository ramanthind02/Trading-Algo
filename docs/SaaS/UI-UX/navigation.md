# Navigation & App Shell

## Information architecture

Portfolios are first-class citizens. Strategies, composition, and deployments are all scoped to a portfolio — users never manage these things globally. The nav reflects that.

```
├── Dashboard      — cross-portfolio overview (signals, deployment health)
├── Portfolios     — create, view, and manage portfolios (the hub)
└── Research       — strategy development pipeline (always portfolio-scoped)
```

Three items. No more. Everything else (strategy library, builder, deployment, API keys) lives **inside** the portfolio.

## Why three

| Old item | Where it goes |
|----------|--------------|
| Strategy Library | Portfolio → Strategies tab |
| Portfolio Builder | Portfolio → Compose tab |
| Deployment | Portfolio → Deploy tab |
| Dashboard | Stays |
| Portfolios | Stays |
| Research Workspace | Stays (renamed Research) |

Removing global silos eliminates the "where do I go?" problem. Users learn one mental model: open the portfolio, everything is there.

## Sidebar behavior

| Property | Spec |
|----------|------|
| Width (expanded) | 240px |
| Width (collapsed) | 56px — icon only |
| Background | `ink-2` |
| Logo | `lockup-dark.svg` expanded; `icon-dark.svg` collapsed |
| Active item | Left 3px `ember` border; `ember-muted` background; `ember` text |
| Inactive item | `text-secondary`; hover `ink-3` fill |
| Fonts | Space Grotesk (labels); JetBrains Mono (section caps) |
| Collapse trigger | Icon-only button at bottom of sidebar |

Research item shows the active portfolio name as a sub-label when a portfolio context is loaded:

```
◉ Research
  Prop futures core      ← muted, 11px mono, truncated
```

## Route map

| Sidebar | Route |
|---------|-------|
| Dashboard | `/dashboard` |
| Portfolios (list) | `/portfolios` |
| Portfolios (new) | `/portfolios/new` |
| Portfolio (overview) | `/portfolios/{id}` |
| Portfolio (strategies) | `/portfolios/{id}/strategies` |
| Portfolio (compose) | `/portfolios/{id}/compose` |
| Portfolio (deploy) | `/portfolios/{id}/deploy` |
| Research | `/research` — portfolio picker if no context |
| Research pipeline | `/research/{id}/setup` → `.../commit` |

Old routes (`/strategy-library`, `/portfolio-builder`, `/deployment`) redirect to the portfolio tabs — no dead links.

## Research pipeline routes

| Step | Route | Doc |
|------|-------|-----|
| Pick portfolio | `/research` | [project_selector.md](research_workspace/project_selector.md) |
| 1 Setup | `/research/{id}/setup` | [project_setup.md](research_workspace/project_setup.md) |
| 2 Editor | `/research/{id}/editor` | [strategy_editor.md](research_workspace/strategy_editor.md) |
| 3 Sweep | `/research/{id}/parameter-sweep` | [parameter_sweep.md](research_workspace/parameter_sweep.md) |
| 4 Robustness | `/research/{id}/robustness` | [strategy_robustness.md](research_workspace/strategy_robustness.md) |
| 5 Addition | `/research/{id}/portfolio-addition` | [portfolio_addition.md](research_workspace/portfolio_addition.md) |
| 6 Commit | `/research/{id}/commit` | [commit_strategy.md](research_workspace/commit_strategy.md) |

Default after portfolio pick: step **1** if strategy zones never saved, else last incomplete step.

## In-research chrome

| Element | Notes |
|---------|-------|
| Breadcrumb | `Portfolios / {portfolio name} / Research / {step name}` — portfolio name links back to `/portfolios/{id}` |
| Stepper | Steps 1–6 across top of content area |
| Zone timeline | Strategy zones editable (step 1); portfolio zone always locked |

The breadcrumb keeps users anchored to their portfolio even inside the pipeline.

## Empty state routing

| State | Redirect |
|-------|---------|
| No portfolios | `/portfolios/new` with onboarding copy |
| Research with no portfolio | Portfolio picker (step 0) |
| Dashboard with no deployments | `/portfolios` with empty state CTA |
