# Shared UI Components

Reusable components referenced across multiple pages. Page-specific layouts live in each area folder.

## Dashboard components

**MVP implementations:** [dashboard/mvp_ui.md](../dashboard/mvp_ui.md)

| Component ID | MVP | Full spec |
|--------------|-----|-----------|
| `DeploymentSummaryCard` | ✅ minimal | [deployment_summary_card.md](../dashboard/deployment_summary_card.md) |
| `ResearchProjectRow` | ✅ minimal | [research_project_row.md](../dashboard/research_project_row.md) |
| `DashboardAccountStrip` | ❌ not shipped | [account_strip.md](../dashboard/account_strip.md) |
| `DashboardLibrarySnapshot` | ❌ not shipped | [library_snapshot.md](../dashboard/library_snapshot.md) |

## Shell components

| Component | Spec |
|-----------|------|
| Left sidebar + logo | [navigation.md](../navigation.md) · assets in `logo (1)/exports/` |
| Zone timeline bar | [research_workspace/README.md](../research_workspace/README.md) |
| Section label (`.label`) | JetBrains Mono 11px uppercase — [brand.md](../brand.md) |
| Status pill | [design_system.md](../design_system.md) + [deployment_summary_card.md](../dashboard/deployment_summary_card.md) |

## Research Workspace components (page-level)

Each workspace subcategory defines its own panels (heatmap, robustness card, correlation matrices). See [research_workspace/](../research_workspace/).

## Future component specs

When implementing QuantFoundry-Web, add Storybook entries keyed to **Component ID** values above for traceability.
