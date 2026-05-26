# Strategy Library (Strategies tab)

**Parent:** [UI/UX README](README.md)  
**Route:** `/portfolios/{id}/strategies` (old `/strategy-library` redirects to the active portfolio's strategies tab)

> **Navigation:** This is the **Strategies tab** inside the portfolio hub — not a top-level nav item. Strategies are always scoped to their portfolio. See [portfolios.md](portfolios.md) for the full hub spec.

## Purpose

Browsable catalog of all **committed** strategy versions for a specific portfolio. Used to select strategies when composing in the Compose tab.

## List view — strategy cards

| Field | Description |
|-------|-------------|
| Name | Strategy name |
| Instrument universe | Tickers / asset class summary |
| Last committed | Date |
| Test evaluated | Indicator if any **test** zone has been evaluated (detail may list which) |

**Actions per card:**

- Open detail
- Add to Portfolio → Portfolio Builder with strategy pre-selected

## Detail view

| Section | Contents |
|---------|----------|
| Version history | All committed versions, timestamps, links to artifacts |
| Validation metrics | From validation-zone runs |
| Test results | Results from test (and validation) runs as applicable |
| Source code | Read-only viewer |
| Portfolio Correlation | Link to [research_workspace/portfolio_correlation.md](research_workspace/portfolio_correlation.md) |

## Dashboard integration

Counts only on dashboard via [dashboard/library_snapshot.md](dashboard/library_snapshot.md) — no full cards on landing page.

## MVP vs future

| Feature | MVP | Future |
|---------|-----|--------|
| Cards + detail + add to portfolio | ✅ | |
| Tagging, team sharing | | ✅ |

See [mvp_scope.md](mvp_scope.md).
