# Component — Library Snapshot

**Component ID:** `DashboardLibrarySnapshot`  
**Used on:** [Dashboard page layout](page_layout.md) §D

## MVP status: **not shipped**

Defer to keep dashboard to two sections (deployments + research). Users reach the library via sidebar. See [mvp_ui.md](mvp_ui.md).

## Purpose (when enabled)

Lightweight pointer to [Strategy Library](../strategy_library.md) without duplicating strategy cards.

## Anatomy

```
Strategy library · 8 committed strategies · 2 portfolios drafted
                                                      [Open library →]
```

## Prerequisite checklist

- [ ] Counts available on list APIs or `GET /api/dashboard` aggregate
- [ ] Product wants footer counts vs sidebar-only navigation
