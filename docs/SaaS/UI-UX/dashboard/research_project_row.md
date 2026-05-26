# Component — Research Project Row

**Component ID:** `ResearchProjectRow`  
**Used on:** [Dashboard page layout](page_layout.md) §C

> **Implement first:** [mvp_ui.md](mvp_ui.md) (minimal row). This document is the **full** row as features land.

## Purpose

Quick resume for in-progress research. Routes to [Research Workspace](../research_workspace/README.md).

## MVP variant (ship first)

```text
┌─────────────────────────────────────────────────────────┐
│ {Project name} · edited {relative time}  Open workspace → │
└─────────────────────────────────────────────────────────┘
```

| Field | MVP |
|-------|-----|
| Project name | ✅ |
| Edited (`updated_at`) | ✅ |
| Open workspace → | ✅ → `/research/{project_id}` |
| Job badge | ❌ deferred |
| Last strategy name | ❌ deferred |
| Mini zone bar | ❌ deferred |

## Full variant

### Section header (parent)

```
Continue research                          [New project +]
```

`New project` only when create flow ships.

### Row anatomy

```
┌─────────────────────────────────────────────────────────────┐
│ {Project name}                              {job badge?}    │
│ Last: {Strategy name} · edited {relative time}              │
│ [▓▓▓▓▓▓▓▓▓░░░░░░░░] train | val | test  (mini zone bar)     │
│                                    [Open workspace →]       │
└─────────────────────────────────────────────────────────────┘
```

## Job badge labels (deferred)

| Job type | Label examples |
|----------|----------------|
| `parameter_sweep` | `Sweep running` |
| `backtest` | `Backtest queued` / `Backtest running` |
| failed | `Failed` |

## List behavior

| Property | Value |
|----------|-------|
| Max rows | 5 |
| Sort | `updated_at` descending |

## Interaction

| Target | Action |
|--------|--------|
| Row / Open workspace | Navigate to `last_workspace_path` or `/research/{id}/zones` |
