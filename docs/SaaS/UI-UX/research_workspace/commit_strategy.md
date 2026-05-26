# Step 6 — Commit to Portfolio

**Flow:** [user_flow.md](user_flow.md) §6  
**Route:** `/research/{project_id}/commit`

## Purpose

Finalize the strategy: create an **immutable committed version** attached to this research project / portfolio. After commit, the version appears in [Strategy Library](../strategy_library.md) and can be included in [Portfolio Builder](../portfolio_builder.md).

## Entry conditions

- Portfolio addition gate passed (step 5)
- Active strategy + locked parameters

## Layout

```text
[Stepper: … | 5 Addition ✓ | 6 Commit ●]

Commit summary (read-only)
  Strategy name · version semver
  Parameter set { … }
  Zones snapshot · tickers · timeframe
  IS + validation summary metrics
  Addition gate ΔSR

[ ] I confirm this version is final for this research cycle

[ Commit strategy ]  (Ember primary)

── After success ───────────────────────────────────────
  ✓ Committed as v1.2.0
  [ Add another strategy ] → editor (step 2)
  [ Open Portfolio Builder ]
```

## Commit action

- Creates `StrategyVersion` immutable record per [`../../technical_design.md`](../../technical_design.md) §6.4
- Snapshots: source, params, `zone_snapshot_json`, methodology `static`

## Post-commit stepper

For this strategy, steps 3–6 show complete. User may start step 2 again for a **new** strategy draft in the same project.

## Exit

No forced next step — user returns to dashboard, editor, or Portfolio Builder.

## MVP

Commit button + success state minimum; full summary panel can be phased.
