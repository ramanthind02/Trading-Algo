# Dashboard

**Version:** 0.2  
**Parent:** [UI/UX README](../README.md)

## Purpose

The **landing page after login**. Surfaces deployed portfolios (when they exist) and quick links to active research projects.

## Start here for implementation

**[mvp_ui.md](mvp_ui.md)** — minimal wireframe, fields, and API compose rules. Build this first; do not implement deferred components until their backends exist.

## Design principles

| Principle | Implementation |
|-----------|----------------|
| Backend-first UI | Only render fields present in API responses — see [mvp_ui.md](mvp_ui.md) |
| Read-only router | No deploy/stop/keys on dashboard — [Deployment](../deployment.md) only |
| Start simple | Two sections (deployments + research); add strips/badges when features land |
| One object per card | One **Deployment** per card |

## Page structure

| Section | MVP | Spec |
|---------|-----|------|
| Account strip | No | [account_strip.md](account_strip.md) |
| Deployed portfolios | Yes | [deployment_summary_card.md](deployment_summary_card.md) → [mvp_ui](mvp_ui.md) |
| Continue research | Yes | [research_project_row.md](research_project_row.md) → [mvp_ui](mvp_ui.md) |
| Library snapshot | No | [library_snapshot.md](library_snapshot.md) |

Layout: [page_layout.md](page_layout.md)

## Empty states

[empty_states.md](empty_states.md)

## API

[api_contract.md](api_contract.md) — MVP uses `GET /api/deployments` + per-deployment signals + projects.

## Future work

[phase_2.md](phase_2.md)

## Navigation map

| From | To | Route |
|------|-----|-------|
| View deployment | Deployment detail | `/deployment/{id}` |
| View all deployments | Deployment list | `/deployment` |
| Open workspace | Research project | `/research/{project_id}` |
| Go to Portfolio Builder | Builder | `/portfolio-builder` |

Signal API and key management: **Deployment detail only** in MVP (not on dashboard card).

## Claude Design brief (MVP)

> QuantFoundry brand v1.0: Ink bg, Paper text, Ember CTAs, Space Grotesk + JetBrains Mono labels. Sidebar: lockup-dark. Dashboard: deployment cards + research rows only (see [mvp_ui](mvp_ui.md)). [brand.md](../brand.md) · [design_system](../design_system.md).
