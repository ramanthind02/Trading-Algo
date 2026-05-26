# Dashboard — Empty States

> Align with [mvp_ui.md](mvp_ui.md) — no template CTAs unless template onboarding ships.

## No deployments, has projects

Replace section B grid:

```text
No live portfolios yet
[Go to Portfolio Builder]
```

Section C (Continue research) remains.

## No deployments, no projects

```text
No live portfolios yet
[Go to Portfolio Builder]
```

Hide section C.

Optional single line: `Or create a research project` → `/research/new` **only if** create flow exists.

## Many deployments (>6)

First 6 cards + `View all {N} deployments →` → `/deployment`

## Loading (MVP)

Skeleton cards (2) + skeleton rows (2), or simple “Loading…” per section — avoid complex shimmer in v0.

## Signals unavailable

Per card when `signals/latest` fails but deployment loads:

```text
Signals unavailable
[View deployment]
```
