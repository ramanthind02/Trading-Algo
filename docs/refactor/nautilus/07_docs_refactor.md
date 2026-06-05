# WP-7 — Docs Folder Refactor (current-state truth + Nautilus alignment)

> **Goal:** make `docs/` accurate again. Two phases: **(A)** fix docs to match the
> **current** codebase (some are stale after recent data-platform/UI/MT5 work); **(B)**
> as each Nautilus WP lands, update the docs it touches. Principle: **docs follow code** —
> a doc change ships with the code change that makes it true.

## Scope

**In scope:** `docs/library/**`, `docs/SaaS/**`, and top-level `docs/*.md`.

**Out of scope (do not edit):**

- `docs/nautilustrader/**` — vendored external Nautilus reference (read-only).
- `docs/refactor/nautilus/**` — this plan set (the source of truth for the migration).

## Method (per doc — the loop every subagent runs)

1. Read the doc.
2. For each code reference (module, class, function, path, command, config), **verify with
   `codegraph`** (`codegraph_search`/`codegraph_explore`/`codegraph_impact`) that it still
   exists and behaves as described. Fix drift; remove references to deleted/renamed symbols.
3. Reconcile the described *workflow* with the current entrypoints (run scripts, configs).
4. Fix internal links (paths moved; `data/` is now untracked — see git `58694bc`).
5. Stamp the doc footer: `> _Verified against commit <sha> on <date>._`
6. If a doc describes something that **will** change under a Nautilus WP, add a one-line
   banner: `> ⚠️ Changes under WP-N (Nautilus refactor) — see docs/refactor/nautilus/`.

Do **not** document the future Nautilus state in Phase A — only correct the present. The
future state is documented in Phase B, when the corresponding WP merges.

## Recent changes that likely invalidated docs (Phase A triage context)

From git history: `58694bc` (data_platform consolidation, full Norgate capture, **`data/`
untracked**), `3f5fb94`/`c5fee3c` (new UI), `4b3ad6b` (MT5 as data source + prod fixes),
`d2c77f5` (CFD prop MT5 phase 1). Also per `CLAUDE.md`: legacy **HRP weight methods
removed**, vault **nested layout** is current. Use these as the first places to look.

## Doc inventory, staleness risk, and Nautilus-WP impact

Risk = likelihood the doc is already wrong vs current code. WP = which Nautilus WP will
later change it (Phase B).

### `docs/library/` (implementation reference)

| Area / file | Phase-A staleness risk | Phase-B WP |
|---|---|---|
| `Data/norgate.md`, `Data/multi_source_update_architecture.md`, `Data/mt5_data_scraper.md`, `Data/back_adjustment_percentage_distortion.md`, `Data/futures_rolling_carver_alignment.md`, `Data/futures_backtesting_data_guide.md`, `Data/carry_on_cfds_etfs.md`, `Data/darwinex_universe.md` | **HIGH** — data_platform just consolidated; paths/loaders/catalog likely renamed; `data/` untracked. | WP-2 (catalog), WP-6 §A (continuous futures) |
| `Deployment/production.md`, `Deployment/live_multi_timeframe.md`, `Deployment/live_cache_refresh.md`, `Deployment/cython.md` | **HIGH** — MT5 mode + prod fixes recent; execution layer about to move. | WP-4 |
| `live_forecast/live_forecast_script.md`, `live_forecast/prop_vs_personal_workflows.md`, `live_forecast/testing_plan.md` | **HIGH** — prop/CFD/MT5 workflows changed. | WP-4 |
| `Ensemble/weight_layer.md`, `Ensemble/portfolio.md`, `Ensemble/base_model.md`, `Ensemble/multi_timeframe.md` | **MEDIUM** — verify against frozen alpha core; HRP methods removed; confirm WeightLayer modes/IDM/FDM formulas match code. | none (alpha core frozen) — but P&L lane note (WP-3) |
| `Feature_selection/*` (pipeline, validation, exploration, permutation_testing, parameter_sensitivity, portfolio_addition, Features/*) | **MEDIUM** — confirm against `feature_research/` + `quantfoundry_core` imports. | WP-1 (parity terms), WP-3 (pnl_engine) |
| `Vault/*` (architecture, user_guide, vault, monitoring, portfolio_snapshot*, predictions) | **MEDIUM** — nested layout current; verify vault paths/profiles. | WP-2 (instruments) minor |
| `Cache/architecture.md`, `Cache/user_guide.md` | **MEDIUM** — confirm `BiasNodeCache`/`CacheManager` still match; cache survives WP-2. | WP-2 (confirm unaffected) |
| `bias_nodes/*` (arch, composed, creating, index) | **LOW** — alpha core frozen; mostly stable. | none |
| `Portfolio_research/holdout.md`, `Testing/prop_firms.md`, `prop_firms/*` | **MEDIUM** — verify against `portfolio_research/holdout/` + `quantfoundry_core.prop_firm`. | WP-3/WP-4 (TradingState prop limits) |
| `index.md`, `remote_ssh_setup.md` | **LOW** | none |

### `docs/SaaS/` (product / design specs)

| Area / file | Phase-A staleness risk | Phase-B WP |
|---|---|---|
| `metrics_library.md` | **HIGH** — must describe `quantfoundry_core` as authoritative; add the new **Nautilus PortfolioAnalyzer additive-reference** lane (WP-6 §F decision). | WP-3/WP-6 |
| `transaction_costs.md`, `position_sizing.md` | **HIGH** — central to the Nautilus execution-realism lane; reconcile with `execution/position_sizer.py` and WP-3 `ExecutionPolicy`. | WP-3/WP-4 |
| `data_flow.md`, `data_source.md` | **HIGH** — data_platform consolidation + WP-2. | WP-2 |
| `research_flow.md`, `weight_layer.md`, `weight_layer_spec.md`, `strategy_spec.md` | **MEDIUM** — reconcile with current pipeline; HRP removed. | WP-1/WP-3 |
| `robustness_tests/*` (in_sample, validation, monitoring, parameter_selection, parameter_sensitivity, portfolio_addition, portfolio_holdout, index) | **MEDIUM** — confirm against `quantfoundry_core.robustness` + `feature_research` pipelines. | WP-1 |
| `portfolio_deployment.md`, `ui_ux.md`, `zone_manager.md`, `product_vision.md`, `technical_design.md` | **MEDIUM/LOW** — UI changed (#121/#126); deployment moving. | WP-4 (deployment), UI per current app |

## Phase-A action: full rewrite vs stamp-only (avoid double-writing)

Some HIGH-risk docs will be **rewritten anyway** when their Nautilus WP lands. Rewriting
them fully now wastes effort. So Phase A treats docs in two buckets, decided by the
"Phase-B WP" column above:

- **STAMP-ONLY (Phase A)** — docs a later WP will *rewrite* (the **WP-2** data docs and the
  **WP-4** deployment/live_forecast docs, incl. `SaaS/data_flow.md`, `SaaS/data_source.md`,
  `SaaS/portfolio_deployment.md`). In Phase A: fix broken links + obviously-wrong
  paths/commands, add the `⚠️ Changes under WP-N` banner, stamp the verification footer,
  and **stop**. The full rewrite happens in that WP's Phase B.
- **FULL REWRITE (Phase A)** — docs **not** slated for a Nautilus rewrite, so fixing them
  now is the only fix they'll get. These are the priority Phase-A targets:
  - `library/Ensemble/*` (alpha core frozen — HRP removed, verify IDM/FDM/WeightLayer math),
  - `library/Feature_selection/*`, `library/bias_nodes/*`, `library/Vault/*`,
    `library/Cache/*`,
  - `SaaS/metrics_library.md` (rewrite to **`quantfoundry_core` authoritative** — the
    Nautilus reference lane is added later in Phase B/WP-6, additively),
  - `SaaS/{research_flow,weight_layer,weight_layer_spec,strategy_spec,robustness_tests/*}.md`,
  - `SaaS/{ui_ux,zone_manager,product_vision}.md` (UI changed via #121/#126; no Nautilus WP).

`SaaS/{transaction_costs,position_sizing}.md` are a judgment call: they're conceptual specs
(not pure how-to), so do a **full Phase-A rewrite to current reality**, then a Phase-B note
when WP-3/WP-4 add the Nautilus execution lane (the lane is additive, not a replacement).

## Structural improvements (do alongside)

- Add/refresh a top-level `docs/README.md` (or `docs/index.md`) mapping the whole tree:
  `library/` (how the code works), `SaaS/` (product/design), `refactor/nautilus/` (active
  migration), `nautilustrader/` (vendored reference — read-only).
- Ensure every area folder has an `index.md` that links its children (most do; fill gaps).
- Normalize cross-links to repo-relative markdown paths (the IDE renders them clickable).
- Add the "Verified against commit" footer convention repo-wide.

## Execution plan

- **Phase A (now, independent of the migration):**
  1. **FULL-REWRITE bucket first** (these get no later WP rewrite, so this is their only
     fix): `library/Ensemble/*`, `library/Feature_selection/*`, `library/Vault/*`,
     `library/Cache/*`, `library/bias_nodes/*`, `SaaS/metrics_library.md`,
     `SaaS/{research_flow,weight_layer,weight_layer_spec,strategy_spec,robustness_tests/*,
     transaction_costs,position_sizing,ui_ux,zone_manager,product_vision}.md`.
  2. **STAMP-ONLY bucket** (WP-2/WP-4 will rewrite): the `Data/*`, `Deployment/*`,
     `live_forecast/*`, and `SaaS/{data_flow,data_source,portfolio_deployment}.md` docs —
     fix links/obvious errors, add the `⚠️ Changes under WP-N` banner, stamp, stop.
  One subagent per area folder; docs-only changes, no code edits.
- **Phase B (per Nautilus WP):** each WP's hand-off includes the full rewrite of the docs in
  its "Phase-B WP" column. The WP is not "done" until its docs are rewritten and stamped.

## Acceptance criteria

- No doc references a symbol/path/command that `codegraph`/filesystem says no longer exists.
- All internal links resolve; no links into untracked `data/` as if tracked.
- HIGH-risk docs reviewed and stamped with a verification commit.
- `metrics_library.md` reflects the QF-authoritative + Nautilus-reference decision.
- Each Nautilus WP, when merged, leaves its mapped docs accurate (Phase B).

## Subagent instructions

- This is a **docs-only** work package in Phase A — do not change code to match a doc; if a
  doc and the code disagree, the **code is truth** (fix the doc), unless the doc reveals an
  actual bug, in which case flag it separately.
- Work one area folder at a time; keep changes reviewable.
- Use `codegraph_explore`/`codegraph_search` to verify every code reference before rewriting.
- Do not touch `docs/nautilustrader/**` or `docs/refactor/nautilus/**`.
