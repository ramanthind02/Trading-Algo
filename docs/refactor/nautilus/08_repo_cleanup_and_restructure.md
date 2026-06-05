# WP-8 — Repo Cleanup & Restructure (pre-Nautilus)

> **Runs BEFORE the Nautilus migration** (after WP-1). Goal: shrink the repo, remove
> dead/outsourced code, and reorganize the scattered folder layout into a clean,
> modular structure — so the Nautilus refactor starts from a tidy base.
>
> **Gate:** WP-1 parity harness must be green first. This WP changes import paths en
> masse; the harness + full test suite are the safety net. **Nothing is deleted or
> moved until the gate proves research/deployment outputs are unchanged.**

## Why before Nautilus

A smaller, well-organized repo makes every later WP cheaper and lower-risk. Culling code
that Nautilus or `quantfoundry_core` already owns avoids migrating dead weight. Doing the
big import-path churn **once**, up front, behind the parity harness, is safer than
interleaving it with the Nautilus changes.

## Hard rules

- **WP-1 first.** Capture golden snapshots before any move/delete. Re-run after every batch.
- **`git mv` for moves** (preserve history). Never copy-delete.
- **Small, independently-verifiable batches.** One concern per batch; run `pytest` +
  parity after each. A red gate reverts that batch.
- **Verify before delete.** Every cull candidate must show **zero live callers** via
  `codegraph_impact` / `codegraph_callers` (and no dynamic/`importlib`/string-based use —
  grep for the module name) before removal.
- **Code is truth, behaviour is frozen.** This WP is structural only: no logic changes,
  no behaviour changes. If a move tempts a logic tweak, stop.
- **Docs + `CLAUDE.md` follow.** Module moves invalidate paths in the docs we just fixed
  (WP-7 Phase A) and in `CLAUDE.md`. The **same move-map** drives an automated docs/paths
  refresh as the closing step (see Phase 3).

---

## Phase 1 — Cull (make it smaller)

Two categories: **(a) outsourced to `quantfoundry_core`** and **(b) unused/dead**.

### Discovery method (per candidate)

1. `codegraph_callers <symbol>` / `codegraph_impact <symbol>` → confirm zero live callers.
2. `Grep` the module/symbol name repo-wide (catch dynamic imports, `__main__`, configs,
   scripts, notebooks, JSON control files).
3. Check `tests/` for the only-caller-is-its-own-test pattern (test + module both cull).
4. If truly unreferenced → delete module + its dead tests + dead imports. Re-run parity.

### Cull candidate ledger (VERIFY before deleting — status PLANNED until confirmed)

| Candidate | Reason | Verify | Status |
|---|---|---|---|
| `feature_selection/base_models/` retired stubs — `BinningModelBase`, `ContinuousBinningModel`, `RuleBasedModel` (raise `RuntimeError`) | Dead: replaced by `create_base_model_from_config` (`signed_signal`). `BinningModelBase` already shows **0 callers**. | confirm the whole stub set is unreferenced; keep `create_base_model_from_config` + live config path | PLANNED |
| `metrics/` package re-export shell | Outsourced: `metrics/__init__.py` re-exports `metric_sharpe`/`metric_sortino` from `feature_selection.validation.objective_metrics` (backed by `quantfoundry_core.metrics`). | keep genuinely-used `metrics/equity`, `metrics/risk`, `metrics/plotting`; cull the redundant shell re-exports / dead `metrics/performance` | PLANNED |
| `utils/simulation/prop_firm_simulator/` | Likely outsourced: superseded by `quantfoundry_core.prop_firm` + `prop_firms/`. | `codegraph_impact` the package; if only legacy callers remain, cull | PLANNED |
| `research/` (top-level, ~empty `__init__.py`) | Unused stub. | confirm no imports; cull or repurpose as the research umbrella (Phase 2) | PLANNED |
| `utils/evaluation/holdout_robustness.py` (and similar QF wrappers) | Possibly thin wrappers over `quantfoundry_core.robustness`. | confirm callers; keep only if it adds repo-specific glue | PLANNED |
| `scripts/_*.py` one-off probes/diagnostics (`_check_catalog`, `_check_ndx_data`, `_etf_position_sizing`, `_mt5_m1_sizing`, `_tick_download_schedule`, `_mt5_*`) | Scratch/diagnostic; some already superseded (probes moved into `data_platform/providers/mt5/probes/`). | per script: keep (→ `scripts/dev/`) or cull if dead/duplicated | PLANNED |
| Any local indicator/metric/robustness code duplicated by `quantfoundry_core` | DRY: QF owns metrics/robustness/prop_firm/portfolio_gate. | grep repo for reimplementations; route to QF | PLANNED |

> Subagents must **add rows** for anything else discovery surfaces, and **never** cull a
> row whose "Verify" step isn't satisfied.

---

## Phase 2 — Restructure (make it modular & clean)

The current top level is scattered: **three** feature dirs (`feature_extraction/`,
`feature_selection/`, `feature_research/`), a `utils/` junk-drawer (7 subdirs + loose
files), an empty `research/`, a `metrics/` shell, and domain dirs (`nodes/`, `ensemble/`,
`execution/`) mixed with infra. Target: **one clear home per concern**, grouped by domain.

### Target layout — DECIDED: Consolidate flat top-level

> ✅ **Approved (2026-06-04):** the **consolidate flat top-level** option below. The two
> alternatives (single `src/` package; minimal cull-only) were declined.

```
configs/                      # unchanged
data/                         # untracked (unchanged)

signals/                      # was nodes/  (bias nodes / signal generators)   [rename optional]
features/                     # merge of the three feature dirs:
  extraction/                 #   <- feature_extraction/
  models/                     #   <- feature_selection/base_models/ (post-cull)
  eda/                        #   <- feature_selection/eda/
  validation/                 #   <- feature_selection/validation/
ensemble/                     # unchanged (reshaped later by Nautilus, not here)
execution/                    # unchanged here (reshaped by WP-4)

research/                     # umbrella for research harnesses (repurpose empty research/)
  feature/                    #   <- feature_research/
  portfolio/                  #   <- portfolio_research/
  evaluation/                 #   <- utils/evaluation/
  prop_firms/                 #   <- prop_firms/            [or keep prop_firms/ top-level]

data_platform/                # unchanged here (reshaped by WP-2)
deployment/                   # unchanged here (reshaped by WP-4)
frontend/                     # unchanged (Flask UI)

lib/                          # dissolve the utils/ junk-drawer into clear infra:
  core/                       #   <- utils/core/ (enums, models, helpers, logger)
  cache/                      #   <- utils/cache/
  compute/                    #   <- utils/compute/ (cython, fast_nodes)
  # utils/simulation -> culled (Phase 1); utils/research_workspace -> frontend or tools/;
  # utils/dev -> tools/; loose utils/*.py -> lib/core or the right domain

scripts/                      # keep; add scripts/dev/ for _*-prefixed diagnostics
tools/                        # dev tooling (was utils/dev, research_workspace) [optional]
tests/                        # mirror the new structure 1:1
```

Culled in Phase 1 (not in the tree): `metrics/` shell, `research/` stub (repurposed),
dead `base_models` stubs, `utils/simulation/prop_firm_simulator/`.

**Declined alternatives (2026-06-04):** (B) single `src/` package — cleanest namespace but
**every** import changes (highest churn/risk); (C) minimal cull-only keeping `nodes/`/
`feature_*` names — lowest churn but smallest clarity gain. We chose **(A)**.

**`signals/` rename note:** renaming `nodes/ → signals/` is the one optional, higher-churn
item in (A). Keep it in scope, but it can be deferred to a final batch (or skipped) without
affecting the rest of the restructure if the churn isn't worth it at execution time.

### Move execution (whichever target is chosen)

1. Lock the **move-map** (old path → new path) as a single source of truth (a CSV/JSON in
   `docs/refactor/nautilus/`), covering every moved module and package.
2. `git mv` in small batches by domain (one subtree per batch).
3. Update imports with a **codemod** driven by the move-map (e.g. an AST/`libcst` script or
   `ruff`/`isort`-assisted find-replace), not hand edits — deterministic and reviewable.
4. Update `__init__.py` re-exports, `pyproject`/`setup`/`setup_cython.py` paths, entry-point
   `python -m ...` targets, `configs/*.yaml`, JSON control files, and `tests/` mirrors.
5. `pytest` + WP-1 parity after **each** batch. Red → revert that batch only.

---

## Phase 3 — Propagate to docs, CLAUDE.md, and the Nautilus plan

The move-map invalidates paths we just fixed in WP-7 Phase A and in `CLAUDE.md` and these
plan files. As the **closing step** of WP-8:

1. Run a move-map-driven find/replace over `docs/**` (excluding `docs/nautilustrader/**`)
   to repoint module/path references; re-stamp touched docs.
2. Update `CLAUDE.md` paths **and** its known-stale facts surfaced during WP-7
   (see "Known `CLAUDE.md` drift" below).
3. Update path references inside `docs/refactor/nautilus/*` (WP-2/3/4/5/6 reference many
   current paths).

### Known `CLAUDE.md` drift to fix here (found during WP-7)

- DiversifiedEnsemble forecast is implemented as `F = τ/σ` clipped at 2.0 (`h_i = 1`, the
  `√h_i` term is dropped) — not `F_i = (τ/(σ·√h_i))·X_i`.
- WeightLayer exposes **9** methods (incl. `ledoit_wolf_min_corr`, `risk_parity_corr`,
  `hierarchy_theme_*`/`inverse_corr_hierarchy`), not 3; genuinely removed set is
  `hrp_cluster_equal`, `hrp_classic`, `optimize_sortino_capped`.
- `VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES` has **13** names, not 5.
- Base-model binning ABCs are retired stubs (raise `RuntimeError`); active path is
  `create_base_model_from_config` (`signed_signal`).

## Acceptance criteria (gate)

- WP-1 parity harness green before, and after, the whole WP (research outputs unchanged).
- Full `pytest` suite green after each batch and at the end.
- `git log --follow` works on moved files (history preserved).
- No dead imports / no references to culled modules anywhere (`grep` clean).
- Top level matches the signed-off target; `tests/` mirrors it.
- Docs + `CLAUDE.md` repointed to new paths; the move-map is committed.

## Risks / notes

- **Import churn is the main risk** — mitigated by codemod + per-batch parity, not manual edits.
- **Dynamic imports / string module names** (registries, `importlib`, JSON control files,
  `python -m` targets) won't be caught by `codegraph` alone — grep for them explicitly.
- **Interaction with later WPs:** WP-2 (data_platform) and WP-4 (execution/deployment) will
  reshape those trees further; keep their top-level homes stable here so the move-map stays
  small for them.
- **Docs rework:** WP-7 Phase A prose stays valid; only paths shift — the Phase-3 codemod
  handles that, so the rework is mechanical, not a re-write.

## Subagent instructions

- Do **Phase 1 (cull) fully** and re-verify parity before starting **Phase 2 (moves)**.
- One subtree per batch; `git mv` + codemod + `pytest` + parity; never hand-edit imports en masse.
- Treat the alpha core's *behaviour* as frozen — you are moving files, not changing logic.
- Add discovered cull candidates to the ledger with their verification evidence; never cull
  unverified.
- Hand back: the committed move-map, the cull ledger with evidence, parity results per batch,
  and the docs/CLAUDE.md repoint diff.
