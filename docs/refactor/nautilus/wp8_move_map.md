# WP-8 — Move Map & Import-Rewrite Plan (Phase 2 restructure)

> **Status: PLAN ONLY — non-destructive.** Nothing in this document has been moved or
> edited. It is the single source of truth for the **approved "consolidate flat top-level"**
> restructure (see `08_repo_cleanup_and_restructure.md` §"Phase 2 — Restructure").
>
> **Gate (unchanged):** WP-1 parity harness + full `pytest` must be green before, between,
> and after every batch. Execute with `git mv` + a move-map-driven codemod, one subtree per
> batch. Phase 1 cull must be done first — culled paths below are **excluded** from the map.

## Executive summary

| Metric | Count |
|---|---|
| Files moved (Python modules, excl. culled, excl. optional `nodes/→signals/`) | **~330** |
| Files moved **including** optional `nodes/→signals/` rename | **~519** |
| Distinct top-level import prefixes that change | **9** (7 mandatory + 1 optional + 1 sub-prefix split) |
| Source files containing imports that must be rewritten | **~430** distinct files (union across prefixes; `utils.*` alone is ~250) |
| Top landmines | (1) `mock.patch`/`monkeypatch` **string** module paths in tests; (2) `nodes/_taxonomy.py` `CANONICAL_MODULE_IMPORTS` string values (only on `nodes→signals`); (3) `utils/compute/cython/setup_cython.py` Extension `name=`/source paths; (4) `importlib.import_module` string args + lazy-alias shims; (5) `python -m` entrypoints in docs/configs/UI command strings |

Culled in Phase 1 (NOT in this map): `metrics/` shell re-exports, top-level `research/` stub
(repurposed as the umbrella below), dead `feature_selection/base_models/base_model.py` stubs
(`BinningModelBase`/`ContinuousBinningModel`/`RuleBasedModel`), `utils/simulation/prop_firm_simulator/`.

---

## 1. Import-prefix change table (the codemod's rename rules)

Apply **longest-prefix-first** (so the `feature_selection.*` sub-splits and `utils.evaluation`
land before any shorter parent rule). Every rule is a dotted-prefix rename.

| # | Old import prefix | New import prefix | Mandatory? | Approx. files w/ matching imports |
|---|---|---|---|---|
| R1 | `feature_research.` | `research.feature.` | yes | 121 |
| R2 | `portfolio_research.` | `research.portfolio.` | yes | 62 |
| R3 | `utils.evaluation.` | `research.evaluation.` | yes | (subset of utils; ~25 import sites) |
| R4 | `prop_firms.` | `research.prop_firms.` | yes¹ | 49 |
| R5 | `feature_extraction.` | `features.extraction.` | yes | 9 |
| R6 | `feature_selection.base_models.` | `features.models.` | yes² | (subset of feature_selection) |
| R7 | `feature_selection.eda.` | `features.eda.` | yes | (subset) |
| R8 | `feature_selection.validation.` | `features.validation.` | yes | 56 (whole `feature_selection.*`) |
| R9 | `utils.core.` | `lib.core.` | yes | (largest slice of utils) |
| R10 | `utils.cache.` | `lib.cache.` | yes | (large slice of utils) |
| R11 | `utils.compute.` | `lib.compute.` | yes | (incl. cython) |
| R12 | `utils.dev.` | `tools.dev.` (see §2.6) | yes | ~0 import sites (empty pkg) |
| R13 | `utils.research_workspace.` | `tools.research_workspace.` | yes³ | ~6 import sites |
| R14 | `utils.<loose>.` (`vault_paths`, `repo_bootstrap`, `futures_micro_specs`) | `lib.core.<loose>` | yes | see §2.6 |
| R15 | `nodes.` | `signals.` | **OPTIONAL — final batch** | 182 |

¹ R4: the plan allows keeping `prop_firms/` top-level. **Recommendation: keep `prop_firms/`
   top-level** (lower churn, 49 files; it is also consumed by `frontend`, `scripts`, and
   `portfolio_research`). If kept, R4 is dropped. The map below documents both; default = keep.
² R6: only the **live** surface (`feature_base_model.py` → `BaseModel`, and the
   `__init__.py` lazy-alias shim, post-cull) moves; the dead `base_model.py` stubs are culled in Phase 1.
³ R13: plan offers `frontend/` OR `tools/` for `research_workspace`. Its only consumers are
   `feature_research.ui.*` and `portfolio_research.ui.*` (now `research.*`). **Recommendation:
   `tools/research_workspace/`** to keep `frontend/` thin, OR fold under `research/` shared. Default below: `tools/`.

---

## 2. Complete move-map (old path → new path)

### 2.1 `feature_research/` → `research/feature/`  (R1)  — 76 files

`git mv feature_research research/feature` then fix the package name. All 76 modules move 1:1:

```
feature_research/__init__.py                                  -> research/feature/__init__.py
feature_research/__main__.py                                  -> research/feature/__main__.py
feature_research/config.py                                    -> research/feature/config.py
feature_research/pipeline.py                                  -> research/feature/pipeline.py
feature_research/inclusion_gates.py                           -> research/feature/inclusion_gates.py
feature_research/run_inclusion_gates.py                       -> research/feature/run_inclusion_gates.py
feature_research/filter_research_labels.py                    -> research/feature/filter_research_labels.py
feature_research/research_table_exports.py                   -> research/feature/research_table_exports.py
feature_research/save_feature_to_vault.py                    -> research/feature/save_feature_to_vault.py
feature_research/_internal/**            (5 files)            -> research/feature/_internal/**
feature_research/binning/**              (5 files)            -> research/feature/binning/**
feature_research/exploration/**          (5 files)            -> research/feature/exploration/**
feature_research/in_sample/**            (5 files)            -> research/feature/in_sample/**
feature_research/oos/**                  (3 files)            -> research/feature/oos/**
feature_research/pipelines/**            (10 files)           -> research/feature/pipelines/**
feature_research/portfolio_addition/**   (3 files)            -> research/feature/portfolio_addition/**
feature_research/shared/**               (6 files)            -> research/feature/shared/**
feature_research/ui/**                   (16 files)           -> research/feature/ui/**
feature_research/validation/**           (6 files)            -> research/feature/validation/**
feature_research/visualization/**        (5 files)            -> research/feature/visualization/**
```
Import prefix: `feature_research.` → `research.feature.`

### 2.2 `portfolio_research/` → `research/portfolio/`  (R2)  — 41 files

```
portfolio_research/__init__.py                               -> research/portfolio/__init__.py
portfolio_research/__main__.py                               -> research/portfolio/__main__.py
portfolio_research/config.py                                 -> research/portfolio/config.py
portfolio_research/correlation_export.py                     -> research/portfolio/correlation_export.py
portfolio_research/futures_sim.py                            -> research/portfolio/futures_sim.py
portfolio_research/portfolio_prop_firm_reports.py            -> research/portfolio/portfolio_prop_firm_reports.py
portfolio_research/prop_firm_bridge.py                       -> research/portfolio/prop_firm_bridge.py
portfolio_research/prop_firm_reports.py                      -> research/portfolio/prop_firm_reports.py
portfolio_research/run_feature_vault_correlation.py          -> research/portfolio/run_feature_vault_correlation.py
portfolio_research/run_portfolio_prop_firm.py                -> research/portfolio/run_portfolio_prop_firm.py
portfolio_research/run_portfolio_test.py                     -> research/portfolio/run_portfolio_test.py
portfolio_research/vault_correlation.py                      -> research/portfolio/vault_correlation.py
portfolio_research/weight_layer_cv.py                        -> research/portfolio/weight_layer_cv.py
portfolio_research/weight_layer_export.py                    -> research/portfolio/weight_layer_export.py
portfolio_research/weight_layer_report.py                    -> research/portfolio/weight_layer_report.py
portfolio_research/holdout/**            (8 files)            -> research/portfolio/holdout/**
portfolio_research/pipelines/**          (2 files)            -> research/portfolio/pipelines/**
portfolio_research/shared/**             (4 files)            -> research/portfolio/shared/**
portfolio_research/ui/**                 (12 files)           -> research/portfolio/ui/**
portfolio_research/visualization/**      (2 files)            -> research/portfolio/visualization/**
```
Import prefix: `portfolio_research.` → `research.portfolio.`

### 2.3 `utils/evaluation/` → `research/evaluation/`  (R3)  — 18 files

```
utils/evaluation/__init__.py                                 -> research/evaluation/__init__.py
utils/evaluation/holdout_robustness.py                       -> research/evaluation/holdout_robustness.py
utils/evaluation/permutation_test/**     (5 files)           -> research/evaluation/permutation_test/**
utils/evaluation/walkforward/**          (12 files)          -> research/evaluation/walkforward/**
```
Import prefix: `utils.evaluation.` → `research.evaluation.`
(Note: this is a `utils.*` subtree that diverges to `research/`, not `lib/` — order R3 before R9–R14.)

### 2.4 `prop_firms/` → `research/prop_firms/`  (R4, OPTIONAL — default KEEP top-level)  — 43 files

If executing R4 (not recommended), `git mv prop_firms research/prop_firms`; all 43 files move 1:1
(`apex/`, `base/`, `fundednext/`, `lucid/`, `mffu/`, `topstep/`, `tradeday/` subpkgs + loose
`comparison.py`, `optimization*.py`, `report*.py`, `reporting.py`, `run_*.py`).
Import prefix: `prop_firms.` → `research.prop_firms.`
**Default: skip R4, keep `prop_firms/` top-level** (the plan explicitly permits this).

### 2.5 `feature_extraction/` + `feature_selection/*` → `features/*`  (R5–R8)  — 19 files (post-cull)

```
feature_extraction/feature_extractor.py                      -> features/extraction/feature_extractor.py
  (add features/extraction/__init__.py)

feature_selection/base_models/__init__.py                    -> features/models/__init__.py
feature_selection/base_models/feature_base_model.py          -> features/models/feature_base_model.py
  # feature_selection/base_models/base_model.py  -> CULLED in Phase 1 (dead stubs); do NOT move.
  # NOTE: the surviving __init__.py lazy-alias for the legacy stubs is removed by the cull —
  #       confirm the __getattr__ shim and its importlib target are deleted, not moved.

feature_selection/eda/**                 (7 files)            -> features/eda/**
feature_selection/validation/**          (8 files)            -> features/validation/**
```
Import prefixes:
- `feature_extraction.` → `features.extraction.`
- `feature_selection.base_models.` → `features.models.`
- `feature_selection.eda.` → `features.eda.`
- `feature_selection.validation.` → `features.validation.`

### 2.6 `utils/` → `lib/` + `tools/`  (R9–R14)  — 33 files (excl. culled `simulation/`, excl. R3 `evaluation/`)

```
utils/__init__.py                                            -> lib/__init__.py  (or drop; see note)
utils/core/**                            (10 files)           -> lib/core/**
utils/cache/**                           (15 files)           -> lib/cache/**
utils/compute/**                         (9 files: incl. cython/)-> lib/compute/**
utils/dev/__init__.py                                         -> tools/dev/__init__.py
utils/research_workspace/**              (7 files)            -> tools/research_workspace/**

# Loose utils/*.py -> lib/core/ (per plan "loose utils/*.py -> lib/core or the right domain"):
utils/vault_paths.py                                          -> lib/core/vault_paths.py
utils/repo_bootstrap.py                                      -> lib/core/repo_bootstrap.py
utils/futures_micro_specs.py                                -> lib/core/futures_micro_specs.py

# CULLED in Phase 1 — do NOT move:
utils/simulation/__init__.py
utils/simulation/prop_firm_simulator/**  (5 files)
```
Import prefixes:
- `utils.core.` → `lib.core.`
- `utils.cache.` → `lib.cache.`
- `utils.compute.` → `lib.compute.`
- `utils.dev.` → `tools.dev.`
- `utils.research_workspace.` → `tools.research_workspace.`
- `utils.vault_paths` → `lib.core.vault_paths`
- `utils.repo_bootstrap` → `lib.core.repo_bootstrap`
- `utils.futures_micro_specs` → `lib.core.futures_micro_specs`

> Note on `utils/__init__.py`: `from utils ...` (bare) appears nowhere except as a namespace
> package. After the split there is no single `utils` package. The codemod must rewrite the
> three loose-module imports explicitly (R14) — they are NOT covered by any `utils.<subpkg>.`
> rule. Grep `from utils import` / `import utils\b` before the batch to confirm none survive.

### 2.7 `nodes/` → `signals/`  (R15, OPTIONAL — own FINAL batch)  — 189 files

`git mv nodes signals`; all 189 files (incl. `_taxonomy.py`, `__init__.py`, the full
`breakout/ mean_reversion/ momentum/ regime/ seasonal/ volatility/ composite/ pairs/ misc/
buy_hold/` trees + legacy flat shims) move 1:1.
Import prefix: `nodes.` → `signals.`
**This batch additionally requires rewriting the 270 string values in
`signals/_taxonomy.py` (`CANONICAL_MODULE_IMPORTS`)** — see Landmine L2.

### 2.8 `scripts/_*.py` → `scripts/dev/`  (one-off probes)  — up to 6 files

```
scripts/_bootstrap.py                                        -> scripts/dev/_bootstrap.py
scripts/_check_catalog.py                                    -> scripts/dev/_check_catalog.py
scripts/_check_ndx_data.py                                   -> scripts/dev/_check_ndx_data.py
scripts/_etf_position_sizing.py                              -> scripts/dev/_etf_position_sizing.py
scripts/_mt5_m1_sizing.py                                    -> scripts/dev/_mt5_m1_sizing.py
scripts/_tick_download_schedule.py                           -> scripts/dev/_tick_download_schedule.py
```
> Per Phase 1, each `_*.py` is a keep-or-cull decision; only **kept** probes move to
> `scripts/dev/`. `scripts/` is normally run as a path (`python scripts/foo.py`), not imported,
> so these moves rarely affect import prefixes — but confirm none are imported by tests
> (`scripts._bootstrap` is referenced; verify before moving).

### 2.9 Unchanged top-level homes (NOT moved this WP)

`configs/`, `data/`, `ensemble/`, `execution/`, `data_platform/`, `deployment/`, `frontend/`,
`tests/` (mirrors the new tree as a closing step), and `metrics/equity|risk|plotting` (the
**kept** parts; only the shell re-exports are culled). These are **consumers** of moved
prefixes — their import statements still get rewritten by R1–R15, but their own paths do not change.

---

## 3. Import-rewrite scope (per distinct prefix)

Counts are files containing ≥1 `import`/`from` statement matching the old prefix (ripgrep over
`*.py`, repo tree excl. `.venv/`/`venv/`). A file can appear under several prefixes.

| Old prefix | Files importing it | Notes |
|---|---|---|
| `utils.` (all subtrees) | **250** | splits into `lib.*` (R9–R11,R14) + `research.evaluation` (R3) + `tools.*` (R12–R13). Largest blast radius. |
| `nodes.` | **182** | OPTIONAL rename; biggest single prefix. Defer to final batch. |
| `feature_research.` | **121** | → `research.feature.` |
| `feature_selection.` | **56** | splits R6/R7/R8. Test dirs are `tests/validators/**` + `tests/unit-tests/validators/**` (note: tests live under `validators/`, not `feature_selection/`). |
| `portfolio_research.` | **62** | → `research.portfolio.` |
| `prop_firms.` | **49** | OPTIONAL R4; default keep top-level (0 if kept). |
| `feature_extraction.` | **9** | → `features.extraction.` |
| **Union of all moved prefixes** | **~430 distinct files** | the codemod's working set |

Highest-fan-in consumers to watch (touch in late batches): `frontend/app.py` (imports both
`feature_research.*` and `portfolio_research.*`), `scripts/enigma_live_forecast.py` (15 `utils.*`),
`ensemble/**` and `execution/position_sizer.py` (`utils.*`, `nodes.*`), `deployment/**`
(`utils.*`), `data_platform/**` (`utils.*`, `feature_*`), and the whole `tests/` tree.

---

## 4. Dynamic-import landmines (a plain import-statement rewrite WILL miss these)

These require **string-literal** rewrites or are special-cased; a libcst transformer that only
visits `Import`/`ImportFrom` nodes will silently leave them broken.

### L1 — `mock.patch` / `monkeypatch.setattr` string targets (HIGH RISK, many files)
Tests patch by dotted string, e.g.:
- `tests/unit-tests/utils/test_central_cache_store.py:262` — `monkeypatch.setattr("utils.cache.runtime.central_cache.BiasNodeCache.load", ...)`
- `tests/feature_research/validation/engine/test_runner.py:666` — `monkeypatch.setattr("utils.evaluation.walkforward.portfolio_evaluator.evaluate_fold_portfolio", ...)`
- `tests/unit-tests/feature_research/test_ui_workspace.py:18,89` — `monkeypatch.setattr("feature_research.ui.workspace._REPO_ROOT", ...)`
- `tests/feature_research/.../test_run_validation_permutation.py`, `test_run_oos_permutation.py` — `patch("feature_research.config.load_config")`
- `tests/unit-tests/nodes/**` — dozens of `patch("nodes.<...>")` (e.g. `test_cython_bias_nodes.py`, `test_atr_percentile_filter_node.py`, `test_rsi_signal_atr_slope_entry.py`, `test_new_bias_nodes.py`).
**Action:** the codemod must also rewrite **string literals** whose value starts with any
moved prefix (with a `.` or `"` boundary), not just import nodes. Restrict to args of
`patch`/`patch.object`/`setattr`/`monkeypatch.setattr`/`import_module`/`find_spec` to avoid
clobbering unrelated strings. The `nodes.*` patch strings are part of the OPTIONAL R15 batch.

### L2 — `nodes/_taxonomy.py` canonical maps (ONLY on the `nodes→signals` batch)
`CANONICAL_MODULE_IMPORTS` (87 entries) holds string values like
`"adaptive_rsi": "nodes.mean_reversion.rsi.adaptive_rsi"` — **270 string values** total across
the dict resolved at runtime via `importlib.import_module` in
`utils/core/helpers.py::_resolve_bias_node_class` (→ `lib/core/helpers.py`). `CANONICAL_MODULE_CLASSES`
holds class *names* (no dotted paths → unaffected).
**Action:** in the R15 batch, rewrite every `"nodes...."` value → `"signals...."`. The dict
**keys** (short module names like `"double7s"`) are the contract used by vault JSONs — DO NOT
touch keys. (If R15 is skipped, this file is untouched.)

### L3 — Vault feature JSONs reference nodes by SHORT NAME, not dotted path (GOOD NEWS)
Sampled `vault/**/features/*.json` (e.g. `double7s_signal_D_*.json`) use
`"bias_node_spec": {"module_name": "double7s", ...}` — a **short** key resolved through L2's
dict, **not** a dotted module path. Therefore vault/control JSONs are **NOT** rewritten by any
batch, including R15. (No grep hits for `module_import_path`/`module_path`/dotted module strings
in `vault/**/*.json`.) Confirm again post-cull, but this removes a large feared surface.

### L4 — `importlib.import_module(...)` string args + lazy-alias shims
- `feature_research/__init__.py` — `_LEGACY_MODULE_ALIASES` maps
  `"core_helpers" -> "feature_research._internal.core_helpers"` etc. (string targets). Moves
  with R1 but the **string values** must be rewritten to `research.feature._internal.*`.
- `feature_research/shared/cli.py:29` — `import_module(module_path)` where `module_path` is the
  `python -m feature_research <command>` dispatch target (see L5).
- `feature_selection/base_models/__init__.py:15` — `import_module(f"{__name__}.base_model")` —
  targets the **culled** stub module; verify removed by Phase 1 (do not carry the shim into `features/models/`).
- `utils/core/helpers.py:120,378` — `import_module(full_module_name)` and a `possible_modules`
  list containing `'utils.core.functime'`; rewrite the **string list entries** → `'lib.core.functime'`.
- `feature_selection/validation/objective_metrics.py:219`, `ensemble/vault_manager.py:7`
  (`import_module("ensemble.vault.manager")` — unchanged, ensemble stays), `portfolio_research/holdout/pipeline.py:20` (`__import__(...)`).
**Action:** rewrite string args of `import_module`/`__import__` matching moved prefixes; audit
the two lazy-alias dicts (`feature_research.__init__`, `feature_research.shared.cli`).

### L5 — `python -m <module>` entrypoints (docs, UI command strings, config docstrings)
Live `python -m ...` targets that must repoint when their package moves:
- `python -m feature_research[.in_sample.run_is | .save_feature_to_vault | <subcommand>]`
  → `python -m research.feature...` (defined in `feature_research/shared/cli.py`,
  `__main__.py`, `in_sample/run_is.py`, `oos/run_oos.py`, `validation/run_validation.py`,
  and emitted as a string in `feature_research/ui/vault_save.py:166`).
- `python -m portfolio_research[ holdout | .run_portfolio_test | .weight_layer_cv | .ui.run_phase]`
  → `python -m research.portfolio...` (incl. the UI-emitted command in
  `portfolio_research/ui/planner.py:115`).
- `python -m utils.cache.runtime.bootstrap_source_candles` / `...cache_manager`
  → `python -m lib.cache.runtime...` (in `utils/cache/runtime/cache_manager.py:23` docstring
  + docs).
- `python utils/compute/cython/setup_cython.py build_ext --inplace`
  → `python lib/compute/cython/setup_cython.py ...` (CLAUDE.md + the file's own docstring).
**Action:** these are NOT Python imports — handle via the Phase-3 docs/string find-replace
**and** fix the UI-emitted command strings in code (`vault_save.py`, `planner.py`).

### L6 — Cython build paths (`utils/compute/cython/setup_cython.py`)
`Extension(name="utils.compute.cython.cython_optimized", sources=["utils/compute/cython/cython_optimized.pyx"])`
and the `cython_nodes` extension. On the R11 batch these must become
`name="lib.compute.cython..."` and `sources=["lib/compute/cython/..."]`. The compiled
`.pyd`/`.so` artifacts must be **recompiled** in-place at the new path
(`python lib/compute/cython/setup_cython.py build_ext --inplace`); stale `.pyd` under the old
path will shadow nothing but the import path changes, so callers of
`utils.compute.cython.cython_*` (and `CYTHON_NODES_AVAILABLE` patches in
`tests/unit-tests/nodes/test_cython_bias_nodes.py`) break under R11.

### L7 — Build / packaging config
- **No root `pyproject.toml` / `setup.py` / `setup.cfg`** exists (only `requirements.txt`, which
  lists PyPI deps — no first-party package names → unaffected).
- `tests/conftest.py` and `tests/unit-tests/utils/conftest.py` — inspect for `sys.path` inserts
  or first-party module strings before moving `tests/`.
- No first-party module paths found in `configs/*.yaml` (`source_priority.yaml` is data-only).

### L8 — Legacy compat-shim tests
`tests/unit-tests/feature_research/test_public_surface.py` asserts
`importlib.import_module("feature_research.core_helpers")` / `"feature_research.bootstrap")`
resolve (via the L4 lazy aliases). After R1 these strings → `research.feature.*` AND the
aliases inside the moved `__init__` must still resolve, or the test must be updated.
`tests/unit-tests/utils/test_cache_module_shims.py` similarly probes `utils.*` cache shims —
update to `lib.cache.*` under R10.

---

## 5. Recommended codemod approach

**Use libcst with a string-aware transform, driven by this move-map** (not pure-AST ImportFrom
rewriting, which misses L1/L2/L4/L5). Concretely:

1. **Source of truth:** serialize §1's rename table to `docs/refactor/nautilus/wp8_renames.json`
   (`[{old, new, optional}]`), longest-prefix-first. The codemod loads this — no rules hardcoded.
2. **Pass A — import statements (libcst):** a `cst.CSTTransformer` that rewrites
   `Import`/`ImportFrom` module names whose dotted prefix matches a rule. Handles `from x.y import z`,
   `import x.y as z`, relative-vs-absolute (these are all absolute here). libcst preserves
   formatting/comments → reviewable diffs.
3. **Pass B — dynamic strings (same transform, `SimpleString`/`ConcatenatedString` visitor):**
   rewrite string literals that (a) equal or start-with-`<prefix>.` a moved prefix AND (b) are an
   argument to one of: `importlib.import_module`, `__import__`, `import_module`, `find_spec`,
   `patch`, `patch.object`, `mocker.patch`, `monkeypatch.setattr`, `setattr` — gated by call-name
   to avoid touching unrelated strings (covers L1, L4, L8, and L2's dict values).
4. **Pass C — `_taxonomy.py` dict values (R15 only):** narrow rewrite of the
   `CANONICAL_MODULE_IMPORTS` value strings (`nodes.` → `signals.`); leave keys untouched.
5. **Pass D — non-Python text (Phase 3, plain templated find-replace):** `python -m ...` strings,
   `setup_cython.py` `name=`/`sources=`, docs under `docs/**` (excl. `docs/nautilustrader/**`),
   `CLAUDE.md`, `.cursor/**` skill/rule files (`005-python-environment.mdc`, etc.), and the
   UI-emitted command strings in `ui/vault_save.py` + `ui/planner.py` (these last two are code,
   so do them in Pass B's file set, not Pass D).
6. **Guard:** after each batch run `pyflakes`/`ruff --select F` for unresolved imports, then
   `pytest` + WP-1 parity. A red gate reverts only that batch (`git mv` is reversible).
7. `git mv` first (preserves history), THEN run the codemod over the whole repo, THEN
   add/adjust `__init__.py` for newly-created package roots (`research/`, `research/feature/`,
   `research/portfolio/`, `research/evaluation/`, `features/`, `features/extraction/`,
   `features/models/`, `features/eda/`, `features/validation/`, `lib/`, `lib/core/`, `lib/cache/`,
   `lib/compute/`, `tools/`, `tools/dev/`, `tools/research_workspace/`, `scripts/dev/`).

---

## 6. Safe batch order (leaf-first, highest-fan-in last; optional rename last)

Each batch = `git mv` subtree + codemod (Passes A–C scoped to that batch's rule set) + `__init__`
fixups + `pytest` + parity. Order minimizes the window where a high-fan-in prefix is half-moved.

| Batch | Scope | Rules | Why this slot |
|---|---|---|---|
| **B0** | Phase 1 cull (prereq) — remove dead `base_models` stubs, `metrics/` shell, `utils/simulation/`, repurpose `research/` stub | — | Map assumes these are gone; do first. |
| **B1** | `scripts/_*.py` → `scripts/dev/` (kept probes only) | §2.8 | Leaf; scripts are run-by-path, ~0 import churn. Cheapest warm-up. |
| **B2** | `feature_extraction/` → `features/extraction/` | R5 | Smallest real package (9 importers); creates `features/` root. |
| **B3** | `feature_selection/{base_models,eda,validation}` → `features/{models,eda,validation}` | R6,R7,R8 | Self-contained; 56 importers but cohesive; tests live under `tests/validators/**`. |
| **B4** | `utils/research_workspace/` → `tools/research_workspace/`; `utils/dev/` → `tools/dev/` | R12,R13 | Leaf infra; few importers (~6); creates `tools/` root. |
| **B5** | `utils/evaluation/` → `research/evaluation/` | R3 | Creates `research/` umbrella; ~25 import sites; must precede R1/R2 which import it. |
| **B6** | `utils/compute/` → `lib/compute/` (+ recompile cython, fix `setup_cython.py`) | R11, L6 | Self-contained compute leaf; isolates the cython rebuild risk in one batch. |
| **B7** | `utils/cache/` → `lib/cache/` | R10 | Higher fan-in infra; after compute so `lib/` exists. |
| **B8** | `utils/core/` → `lib/core/` (+ loose `vault_paths`/`repo_bootstrap`/`futures_micro_specs`) | R9,R14 | **Highest-fan-in infra** (enums/models/helpers/logger); do after the other `utils/*` slices so `utils/` is empty afterward. |
| **B9** | `portfolio_research/` → `research/portfolio/` | R2 | Depends on R3 (evaluation) already moved; 62 importers. |
| **B10** | `feature_research/` → `research/feature/` | R1, L4(aliases), L5(UI strings), L8 | Largest research pkg (121 importers); `frontend/app.py` + `data_platform` consume it — late slot. |
| **B11** | *(optional)* `prop_firms/` → `research/prop_firms/` | R4 | **Skip by default** (keep top-level). If done, isolate here. |
| **B12** | **OPTIONAL FINAL:** `nodes/` → `signals/` | R15, L2(taxonomy), L1(`nodes.*` patches) | Biggest churn (182 importers) + 270 taxonomy strings; isolate so it can be deferred/skipped without disturbing B1–B11. |
| **B13** | Mirror `tests/` to new tree + Phase-3 docs/CLAUDE.md/`.cursor` repoint | Pass D | Closing step; only paths shift. |

> Rationale: leaf/low-fan-in packages (`feature_extraction`, `feature_selection`, the `tools/`
> leaves) go first to surface codemod bugs cheaply; the `utils/` split runs compute→cache→core
> so the junk-drawer empties bottom-up and `lib/core` (everything depends on it) is the last
> infra move; research packages follow once their `utils.evaluation` dependency is relocated;
> and the optional, highest-churn `nodes→signals` rename is fully isolated at the end.

---

## 7. Open decisions to confirm before execution

1. **R4 (`prop_firms/`):** keep top-level (default, recommended) or nest under `research/`?
2. **R13 (`research_workspace`):** `tools/` (default) vs `frontend/` vs `research/` shared?
3. **R15 (`nodes/→signals/`):** execute or defer/skip? (270 taxonomy strings + 182 importers.)
4. **`utils/__init__.py`:** drop entirely (no `utils` namespace remains) — confirm no `import utils`
   bare references survive post-split.
5. **Loose `utils/*.py` homing:** `vault_paths`/`repo_bootstrap`/`futures_micro_specs` → `lib/core/`
   (default) vs a domain home (e.g. `vault_paths` → near `ensemble/vault`).
