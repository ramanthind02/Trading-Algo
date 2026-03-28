
Here is a consolidated report from four parallel **read-only** codebase explorations (ensemble/execution, cache/vault/utils, nodes/feature_selection, deployment/scripts/tests). No files were changed.

---

## Executive summary (Occam’s razor)

The biggest wins come from **fewer parallel “ways to do the same thing”**: duplicate correlation→multiplier math, two portfolio orchestrators, cache shims + singletons, **70+ node import shims**, and **pervasive `sys.path` glue** instead of one installable package. Second tier is **god-modules** (`portfolio.py`, `vault_manager.py`, `feature_base_model.py`) that mix pure math, I/O, and orchestration—splitting by responsibility reduces complexity without changing behavior.

---

## Tier 1 — Highest impact

### 1. Ensemble: one primitive for “diversification multiplier from correlation”
**FDM** (`weight_layer.py`) and **IDM** (`portfolio.py` for TF and global) repeat the same pattern: mean off-diagonal correlation → `sqrt(1/(mean + ε))` → cap, with slightly different matrix handling and caps.  
**Direction:** One parameterized pure function (signals vs returns, cap) removes drift and duplicated comments.

### 2. `ensemble/portfolio.py` as a single huge module
Roughly ~3k lines mixing `TFPortfolio`, `GlobalPortfolio`, cache queries, global `__GLOBAL__` encode/decode for `WeightLayer`, and vault delegation.  
**Direction:** Split **pure math**, **cache I/O**, **global adapter machinery**, and **public orchestration** so each change has a smaller blast radius.

### 3. Global combination via synthetic ticker `__GLOBAL__`
`ClusteredWeightLayer` is reused, but encode/decode/stream-id logic is a **second API** on top of ticker-oriented `WeightLayer`.  
**Direction:** Either isolate that in a small module with a clear name, or replace with an explicit “global combiner” type so `GlobalPortfolio` reads as fit → combine → apply IDM without adapter noise.

### 4. Two portfolio stories: `PortfolioManager` vs `GlobalPortfolio`
`PortfolioManager` appears thin and test/integration-oriented; production-style flows use `GlobalPortfolio`.  
**Direction:** One orchestrator with an explicit mode (TF-only vs global), or deprecate/document the legacy path so there is one mental model.

### 5. Nodes: flat import shims + `sys.modules` hacks
Many files under `nodes/` re-export canonical implementations (often via `sys.modules` replacement); `nodes/_taxonomy.py` already maps logical names to implementations.  
**Direction:** Long-term, **one import surface** (taxonomy + canonical paths) and removing shims deletes a large file count and duplicate “module identity.” High migration cost for external `from nodes.rsi import …` style imports.

### 6. Package install vs `sys.path` everywhere
Deployment modules, scripts, and tests repeat `sys.path` bootstrapping with **inconsistent** `parents[N]` depth.  
**Direction:** Editable install (`pip install -e .`) + optional `console_scripts` removes the most duplicated glue and class of path bugs.

### 7. Cache: shims + singleton `CentralCacheStore`
`utils/cache/*.py` often re-exports `runtime/`; `CacheManager` mutates `CentralCacheStore._instance` and `live_artifact_cache_dir`.  
**Direction:** Fewer shim modules (or exports only from `__init__.py`), and **injected** store/factory instead of global mutation + tests patching `_instance`.

---

## Tier 2 — Medium impact

| Area | Issue | Simpler direction |
|------|--------|-------------------|
| **Vault** | `vault_manager.py` very large; overlaps with `portfolio_vault.py` on JSON paths and snapshots | Submodules or named facets (`ensembles`, `snapshots`); shared path/JSON helpers |
| **Cache API** | `ensure_vault_cache_coverage` in `CacheManager` and again wrapped in `vault_manager` | One public entry; other internal/deprecated |
| **Paths** | `project_root()` in cache vs multi-strategy search in `vault_manager` | One repo-root resolver used everywhere |
| **Aliases** | `CentralCache` = `CentralCacheStore`; `Portfolio` = `TFPortfolio` | One public name each, long-term |
| **Cross-import** | `utils/data/cross_ticker_store.py` re-exports cache | Single import path |
| **`WeightLayer`** | Factory function named like a class | `make_weight_layer` or a thin real class |
| **`DiversifiedEnsemble`** | Many `__init__` paths (control file vs inline models) | Single builder or explicit source type |
| **RSI family** | Repeated rolling RSI state across several `nodes/mean_reversion/rsi/*.py` | One RSI rolling core; nodes only map to outputs |
| **`BaseModel`** | Large orchestrator in `feature_base_model.py` | Split cache vs extraction vs binning vs alignment |
| **Permutation tests** | Parallel continuous vs rule-based runners | One batch runner parameterized by strategy/protocol |
| **Deployment vs scripts** | `forecast_server` and `tws_live_forecast` both orchestrate full flows | Shared “live forecast runtime” library; deployment = thin HTTP/schedule/TWS adapters |

---

## Tier 3 — Cleanup and clarity

- **Nodes:** `nodes/archive/`, monkeypatched shims (e.g. Casey C wrapper mutating canonical class)—delete or fold metadata into canonical modules.  
- **Tests:** Duplicated helpers (`_source_data_available`, etc.) across integration files; heavy `sys.modules` stubs in deployment tests—central `tests.support` or `conftest`.  
- **`tests/integration/test_integration.py`:** Formula/synthetic checks vs repo-backed integration—rename/move so “integration” matches your AGENTS policy.  
- **Scripts:** `run_manual_forecast` reportedly importing a missing `deployment.test_forecast_server`—fix drift between scripts and tree.  
- **`feature_selection/base_models/__init__.py`:** Stale mentions of non-existent model names—docs-only fix.  
- **EDA:** Shared matplotlib/`Agg` and constants in `feature_selection/eda/`.

---

## How to sequence (pragmatic)

1. **Low-risk extractions first:** shared FDM/IDM helper, dedupe integration test helpers, one `project_root`/path resolver.  
2. **Structural:** split `portfolio.py` and trim cache shims once tests pin behavior.  
3. **High-churn:** node shim removal and `pip install -e .` need a migration plan (grep-based codemod, deprecation window).

---

If you want this turned into a tracked refactor backlog (issues or ADRs), switch to **Agent mode** and we can add a short doc or tickets; in **Ask mode** I can only analyze and recommend, not edit the repo.