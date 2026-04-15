
Here is a consolidated **approval brief** from four parallel subagent passes (ensemble, feature_research vs engines, cache/vault/nodes, bootstrap/config/JSON). Nothing was changed in the repo.

---

## 1. Ensemble / portfolio (highest “duplicate product” signal)

| Proposal | Idea | Effort | Risk | LOC / complexity |
|----------|------|--------|------|------------------|
| **A1 — Retire or fold `PortfolioManager`** | `GlobalPortfolio` is the real multi-TF path; `PortfolioManager` looks like a **second** orchestrator (concat per-TF, no cross-TF weighting), used mainly from `ensemble/__init__.py` + `tests/integration/test_portfolio_manager.py`. | **L** | **M–H** if anything external imports it; **L** if truly internal | **~300 LOC** + clearer single entrypoint |
| **A2 — Split `ensemble/portfolio.py` monolith** | `TFPortfolio` + `GlobalPortfolio` + vault/materialization glue in one file → submodules + stable `ensemble.portfolio` re-exports. | **L** | **M** | Mostly **navigation**, modest raw LOC |
| **A3 — “Global combination” subpackage** | Group `global_weight_layer_adapter`, `global_portfolio_runtime`, `global_portfolio_diagnostics`, `portfolio_global_streams` under one package. | **M** | **M** | **Clarity** > LOC; move-only if disciplined |
| **A4 — Single portfolio persistence boundary** | Unify mental model for `portfolio_vault` vs `utils/cache/.../portfolio_materialization` (who owns serialize/load of `GlobalPortfolio`). | **L** | **H** (deployment/cache paths) | **Clarity**; modest LOC if not a big rewrite |
| **A5 — Typed JSON for control + vault** | Overlap with cross-cutting proposal **C4** below. | **L** | **M–H** | Large validator LOC in `ensemble_utils` / `vault_feature_files` |

**Note:** Subagent confirmed **FDM/IDM math is already shared** (`weight_layer`); don’t “merge layers” for that reason alone.

**Approval question:** Do you want to **investigate deprecating `PortfolioManager`** (usage audit + migration plan) as phase 1?

---

## 2. Feature research vs engines (biggest “phase glue” duplication)

| Proposal | Idea | Effort | Risk | LOC / complexity |
|----------|------|--------|------|------------------|
| **B1 — One CLI: `python -m feature_research <phase>`** | Replace many `run_*.py` / `run_phase.py` thin scripts with one dispatcher (argparse/Typer). | **M** | **M** (docs, muscle memory) | Drops **repeated bootstrap/argparse**; not much engine LOC |
| **B2 — One “signed-signal phase runner” in `utils.evaluation`** | Merge overlap between `feature_research/pipelines/_shared.py` and `feature_research/validation/permutation_helpers.py` (load, windows, walkforward wiring). | **L–M** | **M** (layering, cycles) | **Large** if done fully; needs **`Protocol`** to avoid `feature_selection` ↔ `feature_research` cycles |
| **B3 — One permutation config model** | Fold `PermutationResearchConfig` (`feature_research/config.py`) into `PermutationTestConfig` tree (`feature_selection/validation/config.py`) or one shared module. | **M** | **H** (blast radius) | **High** conceptual + LOC win; touches many call sites |
| **B4 — One objective-metric registry** | Unify `feature_selection/validation/objective_metrics.py` vs `utils/evaluation/walkforward/metrics.py` (Sharpe/t-stat etc.). | **M** | **H** (**numeric / test golden** churn) | **Medium–high** LOC; big correctness review |
| **B5 — Thin `feature_research`** | Policy: `feature_research` = config + CLI; engines live in `feature_selection` + `utils` (combine B2–B4). | **L** | **H** | **Largest** long-term simplification if you commit to layering rules |
| **B6 — Rename duplicate `permutation_helpers`** | `walkforward/permutation_helpers.py` vs `feature_research/validation/permutation_helpers.py` → names that encode role. | **S** | **L** | **Clarity**; small LOC |

**Approval question:** Which sequence do you prefer: **(B1 only)** quick win, **(B6 + B2)** structural clarity, or **(B3 + B4)** deep unification (highest risk)?

---

## 3. Cache, vault, nodes (infrastructure sprawl)

| Proposal | Idea | Effort | Risk | LOC / complexity |
|----------|------|--------|------|------------------|
| **C1 — One public cache import tree** | Drop `utils/cache/*.py` shims after codemod to `utils.cache` / `utils.cache.runtime` only. | **M** | **L–M** (mechanical) | **Moderate** file + import churn |
| **C2 — `BiasNodeCache` as façade of central store** | One place for path/exists/load semantics. | **M** | **M** | **Correctness clarity**; moderate LOC |
| **C3 — Single vault module boundary** | Merge or colocate `vault_manager.py` + `vault_feature_files.py` (or `ensemble/vault/` package). | **M** | **L–M** | **Navigation + some LOC** |
| **C4 — Explicit vault root** | Replace cwd walking with repo-anchored default + env override. | **S–M** | **M** (CI/deploy assumptions) | Simpler mental model |
| **C5 — Nodes: registry only, drop `rglob` fallback** | Every `module_name` in `CANONICAL_MODULE_IMPORTS`; remove ambiguous filesystem search in `helpers._resolve_bias_node_import_path`. | **M** | **M** (breaks ad-hoc experiments) | **Safety**; optional later: remove root `nodes/*.py` shims |
| **C6 — Optional: class registry vs `inspect.getmembers`** | Map `module_name → BiasNode` class explicitly. | **M** | **M** | Less fragility; medium churn |

**Approval question:** Are you willing to **mandate taxonomy registration** (C5) for all nodes, or keep the rglob escape hatch?

---

## 4. Bootstrap + config + JSON (cross-cutting, good LOC-per-effort)

| Proposal | Idea | Effort | Risk | LOC / complexity |
|----------|------|--------|------|------------------|
| **D1 — `utils/repo_bootstrap.py` (single `find_repo_root` / `ensure_repo_root_on_syspath`)** | Replace `feature_research/bootstrap`, `scripts/_bootstrap`, `deployment/_bootstrap`, and inlined copies in `portfolio_research`, `prop_firms`, double-bootstrap in `binning/run_phase.py`, etc. | **M** (many small files) | **L** | **High aggregate** small-line deletion |
| **D2 — Unified research settings (e.g. Pydantic)** | One tree for `ResearchConfig`, windows, permutation, walkforward slice, binning; shrink `build_runtime_walkforward_config` and `runner` `getattr` protocols. | **L** | **H** | **Large** over time |
| **D3 — Delete `feature_research/in_sample/config.py` shim** | Import `feature_research.config` everywhere; remove re-export file. | **S** | **L** | Small but removes indirection |
| **D4 — Pydantic (or similar) for ensemble + vault JSON** | Replace imperative validators in `ensemble_utils` / `vault_feature_files` / parts of `vault_manager`. | **L** | **M–H** (on-disk compatibility) | **Very large** potential shrink in validator LOC |
| **D5 — Script policy: one-line bootstrap** | All entrypoints call shared `ensure_repo_root_on_syspath` only. | **S** (policy + pass) | **L** | Stacks with **D1** |
| **D6 — Generated JSON Schema for docs** | Optional; reduces doc/schema drift. | **S–M** | **L** | More **docs** than `.py` LOC |

**Approval question:** Do you want **D1 + D5** approved as a **first milestone** (low behavioral risk), with **D4** as a **second** milestone after schema inventory?

---

## Suggested approval packages (pick one or mix)

1. **Quick wins, low behavior risk:** **D1, D5, D3, B6, C1** (bootstrap + shims + rename + cache imports).  
2. **Portfolio clarity:** **A1** (audit + deprecate `PortfolioManager`) + optionally **A2**.  
3. **Research stack simplification (medium risk):** **B1 + B2** (after Protocol design).  
4. **Deep unification (high risk):** **B3 + B4 + D2 + D4** — schedule explicit numeric regression / test baselines.

Reply with which **IDs** (e.g. “approve D1, D5, A1 audit only”) or a **package number**, and we can turn that into a sequenced plan in Agent mode.