# Bias Node Taxonomy Design

Date: 2026-02-21
Status: Approved (design)

## Context

The `nodes/` package currently mixes many active bias nodes at the top level. This makes strategy-level discovery and maintenance harder as the node set grows.

Current system constraints that must remain unchanged:

- `create_bias_node(module_name, ticker, tf, params)` callers continue passing flat module names (for example, `"rsi"`).
- Callers must not need to specify folders or categories.
- Existing direct imports like `from nodes.rsi import RSI` should keep working.
- `nodes/archive/` remains untouched in this reorganization.

## Goals

- Organize active nodes by strategy category and optional family subfolders.
- Support multi-category placement without duplicating implementation logic.
- Preserve external loading and import behavior.
- Keep migration deterministic and testable.

## Chosen Approach

Adopt a canonical categorized structure with compatibility shims.

### 1) Canonical categorized structure

Move active node implementations from `nodes/*.py` into strategy/family folders under `nodes/`.

Illustrative categories:

- `nodes/mean_reversion/`
- `nodes/momentum/`
- `nodes/seasonal/`
- `nodes/breakout/`
- `nodes/regime/`
- `nodes/volatility/`
- `nodes/buy_hold/`

Family subfolders are used when a category has many close variants (for example RSI family under mean reversion).

### 2) Compatibility shims at nodes root

Keep root modules (for example `nodes/rsi.py`) as thin re-export shims that import from canonical locations.

Result: existing direct imports and tooling expectations continue to work while code ownership moves to categorized folders.

### 3) Multi-category references via re-export modules

Each node has exactly one canonical implementation file. If a node belongs to multiple categories, secondary placements are reference modules that re-export the canonical class.

This avoids code copies while still surfacing nodes in multiple conceptual folders.

### 4) Loader behavior remains externally flat

`create_bias_node(...)` keeps the same public contract and resolves modules recursively under `nodes/`.

Resolution order:

1. Taxonomy manifest mapping (if present)
2. Recursive basename resolution fallback

If recursive fallback finds multiple candidates for the same module name, fail fast with an explicit ambiguity error requiring a canonical manifest entry.

## Taxonomy and Mapping Model

Introduce a small internal registry (for example `nodes/_taxonomy.py`) with:

- canonical module path by flat module name
- optional alias/reference category placements

The registry is used for deterministic resolution and can also drive generation/validation of reference modules.

## Migration Plan

1. Build a classification table for each active node at `nodes/*.py` (excluding `nodes/archive/`).
2. Move implementation files to canonical categorized locations.
3. Replace former root files with compatibility shim modules.
4. Create optional secondary reference modules for multi-category discovery.
5. Update loader resolution internals without changing function signature or caller behavior.
6. Add guardrails for duplicate basenames and missing mappings.

## Validation Plan

### Functional checks

- `create_bias_node("<module>", ...)` works for all migrated modules.
- Existing direct imports from `nodes.<module>` still resolve.
- `extract_features(...)` and downstream research loader paths continue to run without call-site changes.

### Targeted tests

- Node unit tests under `tests/unit-tests/nodes/`
- Cache/extraction tests that instantiate via `create_bias_node`
- Integration path exercising `feature_extraction/feature_extractor.py` and `feature_research/rule_based/data_loader.py`

## Risks and Mitigations

- Ambiguous module name resolution after introducing nested folders.
  - Mitigation: canonical manifest plus explicit ambiguity errors.
- Drift between canonical modules and reference/shim modules.
  - Mitigation: generated references or registry-driven validation.
- Hidden direct imports from old root paths.
  - Mitigation: preserve root compatibility shims for backward compatibility.

## Non-Goals

- Reorganizing `nodes/archive/` in this change.
- Changing public extraction APIs or research config schema.
- Altering bias node mathematical logic.
