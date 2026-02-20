# Permutation Testing Phase Design (Approach 1)

Date: 2026-02-17
Author: OpenCode
Status: Approved

## Context

This design formalizes the permutation-testing phase for feature research using a shared framework across both:

- `feature_research/continuous_binning/`
- `feature_research/rule_based/`

The design aligns with:

- `docs/kanban/complete/permutation_testing/T013_vector_shuffle_stage1.md`
- `docs/kanban/complete/permutation_testing/T014_pipeline_permutation_stage2.md`
- `docs/kanban/complete/permutation_testing/T015_walkforward_stability_stage3.md`
- `docs/kanban/complete/permutation_testing/T016_early_stopping_orchestration.md`
- `docs/kanban/complete/permutation_testing/T017_permutation_report_generation.md`
- `docs/library/Feature_selection/Permutation Testing/`

## Approved Direction

Use **Approach 1**: shared permutation core in `feature_selection/validation/` with thin adapters from both feature-research pipelines.

This keeps one source of truth for permutation logic while avoiding code duplication across continuous and rule-based paths.

## Goals

1. Enforce vector-shuffle-first compute gating before expensive candle-shuffle tests.
2. Run tests over all parameter combinations, then filter using configured pass/fail criteria.
3. Support researcher-provided objective metrics from config.
4. Support custom candle-data injection into bias-node extraction for candle-shuffle permutations.
5. Execute three phases end-to-end:
   - in-sample
   - walkforward diagnostics
   - out-of-sample
6. Produce researcher-ready reports with per-parameter and aggregate statistics.

## Non-Goals

- Automated final ensemble selection (researcher remains final decision-maker).
- Changes to bias-node schema semantics or feature naming conventions.
- Replacement of existing cache strategy.

## High-Level Architecture

### Shared Validation Core

Primary execution remains in `feature_selection/validation/`:

- `config.py`: permutation suite configuration models
- `permutation_tests.py`: stage-level permutation runners
- `stability_analysis.py`: walkforward diagnostics and stability computations
- `orchestration.py`: phase sequencing, early-stopping gates, aggregation
- `reports.py`: report dataclasses and unified suite data contracts
- `report_generator.py`: markdown/json/html and plot generation

### Pipeline Adapters

Thin adapters in:

- `feature_research/continuous_binning/pipeline.py`
- `feature_research/rule_based/pipeline.py`

Adapters are responsible for:

- assembling extractor/model factories
- loading base config
- invoking shared orchestration

Adapters are not responsible for permutation logic.

## Config Design

Extend research configs with nested permutation suite config.

### Objective Metric Spec

Support both built-in and custom callable metric definitions:

- `builtin`: named metric key (for example: `sharpe`, `sortino`, `calmar`, `profit_factor`)
- `callable_path`: import path (`module:function`)
- `kwargs`: optional metric keyword arguments

Validation rule: exactly one of `builtin` or `callable_path` must be provided.

### Phase Config Blocks

- `in_sample`
  - `nreps`, `alpha`, `metric_threshold`, `permute_start_idx`
  - stage-2 mode order for continuous defaults to `feature_shuffle -> candle_shuffle`
- `walkforward`
  - `enabled`, `fold_structure`, `top_k`, `min_folds_stable`
- `out_of_sample`
  - `enabled`, `start`, `end`, `locked_selection_source`
  - `locked_selection_source` values:
    - `stage2_passers`
    - `stable_intersection`

### Reporting and Reproducibility

- `report.output_dir`
- `report.formats` (`markdown`, `json`, optional `html`)
- `report.save_plots`
- `report.save_null_distributions`
- top-level `random_seed` with deterministic derived seeds per repetition

## Phase Execution Flow

### Phase 1: In-Sample Permutation Funnel

For each parameter combination:

1. Run vector shuffle permutation first.
2. If vector stage fails, stop for that parameter.
3. If continuous feature:
   - run feature-shuffle pipeline permutation
   - if pass, run candle-shuffle pipeline permutation
4. If rule-based feature:
   - run candle-shuffle pipeline permutation only

This enforces compute-efficient ordering and protects expensive candle-shuffle runs.

### Phase 2: Walkforward Stability Diagnostics

- Evaluate all parameter combinations across configured folds.
- Compute fold-level top-k and consistency diagnostics.
- Overlay in-sample passers to show overlap between statistical validity and temporal stability.

### Phase 3: Out-of-Sample Validation

- Use configured OOS date range.
- Candidate source chosen from config (`stage2_passers` or `stable_intersection`).
- For each candidate, run OOS vector shuffle first, then OOS candle shuffle for vector passers.
- Produce per-parameter OOS pass/fail plus aggregate OOS funnel statistics.

## Candle Data Injection Contract

Bias-node extraction helpers should accept optional external candles for permutation runs.

Design intent:

- normal runs: existing load/cached behavior
- permutation runs: use `candles_override` for shuffled candle extraction

This enables candle-shuffle tests without duplicating extraction pipelines.

## Reporting Design

### Per-Parameter Decision Record

Produce one normalized record per parameter combo containing:

- phase-1 vector stats and verdict
- phase-1 stage-2 stats and verdict(s)
- phase-2 fold stability diagnostics and flags
- phase-3 OOS stats and verdict
- final status (`candidate`, `rejected`, `needs_review`)

### Aggregate Suite Outputs

Generate:

- `suite_summary.md`
- `suite_summary.json`
- optional `suite_summary.html`
- stage-level plots (including phase-3 OOS null distributions)
- `combo_decision_table.csv`

Include funnel statistics and compute-savings summaries derived from early stopping.

## Testing Strategy

### Unit Tests

Add/extend tests under `tests/unit-tests/validators/permutation/` for:

- metric resolver (built-in + callable)
- vector-first and stage2 ordering gates
- candle override plumbing through extraction entrypoints
- OOS gating and pass/fail behavior
- suite aggregation and funnel math
- report decision table completeness

### Integration Tests

Add/extend integration tests under `tests/integration/feature_validator/`:

- full 3-phase suite for continuous pipeline
- full 3-phase suite for rule-based pipeline
- cache-aware real data flow with skip behavior when cache missing
- report artifact existence and non-empty checks

## Risks and Mitigations

1. **Risk:** Expanded config complexity
   - **Mitigation:** nested dataclasses with strict validation and defaults
2. **Risk:** Runtime cost for wide parameter grids
   - **Mitigation:** hard-gated vector-first and staged continuous mode ordering
3. **Risk:** Extraction-path drift between normal and permutation runs
   - **Mitigation:** reuse the same extraction helpers with optional candle override

## Rollout Plan

1. Extend config/report contracts in shared validation layer.
2. Add OOS phase and sequencing changes in orchestration.
3. Wire candle override support through loaders/extractors.
4. Hook both feature-research pipelines into shared suite runner.
5. Add unit tests, then integration tests, then finalize reporting outputs.
