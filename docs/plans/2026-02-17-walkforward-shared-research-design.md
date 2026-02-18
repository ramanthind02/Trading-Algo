# Design: Shared Walkforward Research Pipeline

**Date:** 2026-02-17
**Branch:** setup-feature-val
**Status:** Approved

---

## Problem

Researchers need a repeatable walkforward workflow that works across both rule-based and continuous features, highlights the selected feature per fold, and makes fold boundaries visually auditable before trusting results.

Current code has split capabilities in multiple places (`WalkForwardSplitter`, Stage 3 stability analysis, report plotting), but no single research-facing pipeline that:
- centralizes walkforward config in feature research configs,
- supports researcher-selected objective metrics by name,
- exports fold-level selection artifacts into a shared output layout by feature type.

---

## Chosen Approach

Build a **shared walkforward core** under `feature_research/walkforward/` and keep **thin adapters** in rule-based and continuous feature research packages.

- Shared core owns fold construction, per-fold grid scoring, best-feature selection, visualization, and artifact export.
- Rule-based and continuous adapters only provide feature extraction/evaluation hooks.
- Reuse existing Stage 3 walkforward stability logic and report plotting where possible; add a fold timeline plot for fold correctness checks.

This maximizes reuse, keeps behavior consistent across feature families, and avoids duplicate implementations.

---

## Architecture

### 1) Shared walkforward package

Add a new package:

- `feature_research/walkforward/config.py`
  - Shared walkforward config dataclass and validation.
- `feature_research/walkforward/metrics.py`
  - Metric-name registry (`sharpe`, `sortino`, `mean_return`, etc.) -> scorer callables.
- `feature_research/walkforward/runner.py`
  - Main orchestration over folds and param grid.
- `feature_research/walkforward/visualization.py`
  - Fold timeline plot and selection summary visual helpers.
- `feature_research/walkforward/io.py`
  - Shared output writing helpers.

### 2) Feature-type adapters

- `feature_research/rule_based/pipeline.py`
  - Integrate optional walkforward execution after existing EDA run path.
- `feature_research/continuous_binning/pipeline.py`
  - Integrate same shared runner via continuous adapter.

Adapters provide:
- param-grid expansion,
- per-param feature extraction,
- fold-level objective computation inputs.

---

## Configuration Design

Add a reusable config dataclass and include it in rule-based config now (continuous adopts same type):

```python
@dataclass(frozen=True)
class WalkforwardResearchConfig:
    enabled: bool
    train_start: datetime
    train_end: datetime
    test_step: int
    num_steps: int
    top_k: int
    objective_metric_name: str
    min_fold_samples: int
    output_root: Path
```

Rule-based config adds:
- `walkforward: WalkforwardResearchConfig`

Metric selection is string-based (approved):
- deterministic registry lookup,
- explicit error on unknown metric names.

---

## Fold Selection And Reporting Behavior

For each fold:
1. Evaluate all parameter combinations.
2. Compute objective score using `objective_metric_name`.
3. Compute neighbor-smoothed score (Stage 3 style).
4. Rank by smoothed score (tie-break by raw score, then canonical param label).
5. Set rank-1 as `selected_feature`.
6. Keep top-K list and permutation overlay flags when passers are available.

This ensures each fold explicitly shows the chosen feature and avoids unstable, isolated peaks via smoothing.

---

## Output Layout

All walkforward artifacts are written to a shared folder partitioned by feature type:

```text
feature_research/shared_results/
  {feature_type}/
    {module_name}/
      walkforward/
        folds.csv
        fold_scores.csv
        selection_summary.csv
        report.json
        walkforward_stability.png
        fold_timeline.png
```

Key files:
- `folds.csv`: fold boundaries and sample counts.
- `selection_summary.csv`: one row per fold with `selected_feature`, selected scores, top-K.
- `fold_timeline.png`: train/test bars per fold to verify fold correctness visually.

---

## Reuse Plan

Reuse existing components:
- `feature_selection.walkforward.walkforward_model.WalkForwardSplitter`
- `feature_selection.validation.stability_analysis.run_walkforward_stability`
- `feature_selection.validation.report_generator.plot_walkforward_stability`

Add only the missing research harness pieces (shared config, objective registry, timeline visualization, output writers).

---

## Testing Strategy

Unit tests:
- walkforward config validation and defaults,
- metric registry lookup/scoring,
- deterministic fold ranking and `selected_feature` emission,
- artifact writer paths and required columns.

Integration tests:
- Rule-based smoke run with cache-backed data asserting generated artifacts and `selected_feature` per fold.
- Continuous smoke run proving the same shared walkforward core works without rule-specific assumptions.

---

## Rollout Sequence

1. Add shared walkforward config and metric registry.
2. Implement shared runner and writers.
3. Add fold timeline visualization.
4. Wire rule-based pipeline to shared runner.
5. Wire continuous pipeline to shared runner.
6. Add/refresh tests and docs.

---

## Out Of Scope

- Changing live execution sizing or production deployment behavior.
- Reworking permutation Stage 1/2 logic.
- Vault schema changes.
