# Feature Selection Validation API

> **Note:** This module provides the public API for the three-phase permutation testing suite.

## Overview

The `feature_selection.validation` package exposes configuration classes and utilities for running permutation tests across the validation pipeline.

## Public Exports

### Configuration Classes

| Class | Description |
|-------|-------------|
| `InSamplePermutationConfig` | Configuration for in-sample permutation stages (Stage 1/2 execution, replicate count, alpha, stage-2 permutation mode, and stage toggles). |
| `OutOfSamplePermutationConfig` | Configuration for out-of-sample validation phase. Used after the ensemble is locked. |
| `PermutationTestConfig` | Composite config for in-sample, walkforward, and OOS permutation phases, with convenience scalar arguments and computed properties. |
| `WalkforwardPermutationConfig` | Configuration for walkforward stability analysis (Phase 3). |
| `PermutationReportConfig` | Configuration for generating permutation test reports. |

### Objective Metrics

| Symbol | Description |
|--------|-------------|
| `ObjectiveMetricSpec` | Specification for the objective metric used across all permutation tests (e.g., Sharpe, Sortino). |
| `resolve_objective_metric` | Utility function to resolve metric name to callable. |

## Usage

```python
from feature_selection.validation import (
    InSamplePermutationConfig,
    ObjectiveMetricSpec,
    resolve_objective_metric,
)

# Configure the permutation test suite
config = InSamplePermutationConfig(
    nreps=500,
    alpha=0.1,
    run_stage1=True,
    run_stage2=False,
)

# Resolve metric
metric_fn = resolve_objective_metric(ObjectiveMetricSpec(builtin="sharpe"))
```

## See Also

- [In-Sample Permutation Testing](../library/Feature_selection/Permutation%20Testing/in-sample_pt.md)
- [Candle Permutation Specs](../library/Feature_selection/Permutation%20Testing/candle_permutation_specs.md)
