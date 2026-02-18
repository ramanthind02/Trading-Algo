# Feature Selection Validation API

> **Note:** This module provides the public API for the three-phase permutation testing suite.

## Overview

The `feature_selection.validation` package exposes configuration classes and utilities for running permutation tests across the validation pipeline.

## Public Exports

### Configuration Classes

| Class | Description |
|-------|-------------|
| `InSamplePermutationConfig` | Configuration for in-sample permutation testing (Phases 1-3). Includes objective metric, significance level, replicate count, and null type selection. |
| `OutOfSamplePermutationConfig` | Configuration for out-of-sample validation phase. Used after the ensemble is locked. |
| `PermutationTestConfig` | Base configuration for individual permutation tests. |
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
    objective_metric=ObjectiveMetricSpec.SHARPE,
    significance_level=0.1,
    replicate_count=500,
)

# Resolve metric
metric_fn = resolve_objective_metric(ObjectiveMetricSpec.SHARPE)
```

## See Also

- [In-Sample Permutation Testing](../library/Feature_selection/Permutation%20Testing/in-sample_pt.md)
- [Candle Permutation Specs](../library/Feature_selection/Permutation%20Testing/candle_permutation_specs.md)
