# Parameter Sensitivity

> [!important]
> The source-of-truth specs for this topic are
> [[SaaS/robustness_tests/parameter_sensitivity]]
> and [[SaaS/robustness_tests/parameter_selection]].

## Role in the workflow

Parameter sensitivity sits inside **exploration**, after the in-sample robustness review and before validation.

Its question is:

> Is the chosen combination sitting on a robust plateau, or on a narrow spike that is unlikely to survive validation?

## Why this matters

Search-bias tests and DSR ask whether the result might be explained by search.

Parameter sensitivity asks something different:

- if the selected point moves slightly, does the result stay good?
- or does the apparent edge disappear as soon as the optimizer is nudged?

That is why the SaaS workflow treats sensitivity as its own step before parameter lock.

## The isolated peak problem

```text
Scenario A - stable region
Lookback:   2    3    4    5    10   14   20
Metric:    0.6  0.9  1.1  1.0  0.5  0.3  0.1
                └── stable plateau ──┘

Scenario B - isolated peak
Lookback:   2    3    4    5    10   14   20
Metric:    0.1  0.2  1.1  0.3  0.2  0.1  0.0
                      ↑
                 isolated spike
```

In Scenario B the best point can still look strong in isolation, but the neighborhood says the optimizer probably found noise rather than a durable edge.

## SaaS perturbation lens

The perturbation test uses **min-step axis-aligned neighbours** (not optimisation-grid spacing). For each configured parameter, the pipeline re-runs the strategy at `chosen ± min_step` while holding other parameters fixed. Typical cost: `2 × n_params` fresh evaluations (e.g. 8 for four parameters).

Configure per-parameter steps in `ParamSensitivityConfig.perturbation_specs` (`ParamPerturbationSpec` from `quantfoundry_core.robustness`).

Reports (artifacts: `perturbation_report.json`, `perturbation_runs.csv`):

- `peak_metric` — chosen combo
- `median_metric` — median across **neighbour** runs only
- `optimism_bias = peak - median`
- `stability_ratio = median / peak` — near 1.0 means the edge barely moves at min-step shifts
- `passed` — perturbation median ≥ configured floor

That perturbation median is the honest shrinkage view of the selected result.

## Grid-neighbor smoothing (EDA surface)

Separate from min-step perturbation: the EDA grid export applies **grid-step** neighbour smoothing for surface plots.

## Local surface-reading lens

Locally, the most useful companion to the perturbation test is the parameter surface itself.

The main artifacts are:

- `param_sensitivity.csv`
- `param_sensitivity_by_ticker.csv`
- `param_combo_long.csv`
- the research workspace pivot explorer (Parameter Sensitivity section)

Read them together:

- the perturbation median tells you how much optimism to remove
- the surface tells you whether the selected point sits inside a believable plateau
- the by-ticker view tells you whether that plateau is broad or driven by one market

## Neighbor smoothing

For each parameter combo `P`, the **smoothed objective** is a weighted average of `P` and its 1-step axis-aligned neighbors.

```text
smoothed(P) = (self_weight * obj(P) + sum(obj(N))) / (self_weight + n_neighbors)
stability_ratio = smoothed(P) / obj(P)
```

Higher `self_weight` keeps boundary points from being over-diluted when they have fewer neighbors.

## Reading the surface

|  | High smoothed | Low smoothed |
|---|---|---|
| High raw | likely genuine signal in a stable region | isolated peak, suspicious |
| Low raw | acceptable neighbor inside a good region | consistently poor |

> [!warning]
> A high stability ratio with a low absolute metric is not good. It often just means the whole neighborhood is consistently weak.

## Decision rule

Use this order during exploration:

1. review the in-sample robustness summary
2. inspect the perturbation summary
3. inspect the surface and heatmaps
4. choose a combination from the robust neighborhood, not just the raw maximum
5. lock that choice for validation

## What this page is not

- not a separate walk-forward phase
- not a replacement for validation
- not permission to keep tuning after seeing the surface

The point of this stage is to lock one defensible definition and move on.

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/exploration]]
- [[Feature_selection/permutation_testing]]
- [[Feature_selection/validation]]
