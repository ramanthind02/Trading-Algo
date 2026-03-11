# Weight Layer

`WeightLayer` is now a single always-clustered combiner in `ensemble/weight_layer.py`.

It is used by `GlobalPortfolio` at the cross-timeframe layer:

```text
Base Models -> DiversifiedEnsemble -> TFPortfolio -> GlobalPortfolio(weight_layer=WeightLayer(...))
```

## Design

The layer only has two modes:

| Mode | Behavior |
|---|---|
| `cluster_equal` | Cluster model forecast streams, equal weight across clusters, equal weight within each cluster |
| `cluster_corr_ulcer` | Cluster first, then tilt cluster weights by inverse average positive correlation and ulcer index on forecast-weighted returns |

There are no manual group definitions, no feature-family grouping, and no HRP path.

## Clustering

Clustering is always automatic on the forecast streams handed into `GlobalPortfolio`:

1. Build a date x model matrix from model forecast values.
2. Compute the model correlation matrix.
3. Clip negative correlations to `0.0`.
4. Convert correlation to distance with `sqrt(0.5 * (1 - rho))`.
5. Run hierarchical clustering and cut at `rho_cut`.

Within each cluster, member models are averaged equally.

## Weighting

### `cluster_equal`

If there are `K` clusters:

```text
cluster_weight_k = 1 / K
model_weight_i = cluster_weight_k / n_members(cluster_k)
```

### `cluster_corr_ulcer`

For each cluster `c`:

```text
score_c = 1 / ((max(ulcer_index_c, eps) ** alpha) * (1 + avg_positive_corr_c))
```

Where:

- `avg_positive_corr_c` is the mean positive correlation of cluster `c` versus the other clusters
- `ulcer_index_c` is computed on `cluster_forecast_c * instrument_return`
- `alpha` controls tilt strength

Scores are normalized, capped by `group_weight_cap`, then distributed equally within each cluster.

## FDM

FDM is computed from cluster-level forecast correlations for both modes:

```text
mean_corr = mean(off_diagonal(cluster_corr))
FDM = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)
```

Single-model and single-cluster cases use `FDM = 1.0`.

## Config

`WeightLayerConfig` exposes only:

| Parameter | Default | Description |
|---|---|---|
| `weighting_method` | `cluster_equal` | `cluster_equal` or `cluster_corr_ulcer` |
| `rho_cut` | `0.70` | Correlation cutoff for automatic clustering |
| `fdm_max` | `2.0` | FDM cap |
| `group_weight_cap` | `0.25` | Hard cap per cluster |
| `risk_tilt_alpha` | `0.5` | Ulcer tilt strength for `cluster_corr_ulcer` |

## Diagnostics

Per ticker, the layer reports:

- `weights`
- `cluster_assignments`
- `cluster_weights`
- `cluster_metrics`
- `fdm`
- `mean_cluster_correlation`

`cluster_metrics` contains per-cluster `avg_positive_corr`, `ulcer_index`, `score`, and `member_count`.
