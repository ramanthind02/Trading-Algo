# Weight Layer

`WeightLayer` in `ensemble/weight_layer.py` combines the global forecast streams produced by `GlobalPortfolio`.

The `WeightLayer` implementation itself is unchanged in the global sector cutover.
`GlobalPortfolio` now adapts multi-ticker/timeframe streams into a synthetic ticker
(`__GLOBAL__`) and decodes weighted outputs back to real tickers after combine.

```text
Base Models -> DiversifiedEnsemble -> TFPortfolio -> GlobalPortfolio(weight_layer=WeightLayer(...))
```

## Modes

| Mode | Behavior |
|---|---|
| `equal_signal` | Equal weight across all signals |
| `inverse_avg_pairwise_corr` | Weight signals by inverse average positive pairwise correlation |
| `hrp_cluster_equal` | Build HRP linkage, cut at `rho_cut`, equal weight across groups, equal weight within each group |
| `hrp_classic` | Classic HRP recursive bisection on the linkage tree |

`equal_signal` is the default.

## Estimation Stack

For each ticker:

1. Build a date x model forecast matrix.
2. Drop all-empty rows and fill remaining gaps with `0.0`.
3. Standardize each signal by its full-sample in-sample sample standard deviation.
4. Fit Ledoit-Wolf covariance on the standardized matrix.
5. Derive correlation from that covariance.
6. Use full correlation for HRP distance/linkage, and positive-clipped correlation for scoring and FDM.

Distance for clustering:

```text
d_ij = sqrt((1 - rho_ij) / 2)
```

Linkage:

```text
Ward linkage
```

## Weighting

### `equal_signal`

```text
model_weight_i = 1 / N
```

### `inverse_avg_pairwise_corr`

For each signal `i`:

```text
score_i = 1 / (1 + mean_positive_corr_i)
weight_i = score_i / sum(scores)
```

### `hrp_cluster_equal`

1. Build the full HRP tree.
2. Cut the tree at `rho_cut`.
3. Assign equal weight across groups.
4. Apply `group_weight_cap`.
5. Split each group equally across members.

### `hrp_classic`

1. Build the full HRP tree.
2. Order leaves quasi-diagonally.
3. Recursively split the ordered tree.
4. Allocate left/right branches inversely to branch variance using inverse-variance branch weights.

`group_weight_cap` does not apply to `hrp_classic`.

## FDM

FDM is based on the raw signal-level positive-clipped correlation matrix for all modes:

```text
mean_corr = mean(off_diagonal(signal_corr_positive))
FDM = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)
```

Single-model and fallback cases use `FDM = 1.0`.

## Config

| Parameter | Default | Description |
|---|---|---|
| `weighting_method` | `equal_signal` | One of the four modes above |
| `rho_cut` | `0.70` | Cut threshold for reporting groups and `hrp_cluster_equal` |
| `fdm_max` | `2.0` | FDM cap |
| `group_weight_cap` | `0.25` | Hard cap for `hrp_cluster_equal` group weights |

## Diagnostics

Per ticker, the layer reports:

- `weights`
- `cluster_assignments`
- `cluster_weights`
- `cluster_metrics`
- `fdm`
- `mean_signal_correlation`
- `mean_cluster_correlation`

The `cluster_*` field names are retained for compatibility with existing reports, but they now represent generic allocation groups for non-clustered modes.
