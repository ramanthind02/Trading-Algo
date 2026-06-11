# Weight Layer

`WeightLayer` in [`ensemble/weight_layer.py`](../../../ensemble/weight_layer.py) combines forecast streams at the **global** stage: it assigns weights across base-model streams (per synthetic global ticker) and applies the Forecast Diversification Multiplier (FDM).

`WeightLayer(...)` is a **factory function** that returns a `ClusteredWeightLayer` (the only concrete implementation; `BaseWeightLayer` is the ABC). [`GlobalPortfolio`](../../../ensemble/portfolio_impl/global_portfolio_impl.py) encodes multi-ticker / multi-timeframe streams into a synthetic global ticker (`__GLOBAL__`) and decodes the weighted output back to real tickers after combine.

```text
Base Models -> DiversifiedEnsemble -> TFPortfolio -> GlobalPortfolio(weight_layer=WeightLayer(...))
```

## Modes

The factory validates `weighting_method` against `_WEIGHT_METHODS`. The supported values are:

| Mode | Behavior |
|---|---|
| `equal_signal` (default) | Equal weight across all signals (`1/N`). |
| `inverse_avg_pairwise_corr` | Weight each signal by inverse average positive pairwise correlation. |
| `hierarchy_equal` | Manual nested tree: equal weight among siblings at each branch; leaf weight is the product of branch fractions. Requires a `hierarchy_spec` or `hierarchy_path`. |
| `inverse_corr_hierarchy` | Equal-split the tree, then redistribute each group's mass by inverse within-group correlation. Requires a hierarchy. |
| `ledoit_wolf_min_corr` | Score each signal by `1 / sum_of_positive_corr` and normalize. |
| `risk_parity_corr` | Score each signal by `1 / sqrt(sum_of_positive_corr)` and normalize. |
| `hierarchy_theme_inv_corr` | Hierarchy with top-level themes scored by inter-theme inverse-average correlation; equal within theme. Requires a hierarchy. |
| `hierarchy_theme_ledoit` | As above, but themes scored with a Ledoit-Wolf penalty. Requires a hierarchy. |
| `ledoit_wolf_hierarchy_within` | Ledoit-Wolf theme scoring **and** Ledoit-Wolf within-theme scoring. Requires a hierarchy. |

The hierarchy-requiring methods (`hierarchy_equal`, `inverse_corr_hierarchy`,
`hierarchy_theme_inv_corr`, `hierarchy_theme_ledoit`, `ledoit_wolf_hierarchy_within`) raise
at config time if neither `hierarchy_spec` nor `hierarchy_path` is provided.

> The recommended production configuration for multi-asset vaults is `hierarchy_equal` with
> an asset-first 3-level tree (see below). `equal_signal` is the factory default and the
> fallback `GlobalPortfolio` uses when no weight layer is supplied.

### Legacy methods removed

HRP-based methods (`hrp_cluster_equal`, `hrp_classic`) and `optimize_sortino_capped` were
removed (`_LEGACY_REMOVED_METHODS`). Saved snapshots that still reference one of these fail
`deserialize_weight_layer_state` with an explicit error — re-fit the portfolio with a current
method (`equal_signal`, `inverse_avg_pairwise_corr`, or `hierarchy_equal`). The `risk_tilt_alpha`
keyword is likewise rejected by the factory. Distance/Ward linkage is **not** used for weighting.

## Estimation stack (correlation and FDM)

For each ticker (`_prepare_signal_matrix` / `_estimate_covariance_and_correlation`):

1. Build a date × model forecast matrix (`_pivot_ticker_forecasts`).
2. Drop all-empty rows and fill remaining gaps with `0.0`.
3. Standardize each signal by its full-sample in-sample standard deviation (ddof=1).
4. Fit Ledoit-Wolf covariance on the standardized matrix.
5. Derive correlation from that covariance.
6. Positive-clip the correlation matrix (`_positive_clipped_correlation`, off-diagonal floored at 0, diagonal set to 1) for FDM and for the correlation-based scoring methods.

Single-model tickers, insufficient history (`< 2` rows), all-constant signals, or covariance
failures fall back to single-model / equal weights with `FDM = 1.0`.

## Weighting

### `equal_signal`

```text
model_weight_i = 1 / N
```

### `inverse_avg_pairwise_corr`

For each signal `i`:

```text
score_i  = 1 / (1 + mean_positive_corr_i)
weight_i = score_i / sum(scores)
```

### Hierarchy methods

Provide either:

- `hierarchy_spec`: nested JSON mapping (root is a `type: "group"` node with `children`), or
- `hierarchy_path`: path to a JSON file with the same shape.

Leaves may specify `stream_id` directly (`ticker::timeframe::model_name`) or `ticker`,
`timeframe`, and `model_name`. The fit step validates that declared leaves match the available
model names (strict coverage). See `parse_hierarchy_spec`, `resolve_hierarchy_for_fit`, and
`compute_equal_split_weights` in [`ensemble/weight_hierarchy.py`](../../../ensemble/weight_hierarchy.py).

`hierarchy_equal` uses the tree only for weights (equal split among siblings); correlation is
still computed for FDM diagnostics. The other hierarchy methods adjust within-group or
inter-theme mass using the positive-clipped correlation matrix as described in the table above.

### Asset-first 3-level hierarchy (recommended for portfolios)

For multi-asset vaults, prefer a **3-level** tree so commodity and equity streams are not
siblings under one strategy label:

```text
root → asset_class → strategy_group (vault weight_hierarchy_group) → stream leaves
```

Vault folder layout is unchanged; rebucketing uses the ticker in each `stream_id`
(`buy_hold`, `seasonal`, and `es_tlt` included — e.g. an ES rebalancing flow →
`equity_indices/es_tlt`). See [`docs/SaaS/weight_layer_spec.md`](../../SaaS/weight_layer_spec.md) for rationale.

| Module | Role |
|---|---|
| [`ensemble/vault/constants.py`](../../../ensemble/vault/constants.py) | `TICKER_ASSET_CLASS`, `STRATEGY_GROUP_ASSET_OVERRIDE`, `ASSET_CLASS_ORDER` |
| [`ensemble/vault/hierarchy_spec.py`](../../../ensemble/vault/hierarchy_spec.py) | `collect_streams_by_asset_and_style*`, `build_asset_first_hierarchy_spec*` |

```python
from ensemble.vault.hierarchy_spec import build_asset_first_hierarchy_spec_for_ensemble_dirs

spec = build_asset_first_hierarchy_spec_for_ensemble_dirs(
    repo_root,
    config.ensemble_dirs,
    strict_group=True,
    portfolio_ticker_names=frozenset(t.name for t in config.tickers),
)
WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=spec, fdm_max=2.0)
```

Feature-research portfolio admission rebuilds this spec whenever `ensemble_dirs` changes
(`research.feature.inclusion_gates.config_with_ensemble_dirs`).

### SR adjustment on `hierarchy_equal`

Optional Carver handcrafting SR multipliers (`sr_adjustment=True`) run after the equal splits
on `hierarchy_equal`, using training-window **lagged** `forecast × instrument_return` PnL
(prior-bar vol-scaled forecast, no same-day lookahead). See
[`docs/SaaS/weight_layer_spec.md`](../../SaaS/weight_layer_spec.md) §2.5.

```python
WeightLayer(
    weight_method="hierarchy_equal",
    hierarchy_spec=spec,
    fdm_max=2.0,
    sr_adjustment=True,
    sr_avg=0.5,
    sr_p_step=0.01,
    sr_min_years=5.0,
)
```

`GlobalPortfolio.fit()` passes per-ticker instrument returns to the weight layer when
`sr_adjustment` is enabled (otherwise it passes the aggregate daily return proxy).

Draft a hierarchy JSON from a vault root:

```bash
python scripts/emit_weight_hierarchy_json.py --stats -o hierarchy_draft.json
```

Use `--legacy-two-level` for the older root → vault-group → leaves layout.

## FDM

FDM uses the raw signal-level positive-clipped correlation matrix for **all** modes
(`_compute_fdm_from_corr_matrix` → `_correlation_multiplier_from_corr_matrix`, `epsilon = 0.01`):

```text
mean_corr = mean(off_diagonal(signal_corr_positive))
FDM       = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)
```

Single-model and fallback cases use `FDM = 1.0`. At `combine()` time the per-ticker
`forecast_score = (weighted_forecast × FDM)` is **clipped to the range `[-2.0, 2.0]`**.

## Config

`WeightLayerConfig` is a frozen dataclass. The factory `WeightLayer(...)` accepts
`weight_method` (mapped onto the config's `weighting_method`) plus keyword overrides:

| Parameter | Default | Description |
|---|---|---|
| `weighting_method` | `equal_signal` | One of the nine methods in `_WEIGHT_METHODS` |
| `fdm_max` | `2.0` | FDM cap (must be `> 0`) |
| `hierarchy_spec` | `None` | Nested dict for hierarchy methods (optional if `hierarchy_path` is set) |
| `hierarchy_path` | `None` | Filesystem path to a hierarchy JSON for hierarchy methods |
| `sr_adjustment` | `False` | Enable Carver SR handcrafting tilt on `hierarchy_equal` |
| `sr_avg` | `0.5` | SR-tilt average target |
| `sr_p_step` | `0.01` | SR-tilt probability step (must be in `(0, 1)`) |
| `sr_std` | `0.15` | SR-tilt standard deviation (must be `> 0`) |
| `sr_min_years` | `5.0` | Minimum PnL history (years) before SR tilt applies |
| `sr_tilt_max_depth` | `None` | Limit SR-tilt to N hierarchy levels (`>= 1`, or `None` = unlimited) |
| `within_group_method` | `equal` | `equal` or `inverse_avg_pairwise_corr` applied within each hierarchy group below the SR-tilt level |

Example:

```python
from ensemble.weight_layer import WeightLayer

WeightLayer(
    weight_method="hierarchy_equal",
    fdm_max=2.0,
    hierarchy_path="path/to/hierarchy.json",
)
```

## Serialization

`serialize_weight_layer_state` / `deserialize_weight_layer_state` round-trip a fitted (or
unfitted) `ClusteredWeightLayer` to a JSON-safe payload (config + fitted state). Deserializing
a payload whose `weighting_method` is in `_LEGACY_REMOVED_METHODS` raises.

## Diagnostics

`get_diagnostics()` returns, per ticker:

- `weights`
- `cluster_assignments` / `cluster_weights` / `cluster_metrics` (historical names; for
  non-clustered modes these summarize per-signal allocation groups)
- `fdm`
- `mean_signal_correlation`

plus a top-level `summary` (mean/min/max FDM, model and cluster counts) and the active
`weight_method` / `fdm_max`.

**See also:** [Portfolio pipeline](portfolio.md), [Multi-timeframe](multi_timeframe.md), [Base model](base_model.md)

> _Verified against current code via CodeGraph on 2026-06-07._
