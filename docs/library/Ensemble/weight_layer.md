# Weight Layer

`WeightLayer` in `ensemble/weight_layer.py` combines forecast streams at the **global** stage: it assigns weights across base-model streams (per synthetic global ticker) and applies the Forecast Diversification Multiplier (FDM).

`GlobalPortfolio` adapts multi-ticker/timeframe streams into a synthetic ticker (`__GLOBAL__`) and decodes weighted outputs back to real tickers after combine.

```text
Base Models -> DiversifiedEnsemble -> TFPortfolio -> GlobalPortfolio(weight_layer=WeightLayer(...))
```

## Modes

| Mode | Behavior |
|---|---|
| `equal_signal` | Equal weight across all signals |
| `inverse_avg_pairwise_corr` | Weight signals by inverse average positive pairwise correlation |
| `hierarchy_equal` | Manual nested tree: equal weight among siblings at each branch; leaf weights are the product of branch fractions (see `ensemble/weight_hierarchy.py`) |

`equal_signal` is the default.

HRP-based methods (`hrp_cluster_equal`, `hrp_classic`) and `optimize_sortino_capped` were removed from the active config surface. Saved snapshots that still reference legacy methods fail deserialization with an explicit error—reload from a current portfolio config or re-fit.

## Estimation stack (correlation and FDM)

For each ticker:

1. Build a date × model forecast matrix.
2. Drop all-empty rows and fill remaining gaps with `0.0`.
3. Standardize each signal by its full-sample in-sample standard deviation.
4. Fit Ledoit–Wolf covariance on the standardized matrix.
5. Derive correlation from that covariance.
6. Use the **positive-clipped** correlation matrix for FDM (and for `inverse_avg_pairwise_corr` scoring). `hierarchy_equal` uses the tree only for weights; correlation is still used for FDM diagnostics.

Distance and Ward linkage are **not** used for weighting in the current implementation.

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

### `hierarchy_equal`

Provide either:

- `hierarchy_spec`: nested JSON mapping (root is a `type: "group"` node with `children`), or
- `hierarchy_path`: path to a JSON file with the same shape.

Leaves may specify `stream_id` directly (`ticker::timeframe::model_name`) or `ticker`, `timeframe`, and `model_name`. The fit step validates that declared leaves match the available model names (strict coverage). See `parse_hierarchy_spec` and `compute_equal_split_weights` in `ensemble/weight_hierarchy.py`.

### Asset-first 3-level hierarchy (recommended for portfolios)

For multi-asset vaults, prefer a **3-level** tree so commodity and equity streams are not siblings under one strategy label:

```text
root → asset_class → strategy_group (vault weight_hierarchy_group) → stream leaves
```

Vault folder layout is unchanged; rebucketing uses the ticker in each `stream_id` (`buy_hold`, `seasonal`, and `es_tlt` included — e.g. ES rebalancing flow → `equity_indices/es_tlt`). See `docs/SaaS/weight_layer_spec.md` for rationale.

| Module | Role |
|---|---|
| `ensemble/vault/constants.py` | `TICKER_ASSET_CLASS`, `STRATEGY_GROUP_ASSET_OVERRIDE`, `ASSET_CLASS_ORDER` |
| `ensemble/vault/hierarchy_spec.py` | `collect_streams_by_asset_and_style*`, `build_asset_first_hierarchy_spec*` |

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

Feature-research portfolio admission rebuilds this spec whenever `ensemble_dirs` changes (`feature_research.inclusion_gates.config_with_ensemble_dirs`). Default `PortfolioSourceConfig.weight_layer_method` is `hierarchy_equal`.

### SR adjustment on `hierarchy_equal`

Optional Carver handcrafting SR multipliers (`sr_adjustment=True`) run after equal splits at **each sibling level** (asset class, style group, leaf streams), using training-window **lagged** `forecast × instrument_return` PnL (prior-bar vol-scaled forecast, no same-day lookahead). See `docs/SaaS/weight_layer_spec.md` §2.5.

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

`GlobalPortfolio.fit()` supplies per-ticker daily returns when SR adjustment is enabled.

Draft JSON from a vault root:

```bash
python scripts/emit_weight_hierarchy_json.py --stats -o hierarchy_draft.json
```

Use `--legacy-two-level` for the older root → vault-group → leaves layout.

## FDM

FDM uses the raw signal-level positive-clipped correlation matrix for **all** modes:

```text
mean_corr = mean(off_diagonal(signal_corr_positive))
FDM = min(sqrt(1 / (mean_corr + 0.01)), fdm_max)
```

Single-model and fallback cases use `FDM = 1.0`.

## Config

| Parameter | Default | Description |
|---|---|---|
| `weighting_method` | `equal_signal` | One of `equal_signal`, `inverse_avg_pairwise_corr`, `hierarchy_equal` |
| `fdm_max` | `2.0` | FDM cap |
| `hierarchy_spec` | `None` | Nested dict for `hierarchy_equal` (optional if `hierarchy_path` is set) |
| `hierarchy_path` | `None` | Filesystem path to hierarchy JSON for `hierarchy_equal` |

Example:

```python
from ensemble.weight_layer import WeightLayer

WeightLayer(
    weight_method="hierarchy_equal",
    fdm_max=2.0,
    hierarchy_path="path/to/hierarchy.json",
)
```

## Diagnostics

Per ticker, the layer reports:

- `weights`
- `cluster_assignments` / `cluster_weights` / `cluster_metrics` (historical names; for non-clustered modes these summarize per-signal allocation groups)
- `fdm`
- `mean_signal_correlation`
