# Weight Layer Spec: Asset-First Hierarchy (implemented)

> **Status:** This describes the **current** weight-layer encoding and the
> asset-first `hierarchy_equal` builder as they exist in code. An earlier version
> of this file was a migration plan ("strategy-first → asset-first"); that
> migration has shipped, so this doc now documents the implemented behaviour. The
> symbols below are verified against `ensemble/weight_layer.py`,
> `ensemble/weight_hierarchy.py`, `ensemble/vault/hierarchy_spec.py`, and
> `ensemble/vault/constants.py`.

## 1. How the weight layer operates

All strategies across all tickers, timeframes, and models are encoded into a single flat pool before the weight layer sees them. The encoding path is:

```
TFPortfolio.predict_base_model_vectors_from_candles()
  → encode_forecast_vectors_for_global_weight_layer()
  → ticker rewritten to __GLOBAL__, model_name = stream_id
  → ClusteredWeightLayer.fit() / .combine()      (via the WeightLayer(...) factory)
  → decode_global_weight_layer_output()
```

Every `stream_id` has the form `{ticker}::{timeframe}::{model_name}` (built by
`build_global_stream_id` in `ensemble/portfolio_impl/global_weight_layer_adapter.py`).
The weight layer treats these as model names on a single synthetic `__GLOBAL__`
ticker.

## 2. Grouping scheme (vault folders)

The vault is organised as `<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble>/`. The valid group names are `VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES` in `ensemble/vault/constants.py`:

```python
VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES = frozenset({
    "mean_reversion_indices",  # MR strategies — equity indices
    "buy_hold",                # long-biased hold positions
    "es_tlt",                  # ES leg vs TLT peer (trades ES only)
    "seasonal",                # calendar rules
    "momentum",                # momentum / trend, mixed asset class
    "trend_following",
    "momentum_gc",             # GC-specific momentum
    "crude_oil_mr",            # CL mean reversion
    "gc_breakout",             # GC Donchian / trend breakout
    "cl_breakout",             # CL breakout
    "breakout",
    "silver_mr",               # SI mean reversion
    "silver_trend",            # SI trend
})
```

A feature JSON carries its group as the `weight_hierarchy_group` field; if absent,
the group is inferred from the `vault/<TF>/<group>/...` path
(`infer_weight_hierarchy_group_from_feature_path`). These group names are the
**second level** (style) of the asset-first hierarchy.

## 3. Asset-first hierarchy structure

The active hierarchy is three levels: **asset class → strategy group (style) → individual stream leaves.**

```
root
├── equity_indices
│   ├── mean_reversion_indices   ← MR streams on ES/NQ
│   ├── momentum                 ← momentum streams on ES/NQ
│   ├── seasonal                 ← calendar rules on ES/NQ
│   ├── buy_hold                 ← per-ticker buy/hold on ES/NQ
│   └── es_tlt                   ← ES leg vs TLT peer (trades ES only)
├── commodities
│   ├── momentum / momentum_gc / gc_breakout / crude_oil_mr / cl_breakout / silver_* …
│   └── buy_hold                 ← e.g. GC::M buy_hold
├── fixed_income                 ← TLT/ZB/ZN strategies (when added)
└── diversified                  ← unknown tickers / multi-asset ensembles
```

Equal weight splits at every level (`compute_equal_split_weights`). A stream in
`equity_indices/mean_reversion_indices` gets
`1/n_asset_classes × 1/n_styles_in_that_asset × 1/n_streams_in_that_style`. Empty
asset classes and empty style groups are omitted by the builder.

### 3.1 Asset class → ticker mapping

`TICKER_ASSET_CLASS` in `ensemble/vault/constants.py` maps each known ticker to its asset class; unknown tickers fall into `"diversified"`:

```python
TICKER_ASSET_CLASS = {
    "ES": "equity_indices", "NQ": "equity_indices", "RTY": "equity_indices",
    "YM": "equity_indices", "DAX": "equity_indices",
    "GC": "commodities", "SI": "commodities", "CL": "commodities", "NG": "commodities",
    "ZB": "fixed_income", "ZN": "fixed_income", "TLT": "fixed_income",
    "EUR": "fx", "JPY": "fx", "GBP": "fx",
    "EU": "fx", "JY": "fx", "BP": "fx", "CD": "fx", "SF": "fx",
}
```

Asset-class display order is `ASSET_CLASS_ORDER = (equity_indices, commodities, fixed_income, fx, diversified)`.

### 3.2 Strategy group → asset override

`STRATEGY_GROUP_ASSET_OVERRIDE` is currently `{}` (empty). Every stream's asset
class is therefore derived from the ticker component of its `stream_id`
(`_asset_class_for_stream`). A group whose streams span asset classes (e.g.
`momentum` with ES, NQ, and GC) is split automatically: one feature file can
contribute streams to two different branches. `es_tlt`, `buy_hold`, and `seasonal`
all land under the traded instrument's asset class (e.g. ES rebalancing flow →
`equity_indices/es_tlt`).

## 4. Optional SR adjustment (`sr_adjustment`)

After equal sibling splits, an optional **Carver mini-bootstrap** step
(`ensemble/sr_adjustment.py`, wired via `_apply_sr_tilt_to_hierarchy_weights` in
`ensemble/weight_layer.py`) tilts budgets among siblings at every hierarchy level.
Parent-group mass is preserved; only internal splits change. `sr_tilt_max_depth`
caps how deep the tilt applies; `within_group_method="inverse_avg_pairwise_corr"`
optionally applies inverse-correlation weighting *within* groups below the
SR-tilt level.

**Fit data (training window only):** `GlobalPortfolio.fit()` passes daily
**instrument returns** (columns = normalized ticker keys) into
`ClusteredWeightLayer.fit(..., returns=...)` when `sr_adjustment=True`. Per-stream
PnL is **lagged** `forecast × ticker_return` (vol-scaled forecast from the prior
bar; avoids same-day lookahead). SR is annualized from daily PnL (`years = n_obs / 252`).
Groups with `years < sr_min_years` are left at equal split. Streams with no return
column for their instrument are skipped (logged).

```python
WeightLayerConfig(
    weighting_method="hierarchy_equal",
    hierarchy_spec=spec,
    fdm_max=2.0,
    sr_adjustment=True,
    sr_avg=0.5,
    sr_p_step=0.01,
    sr_std=0.15,
    sr_min_years=5.0,
    sr_tilt_max_depth=None,        # None = unlimited depth
    within_group_method="equal",   # or "inverse_avg_pairwise_corr"
)
```

The prop portfolio research config (`portfolio_research/config.py`) enables the
SR tilt by default through `weight_layer_kwargs`; portfolio research rebuilds the
`hierarchy_spec` from `ensemble_dirs`
(`portfolio_research.config.rebuild_weight_layer_kwargs`,
`describe_weight_layer_policy`).

## 5. Builder API (`ensemble/vault/hierarchy_spec.py`)

The asset-first spec is assembled by these public functions (all present today):

| Function | Purpose |
|---|---|
| `collect_streams_by_group_from_vault(vault_root, ...)` | `{ group: frozenset[stream_id] }` from validated vault feature JSONs. |
| `collect_streams_by_group_for_ensemble_dirs(repo_root, ensemble_dirs, ...)` | Same, scoped to explicit `ensemble_dirs` (portfolio-research path). |
| `collect_streams_by_asset_and_style(vault_root, ...)` | Re-buckets into `{ asset_class: { strategy_group: frozenset[stream_id] } }`. |
| `collect_streams_by_asset_and_style_for_ensemble_dirs(repo_root, ensemble_dirs, ...)` | `ensemble_dirs` variant. |
| `build_asset_first_hierarchy_spec(streams_by_asset_and_style, ...)` | 3-level `hierarchy_equal` JSON: root → asset → style → leaves. |
| `build_asset_first_hierarchy_spec_from_vault(vault_root, ...)` | Convenience: collect + build from a vault root. |
| `build_asset_first_hierarchy_spec_for_ensemble_dirs(repo_root, ensemble_dirs, ...)` | Convenience for the `ensemble_dirs` path. |
| `build_hierarchy_equal_spec(streams_by_group, ...)` | Legacy 2-level (root → group → leaves) builder; **kept** for backward compat. |
| `build_hierarchy_spec_from_vault(vault_root, ...)` | Returns `(2-level spec, streams_by_group)`. |

Produced spec shape:

```json
{
  "type": "group", "id": "root",
  "children": [
    {
      "type": "group", "id": "equity_indices",
      "children": [
        {
          "type": "group", "id": "mean_reversion_indices",
          "children": [{"type": "leaf", "stream_id": "ES::D::..."}]
        }
      ]
    }
  ]
}
```

## 6. Engine consumption (`ensemble/weight_hierarchy.py`)

The hierarchy engine is depth-agnostic — a 2-level, 3-level, or N-level spec is all valid:

| Component | Behaviour |
|---|---|
| `parse_hierarchy_spec(raw)` | Validates and parses the JSON into a nested `_Group`/`_ResolvedLeaf` tree. Leaves accept `stream_id` or `(ticker, timeframe, model_name)`. |
| `compute_equal_split_weights(root, available_models)` | Equal split among siblings at each level; **strict** coverage check (`validate_strict_stream_coverage`) — declared leaves must exactly match available streams. |
| `resolve_hierarchy_for_fit(hierarchy_spec=, hierarchy_path=)` | Resolves config into a root group; `hierarchy_spec` takes precedence over `hierarchy_path`. |

`hierarchy_equal` mode therefore requires the spec to enumerate every fitted
`stream_id` exactly once. Asset-class proportions follow from the equal-split tree
(e.g. with two non-empty asset classes, each gets 0.5 before within-asset splits).

## 7. Serialization / migration

A fitted `ClusteredWeightLayer` is serialized **by value**: the config dict
(including the nested `hierarchy_spec`) plus fitted weights/FDM/cluster state are
stored by `serialize_weight_layer_state`. Portfolios fitted with an older 2-level
spec continue to deserialize and reproduce the same weights — no automatic
migration is performed. Upgrading an existing portfolio to the asset-first
hierarchy means re-fitting the weight layer with a 3-level spec, which is a weight
layer config change subject to the holdout-contamination doctrine in
`docs/SaaS/weight_layer.md` §5.

> **Removed methods.** Deserializing a portfolio whose `weighting_method` is one
> of `hrp_cluster_equal`, `hrp_classic`, or `optimize_sortino_capped`
> (`_LEGACY_REMOVED_METHODS`) raises a `ValueError` instructing a re-fit with
> `equal_signal`, `inverse_avg_pairwise_corr`, or `hierarchy_equal`.

## 8. Tests

- `tests/unit-tests/ensemble/test_asset_first_hierarchy.py` — asset/style bucketing and 3-level spec building.
- `tests/unit-tests/ensemble/test_weight_hierarchy.py` — parse + equal-split + strict coverage.
- `tests/unit-tests/ensemble/test_sr_adjustment.py` — SR tilt over the hierarchy.
- `tests/integration/ensemble/test_asset_first_hierarchy_portfolio.py` — `GlobalPortfolio` fit with an asset-first spec + round-trip serialization.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
