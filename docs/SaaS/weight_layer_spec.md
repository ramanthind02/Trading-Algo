# Weight Layer Migration Spec: Strategy-First → Asset-First Hierarchy

## 1. Current State

### 1.1 How the weight layer operates

All strategies across all tickers, timeframes, and models are encoded into a single flat pool before the weight layer sees them. The encoding path is:

```
TFPortfolio.predict_base_model_vectors() 
  → encode_forecast_vectors_for_global_weight_layer()
  → ticker rewritten to __GLOBAL__, model_name = stream_id
  → ClusteredWeightLayer.fit() / .combine()
  → decode_global_weight_layer_output()
```

Every stream_id has the form `{ticker}::{timeframe}::{model_name}`. The weight layer treats these as model names on a single synthetic `__GLOBAL__` ticker.

### 1.2 Current grouping scheme

The vault is organised as `vault/<TF>/<weight_hierarchy_group>/<ensemble>/`. There are six known group names (from `ensemble/vault/constants.py`):

```python
VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES = {
    "mean_reversion_indices",  # MR strategies — equity indices
    "momentum",                # Momentum/trend — equity + commodities mixed
    "momentum_gc",             # GC-specific momentum
    "mean_reversion_gc",       # GC-specific mean reversion (commodities sleeve)
    "gc_breakout",             # GC Donchian / trend breakout (e.g. robust_trend_breakout)
    "buy_hold",                # Long-biased hold positions
    "es_tlt",                  # ES/TLT pair
    "seasonal",                # (unpopulated)
}
```

These map to a **2-level hierarchy** in the weight layer:

```
root
├── mean_reversion_indices  (ES, NQ streams)
├── momentum                (ES, NQ, GC streams — mixed asset class)
├── momentum_gc             (GC streams)
├── mean_reversion_gc       (GC MR streams — feature_research GC-only validation)
├── gc_breakout             (GC trend breakout — robust_trend_breakout)
└── buy_hold                (ES, GC, NQ, TLT streams)
```

### 1.3 What the current vault actually contains

| Group | Ensemble | Tickers | TF |
|---|---|---|---|
| `mean_reversion_indices` | `mr_indices_long` (turnaround tuesday) | ES, NQ | D |
| `mean_reversion_indices` | `mr_indices_long_long` (IBS lower band) | NQ | D |
| `momentum` | `sma_regime_long_short` | ES, NQ, GC | D |
| `momentum` | `algomatic_momentum` | NQ | D |
| `momentum` | `filter_sma200_consec_momentum` | ES, NQ | D |
| `momentum_gc` | `gc_sma_above_filter` | GC | D |
| `buy_hold` | `buy_hold_long` | ES, GC, NQ (+ TLT personal) | M |

### 1.4 The current grouping problem

`momentum` mixes equity index streams (ES, NQ) and commodity streams (GC) within a single group. The `sma_regime` ensemble covers all three. The weight layer treats ES::D::sma_regime and GC::D::sma_regime as siblings inside the same group, allocating equal budget to each. This conflates two different asset class exposures under one label.

### 1.5 What the engine already supports

`weight_hierarchy.py` `_walk_assign()` handles arbitrary nesting depth. A 3-level or N-level hierarchy JSON is already valid and works correctly. The gap is entirely in **how the hierarchy spec is built**, not in how it is consumed.

`hierarchy_equal` mode strictly validates that declared leaves match available streams — meaning the hierarchy spec must enumerate every stream_id exactly once.

---

## 2. Target State

### 2.1 Target hierarchy structure

Asset class at top level. Strategy group (style) below that. Individual streams as leaves.

```
root
├── equity_indices
│   ├── mean_reversion         ← all MR strategy streams on ES/NQ
│   ├── momentum               ← all momentum strategy streams on ES/NQ
│   ├── seasonal               ← calendar rules on ES/NQ (e.g. EOY SP500)
│   ├── buy_hold               ← per-ticker buy/hold on ES/NQ (e.g. ES::M)
│   └── es_tlt                 ← ES leg vs TLT peer (rebalancing flow; trades ES only)
├── commodities
│   ├── momentum               ← momentum strategy streams on GC/CL/etc.
│   ├── gc_breakout            ← GC trend breakout (Donchian + EMA + ATR chandelier)
│   └── buy_hold               ← per-ticker buy/hold (e.g. GC::M)
├── fixed_income               ← TLT, ZB, ZN strategies (when added)
│   ├── buy_hold               ← per-ticker buy/hold (e.g. TLT::M)
│   └── seasonal               ← bond calendar rules (e.g. TLT month-end)
└── diversified                ← unknown tickers / multi-asset ensembles only
    └── …
```

Equal weight splits at every level. A stream in `equity_indices/mean_reversion` gets weight `1/n_asset_classes × 1/n_style_groups_in_equity × 1/n_streams_in_that_style`.

### 2.2 Target hierarchy for the current vault

Based on the current vault contents:

```
root (1.0)
├── equity_indices (0.5)
│   ├── mean_reversion (0.25)
│   │   ├── ES::D::turnaround_tuesday (leaf)
│   │   ├── NQ::D::turnaround_tuesday (leaf)
│   │   └── NQ::D::ibs_lower_band (leaf)
│   └── momentum (0.25)
│       ├── ES::D::sma_regime (leaf)
│       ├── NQ::D::sma_regime (leaf)
│       ├── NQ::D::algomatic_momentum (leaf)
│       ├── ES::D::filter_consec_momentum (leaf)
│       └── NQ::D::filter_consec_momentum (leaf)
├── commodities (0.33…)
│   ├── momentum (…)
│   │   ├── GC::D::sma_regime (leaf)
│   │   └── …
│   └── buy_hold (…)
│       └── GC::M::buy_hold (leaf)
├── equity_indices (0.33…)
│   ├── mean_reversion (…)
│   ├── momentum (…)
│   └── buy_hold (…)
│       ├── ES::M::buy_hold (leaf)
│       └── NQ::M::buy_hold (leaf)
└── fixed_income (when present)
    └── buy_hold (…)
        └── TLT::M::buy_hold (leaf — personal vault only)
```

### 2.3 Asset class → ticker mapping

The mapping is hardcoded for the known universe. New tickers are added here when onboarded.

```python
TICKER_ASSET_CLASS: dict[str, str] = {
    "ES":  "equity_indices",
    "NQ":  "equity_indices",
    "RTY": "equity_indices",
    "YM":  "equity_indices",
    "DAX": "equity_indices",
    "GC":  "commodities",
    "SI":  "commodities",
    "CL":  "commodities",
    "NG":  "commodities",
    "ZB":  "fixed_income",
    "ZN":  "fixed_income",
    "TLT": "fixed_income",
    "EUR": "fx",
    "JPY": "fx",
    "GBP": "fx",
}
```

Tickers not in this map fall into a `"diversified"` asset class.

### 2.4 Strategy group → hierarchy slot mapping

The existing `weight_hierarchy_group` values remain unchanged as the second-level labels. Groups that contain streams from multiple asset classes (e.g., `momentum` with ES, NQ, GC) will have their streams split across asset class nodes automatically — one feature file can contribute streams to two different branches of the hierarchy.

```python
STRATEGY_GROUP_ASSET_OVERRIDE: dict[str, str] = {}
```

When empty, every stream uses ticker-based asset class inference. **`es_tlt`**, **`buy_hold`**, and **`seasonal`** therefore land under the traded instrument's asset class (`equity_indices/es_tlt` for ES rebalancing-flow vs TLT; `equity_indices/seasonal` for ES/NQ calendar rules; etc.).

### 2.5 Optional SR adjustment (`sr_adjustment`)

After equal sibling splits, an optional **Carver mini-bootstrap** step (`ensemble/sr_adjustment.py`) tilts budgets among siblings at **every hierarchy level** (root asset classes, style groups, and leaf streams). Parent-group mass is preserved; only internal splits change.

**Fit data (training window only):** `GlobalPortfolio.fit()` passes daily **instrument returns** (`columns` = tickers, normalized keys) into `WeightLayer.fit()` when `sr_adjustment=True`. Per-stream PnL is **lagged** `forecast × ticker_return` (vol-scaled forecast from the prior bar; avoids same-day lookahead). SR is annualized from daily PnL; `years = n_obs / 252`. Groups with `years < sr_min_years` are left at equal split. Streams with no return column for their instrument are skipped (logged).

```python
WeightLayerConfig(
    weighting_method="hierarchy_equal",
    hierarchy_spec=spec,
    fdm_max=2.0,
    sr_adjustment=True,
    sr_avg=0.5,
    sr_p_step=0.01,
    sr_min_years=5.0,
)
```

Feature-research portfolio admission enables SR tilt by default via `PortfolioSourceConfig.weight_layer_kwargs`. Portfolio research rebuilds `hierarchy_spec` from `ensemble_dirs` (`portfolio_research.config.rebuild_weight_layer_kwargs`).

---

## 3. Gap Analysis

### 3.1 Engine (no changes needed)

| Component | Status |
|---|---|
| `weight_hierarchy.py` `_walk_assign()` | Supports arbitrary nesting depth — no change |
| `weight_hierarchy.py` `compute_equal_split_weights()` | Works for 3-level trees — no change |
| `WeightLayerConfig` / `ClusteredWeightLayer` | Consumes any valid hierarchy spec — no change |
| `hierarchy_spec` JSON format (`type: group/leaf`) | Already supports nested groups — no change |
| Serialization / deserialization | Persists the config dict, including nested specs — no change |

### 3.2 Vault structure (no changes needed)

Vault folder layout (`vault/<TF>/<weight_hierarchy_group>/<ensemble>`) does not change. The strategy group names (`mean_reversion_indices`, `momentum`, etc.) remain as-is. No feature JSON renames or moves.

### 3.3 What needs to be added

**A. Ticker-to-asset-class mapping** — `ensemble/vault/constants.py`
- `TICKER_ASSET_CLASS: dict[str, str]` (§2.3 above)
- `STRATEGY_GROUP_ASSET_OVERRIDE: dict[str, str]` (§2.4 above)

**B. 3-level hierarchy builder** — `ensemble/vault/hierarchy_spec.py`
- New function: `collect_streams_by_asset_and_style()`  
  Returns `dict[asset_class, dict[strategy_group, frozenset[stream_id]]]`  
  Derives asset class per stream_id from the ticker component of the stream_id
- New function: `build_asset_first_hierarchy_spec()`  
  Consumes output of `collect_streams_by_asset_and_style()` and produces a valid 3-level `hierarchy_equal` spec JSON

**C. Portfolio-level wiring** — `GlobalPortfolio` construction
- The caller constructs the hierarchy spec via `build_asset_first_hierarchy_spec()` and passes it to `WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=spec)`
- No changes to `GlobalPortfolio` itself; only how the config is assembled changes

**D. Existing `build_hierarchy_equal_spec()` / `build_hierarchy_spec_from_vault()`**  
Kept unchanged for backward compat. The new 3-level builder is additive.

---

## 4. Implementation Tasks

### T1 — Add constants to `ensemble/vault/constants.py`
- `TICKER_ASSET_CLASS: dict[str, str]`
- `STRATEGY_GROUP_ASSET_OVERRIDE: dict[str, str]`

No functional change; sets up the lookup tables for T2.

### T2 — Add `collect_streams_by_asset_and_style()` to `ensemble/vault/hierarchy_spec.py`

```python
def collect_streams_by_asset_and_style(
    vault_root: str | Path,
    *,
    strict_group: bool = False,
    portfolio_ticker_names: frozenset[str] | None = None,
) -> dict[str, dict[str, frozenset[str]]]:
    """
    Returns { asset_class: { strategy_group: frozenset[stream_id] } }
    
    Asset class is derived from the ticker component of each stream_id,
    with ticker-derived asset classes for all strategy groups (including es_tlt on ES).
    Tickers not in TICKER_ASSET_CLASS fall into "diversified".
    """
```

Calls existing `collect_streams_by_group_from_vault()` to get the flat `{group: frozenset[stream_id]}` view, then re-buckets each stream_id into the 2D structure by parsing the ticker out of the stream_id and looking up `TICKER_ASSET_CLASS`.

An equivalent `collect_streams_by_asset_and_style_for_ensemble_dirs()` variant covers the `ensemble_dirs` path for portfolio research configs.

### T3 — Add `build_asset_first_hierarchy_spec()` to `ensemble/vault/hierarchy_spec.py`

```python
def build_asset_first_hierarchy_spec(
    streams_by_asset_and_style: Mapping[str, Mapping[str, Collection[str]]],
    *,
    root_id: str = "root",
    asset_order: Sequence[str] | None = None,
    style_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    3-level hierarchy spec: root → asset_class → strategy_group → stream leaves.
    Empty asset classes and style groups are omitted.
    """
```

Produces:
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

### T4 — Convenience wrapper in `ensemble/vault/hierarchy_spec.py`

```python
def build_asset_first_hierarchy_spec_from_vault(
    vault_root: str | Path,
    *,
    strict_group: bool = False,
    root_id: str = "root",
    portfolio_ticker_names: frozenset[str] | None = None,
) -> dict[str, object]:
    """Single-call convenience: collect + build from vault root."""
```

And the `ensemble_dirs` variant:
```python
def build_asset_first_hierarchy_spec_for_ensemble_dirs(
    repo_root: str | Path,
    ensemble_dirs: Mapping[str, str],
    ...
) -> dict[str, object]:
```

### T5 — Unit tests

File: `tests/unit-tests/ensemble/test_asset_first_hierarchy.py`

- `TICKER_ASSET_CLASS` covers all tickers currently in vaults
- `collect_streams_by_asset_and_style()` correctly splits ES/NQ/GC streams from a mixed `momentum` group
- `build_asset_first_hierarchy_spec()` produces valid 3-level JSON parseable by `parse_hierarchy_spec()`
- `compute_equal_split_weights()` on the 3-level spec gives correct proportional weights (equity 0.5, commodities 0.25, diversified 0.25 for the current vault)
- `validate_strict_stream_coverage()` passes when all streams from the vault are present in the built spec

### T6 — Integration smoke test

File: `tests/integration/ensemble/test_asset_first_hierarchy_portfolio.py`

- Build a `GlobalPortfolio` with `WeightLayer(weight_method="hierarchy_equal", hierarchy_spec=build_asset_first_hierarchy_spec_from_vault(vault_root, ...))` 
- Fit on IS data; assert fitted weights respect asset-class budget proportions (equity streams get ~50% combined)
- Round-trip serialize/deserialize the portfolio and assert weights are unchanged

### T7 — Update `docs/library/Ensemble/weight_layer.md`

Document the 3 new public functions (T3/T4) and the 3-level hierarchy spec format. Note that `hierarchy_equal` mode is the recommended default for asset-first portfolios.

---

## 5. Out of Scope

- Vault folder restructure: not required; the 3-level hierarchy is built from stream_ids, not folder paths
- Changing `weight_hierarchy_group` tags in existing feature JSONs: not required
- Removing `build_hierarchy_equal_spec` / `build_hierarchy_spec_from_vault`: kept for backward compat
- New weighting methods (inverse-corr at each level, Sharpe-ratio tilt): deferred; `hierarchy_equal` (equal at every level) is the implementation target for this migration

---

## 6. Migration for Existing Portfolios

Portfolios fitted with the old 2-level `hierarchy_equal` spec are serialized by value — the config dict is stored, not a reference to the vault. They continue to deserialize and produce correct results with the old 2-level spec. No automatic migration is needed.

To upgrade an existing portfolio to the asset-first hierarchy, the researcher must:
1. Re-fit the weight layer using the new 3-level spec
2. Treat this as a weight layer config change (subject to holdout contamination rules in `docs/SaaS/weight_layer.md` §5)
