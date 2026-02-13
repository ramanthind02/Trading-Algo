# Sector Weight Allocation Specification

**Version**: 1.0.0  
**Date**: 2025-02-10  
**Status**: Specification (Design)  
**Scope**: Portfolio layer only; no changes to DiversifiedEnsemble or WeightLayer.

---

## Table of Contents

1. [Problem: Equal Weight vs Sector Allocation](#1-problem-equal-weight-vs-sector-allocation)
2. [Config File Format and Schema](#2-config-file-format-and-schema)
3. [Resolution Algorithm](#3-resolution-algorithm)
4. [Portfolio API](#4-portfolio-api)
5. [Full JSON Example and Resolved Weights](#5-full-json-example-and-resolved-weights)
6. [Validation and Missing Tickers](#6-validation-and-missing-tickers)
7. [Implementation Scope](#7-implementation-scope)

---

## 1. Problem: Equal Weight vs Sector Allocation

Currently, the Portfolio assigns risk across instruments by **equal weight**: each of N tickers receives weight `1/N`. For example, with tickers X, Y, Z, each gets 1/3 (33.33%). This is simple but does not allow asset-specific or sector-level allocation.

A **sector weight allocation** strategy allows the user to:

- Group tickers into sectors (e.g. equities, metals, commodities).
- Allocate a weight to each sector; within a sector, weights are distributed among its members (equally or via sub-weights).
- Support **nested** structures: e.g. equities → US equities, European equities, Japanese equities, each with their own weights and tickers.

If no sector config is provided, behavior remains **equal weight** (current behavior). The sector config is optional and supplied via a config file path on the Portfolio class.

---

## 2. Config File Format and Schema

The config file is JSON and defines a single **nested tree** of allocation nodes.

### Node structure

Each node (sector or leaf) has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `weight` | number | Yes | Relative weight among siblings. Must be > 0. Sibling weights are normalized to sum to 1.0 at each level. |
| `children` | array of nodes | Conditional | Required for a **branch** node (no `tickers`). |
| `tickers` | array of strings | Conditional | Required for a **leaf** node. List of ticker symbols that share this node’s allocation. |
| `ticker_weights` | object (string → number) | No | At a leaf, optional explicit weights per ticker (e.g. `{"ES": 0.5, "NQ": 0.3}`). Relative weights are normalized among themselves. If omitted, the node’s allocation is split **equally** among `tickers`. |
| `name` | string | No | Optional label for the node (e.g. for diagnostics or docs). |

Rules:

- A node is either a **branch** (`children` present) or a **leaf** (`tickers` present), not both.
- Every leaf must have at least one ticker.
- A ticker must not appear in more than one leaf (no duplicate tickers).
- The **root** is a single object (typically with `weight: 1.0` and `children`).

---

## 3. Resolution Algorithm

Resolution turns the tree into a flat `Dict[str, float]` (ticker → weight) that sums to 1.0.

1. **Normalize sibling weights** at each level: for each set of siblings, divide each `weight` by the sum of their `weight` values so that siblings sum to 1.0.
2. **Recursive walk**:
   - Start at the root with `parent_contribution = 1.0`.
   - For each child: `node_contribution = parent_contribution * normalized_child_weight`.
   - If the node has `children`, recurse with `parent_contribution = node_contribution`.
   - If the node has `tickers` (leaf):
     - If `ticker_weights` is present: distribute `node_contribution` according to `ticker_weights` (normalize `ticker_weights` so they sum to 1.0, then multiply each by `node_contribution`).
     - Else: give each ticker `node_contribution / len(tickers)`.
3. **Aggregate**: Collect one weight per ticker from all leaves (each ticker appears in exactly one leaf).
4. **Final normalization**: Scale all ticker weights so they sum to 1.0. This is the `instrument_weights` map.

---

## 4. Portfolio API

- **New optional parameter**: `sector_allocation_config_path: Optional[str] = None`.
- **When provided**: Portfolio loads the JSON file at construction time, runs the resolution algorithm above, and sets the resulting map as the effective instrument weights (as if `instrument_weights` had been passed with that dict). Existing code paths (`_apply_instrument_weights`, `_apply_risk_management_to_forecasts`) remain unchanged; they already accept a `instrument_weights` dict.
- **Precedence**:
  - If `sector_allocation_config_path` is set, it is used and any constructor `instrument_weights` argument is ignored (sector config wins).
  - If `sector_allocation_config_path` is not set and `instrument_weights` is provided, use `instrument_weights` (current behavior).
  - If neither is set, use equal weight: each ticker gets `1 / n_instruments` (current behavior).

No change to method signatures of `fit`, `predict`, `fit_from_candles`, or `predict_from_candles`; only the constructor gains an optional parameter and the source of `instrument_weights` when the config path is provided.

---

## 5. Full JSON Example and Resolved Weights

Example config: top-level sectors Equities (50%), Metals (30%), Commodities (20%). Equities are split into US (60%), European (30%), Japanese (10%). Leaves list tickers; no `ticker_weights`, so equal split within each leaf.

```json
{
  "weight": 1.0,
  "children": [
    {
      "name": "equities",
      "weight": 0.5,
      "children": [
        {
          "name": "us_equities",
          "weight": 0.6,
          "tickers": ["ES", "NQ", "YM", "RTY"]
        },
        {
          "name": "european_equities",
          "weight": 0.3,
          "tickers": ["DAX"]
        },
        {
          "name": "japanese_equities",
          "weight": 0.1,
          "tickers": ["N225"]
        }
      ]
    },
    {
      "name": "metals",
      "weight": 0.3,
      "tickers": ["GC", "SI"]
    },
    {
      "name": "commodities",
      "weight": 0.2,
      "tickers": ["CL"]
    }
  ]
}
```

**Resolved weights (before final normalization):**

- Equities 0.5 → US 0.6: 0.5 × 0.6 = 0.3 shared by ES, NQ, YM, RTY → **0.075** each.
- Equities 0.5 → European 0.3: 0.5 × 0.3 = **0.15** for DAX.
- Equities 0.5 → Japanese 0.1: 0.5 × 0.1 = **0.05** for N225.
- Metals 0.3 → GC, SI: **0.15** each.
- Commodities 0.2 → **0.2** for CL.

Sum = 0.075×4 + 0.15 + 0.05 + 0.15×2 + 0.2 = 0.3 + 0.15 + 0.05 + 0.3 + 0.2 = 1.0. So after final normalization the map is unchanged:

| Ticker | Weight |
|--------|--------|
| ES     | 0.075  |
| NQ     | 0.075  |
| YM     | 0.075  |
| RTY    | 0.075  |
| DAX    | 0.15   |
| N225   | 0.05   |
| GC     | 0.15   |
| SI     | 0.15   |
| CL     | 0.20   |

Example with **explicit ticker_weights** at a leaf (US equities):

```json
{
  "name": "us_equities",
  "weight": 0.6,
  "tickers": ["ES", "NQ", "YM"],
  "ticker_weights": { "ES": 0.5, "NQ": 0.3, "YM": 0.2 }
}
```

US slice = 0.5 × 0.6 = 0.3. Normalized ticker_weights 0.5, 0.3, 0.2 sum to 1.0, so ES = 0.15, NQ = 0.09, YM = 0.06.

---

## 6. Validation and Behavior for Missing Tickers

**Validation rules (recommended at load/resolution time):**

- All `weight` values must be > 0.
- Every leaf has at least one ticker.
- No ticker appears in more than one leaf.
- File must be valid JSON and conform to the schema (each node has either `children` or `tickers`).

**Tickers in forecasts but not in config:**

- Option A (recommended): Use the **existing Portfolio fallback**: instruments not in the config are assigned an equal share of the “remaining” weight (current behavior when `instrument_weights` is provided but a ticker is missing). This keeps backward compatibility and allows adding new tickers without failing.
- Option B: **Strict mode**: If any ticker in the data is not in the resolved map, raise an error (e.g. at first `predict` or `fit`). This can be a separate flag (e.g. `strict_sector_tickers: bool = False`) if implemented later.

The spec recommends Option A by default; Option B can be documented as an optional strict mode for implementation.

---

## 7. Implementation Scope

- **Portfolio only.** Resolution from sector tree → `Dict[str, float]` can be implemented in a small helper (e.g. in `ensemble/` or `utils/`) or inside the Portfolio class. The Portfolio constructor, when `sector_allocation_config_path` is set, should: load the JSON, resolve the tree to per-ticker weights, and set `self.instrument_weights` to that map. The raw config may optionally be stored for diagnostics (e.g. logging or a getter).
- **No changes** to DiversifiedEnsemble or WeightLayer. DiversifiedEnsemble continues to use explicit `instrument_weights` only when passed at fit time (e.g. by a caller that obtains them from Portfolio or elsewhere). Sector allocation is purely a Portfolio-layer feature.
- **Backward compatibility**: If neither `sector_allocation_config_path` nor `instrument_weights` is provided, behavior remains equal weight (1/N). If only `instrument_weights` is provided, current behavior is unchanged.

Implementation can follow this spec in a later change; this document defines the design only.
