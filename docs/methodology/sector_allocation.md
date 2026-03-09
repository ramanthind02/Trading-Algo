# Sector allocation (instrument weights)

The Portfolio layer can apply ticker-level instrument weights from a **hierarchical sector allocation** JSON config. When `sector_allocation_config_path` is set, it overrides any explicit `instrument_weights` passed to `Portfolio`. The config is resolved into a normalized `ticker -> weight` map (weights sum to 1) and used when applying instrument weights to combined forecasts.

Implementation: [ensemble/portfolio.py](../api/ensemble.md) (`_load_sector_allocation_config`, `_validate_sector_allocation_node`, `_resolve_sector_allocation`).

---

## JSON schema

- **Root node**: Must have `weight` (positive number) and **exactly one** of:
  - `children`: list of child nodes (same shape, recursive)
  - `tickers`: list of non-empty strings (leaf)
- **Internal node**: `weight` (positive) + `children` (non-empty list of nodes).
- **Leaf node**: `weight` (positive) + `tickers` (non-empty list of ticker strings). Optional **`ticker_weights`**: object mapping each ticker in `tickers` to a positive number; if present, keys must equal `tickers` and values define relative weight within that leaf. If `ticker_weights` is omitted, tickers in the leaf split the node’s contribution equally.

Ticker keys in the config should match the identifiers used elsewhere (e.g. `Ticker` enum names: `ES`, `NQ`, `TLT`, `GC` for gold futures, etc.).

---

## Validation rules

- Every node’s `weight` must be a number &gt; 0.
- Each node has exactly one of `children` or `tickers` (not both, not neither).
- `children`, when present, must be a non-empty list of objects.
- Leaf `tickers` must be non-empty strings; no duplicate tickers within a leaf.
- No duplicate tickers across the entire tree.
- If `ticker_weights` is present on a leaf, its keys must equal `tickers` and all values must be &gt; 0.

On load, the Portfolio validates the tree then resolves it to a single `ticker -> weight` map and normalizes so the total is 1.

---

## Example: 60% stock indices / 20% TLT / 20% gold

60% to stock indices (NQ, YM, RTY) split equally; 20% TLT; 20% gold (GC):

```json
{
  "weight": 1.0,
  "children": [
    { "weight": 0.6, "tickers": ["NQ", "YM", "RTY"] },
    { "weight": 0.2, "tickers": ["TLT"] },
    { "weight": 0.2, "tickers": ["GC"] }
  ]
}
```

Resolved weights: NQ 20%, YM 20%, RTY 20%, TLT 20%, GC 20%.

Canonical file: `feature_research/config/sector_buy_hold_60_20_20.json`.

---

## Usage

- **Portfolio**: Pass the file path when constructing the portfolio:
  `Portfolio(sector_allocation_config_path="/path/to/sector.json", ...)`.
- **Config-driven**: `BaseResearchConfig` and `PortfolioResearchConfig` can hold `sector_allocation_config_path`; scripts that build `Portfolio` from config (e.g. `portfolio_research/run_portfolio_test.py`, `scripts/benchmark_portfolio_backtest.py`) pass it through when non-`None`.
