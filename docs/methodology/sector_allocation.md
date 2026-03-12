# Sector allocation (deprecated for active sizing flows)

Sector-allocation JSON is no longer part of active global/research portfolio sizing.

## What changed

- `GlobalPortfolio` no longer accepts `sector_allocation_config_path`.
- `PortfolioResearchConfig` and research pipeline wiring no longer forward sector JSON.
- Global diversification is now handled by adapter-encoding all streams into the existing
  `WeightLayer` (`__GLOBAL__` synthetic ticker), then decoding back to real tickers.

## What remains

- `TFPortfolio`/`Portfolio` still supports explicit `instrument_weights`.
- Legacy sector-allocation helpers may still exist for backward-compatible single-timeframe paths,
  but they are not used by the active global sizing orchestration.

## Migration guidance

1. Remove `sector_allocation_config_path` from global/research configs and constructor calls.
2. Use explicit `instrument_weights` only when you need static ticker-level tilts.
3. Use `GlobalPortfolio.get_diagnostics()["weight_layer"]["adapter_diagnostics"]`
   for ticker/timeframe rollups and stream decode metadata.
