# Norgate Data

Norgate Data is the external market data provider accessed via the `norgatedata` Python package, which reads locally-maintained data synced by the Norgate Data Updater (NDU).

## Requirements

> [!warning]
> Windows only. NDU must be running on a Windows host for extraction calls to succeed.

- Active Norgate subscription (futures coverage required)
- NDU installed and running locally
- Runtime health check: `norgatedata.status()` returns `True`
- Local data root at `.norgatedata` or env var `NORGATEDATA_ROOT`
- Python packages: `norgatedata`, `pandas`, `numpy`, `requests`, `logbook`

## Identifiers

- `assetid(symbol)` - stable numeric ID; preferred for persistent references
- `symbol(assetid)` - reverse lookup; symbols can change, asset IDs do not

## Price schema

Base columns: `Date`, `Open`, `High`, `Low`, `Close`

Additional columns:

- `Volume` - futures, stocks, some indices
- `Open Interest` - futures, some options
- `Delivery Month` - continuous futures only

## Key parameters for `price_timeseries(...)`

| Parameter | Values |
|---|---|
| `interval` | `D`, `W`, `M` |
| `start_date` / `end_date` | `YYYY-MM-DD`, `datetime`, `Timestamp`, `datetime64` |
| Output format | `numpy-recarray`, `numpy-ndarray`, `pandas-dataframe` |

## Futures adjustment context

Norgate continuous futures use volume-based rolls and vendor-defined back-adjustment behavior.

In this repository, Norgate adjusted continuous data is the canonical daily futures history anchor. IB live appends are reconciled into that anchor using append-only plus ratio splice rules — generalised by the multi-source reconciler.

See:

- [[multi_source_update_architecture]] — source priority + Norgate→IB handover
- [[data_platform_README]] — the consolidated data stack
- `docs/SaaS/data_source.md`

## Adapter

The Norgate adapter lives at `data_platform/providers/norgate/`. See
[`data_platform/providers/norgate/README.md`](../../../data_platform/providers/norgate/README.md)
for the full directory layout, schema, CLI usage, and post-subscription migration path.

| Task | Command |
|---|---|
| Full rebuild | `python -m data_platform.providers.norgate.rebuild` |
| Fetch continuous only | `python -m data_platform.providers.norgate.fetch_continuous` |
| Archive contracts | `python -m data_platform.providers.norgate.fetch_contracts` |
| Migrate to ohlc_data | `python -m data_platform.providers.norgate.migrate` |
| Fetch US stocks (full universe) | `python -m data_platform.providers.norgate.stocks --universe both --workers 8` |
| Fetch market series (indices/cash/forex) | `python -m data_platform.providers.norgate.market_series` |
| Seed instrument catalog | `python -m data_platform.providers.norgate.to_catalog` |

### Databases we capture

A US Stocks + Futures subscription exposes these databases; all are now mirrored locally:

| Database | Count | Local store |
|---|---|---|
| Continuous Futures | 224 | `ohlc_data/` (23 trading) + `norgate/archive/continuous/` |
| Futures (individual) | 27,201 | `norgate/archive/contracts/` |
| US Equities + Delisted | 14,223 + 20,988 | `stock_data/` (TR/CAP/UNADJ + membership) |
| US Indices | 1,615 | `norgate/market_series/us_indices/` |
| World Indices | 31 | `norgate/market_series/world_indices/` |
| Cash Commodities | 100 | `norgate/market_series/cash_commodities/` |
| Forex Spot | 57 | `norgate/market_series/forex_spot/` |
| Economic | 148 | `events/econ/` |

## Operational notes

- Invalid symbols/parameters raise `ValueError`
- Missing data typically returns `None`
- Access-denied errors indicate the symbol is outside your subscription tier
- For reproducible workflows, persist adjustment/padding assumptions with extracted outputs

## Related

- [[pipeline]]
- [[vault]]
