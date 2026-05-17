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

In this repository, Norgate adjusted continuous data is the canonical daily futures history anchor. IB live appends are reconciled into that anchor using append-only plus ratio splice rules.

See:

- `docs/library/Data/canonical_data_architecture.md`
- `docs/library/Data/NORGATE_MIGRATION.md`
- `docs/SaaS/data_source.md`

## Operational notes

- Invalid symbols/parameters raise `ValueError`
- Missing data typically returns `None`
- For reproducible workflows, persist adjustment/padding assumptions with extracted outputs

## Related

- [[pipeline]]
- [[vault]]
