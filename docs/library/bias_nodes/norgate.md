# Norgate Data

Norgate Data is the external market data provider accessed via the `norgatedata` Python package, which reads locally-maintained data synced by the Norgate Data Updater (NDU).

## Requirements

> [!warning] Windows only
> NDU is a Windows-only application. Data extraction must run on a Windows host.

- Active Norgate subscription (current coverage: futures)
- NDU installed and running locally
- Runtime health check: `norgatedata.status()` returns `True` when NDU is active
- Local data root at `.norgatedata` or env var `NORGATEDATA_ROOT`
- Python packages: `norgatedata`, `pandas`, `numpy`, `requests`, `logbook`

## Identifiers

- `assetid(symbol)` — stable numeric ID; **use this for storage**, not symbol strings
- `symbol(assetid)` — reverse lookup; symbols can change over time, asset IDs do not

## Price Schema

Base columns: `Date`, `Open`, `High`, `Low`, `Close`

Additional columns:
- `Volume` — futures, stocks, some indices
- `Open Interest` — futures, some options
- `Delivery Month` — continuous futures contracts only

## Key Parameters for `price_timeseries(...)`

| Parameter | Values |
|---|---|
| `interval` | `D`, `W`, `M` (date = last date of interval) |
| `start_date` / `end_date` | `YYYY-MM-DD`, `datetime`, `Timestamp`, `datetime64` |
| Output format | `numpy-recarray` (default), `numpy-ndarray`, `pandas-dataframe` |

## Adjustment Types (`StockPriceAdjustmentType`)

- `NONE` — raw prices
- `CAPITAL` — capital events only
- `CAPITALSPECIAL` — capital + special dividends
- `TOTALRETURN` (default) — full total return adjustment

## Padding Types (`PaddingType`)

- `NONE` (default)
- `ALLMARKETDAYS`, `ALLWEEKDAYS`, `ALLCALENDARDAYS`
- Use `padding_status_timeseries` to identify padded rows

## Futures-Specific Notes

- Continuous futures provide `Delivery Month` column
- First notice date and other date metadata are only available on **individual contracts**, not continuous symbols
- Key metadata fields: tick size, point value, margin, session type, futures market name

> [!important] Roll methodology divergence
> Norgate continuous futures use **volume-based** roll rules. Legacy data uses fixed-date rolls. Roll dates and price levels will diverge — account for this in any comparison workstream.

## Operational Notes

- Invalid symbols/params raise `ValueError`; missing data returns `None`
- Always record adjustment type and padding settings alongside extracted data to keep comparisons reproducible
- Comparing Norgate daily series to legacy 1-minute series requires resampling and timestamp alignment

## Related

- [[pipeline]] — feature extraction pipeline that consumes candle data
- [[vault]] — where validated features are stored after extraction
