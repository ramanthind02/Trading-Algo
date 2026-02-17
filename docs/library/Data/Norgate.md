# Norgate Data Reference

## Overview
Norgate Data is the external market data source for the migration and comparison workstreams. Access is provided via the `norgatedata` Python package, which reads locally-updated data maintained by the Norgate Data Updater (NDU). This reference captures the data structure, formats, identifiers, and constraints used for current futures data and future extensions.

## Operational requirements
- Windows host (NDU is Windows-only).
- Active Norgate Data subscription (current coverage: futures).
- NDU installed and running locally.
- Local writable data root at `.norgatedata` or via `NORGATEDATA_ROOT`.
- Python 3.5+ and `norgatedata` package dependencies (pandas, numpy, requests, logbook).
- The Python version note reflects vendor package compatibility; project runtime may be higher.
- Runtime health check: `norgatedata.status()` returns True when NDU is running.

## Time series formats and parameters
### Output formats
`norgatedata.price_timeseries(...)` supports:
- `numpy-recarray` (default)
- `numpy-ndarray`
- `pandas-dataframe`

### Date parameters
- `start_date`, `end_date`, `limit`, `interval`
- `interval` values: `D`, `W`, `M` (date returned is last date of interval)
- Date inputs accept string (`YYYY-MM-DD` or `YYYYMMDD`), `datetime`, pandas `Timestamp`, or numpy `datetime64`.

### Datetime formats
Defaults:
- NumPy recarray/ndarray: `datetime64` as `<M8[D]>` (timezone-naive)
- pandas DataFrame: `datetime64[ns]` (timezone-naive)

Overrides:
- `datetimeformat`: `datetime`, `date`, `datetime64ns`, `datetime64ms`, `m8d`
- `timezone`: any valid tz string (e.g., `UTC`, `US/Eastern`) while keeping time at 00:00:00

## Adjustment and padding settings
### Price adjustments
`StockPriceAdjustmentType`:
- `NONE`
- `CAPITAL`
- `CAPITALSPECIAL`
- `TOTALRETURN` (default)

### Padding
`PaddingType`:
- `NONE` (default)
- `ALLMARKETDAYS`
- `ALLWEEKDAYS`
- `ALLCALENDARDAYS`

Padding indicators are available via `padding_status_timeseries`.

### Dividend notes
Dividend values depend on adjustment settings:
- Capital-only adjustments show ordinary + special dividends.
- Capital + special adjustments show ordinary dividends.
- Total-return adjustments return no dividend values.
Dividends are reported for the day before the ex-date.

## Price and volume schema
Base columns:
- `Date`, `Open`, `High`, `Low`, `Close`

Additional columns where applicable:
- `Volume` (stocks, some indices, some indicators, futures)
- `Turnover` (stocks, some indices, some indicators)
- `Unadjusted Close` (stocks)
- `Dividend` (stocks)
- `Open Interest` (futures, some options)
- `Delivery Month` (continuous futures)

## Identifiers
- `assetid(symbol)` returns a stable, unchanging numeric id for a symbol.
- `symbol(assetid)` returns the current symbol for an asset id.
- Use `assetid` for storage and order history to avoid symbol changes over time.

## Informational time series and metadata
Time series indicators:
- Index constituents
- Major exchange listed
- Blank check company
- Capital event
- Dividend yield
- Unadjusted close
- Padding status

Single-value metadata:
- Domicile, currency, exchange name (short/long), security name
- Base type and subtype1/2/3
- Financial summary, business summary
- First/last/second-last quoted dates
- Shares outstanding / shares float
- Classification and classification-at-level (GICS/TRBC/etc.)
- Corresponding industry index

## Futures metadata
Key fields:
- Tick size
- Point value
- Margin
- First notice date
- Lowest ever tick size
- Session type
- Futures market name
- Futures market session name
- Futures market session symbol
- Futures market symbol
- Futures market session contracts
- Futures market symbols and session symbols

Date-related metadata (e.g., first notice date) is only available for individual futures contracts, not continuous symbols.

## Databases and watchlists
- `databases()` and `database_symbols()` provide subscription database contents.
- `watchlists()` and `watchlist_symbols()` expose NDU watchlists.

## Error handling and missing data
- Invalid symbols or parameters raise `ValueError`.
- Missing data returns `None` (or `(None, None)` for functions returning value + date).

## Migration implications
- Norgate continuous futures roll methodology is volume-based, while legacy data uses fixed-date roll rules. Roll dates and price levels are expected to diverge.
- Comparisons between legacy 1-minute series and Norgate daily series require resampling and timestamp alignment.
- Adjustment and padding settings must be captured alongside any extracted data to keep comparisons deterministic.
