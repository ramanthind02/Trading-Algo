# Events — market calendar + economic releases

Calendar and macro-event data: holiday closures, FOMC decision dates, and the Norgate
Economic database. Consumed by the seasonal bias nodes (`nodes/seasonal/calendar/`).

## Modules

| Module | Purpose |
|---|---|
| `calendar_loader.py` | `load_calendar_bundle()` — reads holiday + FOMC JSON into a `CalendarBundle` |
| `nyse_holidays.py` | NYSE closure dates and D0 (last session before closure) |
| `fed_fomc.py` | Parse FOMC decision dates from Federal Reserve HTML |
| `trading_day_index.py` | `TradingDayIndex` — trading-day offsets relative to event D0 |
| `econ_releases.py` | Scrape the ~148-series Norgate Economic DB to parquet |
| `diagnostics.py` | Calendar-ensemble diagnostics (event PnL, window variants) |

## Data layout

```
data/events/
  calendar/
    nyse_holiday_events.json     # NYSE closures + D0
    fomc_decision_dates.json     # FOMC policy decision days
    README.md
  econ/
    {SAFE_SYMBOL}.parquet        # one per macro series (date32 index + value)
    _manifest.json               # symbol -> name + coverage
```

## Usage

```python
from data_platform.events import load_calendar_bundle, load_econ_series

bundle = load_calendar_bundle()          # FOMC + holiday events
cpi = load_econ_series("CPISA")          # one econ series by safe symbol
```

## Regenerating

```powershell
# NYSE holiday events (uses ES sessions from data/ohlc_data)
.\.venv\Scripts\python.exe -m scripts.build_nyse_holiday_events

# FOMC decision dates (scrapes Federal Reserve)
.\.venv\Scripts\python.exe -m scripts.scrape_fomc_dates

# Economic releases (~148 Norgate series — requires NDU running)
.\.venv\Scripts\python.exe -m data_platform.events.econ_releases
```

## Note

This was `utils/calendar/` before the data-platform consolidation. The economic-release
scraper and the `data/events/` data layout were added in the same pass.

> _Verified against current code via CodeGraph on 2026-06-07._
