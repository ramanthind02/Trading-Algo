# Calendar event data

Committed JSON calendars for the calendar-ensemble seasonal bias nodes.

## Files

| File | Description |
|------|-------------|
| `fomc_decision_dates.json` | FOMC policy decision days (D0 = last day of each meeting) |
| `nyse_holiday_events.json` | US equity holiday closures with trading-adjusted D0 |

## Regenerating

```bash
python -m scripts.scrape_fomc_dates --start-year 2005 --end-year 2027
python -m scripts.build_nyse_holiday_events --start-year 2005 --end-year 2027
```

FOMC dates are scraped from:

- [Historical materials by year](https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm) (`fomchistorical{year}.htm`, 2005–2020)
- [Meeting calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) (2021+)

NYSE holiday D0 values use closure rules in `utils/calendar/nyse_holidays.py` and ES daily sessions from `data/ohlc_data/`.

## Not for FOMC

Do **not** use `data/release_dates_101.txt` for FOMC meeting dates. That file is an ALFRED export of FOMC press-release *revision* timestamps, not the meeting schedule.
