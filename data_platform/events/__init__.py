"""Market-calendar and economic-event data: loaders, scrapers, parsers.

Two event families:
  - Calendar: NYSE holiday closures + FOMC decision dates (used by seasonal bias nodes).
  - Economic: ~148 Norgate macro series (CPI, NFP, GDP, yields...) under data/events/econ/.
"""

from data_platform.events.calendar_loader import (
    CalendarBundle,
    HolidayAssetBucket,
    HolidayEvent,
    load_calendar_bundle,
    repo_calendar_dir,
)
from data_platform.events.trading_day_index import TradingDayIndex
from data_platform.events.econ_releases import (
    econ_dir,
    load_econ_series,
    scrape as scrape_econ_series,
)

__all__ = [
    # calendar
    "CalendarBundle",
    "HolidayAssetBucket",
    "HolidayEvent",
    "TradingDayIndex",
    "load_calendar_bundle",
    "repo_calendar_dir",
    # economic releases
    "econ_dir",
    "load_econ_series",
    "scrape_econ_series",
]
