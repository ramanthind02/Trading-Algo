"""Norgate data provider adapter.

Public surface — import these rather than reaching into sub-modules:

    from data_platform.providers.norgate import TICKER_TO_CCB, TICKER_TO_RAW
    from data_platform.providers.norgate import fetch_continuous, migrate_all, rebuild
"""
from ._constants import (
    TICKER_TO_CCB,
    TICKER_TO_CONTRACT_PREFIX,
    TICKER_TO_RAW,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
)
from ._paths import (
    norgate_root,
    working_adjusted_dir,
    working_unadjusted_dir,
    archive_continuous_dir,
    archive_contracts_dir,
    ohlc_root,
    ohlc_ticker_dir,
)
from .fetch_continuous import fetch_all as fetch_continuous, ensure_norgate_running
from .fetch_contracts import archive_continuous, archive_all_continuous, archive_contracts
from .fetch_specs import fetch_specs, load as load_contract_specs
from .migrate import migrate_all, migrate_ticker
from .rebuild import rebuild
from .stocks import (
    INDEX_MEMBERSHIP_WATCHLISTS,
    ScrapeSummary,
    StockAdjustment,
    StockFetchResult,
    Universe,
    fetch_symbol,
    list_symbols,
    list_watchlist_symbols,
    membership_path,
    price_path,
    read_raw_symbol,
    safe_symbol,
    scrape,
    scrape_parallel,
    stock_data_root,
    stock_dir,
    symbol_bucket,
)
from .market_series import (
    MarketCategory,
    market_series_root,
    scrape_category,
    series_path,
)
from .stocks_catalog import (
    build_stock_instrument,
    build_stock_instruments,
    seed_stock_catalog,
)

__all__ = [
    # constants
    "TICKER_TO_CCB",
    "TICKER_TO_RAW",
    "TICKER_TO_CONTRACT_PREFIX",
    "PARQUET_COMPRESSION",
    "PARQUET_COMPRESSION_LEVEL",
    # paths
    "norgate_root",
    "working_adjusted_dir",
    "working_unadjusted_dir",
    "archive_continuous_dir",
    "archive_contracts_dir",
    "ohlc_root",
    "ohlc_ticker_dir",
    # operations
    "ensure_norgate_running",
    "fetch_continuous",
    "archive_continuous",
    "archive_all_continuous",
    "archive_contracts",
    "fetch_specs",
    "load_contract_specs",
    "migrate_all",
    "migrate_ticker",
    "rebuild",
    # stocks: scraper
    "INDEX_MEMBERSHIP_WATCHLISTS",
    "ScrapeSummary",
    "StockAdjustment",
    "StockFetchResult",
    "Universe",
    "fetch_symbol",
    "list_symbols",
    "list_watchlist_symbols",
    "membership_path",
    "price_path",
    "read_raw_symbol",
    "safe_symbol",
    "scrape",
    "scrape_parallel",
    "stock_data_root",
    "stock_dir",
    "symbol_bucket",
    # market series: indices / cash / forex
    "MarketCategory",
    "market_series_root",
    "scrape_category",
    "series_path",
    # stocks: catalog
    "build_stock_instrument",
    "build_stock_instruments",
    "seed_stock_catalog",
]
