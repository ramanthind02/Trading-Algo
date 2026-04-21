#!/usr/bin/env python3
"""
TWS Live Forecast Pipeline
===========================

Connects to TWS API, fetches historical data, generates multi-timeframe
forecasts using GlobalPortfolio (auto-loaded from vault), and converts
to ETF positions with optional Telegram notification.

Usage:
    # Paper trading (default)
    python scripts/tws_live_forecast.py

    # Live trading
    python scripts/tws_live_forecast.py --port 7496

    # With capital override
    python scripts/tws_live_forecast.py --capital 5000

    # Dry run (no Telegram)
    python scripts/tws_live_forecast.py --dry-run
"""

import sys
import os
import json
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

# Configure logging - suppress IB API verbose logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
# Suppress ibapi debug logs
logging.getLogger('ibapi').setLevel(logging.WARNING)

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

# IB API imports
from ibapi.contract import Contract

# Project imports
from scripts.demo_ib_data_fetch import IBDataClient, IBConfig
from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.portfolio import TFPortfolio, GlobalPortfolio
from deployment.telegram_notifier import TelegramNotifier
from utils.core.enums import TimeFrame, Ticker
from utils.vault_paths import resolve_vault_root


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG_PATH = "configs/live_forecast_config.json"
DEFAULT_CONFIG_PATH_PROP = "configs/live_forecast_config_prop.json"
DEFAULT_CONFIG_PATH_PERSONAL = "configs/live_forecast_config_personal.json"


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def create_futures_contract(symbol: str, exchange: str) -> Contract:
    """Create a continuous futures contract for data fetching."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "CONTFUT"  # Continuous futures
    contract.exchange = exchange
    contract.currency = "USD"
    return contract


def create_etf_contract(symbol: str) -> Contract:
    """Create a stock/ETF contract for price fetching."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def _create_contract_from_config(ticker: str, instrument_info: Dict) -> Contract:
    """Create an IB contract based on instrument config sec_type."""
    sec_type = instrument_info.get("sec_type", "CONTFUT")
    if sec_type == "STK":
        return create_etf_contract(ticker)
    else:
        exchange = instrument_info.get("exchange", "CME")
        return create_futures_contract(ticker, exchange)


def fetch_historical_candles(
    client: IBDataClient,
    ticker: str,
    instrument_info: Dict,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Fetch historical bars and convert to candles DataFrame format.

    Parameters
    ----------
    client : IBDataClient
        Connected IB client
    ticker : str
        Ticker symbol (ES, NQ, YM, RTY, GC, TLT, etc.)
    instrument_info : dict
        Instrument config with keys: sec_type, exchange, etf
    lookback_days : int
        Number of days of history to fetch

    Returns
    -------
    pd.DataFrame
        Candles with columns: datetime, open, high, low, close, volume, ticker, timeframe
    """
    contract = _create_contract_from_config(ticker, instrument_info)

    # Request historical data
    req_id = client.request_historical_data(
        contract,
        end_date_time="",  # Current time
        duration=f"{lookback_days} D",
        bar_size="1 day",
        what_to_show="TRADES",
        use_rth=1,
        format_date=1,
        keep_up_to_date=False
    )

    # Wait for data
    success = client.wait_for_historical_data(req_id, timeout=30.0)

    # Check if we got bars (even if wait returned false due to late errors)
    if not client.historical_bars:
        if not success:
            print(f"  Warning: Timeout or error fetching data for {ticker}")
        else:
            print(f"  Warning: No bars received for {ticker}")
        return pd.DataFrame()

    # Convert bars to DataFrame
    rows = []
    for bar in client.historical_bars:
        dt = pd.to_datetime(bar.date)
        rows.append({
            "datetime": dt,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(bar.volume),
            "ticker": ticker,
            "timeframe": TimeFrame.D
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"  {ticker}: Fetched {len(df)} bars, last date: {df['datetime'].iloc[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'}")

    return df


def fetch_partial_daily_candle(
    client: IBDataClient,
    ticker: str,
    instrument_info: Dict,
    intraday_bar_size: str = "15 mins",
) -> pd.DataFrame:
    """Build a synthetic *partial* daily candle for today from intraday bars.

    Used by the personal-account profile which runs before the official daily
    candle closes. Fetches intraday bars for today only, then aggregates them
    into a single OHLCV row dated today. O=first, H=max, L=min, C=latest, V=sum.

    Returns a 1-row DataFrame shaped like :func:`fetch_historical_candles`
    output, or an empty frame if no intraday bars are returned.
    """
    contract = _create_contract_from_config(ticker, instrument_info)

    # Ask for "1 D" of intraday bars; IB returns today's session so far when
    # end_date_time is empty.
    req_id = client.request_historical_data(
        contract,
        end_date_time="",
        duration="1 D",
        bar_size=intraday_bar_size,
        what_to_show="TRADES",
        use_rth=1,
        format_date=1,
        keep_up_to_date=False,
    )
    success = client.wait_for_historical_data(req_id, timeout=30.0)

    if not client.historical_bars:
        if not success:
            print(f"  Warning: Timeout or error fetching intraday bars for {ticker}")
        else:
            print(f"  Warning: No intraday bars for {ticker}")
        return pd.DataFrame()

    rows = []
    for bar in client.historical_bars:
        rows.append({
            "dt": pd.to_datetime(bar.date),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(bar.volume),
        })
    bars_df = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
    if bars_df.empty:
        return pd.DataFrame()

    # Aggregate into one synthetic daily bar dated today (date of latest bar).
    latest_date = bars_df["dt"].iloc[-1].normalize()
    synthetic = pd.DataFrame([{
        "datetime": latest_date,
        "open": bars_df["open"].iloc[0],
        "high": bars_df["high"].max(),
        "low": bars_df["low"].min(),
        "close": bars_df["close"].iloc[-1],
        "volume": int(bars_df["volume"].sum()),
        "ticker": ticker,
        "timeframe": TimeFrame.D,
    }])
    print(
        f"  {ticker}: synthesized partial daily from {len(bars_df)} × {intraday_bar_size} bars "
        f"(O {synthetic['open'].iloc[0]:.2f}, H {synthetic['high'].iloc[0]:.2f}, "
        f"L {synthetic['low'].iloc[0]:.2f}, C {synthetic['close'].iloc[0]:.2f})"
    )
    return synthetic


def resample_daily_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily candles to monthly OHLCV candles, per ticker."""
    results = []
    for ticker, group in daily_df.groupby("ticker"):
        g = group.set_index("datetime").sort_index()
        monthly = g.resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["close"])
        monthly["ticker"] = ticker
        monthly["timeframe"] = TimeFrame.M
        monthly = monthly.reset_index()
        results.append(monthly)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def fetch_current_prices(
    client: IBDataClient,
    etf_symbols: List[str]
) -> Dict[str, float]:
    """
    Fetch snapshot prices for ETFs.

    Parameters
    ----------
    client : IBDataClient
        Connected IB client
    etf_symbols : List[str]
        List of ETF symbols (SPY, QQQ, DIA, IWM, GLD, TLT)

    Returns
    -------
    Dict[str, float]
        Mapping from symbol to current price
    """
    prices = {}

    for symbol in etf_symbols:
        contract = create_etf_contract(symbol)

        req_id = client.request_historical_data(
            contract,
            end_date_time="",
            duration="1 D",
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=1,
            format_date=1,
            keep_up_to_date=False
        )

        if client.wait_for_historical_data(req_id, timeout=15.0):
            if client.historical_bars:
                prices[symbol] = client.historical_bars[-1].close
                print(f"  {symbol}: ${prices[symbol]:.2f}")
            else:
                print(f"  Warning: No price data for {symbol}")
        else:
            print(f"  Warning: Timeout fetching price for {symbol}")

    return prices


# ==============================================================================
# PORTFOLIO LOADING
# ==============================================================================

def _load_ensembles_for_tf(vault_root: str, tf: TimeFrame) -> list:
    """Load all ensembles from ``vault_root/{tf.name}/``.

    Supports both the preferred nested layout
    (``vault_root/{tf}/{weight_group}/{ensemble}/``) and the legacy flat
    layout (``vault_root/{tf}/{ensemble}/``).
    """
    from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    tf_dir = Path(vault_root) / tf.name
    if not tf_dir.exists():
        return []

    ensembles = []
    for child in sorted(tf_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
            # Nested: iterate ensembles under this weight group.
            candidates = sorted(d for d in child.iterdir() if d.is_dir())
        else:
            # Legacy flat: ``child`` is the ensemble leaf itself.
            candidates = [child]
        for ens_dir in candidates:
            try:
                ens = load_ensemble_from_vault(str(ens_dir))
                ensembles.append(ens)
                print(f"    Loaded: {ens_dir.name}")
            except Exception as e:
                print(f"    Warning: Skipping {ens_dir.name}: {e}")
    return ensembles


def build_portfolio(config: Dict) -> GlobalPortfolio:
    """
    Build a GlobalPortfolio with auto-loaded TFPortfolio instances from vault.

    Parameters
    ----------
    config : dict
        Loaded config with portfolio.vault_root, target_volatility, etc.

    Returns
    -------
    GlobalPortfolio
        Multi-timeframe portfolio with all vault ensembles
    """
    vault_root = str(resolve_vault_root(config["portfolio"]["vault_root"]))
    target_vol = config["portfolio"]["target_volatility"]
    max_pos = config["portfolio"]["max_position_pct"]
    idm_max = config["portfolio"]["idm_max"]

    tf_portfolios = []
    for tf in [TimeFrame.D, TimeFrame.M]:
        print(f"  Loading {tf.name} ensembles...")
        ensembles = _load_ensembles_for_tf(vault_root, tf)
        if not ensembles:
            print(f"    No ensembles found for {tf.name}, skipping")
            continue

        tf_p = TFPortfolio(
            ensembles=ensembles,
            trading_timeframe=tf,
            target_volatility=target_vol,
            max_position_pct=max_pos,
            idm_max=idm_max,
        )
        tf_portfolios.append(tf_p)
        print(f"    {tf.name}: {len(ensembles)} ensemble(s) loaded")

    if not tf_portfolios:
        raise ValueError(f"No ensembles found in vault root: {vault_root}")

    portfolio = GlobalPortfolio(
        tf_portfolios=tf_portfolios,
        max_position_pct=max_pos,
        idm_max=idm_max,
    )
    total_ensembles = sum(len(tp.ensembles) for tp in tf_portfolios)
    print(f"  GlobalPortfolio created: {len(tf_portfolios)} timeframe(s), {total_ensembles} total ensemble(s)")
    return portfolio


def discover_required_tickers(portfolio: GlobalPortfolio) -> Set[str]:
    """Discover all tickers needed by the portfolio (primary + cross-tickers)."""
    from utils.data.cross_ticker_store import extract_cross_ticker_names

    required: Set[str] = set()
    for tf_p in portfolio.tf_portfolios:
        for ens in tf_p.ensembles:
            # Primary tickers from fitted ensemble state
            if ens.unique_tickers_:
                for t in ens.unique_tickers_:
                    required.add(t if isinstance(t, str) else str(t))
            # Cross-tickers from bias node specs
            for spec in ens.get_required_bias_nodes():
                required.update(extract_cross_ticker_names(spec.get('params', {})))

    return required


def ensure_cache_ready(required_tickers: Set[str]) -> Dict[str, Any]:
    """Bootstrap central cache from repo parquets if not already populated.

    Returns dict with 'bootstrapped' bool and coverage info per ticker.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.runtime.cache_manager import CacheManager

    store = CentralCacheStore.get_instance()
    manager = CacheManager()
    coverage_info = {}
    needs_bootstrap = []

    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            needs_bootstrap.append(ticker_enum)
        else:
            coverage_info[ticker_str] = {
                "start": record.coverage.start,
                "end": record.coverage.end,
                "revision": record.revision,
            }

    bootstrapped = False
    if needs_bootstrap:
        print(f"  Bootstrapping {len(needs_bootstrap)} ticker(s) from repo parquets...")
        result = manager.bootstrap_source_candles(
            tickers=[Ticker[t] for t in sorted(required_tickers)],
            timeframes=[TimeFrame.D, TimeFrame.M],
        )
        bootstrapped = True
        print(f"  Bootstrap: {result['success']} success, {result['failed']} failed")
        for ticker_str in sorted(required_tickers):
            try:
                ticker_enum = Ticker[ticker_str]
            except KeyError:
                continue
            record = store.describe_candle(ticker_enum, TimeFrame.D)
            if record:
                coverage_info[ticker_str] = {
                    "start": record.coverage.start,
                    "end": record.coverage.end,
                    "revision": record.revision,
                }

    return {"bootstrapped": bootstrapped, "coverage": coverage_info}


def upsert_tws_candles(
    daily_candles: pd.DataFrame,
    required_tickers: Set[str],
) -> None:
    """Upsert fetched TWS daily bars into central cache and resample to monthly.

    Parameters
    ----------
    daily_candles : pd.DataFrame
        All fetched daily candles with 'ticker' column.
    required_tickers : Set[str]
        Ticker names to upsert.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()

    for ticker_str in sorted(required_tickers):
        ticker_mask = daily_candles["ticker"].astype(str) == ticker_str
        ticker_candles = daily_candles.loc[ticker_mask].copy()
        if ticker_candles.empty:
            continue
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        store.upsert_candles(ticker_enum, TimeFrame.D, ticker_candles)
        print(f"    {ticker_str} D: upserted {len(ticker_candles)} bars")

    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            continue
        full_daily = store.query_candles(
            ticker_enum, TimeFrame.D,
            start=record.coverage.start,
            end=record.coverage.end,
        ).reset_index()
        if full_daily.empty:
            continue
        full_daily["ticker"] = ticker_str
        monthly = resample_daily_to_monthly(full_daily)
        if not monthly.empty:
            store.upsert_candles(ticker_enum, TimeFrame.M, monthly)


def refresh_bias_caches(
    vault_root: str,
    required_tickers: Set[str],
) -> Dict[str, Any]:
    """Refresh stale bias node artifacts for all vault ensembles.

    Returns the summary dict from ensure_vault_cache_coverage.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.runtime.cache_manager import CacheManager

    store = CentralCacheStore.get_instance()
    manager = CacheManager()

    earliest_start = None
    latest_end = None
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record is None:
            continue
        if record.coverage.start and (earliest_start is None or record.coverage.start < earliest_start):
            earliest_start = record.coverage.start
        if record.coverage.end and (latest_end is None or record.coverage.end > latest_end):
            latest_end = record.coverage.end

    if earliest_start is None or latest_end is None:
        raise ValueError("No candle coverage found in cache. Run bootstrap first.")

    # Discover ensemble directories, supporting both the nested weight-hierarchy
    # layout (vault/<TF>/<weight_group>/<ensemble>/) and the legacy flat layout
    # (vault/<TF>/<ensemble>/).
    from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    vault_dirs: List[str] = []
    for tf_name in ["D", "M"]:
        tf_dir = Path(vault_root) / tf_name
        if not tf_dir.exists():
            continue
        for child in sorted(tf_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
                for ens_dir in sorted(child.iterdir()):
                    if ens_dir.is_dir():
                        vault_dirs.append(str(ens_dir))
            else:
                vault_dirs.append(str(child))

    if not vault_dirs:
        raise ValueError(f"No vault ensembles found in {vault_root}")

    print(f"  Refreshing bias caches for {len(vault_dirs)} ensembles...")
    print(f"  Coverage window: {earliest_start.date()} to {latest_end.date()}")

    summary = manager.ensure_vault_cache_coverage(
        vault_ensemble_dirs=vault_dirs,
        start_date=earliest_start,
        end_date=latest_end,
        refresh_mode="missing_stale_only",
    )

    rebuilt = summary.get("rebuilt", 0)
    failed = summary.get("failed", 0)
    validated = summary.get("validated", 0)
    print(f"  Result: {rebuilt} rebuilt, {validated} already fresh, {failed} failed")

    if failed > 0:
        for d in summary.get("details", []):
            if d.get("status") == "failed":
                print(f"    FAILED: {d.get('module_name')}/{d.get('ticker')}: {d.get('message','')}")

    return summary


def build_cache_query(
    required_tickers: Set[str],
) -> tuple:
    """Build PortfolioCacheQuery and instrument_returns from cached candles.

    Returns (query, instrument_returns) tuple.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore
    from utils.cache.runtime.central_cache_models import ArtifactScope
    from ensemble.portfolio import PortfolioCacheQuery

    store = CentralCacheStore.get_instance()

    starts = []
    ends = []
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        record = store.describe_candle(ticker_enum, TimeFrame.D)
        if record and record.coverage.start and record.coverage.end:
            starts.append(record.coverage.start)
            ends.append(record.coverage.end)

    if not starts:
        raise ValueError("No candle coverage in cache")

    query_start = max(starts)
    query_end = min(ends)

    query = PortfolioCacheQuery(
        tickers=tuple(sorted(required_tickers)),
        start=query_start,
        end=query_end,
        timeframes=(TimeFrame.D, TimeFrame.M),
        scope=ArtifactScope.LIVE,
    )

    frames = []
    for ticker_str in sorted(required_tickers):
        try:
            ticker_enum = Ticker[ticker_str]
        except KeyError:
            continue
        candles = store.query_candles(
            ticker_enum, TimeFrame.D,
            start=query_start, end=query_end,
        ).reset_index()
        if not candles.empty:
            candles = candles.set_index("datetime")["close"].rename(ticker_str)
            frames.append(candles)

    if not frames:
        raise ValueError("No daily candle data in cache for returns computation")

    prices = pd.concat(frames, axis=1).sort_index()
    instrument_returns = prices.pct_change(fill_method=None).dropna(how="all")

    return query, instrument_returns


def compute_fetch_lookback(
    ticker_str: str,
    max_lookback: int = 365,
    min_lookback: int = 10,
    buffer_days: int = 5,
) -> int:
    """Determine how many days to fetch from TWS based on cache gap.

    If cache has recent data, only fetch the gap. If cache is empty or stale,
    fetch max_lookback.
    """
    from utils.cache.runtime.central_cache import CentralCacheStore

    store = CentralCacheStore.get_instance()
    try:
        ticker_enum = Ticker[ticker_str]
    except KeyError:
        return max_lookback

    record = store.describe_candle(ticker_enum, TimeFrame.D)
    if record is None or record.coverage.end is None:
        return max_lookback

    gap_days = (datetime.now() - record.coverage.end).days + buffer_days
    return max(min(gap_days, max_lookback), min_lookback)


# ==============================================================================
# POSITION SIZING
# ==============================================================================

# Futures contract specifications: CME/CBOT/COMEX micro contracts.
# point_value: dollar P&L per 1.00 point move per micro contract.
# For TLT (bond ETF), we map to ZN (10Y Treasury Note future) as the closest
# liquid futures proxy; ZN has no micro variant, so we use the full contract.
FUTURES_CONTRACT_SPECS: Dict[str, Dict[str, Any]] = {
    "ES":  {"micro": "MES", "mini": "ES",  "micro_point_value": 5.0,    "mini_point_value": 50.0,  "exchange": "CME"},
    "NQ":  {"micro": "MNQ", "mini": "NQ",  "micro_point_value": 2.0,    "mini_point_value": 20.0,  "exchange": "CME"},
    "YM":  {"micro": "MYM", "mini": "YM",  "micro_point_value": 0.5,    "mini_point_value": 5.0,   "exchange": "CBOT"},
    "RTY": {"micro": "M2K", "mini": "RTY", "micro_point_value": 5.0,    "mini_point_value": 50.0,  "exchange": "CME"},
    "GC":  {"micro": "MGC", "mini": "GC",  "micro_point_value": 10.0,   "mini_point_value": 100.0, "exchange": "COMEX"},
    "TLT": {"micro": "ZN",  "mini": "ZN",  "micro_point_value": 1000.0, "mini_point_value": 1000.0, "exchange": "CBOT"},
}


def calculate_futures_contracts(
    positions_df: pd.DataFrame,
    prices: Dict[str, float],
    capital_usd: float,
) -> pd.DataFrame:
    """Convert position fractions to micro futures contract quantities.

    ``notional_per_contract = price * micro_point_value``.
    ``contracts_fractional = target_dollars / notional_per_contract``.
    ``contracts_whole = round(contracts_fractional)``.

    Returns a DataFrame with both the exact fractional value (for transparency)
    and the rounded whole integer (for execution).
    """
    results = []
    latest = positions_df.sort_values("datetime").groupby("ticker").last().reset_index()

    for _, row in latest.iterrows():
        ticker = row["ticker"]
        ticker_str = ticker.name if hasattr(ticker, "name") else str(ticker)

        spec = FUTURES_CONTRACT_SPECS.get(ticker_str)
        if spec is None:
            continue
        if ticker_str not in prices:
            continue

        forecast_score = row.get("forecast_score", 0.0)
        position_fraction = row.get("position_fraction", 0.0)
        futures_price = prices[ticker_str]
        point_value = spec["micro_point_value"]
        contract_symbol = spec["micro"]

        notional_per_contract = futures_price * point_value
        target_dollars = position_fraction * capital_usd
        contracts_fractional = (
            target_dollars / notional_per_contract if notional_per_contract > 0 else 0.0
        )
        contracts_whole = int(round(contracts_fractional))
        actual_notional = contracts_whole * notional_per_contract
        actual_pct = (actual_notional / capital_usd * 100) if capital_usd > 0 else 0.0

        results.append({
            "ticker": ticker_str,
            "contract_symbol": contract_symbol,
            "forecast": forecast_score,
            "position_pct": position_fraction * 100,
            "target_dollars": target_dollars,
            "futures_price": futures_price,
            "point_value": point_value,
            "notional_per_contract": notional_per_contract,
            "contracts_fractional": contracts_fractional,
            "contracts_whole": contracts_whole,
            "actual_notional": actual_notional,
            "actual_pct": actual_pct,
        })

    return pd.DataFrame(results)


def calculate_etf_shares(
    positions_df: pd.DataFrame,
    prices: Dict[str, float],
    capital_usd: float,
    instrument_config: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Convert position fractions to ETF share quantities.

    Parameters
    ----------
    positions_df : pd.DataFrame
        Forecast results with ticker and position_fraction columns
    prices : Dict[str, float]
        Current ETF prices
    capital_usd : float
        Account capital in USD
    instrument_config : Dict[str, Dict]
        Configuration mapping futures tickers to ETFs

    Returns
    -------
    pd.DataFrame
        DataFrame with ETF share calculations
    """
    results = []

    # Get the most recent forecast for each ticker
    latest = positions_df.sort_values("datetime").groupby("ticker").last().reset_index()

    for _, row in latest.iterrows():
        ticker = row["ticker"]

        # Handle both string and enum ticker types
        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)

        if ticker_str not in instrument_config:
            continue

        etf = instrument_config[ticker_str]["etf"]
        if etf not in prices:
            continue

        forecast_score = row.get("forecast_score", 0)
        position_fraction = row.get("position_fraction", 0)
        etf_price = prices[etf]

        # Calculate target dollars and fractional shares
        target_dollars = position_fraction * capital_usd
        shares_fractional = target_dollars / etf_price if etf_price > 0 else 0
        shares_whole = round(shares_fractional)
        actual_dollars_whole = shares_whole * etf_price
        actual_pct_whole = (actual_dollars_whole / capital_usd * 100) if capital_usd > 0 else 0

        results.append({
            "ticker": ticker_str,
            "etf": etf,
            "forecast": forecast_score,
            "position_pct": position_fraction * 100,
            "target_dollars": target_dollars,
            "etf_price": etf_price,
            "shares_fractional": shares_fractional,
            "shares_whole": shares_whole,
            "actual_dollars": actual_dollars_whole,
            "actual_pct": actual_pct_whole
        })

    return pd.DataFrame(results)


# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

def _strength_label(forecast: float) -> str:
    if forecast >= 1.5:
        return "Strong Bull"
    if forecast >= 0.5:
        return "Bullish"
    if forecast >= 0:
        return "Weak Bull"
    if forecast >= -0.5:
        return "Weak Bear"
    if forecast >= -1.5:
        return "Bearish"
    return "Strong Bear"


def format_console_output(
    forecasts_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    capital: float,
    profile: str = "prop",
) -> str:
    """Format results for console display (profile-aware: futures vs ETFs)."""
    lines = []
    lines.append("=" * 70)
    title = "PROP (MICRO FUTURES)" if profile == "prop" else "PERSONAL (ETFs)"
    lines.append(
        f"ENIGMA ALGOS FORECAST -- {title} -- "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M PST')}"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Account Capital: ${capital:,.2f} USD")
    lines.append("")

    lines.append("FORECASTS & SIGNALS:")
    lines.append(f"  {'Ticker':<6} {'Forecast':>10} {'Target %':>10}   {'Interpretation':<15}")
    lines.append("  " + "-" * 50)
    for _, row in shares_df.iterrows():
        forecast = row["forecast"]
        position_pct = row["position_pct"]
        sign = "+" if forecast >= 0 else ""
        lines.append(
            f"  {row['ticker']:<6} {sign}{forecast:>9.2f} "
            f"{sign}{position_pct:>9.1f}%   {_strength_label(forecast):<15}"
        )
    lines.append("")

    total_target = 0.0
    if profile == "prop":
        lines.append("FUTURES POSITIONS (micro contracts):")
        lines.append(
            f"  {'Ticker':<6} {'Contract':<8} {'Price':>10} {'Allocate $':>12} "
            f"{'Fractional':>11} {'Whole':>7}"
        )
        lines.append("  " + "-" * 60)
        for _, row in shares_df.iterrows():
            total_target += row["target_dollars"]
            dollars = row["target_dollars"]
            dollars_sign = "+" if dollars >= 0 else "-"
            frac_sign = "+" if row["contracts_fractional"] >= 0 else "-"
            lines.append(
                f"  {row['ticker']:<6} {row['contract_symbol']:<8} "
                f"${row['futures_price']:>9,.2f} "
                f"{dollars_sign}${abs(dollars):>10,.2f} "
                f"{frac_sign}{abs(row['contracts_fractional']):>10.3f} "
                f"{row['contracts_whole']:>+7d}"
            )
        total_pct = (total_target / capital * 100) if capital > 0 else 0
        lines.append("  " + "-" * 60)
        total_sign = "+" if total_target >= 0 else "-"
        lines.append(
            f"  {'TOTAL':<6} {'':<8} {'':<10} "
            f"{total_sign}${abs(total_target):>10,.2f} ({total_pct:>+6.1f}%)"
        )
        lines.append("")
        lines.append("Note: Forecast > 0 = bullish, < 0 = bearish.")
        lines.append("      Fractional = target sizing before rounding; Whole = what to trade.")
        lines.append("      Target % can exceed 100% due to diversification multipliers.")
    else:
        lines.append("ETF POSITIONS (fractional shares):")
        lines.append(
            f"  {'Ticker':<6} {'ETF':<5} {'Price':>9} {'Allocate $':>11} {'Shares':>10}"
        )
        lines.append("  " + "-" * 50)
        for _, row in shares_df.iterrows():
            total_target += row["target_dollars"]
            sign = "+" if row["target_dollars"] >= 0 else ""
            lines.append(
                f"  {row['ticker']:<6} {row['etf']:<5} ${row['etf_price']:>8.2f} "
                f"{sign}${abs(row['target_dollars']):>9.2f} {row['shares_fractional']:>10.3f}"
            )
        total_pct = (total_target / capital * 100) if capital > 0 else 0
        lines.append("  " + "-" * 50)
        lines.append(
            f"  {'TOTAL':<6} {'':<5} {'':<9} "
            f"${total_target:>10.2f} ({total_pct:.1f}%)"
        )
        lines.append("")
        lines.append("Note: Forecast > 0 = bullish, < 0 = bearish. TWS supports fractional shares")
        lines.append("      during RTH (9:30 AM - 4:00 PM ET).")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_telegram_message(
    shares_df: pd.DataFrame,
    capital: float,
    profile: str = "prop",
) -> str:
    """Format results for Telegram notification (profile-aware)."""
    lines = []
    heading = (
        "*ENIGMA ALGOS FORECAST -- PROP*"
        if profile == "prop"
        else "*ENIGMA ALGOS FORECAST -- PERSONAL*"
    )
    lines.append(heading)
    lines.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M PST')}_")
    lines.append("")
    lines.append(f"Capital: ${capital:,.2f}")
    lines.append("")

    lines.append("*SIGNALS* (forecast > 0 = bullish)")
    lines.append("```")
    lines.append(f"{'Ticker':<6} {'Signal':>8} {'Strength':<12}")
    lines.append("-" * 28)
    for _, row in shares_df.iterrows():
        lines.append(
            f"{row['ticker']:<6} {row['forecast']:>+8.2f} {_strength_label(row['forecast']):<12}"
        )
    lines.append("```")
    lines.append("")

    total_dollars = 0.0
    if profile == "prop":
        lines.append("*POSITIONS* (micro futures)")
        lines.append("```")
        lines.append(f"{'Sym':<4} {'Price':>10} {'Frac':>7} {'Whole':>6}")
        lines.append("-" * 32)
        for _, row in shares_df.iterrows():
            total_dollars += row["target_dollars"]
            price_str = f"${row['futures_price']:,.2f}"
            lines.append(
                f"{row['contract_symbol']:<4} {price_str:>10} "
                f"{row['contracts_fractional']:>+7.3f} {row['contracts_whole']:>+6d}"
            )
        lines.append("-" * 32)
        total_sign = "+" if total_dollars >= 0 else "-"
        total_str = f"{total_sign}${abs(total_dollars):,.0f}"
        total_pct = (total_dollars / capital * 100) if capital > 0 else 0
        lines.append(f"{'TOTAL':<4} {total_str:>10} {total_pct:>+7.0f}%")
        lines.append("```")
    else:
        lines.append("*POSITIONS* (ETF fractional shares)")
        lines.append("```")
        lines.append(f"{'ETF':<5} {'Price':>9} {'Dollars':>8} {'Shares':>7}")
        lines.append("-" * 33)
        for _, row in shares_df.iterrows():
            total_dollars += row["target_dollars"]
            price_str = f"${row['etf_price']:.2f}"
            dollars = row["target_dollars"]
            dollars_str = f"${abs(dollars):.0f}" if dollars >= 0 else f"-${abs(dollars):.0f}"
            lines.append(
                f"{row['etf']:<5} {price_str:>9} {dollars_str:>8} {row['shares_fractional']:>7.2f}"
            )
        lines.append("-" * 33)
        total_pct = (total_dollars / capital * 100) if capital > 0 else 0
        total_str = f"${total_dollars:.0f}"
        lines.append(f"{'TOTAL':<5} {'':>9} {total_str:>8} ({total_pct:.0f}%)")
        lines.append("```")

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enigma Live Forecast Pipeline")
    parser.add_argument(
        "--profile",
        choices=["prop", "personal"],
        default="prop",
        help=(
            "Signal profile. 'prop' uses vault/ and sizes micro futures; "
            "'personal' uses vault_personal/, sizes ETF shares, fetches a "
            "partial 15-min daily candle for today, and posts to the personal channel."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to config file (profile default if omitted)")
    parser.add_argument("--capital", type=float, help="Override account capital")
    parser.add_argument("--port", type=int, help="Override TWS port (7497=paper, 7496=live)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without sending Telegram")
    args = parser.parse_args()

    profile = args.profile

    # Pick config file based on profile (fall back to legacy shared config if
    # a profile-specific one does not exist yet).
    project_root = Path(__file__).parent.parent
    if args.config is not None:
        config_candidate = Path(args.config)
        if not config_candidate.is_absolute():
            config_candidate = project_root / args.config
        config_path = config_candidate
    else:
        profile_default = (
            DEFAULT_CONFIG_PATH_PROP if profile == "prop" else DEFAULT_CONFIG_PATH_PERSONAL
        )
        profile_path = project_root / profile_default
        legacy_path = project_root / DEFAULT_CONFIG_PATH
        config_path = profile_path if profile_path.exists() else legacy_path

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Profile-specific vault root override: prop -> vault/, personal -> vault_personal/
    # unless the config already has an explicit override.
    if "portfolio" in config and "vault_root" not in config.get("portfolio", {}):
        config.setdefault("portfolio", {})
    if profile == "personal":
        config["portfolio"]["vault_root"] = config["portfolio"].get("vault_root") or "vault_personal"
    else:
        config["portfolio"]["vault_root"] = config["portfolio"].get("vault_root") or "vault"

    # Apply overrides
    capital = args.capital or config["account"]["capital_usd"]
    port = args.port or config["connection"]["port"]

    print("=" * 60)
    header = "Prop Firms (futures)" if profile == "prop" else "Personal Account (ETFs)"
    print(f"Enigma Live Forecast Pipeline -- {header}")
    print("=" * 60)
    print(f"Profile: {profile}")
    print(f"Config: {config_path}")
    print(f"Vault: {config['portfolio']['vault_root']}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Port: {port} ({'Paper' if port == 7497 else 'Live' if port == 7496 else 'Custom'})")
    print(f"Dry Run: {args.dry_run}")

    # Step 1: Build portfolio from vault (before data fetching so we know what tickers to fetch)
    print("\n1. Building portfolio from vault...")
    portfolio = build_portfolio(config)

    # Step 2: Discover all tickers needed
    required_tickers = discover_required_tickers(portfolio)
    print(f"\n   Required tickers: {sorted(required_tickers)}")

    # Validate that all required tickers are in the config
    missing_tickers = required_tickers - set(config["instruments"].keys())
    if missing_tickers:
        print(f"   Warning: Tickers not in config (will be skipped): {sorted(missing_tickers)}")

    # Step 3: Ensure central cache has baseline data (bootstrap if empty)
    print("\n2. Ensuring cache is ready...")
    cache_status = ensure_cache_ready(required_tickers)
    if cache_status["bootstrapped"]:
        print("  Cache bootstrapped from repo parquets.")
    for tk, cov in cache_status.get("coverage", {}).items():
        print(f"  {tk}: {cov['start'].date()} to {cov['end'].date()}")

    # Create IB client
    ib_config = IBConfig(
        host=config["connection"]["host"],
        port=port,
        client_id=config["connection"]["client_id"]
    )
    client = IBDataClient(ib_config)

    try:
        # Connect to TWS
        print("\n3. Connecting to TWS...")
        client.connect_to_ib()

        # Wait for connection (nextValidId callback sets connected=True)
        max_wait = 10
        waited = 0
        while not client.connected and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5

        if not client.connected:
            print("ERROR: Could not connect to TWS.")
            print("Please ensure:")
            print("  1. TWS or IB Gateway is running")
            print("  2. API is enabled in TWS settings")
            print(f"  3. Port {port} is correct")
            sys.exit(1)

        print("Connected to TWS successfully!")

        # Brief pause to ensure API is fully ready
        time.sleep(1)

        # Step 4: Fetch historical data with smart lookback per ticker
        print("\n4. Fetching historical data...")
        all_candles = []
        instruments = config["instruments"]
        data_cfg = config["data"]
        min_lb = data_cfg.get("min_lookback_days", 10)
        max_lb = data_cfg.get("max_lookback_days", 365)

        for ticker in sorted(required_tickers):
            if ticker not in instruments:
                continue
            lookback = compute_fetch_lookback(ticker, max_lookback=max_lb, min_lookback=min_lb)
            print(f"  {ticker}: fetching {lookback} days")
            candles = fetch_historical_candles(
                client,
                ticker,
                instruments[ticker],
                lookback_days=lookback,
            )
            if not candles.empty:
                all_candles.append(candles)

        if not all_candles:
            print("ERROR: No historical data fetched for any instrument.")
            sys.exit(1)

        daily_candles = pd.concat(all_candles, ignore_index=True)
        print(f"Total daily candles: {len(daily_candles)} across {daily_candles['ticker'].nunique()} instruments")

        # Personal profile: append a synthetic partial daily candle for today
        # built from 15-min intraday bars. We run the script before the regular
        # daily close, so IB's daily bar for today isn't available yet.
        if profile == "personal":
            print("\n4b. Fetching partial daily candles (15-min aggregation) for today...")
            partial_frames = []
            for ticker in sorted(required_tickers):
                if ticker not in instruments:
                    continue
                partial = fetch_partial_daily_candle(
                    client,
                    ticker,
                    instruments[ticker],
                    intraday_bar_size="15 mins",
                )
                if partial.empty:
                    continue
                # Drop any existing row for the same date/ticker so we replace
                # whatever IB returned with the fresher synthetic bar.
                partial_date = partial["datetime"].iloc[0]
                mask = (
                    (daily_candles["ticker"] == ticker)
                    & (pd.to_datetime(daily_candles["datetime"]).dt.normalize() == partial_date)
                )
                daily_candles = daily_candles.loc[~mask]
                partial_frames.append(partial)
            if partial_frames:
                daily_candles = pd.concat(
                    [daily_candles] + partial_frames, ignore_index=True
                ).sort_values(["ticker", "datetime"]).reset_index(drop=True)
                print(
                    f"  Appended {len(partial_frames)} partial daily candle(s) dated today. "
                    f"Total daily candles: {len(daily_candles)}."
                )
            else:
                print("  No partial daily candles synthesized (no intraday bars returned).")

        # Step 5: Upsert fetched candles into central cache (+ auto monthly resample)
        print("\n5. Upserting TWS candles into cache...")
        upsert_tws_candles(daily_candles, required_tickers)

        # Populate cross-ticker store with ALL fetched tickers.
        # Ensembles with cross-ticker bias nodes (e.g., rebalancing) look up
        # data from the cross-ticker store at predict time. We pre-load every
        # fetched ticker so lookups succeed regardless of camelCase/snake_case
        # param key differences.
        print("\n   Populating cross-ticker data store...")
        from utils.data.cross_ticker_store import CrossTickerDataStore

        ct_store = CrossTickerDataStore.get_instance()
        for ticker_name in sorted(daily_candles['ticker'].unique()):
            ticker_str = str(ticker_name)
            try:
                ct_ticker = Ticker[ticker_str]
            except KeyError:
                continue
            if ct_store.is_loaded(ct_ticker, TimeFrame.D):
                continue
            ticker_mask = daily_candles['ticker'] == ticker_name
            ct_store.set_data(ct_ticker, TimeFrame.D, daily_candles.loc[ticker_mask].copy())
            print(f"   Cross-ticker {ticker_str}: loaded from fetched data")

        # Step 6: Refresh stale bias node caches
        print("\n6. Refreshing bias caches...")
        refresh_bias_caches(
            str(resolve_vault_root(config["portfolio"]["vault_root"])),
            required_tickers,
        )

        # Step 7: Build cache query and fit portfolio
        print("\n7. Fitting portfolio from cache...")
        query, instrument_returns = build_cache_query(required_tickers)
        portfolio.fit_from_cache(query, instrument_returns)

        # Log portfolio parameters
        print("\n" + "=" * 60)
        print("PORTFOLIO PARAMETERS (after fitting):")
        for tf_p in portfolio.tf_portfolios:
            print(f"  {tf_p.trading_timeframe.name}:")
            print(f"    Ensembles: {len(tf_p.ensembles)}")
            print(f"    IDM: {tf_p.idm_ if tf_p.idm_ else 'N/A'}")
            print(f"    Instruments: {tf_p.instruments_ if tf_p.instruments_ else 'N/A'}")
        print(f"  Global IDM: {portfolio.global_idm_ if portfolio.global_idm_ else 'N/A'}")
        print(f"  Max Position %: {portfolio.max_position_pct}")
        print("=" * 60)

        # Step 8: Generate forecasts from cache
        print("\n8. Generating forecasts from cache...")
        positions_df = portfolio.predict_from_cache(query)

        if positions_df.empty:
            print("ERROR: No forecasts generated.")
            sys.exit(1)

        # Log raw forecast details
        print("\n" + "=" * 60)
        print("RAW FORECAST OUTPUT FROM PORTFOLIO:")
        latest = positions_df.sort_values("datetime").groupby("ticker").last().reset_index()
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
            forecast = row.get("forecast_score", "N/A")
            position_frac = row.get("position_fraction", "N/A")
            print(f"  {ticker_str}: forecast_score={forecast:.4f}, position_fraction={position_frac:.4f}")
        print("=" * 60)

        # Step 9: Position sizing (profile-specific)
        position_tickers = set()
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            ticker_str = ticker.name if hasattr(ticker, "name") else str(ticker)
            if ticker_str in instruments:
                position_tickers.add(ticker_str)

        if profile == "prop":
            # Futures sizing: use last-close futures prices from fetched candles
            # (no extra TWS round-trip needed) and compute micro contract counts.
            print("\n9. Using latest futures prices from fetched candles...")
            futures_prices: Dict[str, float] = {}
            last_closes = (
                daily_candles.sort_values("datetime").groupby("ticker")["close"].last()
            )
            for ticker_val, close_price in last_closes.items():
                ticker_str = ticker_val.name if hasattr(ticker_val, "name") else str(ticker_val)
                futures_prices[ticker_str] = float(close_price)
                print(f"  {ticker_str}: ${float(close_price):,.2f}")
            if not futures_prices:
                print("ERROR: No futures prices available.")
                sys.exit(1)
            print("\n10. Calculating futures positions...")
            shares_df = calculate_futures_contracts(positions_df, futures_prices, capital)
        else:
            # ETF sizing: fetch current ETF prices from TWS and compute fractional shares.
            print("\n9. Fetching current ETF prices...")
            etf_symbols = list({
                instruments[t]["etf"]
                for t in position_tickers
                if t in instruments
            })
            prices = fetch_current_prices(client, etf_symbols)
            if not prices:
                print("ERROR: No ETF prices fetched.")
                sys.exit(1)
            print("\n10. Calculating ETF positions...")
            shares_df = calculate_etf_shares(positions_df, prices, capital, instruments)

        # Log position sizing calculation
        print("\n" + "=" * 60)
        print("POSITION SIZING CALCULATION:")
        print(f"  Capital: ${capital:,.2f}")
        total_dollars = 0
        for _, row in shares_df.iterrows():
            ticker = row["ticker"]
            pos_frac = row["position_pct"] / 100
            target_dollars = row["target_dollars"]
            total_dollars += target_dollars
            print(f"  {ticker}: position_fraction={pos_frac:.4f} x ${capital:,.0f} = ${target_dollars:.2f}")
        print(f"  TOTAL: ${total_dollars:.2f} ({total_dollars/capital*100:.1f}% of capital)")
        print("=" * 60)

        # Display results
        print("\n" + format_console_output(positions_df, shares_df, capital, profile=profile))

        # Telegram: route to the profile-appropriate bot/channel.
        notifier = (
            TelegramNotifier.for_prop_firms()
            if profile == "prop"
            else TelegramNotifier.for_personal_account()
        )
        message = format_telegram_message(shares_df, capital, profile=profile)
        if not args.dry_run:
            print(f"\n11. Sending Telegram notification ({profile})...")
            if notifier.send_message(message):
                print("Telegram notification sent!")
            else:
                print("Warning: Failed to send Telegram notification")
        else:
            print("\n11. Skipping Telegram (dry-run mode)")
            print("\nTelegram message would be:")
            print("-" * 40)
            print(message)
            print("-" * 40)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect
        if client.connected:
            print("\nDisconnecting from TWS...")
            client.disconnect_from_ib()

    print("\nDone!")


if __name__ == "__main__":
    main()
