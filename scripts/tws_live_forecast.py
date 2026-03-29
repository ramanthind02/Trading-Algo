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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IB API imports
from ibapi.contract import Contract

# Project imports
from scripts.demo_ib_data_fetch import IBDataClient, IBConfig
from ensemble.vault_manager import load_ensemble_from_vault
from ensemble.portfolio import TFPortfolio, GlobalPortfolio
from deployment.telegram_notifier import TelegramNotifier
from utils.core.enums import TimeFrame, Ticker


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG_PATH = "configs/live_forecast_config.json"


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
    """Load all ensembles from vault/{tf.name}/ for a single timeframe."""
    tf_dir = Path(vault_root) / tf.name
    if not tf_dir.exists():
        return []
    ensembles = []
    for ens_dir in sorted(tf_dir.iterdir()):
        if not ens_dir.is_dir():
            continue
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
    vault_root = config["portfolio"]["vault_root"]
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
    from utils.cache.cache_manager import CacheManager

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
    from utils.cache.cache_manager import CacheManager

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

    vault_dirs = []
    for tf_name in ["D", "M"]:
        tf_dir = Path(vault_root) / tf_name
        if not tf_dir.exists():
            continue
        for ens_dir in sorted(tf_dir.iterdir()):
            if ens_dir.is_dir():
                vault_dirs.append(str(ens_dir))

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
    from utils.cache.central_cache_models import ArtifactScope
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

def format_console_output(
    forecasts_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    capital: float
) -> str:
    """Format results for console display."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"TWS LIVE FORECAST (Multi-TF) - {datetime.now().strftime('%Y-%m-%d %H:%M PST')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Account Capital: ${capital:,.2f} USD")
    lines.append("")

    # Forecasts section with interpretation
    lines.append("FORECASTS & SIGNALS:")
    lines.append(f"  {'Ticker':<6} {'Forecast':>10} {'Target %':>10}   {'Interpretation':<15}")
    lines.append("  " + "-" * 50)
    for _, row in shares_df.iterrows():
        forecast = row["forecast"]
        position_pct = row["position_pct"]
        sign = "+" if forecast >= 0 else ""

        # Interpret forecast
        if forecast >= 1.5:
            interp = "Strong Bullish"
        elif forecast >= 0.5:
            interp = "Bullish"
        elif forecast >= 0:
            interp = "Weak Bullish"
        elif forecast >= -0.5:
            interp = "Weak Bearish"
        elif forecast >= -1.5:
            interp = "Bearish"
        else:
            interp = "Strong Bearish"

        lines.append(f"  {row['ticker']:<6} {sign}{forecast:>9.2f} {sign}{position_pct:>9.1f}%   {interp:<15}")
    lines.append("")

    # ETF positions section - fractional shares
    lines.append("ETF POSITIONS (fractional shares for precise allocation):")
    lines.append(f"  {'Ticker':<6} {'ETF':<5} {'Price':>9} {'Allocate $':>11} {'Shares':>10}")
    lines.append("  " + "-" * 45)

    total_target = 0

    for _, row in shares_df.iterrows():
        total_target += row["target_dollars"]
        sign = "+" if row["target_dollars"] >= 0 else ""

        lines.append(
            f"  {row['ticker']:<6} {row['etf']:<5} ${row['etf_price']:>8.2f} "
            f"{sign}${abs(row['target_dollars']):>9.2f} {row['shares_fractional']:>10.3f}"
        )

    # Totals
    total_pct = (total_target / capital * 100) if capital > 0 else 0
    lines.append("  " + "-" * 45)
    lines.append(
        f"  {'TOTAL':<6} {'':<5} {'':<9} "
        f"${total_target:>10.2f} ({total_pct:.1f}%)"
    )

    lines.append("")
    lines.append("Note: Forecast > 0 means bullish, < 0 means bearish.")
    lines.append("      Target % can exceed 100% due to diversification multipliers (IDM/FDM).")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_telegram_message(
    shares_df: pd.DataFrame,
    capital: float
) -> str:
    """Format results for Telegram notification."""
    lines = []
    lines.append("*TWS LIVE FORECAST (Multi-TF)*")
    lines.append(f"_{datetime.now().strftime('%Y-%m-%d %H:%M PST')}_")
    lines.append("")
    lines.append(f"Capital: ${capital:,.2f}")
    lines.append("")

    # Forecast explanation
    lines.append("*SIGNALS* (forecast > 0 = bullish)")
    lines.append("```")
    lines.append(f"{'Ticker':<6} {'Signal':>8} {'Strength':<12}")
    lines.append("-" * 28)

    for _, row in shares_df.iterrows():
        forecast = row["forecast"]
        ticker = row["ticker"]

        # Interpret forecast strength
        if forecast >= 1.5:
            strength = "Strong Bull"
        elif forecast >= 0.5:
            strength = "Bullish"
        elif forecast >= 0:
            strength = "Weak Bull"
        elif forecast >= -0.5:
            strength = "Weak Bear"
        elif forecast >= -1.5:
            strength = "Bearish"
        else:
            strength = "Strong Bear"

        sign = "+" if forecast >= 0 else ""
        lines.append(f"{ticker:<6} {sign}{forecast:>7.2f} {strength:<12}")

    lines.append("```")
    lines.append("")

    # Position sizing
    lines.append("*POSITIONS* (fractional shares)")
    lines.append("```")
    lines.append(f"{'ETF':<5} {'Price':>8} {'Dollars':>9} {'Shares':>8}")
    lines.append("-" * 32)

    total_dollars = 0

    for _, row in shares_df.iterrows():
        etf = row["etf"]
        price = row["etf_price"]
        dollars = row["target_dollars"]
        shares = row["shares_fractional"]
        total_dollars += dollars

        # Show direction
        direction = "+" if dollars >= 0 else "-"
        lines.append(f"{etf:<5} ${price:>7.2f} {direction}${abs(dollars):>7.0f} {shares:>8.2f}")

    lines.append("-" * 32)
    total_pct = (total_dollars / capital * 100) if capital > 0 else 0
    lines.append(f"{'TOTAL':<5} {'':<8} ${total_dollars:>8.0f} ({total_pct:.0f}%)")
    lines.append("```")

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="TWS Live Forecast Pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config file")
    parser.add_argument("--capital", type=float, help="Override account capital")
    parser.add_argument("--port", type=int, help="Override TWS port (7497=paper, 7496=live)")
    parser.add_argument("--dry-run", action="store_true", help="Print results without sending Telegram")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        project_root = Path(__file__).parent.parent
        config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Apply overrides
    capital = args.capital or config["account"]["capital_usd"]
    port = args.port or config["connection"]["port"]

    print("=" * 60)
    print("TWS Live Forecast Pipeline (Multi-TF)")
    print("=" * 60)
    print(f"Config: {config_path}")
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
        refresh_bias_caches(config["portfolio"]["vault_root"], required_tickers)

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

        # Step 9: Fetch current ETF prices
        print("\n9. Fetching current ETF prices...")
        # Only fetch ETF prices for tickers that appear in positions
        position_tickers = set()
        for _, row in latest.iterrows():
            ticker = row["ticker"]
            ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
            if ticker_str in instruments:
                position_tickers.add(ticker_str)

        etf_symbols = list({
            instruments[t]["etf"]
            for t in position_tickers
            if t in instruments
        })
        prices = fetch_current_prices(client, etf_symbols)

        if not prices:
            print("ERROR: No ETF prices fetched.")
            sys.exit(1)

        # Step 10: Calculate ETF shares
        print("\n10. Calculating ETF positions...")
        shares_df = calculate_etf_shares(
            positions_df,
            prices,
            capital,
            instruments
        )

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
        print("\n" + format_console_output(positions_df, shares_df, capital))

        # Send Telegram notification
        if not args.dry_run:
            print("\n11. Sending Telegram notification...")
            notifier = TelegramNotifier()
            message = format_telegram_message(shares_df, capital)
            if notifier.send_message(message):
                print("Telegram notification sent!")
            else:
                print("Warning: Failed to send Telegram notification")
        else:
            print("\n11. Skipping Telegram (dry-run mode)")
            print("\nTelegram message would be:")
            print("-" * 40)
            print(format_telegram_message(shares_df, capital))
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
