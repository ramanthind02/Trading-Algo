#!/usr/bin/env python3
"""
TWS Live Forecast Pipeline
===========================

Connects to TWS API, fetches historical data for equity index futures,
generates forecasts using the Portfolio class, and converts to ETF positions.

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
from typing import Dict, List
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
from ensemble.portfolio import Portfolio
from deployment.telegram_notifier import TelegramNotifier
from utils.enums import TimeFrame


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG_PATH = "configs/live_forecast_config.json"

# Mapping from our ticker symbols to TWS CONTFUT symbols
TICKER_TO_TWS = {
    "ES": ("ES", "CME"),
    "NQ": ("NQ", "CME"),
    "YM": ("YM", "CBOT"),
    "RTY": ("RTY", "CME"),
}


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


def fetch_historical_candles(
    client: IBDataClient,
    ticker: str,
    lookback_days: int = 60
) -> pd.DataFrame:
    """
    Fetch CONTFUT bars and convert to candles DataFrame format.

    Parameters
    ----------
    client : IBDataClient
        Connected IB client
    ticker : str
        Ticker symbol (ES, NQ, YM, RTY)
    lookback_days : int
        Number of days of history to fetch

    Returns
    -------
    pd.DataFrame
        Candles with columns: datetime, open, high, low, close, volume, ticker, timeframe
    """
    if ticker not in TICKER_TO_TWS:
        raise ValueError(f"Unknown ticker: {ticker}. Valid tickers: {list(TICKER_TO_TWS.keys())}")

    tws_symbol, exchange = TICKER_TO_TWS[ticker]
    contract = create_futures_contract(tws_symbol, exchange)

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

    # Convert bars to DataFrame - we have data regardless of wait result
    rows = []
    for bar in client.historical_bars:
        # Parse date - IB returns format like "20260124"
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
        List of ETF symbols (SPY, QQQ, DIA, IWM)

    Returns
    -------
    Dict[str, float]
        Mapping from symbol to current price
    """
    prices = {}

    for symbol in etf_symbols:
        contract = create_etf_contract(symbol)

        # Request snapshot market data
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

        # Wait for data
        if client.wait_for_historical_data(req_id, timeout=15.0):
            if client.historical_bars:
                # Use the last bar's close as the current price
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

def load_portfolio(vault_paths: List[str], target_vol: float = 0.10) -> Portfolio:
    """
    Load pre-fitted ensembles and create Portfolio.

    Parameters
    ----------
    vault_paths : List[str]
        List of paths to vault ensemble directories
    target_vol : float
        Target volatility for the portfolio

    Returns
    -------
    Portfolio
        Configured portfolio with loaded ensembles
    """
    ensembles = []

    for vault_path in vault_paths:
        print(f"  Loading ensemble from: {vault_path}")
        ensemble = load_ensemble_from_vault(
            ensemble_dir=vault_path,
            refit=False,  # Use pre-fitted params
            target_volatility=target_vol
        )
        ensembles.append(ensemble)
        print(f"    -> Loaded: {ensemble}")

    # Create portfolio with all ensembles
    portfolio = Portfolio(
        ensembles=ensembles,
        trading_timeframe=TimeFrame.D,
        target_volatility=target_vol,
        max_position_pct=2.5,  # Allow up to 250% for diversification
        idm_max=2.5
    )

    print(f"  Portfolio created with {len(ensembles)} ensemble(s)")
    return portfolio


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
    lines.append(f"TWS LIVE FORECAST - {datetime.now().strftime('%Y-%m-%d %H:%M PST')}")
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
    lines.append("*TWS LIVE FORECAST*")
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
        direction = "+" if dollars >= 0 else ""
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
        # Try relative to project root
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
    print("TWS Live Forecast Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Port: {port} ({'Paper' if port == 7497 else 'Live' if port == 7496 else 'Custom'})")
    print(f"Dry Run: {args.dry_run}")

    # Create IB client
    ib_config = IBConfig(
        host=config["connection"]["host"],
        port=port,
        client_id=config["connection"]["client_id"]
    )
    client = IBDataClient(ib_config)

    try:
        # Connect to TWS
        print("\n1. Connecting to TWS...")
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

        # Fetch historical data for all instruments
        print("\n2. Fetching historical data...")
        all_candles = []

        for ticker in config["instruments"].keys():
            candles = fetch_historical_candles(
                client,
                ticker,
                lookback_days=config["data"]["lookback_days"]
            )
            if not candles.empty:
                all_candles.append(candles)

        if not all_candles:
            print("ERROR: No historical data fetched for any instrument.")
            sys.exit(1)

        candles_df = pd.concat(all_candles, ignore_index=True)
        print(f"Total candles: {len(candles_df)} across {candles_df['ticker'].nunique()} instruments")

        # Load portfolio
        print("\n3. Loading portfolio...")
        portfolio = load_portfolio(
            config["portfolio"]["vault_paths"],
            config["portfolio"]["target_volatility"]
        )

        # Fit portfolio with historical data
        print("Fitting portfolio with historical data...")
        try:
            portfolio.fit_from_candles(candles_df, target_data=candles_df.groupby('ticker')['close'].pct_change())
            fit_success = True
        except Exception as e:
            print(f"  Warning: Portfolio fitting failed ({e}), using default weights")
            fit_success = False

        # Log portfolio parameters
        print("\n" + "=" * 60)
        print("PORTFOLIO PARAMETERS" + (" (after fitting):" if fit_success else " (using defaults):"))
        print(f"  Target Volatility: {portfolio.target_volatility}")
        print(f"  IDM (Instrument Diversification Multiplier): {portfolio.idm_ if portfolio.idm_ else 'N/A (will use default)'}")
        print(f"  IDM Max Cap: {portfolio.idm_max}")
        print(f"  Max Position %: {portfolio.max_position_pct}")
        print(f"  Instruments: {portfolio.instruments_ if portfolio.instruments_ else 'N/A'}")
        print(f"  Ensembles: {len(portfolio.ensembles)}")
        print("=" * 60)

        # Generate forecasts
        print("\n4. Generating forecasts...")
        positions_df = portfolio.predict_from_candles(candles_df)

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

        # Fetch current ETF prices
        print("\n5. Fetching current ETF prices...")
        etf_symbols = [config["instruments"][t]["etf"] for t in config["instruments"]]
        prices = fetch_current_prices(client, etf_symbols)

        if not prices:
            print("ERROR: No ETF prices fetched.")
            sys.exit(1)

        # Calculate ETF shares
        print("\n6. Calculating ETF positions...")
        shares_df = calculate_etf_shares(
            positions_df,
            prices,
            capital,
            config["instruments"]
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
            print("\n7. Sending Telegram notification...")
            notifier = TelegramNotifier()
            message = format_telegram_message(shares_df, capital)
            if notifier.send_message(message):
                print("Telegram notification sent!")
            else:
                print("Warning: Failed to send Telegram notification")
        else:
            print("\n7. Skipping Telegram (dry-run mode)")
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
