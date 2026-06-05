"""Run a manual forecast pass without waiting for the deployment scheduler."""

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from deployment.forecast_server import ForecastServer
from lib.core.enums import TimeFrame
from lib.core.logger import get_logger

logger = get_logger(__name__)


def print_forecast_results(results: dict) -> None:
    """Pretty print forecast results."""
    print("\n" + "="*80)
    print("FORECAST TEST RESULTS")
    print("="*80)
    print(f"Timestamp: {results['timestamp']}\n")
    
    for timeframe, data in results['forecasts'].items():
        print(f"{timeframe} Forecasts: {data['count']} total")
        # predictions is now a dict: {ticker_name: forecast_value}
        for ticker, prediction in data['predictions'].items():
            print(f"  {ticker}: {prediction:+.4f}")
        print()
    
    if results['errors']:
        print("ERRORS:")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    print("="*80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manual forecast test")
    parser.add_argument("--timeframe", choices=["D", "W"], help="Specific timeframe to test")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()
    
    try:
        print("🚀 Initializing test server...")
        logger.info("🚀 Initializing test server...")
        server = ForecastServer()
        print("✅ Server initialized")
        
        print("📚 Loading historical data...")
        logger.info("📚 Loading historical data...")
        server.load_historical_data()
        print("✅ Historical data loaded")
        
        timeframe = TimeFrame[args.timeframe] if args.timeframe else None
        
        print("🧪 Running test forecast...")
        logger.info("🧪 Running test forecast...")
        results = server.run_test_forecast(timeframe=timeframe)
        print("✅ Forecast complete")
        
        print_forecast_results(results)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to {args.output}")
        
        print("🧹 Cleaning up...")
        server.stop()
        logger.info("✅ Test complete!")
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        logger.error(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
