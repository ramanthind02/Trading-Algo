"""
Forecast Server

Production deployment server that uses Portfolio to manage multiple ensembles.
Generates forecasts for multiple ticker/timeframe combinations efficiently.
"""

import os
import time
import schedule
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

try:
    from deployment._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from utils.core.logger import get_logger
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle
import utils.core.helpers as helpers
from deployment.forecast_live_inputs import ForecastLiveInputs
from deployment.forecast_prediction_runtime import ForecastPredictionRuntime
from deployment.mt5_data_connector import ForecastMT5DataConnector
from deployment.telegram_notifier import TelegramNotifier
from utils.compute.daily_ewsd_volatility import DailyEWSDVolatilityService

logger = get_logger(__name__)
NY_TZ = ZoneInfo("America/New_York")

class ForecastServer:
    """
    Production forecasting server.
    
    Uses one ensemble and MLManager per ticker/timeframe combination.
    Generates one forecast per ticker/timeframe combination.
    
    Architecture:
    - One MLManager per ticker/timeframe (for feature extraction)
    - Features computed from real-time market data
    - Forecasts generated on schedule and sent to Telegram
    """
    
    def __init__(self, config_dir: str = "deployment/config"):
        """
        Initialize forecast server.
        
        Parameters
        ----------
        config_dir : str
            Path to configuration directory containing ensemble control files
            organized by timeframe (e.g., config/D/, config/W/)
        """
        self.config_dir = config_dir
        logger.info(f"🚀 Initializing ForecastServer with config dir: {config_dir}")
        
        # Initialize components
        self.mt5_connector = ForecastMT5DataConnector()
        self.telegram = TelegramNotifier()
        
        # Portfolios - one per timeframe
        # Changed: Now we store ensembles per ticker/timeframe, not one portfolio per timeframe
        self.ensembles: Dict[tuple, Any] = {}  # (ticker, timeframe) -> DiversifiedEnsemble
        
        # MLManagers - one per ticker/timeframe for feature extraction
        # We need separate MLManagers per ticker because each ticker has different price data
        self.ml_managers: Dict[tuple, Any] = {}  # (ticker, timeframe) -> MLManager-like
        
        # Candle buffers for each ticker/timeframe
        self.candle_buffers: Dict[tuple, List[Candle]] = {}  # (ticker, timeframe) -> [candles]
        self.lookback_candles = 50
        
        # Configuration
        self.tickers = [Ticker.EU, Ticker.BP, Ticker.ES, Ticker.NQ]
        self.timeframes = [TimeFrame.D, TimeFrame.W]  # Daily and Weekly
        self.volatility_service = DailyEWSDVolatilityService(
            store_dir=str(Path(__file__).resolve().parent / "state" / "ewsd_volatility")
        )
        
        # Initialize portfolios and MLManagers
        self._setup_ensembles()
        self._setup_ml_managers()
        
        # Setup scheduling
        self._setup_scheduling()
        
        logger.info(f"✅ ForecastServer initialized")
        logger.info(f"   Ensembles: {len(self.ensembles)}")
        logger.info(f"   MLManagers: {len(self.ml_managers)}")

    def _live_inputs(self) -> ForecastLiveInputs:
        """Build the live-input runtime helper around current server state."""
        return ForecastLiveInputs(
            mt5_connector=getattr(self, "mt5_connector", None),
            volatility_service=getattr(self, "volatility_service", None),
            candle_buffers=getattr(self, "candle_buffers", {}),
            ml_managers=getattr(self, "ml_managers", {}),
            ensembles=getattr(self, "ensembles", {}),
            lookback_candles=getattr(self, "lookback_candles", 0),
            tickers=getattr(self, "tickers", []),
            logger=logger,
        )

    def _prediction_runtime(self) -> ForecastPredictionRuntime:
        """Build the forecast-prediction runtime helper around current server state."""
        return ForecastPredictionRuntime(
            volatility_service=getattr(self, "volatility_service", None),
            ensembles=getattr(self, "ensembles", {}),
            ml_managers=getattr(self, "ml_managers", {}),
            candle_buffers=getattr(self, "candle_buffers", {}),
            tickers=getattr(self, "tickers", []),
            logger=logger,
        )
    
    def _setup_ensembles(self) -> None:
        """Setup one ensemble per ticker/timeframe combination."""
        logger.info("Setting up ensembles...")
        
        from ensemble.diversified_ensemble import DiversifiedEnsemble
        
        for timeframe in self.timeframes:
            # Each timeframe has its own subdirectory (e.g., config/D/, config/W/)
            tf_dir = os.path.join(self.config_dir, timeframe.name)
            
            if not os.path.exists(tf_dir):
                logger.warning(f"Timeframe directory not found: {tf_dir}")
                continue
            
            for ticker in self.tickers:
                try:
                    # Look for ensemble file: {TICKER}_{TF}.json (e.g., EU_D.json)
                    ensemble_filename = f"{ticker.name}_{timeframe.name}.json"
                    ensemble_path = os.path.join(tf_dir, ensemble_filename)
                    
                    if not os.path.exists(ensemble_path):
                        logger.warning(f"Ensemble file not found: {ensemble_path}")
                        continue
                    
                    # Load ensemble
                    ensemble = DiversifiedEnsemble(
                        control_file_path=ensemble_path,
                        base_tf=timeframe
                    )
                    
                    # Store by (ticker, timeframe) key
                    self.ensembles[(ticker, timeframe)] = ensemble
                    logger.info(f"✅ Loaded ensemble: {ticker.name} {timeframe.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load ensemble {ticker.name} {timeframe.name}: {e}")
        
        logger.info(f"Setup complete: {len(self.ensembles)} ensembles loaded")
    
    def _setup_ml_managers(self) -> None:
        """
        Setup MLManagers for feature extraction.
        
        Creates one MLManager per ticker/timeframe combination.
        Each MLManager gets the bias nodes required by the ensemble for that ticker/timeframe.
        """
        logger.info("Setting up MLManagers...")
        
        # Create MLManager for each ticker/timeframe combination
        for ticker in self.tickers:
            for timeframe in self.timeframes:
                try:
                    ensemble_key = (ticker, timeframe)
                    
                    if ensemble_key not in self.ensembles:
                        logger.warning(f"No ensemble for {ticker.name} {timeframe.name}, skipping MLManager")
                        continue
                    
                    ensemble = self.ensembles[ensemble_key]
                    
                    # Get bias nodes needed by this ensemble
                    bias_node_specs = ensemble.get_required_bias_nodes()
                    
                    logger.info(f"Ensemble {ticker.name} {timeframe.name} requires {len(bias_node_specs)} bias node types")
                    
                    # Create MLManager using helper function
                    ml_manager = helpers.create_ml_manager(
                        ticker=ticker,
                        base_tf=timeframe,
                        build_matrix=True,
                        bias_node_specs=bias_node_specs
                    )
                    
                    self.ml_managers[(ticker, timeframe)] = ml_manager
                    self.candle_buffers[(ticker, timeframe)] = []
                    
                    logger.info(f"✅ Setup MLManager: {ticker.name} {timeframe.name} ({len(ml_manager.bias_nodes)} bias nodes)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to setup MLManager for {ticker.name} {timeframe.name}: {e}")
        
        logger.info(f"Setup complete: {len(self.ml_managers)} MLManagers")
    
    def _setup_scheduling(self) -> None:
        """Setup forecast scheduling."""
        logger.info("Setting up forecast scheduling...")
        
        # Daily forecasts at midnight EST
        schedule.every().day.at("00:00").do(
            lambda: self._run_forecasts(TimeFrame.D, "Daily")
        )
        
        # Weekly forecasts on Sunday at 18:00 EST (end of week)
        schedule.every().sunday.at("18:00").do(
            lambda: self._run_forecasts(TimeFrame.W, "Weekly")
        )
        
        # Test forecast every 5 minutes (for development/testing)
        # Comment out for production
        schedule.every(5).minutes.do(
            lambda: self._run_test_forecast()
        )
        
        logger.info("✅ Scheduling setup complete")
        logger.info("   - Daily: 00:00 EST")
        logger.info("   - Weekly: Sunday 18:00 EST")
        logger.info("   - Test: Every 5 minutes")
    
    def _run_forecasts(self, timeframe: TimeFrame, timeframe_name: str) -> None:
        """
        Run forecasts for all tickers in specified timeframe.
        
        Parameters
        ----------
        timeframe : TimeFrame
            Timeframe to forecast
        timeframe_name : str
            Human-readable timeframe name
        """
        logger.info(f"🔮 Running {timeframe_name} forecasts...")
        timestamp = datetime.now(NY_TZ)
        
        # Check market status
        market_open = self.mt5_connector.is_market_open()
        market_status = "Open" if market_open else "Closed"
        logger.info(f"Market status: {market_status}")

        self._update_live_inputs_for_timeframe(timeframe)
        
        # Generate forecasts for all tickers using portfolio
        forecasts = self._generate_portfolio_forecasts(timeframe)
        
        # Send results to Telegram
        if forecasts:
            logger.info(f"📤 Sending {len(forecasts)} forecasts to Telegram...")
            
            success = self.telegram.send_forecast_update(
                forecasts=forecasts,
                timeframe=timeframe,
                timestamp=timestamp,
                market_status=market_status
            )
            
            if success:
                logger.info("✅ Forecast update sent to Telegram")
            else:
                logger.error("❌ Failed to send forecast update")
        else:
            logger.warning("⚠️ No forecasts to send")
        
        successful_count = len(forecasts)
        logger.info(f"🏁 {timeframe_name} forecast run complete: {successful_count}/{len(self.tickers)} successful")

    def _update_live_inputs_for_timeframe(self, timeframe: TimeFrame) -> None:
        """Refresh live candles, cross-ticker data, and daily volatility for one timeframe."""
        self._live_inputs().update_live_inputs_for_timeframe(timeframe)

    def run_test_forecast(
        self,
        timeframe: Optional[TimeFrame] = None,
    ) -> dict[str, Any]:
        """Run a manual forecast pass and return structured results."""
        requested_timeframes = [timeframe] if timeframe is not None else list(self.timeframes)
        timestamp = datetime.now(NY_TZ).isoformat()
        results: dict[str, Any] = {
            "timestamp": timestamp,
            "forecasts": {},
            "errors": [],
        }

        for requested_timeframe in requested_timeframes:
            try:
                self._update_live_inputs_for_timeframe(requested_timeframe)
                predictions = self._generate_portfolio_forecasts(requested_timeframe)
                results["forecasts"][requested_timeframe.name] = {
                    "count": len(predictions),
                    "predictions": predictions,
                }
            except Exception as exc:
                logger.error(
                    "Manual forecast failed for %s: %s",
                    requested_timeframe.name,
                    exc,
                    exc_info=True,
                )
                results["errors"].append(f"{requested_timeframe.name}: {exc}")
                results["forecasts"][requested_timeframe.name] = {
                    "count": 0,
                    "predictions": {},
                }

        return results
    
    def _add_candle(self, ticker: Ticker, timeframe: TimeFrame, candle: Candle) -> None:
        """Add a live candle to buffers, cache, MLManager, and volatility state."""
        self._live_inputs().add_candle(ticker, timeframe, candle)

    def _refresh_daily_volatility(self, ticker: Ticker) -> None:
        """Refresh incremental daily EWSD state for a ticker using latest daily candle."""
        self._live_inputs().refresh_daily_volatility(ticker)
    
    def _generate_portfolio_forecasts(self, timeframe: TimeFrame) -> Dict[str, float]:
        """
        Generate forecasts for all tickers using individual ensembles.
        
        Parameters
        ----------
        timeframe : TimeFrame
            Timeframe to forecast
            
        Returns
        -------
        Dict[str, float]
            Mapping of ticker name to forecast value (0-1)
        """
        return self._prediction_runtime().generate_portfolio_forecasts(timeframe)
    
    def _prepare_prediction_data(self, ticker: Ticker, ml_manager_key: tuple, n_samples: int) -> tuple:
        """
        Prepare ticker and volatility data for portfolio prediction.
        
        Parameters
        ----------
        ticker : Ticker
            Ticker enum
        ml_manager_key : tuple
            (ticker, timeframe) key for candle buffer
        n_samples : int
            Number of samples
            
        Returns
        -------
        tuple
            (ticker_series, volatility_series)
        """
        _ = ml_manager_key
        return self._prediction_runtime().prepare_prediction_data(ticker, n_samples)
    
    def _run_test_forecast(self) -> None:
        """Run a test forecast to verify system is working."""
        logger.info("🧪 Running test forecast...")
        
        # Just test one ticker/timeframe to avoid spam
        test_ticker = Ticker.EU
        test_timeframe = TimeFrame.D
        
        try:
            ml_manager_key = (test_ticker, test_timeframe)
            
            if ml_manager_key not in self.ml_managers:
                logger.warning(f"No test MLManager for {test_ticker.name} {test_timeframe.name}")
                return
            
            # Get latest candle
            latest_candle = self.mt5_connector.get_latest_candle(test_ticker.value, test_timeframe)
            
            if latest_candle is None:
                logger.warning(f"No test candle data for {test_ticker.name}")
                return
            
            # Add candle
            self._add_candle(test_ticker, test_timeframe, latest_candle)
            
            # Generate forecast
            forecasts = self._generate_portfolio_forecasts(test_timeframe)
            
            if test_ticker.name in forecasts:
                forecast = forecasts[test_ticker.name]
                logger.info(f"🧪 Test forecast {test_ticker.name}: {forecast:.4f}")
                
                # Send test status (less frequently)
                now = datetime.now()
                if now.minute % 15 == 0:  # Only every 15 minutes
                    self.telegram.send_status_update(
                        status="System operational",
                        details=f"Test forecast: {test_ticker.name} = {forecast:.4f}"
                    )
            else:
                logger.warning(f"🧪 Test forecast failed for {test_ticker.name}")
                
        except Exception as e:
            logger.error(f"🧪 Test forecast error: {e}")
    
    def load_historical_data(self, days_back: int = 60) -> None:
        """
        Load historical data for all MLManagers.
        
        ⚠️ IMPORTANT: This should only be called ONCE during initialization!
        
        This fetches enough historical candles to properly initialize all bias nodes
        with their required lookback periods.
        
        Parameters
        ----------
        days_back : int
            Number of days of historical data to load
        """
        logger.info(f"📚 Loading historical data for all MLManagers (INITIALIZATION ONLY)...")
        
        # Determine how many candles we need based on lookback_candles
        candles_needed = max(self.lookback_candles, days_back)
        
        for (ticker, timeframe), _ in self.ml_managers.items():
            try:
                logger.info(f"Loading {candles_needed} candles for {ticker.name} {timeframe.name}...")
                
                # Fetch historical candles
                candles = self.mt5_connector.get_historical_candles(
                    ticker.value, 
                    timeframe, 
                    count=candles_needed
                )
                
                if candles:
                    # Add all candles to buffer and MLManager
                    for candle in candles:
                        self._add_candle(ticker, timeframe, candle)
                    
                    logger.info(f"✅ Loaded {len(candles)} candles for {ticker.name} {timeframe.name}")
                else:
                    logger.warning(f"❌ No historical data for {ticker.name} {timeframe.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading history for {ticker.name} {timeframe.name}: {e}")
        
        logger.info("Historical data loading complete")

        # Pre-load cross-ticker data referenced in bias node params.
        self._load_cross_ticker_history()


    def _get_cross_tickers(self) -> set[tuple[Ticker, TimeFrame]]:
        """Discover cross-tickers from ensemble bias node specs."""
        return self._live_inputs().get_cross_tickers()

    def _load_cross_ticker_history(self) -> None:
        """Fetch historical data for cross-tickers from MT5 and load into store."""
        self._live_inputs().load_cross_ticker_history()

    def _upsert_cross_ticker_candle(self, ticker: Ticker, tf: TimeFrame, candle: Candle) -> None:
        """Insert or replace the latest candle for a cross-ticker in the store."""
        self._live_inputs().upsert_cross_ticker_candle(ticker, tf, candle)

    def _update_cross_ticker_latest(
        self,
        timeframe: TimeFrame,
        skip_tickers: Optional[set[Ticker]] = None,
    ) -> None:
        """Fetch latest candle for each cross-ticker and update the store."""
        self._live_inputs().update_cross_ticker_latest(
            timeframe,
            skip_tickers=skip_tickers,
        )
    
    def start(self) -> None:
        """Start the forecast server."""
        logger.info("🎬 Starting ForecastServer...")
        
        # Send startup notification
        self.telegram.send_status_update(
            status="ForecastServer starting",
            details=f"Managing {len(self.ensembles)} ensembles, {len(self.ml_managers)} MLManagers"
        )
        
        # Load historical data
        self.load_historical_data()
        
        # Test connections
        logger.info("🔌 Testing connections...")
        
        # Test MT5
        if self.mt5_connector.is_market_open():
            logger.info("✅ MT5 connection working")
        else:
            logger.warning("⚠️ MT5 connection issue or market closed")
        
        # Test Telegram
        if self.telegram.test_connection():
            logger.info("✅ Telegram connection working")
        else:
            logger.warning("⚠️ Telegram connection issue")
        
        # Send ready notification
        self.telegram.send_status_update(
            status="ForecastServer ready",
            details="All systems operational, monitoring market data"
        )
        
        logger.info("✅ ForecastServer started successfully")
        logger.info("📊 Scheduled forecasts:")
        logger.info("   - Daily: 00:00 EST")
        logger.info("   - Weekly: Sunday 18:00 EST")
        logger.info("🔄 Running scheduler loop...")
        
        # Main scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 ForecastServer stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f"💥 ForecastServer error: {e}")
            self.telegram.send_error_notification(str(e), "ForecastServer")
            raise
    
    def stop(self) -> None:
        """Stop the forecast server."""
        logger.info("🛑 Stopping ForecastServer...")
        
        # Close connections
        self.mt5_connector.shutdown()
        
        # Send shutdown notification
        self.telegram.send_status_update("ForecastServer stopped")
        
        logger.info("✅ ForecastServer stopped cleanly")
    
    def get_status(self) -> Dict:
        """Get server status information."""
        # Count ensembles per timeframe
        ensembles_by_tf = {}
        for (ticker, tf), ensemble in self.ensembles.items():
            if tf.name not in ensembles_by_tf:
                ensembles_by_tf[tf.name] = 0
            ensembles_by_tf[tf.name] += 1
        
        return {
            'ensembles': ensembles_by_tf,
            'ml_managers': len(self.ml_managers),
            'tickers': [t.name for t in self.tickers],
            'timeframes': [tf.name for tf in self.timeframes],
            'mt5_connected': self.mt5_connector.is_market_open(),
        }

# Main execution
if __name__ == '__main__':
    """
    Run forecast server as standalone application.
    """
    try:
        server = ForecastServer()
        server.start()
    except Exception as e:
        logger.error(f"💥 Failed to start ForecastServer: {e}")
        raise
