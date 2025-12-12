"""
Forecast Server

Production deployment server that uses Portfolio to manage multiple ensembles.
Generates forecasts for multiple ticker/timeframe combinations efficiently.
"""

import os
import sys
import time
import schedule
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.enums import TimeFrame, Ticker
from utils.models import Candle
import utils.helpers as helpers
from deployment.mt5_data_connector import ForecastMT5DataConnector
from deployment.telegram_notifier import TelegramNotifier
from ensemble.portfolio import Portfolio
from feature_extraction.ml_manager import MLManager

logger = get_logger(__name__)
NY_TZ = ZoneInfo("America/New_York")

class ForecastServer:
    """
    Production forecasting server.
    
    Uses Portfolio class to manage multiple ensembles efficiently.
    Generates one forecast per ticker/timeframe combination.
    
    Architecture:
    - One Portfolio per timeframe (Daily, Weekly, etc.)
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
        self.portfolios: Dict[TimeFrame, Portfolio] = {}
        
        # MLManagers - one per ticker/timeframe for feature extraction
        # We need separate MLManagers per ticker because each ticker has different price data
        self.ml_managers: Dict[tuple, MLManager] = {}  # (ticker, timeframe) -> MLManager
        
        # Candle buffers for each ticker/timeframe
        self.candle_buffers: Dict[tuple, List[Candle]] = {}  # (ticker, timeframe) -> [candles]
        self.lookback_candles = 50
        
        # Configuration
        self.tickers = [Ticker.EU, Ticker.BP, Ticker.ES, Ticker.NQ]
        self.timeframes = [TimeFrame.D, TimeFrame.W]  # Daily and Weekly
        
        # Initialize portfolios and MLManagers
        self._setup_portfolios()
        self._setup_ml_managers()
        
        # Setup scheduling
        self._setup_scheduling()
        
        logger.info(f"✅ ForecastServer initialized")
        logger.info(f"   Portfolios: {len(self.portfolios)}")
        logger.info(f"   MLManagers: {len(self.ml_managers)}")
    
    def _setup_portfolios(self) -> None:
        """Setup portfolio for each timeframe."""
        logger.info("Setting up portfolios...")
        
        for timeframe in self.timeframes:
            try:
                # Each timeframe has its own subdirectory (e.g., config/D/, config/W/)
                tf_dir = os.path.join(self.config_dir, timeframe.name)
                
                if not os.path.exists(tf_dir):
                    logger.warning(f"Timeframe directory not found: {tf_dir}")
                    continue
                
                # Create portfolio for this timeframe
                # Portfolio will load ALL control files in the directory
                portfolio = Portfolio(
                    control_file_dir=tf_dir,
                    is_fit=True,  # Load fitted ensembles
                    ticker=None,  # Will infer from control files
                    base_tf=timeframe
                )
                
                if len(portfolio.ensembles) == 0:
                    logger.warning(f"No ensembles loaded for timeframe {timeframe.name}")
                    continue
                
                self.portfolios[timeframe] = portfolio
                logger.info(f"✅ Setup portfolio for {timeframe.name}: {len(portfolio.ensembles)} ensembles")
                
            except Exception as e:
                logger.error(f"❌ Failed to setup portfolio for {timeframe.name}: {e}")
        
        logger.info(f"Setup complete: {len(self.portfolios)} portfolios")
    
    def _setup_ml_managers(self) -> None:
        """
        Setup MLManagers for feature extraction.
        
        Creates one MLManager per ticker/timeframe combination.
        Each MLManager gets the bias nodes required by the portfolio for that timeframe.
        """
        logger.info("Setting up MLManagers...")
        
        for timeframe, portfolio in self.portfolios.items():
            # Get all bias nodes needed by this portfolio
            bias_node_specs = portfolio.get_required_bias_nodes()
            
            logger.info(f"Portfolio {timeframe.name} requires {len(bias_node_specs)} bias node types")
            
            # Create MLManager for each ticker (each ticker needs its own data)
            for ticker in self.tickers:
                try:
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
        
        # Check if we have a portfolio for this timeframe
        if timeframe not in self.portfolios:
            logger.warning(f"No portfolio for timeframe {timeframe.name}")
            return
        
        portfolio = self.portfolios[timeframe]
        
        # Check market status
        market_open = self.mt5_connector.is_market_open()
        market_status = "Open" if market_open else "Closed"
        logger.info(f"Market status: {market_status}")
        
        # Get latest candles and update MLManagers
        for ticker in self.tickers:
            try:
                ml_manager_key = (ticker, timeframe)
                
                if ml_manager_key not in self.ml_managers:
                    logger.warning(f"No MLManager for {ticker.name} {timeframe.name}")
                    continue
                
                # Get latest candle
                latest_candle = self.mt5_connector.get_latest_candle(ticker.value, timeframe)
                
                if latest_candle is None:
                    logger.warning(f"No candle data for {ticker.name}")
                    continue
                
                # Add candle to buffer and MLManager
                self._add_candle(ticker, timeframe, latest_candle)
                
            except Exception as e:
                logger.error(f"❌ Error updating {ticker.name}: {e}")
        
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
    
    def _add_candle(self, ticker: Ticker, timeframe: TimeFrame, candle: Candle) -> None:
        """
        Add candle to buffer and MLManager.
        
        Parameters
        ----------
        ticker : Ticker
            Ticker symbol
        timeframe : TimeFrame
            Timeframe
        candle : Candle
            Market candle
        """
        key = (ticker, timeframe)
        
        # Add to buffer
        if key not in self.candle_buffers:
            self.candle_buffers[key] = []
        
        self.candle_buffers[key].append(candle)
        
        # Keep only required number of candles
        if len(self.candle_buffers[key]) > self.lookback_candles:
            self.candle_buffers[key] = self.candle_buffers[key][-self.lookback_candles:]
        
        # Add to MLManager
        if key in self.ml_managers:
            ml_manager = self.ml_managers[key]
            ml_manager.add_candle(candle, timeframe)
            
            logger.debug(f"Added candle to {ticker.name} {timeframe.name}: {candle.datetime}")
    
    def _generate_portfolio_forecasts(self, timeframe: TimeFrame) -> Dict[str, float]:
        """
        Generate forecasts for all tickers using portfolio.
        
        Parameters
        ----------
        timeframe : TimeFrame
            Timeframe to forecast
            
        Returns
        -------
        Dict[str, float]
            Mapping of ticker name to forecast value (0-1)
        """
        forecasts = {}
        
        if timeframe not in self.portfolios:
            return forecasts
        
        portfolio = self.portfolios[timeframe]
        
        # Generate forecast for each ticker
        for ticker in self.tickers:
            try:
                ml_manager_key = (ticker, timeframe)
                
                if ml_manager_key not in self.ml_managers:
                    continue
                
                ml_manager = self.ml_managers[ml_manager_key]
                
                # Check if we have enough data
                if len(self.candle_buffers.get(ml_manager_key, [])) < 20:
                    logger.warning(f"Insufficient candles for {ticker.name}: {len(self.candle_buffers.get(ml_manager_key, []))}/20")
                    continue
                
                # Get features from MLManager
                features_df = ml_manager.matrix_df
                
                if features_df is None or features_df.empty:
                    logger.warning(f"No features for {ticker.name}")
                    continue
                
                # Get latest row of features
                latest_features = features_df.iloc[[-1]]
                
                # Prepare ticker and volatility series
                ticker_series, volatility_series = self._prepare_prediction_data(
                    ticker, 
                    ml_manager_key,
                    len(latest_features)
                )
                
                # Generate prediction using portfolio
                # Portfolio.predict expects features for a specific timeframe
                predictions = portfolio.predict(
                    X=latest_features,
                    ticker=ticker_series,
                    volatility=volatility_series,
                    timeframe=timeframe
                )
                
                if len(predictions) == 0:
                    logger.warning(f"No predictions for {ticker.name}")
                    continue
                
                # Extract forecast value (portfolio returns DataFrame with %_to_risk column)
                forecast = float(predictions['%_to_risk'].iloc[0])
                
                # Normalize to 0-1 range if needed (portfolio already returns position sizes)
                # The ensemble predict() method returns values in a specific range
                # We may want to apply sigmoid normalization for consistency
                forecast_normalized = 1.0 / (1.0 + np.exp(-forecast * 2))
                
                forecasts[ticker.name] = forecast_normalized
                logger.info(f"✅ {ticker.name}: {forecast_normalized:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error forecasting {ticker.name}: {e}")
                logger.exception("Full traceback:")
        
        return forecasts
    
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
        import pandas as pd
        import numpy as np
        
        # Create ticker series - use ticker value
        ticker_value = ticker.value
        ticker_series = pd.Series([ticker_value] * n_samples)
        
        # Calculate volatility from recent candles
        candles = self.candle_buffers.get(ml_manager_key, [])
        
        if len(candles) >= 20:
            returns = []
            for i in range(1, min(21, len(candles))):
                ret = np.log(candles[-i].close / candles[-i-1].close)
                returns.append(ret)
            
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0.15  # Default to 15%
        
        volatility_series = pd.Series([volatility] * n_samples)
        
        return ticker_series, volatility_series
    
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
    
    def load_historical_data(self, days_back: int = 30) -> None:
        """
        Load historical data for all MLManagers.
        
        Parameters
        ----------
        days_back : int
            Number of days of historical data to load
        """
        logger.info(f"📚 Loading {days_back} days of historical data...")
        
        for (ticker, timeframe), ml_manager in self.ml_managers.items():
            try:
                logger.info(f"Loading history for {ticker.name} {timeframe.name}...")
                
                # This is simplified - in reality we'd need to fetch historical candles
                # For now, we'll just add the latest candle to initialize
                latest_candle = self.mt5_connector.get_latest_candle(ticker.value, timeframe)
                
                if latest_candle is not None:
                    self._add_candle(ticker, timeframe, latest_candle)
                    logger.info(f"✅ Initialized {ticker.name} {timeframe.name}")
                else:
                    logger.warning(f"❌ No data for {ticker.name} {timeframe.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading history for {ticker.name} {timeframe.name}: {e}")
        
        logger.info("📚 Historical data loading complete")
    
    def start(self) -> None:
        """Start the forecast server."""
        logger.info("🎬 Starting ForecastServer...")
        
        # Send startup notification
        self.telegram.send_status_update(
            status="ForecastServer starting",
            details=f"Managing {len(self.portfolios)} portfolios, {len(self.ml_managers)} MLManagers"
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
        return {
            'portfolios': {
                tf.name: len(portfolio.ensembles)
                for tf, portfolio in self.portfolios.items()
            },
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
    import numpy as np  # Need numpy for normalization
    
    try:
        server = ForecastServer()
        server.start()
    except Exception as e:
        logger.error(f"💥 Failed to start ForecastServer: {e}")
        raise
