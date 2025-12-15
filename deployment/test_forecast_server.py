"""
Test Forecast Server

Extends ForecastServer with testing capabilities. Separates test logic from
production code for cleaner architecture.
"""

import sys
import os
from typing import Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.forecast_server import ForecastServer
from utils.enums import TimeFrame
from utils.logger import get_logger
from zoneinfo import ZoneInfo

logger = get_logger(__name__)
NY_TZ = ZoneInfo("America/New_York")


class TestForecastServer(ForecastServer):
    """
    Test wrapper for ForecastServer that adds testing capabilities
    without polluting production code.
    """
    
    def run_test_forecast(self, timeframe: Optional[TimeFrame] = None) -> Dict:
        """
        Run a test forecast immediately without waiting for schedule.
        
        Parameters
        ----------
        timeframe : TimeFrame, optional
            Specific timeframe to test (D or W). If None, tests both.
            
        Returns
        -------
        Dict
            Dictionary containing forecast results and status information
        """
        logger.info("🧪 Running test forecast...")
        
        results = {
            'timestamp': datetime.now(NY_TZ).isoformat(),
            'forecasts': {},
            'errors': []
        }
        
        # Determine which timeframes to test
        timeframes_to_test = []
        if timeframe is not None:
            # Check if we have any ensembles for this timeframe
            has_ensemble = any(tf == timeframe for (_, tf) in self.ensembles.keys())
            if has_ensemble:
                timeframes_to_test = [(timeframe, timeframe.name)]
        else:
            # Test all available timeframes
            for tf in [TimeFrame.D, TimeFrame.W]:
                # Check if we have any ensembles for this timeframe
                has_ensemble = any(timeframe_key == tf for (_, timeframe_key) in self.ensembles.keys())
                if has_ensemble:
                    timeframes_to_test.append((tf, tf.name))
        
        # Run forecasts for each timeframe
        for tf, tf_name in timeframes_to_test:
            try:
                print(f"🔮 Testing {tf_name} forecasts...")
                logger.info(f"🔮 Testing {tf_name} forecasts...")
                
                # Get latest data and run forecasts
                print(f"  Generating forecasts for {tf_name}...")
                forecasts = self._generate_portfolio_forecasts(tf)
                print(f"  Generated {len(forecasts)} forecasts")
                
                results['forecasts'][tf_name] = {
                    'count': len(forecasts),
                    'tickers': [f['ticker'] for f in forecasts],
                    'predictions': forecasts
                }
                
                print(f"✅ {tf_name} test complete: {len(forecasts)} forecasts")
                logger.info(f"✅ {tf_name} test complete: {len(forecasts)} forecasts generated")
                
            except Exception as e:
                error_msg = f"Error testing {tf_name}: {e}"
                print(f"❌ {error_msg}")
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                import traceback
                traceback.print_exc()
        
        # Summary
        total_forecasts = sum(r['count'] for r in results['forecasts'].values())
        logger.info(f"🏁 Test forecast complete: {total_forecasts} total forecasts, {len(results['errors'])} errors")
        
        return results
