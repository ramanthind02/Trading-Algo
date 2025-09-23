import polars as pl
from typing import Dict, List
from datetime import datetime
from machine_learning.data_collection.feature_extraction import FeatureExtractor
from utils.helpers import load_numpy_data
from utils.enums import TimeFrame
import numpy as np
import gc
from backtest import Backtest
from utils.enums import TimeFrame 
import utils.helpers as helpers
import traceback

class BiasFeatureExtractor(FeatureExtractor):
    """
    Specialized extractor for bias features.
    
    Handles the extraction of features related to market bias across multiple timeframes.
    """
    
    def extract_bias(
            self,
            timeframes: List[TimeFrame],
            name: str = ""
    ) -> Dict[TimeFrame, Dict]:
        """
        Extract bias features across multiple timeframes.
        
        Parameters
        ----------
        timeframes : List[TimeFrame]
            List of timeframes to process
        name : str
            Name identifier for the features
            
        Returns
        -------
        Dict[TimeFrame, Dict]
            Dictionary with keys:
            - 'features': Processed features DataFrame
            - 'lambdas': Feature lambdas for normalization
            - 'pipeline': Scikit-learn pipeline for future transformations
        """
        bias_map = {}
        start_time = datetime.now()

        for tf in timeframes:
            print(f"Processing timeframe {tf}")
            tf_start_time = datetime.now()

            data = load_numpy_data(self.ticker, tf, start=self.DEFAULT_START_DATE, end=self.DEFAULT_END_DATE)
            
            try:
                bias_map[tf] = {}
                # Process one timeframe at a time
                features_df, lambdas, pipeline = self._process_single_timeframe(tf, data, name)
                if features_df is not None:
                    bias_map[tf]['features'] = features_df
                    bias_map[tf]['lambdas'] = lambdas
                    bias_map[tf]['pipeline'] = pipeline
                    print(f"Created pipeline for timeframe {tf}")
                
                tf_duration = (datetime.now() - tf_start_time).total_seconds()
                print(f"Completed timeframe {tf} in {tf_duration:.2f} seconds")
                
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"\nError processing timeframe {tf}: {e}")
                print(f"\nDetailed error traceback:\n{error_traceback}")
                continue
            finally:
                gc.collect()

        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"Total bias feature extraction completed in {total_duration:.2f} seconds")
        return bias_map
    
    def _process_single_timeframe(
            self, 
            tf: TimeFrame, 
            data: np.ndarray, 
            name: str 
    ) -> tuple[pl.DataFrame, dict, object]:
        """
        Process a single timeframe for bias feature extraction.
        
        Parameters
        ----------
        tf : TimeFrame
            Timeframe to process
        data : np.ndarray
            Numpy data for the timeframe
        name : str
            Name identifier for the features
            
        Returns
        -------
        tuple[pl.DataFrame, dict, object]
            Tuple containing:
            - Processed and normalized features dataframe
            - Lambdas dictionary for feature normalization
            - Complete scikit-learn pipeline for future use
        """
        # Build middleman and run backtest

        middleman = helpers.build_bias_middleman(self.ticker, tf)
        middleman.bias_manager.ml_managers[0].build_matrix = True
        features_backtest = Backtest("features", data, middleman, self.candle_fetcher, tf)
        features_backtest.run()
        
        # Extract features and clear middleman
        features = middleman.bias_manager.ml_managers[0].matrix
        columns = middleman.bias_manager.ml_managers[0].columns
        del middleman, features_backtest
        gc.collect()
        
        return features, columns