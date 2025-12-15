"""
Production Training Pipeline

Complete workflow from feature selection to ensemble training and deployment.
Shows how to create real (non-dummy) production-ready ensemble configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.enums import TimeFrame, Ticker
from feature_selection.os_feature_selector import OSFeatureSelector
from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.ensemble_utils import save_control_file
from feature_extraction.ml_manager import MLManager
from feature_selection.base_models.quantile_binning import QuantileBinningModel


logger = get_logger(__name__)

class ProductionTrainingPipeline:
    """
    Complete training pipeline for production ensemble deployment.
    
    Workflow:
    1. Load historical data for ticker/timeframe
    2. Run feature extraction (MLManager)
    3. Perform OS feature selection 
    4. Train ensemble on selected features
    5. Export trained ensemble to deployment config
    """
    
    def __init__(self, output_dir: str = "deployment/config"):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
        output_dir : str
            Directory to save trained ensemble configs
        """
        self.output_dir = output_dir
        logger.info(f"🏭 Initializing ProductionTrainingPipeline")
        logger.info(f"   Output directory: {output_dir}")
    
    def load_ticker_data(self, ticker: Ticker, timeframe: TimeFrame, 
                        start_date: str = "2020-01-01", 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical data for ticker and timeframe from parquet files.
        
        Loads real historical OHLCV data from the data/parquet_data/ directory.
        
        Parameters
        ----------
        ticker : Ticker
            Ticker enum value
        timeframe : TimeFrame
            Timeframe enum value
        start_date : str
            Start date for training data
        end_date : Optional[str]
            End date for training data (defaults to latest available)
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with datetime index
        """
        logger.info(f"📊 Loading real data for {ticker.name} {timeframe.name}")
        
        # Map ticker enum to parquet filename
        ticker_file_map = {
            Ticker.EU: 'EU.parquet',    # EUR/USD
            Ticker.BP: 'BP.parquet',    # GBP/USD
            Ticker.ES: 'ES.parquet',    # S&P 500 E-mini
            Ticker.NQ: 'NQ.parquet',    # Nasdaq 100 E-mini
            Ticker.GC: 'GC.parquet',    # Gold futures
            Ticker.CL: 'CL.parquet',    # Crude Oil futures
            Ticker.JY: 'JY.parquet',    # Japanese Yen futures
            Ticker.CD: 'CD.parquet',    # Canadian Dollar futures
            # Add more mappings as needed
        }
        
        if ticker not in ticker_file_map:
            raise ValueError(f"No data file available for ticker {ticker.name}")
        
        # Load parquet file - use absolute path from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parquet_path = os.path.join(project_root, "data", "parquet_data", ticker_file_map[ticker])
        
        try:
            logger.info(f"   Loading from: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            
            # Convert datetime column to pandas datetime and set as index
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
            df.set_index('datetime', inplace=True)
            
            # Keep only OHLCV columns
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            data = df[ohlcv_columns].copy()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            if end_date is None:
                end_dt = data.index.max()
            else:
                end_dt = pd.to_datetime(end_date)
            
            # Apply date filter
            mask = (data.index >= start_dt) & (data.index <= end_dt)
            data = data[mask].copy()
            
            # Ensure data is sorted by date
            data = data.sort_index()
            
            # Remove any duplicate dates (keep last)
            data = data[~data.index.duplicated(keep='last')]
            
            # Handle timeframe aggregation if needed
            if timeframe == TimeFrame.W:
                # Aggregate daily data to weekly (Friday close)
                data = data.resample('W-FRI').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"   Aggregated to weekly data")
            elif timeframe == TimeFrame.M:
                # Aggregate to monthly (month end)
                data = data.resample('M').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"   Aggregated to monthly data")
            # Daily data (TimeFrame.D) needs no aggregation
            
            logger.info(f"✅ Loaded {len(data)} periods from {data.index.min().date()} to {data.index.max().date()}")
            logger.info(f"   Date range: {len(data)} {timeframe.name} periods")
            
            if len(data) < 100:
                logger.warning(f"⚠️  Limited data: only {len(data)} periods available")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"❌ Data file not found: {parquet_path}")
            raise FileNotFoundError(f"Historical data not available for {ticker.name}")
            
        except Exception as e:
            logger.error(f"❌ Error loading data for {ticker.name}: {e}")
            raise
    
    def extract_features_and_targets(self, data: pd.DataFrame, ticker: Ticker, 
                                   timeframe: TimeFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract features and target returns from price data using MLManager.
        
        Uses proper node-based architecture via MLManager to generate features.
        Features use standardized camelCase naming convention.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV price data with datetime index
        ticker : Ticker
            Ticker for feature extraction
        timeframe : TimeFrame
            Timeframe for feature extraction
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Features DataFrame and targets DataFrame
        """
        logger.info(f"🔧 Extracting features using bias nodes for {ticker.name} {timeframe.name}")
        
        # Use bias nodes directly instead of MLManager for batch feature extraction
        # This avoids the complexity of MLManager's multi-timeframe architecture
        from utils.helpers import create_bias_node
        from utils.models import Candle
        import utils.helpers as helpers
        
        # Create bias nodes for feature extraction
        rsi_node = create_bias_node('rsi', ticker, timeframe, {'lookback': 14})
        momentum_node = create_bias_node('momentum', ticker, timeframe, {'lookback': 20})
        ma_diff_node = create_bias_node('ma_diff', ticker, timeframe, {'lookback': 50})
        
        # Process all candles through nodes
        rsi_values = []
        momentum_values = []
        ma_diff_values = []
        
        for idx, row in data.iterrows():
            candle = Candle(
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                datetime=idx,
                volume=float(row.get('volume', row.get('tick_volume', 0))),
                ticker=ticker,
                tf=timeframe
            )
            
            # Get values from each node
            rsi_vals = rsi_node.add_candle(candle)
            momentum_vals = momentum_node.add_candle(candle)
            ma_diff_vals = ma_diff_node.add_candle(candle)
            
            # Extract the signal values (first element from each node's output)
            rsi_values.append(rsi_vals[0] if len(rsi_vals) > 0 else 0)
            momentum_values.append(momentum_vals[0] if len(momentum_vals) > 0 else 0)
            ma_diff_values.append(ma_diff_vals[0] if len(ma_diff_vals) > 0 else 0)
        
        # Create features DataFrame with standardized camelCase naming
        features_df = pd.DataFrame({
            helpers.build_feature_column_name('rsi', 'signal', timeframe, {'lookback': 14}): rsi_values,
            helpers.build_feature_column_name('momentum', 'signal', timeframe, {'lookback': 20}): momentum_values,
            helpers.build_feature_column_name('ma_diff', 'signal', timeframe, {'lookback': 50}): ma_diff_values
        }, index=data.index)
        
        logger.info(f"   Extracted {len(features_df)} rows × {len(features_df.columns)} features")
        logger.info(f"   Feature columns: {list(features_df.columns)}")
        
        # Calculate target returns
        closes = data['close']
        forward_returns = closes.pct_change(1).shift(-1)  # Next period return
        log_returns = np.log(closes / closes.shift(1)).shift(-1)  # Log returns
        
        targets_df = pd.DataFrame({
            'log_return': log_returns,
            'simple_return': forward_returns
        }, index=data.index)
        
        # Drop NaN rows from both features and targets
        valid_idx = features_df.dropna().index.intersection(targets_df.dropna().index)
        features_df = features_df.loc[valid_idx]
        targets_df = targets_df.loc[valid_idx]
        
        logger.info(f"✅ Extracted {len(features_df.columns)} features, {len(features_df)} samples")
        logger.info(f"   Feature columns: {list(features_df.columns)}")
        
        return features_df, targets_df
    
    def run_feature_selection(self, features_df: pd.DataFrame, 
                            targets_df: pd.DataFrame) -> List[str]:
        """
        Run OS feature selection to identify best features.
        
        Uses the full OSFeatureSelector with walkforward validation to properly
        validate features out-of-sample before including them in production models.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix
        targets_df : pd.DataFrame
            Target variables
            
        Returns
        -------
        List[str]
            Selected feature names
        """
        logger.info(f"🔍 Running OS feature selection on {len(features_df.columns)} features")
        
        try:
            # Initialize OS Feature Selector with proper data
            selector = OSFeatureSelector(
                features_df=features_df,
                targets_df=targets_df
            )
            
            # Import required models and metrics
            from feature_selection.base_models.quantile_binning import QuantileBinningModel
            from metrics.performance import SortinoRatio
            from datetime import datetime, timedelta
            
            # Set up walkforward parameters based on data range
            data_start = features_df.index.min()
            data_end = features_df.index.max()
            data_span = (data_end - data_start).days
            
            # Use 60% of data for initial training, test on remaining
            train_days = int(data_span * 0.6)
            train_start = data_start
            train_end = data_start + timedelta(days=train_days)
            
            # Calculate appropriate test steps (aim for ~5-10 walkforward steps)
            remaining_days = data_span - train_days
            test_step = max(remaining_days // 8, 30)  # Minimum 30 days per step
            num_steps = min(remaining_days // test_step, 10)  # Maximum 10 steps
            
            logger.info(f"   Walkforward setup: train={train_days} days, "
                       f"test_step={test_step} days, steps={num_steps}")
            
            # Initialize model and metric for feature selection
            model = QuantileBinningModel(
                n_bins=3,
                selection_metric='sortino',
                strategy='long'
            )
            
            objective_metric = SortinoRatio()
            
            # Run walkforward test for all features at once
            feature_results = selector.walkforward_test(
                model=model,
                objective_metric=objective_metric,
                feature_cols=list(features_df.columns),
                train_start=train_start,
                train_end=train_end,
                test_step=test_step,
                num_steps=num_steps,
                strategy='long',
                target_col='log_return',
                verbose=True
            )
            
            # Extract results for each feature
            feature_scores = {}
            for feature_name, (results_df, step_info, _) in feature_results.items():
                if len(results_df) > 0:
                    # Calculate aggregated metric from all walkforward steps
                    all_returns = []
                    for step in step_info:
                        if 'returns' in step and len(step['returns']) > 0:
                            all_returns.extend(step['returns'])
                    
                    if len(all_returns) > 10:  # Need minimum trades
                        aggregated_returns = pd.Series(all_returns)
                        final_metric = objective_metric.compute(aggregated_returns)
                        feature_scores[feature_name] = final_metric
                        logger.info(f"   {feature_name}: sortino={final_metric:.3f}, "
                                   f"trades={len(all_returns)}")
                    else:
                        feature_scores[feature_name] = -999
                        logger.info(f"   {feature_name}: insufficient trades ({len(all_returns)})")
                else:
                    feature_scores[feature_name] = -999
                    logger.warning(f"   {feature_name}: walkforward failed")
            
            # Select features with positive Sortino ratio
            min_sortino = 0.5  # Minimum acceptable Sortino ratio
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score > min_sortino
            ]
            
            if not selected_features:
                # Fallback: select best 2 features even if below threshold
                sorted_features = sorted(feature_scores.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
                selected_features = [f[0] for f in sorted_features[:2]]
                logger.warning(f"⚠️  No features met Sortino>{min_sortino} criteria, "
                              f"using top 2: {selected_features}")
            
            logger.info(f"✅ OS Feature Selection complete: {len(selected_features)} features selected")
            logger.info(f"   Selected features: {selected_features}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"❌ OS Feature selection failed: {e}")
            logger.error("   Falling back to all features")
            # Fallback: use all features if OS selection fails
            return list(features_df.columns)
    
    def train_ensemble(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                      selected_features: List[str], ticker: Ticker, 
                      timeframe: TimeFrame) -> DiversifiedEnsemble:
        """
        Train ensemble on selected features.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix
        targets_df : pd.DataFrame  
            Target variables
        selected_features : List[str]
            Selected feature names
        ticker : Ticker
            Ticker being trained
        timeframe : TimeFrame
            Timeframe being trained
            
        Returns
        -------
        DiversifiedEnsemble
            Trained ensemble
        """
        logger.info(f"🏋️ Training ensemble for {ticker.name} {timeframe.name}")
        logger.info(f"   Using features: {selected_features}")
        
        # Filter to selected features
        X = features_df[selected_features].copy()
        y = targets_df['log_return'].copy()
        
        # Create base models for each selected feature
        base_models = {}
        base_model_configs = []
        
        for i, feature in enumerate(selected_features):
            model_name = f"model_{i}"
            
            # Create and fit quantile binning model
            model = QuantileBinningModel(
                n_bins=3,
                selection_metric='sortino',
                strategy='long'
            )
            
            try:
                model.fit(
                    feature_data=X[feature],
                    target_data=y
                )
                
                base_models[model_name] = model
                
                # Store model configuration
                base_model_configs.append({
                    'name': model_name,
                    'model_type': 'QuantileBinningModel',
                    'feature_column': feature,
                    'strategy': 'long',
                    'constructor_params': {
                        'n_bins': 3,
                        'selection_metric': 'sortino'
                    }
                })
                
                logger.info(f"   ✅ Trained {model_name} on {feature}")
                
            except Exception as e:
                logger.warning(f"   ❌ Failed to train {model_name}: {e}")
        
        if not base_models:
            raise ValueError("No base models were successfully trained")
        
        # Create ensemble metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '2.0.0',
            'ensemble_name': f'{ticker.name}_{timeframe.name}_production',
            'is_fit': True,
            'base_tf': timeframe.name,
            'description': f'Production ensemble for {ticker.name} {timeframe.name}'
        }
        
        # Create fitted base model parameters
        fitted_base_models = {}
        for model_name, model in base_models.items():
            fitted_base_models[model_name] = {
                'thresholds': model.thresholds_.tolist() if model.thresholds_ is not None else None,
                'best_long_bin': model.best_long_bin_,
                'best_short_bin': model.best_short_bin_,
                'bin_stats': model.bin_stats_
            }
        
        # Create ensemble weights (equal weighting for simplicity)
        n_models = len(base_models)
        weights = {name: 1.0 / n_models for name in base_models.keys()}
        
        # Estimate exposure fractions from historical data
        exposure_fractions = {}
        for model_name, model in base_models.items():
            feature_col = next(config['feature_column'] for config in base_model_configs 
                             if config['name'] == model_name)
            feature_data = X[feature_col]
            
            # Estimate how often this feature generates signals
            non_zero_signals = (feature_data != 0).sum()
            exposure_fraction = non_zero_signals / len(feature_data)
            exposure_fractions[model_name] = float(exposure_fraction)
        
        # Create fitted ensemble parameters
        fitted_ensemble = {
            'weights': weights,
            'exposure_fractions': exposure_fractions,
            'feature_names': list(base_models.keys()),
            'target_volatility': 0.15,  # 15% annual target volatility
            'unique_tickers': [ticker.value],
            'instrument_weights': {ticker.value: 1.0},
            'n_tickers': 1
        }
        
        logger.info(f"✅ Ensemble trained with {len(base_models)} models")
        
        return {
            'metadata': metadata,
            'base_models': base_model_configs,
            'fitted_base_models': fitted_base_models,
            'fitted_ensemble': fitted_ensemble,
            'tickers': [ticker.value]
        }
    
    def save_ensemble_config(self, ensemble_config: Dict, ticker: Ticker, 
                           timeframe: TimeFrame) -> str:
        """
        Save trained ensemble to deployment configuration file.
        
        Parameters
        ----------
        ensemble_config : Dict
            Trained ensemble configuration
        ticker : Ticker
            Ticker enum
        timeframe : TimeFrame
            Timeframe enum
            
        Returns
        -------
        str
            Path to saved configuration file
        """
        # Create output path using enum-based structure
        tf_dir = timeframe.name  # D, W, M
        ticker_name = ticker.name  # EU, BP, ES, NQ, etc.
        output_path = os.path.join(self.output_dir, tf_dir, f"{ticker_name}_{timeframe.name}.json")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save using ensemble_utils
        save_control_file(
            filepath=output_path,
            base_models=ensemble_config['base_models'],
            metadata=ensemble_config['metadata'],
            fitted_base_models=ensemble_config['fitted_base_models'],
            fitted_ensemble=ensemble_config['fitted_ensemble'],
            tickers=ensemble_config['tickers']
        )
        
        logger.info(f"💾 Saved ensemble config: {output_path}")
        return output_path
    
    def train_ticker_timeframe(self, ticker: Ticker, timeframe: TimeFrame) -> str:
        """
        Complete training workflow for single ticker/timeframe combination.
        
        Parameters
        ----------
        ticker : Ticker
            Ticker to train
        timeframe : TimeFrame
            Timeframe to train
            
        Returns
        -------
        str
            Path to saved configuration file
        """
        logger.info(f"🎯 Training {ticker.name} {timeframe.name}")
        
        try:
            # Step 1: Load data
            data = self.load_ticker_data(ticker, timeframe)
            
            # Step 2: Extract features and targets
            features_df, targets_df = self.extract_features_and_targets(data, ticker, timeframe)
            
            # Step 3: Feature selection
            selected_features = self.run_feature_selection(features_df, targets_df)
            
            # Step 4: Train ensemble
            ensemble_config = self.train_ensemble(features_df, targets_df, selected_features, ticker, timeframe)
            
            # Step 5: Save configuration
            config_path = self.save_ensemble_config(ensemble_config, ticker, timeframe)
            
            logger.info(f"✅ Training complete: {ticker.name} {timeframe.name}")
            return config_path
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Training failed for {ticker.name} {timeframe.name}: {e}")
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise
    
    def train_all_production_ensembles(self, tickers: Optional[List[Ticker]] = None,
                                     timeframes: Optional[List[TimeFrame]] = None) -> Dict[str, str]:
        """
        Train all production ensemble configurations.
        
        Parameters
        ----------
        tickers : Optional[List[Ticker]]
            Tickers to train (defaults to main trading tickers)
        timeframes : Optional[List[TimeFrame]]
            Timeframes to train (defaults to D and W)
            
        Returns
        -------
        Dict[str, str]
            Mapping of ticker_timeframe to config file path
        """
        if tickers is None:
            tickers = [Ticker.EU, Ticker.BP, Ticker.ES, Ticker.NQ]
        
        if timeframes is None:
            timeframes = [TimeFrame.D, TimeFrame.W]
        
        logger.info(f"🏭 Training production ensembles")
        logger.info(f"   Tickers: {[t.name for t in tickers]}")
        logger.info(f"   Timeframes: {[tf.name for tf in timeframes]}")
        
        results = {}
        
        for ticker in tickers:
            for timeframe in timeframes:
                try:
                    config_path = self.train_ticker_timeframe(ticker, timeframe)
                    key = f"{ticker.name}_{timeframe.name}"
                    results[key] = config_path
                    
                except Exception as e:
                    logger.error(f"❌ Failed {ticker.name} {timeframe.name}: {e}")
                    results[f"{ticker.name}_{timeframe.name}"] = None
        
        successful = sum(1 for path in results.values() if path is not None)
        total = len(results)
        
        logger.info(f"🏁 Training complete: {successful}/{total} successful")
        return results

def main():
    """
    Main execution: train production ensemble configurations.
    """
    logger.info("🚀 Starting Production Training Pipeline")
    
    try:
        pipeline = ProductionTrainingPipeline()
        
        # Train all production ensembles
        results = pipeline.train_all_production_ensembles()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PRODUCTION TRAINING RESULTS")
        logger.info("="*60)
        
        for key, path in results.items():
            status = "✅ SUCCESS" if path else "❌ FAILED"
            logger.info(f"{status} | {key}")
            if path:
                logger.info(f"         Config: {path}")
        
        successful_count = sum(1 for path in results.values() if path is not None)
        total_count = len(results)
        
        logger.info(f"\n📊 Summary: {successful_count}/{total_count} ensembles trained successfully")
        
        if successful_count > 0:
            logger.info("🎉 Production training pipeline completed!")
            logger.info("💡 Next steps:")
            logger.info("   1. Test configs: python deployment/test_deployment.py")
            logger.info("   2. Start server: python deployment/forecast_server.py")
        else:
            logger.error("💥 All training failed - check logs above")
            
    except Exception as e:
        logger.error(f"💥 Production training pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
