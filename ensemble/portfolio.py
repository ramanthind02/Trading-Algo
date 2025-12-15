"""
Portfolio Class for Managing Multiple Ensembles

This module provides a Portfolio class that manages multiple DiversifiedEnsemble
instances, coordinates feature extraction via MLManager, filters data by timeframe
and required columns, and combines predictions into a single DataFrame output.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from utils.enums import TimeFrame, Ticker
import utils.helpers as helpers

from .diversified_ensemble import DiversifiedEnsemble
from .ensemble_utils import (
    parse_control_file,
    extract_bias_node_specs_from_control_file,
    aggregate_bias_node_specs_from_directory
)


class Portfolio:
    """
    Portfolio class that manages multiple DiversifiedEnsemble instances.
    
    The Portfolio coordinates feature extraction, filters data by timeframe
    and required columns for each ensemble, and combines predictions into
    a single output DataFrame.
    
    Parameters
    ----------
    feature_list_dir : str
        Path to directory containing feature_list JSON files (one per ensemble)
    ensemble_model_dir : str, optional
        Path to directory containing ensemble_model JSON files (for fitted models).
        If provided, ensembles will be loaded from these files instead of feature_list files.
    ticker : Ticker, optional
        Ticker symbol for MLManager initialization. If None, will try to infer
        from feature_list files or use first ticker found.
    base_tf : TimeFrame, optional
        Base timeframe for MLManager. If None, will use TimeFrame.D as default.
    """
    
    def __init__(
        self,
        control_file_dir: str,
        is_fit: bool = False,
        ticker: Optional[Ticker] = None,
        base_tf: TimeFrame = TimeFrame.D
    ):
        """
        Initialize Portfolio from control file directory.
        
        Parameters
        ----------
        control_file_dir : str
            Path to directory containing control file JSON files (one per ensemble)
        is_fit : bool, default=False
            If True, only load control files with is_fit=True (fitted ensembles).
            If False, only load control files with is_fit=False (unfitted ensembles).
        ticker : Ticker, optional
            Ticker symbol for MLManager initialization. If None, will try to infer
            from control files or use first ticker found.
        base_tf : TimeFrame, optional
            Base timeframe for MLManager. If None, will use TimeFrame.D as default.
        """
        self.control_file_dir = control_file_dir
        self.is_fit = is_fit
        self.ticker = ticker
        self.base_tf = base_tf
        
        # Dictionary mapping ensemble names to (ensemble, base_tf) tuples
        self.ensembles: Dict[str, tuple] = {}
        
        # Initialize ensembles from directory
        self._initialize_ensembles()
    
    def _initialize_ensembles(self) -> None:
        """
        Initialize all ensembles from control file directory.
        Only loads control files matching the is_fit flag.
        """
        if not os.path.isdir(self.control_file_dir):
            raise FileNotFoundError(f"Control file directory not found: {self.control_file_dir}")
        
        # Find all JSON files in directory
        for filename in os.listdir(self.control_file_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.control_file_dir, filename)
            try:
                # Parse control file to get ensemble name and base_tf
                from .ensemble_utils import parse_control_file
                control_file = parse_control_file(filepath)
                metadata = control_file.get('metadata', {})
                
                # Check if is_fit flag matches
                file_is_fit = metadata.get('is_fit', False)
                if file_is_fit != self.is_fit:
                    # Skip files that don't match the desired is_fit state
                    continue
                
                # Get ensemble name from metadata or filename
                ensemble_name = metadata.get('ensemble_name', os.path.splitext(filename)[0])
                
                # Get base_tf from metadata or parse from filename
                base_tf = None
                if 'base_tf' in metadata:
                    tf_str = metadata['base_tf']
                    try:
                        base_tf = TimeFrame[tf_str]
                    except (KeyError, AttributeError):
                        pass
                
                # If not in metadata, try to parse from filename (e.g., "indices_D.json")
                if base_tf is None:
                    name_parts = os.path.splitext(filename)[0].split('_')
                    if len(name_parts) > 1:
                        tf_str = name_parts[-1]
                        try:
                            base_tf = TimeFrame[tf_str]
                        except (KeyError, AttributeError):
                            base_tf = TimeFrame.D  # Default
                    else:
                        base_tf = TimeFrame.D  # Default
                
                # Initialize ensemble
                ensemble = DiversifiedEnsemble(
                    control_file_path=filepath,
                    base_tf=base_tf
                )
                
                # Store ensemble
                self.ensembles[ensemble_name] = (ensemble, base_tf)
                
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                # Skip files that aren't valid control files
                continue
    
    def get_required_bias_nodes(self) -> List[Dict[str, Any]]:
        """
        Get list of bias node specifications needed by all ensembles.
        
        Aggregates bias node specs from all ensembles in the portfolio.
        Each ensemble returns its required bias nodes, and they are combined
        and deduplicated. Specs with the same module_name and params but different
        timeframes are merged into a single spec with multiple timeframes.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of bias node specifications in format:
            [{'module_name': str, 'timeframes': [TimeFrame], 'params': dict}, ...]
        """
        # Use a dict to merge specs with same module_name and params
        # Key: (module_name, tuple of sorted params)
        # Value: set of timeframes
        merged_specs = {}
        
        # Get bias node specs from each ensemble
        for ensemble_name, (ensemble, _) in self.ensembles.items():
            ensemble_specs = ensemble.get_required_bias_nodes()
            
            # Process each spec
            for spec in ensemble_specs:
                module_name = spec['module_name']
                timeframes = spec['timeframes']
                params = spec['params']
                
                # Create key for merging (module_name + params, ignoring timeframes)
                params_key = tuple(sorted(params.items()))
                merge_key = (module_name, params_key)
                
                # Initialize or update the merged spec
                if merge_key not in merged_specs:
                    merged_specs[merge_key] = {
                        'module_name': module_name,
                        'timeframes': set(),
                        'params': params
                    }
                
                # Add timeframes to the set
                for tf in timeframes:
                    if isinstance(tf, TimeFrame):
                        merged_specs[merge_key]['timeframes'].add(tf)
                    else:
                        # Convert string to TimeFrame if needed
                        try:
                            merged_specs[merge_key]['timeframes'].add(TimeFrame[tf])
                        except (KeyError, AttributeError):
                            pass
        
        # Convert sets to sorted lists for consistent output
        result = []
        for merge_key, spec in merged_specs.items():
            # Sort timeframes for consistency
            sorted_tfs = sorted(spec['timeframes'], key=lambda tf: tf.value if hasattr(tf, 'value') else str(tf))
            result.append({
                'module_name': spec['module_name'],
                'timeframes': sorted_tfs,
                'params': spec['params']
            })
        
        return result
    
    def fit(
        self,
        X: pd.DataFrame,
        ticker: pd.Series,
        volatility: pd.Series,
        y: pd.Series,
        normalization_data: Optional[pd.DataFrame] = None
    ) -> 'Portfolio':
        """
        Fit all ensembles in the portfolio.
        
        Passes the full DataFrame to each ensemble, which will filter
        by timeframe and required columns internally.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with all features (all timeframes)
        ticker : pd.Series
            Ticker symbols for each sample
        volatility : pd.Series
            Volatility values for each sample
        y : pd.Series
            Target values (returns) for training
        normalization_data : pd.DataFrame, optional
            Normalization data (EWSD/ATR) for each feature column
            
        Returns
        -------
        self
            Fitted portfolio
            
        Raises
        ------
        ValueError
            If portfolio was initialized with is_fit=True (cannot fit already fitted ensembles)
        """
        if self.is_fit:
            raise ValueError(
                "Cannot fit portfolio initialized with is_fit=True. "
                "Initialize with is_fit=False to fit ensembles."
            )
        
        # Fit each ensemble - let ensembles filter data themselves
        for ensemble_name, (ensemble, _) in self.ensembles.items():
            ensemble.fit(
                X=X,
                ticker=ticker,
                volatility=volatility,
                y=y,
                normalization_data=normalization_data
            )
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        ticker: pd.Series,
        volatility: pd.Series,
        timeframe: TimeFrame,
        normalization_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate predictions from ensembles matching the given timeframe and average them.
        
        MLManager feeds features one timeframe at a time. This method only processes
        ensembles that match the provided timeframe. Predictions from matching ensembles
        are averaged to produce a single prediction per sample.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with features for the current timeframe only.
            MLManager provides features for one timeframe at a time.
        ticker : pd.Series
            Ticker symbols for each sample
        volatility : pd.Series
            Volatility values for each sample
        timeframe : TimeFrame
            Current timeframe being processed. Only ensembles matching this timeframe
            will be used for prediction.
        normalization_data : pd.DataFrame, optional
            Normalization data (EWSD/ATR) for each feature column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [ticker, %_to_risk]
            Each row represents the averaged prediction across matching ensembles for one sample.
        """
        if not self.ensembles:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['ticker', '%_to_risk'])
        
        # Collect predictions from ensembles matching the timeframe
        ensemble_predictions = []
        
        # DEBUG LOGGING
        from utils.logger import logger
        logger.info(f"🔍 Portfolio.predict called:")
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   X columns: {list(X.columns)}")
        logger.info(f"   ticker type: {type(ticker)}")
        logger.info(f"   ticker values: {ticker.tolist() if hasattr(ticker, 'tolist') else ticker}")
        logger.info(f"   timeframe: {timeframe.name}")
        logger.info(f"   Total ensembles: {len(self.ensembles)}")
        
        for ensemble_name, (ensemble, ensemble_tf) in self.ensembles.items():
            # Only process ensembles that match the current timeframe
            if ensemble_tf != timeframe:
                continue
            
            logger.info(f"   Processing ensemble: {ensemble_name}, TF: {ensemble_tf.name}")
            logger.info(f"      Ensemble unique_tickers: {getattr(ensemble, 'unique_tickers_', 'NOT SET')}")
            
            # Get predictions from this ensemble
            predictions = ensemble.predict(
                X=X,
                ticker=ticker,
                volatility=volatility,
                normalization_data=normalization_data
            )
            ensemble_predictions.append(predictions)
        
        # If no matching ensembles, return empty DataFrame
        if not ensemble_predictions:
            return pd.DataFrame(columns=['ticker', '%_to_risk'])
        
        # Stack predictions: each row is a sample, each column is an ensemble
        predictions_matrix = np.column_stack(ensemble_predictions)
        
        # Average across ensembles (axis=1 means average across columns/ensembles)
        averaged_predictions = np.mean(predictions_matrix, axis=1)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ticker': ticker.values if isinstance(ticker, pd.Series) else ticker,
            '%_to_risk': averaged_predictions
        })
        
        return result
    
    def save_control_files(self, output_dir: str) -> Dict[str, str]:
        """
        Save all fitted ensemble control files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save control files
            
        Returns
        -------
        Dict[str, str]
            Mapping of ensemble names to saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        for ensemble_name, (ensemble, base_tf) in self.ensembles.items():
            if not ensemble.is_fitted_:
                continue
            
            # Create filename: {ensemble_name}_{timeframe}.json
            filename = f"{ensemble_name}_{base_tf.name}.json"
            filepath = os.path.join(output_dir, filename)
            
            ensemble.save_control_file(filepath)
            saved_paths[ensemble_name] = filepath
        
        return saved_paths
    
    def __repr__(self) -> str:
        """String representation of the portfolio."""
        return f"Portfolio(n_ensembles={len(self.ensembles)}, base_tf={self.base_tf.name})"
    
    def __str__(self) -> str:
        """Detailed string description of the portfolio."""
        lines = [
            f"Portfolio",
            f"  Ensembles: {len(self.ensembles)}",
            f"  Base Timeframe: {self.base_tf.name}",
            f"  Ensemble Names: {list(self.ensembles.keys())}"
        ]
        return "\n".join(lines)

