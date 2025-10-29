"""
FeatureExtractor Class - Centralized Feature Extraction

This module provides a high-level class for extracting features from bias nodes
across multiple tickers and parameter configurations. It follows the Single
Responsibility Principle by handling ONLY feature extraction, returning clean
dataframes for downstream analysis.

The FeatureExtractor class:
- Manages feature extraction from multiple bias nodes
- Handles multiple tickers with automatic alignment
- Supports parameter grid exploration
- Returns organized dataframes (per-module and combined)
- Decouples extraction logic from analysis (FeatureExplorer, FeatureSelector)

Author: Trading Research Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime as dt
from utils.enums import Ticker, TimeFrame
from feature_extraction.feature_extractor import (
    extract_features_from_bias_node,
    compute_target_columns
)
import utils.helpers as helpers


class FeatureExtractor:
    """
    High-level feature extraction class for systematic feature engineering.
    
    This class provides a clean interface for extracting features from multiple
    bias nodes across different tickers and parameter configurations. It returns
    organized dataframes that can be directly consumed by FeatureExplorer and
    FeatureSelector without any knowledge of the extraction process.
    
    Key Features:
    - Extract from single or multiple tickers
    - Extract from single or multiple bias nodes
    - Support parameter grid exploration
    - Return per-module dataframes + combined dataframe
    - Automatic alignment and target computation
    - Clean separation of concerns
    
    Parameters
    ----------
    tickers : Ticker or List[Ticker]
        Single ticker or list of tickers to extract features for
    start : datetime, optional
        Start date for extraction. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date for extraction. Defaults to datetime.now()
    use_millisecond_offset : bool, default=True
        For multi-ticker: add millisecond offsets to avoid duplicate indices
        
    Attributes
    ----------
    tickers : List[Ticker]
        List of tickers being processed
    start : datetime
        Start date for extraction
    end : datetime
        End date for extraction
    use_millisecond_offset : bool
        Whether to use millisecond offsets for multi-ticker
    features_by_module : Dict[str, pd.DataFrame]
        Dictionary mapping module names to their feature dataframes
    targets_by_module : Dict[str, pd.DataFrame]
        Dictionary mapping module names to their target dataframes
    combined_features : pd.DataFrame or None
        Combined dataframe with all features from all modules
    combined_targets : pd.DataFrame or None
        Combined dataframe with all targets (same across modules)
        
    Examples
    --------
    >>> # Single ticker, single module
    >>> extractor = FeatureExtractor(ticker=Ticker.SPY)
    >>> features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
    ...     modules={'rsi': {'lookback': [14, 21]}}
    ... )
    >>> 
    >>> # Multiple tickers, multiple modules
    >>> extractor = FeatureExtractor(ticker=[Ticker.ES, Ticker.NQ, Ticker.YM])
    >>> features_dict, targets_dict, combined_features, combined_targets = extractor.extract(
    ...     modules={
    ...         'rsi': {'lookback': [14, 21]},
    ...         'cmma': {'lookback': [20, 50], 'atr_length': [252]},
    ...         'volatility_regime': {'window': [20, 60], 'lookback': [252]}
    ...     }
    ... )
    >>> 
    >>> # Access per-module features
    >>> rsi_features = features_dict['rsi']
    >>> 
    >>> # Access combined features
    >>> all_features = combined_features
    """
    
    def __init__(
        self,
        tickers: Union[Ticker, List[Ticker]],
        start: dt = None,
        end: dt = None,
        use_millisecond_offset: bool = True
    ):
        """Initialize FeatureExtractor with tickers and date range."""
        # Convert single ticker to list
        if isinstance(tickers, Ticker):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
        
        # Set defaults
        self.start = start if start is not None else dt(1990, 1, 1)
        self.end = end if end is not None else dt.now()
        self.use_millisecond_offset = use_millisecond_offset
        
        # Storage for extracted features
        self.features_by_module = {}
        self.targets_by_module = {}
        self.combined_features = None
        self.combined_targets = None
        
        print(f"\n{'='*70}")
        print("FeatureExtractor Initialized")
        print(f"{'='*70}")
        print(f"Tickers: {[t.name for t in self.tickers]}")
        print(f"Date range: {self.start.date()} to {self.end.date()}")
        print(f"Multi-ticker offset: {self.use_millisecond_offset}")
        print(f"{'='*70}\n")
    
    def extract(
        self,
        bias_node_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract features from bias nodes.
        
        This is the main extraction method. It extracts features from all specified
        bias nodes in a single backtest run and returns organized dataframes.
        
        Parameters
        ----------
        bias_node_specs : List[Dict[str, Any]]
            List of bias node specifications. Each spec should be a dict with:
            - 'module_name' (str): Name of the bias node module (e.g., 'rsi', 'cmma')
            - 'params' (dict): Parameters for the bias node
            - 'timeframes' (List[TimeFrame], optional): Timeframes to use, defaults to [TimeFrame.D]
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'features': pd.DataFrame - Combined features (excluding ATR/EWSD)
            - 'normalization': pd.DataFrame - ATR and EWSD columns for normalization
            - 'targets': pd.DataFrame - Target columns (raw_return, log_return, etc.)
            - 'features_by_module': Dict[str, pd.DataFrame] - Features per module
            - 'normalization_by_module': Dict[str, pd.DataFrame] - Normalization per module
            - 'targets_by_module': Dict[str, pd.DataFrame] - Targets per module (same across modules)
            - 'feature_metadata': Dict[str, Dict[str, Any]] - Metadata about each feature, including parameters
        ...     ]
        ... )
        >>> features = result['features']
        >>> normalization = result['normalization']
        >>> targets = result['targets']
        >>> 
        >>> # Extract multiple modules
        >>> result = extractor.extract(
        ...     bias_node_specs=[
        ...         {'module_name': 'rsi', 'params': {'lookback': 14}},
        ...         {'module_name': 'cmma', 'params': {'lookback': 20, 'atr_length': 252}},
        ...         {'module_name': 'volatility_regime', 'params': {'window': 252}}
        ...     ]
        ... )
        """
        # Add default timeframes if not specified
        for spec in bias_node_specs:
            if 'timeframes' not in spec:
                spec['timeframes'] = [TimeFrame.D]
            
            # Ensure parameters are passed correctly
            if 'params' not in spec:
                spec['params'] = {}
            
            # Make a copy of parameters to avoid mutation
            spec['params'] = spec['params'].copy()
        
        print(f"\n{'='*70}")
        print("Starting Feature Extraction")
        print(f"{'='*70}")
        print(f"Bias nodes: {len(bias_node_specs)}")
        print(f"{'='*70}\n")
        
        # Extract all bias nodes in a single backtest run
        print(f"Extracting {len(bias_node_specs)} bias node(s) in a single backtest...")
        
        try:
            # Single extraction call for all bias nodes
            features_df, targets_df = extract_features_from_bias_node(
                ticker=self.tickers,
                bias_node_specs=bias_node_specs,
                start=self.start,
                end=self.end,
                use_millisecond_offset=self.use_millisecond_offset
            )
            
            print(f"✓ Extracted {len([c for c in features_df.columns if c != 'ticker'])} features in single backtest")
            
            # Group features by module for per-module access
            module_groups = {}
            for spec in bias_node_specs:
                module_name = spec['module_name']
                if module_name not in module_groups:
                    module_groups[module_name] = []
                module_groups[module_name].append(spec)
            
            # Store per-module results by filtering columns
            for module_name in module_groups.keys():
                # Find columns that belong to this module
                module_cols = [col for col in features_df.columns 
                              if col == 'ticker' or module_name in col.lower()]
                
                self.features_by_module[module_name] = features_df[module_cols]
                self.targets_by_module[module_name] = targets_df
            
            all_module_features = [features_df]
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            all_module_features = []
        
        # Create combined dataframe with all features
        if all_module_features:
            print(f"\n{'='*70}")
            print("Combining all modules...")
            print(f"{'='*70}")
            
            # Merge all modules on index; ticker column will be preserved automatically
            self.combined_features = all_module_features[0]
            for features_df in all_module_features[1:]:
                self.combined_features = self.combined_features.merge(
                    features_df,
                    left_index=True,
                    right_index=True,
                    how='outer',
                    suffixes=('', '_dup')  # Handle duplicate ticker column
                )
                # Remove duplicate ticker column if it exists
                if 'ticker_dup' in self.combined_features.columns:
                    self.combined_features = self.combined_features.drop(columns=['ticker_dup'])
                    
        # Build feature metadata
        feature_metadata = {}
        print(f"Generating feature metadata for {len(bias_node_specs)} specs")
        for i, spec in enumerate(bias_node_specs):
            module_name = spec['module_name']
            params = spec['params']
            timeframes = spec['timeframes']
            
            print(f"  Spec {i+1}: {module_name} with params {params}")
            
            # Generate unique base name using parameter values only
            param_values = '_'.join(str(v) for v in params.values())
            base_name = f"{module_name}_{param_values}"
            
            # Find features that start with the base name
            module_cols = [col for col in self.combined_features.columns 
                         if col.startswith(base_name)]
            
            print(f"    Found {len(module_cols)} columns for {base_name}")
            
            for col in module_cols:
                feature_metadata[col] = {
                    'module': module_name,
                    'parameters': params.copy(),
                    'timeframes': timeframes.copy(),
                    'base_name': base_name,
                    'full_name': col
                }
                print(f"      Metadata for {col}: {feature_metadata[col]['parameters']}")
        
        # After processing all specs, set up the combined data
        if self.combined_features is not None:
            # Targets are the same across all modules, just use the first one
            if self.targets_by_module:
                self.combined_targets = list(self.targets_by_module.values())[0]
            
            # Separate features from normalization columns (ATR/EWSD)
            all_columns = [col for col in self.combined_features.columns if col != 'ticker']
            
            # Identify normalization columns (use substring search, not word boundaries)
            # Column names like "atr_252_D_atr_252_lookback2" contain "atr" as substring
            normalization_cols = [
                col for col in all_columns
                if 'atr' in col.lower() or 'ewsd' in col.lower()
            ]
            
            # Identify feature columns (everything else)
            feature_cols = [col for col in all_columns if col not in normalization_cols]
            
            # Create separate dataframes
            combined_features_only = self.combined_features[feature_cols + (['ticker'] if 'ticker' in self.combined_features.columns else [])]
            combined_normalization = self.combined_features[normalization_cols + (['ticker'] if 'ticker' in self.combined_features.columns else [])]
            
            print(f"✓ Combined features: {len(feature_cols)} feature columns")
            print(f"✓ Normalization columns: {len(normalization_cols)} (ATR/EWSD)")
            print(f"✓ Date range: {self.combined_features.index.min()} to {self.combined_features.index.max()}")
            print(f"✓ Samples: {len(self.combined_features)}")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print("⚠️  No features extracted from any module")
            print(f"{'='*70}\n")
            combined_features_only = pd.DataFrame()
            combined_normalization = pd.DataFrame()
            self.combined_targets = pd.DataFrame()
            return {
                'features': combined_features_only,
                'normalization': combined_normalization,
                'targets': self.combined_targets,
                'features_by_module': {},
                'normalization_by_module': {},
                'targets_by_module': {},
                'feature_metadata': {}
            }
        
        # Also separate features and normalization for per-module dataframes
        features_by_module_only = {}
        normalization_by_module = {}
        
        if self.features_by_module:
            for module_name, module_df in self.features_by_module.items():
                all_cols = [col for col in module_df.columns if col != 'ticker']
                
                # Identify normalization columns (use substring search)
                norm_cols = [
                    col for col in all_cols
                    if 'atr' in col.lower() or 'ewsd' in col.lower()
                ]
                
                # Identify feature columns
                feat_cols = [col for col in all_cols if col not in norm_cols]
                
                # Create separate dataframes
                features_by_module_only[module_name] = module_df[feat_cols + (['ticker'] if 'ticker' in module_df.columns else [])]
                normalization_by_module[module_name] = module_df[norm_cols + (['ticker'] if 'ticker' in module_df.columns else [])]
        
        # Return dictionary with all data
        if self.combined_features is not None:
            return {
                'features': combined_features_only,
                'normalization': combined_normalization,
                'targets': self.combined_targets,
                'features_by_module': features_by_module_only,
                'normalization_by_module': normalization_by_module,
                'targets_by_module': self.targets_by_module,
                'feature_metadata': feature_metadata
            }
        else:
            return {
                'features': pd.DataFrame(),
                'normalization': pd.DataFrame(),
                'targets': pd.DataFrame(),
                'features_by_module': {},
                'normalization_by_module': {},
                'targets_by_module': {},
                'feature_metadata': {}
            }
    

    
    def get_module_features(self, module_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get features and targets for a specific module.
        
        Parameters
        ----------
        module_name : str
            Name of the module
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (features_df, targets_df) for the specified module
            
        Raises
        ------
        KeyError
            If module_name not found
        """
        if module_name not in self.features_by_module:
            raise KeyError(
                f"Module '{module_name}' not found. "
                f"Available modules: {list(self.features_by_module.keys())}"
            )
        
        return self.features_by_module[module_name], self.targets_by_module[module_name]
    
    def get_combined_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get combined features and targets from all modules.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (combined_features, combined_targets)
            
        Raises
        ------
        ValueError
            If no features have been extracted yet
        """
        if self.combined_features is None:
            raise ValueError(
                "No features extracted yet. Call extract() first."
            )
        
        return self.combined_features, self.combined_targets
    
    def summary(self) -> pd.DataFrame:
        """
        Get a summary of extracted features.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: module, n_features, n_samples, date_range
        """
        if not self.features_by_module:
            print("No features extracted yet. Call extract() first.")
            return pd.DataFrame()
        
        summaries = []
        for module_name, features_df in self.features_by_module.items():
            n_features = len([col for col in features_df.columns if col != 'ticker'])
            summaries.append({
                'module': module_name,
                'n_features': n_features,
                'n_samples': len(features_df),
                'start_date': features_df.index.min(),
                'end_date': features_df.index.max()
            })
        
        return pd.DataFrame(summaries)
    
    def __repr__(self) -> str:
        """String representation."""
        n_modules = len(self.features_by_module)
        n_features = len(self.combined_features.columns) - 1 if self.combined_features is not None else 0
        return (
            f"FeatureExtractor(tickers={[t.name for t in self.tickers]}, "
            f"n_modules={n_modules}, n_features={n_features})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        if not self.features_by_module:
            return "FeatureExtractor (no features extracted yet)"
        
        lines = [f"FeatureExtractor with {len(self.tickers)} ticker(s):"]
        for ticker in self.tickers:
            lines.append(f"  - {ticker.name}")
        
        lines.append(f"\nExtracted {len(self.features_by_module)} module(s):")
        for module_name, features_df in self.features_by_module.items():
            n_features = len([col for col in features_df.columns if col != 'ticker'])
            lines.append(f"  - {module_name}: {n_features} features")
        
        if self.combined_features is not None:
            n_total = len([col for col in self.combined_features.columns if col != 'ticker'])
            lines.append(f"\nTotal: {n_total} features combined")
        
        return '\n'.join(lines)
