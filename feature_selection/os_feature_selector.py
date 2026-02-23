"""
Out-of-Sample Feature Selector

This module provides a framework for out-of-sample feature validation using
walk-forward analysis, cross-validation, and permutation tests. It supports
multiple features simultaneously and follows the same workflow as FeatureExplorer.

The OSFeatureSelector class includes methods for:
- Walk-forward out-of-sample validation with any BaseModel
- Cross-validation testing with any BaseModel
- Modular permutation tests with objective functions
- Multiple feature support in a single instance

Author: Trading Research Team
Date: 2025-11-28
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable, TYPE_CHECKING
from datetime import datetime as dt
from sklearn.model_selection import KFold

from feature_selection.base_models.base_model import BinningModelBase
from feature_selection.walkforward.walkforward_model import WalkForwardSplitter
from metrics.performance import ObjectiveMetric
from utils.evaluation.permutation_test.permutation_engine import (
    PermutationEngine, 
    FeaturePermutationStrategy
)

if TYPE_CHECKING:
    from feature_selection.base_models.feature_base_model import BaseModel
else:
    # For runtime, use BinningModelBase as the base type
    BaseModel = BinningModelBase


class OSFeatureSelector:
    """
    Out-of-sample feature selection framework using model-based approaches.
    
    This class provides methods for validating features using out-of-sample
    techniques including walk-forward analysis, cross-validation, and permutation
    tests. It works with multiple features simultaneously and any model that 
    implements the BaseModel interface.
    
    Similar to FeatureExplorer, it takes features_df and targets_df as input
    and supports multiple features in a single instance.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features dataframe with feature columns
        Columns: feature columns + optional 'ticker' column
        Must have datetime index matching targets_df
    targets_df : pd.DataFrame
        Targets dataframe with target columns
        Columns: raw_return, log_return, log_return_atr, log_return_ewsd, optional 'ticker'
        Must have datetime index matching features_df
    metadata : Optional[Dict[str, Any]], default=None
        Optional metadata about the features
        
    Attributes
    ----------
    features_df : pd.DataFrame
        Features dataframe
    targets_df : pd.DataFrame
        Targets dataframe
    feature_names : List[str]
        List of feature column names (excluding 'ticker')
    n_features : int
        Number of features
    n_samples : int
        Number of samples
    date_range : Tuple[datetime, datetime]
        Date range of the data
    metadata : Dict[str, Any]
        Feature metadata
    results : Dict[str, Any]
        Dictionary storing all test results
        
    Examples
    --------
    >>> # Step 1: Extract features using FeatureExtractor
    >>> from feature_extraction.feature_extractor_class import FeatureExtractor
    >>> extractor = FeatureExtractor(ticker=Ticker.SPY)
    >>> result = extractor.extract(bias_node_specs=[...])
    >>> 
    >>> # Step 2: Create OSFeatureSelector from extracted dataframes
    >>> from feature_selection.os_feature_selector import OSFeatureSelector
    >>> selector = OSFeatureSelector(
    ...     features_df=result['features'],
    ...     targets_df=result['targets'],
    ...     metadata=result
    ... )
    >>> 
    >>> # Step 3: Run out-of-sample tests
    >>> results = selector.walkforward_test(
    ...     model=model,
    ...     objective_metric=metric,
    ...     feature_cols=['rsi_signal_D_lookback_14', 'momentum_signal_D_lookback_20'],
    ...     target_col='log_return',
    ...     train_start=datetime(2000, 1, 1),
    ...     train_end=datetime(2010, 1, 1)
    ... )
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize OSFeatureSelector with features and targets dataframes."""
        # Validate inputs
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame")
        if not isinstance(targets_df, pd.DataFrame):
            raise TypeError("targets_df must be a pandas DataFrame")
        
        # Check that indices match
        if not features_df.index.equals(targets_df.index):
            raise ValueError(
                "features_df and targets_df must have matching indices. "
                "Use FeatureExtractor to ensure proper alignment."
            )
        
        # Store dataframes
        self.features_df = features_df
        self.targets_df = targets_df
        self.metadata = metadata or {}
        self.results = {}
        
        # Extract feature names (exclude 'ticker' column)
        self.feature_names = [
            col for col in features_df.columns 
            if col != 'ticker'
        ]
        self.n_features = len(self.feature_names)
        
        # Store basic statistics
        self.n_samples = len(features_df)
        self.date_range = (features_df.index.min(), features_df.index.max())
        
        # Check if multi-ticker
        self.has_ticker = 'ticker' in features_df.columns
        if self.has_ticker:
            self.tickers = features_df['ticker'].unique().tolist()
            self.n_tickers = len(self.tickers)
        else:
            self.tickers = None
            self.n_tickers = 1
        
        # Create combined dataframe for internal use
        self.df = pd.concat([features_df, targets_df], axis=1)
        # Remove duplicate columns if any
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        
    def walkforward_test(
        self,
        model: BinningModelBase,
        objective_metric: ObjectiveMetric,
        feature_cols: Union[str, List[str]],
        train_start: dt,
        train_end: dt,
        test_step: int = 252,
        num_steps: int = 10,
        strategy: str = 'long',
        target_col: str = 'log_return',
        normalization_data: Optional[pd.DataFrame] = None,
        verbose: bool = True
    ) -> Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[pd.DataFrame, List[Dict], Optional[Any]]]]:
        """
        Perform walk-forward analysis using any BaseModel.
        
        For each walk-forward step:
        1. Fit model on training window
        2. Generate predictions on test window
        3. Compute objective metric on test returns
        4. Track performance
        
        Parameters
        ----------
        model : BinningModelBase
            Model to use for walk-forward analysis (e.g., ContinuousBinningModel,
            DecisionTreeBinningModel). The model will be fitted on each training
            window and used to predict on test windows.
        objective_metric : ObjectiveMetric
            Metric to compute on test returns (e.g., SortinoRatio, SharpeRatio)
        feature_cols : Union[str, List[str]]
            Single feature column name or list of feature columns to test
        train_start : datetime
            Start date for initial training window
        train_end : datetime
            End date for initial training window
        test_step : int, default=252
            Number of days for test period (~1 year)
        num_steps : int, default=10
            Number of walk-forward steps
        strategy : str, default='long'
            Trading strategy: 'long' or 'short'
        target_col : str, default='log_return'
            Which target column to use. Options: 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
        normalization_data : Optional[pd.DataFrame], default=None
            Volatility data DataFrame if model requires it (columns match feature_cols)
        verbose : bool, default=True
            Print detailed progress
            
        Returns
        -------
        Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[pd.DataFrame, List[Dict], Optional[Any]]]]
            If single feature (feature_cols is str): Returns (results_df, step_info, fig)
            If multiple features (feature_cols is List[str]): Returns Dict[str, Tuple]
            where keys are feature names and values are (results_df, step_info, fig) tuples
        """
        # Validate inputs
        self._validate_model_and_metric(model, objective_metric)
        
        # Convert single feature to list
        if isinstance(feature_cols, str):
            feature_cols_list = [feature_cols]
            single_feature = True
        else:
            feature_cols_list = feature_cols
            single_feature = False
        
        # Validate feature columns
        missing_features = [col for col in feature_cols_list if col not in self.feature_names]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.targets_df.columns)}"
            )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"WALK-FORWARD OUT-OF-SAMPLE TEST")
            print(f"{'='*70}")
            print(f"Features: {len(feature_cols_list)}")
            print(f"Model: {model.__class__.__name__}")
            print(f"Metric: {objective_metric.__class__.__name__}")
            print(f"Strategy: {strategy}")
            print(f"Target: {target_col}")
            print(f"Steps: {num_steps}")
            print(f"{'='*70}")
        
        # Process each feature
        results_by_feature = {}
        
        for i, feature_col in enumerate(feature_cols_list, 1):
            if verbose:
                print(f"\n[{i}/{len(feature_cols_list)}] Processing feature: {feature_col}")
            
            try:
                # Get feature and target data
                feature_data = self.df[feature_col]
                target_data = self.df[target_col]
                
                # Get normalization data if needed
                norm_data = None
                if normalization_data is not None and feature_col in normalization_data.columns:
                    norm_data = normalization_data[feature_col]
                
                # Run walk-forward test for single feature
                results_df, step_info = self._run_single_walkforward_test(
                    feature_data=feature_data,
                    target_data=target_data,
                    model=model,
                    objective_metric=objective_metric,
                    train_start=train_start,
                    train_end=train_end,
                    test_step=test_step,
                    num_steps=num_steps,
                    strategy=strategy,
                    normalization_data=norm_data,
                    verbose=verbose
                )
                
                # Store results
                results_by_feature[feature_col] = (results_df, step_info, None)
                
                if verbose:
                    final_metric = results_df['test_metric'].mean() if len(results_df) > 0 else 0.0
                    total_trades = results_df['n_trades'].sum() if len(results_df) > 0 else 0
                    print(f"  ✓ Complete: metric={final_metric:.4f}, trades={total_trades}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                # Store error result
                results_by_feature[feature_col] = (pd.DataFrame(), [], None)
                continue
        
        # Create summary results
        summary_rows = []
        for feature_col, (results_df, step_info, fig) in results_by_feature.items():
            if len(results_df) > 0:
                # Compute aggregated OOS metric from all step returns
                all_step_returns = []
                for step in step_info:
                    if 'returns' in step:
                        all_step_returns.extend(step['returns'])
                
                if len(all_step_returns) > 0:
                    aggregated_returns = pd.Series(all_step_returns)
                    final_oos_metric = objective_metric.compute(aggregated_returns)
                else:
                    final_oos_metric = 0.0
                
                mean_oos_metric = results_df['test_metric'].mean()
                total_trades = results_df['n_trades'].sum()
            else:
                final_oos_metric = 0.0
                mean_oos_metric = 0.0
                total_trades = 0
            
            summary_rows.append({
                'feature': feature_col,
                'final_oos_metric': final_oos_metric,
                'mean_oos_metric': mean_oos_metric,
                'total_trades': total_trades
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Store aggregated results
        self.results['walkforward_test'] = {
            'results_by_feature': results_by_feature,
            'summary_df': summary_df,
            'model': model.__class__.__name__,
            'objective_metric': objective_metric.__class__.__name__,
            'strategy': strategy,
            'train_start': train_start,
            'train_end': train_end,
            'test_step': test_step,
            'num_steps': num_steps
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"WALK-FORWARD SUMMARY")
            print(f"{'='*70}")
            print(summary_df.to_string(index=False))
            print(f"{'='*70}")
        
        # Return appropriate format
        if single_feature:
            return results_by_feature[feature_cols_list[0]]
        else:
            return results_by_feature
    
    def cv_test(
        self,
        model: BinningModelBase,
        objective_metric: ObjectiveMetric,
        feature_cols: Union[str, List[str]],
        n_splits: int = 5,
        strategy: str = 'long',
        target_col: str = 'log_return',
        normalization_data: Optional[pd.DataFrame] = None,
        shuffle: bool = False,
        verbose: bool = True
    ) -> Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[pd.DataFrame, List[Dict], Optional[Any]]]]:
        """
        K-fold cross-validation for out-of-sample validation.
        
        Parameters
        ----------
        model : BinningModelBase
            Model instance
        objective_metric : ObjectiveMetric
            Custom metric
        feature_cols : Union[str, List[str]]
            Single feature column name or list of feature columns to test
        n_splits : int, default=5
            Number of CV folds
        strategy : str, default='long'
            Trading strategy: 'long' or 'short'
        target_col : str, default='log_return'
            Target column
        normalization_data : Optional[pd.DataFrame], default=None
            Volatility data DataFrame if needed (columns match feature_cols)
        shuffle : bool, default=False
            Whether to shuffle before splitting (preserve time order by default)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[pd.DataFrame, List[Dict], Optional[Any]]]]
            If single feature: Returns (results_df, fold_info, fig)
            If multiple features: Returns Dict[str, Tuple] with results for each feature
        """
        # Validate inputs
        self._validate_model_and_metric(model, objective_metric)
        
        # Convert single feature to list
        if isinstance(feature_cols, str):
            feature_cols_list = [feature_cols]
            single_feature = True
        else:
            feature_cols_list = feature_cols
            single_feature = False
        
        # Validate feature columns
        missing_features = [col for col in feature_cols_list if col not in self.feature_names]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.targets_df.columns)}"
            )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"CROSS-VALIDATION OUT-OF-SAMPLE TEST")
            print(f"{'='*70}")
            print(f"Features: {len(feature_cols_list)}")
            print(f"Model: {model.__class__.__name__}")
            print(f"Metric: {objective_metric.__class__.__name__}")
            print(f"Strategy: {strategy}")
            print(f"Target: {target_col}")
            print(f"Folds: {n_splits}")
            print(f"Shuffle: {shuffle}")
            print(f"{'='*70}")
        
        # Process each feature
        results_by_feature = {}
        
        for i, feature_col in enumerate(feature_cols_list, 1):
            if verbose:
                print(f"\n[{i}/{len(feature_cols_list)}] Processing feature: {feature_col}")
            
            try:
                # Get feature and target data
                feature_data = self.df[feature_col]
                target_data = self.df[target_col]
                
                # Remove NaN values
                valid_mask = ~(feature_data.isna() | target_data.isna())
                feature_clean = feature_data[valid_mask]
                target_clean = target_data[valid_mask]
                
                if len(feature_clean) < n_splits * 10:
                    raise ValueError(f"Insufficient data: need at least {n_splits * 10} samples")
                
                # Get normalization data if needed
                norm_clean = None
                if normalization_data is not None and feature_col in normalization_data.columns:
                    norm_clean = normalization_data[feature_col][valid_mask]
                
                # Create CV splitter
                cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42 if shuffle else None)
                
                # Run CV folds
                fold_results = []
                all_fold_returns = []
                
                for fold, (train_idx, test_idx) in enumerate(cv.split(feature_clean)):
                    # Get train/test data
                    X_train = feature_clean.iloc[train_idx]
                    y_train = target_clean.iloc[train_idx]
                    X_test = feature_clean.iloc[test_idx]
                    y_test = target_clean.iloc[test_idx]
                    
                    # Get normalization data for this fold if needed
                    norm_train = None
                    norm_test = None
                    if norm_clean is not None:
                        norm_train = norm_clean.iloc[train_idx]
                        norm_test = norm_clean.iloc[test_idx]
                    
                    if verbose:
                        print(f"  Fold {fold + 1}/{n_splits}: "
                              f"train={len(X_train)}, test={len(X_test)}")
                    
                    try:
                        # Fit model on training data
                        model.fit(X_train, y_train, normalization_data=norm_train)
                        
                        # Generate predictions on test data
                        test_signals = model.predict(
                            X_test,
                            strategy=strategy,
                            normalization_data=norm_test
                        )
                        
                        # Get test returns for selected signals
                        test_returns = y_test[test_signals > 0]
                        
                        # Compute metrics
                        if len(test_returns) > 0:
                            test_metric = objective_metric.compute(test_returns)
                            n_trades = len(test_returns)
                            mean_return = test_returns.mean()
                            all_fold_returns.extend(test_returns.values)
                        else:
                            test_metric = 0.0
                            n_trades = 0
                            mean_return = 0.0
                        
                        fold_results.append({
                            'fold': fold,
                            'test_metric': test_metric,
                            'mean_return': mean_return,
                            'n_trades': n_trades,
                            'strategy': strategy
                        })
                        
                    except Exception as e:
                        if verbose:
                            print(f"    ERROR: {e}")
                        fold_results.append({
                            'fold': fold,
                            'test_metric': np.nan,
                            'mean_return': np.nan,
                            'n_trades': 0,
                            'strategy': strategy,
                            'error': str(e)
                        })
                
                # Convert to DataFrame
                results_df = pd.DataFrame(fold_results)
                
                # Store results
                results_by_feature[feature_col] = (results_df, fold_results, None)
                
                if verbose:
                    final_metric = results_df['test_metric'].mean() if len(results_df) > 0 else 0.0
                    total_trades = results_df['n_trades'].sum() if len(results_df) > 0 else 0
                    print(f"  ✓ Complete: metric={final_metric:.4f}, trades={total_trades}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                # Store error result
                results_by_feature[feature_col] = (pd.DataFrame(), [], None)
                continue
        
        # Create summary results
        summary_rows = []
        for feature_col, (results_df, fold_info, fig) in results_by_feature.items():
            if len(results_df) > 0:
                # Compute aggregated OOS metric from all fold returns
                all_fold_returns = []
                for fold in fold_info:
                    if 'returns' in fold:
                        all_fold_returns.extend(fold['returns'])
                
                if len(all_fold_returns) > 0:
                    aggregated_returns = pd.Series(all_fold_returns)
                    final_oos_metric = objective_metric.compute(aggregated_returns)
                else:
                    final_oos_metric = 0.0
                
                mean_oos_metric = results_df['test_metric'].mean()
                total_trades = results_df['n_trades'].sum()
            else:
                final_oos_metric = 0.0
                mean_oos_metric = 0.0
                total_trades = 0
            
            summary_rows.append({
                'feature': feature_col,
                'final_oos_metric': final_oos_metric,
                'mean_oos_metric': mean_oos_metric,
                'total_trades': total_trades
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Store aggregated results
        self.results['cv_test'] = {
            'results_by_feature': results_by_feature,
            'summary_df': summary_df,
            'model': model.__class__.__name__,
            'objective_metric': objective_metric.__class__.__name__,
            'strategy': strategy,
            'n_splits': n_splits,
            'shuffle': shuffle
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"CROSS-VALIDATION SUMMARY")
            print(f"{'='*70}")
            print(summary_df.to_string(index=False))
            print(f"{'='*70}")
        
        # Return appropriate format
        if single_feature:
            return results_by_feature[feature_cols_list[0]]
        else:
            return results_by_feature
    
    def permutation_test(
        self,
        feature_cols: Union[str, List[str]],
        objective_func: Callable,
        permutation_mode: str = 'full_shuffle',
        nreps: int = 100,
        target_col: str = 'log_return',
        normalization_data: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        random_seed: Optional[int] = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Permutation-based robustness testing using feature shuffling.
        
        This method uses a modular, DRY design by accepting an objective function
        that encapsulates the CV/walkforward method. The permutation test is
        agnostic to how the metric is computed.
        
        Parameters
        ----------
        feature_cols : Union[str, List[str]]
            Single feature column name or list of feature columns to test
        objective_func : Callable
            Objective function that includes the test method (CV/walkforward).
            Takes (feature_data: pd.Series, target_data: pd.Series, 
                   normalization_data: Optional[pd.Series]) as input
            Returns a float representing the metric value
            Encapsulates the entire test logic (walkforward/CV) internally
        permutation_mode : str, default='full_shuffle'
            Permutation strategy: 'full_shuffle' or 'fold_shuffle'
        nreps : int, default=100
            Number of permutation replications
        target_col : str, default='log_return'
            Target column
        normalization_data : Optional[pd.DataFrame], default=None
            Volatility data DataFrame if needed (columns match feature_cols)
        alpha : float, default=0.05
            Significance level
        random_seed : Optional[int], default=42
            Random seed
        n_jobs : int, default=-1
            Parallel jobs (-1 = all CPUs)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        pd.DataFrame
            Results with columns:
            - feature: Feature name
            - original_metric: Original metric value (before permutation)
            - pval: Permutation p-value (fraction of permutations >= original)
            - significant: Boolean (pval <= alpha)
            - n_valid_permutations: Number of successful permutations
            - permutation_mode: Which strategy was used
        """
        # Convert single feature to list
        if isinstance(feature_cols, str):
            feature_cols_list = [feature_cols]
        else:
            feature_cols_list = feature_cols
        
        # Validate feature columns
        missing_features = [col for col in feature_cols_list if col not in self.feature_names]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.targets_df.columns)}"
            )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PERMUTATION TEST")
            print(f"{'='*70}")
            print(f"Features: {len(feature_cols_list)}")
            print(f"Mode: {permutation_mode}")
            print(f"Replications: {nreps}")
            print(f"Target: {target_col}")
            print(f"Alpha: {alpha}")
            print(f"{'='*70}")
        
        # Store objective function and related data for criterion function
        self._temp_objective_func = objective_func
        self._temp_target_col = target_col
        self._temp_normalization_data = normalization_data
        
        # Create permutation strategy
        strategy = FeaturePermutationStrategy()
        
        # Create engine
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        
        # Run permutation test
        results = engine.run_permutation_test(
            data=self.df,
            feature_cols=feature_cols_list,
            criterion_func=self._permutation_criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col
        )
        
        # Clean up temporary attributes
        delattr(self, '_temp_objective_func')
        delattr(self, '_temp_target_col')
        delattr(self, '_temp_normalization_data')
        
        # Add permutation mode to results
        results['permutation_mode'] = permutation_mode
        results['n_valid_permutations'] = nreps  # Assuming all successful for now
        
        # Store results
        self.results['permutation_test'] = {
            'results': results,
            'permutation_mode': permutation_mode,
            'nreps': nreps,
            'target_col': target_col,
            'alpha': alpha
        }
        
        if verbose:
            n_significant = results['significant'].sum()
            print(f"\n{'='*70}")
            print(f"PERMUTATION TEST SUMMARY")
            print(f"{'='*70}")
            print(f"Significant features (p <= {alpha}): {n_significant}/{len(feature_cols_list)}")
            if n_significant > 0:
                print(f"\nSignificant features:")
                sig_features = results[results['significant']].copy()
                sig_features = sig_features.sort_values('original_criterion', ascending=False)
                print(sig_features[['feature', 'original_criterion', 'pval']].to_string(index=False))
            print(f"{'='*70}")
        
        return results
    
    def get_results(self, test_name: str) -> Dict[str, Any]:
        """
        Retrieve stored results for a specific test.
        
        Parameters
        ----------
        test_name : str
            Name of the test ('walkforward_test', 'cv_test', 'permutation_test')
            
        Returns
        -------
        Dict[str, Any]
            The dictionary stored in self.results[test_name]
        """
        if test_name not in self.results:
            raise ValueError(f"Test '{test_name}' not found. Available: {list(self.results.keys())}")
        return self.results[test_name]
    
    def get_summary(self) -> pd.DataFrame:
        """
        Aggregate summary across all tests run.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: test_name, metric_value, pval (if permutation), significant, etc.
        """
        summary_rows = []
        
        for test_name, test_results in self.results.items():
            if test_name in ['walkforward_test', 'cv_test']:
                if 'summary_df' in test_results:
                    summary_df = test_results['summary_df']
                    for _, row in summary_df.iterrows():
                        summary_rows.append({
                            'test_name': test_name,
                            'feature': row['feature'],
                            'metric_value': row.get('final_oos_metric', 0.0),
                            'total_trades': row.get('total_trades', 0),
                            'pval': None,
                            'significant': None
                        })
            elif test_name == 'permutation_test':
                if 'results' in test_results:
                    perm_results = test_results['results']
                    for _, row in perm_results.iterrows():
                        summary_rows.append({
                            'test_name': test_name,
                            'feature': row['feature'],
                            'metric_value': row['original_criterion'],
                            'total_trades': None,
                            'pval': row['pval'],
                            'significant': row['significant']
                        })
        
        return pd.DataFrame(summary_rows)
    
    def _validate_model_and_metric(self, model: BaseModel, objective_metric: ObjectiveMetric) -> None:
        """Validate that model implements BaseModel interface and metric implements ObjectiveMetric interface."""
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise ValueError("Model must implement BaseModel interface with fit() and predict() methods")
        
        if not hasattr(objective_metric, 'compute'):
            raise ValueError("Objective metric must implement ObjectiveMetric interface with compute() method")
    
    def _permutation_criterion_func(self, data: pd.DataFrame, feature_col: str) -> float:
        """
        Criterion function for permutation test that can be pickled.
        
        This method uses temporary attributes set by permutation_test() to access
        the objective function and related data. This approach avoids the pickling
        issues that occur with nested local functions.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing features and targets (potentially permuted)
        feature_col : str
            Name of the feature column to evaluate
            
        Returns
        -------
        float
            Computed criterion value
        """
        # Extract feature and target data
        feature_data = data[feature_col]
        target_data = data[self._temp_target_col]
        
        # Get normalization data if available
        normalization_series = None
        if self._temp_normalization_data is not None and feature_col in self._temp_normalization_data.columns:
            normalization_series = self._temp_normalization_data[feature_col]
        
        # Call the objective function
        return self._temp_objective_func(feature_data, target_data, normalization_series)
    
    def _extract_normalization_data(self, model: BaseModel, feature_col: str) -> Optional[pd.Series]:
        """Extract normalization column from self.df based on model's normalize_by attribute."""
        if not hasattr(model, 'normalize_by') or model.normalize_by is None:
            return None
        
        norm_col = model.normalize_by.lower()  # 'ewsd' or 'atr'
        
        # Search for the normalization column in the dataframe (case-insensitive)
        # Look for columns like 'ewsd_252_D', 'atr_252_D', etc.
        matching_cols = [col for col in self.df.columns if col.lower().startswith(norm_col)]
        
        if len(matching_cols) == 0:
            raise ValueError(
                f"Model requires normalization by '{model.normalize_by}' but no {norm_col} column found. "
                f"Available columns: {list(self.df.columns)}. "
                f"Make sure to include {norm_col} in your feature extraction."
            )
        
        # Use the first matching column (typically there's only one)
        norm_col_name = matching_cols[0]
        return self.df[norm_col_name]
    
    def _create_walkforward_objective_func(
        self,
        model: BinningModelBase,
        objective_metric: ObjectiveMetric,
        train_start: dt,
        train_end: dt,
        test_step: int,
        num_steps: int,
        strategy: str,
        target_col: str
    ) -> Callable:
        """Factory function to create objective function for walkforward permutation tests."""
        def objective_func(
            feature_data: pd.Series,
            target_data: pd.Series,
            normalization_data: Optional[pd.Series] = None
        ) -> float:
            """Run walkforward test internally and return final metric."""
            try:
                results_df, step_info = self._run_single_walkforward_test(
                    feature_data=feature_data,
                    target_data=target_data,
                    model=model,
                    objective_metric=objective_metric,
                    train_start=train_start,
                    train_end=train_end,
                    test_step=test_step,
                    num_steps=num_steps,
                    strategy=strategy,
                    normalization_data=normalization_data,
                    verbose=False
                )
                
                # Aggregate all returns from all steps
                all_returns = []
                for step in step_info:
                    if 'returns' in step:
                        all_returns.extend(step['returns'])
                
                if len(all_returns) > 0:
                    aggregated_returns = pd.Series(all_returns)
                    return objective_metric.compute(aggregated_returns)
                else:
                    return 0.0
                    
            except Exception:
                return 0.0
        
        return objective_func
    
    def _create_cv_objective_func(
        self,
        model: BinningModelBase,
        objective_metric: ObjectiveMetric,
        n_splits: int,
        strategy: str,
        target_col: str
    ) -> Callable:
        """Factory function to create objective function for CV permutation tests."""
        def objective_func(
            feature_data: pd.Series,
            target_data: pd.Series,
            normalization_data: Optional[pd.Series] = None
        ) -> float:
            """Run CV test internally and return final metric."""
            try:
                from sklearn.model_selection import KFold
                
                # Remove NaN values
                valid_mask = ~(feature_data.isna() | target_data.isna())
                feature_clean = feature_data[valid_mask]
                target_clean = target_data[valid_mask]
                
                if len(feature_clean) < n_splits * 10:
                    return 0.0
                
                norm_clean = None
                if normalization_data is not None:
                    norm_clean = normalization_data[valid_mask]
                
                # Create CV splitter
                cv = KFold(n_splits=n_splits, shuffle=False, random_state=42)
                
                # Collect all fold returns
                all_fold_returns = []
                
                for fold, (train_idx, test_idx) in enumerate(cv.split(feature_clean)):
                    # Get train/test data
                    X_train = feature_clean.iloc[train_idx]
                    y_train = target_clean.iloc[train_idx]
                    X_test = feature_clean.iloc[test_idx]
                    y_test = target_clean.iloc[test_idx]
                    
                    # Get normalization data for this fold if needed
                    norm_train = None
                    norm_test = None
                    if norm_clean is not None:
                        norm_train = norm_clean.iloc[train_idx]
                        norm_test = norm_clean.iloc[test_idx]
                    
                    # Fit model on training data
                    model.fit(X_train, y_train, normalization_data=norm_train)
                    
                    # Generate predictions on test data
                    test_signals = model.predict(
                        X_test,
                        strategy=strategy,
                        normalization_data=norm_test
                    )
                    
                    # Get test returns for selected signals
                    test_returns = y_test[test_signals > 0]
                    
                    if len(test_returns) > 0:
                        all_fold_returns.extend(test_returns.values)
                
                if len(all_fold_returns) > 0:
                    aggregated_returns = pd.Series(all_fold_returns)
                    return objective_metric.compute(aggregated_returns)
                else:
                    return 0.0
                    
            except Exception:
                return 0.0
        
        return objective_func
    
    def _run_single_walkforward_test(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        model: BinningModelBase,
        objective_metric: ObjectiveMetric,
        train_start: dt,
        train_end: dt,
        test_step: int,
        num_steps: int,
        strategy: str,
        normalization_data: Optional[pd.Series] = None,
        verbose: bool = False
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Run walk-forward test for a single feature."""
        # Create splitter
        splitter = WalkForwardSplitter(
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps
        )
        
        # Get splits
        splits = splitter.split(feature_data.index)
        
        if len(splits) == 0:
            raise ValueError("No valid walk-forward splits found")
        
        # Process each step
        step_results = []
        
        for step, (train_idx, test_idx) in enumerate(splits):
            # Get train/test data
            X_train = feature_data.iloc[train_idx]
            y_train = target_data.iloc[train_idx]
            X_test = feature_data.iloc[test_idx]
            y_test = target_data.iloc[test_idx]
            
            if verbose:
                print(f"    Step {step + 1}/{len(splits)}: "
                      f"train={len(X_train)}, test={len(X_test)}")
            
            try:
                # Get normalization data for this split if needed
                norm_train = None
                norm_test = None
                if normalization_data is not None:
                    norm_train = normalization_data.iloc[train_idx]
                    norm_test = normalization_data.iloc[test_idx]
                
                # Fit model on training data
                model.fit(X_train, y_train, normalization_data=norm_train)
                
                # Generate predictions on test data
                test_signals = model.predict(
                    X_test,
                    strategy=strategy,
                    normalization_data=norm_test
                )
                
                # Get test returns for selected signals
                test_returns = y_test[test_signals > 0]
                
                # Compute metrics
                if len(test_returns) > 0:
                    test_metric = objective_metric.compute(test_returns)
                    n_trades = len(test_returns)
                    mean_return = test_returns.mean()
                    returns_list = test_returns.values.tolist()
                else:
                    test_metric = 0.0
                    n_trades = 0
                    mean_return = 0.0
                    returns_list = []
                
                step_results.append({
                    'step': step,
                    'test_start': X_test.index.min(),
                    'test_end': X_test.index.max(),
                    'test_metric': test_metric,
                    'mean_return': mean_return,
                    'n_trades': n_trades,
                    'strategy': strategy,
                    'returns': returns_list  # Store returns for aggregation
                })
                
            except Exception as e:
                if verbose:
                    print(f"      ERROR: {e}")
                step_results.append({
                    'step': step,
                    'test_start': X_test.index.min(),
                    'test_end': X_test.index.max(),
                    'test_metric': np.nan,
                    'mean_return': np.nan,
                    'n_trades': 0,
                    'strategy': strategy,
                    'error': str(e),
                    'returns': []
                })
        
        # Convert to DataFrame (exclude 'returns' for DataFrame)
        df_results = []
        for result in step_results:
            df_result = {k: v for k, v in result.items() if k != 'returns'}
            df_results.append(df_result)
        
        results_df = pd.DataFrame(df_results)
        
        return results_df, step_results
    
    def __repr__(self) -> str:
        """String representation of the OSFeatureSelector."""
        return (
            f"OSFeatureSelector(n_features={self.n_features}, "
            f"n_samples={self.n_samples}, "
            f"date_range={self.date_range[0].date()} to {self.date_range[1].date()})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"OSFeatureSelector with {self.n_features} features:"]
        
        # Show first few features
        for feature_name in self.feature_names[:5]:
            lines.append(f"  - {feature_name}")
        
        if self.n_features > 5:
            lines.append(f"  ... and {self.n_features - 5} more")
        
        lines.append(f"\nSamples: {self.n_samples}")
        lines.append(f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        
        if self.has_ticker:
            lines.append(f"Tickers: {self.tickers}")
        
        lines.append(f"Tests run: {list(self.results.keys())}")
        
        return '\n'.join(lines)
