import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
import sys
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.enums import Ticker, TimeFrame
from utils.helpers import load_data


def compute_profit_factor(returns: np.ndarray) -> float:
    """Compute profit factor for a series of returns.
    
    Args:
        returns: Array of returns (can be positive or negative)
        
    Returns:
        float: Profit factor (sum of gains / sum of losses)
    """
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return gains / max(losses, 1e-10)  # Avoid division by zero

# Make optimize_threshold optional
OPTIMIZE_THRESHOLD_AVAILABLE = False
try:
    from feature_selection.opt_thresh import optimize_threshold
    OPTIMIZE_THRESHOLD_AVAILABLE = True
except ImportError:
    import warnings
    from dataclasses import dataclass
    
    warnings.warn(
        "feature_selection.opt_thresh not found. Using simplified threshold optimization. "
        "For full functionality, ensure the opt_thresh module is available."
    )
    
    @dataclass
    class OptimizationResult:
        """Simple result container for threshold optimization."""
        profit_factor: float
        threshold: float = 0.0
        n_trades: int = 0
        
    def optimize_threshold(feature, target, min_kept=0.1, **kwargs):
        """Simplified threshold optimization when opt_thresh is not available.
        
        Args:
            feature: Feature values
            target: Target returns
            min_kept: Minimum fraction of samples to keep (0-1)
            **kwargs: Ignored, for compatibility only
            
        Returns:
            OptimizationResult with profit_factor, threshold, and n_trades
        """
        # Simple median threshold
        threshold = feature.median()
        signals = (feature > threshold).astype(int)
        returns = signals * target
        
        # Calculate profit factor
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        profit_factor = gains / max(losses, 1e-10)
        
        return OptimizationResult(
            profit_factor=profit_factor,
            threshold=threshold,
            n_trades=len(signals[signals != 0])
        )


class BarPermute:
    """
    Bar permutation class that shuffles price bars while preserving bar structure.
    
    This follows the C++ algorithm:
    1. Convert bars to relative changes (open-to-open, high-open, low-open, close-open)
    2. Shuffle the changes (separately for intrabar and close-to-open gaps)
    3. Rebuild bars from shuffled changes
    
    This is more realistic than feature permutation because:
    - It preserves the feature extraction logic
    - It destroys predictive relationships in the price data itself
    - Features are re-computed from shuffled bars
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        permute_start_idx: int = 0
    ):
        """
        Initialize bar permutation.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'datetime']
            permute_start_idx: Index to start permuting from (preserves earlier data)
        """
        self.df = df.copy()
        self.permute_start_idx = permute_start_idx
        self.n_bars = len(df)
        
        # Store basis bar (the bar before permutation starts)
        if permute_start_idx > 0:
            self.basis_idx = permute_start_idx - 1
        else:
            self.basis_idx = 0
            
        # Compute and store relative changes
        self._compute_changes()
    
    def _compute_changes(self):
        """
        Compute relative price changes for permutation.
        
        Following C++ logic:
        - rel_open[i] = open[i] - close[i-1]  (close-to-open gap)
        - rel_high[i] = high[i] - open[i]     (intrabar high)
        - rel_low[i] = low[i] - open[i]       (intrabar low)
        - rel_close[i] = close[i] - open[i]   (intrabar close)
        """
        n = self.n_bars
        start = self.permute_start_idx
        
        # Pre-allocate arrays for changes
        self.rel_open = np.zeros(n)
        self.rel_high = np.zeros(n)
        self.rel_low = np.zeros(n)
        self.rel_close = np.zeros(n)
        
        # Get price arrays
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        closes = self.df['close'].values
        
        # Compute changes from permute_start_idx onwards
        for i in range(start, n):
            if i > 0:
                self.rel_open[i] = opens[i] - closes[i-1]
            else:
                self.rel_open[i] = 0  # First bar has no previous close
            
            self.rel_high[i] = highs[i] - opens[i]
            self.rel_low[i] = lows[i] - opens[i]
            self.rel_close[i] = closes[i] - opens[i]
    
    def permute(self) -> pd.DataFrame:
        """
        Shuffle bar changes and rebuild bars.
        
        Returns:
            DataFrame with shuffled bars
        """
        start = self.permute_start_idx
        n = self.n_bars
        
        # Extract the changes that will be shuffled
        n_to_shuffle = n - start
        
        if n_to_shuffle > 1:
            # Create shuffle index for intrabar changes
            intrabar_shuffle = np.arange(n_to_shuffle)
            np.random.shuffle(intrabar_shuffle)
            
            # Create shuffle index for overnight gaps
            gap_shuffle = np.arange(n_to_shuffle)
            np.random.shuffle(gap_shuffle)
            
            # Extract and shuffle the changes
            # Intrabar changes (shuffled together to preserve bar structure)
            shuffled_high = self.rel_high[start:][intrabar_shuffle]
            shuffled_low = self.rel_low[start:][intrabar_shuffle]
            shuffled_close = self.rel_close[start:][intrabar_shuffle]
            
            # Overnight gaps (shuffled separately)
            shuffled_open = self.rel_open[start:][gap_shuffle]
        else:
            # Not enough data to shuffle
            shuffled_high = self.rel_high[start:]
            shuffled_low = self.rel_low[start:]
            shuffled_close = self.rel_close[start:]
            shuffled_open = self.rel_open[start:]
        
        # Rebuild bars from shuffled changes
        new_df = self.df.copy()
        
        # CRITICAL: Preserve the original sequential datetime column!
        # The datetime should NOT shuffle with the bars.
        # This ensures that features based on datetime (like random_feature)
        # are not artificially paired with the bar's OHLC data.
        original_datetimes = self.df['datetime'].values.copy()
        
        # Create new arrays for reconstructed prices
        opens = new_df['open'].values.copy()
        highs = new_df['high'].values.copy()
        lows = new_df['low'].values.copy()
        closes = new_df['close'].values.copy()
        
        # Reconstruct bars starting from permute_start_idx using SHUFFLED changes
        for i in range(start, n):
            idx = i - start  # Index into shuffled arrays
            
            if i > 0:
                # Use the reconstructed close price from previous bar + shuffled gap
                opens[i] = closes[i-1] + shuffled_open[idx]
            
            # Apply shuffled intrabar changes
            highs[i] = opens[i] + shuffled_high[idx]
            lows[i] = opens[i] + shuffled_low[idx]
            closes[i] = opens[i] + shuffled_close[idx]
        
        # Update dataframe with reconstructed OHLC
        new_df['open'] = opens
        new_df['high'] = highs
        new_df['low'] = lows
        new_df['close'] = closes
        
        # CRITICAL: Restore original sequential datetimes
        # This breaks the datetime-OHLC pairing that was causing spurious correlations
        new_df['datetime'] = original_datetimes
        
        return new_df


class BarPermuteWalkForward:
    """
    Walk-forward bar permutation class that shuffles bars independently within each training window.
    
    This prevents data contamination in walk-forward permutation tests by ensuring that:
    1. Each training window's bars are shuffled only among themselves
    2. No information from future periods leaks into earlier training sets
    3. Test periods remain completely untouched
    
    This is critical for realistic walk-forward validation where we want to test if
    the price structure within each training period contains predictive information,
    without allowing future data to contaminate the permutation.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        train_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ):
        """
        Initialize walk-forward bar permutation.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'datetime']
            train_windows: List of (start, end) datetime tuples defining training windows
                          to shuffle independently
        """
        self.df = df.copy()
        self.train_windows = train_windows
        self.n_bars = len(df)
        
        # Ensure datetime column is timezone-aware for comparison
        if 'datetime' not in self.df.columns:
            raise ValueError("DataFrame must have 'datetime' column")
        
        # Convert datetime to pandas Timestamp if needed
        if not isinstance(self.df['datetime'].iloc[0], pd.Timestamp):
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
    
    def _get_window_indices(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[int, int]:
        """
        Get the start and end indices for a datetime window.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            Tuple of (start_idx, end_idx) where end_idx is exclusive
        """
        # Handle timezone compatibility
        df_tz = self.df['datetime'].iloc[0].tz if hasattr(self.df['datetime'].iloc[0], 'tz') else None
        start_tz = start.tz if hasattr(start, 'tz') else None
        
        # Use replace(tzinfo=...) for datetime objects, not tz_localize() which is for pandas DatetimeIndex
        if df_tz is not None and start_tz is None:
            start = start.replace(tzinfo=df_tz)
            end = end.replace(tzinfo=df_tz)
        elif df_tz is None and start_tz is not None:
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        
        # Find indices
        mask = (self.df['datetime'] >= start) & (self.df['datetime'] < end)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return (0, 0)
        
        return (indices[0], indices[-1] + 1)
    
    def _shuffle_window(
        self,
        start_idx: int,
        end_idx: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> None:
        """
        Shuffle bars within a specific window in-place.
        
        Args:
            start_idx: Start index of window (inclusive)
            end_idx: End index of window (exclusive)
            opens: Array of open prices (modified in-place)
            highs: Array of high prices (modified in-place)
            lows: Array of low prices (modified in-place)
            closes: Array of close prices (modified in-place)
        """
        if end_idx <= start_idx or end_idx - start_idx < 2:
            return  # Not enough bars to shuffle
        
        # Compute relative changes within this window
        n_window = end_idx - start_idx
        rel_open = np.zeros(n_window)
        rel_high = np.zeros(n_window)
        rel_low = np.zeros(n_window)
        rel_close = np.zeros(n_window)
        
        # Compute changes for this window
        for i in range(n_window):
            abs_idx = start_idx + i
            
            if i > 0:
                # Use previous bar's close within the window
                rel_open[i] = opens[abs_idx] - closes[abs_idx - 1]
            else:
                # First bar in window: use gap from previous bar outside window
                if start_idx > 0:
                    rel_open[i] = opens[abs_idx] - closes[abs_idx - 1]
                else:
                    rel_open[i] = 0
            
            rel_high[i] = highs[abs_idx] - opens[abs_idx]
            rel_low[i] = lows[abs_idx] - opens[abs_idx]
            rel_close[i] = closes[abs_idx] - opens[abs_idx]
        
        # Shuffle the changes independently
        intrabar_shuffle = np.arange(n_window)
        np.random.shuffle(intrabar_shuffle)
        
        gap_shuffle = np.arange(n_window)
        np.random.shuffle(gap_shuffle)
        
        shuffled_high = rel_high[intrabar_shuffle]
        shuffled_low = rel_low[intrabar_shuffle]
        shuffled_close = rel_close[intrabar_shuffle]
        shuffled_open = rel_open[gap_shuffle]
        
        # Reconstruct bars within this window
        for i in range(n_window):
            abs_idx = start_idx + i
            
            if i > 0:
                # Use reconstructed close from previous bar in window
                opens[abs_idx] = closes[abs_idx - 1] + shuffled_open[i]
            else:
                # First bar: use the gap from the bar before the window
                if start_idx > 0:
                    opens[abs_idx] = closes[start_idx - 1] + shuffled_open[i]
                # else: keep original open for first bar in dataset
            
            highs[abs_idx] = opens[abs_idx] + shuffled_high[i]
            lows[abs_idx] = opens[abs_idx] + shuffled_low[i]
            closes[abs_idx] = opens[abs_idx] + shuffled_close[i]
    
    def permute(self) -> pd.DataFrame:
        """
        Shuffle bars independently within each training window.
        
        Returns:
            DataFrame with shuffled bars (training windows shuffled, test periods preserved)
        """
        new_df = self.df.copy()
        
        # Preserve original datetimes
        original_datetimes = self.df['datetime'].values.copy()
        
        # Get price arrays
        opens = new_df['open'].values.copy()
        highs = new_df['high'].values.copy()
        lows = new_df['low'].values.copy()
        closes = new_df['close'].values.copy()
        
        # Shuffle each training window independently
        for train_start, train_end in self.train_windows:
            start_idx, end_idx = self._get_window_indices(train_start, train_end)
            
            if end_idx > start_idx:
                self._shuffle_window(start_idx, end_idx, opens, highs, lows, closes)
        
        # Update dataframe with shuffled OHLC
        new_df['open'] = opens
        new_df['high'] = highs
        new_df['low'] = lows
        new_df['close'] = closes
        
        # Restore original sequential datetimes
        new_df['datetime'] = original_datetimes
        
        return new_df




def _get_available_features(ticker: Ticker, base_tf: TimeFrame = TimeFrame.D) -> List[str]:
    """
    Get list of available feature names from bias strategies.
    
    This extracts all possible feature column names that will be generated
    by the bias strategies defined in get_bias_nodes.
    
    Args:
        ticker: Ticker symbol
        base_tf: Base timeframe
        
    Returns:
        List of feature column names
    """
    from utils.helpers import get_bias_nodes
    
    bias_strategies = get_bias_nodes(ticker)
    available_features = []
    
    # Iterate through bias strategies to extract feature names
    # Feature names follow pattern: {strategy_key}_{timeframe_name}_{param_info}
    for strategy_key, (timeframes, params) in bias_strategies.items():
        for tf in timeframes:
            # Basic feature name pattern
            if tf == base_tf:
                # This is a simplified approach - actual feature names depend on
                # what each bias node outputs. Some nodes output multiple features.
                # For now, we'll just add the basic strategy key pattern
                feature_name = f"{strategy_key}_{tf.name}"
                available_features.append(feature_name)
    
    return available_features


def bar_permutation_test(
    ticker: Ticker,
    feature_cols: Optional[List[str]] = None,
    start: datetime = datetime(2000, 1, 1),
    end: datetime = datetime.now(),
    base_tf: TimeFrame = TimeFrame.D,
    target_col: str = 'target',
    atr_feature: str = 'atr_252_D_atr_pct_252',
    permute_start_idx: int = 252,
    floor: float = 0.1,
    alpha: float = 0.1,
    nreps: int = 100,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform bar permutation test by shuffling price bars and re-extracting features.
    
    This is a more realistic permutation test than feature shuffling because:
    - It tests whether the PRICE DATA contains predictive information
    - Features are re-computed from shuffled bars, not just shuffled directly
    - It preserves the feature extraction logic and dependencies
    
    Algorithm:
    1. Load original price bars and extract features
    2. Filter features to only those from bias strategies
    3. Compute criterion (e.g., profit factor) for each feature
    4. For each permutation:
        a. Shuffle price bars (preserving bar structure)
        b. Re-extract features from shuffled bars
        c. Re-compute criterion for each feature
        d. Count how often permuted >= original
    5. Compute p-values and significance
    
    Args:
        ticker: Ticker symbol to test
        feature_cols: List of feature column names to test (optional, uses all if None)
        start: Start date for data
        end: End date for data
        base_tf: Base timeframe for feature extraction
        target_col: Name of target column (default: 'target')
        atr_feature: Name of ATR feature for target normalization (default: 'atr_252_D_atr_pct_252')
        permute_start_idx: Index to start permuting from (preserves earlier data for warmup)
        floor: Minimum fraction (0-0.5) of cases that must trade (default: 0.1)
        alpha: Significance level (0-1) (default: 0.1)
        nreps: Number of permutation replications (default: 10)
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress messages
        
    Returns:
        pd.DataFrame: Results with columns:
            - 'feature': Feature name
            - 'original_pf': Original profit factor
            - 'pval': Permutation p-value
            - 'significant': Whether feature passes test (pval <= alpha)
            
    Example:
        >>> from utils.enums import Ticker, TimeFrame
        >>> results = bar_permutation_test(
        ...     ticker=Ticker.SPY,
        ...     feature_cols=['bb_20_D_bb_20', 'dc_50_D_dc_50'],
        ...     base_tf=TimeFrame.D,
        ...     atr_feature='atr_252_D_atr_pct_252',
        ...     nreps=100,
        ...     alpha=0.05
        ... )
        >>> selected = results[results['significant']]
        >>> print(f"Selected {len(selected)} features")
    """
    
    if verbose:
        print("="*80)
        print("BAR PERMUTATION TEST")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ticker: {ticker.name}")
        print(f"Base timeframe: {base_tf.name}")
        print(f"Date range: {start.date()} to {end.date()}")
        print(f"ATR feature for target normalization: {atr_feature}")
        print(f"Permute start index: {permute_start_idx}")
        print(f"Floor (min fraction to trade): {floor}")
        print(f"Alpha level: {alpha}")
        print(f"Number of replications: {nreps}")
        if random_seed is not None:
            print(f"Random seed: {random_seed}")
        print("="*80)
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # STEP 1: Extract features from ORIGINAL bars
    if verbose:
        print("\nSTEP 1: Extracting features from original bars")
        print("="*80)
    
    # Load original price data
    price_df = load_data(ticker, base_tf, start=start, end=end)
    price_df = price_df.reset_index(drop=True)
    
    # Compute target
    price_df['target'] = np.log(price_df['close'] / price_df['open'])
    
    # Extract bias features using the existing infrastructure
    if verbose:
        print("Extracting bias features (this may take a moment)...")
        if feature_cols is not None:
            print(f"  Filtering to {len(feature_cols)} requested features for performance")
    
    # features_df, cols_dict = extract_bias(ticker=ticker, start=start, end=end)
    features_df = pd.DataFrame()
    
    # Join price data with features
    price_df.set_index('datetime', inplace=True)
    price_df.index = price_df.index.tz_localize('UTC')
    full_df = price_df.join(features_df, how='inner')
    full_df = full_df.fillna(0)
    
    if verbose:
        print(f"Extracted {len(full_df)} bars with {len(features_df.columns)} features")
    
    # Filter feature_cols to only include features from bias strategies
    # Get all available bias strategy features from get_bias_nodes
    from utils.helpers import get_bias_nodes
    bias_strategies = get_bias_nodes(ticker)
    
    # If no features specified, use all available features (excluding ATR and other special cols)
    if feature_cols is None:
        # Use all columns except price data, target, and index columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', target_col, 'timestamp']
        feature_cols = [col for col in full_df.columns if col not in exclude_cols]
        if verbose:
            print(f"No features specified, testing all {len(feature_cols)} available features")
    else:
        # Match features using prefix matching (user can pass 'rsi_5' to match 'rsi_5_D_rsi_5')
        original_count = len(feature_cols)
        original_names = feature_cols.copy()
        matched_features = []
        
        for requested_name in original_names:
            # Try exact match first
            if requested_name in full_df.columns:
                matched_features.append(requested_name)
            else:
                # Try prefix match - find all columns that start with requested_name
                prefix_matches = [col for col in full_df.columns if col.startswith(requested_name + '_')]
                if prefix_matches:
                    matched_features.extend(prefix_matches)
                    if verbose and len(prefix_matches) > 1:
                        print(f"  '{requested_name}' matched {len(prefix_matches)} features: {prefix_matches}")
                elif verbose:
                    print(f"  Warning: No match found for '{requested_name}'")
        
        feature_cols = matched_features
        
        if verbose:
            if len(feature_cols) < original_count:
                print(f"Matched {len(feature_cols)} features from {original_count} requested names")
            elif len(feature_cols) > original_count:
                print(f"Matched {len(feature_cols)} features from {original_count} requested names (some matched multiple)")
            else:
                print(f"Testing {len(feature_cols)} specified features")
    
    # Ensure ATR feature is always included (needed for target computation)
    if atr_feature not in feature_cols and atr_feature in full_df.columns:
        feature_cols.append(atr_feature)
        if verbose:
            print(f"Added ATR feature '{atr_feature}' to feature list (required for target normalization)")
    
    # Validate we have features to test
    if len(feature_cols) == 0:
        raise ValueError("No valid features to test after filtering")
    
    # Compute min_kept based on data size
    n_cases = len(full_df)
    min_kept = int(floor * n_cases + 0.5)
    
    if verbose:
        print(f"Number of cases: {n_cases}")
        print(f"Minimum cases to keep per threshold: {min_kept}")
    
    # STEP 2: Compute ORIGINAL criteria for all features
    if verbose:
        print("\nSTEP 2: Computing original (unpermuted) criteria")
        print("="*80)
    
    original_results = []
    
    for i, col in enumerate(feature_cols):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing feature {i+1}/{len(feature_cols)}: {col}")
        
        try:
            # Use optimize_threshold to compute profit factors
            result = optimize_threshold(full_df[col], full_df[target_col], min_kept=min_kept)
            
            original_results.append({
                'feature': col,
                'long_pf': result['pf_high'],
                'short_pf': result['pf_low'],
                'best_pf': result['max_pf'],
            })
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to optimize feature '{col}': {str(e)}")
            continue
    
    results_df = pd.DataFrame(original_results)
    
    if len(results_df) == 0:
        raise ValueError("No features could be successfully optimized")
    
    if verbose:
        print(f"\nSuccessfully optimized {len(results_df)} features")
        print(f"\nTop 5 features by best profit factor:")
        top5 = results_df.nlargest(5, 'best_pf')[['feature', 'best_pf']]
        print(top5.to_string(index=False))
    
    # STEP 3: Run bar permutation replications
    if verbose:
        print(f"\nSTEP 3: Running {nreps-1} bar permutation replications")
        print("="*80)
        print("NOTE: This FULLY RE-EXTRACTS all bias features for each permutation.")
        print("This is computationally intensive but provides a truly realistic test.")
        print("Each replication shuffles price bars and recomputes all features.")
        print("Consider using fewer replications (10-50) for initial testing.")
        print("="*80)
    
    # Initialize counters (includes original unpermuted run)
    results_df['long_count'] = 1
    results_df['short_count'] = 1
    results_df['best_count'] = 1
    
    # Create bar permuter
    permuter = BarPermute(price_df.reset_index(), permute_start_idx=permute_start_idx)
    
    # Run permutation replications
    for irep in range(nreps - 1):
        if verbose and (irep + 1) % 10 == 0:
            print(f"Replication {irep+1}/{nreps-1}...")
        
        # Shuffle bars
        permuted_bars = permuter.permute()
        
        # FULLY RE-EXTRACT features from permuted bars using backtester
        # This is the REALISTIC approach - all bias features are recomputed
        # from shuffled price data, testing whether the price structure itself
        # contains predictive information
        try:
            if verbose and (irep + 1) % 10 == 0:
                print(f"  Re-extracting features from permuted bars...")
            
            # Extract features from permuted bars using backtester
            # Pass feature_cols to only compute needed features (performance optimization)
            permuted_features_df, permuted_price_df = _extract_features_from_bars(
                df=permuted_bars,
                ticker=ticker,
                start=start,
                end=end,
                base_tf=base_tf,
                atr_feature=atr_feature,
                feature_filter=feature_cols,
                verbose=False  # Suppress logging during permutations
            )
            
            # Join permuted features with permuted price data
            # Handle timezone matching - features_df from ml_manager likely has UTC timezone
            if permuted_features_df.index.tz is not None:
                # Features has timezone, ensure price_df matches
                if permuted_price_df.index.tz is None:
                    permuted_price_df.index = permuted_price_df.index.tz_localize('UTC')
            else:
                # Features has no timezone, ensure price_df also has none
                if permuted_price_df.index.tz is not None:
                    permuted_price_df.index = permuted_price_df.index.tz_localize(None)
            
            permuted_full_df = permuted_price_df.join(permuted_features_df, how='inner')
            # Use infer_objects to avoid FutureWarning
            permuted_full_df = permuted_full_df.fillna(0).infer_objects(copy=False)
            
            # Validate we have the required feature columns
            missing_in_perm = [col for col in feature_cols if col not in permuted_full_df.columns]
            if missing_in_perm:
                if verbose:
                    print(f"  Warning: Features missing in permuted data, skipping replication {irep+1}")
                continue
            
            # Compute criteria for all features with FULLY RE-EXTRACTED features
            for idx, row in results_df.iterrows():
                feature_name = row['feature']
                
                try:
                    # Use optimize_threshold to compute profit factors
                    perm_result = optimize_threshold(
                        permuted_full_df[feature_name],
                        permuted_full_df[target_col],
                        min_kept=min_kept
                    )
                    
                    perm_long_pf = perm_result['pf_high']
                    perm_short_pf = perm_result['pf_low']
                    perm_best_pf = perm_result['max_pf']
                    
                    # Count how often permuted >= original
                    if perm_long_pf >= row['long_pf']:
                        results_df.loc[idx, 'long_count'] += 1
                    if perm_short_pf >= row['short_pf']:
                        results_df.loc[idx, 'short_count'] += 1
                    if perm_best_pf >= row['best_pf']:
                        results_df.loc[idx, 'best_count'] += 1
                        
                except Exception as e:
                    # If criterion computation fails for this feature, skip it
                    if verbose and (irep + 1) == 1:
                        print(f"  Warning: Failed to compute criterion for '{feature_name}': {str(e)}")
                    continue
                    
        except Exception as e:
            # If feature extraction fails completely, skip this replication
            if verbose:
                print(f"  Warning: Failed to extract features for replication {irep+1}: {str(e)}")
            continue
    
    # STEP 4: Finalize results
    if verbose:
        print(f"\nCompleted {nreps-1} permutations")
        print("\nSTEP 4: Computing p-values")
        print("="*80)
    
    # Compute p-values
    results_df['long_pval'] = results_df['long_count'] / nreps
    results_df['short_pval'] = results_df['short_count'] / nreps
    results_df['best_pval'] = results_df['best_count'] / nreps
    
    # Determine preferred strategy
    results_df['preferred_strategy'] = results_df.apply(
        lambda row: 'long' if row['long_pf'] >= row['short_pf'] else 'short',
        axis=1
    )
    
    # Mark significant features
    results_df['significant'] = results_df.apply(
        lambda row: row['long_pval'] <= alpha if row['preferred_strategy'] == 'long'
                    else row['short_pval'] <= alpha,
        axis=1
    )
    
    # Drop count columns
    results_df.drop(columns=['long_count', 'short_count', 'best_count'], inplace=True)
    
    # Sort by best_pf
    results_df = results_df.sort_values('best_pf', ascending=False).reset_index(drop=True)
    
    n_significant = results_df['significant'].sum()
    
    if verbose:
        print(f"\nNumber of significant features (p <= {alpha}): {n_significant}")
        
        if n_significant > 0:
            print(f"\nSignificant features:")
            sig_features = results_df[results_df['significant']][[
                'feature', 'best_pf', 'preferred_strategy', 'long_pval', 'short_pval'
            ]]
            print(sig_features.to_string(index=False))
        else:
            print("\nNo features passed the test.")
            print(f"Best long p-value: {results_df['long_pval'].min():.4f}")
            print(f"Best short p-value: {results_df['short_pval'].min():.4f}")
        
        print("\n" + "="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    return results_df


class WalkForwardValidator:
    """
    Reusable walk-forward validation engine.
    
    This class encapsulates the walk-forward validation logic to make it
    reusable across different contexts (e.g., feature selection, permutation testing).
    
    The walk-forward approach:
    1. Train on a fixed historical window
    2. Test on the next period immediately after training
    3. Roll forward and repeat
    4. Aggregate all out-of-sample predictions
    """
    
    def __init__(
        self,
        train_start: datetime,
        train_end: datetime,
        test_step: int,
        num_steps: int,
        floor: float = 0.1,
        min_test_samples: int = 10,
        verbose: bool = False
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_start: Start date for initial training period
            train_end: End date for initial training period
            test_step: Number of days for test period (also roll-forward step)
            num_steps: Number of walk-forward steps to perform
            floor: Minimum fraction of training cases to trade
            min_test_samples: Minimum test samples required per strategy
            verbose: Whether to print progress
        """
        self.train_start = train_start
        self.train_end = train_end
        self.test_step = test_step
        self.num_steps = num_steps
        self.floor = floor
        self.min_test_samples = min_test_samples
        self.verbose = verbose
    
    def validate_single_feature(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        force_steps: Optional[List[int]] = None
    ) -> Dict:
        """
        Run walk-forward validation for a single feature.
        
        Args:
            df: DataFrame with features and target (datetime-indexed)
            feature_col: Name of feature column
            target_col: Name of target column
            force_steps: Optional list of step indices to use. If provided, only these
                        steps will be evaluated, ensuring permuted data uses the same
                        folds as the original for fair comparison.
            
        Returns:
            Dict containing:
                - 'all_oos_returns': Concatenated OOS returns across all folds
                - 'oos_profit_factor': Aggregated OOS profit factor
                - 'n_valid_steps': Number of valid walk-forward steps
                - 'step_results': List of results per step
                - 'valid_steps': List of step indices that were valid (for forcing in permutations)
        """
        # Ensure datetime index timezone compatibility
        if df.index.tz is not None:
            if self.train_start.tzinfo is None:
                train_start = self.train_start.replace(tzinfo=df.index.tz)
                train_end = self.train_end.replace(tzinfo=df.index.tz)
            else:
                train_start = self.train_start
                train_end = self.train_end
        else:
            train_start = self.train_start
            train_end = self.train_end
        
        all_oos_returns = []
        step_results = []
        valid_steps = []  # Track which steps were valid
        
        # Determine which steps to evaluate
        steps_to_evaluate = force_steps if force_steps is not None else range(self.num_steps)
        
        for step in steps_to_evaluate:
            # Calculate date ranges for this step
            current_train_start = train_start + timedelta(days=step * self.test_step)
            current_train_end = train_end + timedelta(days=step * self.test_step)
            current_test_start = current_train_end
            current_test_end = current_test_start + timedelta(days=self.test_step)
            
            # Filter data
            train_mask = (df.index >= current_train_start) & (df.index < current_train_end)
            test_mask = (df.index >= current_test_start) & (df.index < current_test_end)
            
            train_feature = df[feature_col][train_mask]
            train_target = df[target_col][train_mask]
            test_feature = df[feature_col][test_mask]
            test_target = df[target_col][test_mask]
            
            # Skip if insufficient data
            if len(train_feature) < 100 or len(test_feature) < 10:
                if self.verbose:
                    print(f"  Step {step}: Skipping (train={len(train_feature)}, test={len(test_feature)})")
                continue
            
            # Optimize thresholds on training data
            min_kept = int(self.floor * len(train_feature) + 0.5)
            
            try:
                train_result = optimize_threshold(
                    train_feature,
                    train_target,
                    min_kept=min_kept
                )
                
                # Apply to test data
                test_result = apply_threshold_strategy(
                    test_feature,
                    test_target,
                    train_result['high_thresh'],
                    train_result['low_thresh'],
                    train_result['sign_flipped']
                )
                
                # Determine which strategy to use (based on training performance)
                use_long = train_result['pf_high'] >= train_result['pf_low']
                
                # Collect OOS returns for the preferred strategy
                if use_long and test_result['n_high'] >= self.min_test_samples:
                    all_oos_returns.append(test_result['high_returns'])
                    step_results.append({
                        'step': step,
                        'strategy': 'long',
                        'pf': test_result['pf_high'],
                        'n_trades': test_result['n_high']
                    })
                    if force_steps is None:  # Only track valid steps if not forcing
                        valid_steps.append(step)
                elif not use_long and test_result['n_low'] >= self.min_test_samples:
                    all_oos_returns.append(test_result['low_returns'])
                    step_results.append({
                        'step': step,
                        'strategy': 'short',
                        'pf': test_result['pf_low'],
                        'n_trades': test_result['n_low']
                    })
                    if force_steps is None:  # Only track valid steps if not forcing
                        valid_steps.append(step)
                else:
                    if self.verbose:
                        print(f"  Step {step}: Insufficient test samples")
                    # If forcing steps, we must skip this step entirely
                    if force_steps is not None:
                        if self.verbose:
                            print(f"  WARNING: Forced step {step} has insufficient samples!")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Step {step}: Error - {str(e)}")
                continue
        
        # Aggregate all OOS returns
        if len(all_oos_returns) > 0:
            all_returns_series = pd.concat(all_oos_returns, axis=0)
            oos_pf = compute_profit_factor(all_returns_series.values)
        else:
            all_returns_series = pd.Series([], dtype=float)
            oos_pf = 0.0
        
        return {
            'all_oos_returns': all_returns_series,
            'oos_profit_factor': oos_pf,
            'n_valid_steps': len(step_results),
            'step_results': step_results,
            'valid_steps': valid_steps  # Return which steps were valid
        }


def _run_single_permutation(
    irep: int,
    permutation_mode: str,
    full_df: pd.DataFrame,
    feature_cols: List[str],
    atr_feature: str,
    price_df: Optional[pd.DataFrame],
    permute_start_idx: int,
    ticker: Ticker,
    start: datetime,
    end: datetime,
    base_tf: TimeFrame,
    validator: WalkForwardValidator,
    target_col: str,
    original_pfs: Dict[str, float],
    random_seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Run a single permutation iteration.
    
    This function is designed to be called in parallel via multiprocessing.
    Each call is completely independent and can run on a separate CPU core.
    
    Args:
        irep: Permutation iteration number (for seeding)
        permutation_mode: 'feature' or 'bar'
        full_df: Original dataframe with features and target
        feature_cols: List of feature names to test
        atr_feature: ATR feature name (not shuffled)
        price_df: Price dataframe (for bar mode, to create new BarPermute)
        permute_start_idx: Index to start permuting from (for bar mode)
        ticker: Ticker symbol
        start: Start date
        end: End date
        base_tf: Base timeframe
        validator: WalkForwardValidator instance
        target_col: Target column name
        original_pfs: Dict mapping feature names to original profit factors
        random_seed: Base random seed (will be offset by irep)
        
    Returns:
        Dict mapping feature names to count (1 if permuted >= original, 0 otherwise)
    """
    # Set random seed for reproducibility (offset by iteration)
    if random_seed is not None:
        np.random.seed(random_seed + irep)
    
    counts = {}
    
    try:
        # Handle permutation based on mode
        if permutation_mode == 'feature':
            # Feature shuffling: just shuffle feature values in place
            permuted_full_df = full_df.copy()
            
            # Shuffle each feature independently
            for feature_name in feature_cols:
                if feature_name != atr_feature:  # Don't shuffle ATR
                    permuted_full_df[feature_name] = np.random.permutation(
                        permuted_full_df[feature_name].values
                    )
        
        else:  # bar mode
            # Bar permutation: shuffle bars and re-extract features
            # CRITICAL: Create a NEW BarPermute instance for each permutation
            # This ensures each worker has its own independent permuter
            permuter = BarPermute(price_df, permute_start_idx=permute_start_idx)
            permuted_bars = permuter.permute()
            
            # Re-extract features from permuted bars
            permuted_features_df, permuted_price_df = _extract_features_from_bars(
                df=permuted_bars,
                ticker=ticker,
                start=start,
                end=end,
                base_tf=base_tf,
                atr_feature=atr_feature,
                feature_filter=feature_cols,
                verbose=False
            )
            
            # Join permuted features with permuted price data
            if permuted_features_df.index.tz is not None:
                if permuted_price_df.index.tz is None:
                    permuted_price_df.index = permuted_price_df.index.tz_localize('UTC')
            else:
                if permuted_price_df.index.tz is not None:
                    permuted_price_df.index = permuted_price_df.index.tz_localize(None)
            
            permuted_full_df = permuted_price_df.join(permuted_features_df, how='inner')
            permuted_full_df = permuted_full_df.fillna(0).infer_objects(copy=False)
            
            # Validate we have required features
            missing_features = [col for col in feature_cols if col not in permuted_full_df.columns]
            if missing_features:
                # Return zeros for all features if extraction failed
                return {feature: 0 for feature in original_pfs.keys()}
        
        # Run walk-forward validation on permuted data for each feature
        for feature_name, original_pf in original_pfs.items():
            try:
                perm_result = validator.validate_single_feature(
                    df=permuted_full_df,
                    feature_col=feature_name,
                    target_col=target_col
                )
                
                # Compare aggregated OOS profit factor
                perm_oos_pf = perm_result['oos_profit_factor']
                
                # Count how often permuted >= original
                counts[feature_name] = 1 if perm_oos_pf >= original_pf else 0
                
            except Exception:
                # If validation fails, count as 0
                counts[feature_name] = 0
    
    except Exception:
        # If entire permutation fails, return zeros
        counts = {feature: 0 for feature in original_pfs.keys()}
    
    return counts



