"""
Feature Extraction Cache

Caches extracted features to avoid recomputation when multiple base models
share the same bias node spec, ticker, and date range.

Author: Trading Research Team
Date: 2025-01-22
"""

import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
from utils.enums import Ticker, TimeFrame


class FeatureCache:
    """
    Singleton cache for extracted features.
    
    Caches features by (bias_node_spec, ticker, sorted_datetimes) to avoid
    recomputing features when multiple base models share the same configuration.
    
    Cache Key Components:
    - bias_node_spec_hash: Hash of (module_name, timeframes, params)
    - ticker: Ticker enum
    - sorted_datetimes: Sorted tuple of unique datetimes (for exact matching)
    """
    
    _instance: Optional['FeatureCache'] = None
    _cache: Dict[Tuple, pd.Series]
    _hits: int
    _misses: int
    
    def __new__(cls) -> 'FeatureCache':
        """Singleton pattern: return existing instance or create new one."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._hits = 0
            cls._instance._misses = 0
        return cls._instance
    
    def get_cache_key(
        self,
        bias_node_spec: Dict[str, Any],
        ticker: Ticker,
        candles_df: pd.DataFrame
    ) -> Tuple:
        """
        Generate hashable cache key from bias node spec, ticker, and candles.
        
        Parameters
        ----------
        bias_node_spec : Dict[str, Any]
            Bias node specification with keys: module_name, timeframes, params
        ticker : Ticker
            Ticker enum
        candles_df : pd.DataFrame
            Candles DataFrame with 'datetime' column
            
        Returns
        -------
        Tuple
            Hashable cache key: (bias_node_spec_hash, ticker, sorted_datetimes_tuple)
        """
        # Hash bias node spec
        spec_hash = self._hash_bias_node_spec(bias_node_spec)
        
        # Extract sorted datetimes for exact matching
        sorted_datetimes = self._hash_datetime_range(candles_df)
        
        # Create cache key
        cache_key = (spec_hash, ticker, sorted_datetimes)
        
        return cache_key
    
    def _hash_bias_node_spec(self, spec: Dict[str, Any]) -> int:
        """
        Create hashable representation of bias node spec.
        
        Parameters
        ----------
        spec : Dict[str, Any]
            Bias node spec with module_name, timeframes, params
            
        Returns
        -------
        int
            Hash of the spec
        """
        module_name = spec.get('module_name', '')
        timeframes = spec.get('timeframes', [])
        params = spec.get('params', {})
        
        # Convert timeframes to hashable format (sorted tuple of names)
        tf_names = tuple(sorted(tf.name if hasattr(tf, 'name') else str(tf) for tf in timeframes))
        
        # Convert params to hashable format (frozenset of sorted items)
        # Handle nested dicts and lists in params
        params_hashable = self._make_hashable(params)
        
        # Create hashable tuple
        spec_tuple = (module_name, tf_names, params_hashable)
        
        return hash(spec_tuple)
    
    def _make_hashable(self, obj: Any) -> Any:
        """
        Convert object to hashable format.
        
        Handles dicts, lists, and other types recursively.
        
        Parameters
        ----------
        obj : Any
            Object to make hashable
            
        Returns
        -------
        Any
            Hashable representation
        """
        if isinstance(obj, dict):
            # Sort items and make values hashable
            return frozenset(
                (k, self._make_hashable(v))
                for k, v in sorted(obj.items())
            )
        elif isinstance(obj, (list, tuple)):
            # Convert to tuple and make elements hashable
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Already hashable
            return obj
        elif hasattr(obj, 'name'):
            # Handle enums (e.g., TimeFrame)
            return obj.name
        else:
            # Fallback: convert to string
            return str(obj)
    
    def _hash_datetime_range(self, candles_df: pd.DataFrame) -> Tuple:
        """
        Generate hashable key from candle datetimes.
        
        Uses sorted tuple of unique datetimes for exact matching.
        This ensures cache hits when the same candles are used.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            DataFrame with 'datetime' column
            
        Returns
        -------
        Tuple
            Sorted tuple of unique datetimes (for exact matching)
        """
        if candles_df.empty or 'datetime' not in candles_df.columns:
            return tuple()
        
        # Get sorted unique datetimes
        datetimes = pd.to_datetime(candles_df['datetime']).unique()
        sorted_dts = tuple(sorted(datetimes))
        
        return sorted_dts
    
    def get(
        self,
        bias_node_spec: Dict[str, Any],
        ticker: Ticker,
        candles_df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """
        Retrieve cached features if available.
        
        Parameters
        ----------
        bias_node_spec : Dict[str, Any]
            Bias node specification
        ticker : Ticker
            Ticker enum
        candles_df : pd.DataFrame
            Candles DataFrame
            
        Returns
        -------
        Optional[pd.Series]
            Cached features if found, None otherwise
        """
        cache_key = self.get_cache_key(bias_node_spec, ticker, candles_df)
        result = self._cache.get(cache_key)
        
        # Track cache statistics
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        
        return result
    
    def set(
        self,
        bias_node_spec: Dict[str, Any],
        ticker: Ticker,
        candles_df: pd.DataFrame,
        features: pd.Series
    ) -> None:
        """
        Store features in cache.
        
        Parameters
        ----------
        bias_node_spec : Dict[str, Any]
            Bias node specification
        ticker : Ticker
            Ticker enum
        candles_df : pd.DataFrame
            Candles DataFrame
        features : pd.Series
            Extracted features to cache
        """
        cache_key = self.get_cache_key(bias_node_spec, ticker, candles_df)
        self._cache[cache_key] = features.copy()
    
    def clear(self) -> None:
        """Clear all cached features and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_cache_size(self) -> int:
        """
        Get number of cached entries.
        
        Returns
        -------
        int
            Number of cached feature series
        """
        return len(self._cache)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit/miss rates.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with cache statistics including:
            - num_entries: Number of cached entries
            - total_memory_bytes: Total memory used
            - total_memory_mb: Total memory in MB
            - hits: Number of cache hits
            - misses: Number of cache misses
            - total_requests: Total cache requests (hits + misses)
            - hit_rate: Cache hit rate as percentage
        """
        total_size = len(self._cache)
        total_memory = sum(
            series.memory_usage(deep=True) if isinstance(series, pd.Series) else 0
            for series in self._cache.values()
        )
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'num_entries': total_size,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024) if total_memory > 0 else 0,
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate
        }
