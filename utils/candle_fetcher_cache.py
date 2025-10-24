"""
Candle Fetcher Cache for Performance Optimization.

This module provides a singleton cache for CandleFetcher instances to avoid
reloading the same data multiple times during permutation tests.

Expected speedup: 15-20% for permutation tests with bar mode.
"""

from typing import Dict, Tuple
from utils.enums import Ticker, TimeFrame
from utils.candle_fetcher import CandleFetcher


class CandleFetcherCache:
    """
    Singleton cache for CandleFetcher instances.
    
    This cache stores CandleFetcher instances keyed by (ticker, timeframes tuple)
    to avoid reloading the same data multiple times.
    
    Usage:
        cache = CandleFetcherCache.get_instance()
        fetcher = cache.get_fetcher(Ticker.ES, [TimeFrame.W, TimeFrame.M])
    """
    
    _instance = None
    
    def __init__(self):
        """Initialize the cache."""
        self._cache: Dict[Tuple, CandleFetcher] = {}
        self._hits = 0
        self._misses = 0
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_fetcher(
        self,
        ticker: Ticker,
        tfs: list,
        force_reload: bool = False
    ) -> CandleFetcher:
        """
        Get a CandleFetcher instance from cache or create new one.
        
        Args:
            ticker: Ticker symbol
            tfs: List of timeframes to fetch
            force_reload: If True, bypass cache and create new fetcher
            
        Returns:
            CandleFetcher instance
        """
        # Create cache key
        tfs_tuple = tuple(sorted(tfs, key=lambda x: x.value))
        cache_key = (ticker, tfs_tuple)
        
        # Check if force reload
        if force_reload:
            self._misses += 1
            fetcher = CandleFetcher(ticker=ticker, tfs=list(tfs_tuple))
            self._cache[cache_key] = fetcher
            return fetcher
        
        # Check cache
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        
        # Cache miss - create new fetcher
        self._misses += 1
        fetcher = CandleFetcher(ticker=ticker, tfs=list(tfs_tuple))
        self._cache[cache_key] = fetcher
        
        return fetcher
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache hits, misses, and hit rate
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'total': total,
            'hit_rate': hit_rate,
            'cached_items': len(self._cache)
        }
    
    def __repr__(self):
        """String representation."""
        stats = self.get_stats()
        return (
            f"CandleFetcherCache("
            f"hits={stats['hits']}, "
            f"misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1f}%, "
            f"cached={stats['cached_items']})"
        )


# Convenience function for easy access
def get_cached_fetcher(
    ticker: Ticker,
    tfs: list,
    use_cache: bool = True
) -> CandleFetcher:
    """
    Get a CandleFetcher instance with optional caching.
    
    Args:
        ticker: Ticker symbol
        tfs: List of timeframes
        use_cache: Whether to use cache (default: True)
        
    Returns:
        CandleFetcher instance
    """
    if use_cache:
        cache = CandleFetcherCache.get_instance()
        return cache.get_fetcher(ticker, tfs)
    else:
        return CandleFetcher(ticker=ticker, tfs=tfs)


def clear_cache():
    """Clear the candle fetcher cache."""
    cache = CandleFetcherCache.get_instance()
    cache.clear()


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache = CandleFetcherCache.get_instance()
    return cache.get_stats()
