from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import math
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, TypeVar

import pandas as pd

import utils.core.helpers as _helpers
from utils.core.enums import Bias, Ticker, TimeFrame
from utils.core.models import Candle

if TYPE_CHECKING:
    from utils.cache.runtime.bias_node_cache import BiasNodeCache


T = TypeVar('T', bound='BiasNode')


@dataclass(frozen=True)
class LookbackContribution:
    """Machine-readable bar-count contribution used for cold cache rebuilds."""

    label: str
    bars: int


LookbackWindow = LookbackContribution


class BiasNode(ABC):
    lookback_param_names: ClassVar[frozenset[str]] = frozenset()
    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = ()
    cold_rebuild_buffer_ratio: ClassVar[float] = 0.2

    _instances: Dict[tuple, 'BiasNode'] = {}

    def __init__(self, ticker: Ticker, tf: TimeFrame):
        """
        Superclass for bias nodes

        Parameters:
        - ticker (Ticker): The ticker symbol
        - tf (TimeFrame): The timeframe for analysis

        Returns: None
        """
        self.name = self.__class__.__name__
        self.ticker = ticker
        self.tf = tf
        self.bias = Bias.NEUTRAL
        self.output = []
        # Optional standardization metadata (used for column naming)
        self.module_name: str = ''
        self.output_features: list = []  # e.g., ["signal"] or ["atr", "atrPct"]
        self.params: Dict[str, Any] = {}
        self._cache = {
            'last_candle_id': None,
            'last_result': None
        }

        # Bias node cache attributes (initialized after params are set)
        self._bias_node_cache: Optional['BiasNodeCache'] = None
        self._cache_loaded: bool = False
        self._cached_data: Optional['pd.DataFrame'] = None
    
    @classmethod
    def get_instance(cls: Type[T], *args, **kwargs) -> T:
        """
        Singleton implementation to return existing instance or creates new one based on config parameters

        Parameters:
        - args
        - kwargs

        Returns: T
        """
        def make_hashable(obj):
            """Recursively convert objects to hashable types for use as dictionary keys."""
            if isinstance(obj, dict):
                # Convert dict to sorted tuple of (key, value) pairs with hashable values
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, (list, tuple)):
                # Convert list/tuple to tuple with hashable elements
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, set):
                # Convert set to frozenset with hashable elements
                return frozenset(make_hashable(item) for item in obj)
            elif callable(obj):
                # For function objects, use the function name and module
                # This allows same-named functions from different modules to be distinct
                func_name = getattr(obj, '__name__', str(obj))
                func_module = getattr(obj, '__module__', 'unknown')
                return (func_module, func_name)
            else:
                # For primitive types (str, int, float, bool, None) and other hashable types
                # Try to return as-is, but if it's not hashable, convert to string
                try:
                    hash(obj)
                    return obj
                except TypeError:
                    # Object is not hashable, convert to string representation
                    return str(obj)
        
        # Convert args and kwargs to hashable format
        hashable_args = tuple(make_hashable(arg) for arg in args)
        hashable_kwargs = tuple(sorted((k, make_hashable(v)) for k, v in kwargs.items()))
        key = (cls, *hashable_args, hashable_kwargs)
        
        if key not in cls._instances:
            cls._instances[key] = cls(*args, **kwargs)
        return cls._instances[key]

    def add_candle(self, candle: Candle) -> List:
        """
        Wrapper method to implement caching logic for bias nodes

        Parameters:
        - candle (Candle)

        Returns:
        - List
        """
        # Create cache key from datetime and ticker (works for both Candle and FastCandle)
        cache_key = (candle.datetime, candle.ticker)
        
        if self._cache['last_candle_id'] == cache_key:
            return self._cache['last_result']
        result = self._compute_candle(candle)
        self._cache['last_candle_id'] = cache_key
        self._cache['last_result'] = result
        return result

    @abstractmethod
    def _compute_candle(self, candle: Candle) -> List:
        """
        Abstract method to be implemented by subclasses for actual computation

        Parameters:
        - candle (Candle)

        Returns:
        - List
        """
        pass

    # ----------------------------------------------------------------------
    # Lookback metadata API
    # ----------------------------------------------------------------------
    def _coerce_lookback_bars(self, label: str, value: object) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"Lookback contribution '{label}' on {self.__class__.__name__} "
                "must be int-like, not bool."
            )
        if isinstance(value, int):
            bars = value
        elif isinstance(value, float) and value.is_integer():
            bars = int(value)
        else:
            raise TypeError(
                f"Lookback contribution '{label}' on {self.__class__.__name__} "
                f"must be int-like, got {type(value).__name__}."
            )
        if bars < 0:
            raise ValueError(
                f"Lookback contribution '{label}' on {self.__class__.__name__} "
                f"must be non-negative, got {bars}."
            )
        return bars

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        """Allow subclasses to add dynamic or wrapped lookback windows."""
        return ()

    def lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        """Return all declared lookback windows for this node instance."""
        contributions: list[LookbackContribution] = []
        params = self.params if isinstance(self.params, dict) else {}

        for label in sorted(self.lookback_param_names):
            if label not in params:
                continue
            bars = self._coerce_lookback_bars(label, params[label])
            contributions.append(LookbackContribution(label=label, bars=bars))

        for label, raw_bars in self.hardcoded_lookbacks:
            bars = self._coerce_lookback_bars(label, raw_bars)
            contributions.append(LookbackContribution(label=label, bars=bars))

        contributions.extend(self._extra_lookback_contributions())

        front_bad = getattr(self, "front_bad", None)
        if front_bad is not None:
            contributions.append(
                LookbackContribution(
                    label="front_bad",
                    bars=self._coerce_lookback_bars("front_bad", front_bad),
                )
            )

        return tuple(contributions)

    def max_lookback(self) -> int:
        """Return the maximum bar-count needed to warm this node from scratch."""
        return max((item.bars for item in self.lookback_contributions()), default=0)

    def cold_rebuild_candle_count(self) -> int:
        """Return total candles to stream for a stateless cold rebuild."""
        max_lookback = self.max_lookback()
        if max_lookback <= 0:
            return 1
        buffer_bars = max(1, math.ceil(max_lookback * self.cold_rebuild_buffer_ratio))
        return max_lookback + buffer_bars

    # ----------------------------------------------------------------------
    # Standardized column naming API
    # ----------------------------------------------------------------------
    def get_column_names(self) -> List[str]:
        """
        Return standardized column names for this node if metadata is available.
        Falls back to existing self.columns if standardization metadata is missing.
        """
        try:
            if hasattr(self, 'output_features') and self.output_features and hasattr(self, 'module_name') and self.module_name:
                # Build names using standardized helper
                names: List[str] = []
                for feat in self.output_features:
                    names.append(
                        _helpers.build_feature_column_name(
                            module=self.module_name,
                            feature=feat,
                            tf=self.tf,
                            params=self.params if isinstance(self.params, dict) else {}
                        )
                    )
                return names
        except Exception:
            # If anything goes wrong, fall back
            pass

        # Fallback to legacy columns if present
        return getattr(self, 'columns', [])

    def ensure_standardized_columns(self) -> None:
        """Set self.columns to standardized names when possible."""
        try:
            names = self.get_column_names()
            if names:
                self.columns = names
        except Exception:
            # Do not raise to preserve backwards compatibility
            pass

    # ----------------------------------------------------------------------
    # Bias Node Cache API
    # ----------------------------------------------------------------------
    def _init_cache_after_params(self) -> None:
        """
        Initialize the bias node cache after params are set.

        This method should be called by subclasses at the END of their
        __init__ method, after setting module_name and params.

        If the cache file exists, it will be loaded automatically.

        Example usage in subclass __init__:
            def __init__(self, ticker, tf, lookback=14):
                super().__init__(ticker, tf)
                self.module_name = 'rsi'
                self.params = {'lookback': lookback}
                # ... other initialization ...
                self._init_cache_after_params()  # Call at end
        """
        # Only initialize cache if module_name and params are set
        if not self.module_name:
            return

        try:
            from utils.cache.runtime.bias_node_cache import BiasNodeCache

            self._bias_node_cache = BiasNodeCache(
                module_name=self.module_name,
                params=self.params,
                ticker=self.ticker,
                tf=self.tf
            )

            # Load cache if it exists
            if self._bias_node_cache.exists():
                self._cached_data = self._bias_node_cache.load()
                self._cache_loaded = True

        except ImportError:
            # BiasNodeCache not available, continue without caching
            pass
        except Exception as e:
            # Log warning but don't fail initialization
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to initialize cache for {self.module_name}: {e}")

    def _central_cache_descriptor(self):
        from utils.cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope

        return ArtifactDescriptor(
            family="bias",
            ticker=self.ticker,
            timeframe=self.tf,
            module_name=self.module_name or None,
            params=self.params,
            scope=ArtifactScope.LIVE,
            artifact_name=self.module_name or None,
        )

    def _raise_cache_miss(
        self,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
        reason: str,
    ) -> None:
        from utils.cache.runtime.bias_node_cache import CacheMissError

        raise CacheMissError(
            module_name=self.module_name,
            params=self.params,
            ticker=self.ticker,
            tf=self.tf,
            date_range=(start, end),
            cache_path=self.get_cache_path(),
            reason=reason,
        )

    def _read_cached_artifact(
        self,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> "pd.DataFrame":
        from utils.cache.runtime.central_cache import CentralCacheStore
        from utils.cache.runtime.central_cache_errors import CacheCoverageError
        from utils.cache.runtime.central_cache_models import CacheRequest

        store = CentralCacheStore.get_instance()
        descriptor = self._central_cache_descriptor()
        clamped_request = CacheRequest(start=start, end=end)

        if start is not None or end is not None:
            record = store.describe_artifact(descriptor)
            if record is not None:
                effective_start = start
                effective_end = end
                if start is not None and record.coverage.start is not None:
                    effective_start = max(
                        pd.Timestamp(start),
                        pd.Timestamp(record.coverage.start),
                    ).to_pydatetime()
                if end is not None and record.coverage.end is not None:
                    effective_end = min(
                        pd.Timestamp(end),
                        pd.Timestamp(record.coverage.end),
                    ).to_pydatetime()
                if (
                    effective_start is not None
                    and effective_end is not None
                    and pd.Timestamp(effective_start) > pd.Timestamp(effective_end)
                ):
                    raise CacheCoverageError(
                        module_name=self.module_name,
                        ticker=self.ticker,
                        timeframe=self.tf,
                        start=start,
                        end=end,
                        coverage_start=record.coverage.start,
                        coverage_end=record.coverage.end,
                    )
                clamped_request = CacheRequest(start=effective_start, end=effective_end)

        return store.read_artifact(
            descriptor,
            request=clamped_request,
        )

    def get_cached_values(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        require_cache: bool = True
    ) -> Optional['pd.Series']:
        """
        Get cached output values for a date range.

        This is the primary API for vectorized access to bias node outputs.
        Use this instead of streaming candles when cache is available.

        Parameters
        ----------
        start : datetime, optional
            Start of date range (inclusive). If None, uses earliest available.
        end : datetime, optional
            End of date range (inclusive). If None, uses latest available.
        require_cache : bool, default=True
            If True and cache is missing: logs warning and raises CacheMissError.
            If False and cache is missing: returns None.

        Returns
        -------
        pd.Series or None
            Cached output values indexed by datetime.
            Returns None if cache is missing and require_cache=False.

        Raises
        ------
        CacheMissError
            If require_cache=True and cache is missing or incomplete.

        Example
        -------
        >>> # Vectorized access (fast)
        >>> values = bias_node.get_cached_values(start, end)
        >>> if values is not None:
        ...     features = values.reindex(candles_df['datetime'])
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check if cache is initialized
        if self._bias_node_cache is None:
            if self.module_name:
                self._init_cache_after_params()
        
        if self._bias_node_cache is None:
            if require_cache:
                logger.warning(
                    f"Cache not initialized for {self.module_name} ({self.ticker}, {self.tf}). "
                    f"Call _init_cache_after_params() in subclass __init__."
                )
                self._raise_cache_miss(
                    start=start,
                    end=end,
                    reason="Cache not initialized. Subclass must call _init_cache_after_params().",
                )
            return None

        from utils.cache.runtime.central_cache_errors import (
            ArtifactLifecycleError,
            ArtifactMissingError,
            CacheCoverageError,
        )

        try:
            cached_df = self._read_cached_artifact(start=start, end=end)
        except (ArtifactLifecycleError, ArtifactMissingError, CacheCoverageError) as exc:
            if not require_cache:
                return None
            self._raise_cache_miss(start=start, end=end, reason=str(exc))

        self._cached_data = cached_df
        self._cache_loaded = True

        if 'value' in cached_df.columns:
            return cached_df['value']
        if len(cached_df.columns) == 1:
            return cached_df.iloc[:, 0]
        return cached_df.iloc[:, 0]

    def get_cached_dataframe(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        require_cache: bool = True
    ) -> Optional['pd.DataFrame']:
        """
        Get cached output as DataFrame (for multi-output nodes).

        Similar to get_cached_values() but returns all output columns
        for nodes with multiple outputs (e.g., EWMAC with signal and signalBool).

        Parameters
        ----------
        start : datetime, optional
            Start of date range (inclusive)
        end : datetime, optional
            End of date range (inclusive)
        require_cache : bool, default=True
            If True and cache is missing: raises CacheMissError.

        Returns
        -------
        pd.DataFrame or None
            All cached columns for the date range.
        """
        if self._bias_node_cache is None:
            if self.module_name:
                self._init_cache_after_params()

        if self._bias_node_cache is None:
            if require_cache:
                self._raise_cache_miss(start=start, end=end, reason="Cache not initialized")
            return None

        from utils.cache.runtime.central_cache_errors import (
            ArtifactLifecycleError,
            ArtifactMissingError,
            CacheCoverageError,
        )

        try:
            cached_df = self._read_cached_artifact(start=start, end=end)
        except (ArtifactLifecycleError, ArtifactMissingError, CacheCoverageError) as exc:
            if not require_cache:
                return None
            self._raise_cache_miss(start=start, end=end, reason=str(exc))

        self._cached_data = cached_df
        self._cache_loaded = True
        return cached_df

    def is_cache_loaded(self) -> bool:
        """
        Check if cache data is loaded.

        Returns
        -------
        bool
            True if cache exists and has been loaded.
        """
        return self._cache_loaded

    def cache_exists(self) -> bool:
        """
        Check if cache file exists on disk.

        Returns
        -------
        bool
            True if cache file exists.
        """
        if self._bias_node_cache is None:
            return False
        return self._bias_node_cache.exists()

    def get_cache_path(self) -> Optional[str]:
        """
        Get the cache file path.

        Returns
        -------
        str or None
            Path to cache file, or None if cache not initialized.
        """
        if self._bias_node_cache is None:
            return None
        return self._bias_node_cache.cache_path
