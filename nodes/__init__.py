from utils.models import Candle
from abc import ABC, abstractmethod
from typing import List, Dict, Type, TypeVar
from utils.enums import Bias, Ticker, TimeFrame
from typing import Any, Dict
import utils.helpers as _helpers


T = TypeVar('T', bound='BiasNode')


class BiasNode(ABC):

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
    