from utils.models import Candle
from abc import ABC, abstractmethod
from typing import List, Dict, Type, TypeVar
from utils.enums import Bias, Ticker, TimeFrame


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
        key = (cls, *args, tuple(sorted(kwargs.items())))
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
        if self._cache['last_candle_id'] == candle.id:
            return self._cache['last_result']
        result = self._compute_candle(candle)
        self._cache['last_candle_id'] = candle.id
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
    