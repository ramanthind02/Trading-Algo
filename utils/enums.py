from enum import Enum
from dataclasses import dataclass, field


class TimeFrame(Enum):
    """Simplified TimeFrame enum for daily, weekly, and monthly data only."""
    D = 60 * 60 * 24
    W = 60 * 60 * 24 * 7
    M = 60 * 60 * 24 * 30


    def __lt__(self, other):
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)

    @classmethod
    def higher_timeframes(cls, current_tf):
        current_index = list(cls).index(current_tf)
        return [tf for tf in cls if list(cls).index(tf) > current_index]

class Ticker(Enum):
    # Equity Indices
    ES = 'US500'  # E-Mini S&P
    NQ = 'US100'  # E-Mini Nasdaq
    YM = 'US30'   # E-Mini Dow
    RTY = 'RUSSELL_2000'  # E-Mini Russell 2000
    
    # Energy
    CL = 'OIL_CRUDE'  # Crude Oil (WTI)
    HO = 'HEATING_OIL'  # ULSD (Heating Oil)
    
    # Metals
    GC = 'GOLD'  # Gold
    HG = 'COPPER'  # Copper
    SI = 'SILVER'  # Silver
    PL = 'PLATINUM'  # Platinum

    
    # Currencies (FX)
    EU = 'EURUSD'  # Euro Currency
    JY = 'YEN'  # Japanese Yen
    BP = 'GBPUSD'  # British Pound
    CD = 'CADUSD'  # Canadian Dollar
    SF = 'CHFUSD'  # Swiss Franc

    
    # Agricultural (Food Grains)
    C = 'CORN'  # Corn
    S = 'SOYBEANS'  # Soybean
    W = 'WHEAT'  # Wheat (SRW)

    # Meat
    GF = 'FEEDER_CATTLE'  # Feeder Cattle
    
    # Fixed Income
    TY = 'TEN_YR_NOTE'  # 10Yr T.Note
    FV = 'FIVE_YR_NOTE'  # 5Yr T.Note
    US = 'US_BONDS'  # 30Yr T.Bond
    TU = 'TWO_YR_NOTE'  # 2Yr T.Note

    def __lt__(self, other):
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)



class Bias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0
    ANY = None


class ResamplingMethod(Enum):
    """Enum for probabilistic resampling methods used in robustness testing."""
    MONTE_CARLO = "monte_carlo"  # Random permutation/shuffle without replacement
    BOOTSTRAP = "bootstrap"  # Simple bootstrap (sampling with replacement)
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Block bootstrap (preserves serial correlation)


class Direction(Enum):
    """
    Trading direction for ensemble strategies.
    
    Ensembles are separated by direction to allow portfolio-level
    allocation between long and short strategies (e.g., 60% long, 40% short).
    """
    
    LONG = 'long'
    SHORT = 'short'
    
    def __str__(self) -> str:
        """String representation returns the value."""
        return self.value
    
    def __lt__(self, other):
        """Enable sorting (LONG before SHORT)."""
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)
    
    @classmethod
    def from_string(cls, direction_str: str) -> 'Direction':
        """
        Convert string to Direction enum (case-insensitive).
        
        Parameters
        ----------
        direction_str : str
            Direction string ('long' or 'short')
            
        Returns
        -------
        Direction
            The matching direction
            
        Raises
        ------
        ValueError
            If direction string is invalid
        """
        direction_lower = direction_str.lower()
        for direction in cls:
            if direction.value == direction_lower:
                return direction
        raise ValueError(
            f"Invalid direction: '{direction_str}'. "
            f"Must be 'long' or 'short'"
        )
