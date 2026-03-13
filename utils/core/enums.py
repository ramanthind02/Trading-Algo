from enum import Enum
from dataclasses import dataclass, field


class TimeFrame(Enum):
    """TimeFrame enum covering intraday (M1-H4) and daily+ (D/W/M) data."""
    M1  = 60
    M2  = 60 * 2
    M3  = 60 * 3
    M4  = 60 * 4
    M5  = 60 * 5
    M6  = 60 * 6
    M7  = 60 * 7
    M8  = 60 * 8
    M9  = 60 * 9
    M10 = 60 * 10
    M15 = 60 * 15
    M30 = 60 * 30
    H1  = 60 * 60
    H2  = 60 * 60 * 2
    H4  = 60 * 60 * 4
    D   = 60 * 60 * 24
    W   = 60 * 60 * 24 * 7
    M   = 60 * 60 * 24 * 30


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
    TLT = 'TWENTY_YR_NOTE'  # 20Yr T.Note

    def __lt__(self, other):
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)



class Bias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0
    ANY = None


class PositionMode(Enum):
    """
    Position mode for bias nodes that can restrict output to long-only or short-only.
    Used in BasicBreakout / BasicMR: LONG_ONLY clamps raw -1 to 0; SHORT_ONLY clamps raw 1 to 0.
    """
    LONG_SHORT = "long_short"   # Default: output 1, 0, -1
    LONG_ONLY = "long_only"     # Clamp raw -1 to 0
    SHORT_ONLY = "short_only"   # Clamp raw 1 to 0


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
    LONG_SHORT allows rule-based models that output -1, 0, 1 (short, flat, long).
    """
    LONG = 'long'
    SHORT = 'short'
    LONG_SHORT = 'long_short'

    def __str__(self) -> str:
        """String representation returns the value."""
        return self.value

    def __lt__(self, other):
        """Enable sorting (definition order: LONG, SHORT, LONG_SHORT)."""
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)

    @classmethod
    def from_string(cls, direction_str: str) -> 'Direction':
        """
        Convert string to Direction enum (case-insensitive).
        
        Parameters
        ----------
        direction_str : str
            Direction string ('long', 'short', or 'long_short' / 'both')
            
        Returns
        -------
        Direction
            The matching direction
            
        Raises
        ------
        ValueError
            If direction string is invalid
        """
        direction_lower = direction_str.lower().strip()
        if direction_lower in ('both', 'long_short'):
            return cls.LONG_SHORT
        for direction in cls:
            if direction.value == direction_lower:
                return direction
        raise ValueError(
            f"Invalid direction: '{direction_str}'. "
            f"Must be 'long', 'short', or 'long_short'/'both'"
        )
