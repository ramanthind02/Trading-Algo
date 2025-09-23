from enum import Enum
from dataclasses import dataclass, field


class TimeFrame(Enum):
    """Simplified TimeFrame enum for daily, weekly, and monthly data only."""
    D = 'D'  # Daily
    W = 'W'  # Weekly
    M = 'M'  # Monthly

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
    NN = 'NIKKEI_225'  # Nikkei 225 Yen Index
    
    # Energy
    CL = 'OIL_CRUDE'  # Crude Oil (WTI)
    NG = 'NAT_GAS'  # Natural Gas
    HO = 'HEATING_OIL'  # ULSD (Heating Oil)
    
    # Metals
    GC = 'GOLD'  # Gold
    HG = 'COPPER'  # Copper
    SI = 'SILVER'  # Silver
    PL = 'PLATINUM'  # Platinum
    PA = 'PALLADIUM'  # Palladium
    
    # Currencies (FX)
    EU = 'EURUSD'  # Euro Currency
    JY = 'YEN'  # Japanese Yen
    BP = 'GBPUSD'  # British Pound
    CD = 'CADUSD'  # Canadian Dollar
    SF = 'CHFUSD'  # Swiss Franc
    AD = 'AUDUSD'  # Australian Dollar
    RP = 'EURGBP'  # Euro/British Pound
    
    # Cryptocurrencies
    BTC = 'BTCUSD'  # Bitcoin
    
    # Agricultural (Food Grains)
    C = 'CORN'  # Corn
    S = 'SOYBEANS'  # Soybean
    W = 'WHEAT'  # Wheat (SRW)
    
    # Softs
    SB = 'SUGAR'  # Sugar
    KC = 'COFFEE'  # Coffee
    CT = 'COTTON'  # Cotton
    
    # Meat
    LC = 'LIVE_CATTLE'  # Live Cattle
    LH = 'LEAN_HOG'  # Lean Hog
    GF = 'FEEDER_CATTLE'  # Feeder Cattle
    
    # Fixed Income
    TY = 'TEN_YR_NOTE'  # 10Yr T.Note
    FV = 'FIVE_YR_NOTE'  # 5Yr T.Note
    US = 'US_BONDS'  # 30Yr T.Bond
    TU = 'TWO_YR_NOTE'  # 2Yr T.Note

    def __lt__(self, other):
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)
