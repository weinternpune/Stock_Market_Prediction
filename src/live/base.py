"""
base.py
-------
Abstract Base Class for Live Market Data Providers.
Defines the required contract for fetching single quotes, batch quotes,
market operating status, and health connectivity checks.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict
from zoneinfo import ZoneInfo

from src.live.models import MarketQuote
from config.live_config import (
    KOLKATA_TZ,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    PRE_OPEN_START_HOUR,
    PRE_OPEN_START_MINUTE
)

def determine_market_status(now_ist: Optional[datetime] = None) -> str:
    """
    Determines NSE market operating state based on Asia/Kolkata official hours.
    - Regular Trading: 09:15 to 15:30 IST (Monday - Friday)
    - Pre-Open: 09:00 to 09:15 IST (Monday - Friday)
    - Closed: Weekends or outside regular hours
    """
    if now_ist is None:
        now_ist = datetime.now(KOLKATA_TZ)
    elif now_ist.tzinfo is None:
        now_ist = now_ist.replace(tzinfo=KOLKATA_TZ)
    else:
        now_ist = now_ist.astimezone(KOLKATA_TZ)

    # 0 = Monday, 6 = Sunday
    if now_ist.weekday() >= 5:
        return "CLOSED (WEEKEND)"

    time_mins = now_ist.hour * 60 + now_ist.minute
    pre_open_start = PRE_OPEN_START_HOUR * 60 + PRE_OPEN_START_MINUTE
    market_open = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
    market_close = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE

    if pre_open_start <= time_mins < market_open:
        return "PRE_OPEN"
    elif market_open <= time_mins <= market_close:
        return "OPEN"
    else:
        return "CLOSED"


class BaseLiveMarketDataProvider(ABC):
    """
    Abstract Interface for Market Data Providers.
    All live providers (Demo, Kite, Upstox, etc.) must implement these methods.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the data provider (e.g. 'DemoMockProvider', 'UpstoxProvider')."""
        pass

    @property
    @abstractmethod
    def is_demo(self) -> bool:
        """True if the provider is producing simulated demo data, False if real market feed."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Fetches the latest market quote for a single symbol.
        Returns MarketQuote instance, or None if symbol not found / error.
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: Optional[List[str]] = None) -> List[MarketQuote]:
        """
        Fetches the latest quotes for a list of symbols (or all 50 NIFTY constituents if None).
        Returns a list of MarketQuote instances.
        """
        pass

    def get_market_status(self) -> str:
        """Returns standard current market operating state in Asia/Kolkata."""
        return determine_market_status()

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the provider is operational and accessible."""
        pass
