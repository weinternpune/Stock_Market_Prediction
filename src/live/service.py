"""
service.py
----------
High-Level Service Facade for Live Market Data.
Orchestrates provider selection (Demo vs. Real), quote fetching, normalization,
validation, duplicate detection, and snapshot persistence.
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd

from src.live.base import BaseLiveMarketDataProvider
from src.live.demo_provider import DemoMarketDataProvider
from src.live.real_provider import RealMarketDataProvider, RealMarketDataConfigurationError
from src.live.validator import LiveQuoteValidator
from src.live.storage import LiveQuoteStorage
from src.live.models import MarketQuote
from config.live_config import (
    MARKET_DATA_PROVIDER,
    CONSTITUENTS_CSV
)
from src.utilities.logger import get_logger

logger = get_logger("LiveMarketDataService")


class LiveMarketDataService:
    """
    Central orchestration service for live market data operations.
    Seamlessly manages provider lifecycle, validation pipelines, and storage.
    """

    def __init__(
        self,
        provider: Optional[BaseLiveMarketDataProvider] = None,
        validator: Optional[LiveQuoteValidator] = None,
        storage: Optional[LiveQuoteStorage] = None
    ):
        self.validator = validator or LiveQuoteValidator()
        self.storage = storage or LiveQuoteStorage()

        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._initialize_provider_from_config()

        logger.info(
            f"Initialized LiveMarketDataService with provider='{self.provider.provider_name}' "
            f"(is_demo={self.provider.is_demo})"
        )

    def _initialize_provider_from_config(self) -> BaseLiveMarketDataProvider:
        """Instantiates provider according to MARKET_DATA_PROVIDER environment configuration."""
        provider_type = MARKET_DATA_PROVIDER.lower()

        if provider_type == "real":
            try:
                return RealMarketDataProvider()
            except RealMarketDataConfigurationError as e:
                logger.error(f"Cannot initialize RealMarketDataProvider: {e}")
                raise
        else:
            # Default to Demo/Mock Provider
            return DemoMarketDataProvider()

    @property
    def is_demo(self) -> bool:
        """True if the active provider is running in DEMO mode."""
        return self.provider.is_demo

    @property
    def provider_name(self) -> str:
        """Active provider identification string."""
        return self.provider.provider_name

    def get_provider_info(self) -> Dict[str, Any]:
        """Returns metadata on active provider, mode, and market operating status."""
        return {
            "provider_name": self.provider.provider_name,
            "is_demo": self.provider.is_demo,
            "data_source": "DEMO DATA" if self.provider.is_demo else "REAL MARKET FEED",
            "market_status": self.provider.get_market_status(),
            "healthy": self.provider.health_check()
        }

    def fetch_quotes(
        self,
        symbols: Optional[List[str]] = None,
        persist: bool = True
    ) -> Tuple[List[MarketQuote], List[Dict[str, Any]]]:
        """
        Fetches quotes from active provider, applies validation & deduplication,
        and optionally persists snapshots to storage.
        
        Returns:
            (valid_quotes, dropped_audit_records)
        """
        raw_quotes = self.provider.get_quotes(symbols)
        valid_quotes, dropped = self.validator.filter_and_validate(raw_quotes)

        if persist and valid_quotes:
            self.storage.save_quotes(valid_quotes)

        return valid_quotes, dropped

    def fetch_single_quote(self, symbol: str, persist: bool = False) -> Optional[MarketQuote]:
        """Fetches, validates, and returns quote for an individual stock."""
        quote = self.provider.get_quote(symbol)
        if quote is None:
            return None

        is_valid, errors = self.validator.validate_quote(quote)
        if not is_valid:
            logger.warning(f"Quote for {symbol} failed validation: {errors}")
            return None

        if persist:
            self.storage.save_snapshot([quote])

        return quote

    def get_cached_snapshot(self) -> Optional[pd.DataFrame]:
        """Loads previously saved live quote snapshot without making a network or tick request."""
        return self.storage.load_latest_snapshot()

    def get_historical_ticks(self, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Loads historical live ticks from archive."""
        return self.storage.load_history(symbol=symbol)

    def get_constituents(self) -> List[Dict[str, str]]:
        """Returns list of all 50 constituent companies loaded from official metadata."""
        if hasattr(self.provider, "_constituents"):
            return list(self.provider._constituents.values())
        df = pd.read_csv(CONSTITUENTS_CSV)
        return df.to_dict(orient="records")

    def health_check(self) -> bool:
        """Validates provider connectivity."""
        return self.provider.health_check()
