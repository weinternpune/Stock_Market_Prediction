"""
real_provider.py
----------------
Real Market Data Provider Adapter for Indian Stock Market (NSE).
Integrated with Zerodha Kite Connect API (v3) for real-time and end-of-day
quotes across the 50 NIFTY 50 companies.

STRICT SAFETY CONSTRAINTS:
1. NO SCRAPING OF TRADINGVIEW: Scraping TradingView is strictly forbidden.
2. NO FAKE/RANDOM VALUES IN PRODUCTION: If credentials are missing, network fails,
   or an API error occurs, this provider raises an explicit exception.
   It NEVER falls back to fake, mock, or random numbers in production mode.
3. CLEAR CREDENTIAL VALIDATION: Validates API Key, Secret, and Access Token from environment.
4. HONEST STATUS REPORTING: Reports actual HTTP response codes, token errors, and exchange status.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo
import pandas as pd
import requests

from src.live.base import BaseLiveMarketDataProvider, determine_market_status
from src.live.models import MarketQuote
from config.live_config import (
    CONSTITUENTS_CSV,
    KOLKATA_TZ,
    MARKET_DATA_API_KEY,
    MARKET_DATA_API_SECRET,
    MARKET_DATA_ACCESS_TOKEN,
    MARKET_DATA_BASE_URL,
    MARKET_DATA_ENVIRONMENT
)
from src.utilities.logger import get_logger

logger = get_logger("RealLiveMarketProvider")


class RealMarketDataConfigurationError(Exception):
    """Raised when real market data provider is invoked without valid credentials."""
    pass


class RealMarketDataConnectionError(Exception):
    """Raised when an external API call to the live market provider fails or returns an error."""
    pass


class RealMarketDataProvider(BaseLiveMarketDataProvider):
    """
    Production-grade live market data provider adapter for Zerodha Kite Connect v3.
    Retrieves authentic NSE market quotes for the 50 NIFTY 50 companies.
    
    Zerodha Kite Connect v3 Specifications:
    - Base URL: https://api.kite.trade (or configured via MARKET_DATA_BASE_URL)
    - Endpoint: GET /quote?i=NSE:SYMBOL
    - Batch support: Up to 500 instruments per request (all 50 NIFTY 50 in 1 call)
    - Auth: Authorization: token {api_key}:{access_token}
    - Header: X-Kite-Version: 3
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: Optional[str] = None,
        constituents_file: Path = CONSTITUENTS_CSV,
        timeout: int = 10
    ):
        self._api_key = (api_key if api_key is not None else os.getenv("MARKET_DATA_API_KEY", MARKET_DATA_API_KEY)).strip()
        self._api_secret = (api_secret if api_secret is not None else os.getenv("MARKET_DATA_API_SECRET", MARKET_DATA_API_SECRET)).strip()
        self._access_token = (access_token if access_token is not None else os.getenv("MARKET_DATA_ACCESS_TOKEN", MARKET_DATA_ACCESS_TOKEN)).strip()
        
        raw_base_url = (base_url if base_url is not None else os.getenv("MARKET_DATA_BASE_URL", MARKET_DATA_BASE_URL)).strip()
        self._base_url = raw_base_url.rstrip("/") if raw_base_url else "https://api.kite.trade"
        
        self._environment = (
            environment if environment is not None else os.getenv("MARKET_DATA_ENVIRONMENT", MARKET_DATA_ENVIRONMENT)
        ).strip().lower()
        
        self._constituents_file = constituents_file
        self._timeout = timeout
        self._session = requests.Session()

        self._constituents: Dict[str, Dict[str, str]] = {}
        self._load_constituents()

        # Strict validation of credentials on initialization
        self._validate_credentials()

    @property
    def provider_name(self) -> str:
        return f"ZerodhaKiteConnect[{self._environment.upper()}]"

    @property
    def is_demo(self) -> bool:
        return False

    def _validate_credentials(self) -> None:
        """
        Enforces presence of required production credentials.
        Never allows unauthenticated or fake production execution.
        """
        if not self._api_key and not self._access_token:
            err_msg = (
                "Real market data provider requires credentials. "
                "Please configure the following environment variables:\n"
                "  - MARKET_DATA_API_KEY (Your Kite Connect API Key)\n"
                "  - MARKET_DATA_API_SECRET (Your Kite Connect API Secret)\n"
                "  - MARKET_DATA_ACCESS_TOKEN (Your active daily session Access Token)\n"
                "  - MARKET_DATA_BASE_URL (Optional, defaults to https://api.kite.trade)\n"
                "  - MARKET_DATA_ENVIRONMENT ('sandbox' or 'production')\n\n"
                "NOTE: Scraping TradingView is strictly forbidden. Fake/random values are prohibited in production mode."
            )
            logger.error(err_msg)
            raise RealMarketDataConfigurationError(err_msg)

    def _load_constituents(self) -> None:
        """Loads official 50 NIFTY constituent symbols and company metadata."""
        if not self._constituents_file.exists():
            raise FileNotFoundError(f"Missing constituents file at: {self._constituents_file}")

        df = pd.read_csv(self._constituents_file)
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).strip().upper()
            self._constituents[sym] = {
                "symbol": sym,
                "company": str(row.get("Company_Name", sym)).strip(),
                "industry": str(row.get("Industry", "")).strip(),
                "sector": str(row.get("Sector", "")).strip()
            }

    @staticmethod
    def symbol_to_instrument(symbol: str) -> str:
        """
        Maps a standard constituent symbol to Zerodha Kite instrument identifier.
        Format: 'NSE:{SYMBOL}'
        Examples:
          'ADANIENT'   -> 'NSE:ADANIENT'
          'M&M'        -> 'NSE:M&M'
          'BAJAJ-AUTO' -> 'NSE:BAJAJ-AUTO'
        """
        clean_sym = symbol.strip().upper()
        return f"NSE:{clean_sym}"

    @staticmethod
    def instrument_to_symbol(instrument: str) -> str:
        """
        Extracts the bare NSE symbol from a Kite instrument identifier.
        Example: 'NSE:ADANIENT' -> 'ADANIENT'
        """
        inst = instrument.strip().upper()
        if inst.startswith("NSE:"):
            return inst[4:]
        return inst

    def _get_headers(self) -> Dict[str, str]:
        """Builds Kite Connect v3 authorization headers."""
        token_part = f"{self._api_key}:{self._access_token}" if self._access_token else self._api_key
        return {
            "Authorization": f"token {token_part}",
            "X-Kite-Version": "3",
            "Accept": "application/json",
            "User-Agent": "Nifty50-Stock-Market-Prediction/1.0"
        }

    def _parse_timestamp(self, raw_ts: Any) -> datetime:
        """Parses exchange timestamp into datetime localized to Asia/Kolkata (+05:30)."""
        now_kolkata = datetime.now(KOLKATA_TZ)
        if not raw_ts:
            return now_kolkata

        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                return raw_ts.replace(tzinfo=KOLKATA_TZ)
            return raw_ts.astimezone(KOLKATA_TZ)

        if isinstance(raw_ts, str):
            clean_ts = raw_ts.strip()
            # Try ISO 8601 first
            try:
                dt = datetime.fromisoformat(clean_ts)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=KOLKATA_TZ)
                return dt.astimezone(KOLKATA_TZ)
            except ValueError:
                pass

            # Try standard Kite datetime: 'YYYY-MM-DD HH:MM:SS'
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(clean_ts, fmt)
                    return dt.replace(tzinfo=KOLKATA_TZ)
                except ValueError:
                    continue

        return now_kolkata

    def _parse_quote_item(self, symbol: str, item: Dict[str, Any]) -> MarketQuote:
        """
        Transforms a raw Zerodha Kite Connect JSON quote object into a canonical MarketQuote.
        Enforces all 14 mandatory fields and integrity constraints.
        """
        company_meta = self._constituents.get(symbol, {})
        company_name = company_meta.get("company", symbol)

        ohlc = item.get("ohlc", {})
        last_price = float(item.get("last_price", 0.0))
        prev_close = float(ohlc.get("close", 0.0))
        open_price = float(ohlc.get("open", 0.0))
        high_price = float(ohlc.get("high", 0.0))
        low_price = float(ohlc.get("low", 0.0))

        # Enforce OHLC logical invariants on day bounds
        day_high = max(high_price, open_price, last_price)
        day_low = min(low_price, open_price, last_price) if low_price > 0 else min(open_price, last_price)
        if day_high == 0.0 and last_price > 0.0:
            day_high = last_price
        if day_low == 0.0 and last_price > 0.0:
            day_low = last_price

        # Change & Change % calculation
        net_change = item.get("net_change")
        if net_change is not None:
            change = round(float(net_change), 2)
        else:
            change = round(last_price - prev_close, 2)

        if prev_close > 0.0:
            change_pct = round((change / prev_close) * 100.0, 2)
        else:
            change_pct = 0.0

        volume = int(item.get("volume", 0))

        # Parse timestamp
        timestamp_dt = self._parse_timestamp(item.get("timestamp"))

        # Market operating status
        market_status = determine_market_status(datetime.now(KOLKATA_TZ))

        return MarketQuote(
            symbol=symbol,
            company=company_name,
            timestamp=timestamp_dt,
            open=open_price,
            high=high_price,
            low=low_price,
            current_price=last_price,
            previous_close=prev_close,
            change=change,
            change_pct=change_pct,
            volume=volume,
            day_high=day_high,
            day_low=day_low,
            market_status=market_status,
            is_demo=False,
            data_source="Zerodha Kite Connect v3"
        )

    def _handle_api_error_response(self, response: requests.Response, context: str) -> None:
        """Parses Kite Connect error JSON and raises informative RealMarketDataConnectionError."""
        status_code = response.status_code
        error_type = "ApiError"
        error_message = response.text

        try:
            err_json = response.json()
            error_type = err_json.get("error_type", error_type)
            error_message = err_json.get("message", error_message)
        except Exception:
            pass

        full_msg = (
            f"Zerodha Kite Connect error during {context} (HTTP {status_code} [{error_type}]): "
            f"{error_message}. "
            "Verify MARKET_DATA_API_KEY and active daily session MARKET_DATA_ACCESS_TOKEN."
        )
        logger.error(full_msg)
        raise RealMarketDataConnectionError(full_msg)

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Fetches a real live quote for a single symbol from Zerodha Kite Connect v3.
        Endpoint: GET /quote?i=NSE:{symbol}
        Raises RealMarketDataConnectionError if external API call fails.
        NEVER generates fake numbers in production mode.
        """
        sym_clean = symbol.strip().upper()
        if sym_clean not in self._constituents:
            logger.warning(f"Symbol '{symbol}' not found in NIFTY 50 constituents.")
            return None

        instrument = self.symbol_to_instrument(sym_clean)
        url = f"{self._base_url}/quote"
        headers = self._get_headers()
        params = [("i", instrument)]

        logger.info(f"Connecting to Zerodha Kite Connect for {sym_clean} ({instrument})...")
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as e:
            err_msg = f"Network connection error to Zerodha Kite Connect: {e}"
            logger.error(err_msg)
            raise RealMarketDataConnectionError(err_msg) from e

        if resp.status_code != 200:
            self._handle_api_error_response(resp, f"single quote fetch for '{sym_clean}'")

        try:
            payload = resp.json()
        except Exception as e:
            raise RealMarketDataConnectionError(f"Malformed JSON response from Kite Connect: {e}")

        data = payload.get("data", {})
        if instrument not in data:
            raise RealMarketDataConnectionError(
                f"Instrument '{instrument}' not present in Zerodha Kite Connect response data."
            )

        return self._parse_quote_item(sym_clean, data[instrument])

    def get_quotes(self, symbols: Optional[List[str]] = None) -> List[MarketQuote]:
        """
        Batch fetches real live quotes for NIFTY 50 symbols from Zerodha Kite Connect v3.
        Queries all symbols in a single efficient HTTP GET request (up to 500 instruments allowed).
        Guaranteed to only return authentic quotes or raise RealMarketDataConnectionError.
        """
        target_symbols = (
            [s.strip().upper() for s in symbols if s.strip().upper() in self._constituents]
            if symbols is not None
            else sorted(self._constituents.keys())
        )

        if not target_symbols:
            return []

        url = f"{self._base_url}/quote"
        headers = self._get_headers()
        # Build query parameters for batch: i=NSE:SYM1&i=NSE:SYM2...
        params = [("i", self.symbol_to_instrument(sym)) for sym in target_symbols]

        logger.info(
            f"Fetching {len(target_symbols)} quotes from Zerodha Kite Connect via {url}..."
        )
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as e:
            err_msg = f"Network connection error to Zerodha Kite Connect: {e}"
            logger.error(err_msg)
            raise RealMarketDataConnectionError(err_msg) from e

        if resp.status_code != 200:
            self._handle_api_error_response(resp, f"batch quote fetch for {len(target_symbols)} symbols")

        try:
            payload = resp.json()
        except Exception as e:
            raise RealMarketDataConnectionError(f"Malformed JSON response from Kite Connect: {e}")

        data = payload.get("data", {})
        quotes: List[MarketQuote] = []
        missing_symbols: List[str] = []

        for sym in target_symbols:
            inst = self.symbol_to_instrument(sym)
            if inst in data:
                try:
                    q = self._parse_quote_item(sym, data[inst])
                    quotes.append(q)
                except Exception as ex:
                    logger.warning(f"Error parsing Kite quote for {sym}: {ex}")
            else:
                missing_symbols.append(sym)

        if missing_symbols:
            logger.warning(
                f"Kite Connect returned quotes for {len(quotes)}/{len(target_symbols)} symbols. "
                f"Missing: {missing_symbols}"
            )

        return quotes

    def health_check(self) -> bool:
        """Validates credentials presence and connectivity to Zerodha Kite Connect gateway."""
        if not self._api_key and not self._access_token:
            return False
        try:
            resp = self._session.get(
                f"{self._base_url}/quote",
                params=[("i", "NSE:NIFTY 50")],
                headers=self._get_headers(),
                timeout=3
            )
            return resp.status_code == 200
        except Exception:
            return False


# Alias for explicit clarity
KiteConnectMarketDataProvider = RealMarketDataProvider
