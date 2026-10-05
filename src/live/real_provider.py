"""
real_provider.py
----------------
Real Market Data Provider Adapters for Indian Stock Market (NSE).
Features primary integration with the official Groww Trade API (`growwapi`),
with secondary support for Zerodha Kite Connect v3.

STRICT SAFETY CONSTRAINTS:
1. NO SCRAPING OF TRADINGVIEW: Scraping TradingView is strictly forbidden.
2. NO FAKE/RANDOM VALUES IN PRODUCTION: If credentials are missing, network fails,
   or an API error occurs, this provider raises an explicit exception.
   It NEVER falls back to fake, mock, or random numbers in production mode.
3. CLEAR CREDENTIAL VALIDATION: Validates API Key, Secret, and Access Token from environment.
4. HONEST STATUS REPORTING: Reports actual HTTP response codes, token errors, and exchange status.
5. NEVER EXPOSE OR LOG API KEYS OR SECRETS.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import pyotp
from growwapi import GrowwAPI
from growwapi.groww.exceptions import (
    GrowwAPIException,
    GrowwAPIAuthenticationException,
    GrowwAPIAuthorisationException,
    GrowwAPIRateLimitException,
    GrowwAPIBadRequestException,
    GrowwAPINotFoundException,
    GrowwAPITimeoutException
)

from src.live.base import BaseLiveMarketDataProvider, determine_market_status
from src.live.models import MarketQuote
from config.live_config import (
    CONSTITUENTS_CSV,
    KOLKATA_TZ,
    GROWW_API_KEY,
    GROWW_API_SECRET,
    GROWW_ACCESS_TOKEN,
    GROWW_BASE_URL,
    MARKET_DATA_API_KEY,
    MARKET_DATA_API_SECRET,
    MARKET_DATA_ACCESS_TOKEN,
    MARKET_DATA_BASE_URL,
    MARKET_DATA_ENVIRONMENT
)
from src.utilities.logger import get_logger

logger = get_logger("RealLiveMarketProvider")


# ==============================================================================
# Custom Exceptions for Real Market Data
# ==============================================================================

class RealMarketDataConfigurationError(Exception):
    """Raised when real market data provider is invoked without valid credentials."""
    pass


class RealMarketDataConnectionError(Exception):
    """Raised when an external API call to the live market provider fails or returns an error."""
    pass


class GrowwMarketDataConfigurationError(RealMarketDataConfigurationError):
    """Raised when Groww API credentials are missing or improperly configured."""
    pass


class GrowwMarketDataAuthenticationError(RealMarketDataConnectionError):
    """Raised when Groww API authentication fails (HTTP 401 / Invalid TOTP / Token expired)."""
    pass


class GrowwMarketDataAuthorisationError(RealMarketDataConnectionError):
    """Raised when Groww API access is forbidden (HTTP 403 / Missing live market data subscription)."""
    pass


class GrowwMarketDataRateLimitError(RealMarketDataConnectionError):
    """Raised when Groww API rate limits are exceeded (HTTP 429)."""
    pass


# ==============================================================================
# Groww Trade API Provider (Primary Implementation)
# ==============================================================================

class GrowwMarketDataProvider(BaseLiveMarketDataProvider):
    """
    Production-grade live market data provider adapter for the Groww Trade API (`growwapi`).
    Retrieves authentic NSE market quotes for the 50 NIFTY 50 companies.

    Groww Trade API Architecture:
    - Official Library: `growwapi` (v1.5.0+) with `pyotp` for TOTP authentication
    - Auth: TOTP Flow (`api_key` + `totp=pyotp.TOTP(secret).now()`) or Approval Secret
    - Rate Limits:
        * Authentication: 5 req/s, 30 req/min, 150 token requests per 24 hours
        * Live Data: 10 req/s, 300 req/min
    - Single Quote: `groww.get_quote(trading_symbol="RELIANCE", exchange="NSE", segment="CASH")`
    - Batch OHLC/LTP: `groww.get_ohlc(...)` and `groww.get_ltp(...)` supporting up to 50 instruments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        constituents_file: Path = CONSTITUENTS_CSV,
        timeout: int = 15,
        groww_client: Optional[Any] = None
    ):
        self._api_key = (
            api_key if api_key is not None else os.getenv("GROWW_API_KEY", GROWW_API_KEY)
        ).strip()
        self._api_secret = (
            api_secret if api_secret is not None else os.getenv("GROWW_API_SECRET", GROWW_API_SECRET)
        ).strip()
        self._access_token = (
            access_token if access_token is not None else os.getenv("GROWW_ACCESS_TOKEN", GROWW_ACCESS_TOKEN)
        ).strip()
        
        raw_base_url = (
            base_url if base_url is not None else os.getenv("GROWW_BASE_URL", GROWW_BASE_URL)
        ).strip()
        self._base_url = raw_base_url.rstrip("/") if raw_base_url else "https://api.groww.in"

        self._constituents_file = constituents_file
        self._timeout = timeout
        self._groww: Optional[GrowwAPI] = groww_client

        self._constituents: Dict[str, Dict[str, str]] = {}
        self._load_constituents()

        # Validate credential availability (without logging or exposing values)
        self._validate_credentials()

        # If a pre-initialized groww_client was not injected, ensure session can be created
        if self._groww is None and self._access_token:
            self._groww = GrowwAPI(self._access_token)

    @property
    def provider_name(self) -> str:
        return "GrowwTradeAPI[PRODUCTION]"

    @property
    def is_demo(self) -> bool:
        return False

    def _validate_credentials(self) -> None:
        """
        Enforces presence of required Groww production credentials.
        Never allows unauthenticated execution or fake production numbers.
        """
        if not self._access_token and (not self._api_key or not self._api_secret):
            err_msg = (
                "Groww Trade API provider requires credentials. "
                "Please configure the following environment variables in your .env file:\n"
                "  - GROWW_API_KEY (Your Groww API Key or TOTP Token)\n"
                "  - GROWW_API_SECRET (Your 32-character TOTP Secret Key or Approval Secret)\n"
                "  - GROWW_ACCESS_TOKEN (Optional: Pre-generated active session token)\n\n"
                "NOTE: Scraping TradingView is strictly forbidden. Fake/random values are prohibited in production mode."
            )
            logger.error(err_msg)
            raise GrowwMarketDataConfigurationError(err_msg)

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
    def symbol_to_trading_symbol(symbol: str) -> str:
        """
        Normalizes a symbol to the standard NSE trading symbol used in Groww get_quote.
        Examples:
          'NSE:ADANIENT' -> 'ADANIENT'
          'NSE_ADANIENT' -> 'ADANIENT'
          'M&M'          -> 'M&M'
          'BAJAJ-AUTO'   -> 'BAJAJ-AUTO'
        """
        sym = symbol.strip().upper()
        if sym.startswith("NSE:"):
            sym = sym[4:]
        elif sym.startswith("NSE_"):
            sym = sym[4:]
        return sym

    @staticmethod
    def symbol_to_batch_exchange_symbol(symbol: str) -> str:
        """
        Maps a constituent symbol to the Groww batch exchange_trading_symbols format: 'NSE_{SYMBOL}'.
        Examples:
          'RELIANCE'   -> 'NSE_RELIANCE'
          'M&M'        -> 'NSE_M&M'
          'BAJAJ-AUTO' -> 'NSE_BAJAJ-AUTO'
        """
        clean = GrowwMarketDataProvider.symbol_to_trading_symbol(symbol)
        return f"NSE_{clean}"

    def _get_or_create_session(self) -> GrowwAPI:
        """
        Initializes or caches the authenticated GrowwAPI instance.
        Respects the 150 token requests per 24 hours rate limit by persisting
        the generated session token in memory.
        """
        if self._groww is not None:
            return self._groww

        if not self._access_token:
            totp_code = None
            secret_arg = None
            try:
                totp_gen = pyotp.TOTP(self._api_secret)
                totp_code = totp_gen.now()
            except Exception:
                secret_arg = self._api_secret

            logger.info("Requesting daily access token from Groww Trade API...")
            try:
                token_resp = GrowwAPI.get_access_token(
                    api_key=self._api_key,
                    totp=totp_code,
                    secret=secret_arg
                )
                if isinstance(token_resp, dict) and "token" in token_resp:
                    self._access_token = str(token_resp["token"])
                else:
                    self._access_token = str(token_resp)
            except GrowwAPIAuthenticationException as e:
                err_msg = (
                    "Groww API authentication failed (HTTP 401). "
                    "Please verify GROWW_API_KEY and GROWW_API_SECRET in your .env file."
                )
                logger.error(err_msg)
                raise GrowwMarketDataAuthenticationError(err_msg) from e
            except GrowwAPIAuthorisationException as e:
                err_msg = (
                    "Groww API authorization forbidden (HTTP 403) during token creation. "
                    "Verify your API Key status at https://groww.in/trade-api/api-keys."
                )
                logger.error(err_msg)
                raise GrowwMarketDataAuthorisationError(err_msg) from e
            except Exception as e:
                err_msg = f"Failed to authenticate with Groww Trade API: {e}"
                logger.error(err_msg)
                raise RealMarketDataConnectionError(err_msg) from e

        self._groww = GrowwAPI(self._access_token)
        return self._groww

    def _parse_timestamp(self, raw_ts: Any) -> datetime:
        """Parses exchange or trade timestamp into localized Asia/Kolkata (+05:30) datetime."""
        now_kolkata = datetime.now(KOLKATA_TZ)
        if not raw_ts:
            return now_kolkata

        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                return raw_ts.replace(tzinfo=KOLKATA_TZ)
            return raw_ts.astimezone(KOLKATA_TZ)

        if isinstance(raw_ts, (int, float)):
            # Epoch milliseconds vs seconds
            try:
                if raw_ts > 1e11:
                    return datetime.fromtimestamp(raw_ts / 1000.0, tz=KOLKATA_TZ)
                return datetime.fromtimestamp(raw_ts, tz=KOLKATA_TZ)
            except (ValueError, OSError):
                return now_kolkata

        if isinstance(raw_ts, str):
            clean_ts = raw_ts.strip()
            try:
                dt = datetime.fromisoformat(clean_ts)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=KOLKATA_TZ)
                return dt.astimezone(KOLKATA_TZ)
            except ValueError:
                pass

            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(clean_ts, fmt)
                    return dt.replace(tzinfo=KOLKATA_TZ)
                except ValueError:
                    continue

        return now_kolkata

    def _parse_quote_item(self, symbol: str, item: Dict[str, Any]) -> MarketQuote:
        """
        Transforms a raw Groww JSON quote object into a canonical MarketQuote dataclass.
        Enforces all required fields and invariants:
        symbol, company, exchange, timestamp, open, high, low, current_price (last_price),
        previous_close, change, change_pct (change_percent), volume, day_high, day_low, market_status.
        """
        company_meta = self._constituents.get(symbol, {})
        company_name = company_meta.get("company", symbol)

        ohlc = item.get("ohlc") or {}
        if not isinstance(ohlc, dict):
            ohlc = {}

        last_price = float(item.get("last_price") or item.get("ltp") or 0.0)
        open_price = float(ohlc.get("open") or item.get("open") or 0.0)
        high_price = float(ohlc.get("high") or item.get("high") or 0.0)
        low_price = float(ohlc.get("low") or item.get("low") or 0.0)
        prev_close = float(ohlc.get("close") or item.get("close") or 0.0)

        # Day high & low bounds
        high_range = item.get("high_trade_range")
        low_range = item.get("low_trade_range")
        
        day_high = float(high_range) if high_range is not None else max(high_price, open_price, last_price)
        if low_range is not None:
            day_low = float(low_range)
        elif low_price > 0:
            day_low = min(low_price, open_price, last_price)
        else:
            day_low = min(open_price, last_price) if (open_price > 0 and last_price > 0) else (open_price or last_price)

        if day_high == 0.0 and last_price > 0.0:
            day_high = last_price
        if day_low == 0.0 and last_price > 0.0:
            day_low = last_price

        # Net change & percentage
        if item.get("day_change") is not None:
            change = round(float(item["day_change"]), 2)
        elif last_price > 0.0 and prev_close > 0.0:
            change = round(last_price - prev_close, 2)
        else:
            change = 0.0

        if item.get("day_change_perc") is not None:
            change_pct = round(float(item["day_change_perc"]), 2)
        elif prev_close > 0.0:
            change_pct = round((change / prev_close) * 100.0, 2)
        else:
            change_pct = 0.0

        volume = int(item.get("volume", 0))

        # Parse timestamp from last_trade_time or system time
        ts_val = item.get("last_trade_time") or item.get("timestamp")
        timestamp_dt = self._parse_timestamp(ts_val)

        # Operating market status in Asia/Kolkata
        market_status = determine_market_status(timestamp_dt)

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
            exchange="NSE",
            is_demo=False,
            data_source="Groww Trade API"
        )

    def _handle_groww_exception(self, exc: Exception, context: str) -> None:
        """Categorizes and translates Groww SDK exceptions with explicit guidance."""
        code_str = str(getattr(exc, "code", "")).strip()
        msg_str = str(getattr(exc, "message", getattr(exc, "msg", str(exc)))).lower()

        # 403 Forbidden / Market Data permission or subscription required
        if isinstance(exc, GrowwAPIAuthorisationException) or code_str == "403" or "forbidden" in msg_str:
            err_msg = (
                f"Groww Trade API returned 403 Forbidden during {context}. "
                "Authentication succeeded, but your account lacks an active Live Market Data subscription "
                "or market data permission for this API key. "
                "To enable real-time quotes, activate the Market Data subscription / permissions at: "
                "https://groww.in/user/profile/trading-apis or https://groww.in/trade-api/api-keys."
            )
            logger.error(err_msg)
            raise GrowwMarketDataAuthorisationError(err_msg) from exc

        # 401 Unauthorized / Token expired
        elif isinstance(exc, GrowwAPIAuthenticationException) or code_str == "401" or "unauthorized" in msg_str:
            err_msg = (
                f"Groww API authentication failed (HTTP 401) during {context}. "
                "Session token is expired or credentials are invalid. Please verify GROWW_API_KEY in .env."
            )
            logger.error(err_msg)
            raise GrowwMarketDataAuthenticationError(err_msg) from exc

        # 429 Rate Limit
        elif isinstance(exc, GrowwAPIRateLimitException) or code_str == "429" or "rate limit" in msg_str:
            err_msg = (
                f"Groww API rate limit exceeded (HTTP 429) during {context}. "
                "Live Data limit is 10 requests/sec, 300 requests/min. Please throttle requests."
            )
            logger.error(err_msg)
            raise GrowwMarketDataRateLimitError(err_msg) from exc

        # Timeout
        elif isinstance(exc, GrowwAPITimeoutException) or code_str in ("408", "504") or "timed out" in msg_str:
            err_msg = f"Groww API request timed out during {context}."
            logger.error(err_msg)
            raise RealMarketDataConnectionError(err_msg) from exc

        # Generic / Other
        else:
            err_msg = f"Groww API error during {context}: {exc}"
            logger.error(err_msg)
            raise RealMarketDataConnectionError(err_msg) from exc

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        """
        Fetches a real live quote for a single symbol from the Groww Trade API.
        Method: groww.get_quote(trading_symbol, exchange="NSE", segment="CASH")
        Raises RealMarketDataConnectionError / GrowwMarketDataAuthorisationError on failure.
        NEVER generates fake numbers in production mode.
        """
        client = self._get_or_create_session()
        sym_clean = self.symbol_to_trading_symbol(symbol)

        if sym_clean not in self._constituents:
            logger.warning(f"Symbol '{symbol}' not found in NIFTY 50 constituents.")
            return None

        logger.info(f"Connecting to Groww Trade API for single quote '{sym_clean}'...")
        try:
            raw_quote = client.get_quote(
                trading_symbol=sym_clean,
                exchange="NSE",
                segment="CASH",
                timeout=self._timeout
            )
        except Exception as e:
            self._handle_groww_exception(e, f"single quote fetch for '{sym_clean}'")

        if not raw_quote or not isinstance(raw_quote, dict):
            logger.warning(f"Empty quote response received from Groww for {sym_clean}")
            return None

        return self._parse_quote_item(sym_clean, raw_quote)

    def get_quotes(self, symbols: Optional[List[str]] = None) -> List[MarketQuote]:
        """
        Batch fetches authentic live quotes for NIFTY 50 symbols from Groww Trade API.
        Uses Groww's batch methods `get_ohlc` and `get_ltp` (supporting up to 50 instruments per call).
        Queries all 50 NIFTY 50 stocks with minimal API calls respecting rate limits.
        """
        client = self._get_or_create_session()
        target_symbols = (
            [self.symbol_to_trading_symbol(s) for s in symbols if self.symbol_to_trading_symbol(s) in self._constituents]
            if symbols is not None
            else sorted(self._constituents.keys())
        )

        if not target_symbols:
            return []

        quotes: List[MarketQuote] = []
        chunk_size = 50  # Groww batch methods support up to 50 instruments per call

        logger.info(f"Batch fetching {len(target_symbols)} quotes from Groww Trade API...")

        for i in range(0, len(target_symbols), chunk_size):
            chunk = target_symbols[i:i + chunk_size]
            exchange_symbols = tuple(self.symbol_to_batch_exchange_symbol(s) for s in chunk)

            ohlc_resp: Dict[str, Any] = {}
            ltp_resp: Dict[str, Any] = {}

            try:
                ohlc_resp = client.get_ohlc(
                    segment="CASH",
                    exchange_trading_symbols=exchange_symbols,
                    timeout=self._timeout
                )
                ltp_resp = client.get_ltp(
                    segment="CASH",
                    exchange_trading_symbols=exchange_symbols,
                    timeout=self._timeout
                )
            except Exception as e:
                self._handle_groww_exception(e, f"batch fetch for {len(chunk)} symbols")

            if not isinstance(ohlc_resp, dict):
                ohlc_resp = {}
            if not isinstance(ltp_resp, dict):
                ltp_resp = {}

            missing_symbols: List[str] = []

            for sym in chunk:
                batch_key = self.symbol_to_batch_exchange_symbol(sym)
                ltp_val = ltp_resp.get(batch_key)
                ohlc_data = ohlc_resp.get(batch_key)

                if ltp_val is None and not ohlc_data:
                    missing_symbols.append(sym)
                    continue

                quote_item = {
                    "last_price": ltp_val,
                    "ltp": ltp_val,
                    "ohlc": ohlc_data or {},
                    "open": ohlc_data.get("open") if isinstance(ohlc_data, dict) else None,
                    "high": ohlc_data.get("high") if isinstance(ohlc_data, dict) else None,
                    "low": ohlc_data.get("low") if isinstance(ohlc_data, dict) else None,
                    "close": ohlc_data.get("close") if isinstance(ohlc_data, dict) else None,
                }
                try:
                    q = self._parse_quote_item(sym, quote_item)
                    quotes.append(q)
                except Exception as ex:
                    logger.warning(f"Error parsing batch quote for {sym}: {ex}")

            if missing_symbols:
                logger.warning(
                    f"Groww batch returned quotes for {len(quotes)}/{len(chunk)} symbols. "
                    f"Missing: {missing_symbols}"
                )

        return quotes

    def health_check(self) -> bool:
        """
        Validates credentials and live connection to Groww API.
        Attempts user profile inspection to verify authentication status.
        """
        if not self._access_token and (not self._api_key or not self._api_secret):
            return False
        try:
            client = self._get_or_create_session()
            profile = client.get_user_profile()
            return bool(profile and isinstance(profile, dict))
        except Exception:
            return False


# ==============================================================================
# Zerodha Kite Connect v3 Provider (Secondary / Alternative Implementation)
# ==============================================================================

class KiteConnectMarketDataProvider(BaseLiveMarketDataProvider):
    """
    Alternative live market data provider adapter for Zerodha Kite Connect v3.
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
        self._validate_credentials()

    @property
    def provider_name(self) -> str:
        return f"ZerodhaKiteConnect[{self._environment.upper()}]"

    @property
    def is_demo(self) -> bool:
        return False

    def _validate_credentials(self) -> None:
        if not self._api_key and not self._access_token:
            err_msg = (
                "Zerodha Kite provider requires credentials. "
                "Please configure MARKET_DATA_API_KEY and MARKET_DATA_ACCESS_TOKEN."
            )
            logger.error(err_msg)
            raise RealMarketDataConfigurationError(err_msg)

    def _load_constituents(self) -> None:
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
        clean_sym = symbol.strip().upper()
        return f"NSE:{clean_sym}"

    @staticmethod
    def instrument_to_symbol(instrument: str) -> str:
        inst = instrument.strip().upper()
        if inst.startswith("NSE:"):
            return inst[4:]
        return inst

    def _get_headers(self) -> Dict[str, str]:
        token_part = f"{self._api_key}:{self._access_token}" if self._access_token else self._api_key
        return {
            "Authorization": f"token {token_part}",
            "X-Kite-Version": "3",
            "Accept": "application/json",
            "User-Agent": "Nifty50-Stock-Market-Prediction/1.0"
        }

    def _parse_timestamp(self, raw_ts: Any) -> datetime:
        now_kolkata = datetime.now(KOLKATA_TZ)
        if not raw_ts:
            return now_kolkata
        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                return raw_ts.replace(tzinfo=KOLKATA_TZ)
            return raw_ts.astimezone(KOLKATA_TZ)
        if isinstance(raw_ts, str):
            clean_ts = raw_ts.strip()
            try:
                dt = datetime.fromisoformat(clean_ts)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=KOLKATA_TZ)
                return dt.astimezone(KOLKATA_TZ)
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(clean_ts, fmt)
                    return dt.replace(tzinfo=KOLKATA_TZ)
                except ValueError:
                    continue
        return now_kolkata

    def _parse_quote_item(self, symbol: str, item: Dict[str, Any]) -> MarketQuote:
        company_meta = self._constituents.get(symbol, {})
        company_name = company_meta.get("company", symbol)

        ohlc = item.get("ohlc", {})
        last_price = float(item.get("last_price", 0.0))
        prev_close = float(ohlc.get("close", 0.0))
        open_price = float(ohlc.get("open", 0.0))
        high_price = float(ohlc.get("high", 0.0))
        low_price = float(ohlc.get("low", 0.0))

        day_high = max(high_price, open_price, last_price)
        day_low = min(low_price, open_price, last_price) if low_price > 0 else min(open_price, last_price)
        if day_high == 0.0 and last_price > 0.0:
            day_high = last_price
        if day_low == 0.0 and last_price > 0.0:
            day_low = last_price

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
        timestamp_dt = self._parse_timestamp(item.get("timestamp"))
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
            exchange="NSE",
            is_demo=False,
            data_source="Zerodha Kite Connect v3"
        )

    def _handle_api_error_response(self, response: requests.Response, context: str) -> None:
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
            f"Zerodha Kite Connect error during {context} (HTTP {status_code} [{error_type}]): {error_message}"
        )
        logger.error(full_msg)
        raise RealMarketDataConnectionError(full_msg)

    def get_quote(self, symbol: str) -> Optional[MarketQuote]:
        sym_clean = symbol.strip().upper()
        if sym_clean not in self._constituents:
            logger.warning(f"Symbol '{symbol}' not found in NIFTY 50 constituents.")
            return None

        instrument = self.symbol_to_instrument(sym_clean)
        url = f"{self._base_url}/quote"
        headers = self._get_headers()
        params = [("i", instrument)]

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
        target_symbols = (
            [s.strip().upper() for s in symbols if s.strip().upper() in self._constituents]
            if symbols is not None
            else sorted(self._constituents.keys())
        )

        if not target_symbols:
            return []

        url = f"{self._base_url}/quote"
        headers = self._get_headers()
        params = [("i", self.symbol_to_instrument(sym)) for sym in target_symbols]

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
                f"Kite Connect returned quotes for {len(quotes)}/{len(target_symbols)} symbols. Missing: {missing_symbols}"
            )

        return quotes

    def health_check(self) -> bool:
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


# ==============================================================================
# Provider Aliases
# ==============================================================================

# Primary real provider alias is GrowwMarketDataProvider
RealMarketDataProvider = GrowwMarketDataProvider
