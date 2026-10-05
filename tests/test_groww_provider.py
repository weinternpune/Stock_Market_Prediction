"""
test_groww_provider.py
----------------------
Unit and integration tests for GrowwMarketDataProvider:
1. Configuration and credential presence validation
2. Symbol to trading symbol and batch exchange symbol mapping
3. TOTP generation and session token caching
4. Single quote parsing and MarketQuote conversion with all 12+ required fields
5. Batch quotes parsing using get_ohlc and get_ltp
6. HTTP 403 Forbidden handling (diagnostic guidance for market data subscription)
7. HTTP 401 Authentication failure handling
8. HTTP 429 Rate limit error handling
9. Missing fields and partial response robustness
10. Network failure and API timeout handling
11. Health check connectivity verification
12. Integration with LiveMarketDataService, LiveQuoteValidator, and LiveQuoteStorage
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.live.real_provider import (
    GrowwMarketDataProvider,
    GrowwMarketDataConfigurationError,
    GrowwMarketDataAuthenticationError,
    GrowwMarketDataAuthorisationError,
    GrowwMarketDataRateLimitError,
    RealMarketDataConnectionError
)
from src.live.models import MarketQuote
from src.live.service import LiveMarketDataService
from src.live.storage import LiveQuoteStorage
from src.live.validator import LiveQuoteValidator
from config.live_config import CONSTITUENTS_CSV, KOLKATA_TZ
from growwapi.groww.exceptions import (
    GrowwAPIException,
    GrowwAPIAuthenticationException,
    GrowwAPIAuthorisationException,
    GrowwAPIRateLimitException,
    GrowwAPITimeoutException
)


class TestGrowwMarketDataProvider(unittest.TestCase):
    """Test suite for Groww Trade API live provider."""

    @classmethod
    def setUpClass(cls):
        cls.test_storage_dir = Path("data/test_groww_tmp")
        cls.storage = LiveQuoteStorage(
            storage_dir=cls.test_storage_dir,
            snapshot_file=cls.test_storage_dir / "test_snapshot.csv",
            history_parquet=cls.test_storage_dir / "test_history.parquet",
            history_csv=cls.test_storage_dir / "test_history.csv"
        )
        cls.validator = LiveQuoteValidator()

    @classmethod
    def tearDownClass(cls):
        if cls.test_storage_dir.exists():
            for f in cls.test_storage_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                cls.test_storage_dir.rmdir()
            except Exception:
                pass

    def test_01_configuration_validation_missing_credentials(self):
        """Verifies GrowwMarketDataConfigurationError when credentials are missing."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GrowwMarketDataConfigurationError) as ctx:
                GrowwMarketDataProvider(api_key="", api_secret="", access_token="")
            self.assertIn("Groww Trade API provider requires credentials", str(ctx.exception))

    def test_02_symbol_mapping_trading_and_batch(self):
        """Tests symbol mapping for standard, hyphenated, and ampersand symbols."""
        # Single quote trading symbol mapping
        self.assertEqual(GrowwMarketDataProvider.symbol_to_trading_symbol("RELIANCE"), "RELIANCE")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_trading_symbol("M&M"), "M&M")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_trading_symbol("BAJAJ-AUTO"), "BAJAJ-AUTO")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_trading_symbol("NSE:ADANIENT"), "ADANIENT")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_trading_symbol("NSE_TCS"), "TCS")

        # Batch exchange symbol mapping
        self.assertEqual(GrowwMarketDataProvider.symbol_to_batch_exchange_symbol("RELIANCE"), "NSE_RELIANCE")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_batch_exchange_symbol("M&M"), "NSE_M&M")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_batch_exchange_symbol("BAJAJ-AUTO"), "NSE_BAJAJ-AUTO")
        self.assertEqual(GrowwMarketDataProvider.symbol_to_batch_exchange_symbol("NSE:ADANIENT"), "NSE_ADANIENT")

    @patch("growwapi.GrowwAPI.get_access_token")
    @patch("growwapi.GrowwAPI.__init__", return_value=None)
    def test_03_totp_and_token_caching(self, mock_groww_init, mock_get_token):
        """Tests that access token is obtained via TOTP and cached in memory."""
        mock_get_token.return_value = "mock_session_token_12345"

        provider = GrowwMarketDataProvider(
            api_key="mock_key",
            api_secret="DZG4UOJJLKVFLG5BNFUTY4YNRKKQHY3C"  # 32-char Base32 TOTP secret
        )

        session = provider._get_or_create_session()
        self.assertEqual(provider._access_token, "mock_session_token_12345")
        mock_get_token.assert_called_once()

        # Calling again should reuse cached token, not calling get_access_token again
        session_again = provider._get_or_create_session()
        self.assertEqual(session, session_again)
        mock_get_token.assert_called_once()

    def test_04_single_quote_parsing_and_market_quote_conversion(self):
        """
        Tests parsing an authentic Groww get_quote payload into MarketQuote.
        Verifies:
        - Symbol, Company, Exchange
        - Current Price (last_price)
        - Open, High, Low, Previous Close
        - Change, Change %
        - Volume
        - Timestamp in Asia/Kolkata
        - Market Status
        - is_demo = False, data_source = 'Groww Trade API'
        """
        mock_client = MagicMock()
        mock_client.get_quote.return_value = {
            "average_price": 2505.50,
            "bid_price": 2500.0,
            "bid_quantity": 1000,
            "day_change": 25.50,
            "day_change_perc": 1.03,
            "high_trade_range": 2520.0,
            "low_trade_range": 2480.0,
            "last_price": 2505.50,
            "last_trade_time": 1727670600000,  # Epoch ms
            "ohlc": {
                "open": 2490.0,
                "high": 2520.0,
                "low": 2480.0,
                "close": 2480.0  # Previous close
            },
            "volume": 1250000
        }

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        quote = provider.get_quote("RELIANCE")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.symbol, "RELIANCE")
        self.assertEqual(quote.exchange, "NSE")
        self.assertEqual(quote.current_price, 2505.50)
        self.assertEqual(quote.last_price, 2505.50)
        self.assertEqual(quote.open, 2490.0)
        self.assertEqual(quote.high, 2520.0)
        self.assertEqual(quote.low, 2480.0)
        self.assertEqual(quote.previous_close, 2480.0)
        self.assertEqual(quote.change, 25.50)
        self.assertEqual(quote.change_pct, 1.03)
        self.assertEqual(quote.change_percent, 1.03)
        self.assertEqual(quote.volume, 1250000)
        self.assertEqual(quote.day_high, 2520.0)
        self.assertEqual(quote.day_low, 2480.0)
        self.assertFalse(quote.is_demo)
        self.assertEqual(quote.data_source, "Groww Trade API")
        self.assertEqual(quote.timestamp.tzinfo.key, "Asia/Kolkata")
        self.assertIn(quote.market_status, ["OPEN", "CLOSED", "PRE_OPEN", "CLOSED (WEEKEND)"])

    def test_05_batch_quotes_parsing(self):
        """Tests batch quote retrieval using get_ohlc and get_ltp."""
        mock_client = MagicMock()
        mock_client.get_ltp.return_value = {
            "NSE_RELIANCE": 2500.0,
            "NSE_TCS": 3800.0
        }
        mock_client.get_ohlc.return_value = {
            "NSE_RELIANCE": {"open": 2480.0, "high": 2510.0, "low": 2470.0, "close": 2480.0},
            "NSE_TCS": {"open": 3750.0, "high": 3820.0, "low": 3740.0, "close": 3750.0}
        }

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        quotes = provider.get_quotes(["RELIANCE", "TCS"])
        self.assertEqual(len(quotes), 2)
        symbols = [q.symbol for q in quotes]
        self.assertIn("RELIANCE", symbols)
        self.assertIn("TCS", symbols)

        rel_quote = next(q for q in quotes if q.symbol == "RELIANCE")
        self.assertEqual(rel_quote.current_price, 2500.0)
        self.assertEqual(rel_quote.previous_close, 2480.0)
        self.assertEqual(rel_quote.change, 20.0)
        self.assertEqual(rel_quote.change_pct, 0.81)

    def test_06_http_403_authorisation_forbidden_error(self):
        """
        Tests that GrowwAPIAuthorisationException triggers GrowwMarketDataAuthorisationError
        with diagnostic guidance for activating Live Market Data on Groww Cloud.
        """
        mock_client = MagicMock()
        mock_client.get_quote.side_effect = GrowwAPIAuthorisationException()

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        with self.assertRaises(GrowwMarketDataAuthorisationError) as ctx:
            provider.get_quote("ADANIENT")

        self.assertIn("403 Forbidden", str(ctx.exception))
        self.assertIn("Live Market Data subscription", str(ctx.exception))
        self.assertIn("https://groww.in/user/profile/trading-apis", str(ctx.exception))

    def test_07_http_401_authentication_error(self):
        """Tests that GrowwAPIAuthenticationException triggers GrowwMarketDataAuthenticationError."""
        mock_client = MagicMock()
        mock_client.get_quote.side_effect = GrowwAPIAuthenticationException()

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        with self.assertRaises(GrowwMarketDataAuthenticationError) as ctx:
            provider.get_quote("INFY")

        self.assertIn("401", str(ctx.exception))

    def test_08_http_429_rate_limit_error(self):
        """Tests that GrowwAPIRateLimitException triggers GrowwMarketDataRateLimitError."""
        mock_client = MagicMock()
        mock_client.get_quote.side_effect = GrowwAPIRateLimitException()

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        with self.assertRaises(GrowwMarketDataRateLimitError) as ctx:
            provider.get_quote("TCS")

        self.assertIn("429", str(ctx.exception))
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_09_missing_fields_and_partial_response(self):
        """Tests quote parsing when optional fields or OHLC are missing or zero."""
        mock_client = MagicMock()
        mock_client.get_quote.return_value = {
            "last_price": 100.0
            # missing ohlc, volume, day_change, last_trade_time
        }

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        quote = provider.get_quote("ITC")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.current_price, 100.0)
        self.assertEqual(quote.volume, 0)
        self.assertEqual(quote.change, 0.0)
        self.assertEqual(quote.change_pct, 0.0)
        self.assertEqual(quote.day_high, 100.0)
        self.assertEqual(quote.day_low, 100.0)

    def test_10_network_failure_and_timeout(self):
        """Tests that API timeout triggers RealMarketDataConnectionError."""
        mock_client = MagicMock()
        mock_client.get_quote.side_effect = GrowwAPITimeoutException()

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )

        with self.assertRaises(RealMarketDataConnectionError) as ctx:
            provider.get_quote("SBIN")

        self.assertIn("timed out", str(ctx.exception).lower())

    def test_11_health_check(self):
        """Tests health_check returns True on successful profile check, False on exception."""
        mock_client = MagicMock()
        mock_client.get_user_profile.return_value = {"ucc": "7196862693", "active_segments": ["CASH"]}

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )
        self.assertTrue(provider.health_check())

        mock_client.get_user_profile.side_effect = Exception("Network offline")
        self.assertFalse(provider.health_check())

    def test_12_service_and_storage_integration(self):
        """Tests full pipeline: GrowwProvider -> Validator -> Storage via LiveMarketDataService."""
        mock_client = MagicMock()
        mock_client.get_ltp.return_value = {"NSE_ADANIENT": 3100.0}
        mock_client.get_ohlc.return_value = {
            "NSE_ADANIENT": {"open": 3080.0, "high": 3120.0, "low": 3070.0, "close": 3080.0}
        }

        provider = GrowwMarketDataProvider(
            access_token="mock_token",
            groww_client=mock_client
        )
        service = LiveMarketDataService(
            provider=provider,
            validator=self.validator,
            storage=self.storage
        )

        valid_quotes, dropped = service.fetch_quotes(symbols=["ADANIENT"], persist=True)
        self.assertEqual(len(valid_quotes), 1)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(valid_quotes[0].symbol, "ADANIENT")
        self.assertEqual(valid_quotes[0].current_price, 3100.0)

        # Check that snapshot was saved
        snapshot_df = self.storage.load_latest_snapshot()
        self.assertIsNotNone(snapshot_df)
        self.assertEqual(len(snapshot_df), 1)
        sym_col = "Symbol" if "Symbol" in snapshot_df.columns else "symbol"
        self.assertEqual(snapshot_df.iloc[0][sym_col], "ADANIENT")


if __name__ == "__main__":
    unittest.main()
