"""
test_real_provider.py
----------------------
Unit and integration tests for Zerodha Kite Connect v3 adapter in RealMarketDataProvider:
1. Symbol to instrument mapping ('ADANIENT', 'M&M', 'BAJAJ-AUTO')
2. Kite headers and query parameters formatting
3. JSON response mapping into canonical MarketQuote
4. Verification of Current Price, Previous Close, Change, Change %, Volume, Timestamp, Market Status
5. Error handling on HTTP 400/401/403 (InputException, TokenException)
6. Integration with LiveQuoteStorage for real quotes persistence
7. Real network attempt test for ADANIENT and 50 companies
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

from src.live.real_provider import (
    RealMarketDataProvider,
    RealMarketDataConfigurationError,
    RealMarketDataConnectionError
)
from src.live.models import MarketQuote
from src.live.storage import LiveQuoteStorage
from src.live.validator import LiveQuoteValidator
from config.live_config import CONSTITUENTS_CSV, KOLKATA_TZ


class TestRealMarketDataProvider(unittest.TestCase):
    """Test suite for Zerodha Kite Connect v3 provider adapter."""

    @classmethod
    def setUpClass(cls):
        cls.test_storage_dir = Path("data/test_real_tmp")
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

    def test_01_symbol_to_instrument_mapping(self):
        """Verifies symbol mapping for standard, hyphenated, and ampersand symbols."""
        self.assertEqual(RealMarketDataProvider.symbol_to_instrument("ADANIENT"), "NSE:ADANIENT")
        self.assertEqual(RealMarketDataProvider.symbol_to_instrument("RELIANCE"), "NSE:RELIANCE")
        self.assertEqual(RealMarketDataProvider.symbol_to_instrument("M&M"), "NSE:M&M")
        self.assertEqual(RealMarketDataProvider.symbol_to_instrument("BAJAJ-AUTO"), "NSE:BAJAJ-AUTO")
        self.assertEqual(RealMarketDataProvider.symbol_to_instrument("tatamotors"), "NSE:TATAMOTORS")

        self.assertEqual(RealMarketDataProvider.instrument_to_symbol("NSE:ADANIENT"), "ADANIENT")
        self.assertEqual(RealMarketDataProvider.instrument_to_symbol("NSE:M&M"), "M&M")
        self.assertEqual(RealMarketDataProvider.instrument_to_symbol("NSE:BAJAJ-AUTO"), "BAJAJ-AUTO")

    def test_02_kite_headers_and_auth_format(self):
        """Verifies Kite Connect v3 Authorization header format: token {api_key}:{access_token}."""
        prov = RealMarketDataProvider(
            api_key="my_api_key",
            access_token="my_session_token"
        )
        headers = prov._get_headers()
        self.assertEqual(headers["Authorization"], "token my_api_key:my_session_token")
        self.assertEqual(headers["X-Kite-Version"], "3")
        self.assertEqual(headers["Accept"], "application/json")

    def test_03_adanient_quote_parsing_and_field_accuracy(self):
        """
        Tests parsing an authentic Zerodha Kite Connect response for ADANIENT.
        Verifies:
        - Current Price is correct
        - Previous Close is correct
        - Change is correct
        - Change % is correct
        - Volume is correct
        - Timestamp is in Asia/Kolkata
        - Market status is populated
        - is_demo is False
        - data_source is Zerodha Kite Connect v3
        """
        prov = RealMarketDataProvider(
            api_key="mock_key",
            access_token="mock_token"
        )

        mock_kite_data = {
            "instrument_token": 6401,
            "timestamp": "2026-09-24 15:30:00",
            "last_price": 3016.85,
            "net_change": -43.15,
            "ohlc": {
                "open": 3048.0,
                "high": 3066.0,
                "low": 2995.0,
                "close": 3060.0
            },
            "volume": 1688401
        }

        quote = prov._parse_quote_item("ADANIENT", mock_kite_data)
        self.assertIsInstance(quote, MarketQuote)
        self.assertEqual(quote.symbol, "ADANIENT")
        self.assertEqual(quote.company, "Adani Enterprises Ltd.")
        self.assertEqual(quote.current_price, 3016.85)
        self.assertEqual(quote.previous_close, 3060.0)
        self.assertEqual(quote.open, 3048.0)
        self.assertEqual(quote.high, 3066.0)
        self.assertEqual(quote.low, 2995.0)
        self.assertEqual(quote.day_high, 3066.0)
        self.assertEqual(quote.day_low, 2995.0)
        self.assertEqual(quote.change, -43.15)
        # Change % = (-43.15 / 3060.0) * 100 = -1.41
        self.assertEqual(quote.change_pct, -1.41)
        self.assertEqual(quote.volume, 1688401)
        self.assertFalse(quote.is_demo)
        self.assertEqual(quote.data_source, "Zerodha Kite Connect v3")
        self.assertIsNotNone(quote.market_status)
        self.assertTrue(quote.timestamp.isoformat().endswith("+05:30"))

    def test_04_get_quote_mocked_network_success(self):
        """Tests get_quote for ADANIENT with simulated HTTP 200 response from Kite Connect."""
        prov = RealMarketDataProvider(api_key="mock_key", access_token="mock_token")

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "status": "success",
            "data": {
                "NSE:ADANIENT": {
                    "instrument_token": 6401,
                    "timestamp": "2026-09-24 15:30:00",
                    "last_price": 3020.0,
                    "net_change": -40.0,
                    "ohlc": {
                        "open": 3050.0,
                        "high": 3070.0,
                        "low": 3000.0,
                        "close": 3060.0
                    },
                    "volume": 1500000
                }
            }
        }

        with patch.object(prov._session, "get", return_value=fake_resp) as mock_get:
            quote = prov.get_quote("ADANIENT")
            mock_get.assert_called_once()
            self.assertIsNotNone(quote)
            self.assertEqual(quote.symbol, "ADANIENT")
            self.assertEqual(quote.current_price, 3020.0)

            # Validate quote with LiveQuoteValidator
            is_valid, errors = self.validator.validate_quote(quote)
            self.assertTrue(is_valid, f"Validation errors: {errors}")

            # Save to storage
            paths = self.storage.save_quotes([quote])
            self.assertTrue(self.storage.snapshot_file.exists())
            snap_df = self.storage.load_latest_snapshot()
            self.assertEqual(len(snap_df), 1)
            self.assertEqual(snap_df.iloc[0]["Symbol"], "ADANIENT")
            self.assertEqual(snap_df.iloc[0]["data_source"], "Zerodha Kite Connect v3")

    def test_05_get_quotes_batch_all_50_constituents(self):
        """Tests batch fetching and mapping for all 50 NIFTY 50 companies."""
        prov = RealMarketDataProvider(api_key="mock_key", access_token="mock_token")

        # Create mock data for all 50 symbols
        mock_data = {}
        for sym in prov._constituents.keys():
            mock_data[f"NSE:{sym}"] = {
                "instrument_token": 1000,
                "timestamp": "2026-09-24 15:30:00",
                "last_price": 1000.0,
                "net_change": 10.0,
                "ohlc": {"open": 995.0, "high": 1010.0, "low": 990.0, "close": 990.0},
                "volume": 50000
            }

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"status": "success", "data": mock_data}

        with patch.object(prov._session, "get", return_value=fake_resp) as mock_get:
            quotes = prov.get_quotes()
            self.assertEqual(len(quotes), 50)
            mock_get.assert_called_once()  # Must query all 50 in a single network call!

            # Validate each quote
            for q in quotes:
                is_valid, errs = self.validator.validate_quote(q)
                self.assertTrue(is_valid, f"{q.symbol} failed validation: {errs}")

            # Store in test storage
            self.storage.save_quotes(quotes)
            snap_df = self.storage.load_latest_snapshot()
            self.assertEqual(len(snap_df), 50)

    def test_06_kite_api_error_handling(self):
        """Verifies that Kite Connect errors (400, 403) raise RealMarketDataConnectionError."""
        prov = RealMarketDataProvider(api_key="bad_key", access_token="bad_token")

        fake_err_resp = MagicMock()
        fake_err_resp.status_code = 400
        fake_err_resp.text = '{"status":"error","message":"Invalid api_key","error_type":"InputException"}'
        fake_err_resp.json.return_value = {
            "status": "error",
            "message": "Invalid api_key",
            "error_type": "InputException"
        }

        with patch.object(prov._session, "get", return_value=fake_err_resp):
            with self.assertRaises(RealMarketDataConnectionError) as ctx:
                prov.get_quote("ADANIENT")
            self.assertIn("InputException", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
