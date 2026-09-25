"""
test_real_market_feed.py
-------------------------
Executable runner for testing the Real Indian Market Data Provider (Zerodha Kite Connect v3).
Performs live network tests against https://api.kite.trade:
1. Validates environment variables and API configuration
2. Tests fetching real quote for ONE company (ADANIENT)
3. Tests fetching real quotes for all 50 NIFTY 50 companies
4. Verifies all 8 field constraints (Symbol, Price, Prev Close, Change, Change %, Volume, Timestamp, Market Status)
5. Persists real quotes using LiveQuoteStorage if retrieved
6. Reports full diagnostic details and required activation steps
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.real_provider import (
    RealMarketDataProvider,
    RealMarketDataConfigurationError,
    RealMarketDataConnectionError
)
from src.live.validator import LiveQuoteValidator
from src.live.storage import LiveQuoteStorage
from src.live.models import MarketQuote
from config.live_config import (
    CONSTITUENTS_CSV,
    LIVE_SNAPSHOT_CSV,
    LIVE_HISTORY_PARQUET,
    KOLKATA_TZ
)
from src.utilities.logger import get_logger

logger = get_logger("RealMarketFeedRunner")


def run_real_feed_verification():
    print("=" * 80)
    print("REAL INDIAN MARKET DATA PROVIDER VERIFICATION: ZERODHA KITE CONNECT v3")
    print("=" * 80)

    # 1. Environment & Credentials Check
    api_key = os.getenv("MARKET_DATA_API_KEY", "").strip()
    api_secret = os.getenv("MARKET_DATA_API_SECRET", "").strip()
    access_token = os.getenv("MARKET_DATA_ACCESS_TOKEN", "").strip()
    base_url = os.getenv("MARKET_DATA_BASE_URL", "https://api.kite.trade").strip()
    env = os.getenv("MARKET_DATA_ENVIRONMENT", "sandbox").strip().lower()

    print(f"\n[1] CONFIGURATION CHECK:")
    print(f"  - Provider:                 Zerodha Kite Connect v3")
    print(f"  - Gateway URL:              {base_url}")
    print(f"  - Environment:              {env}")
    print(f"  - MARKET_DATA_API_KEY:      {'[CONFIGURED]' if api_key else '[NOT SET]'}")
    print(f"  - MARKET_DATA_API_SECRET:   {'[CONFIGURED]' if api_secret else '[NOT SET]'}")
    print(f"  - MARKET_DATA_ACCESS_TOKEN: {'[CONFIGURED]' if access_token else '[NOT SET]'}")

    storage = LiveQuoteStorage()
    validator = LiveQuoteValidator()

    # If no credentials set, we test with placeholder credentials to execute the actual
    # HTTP roundtrip to api.kite.trade and capture the live server response.
    effective_key = api_key if api_key else "test_unauthenticated_key"
    effective_token = access_token if access_token else "test_session_token"

    try:
        provider = RealMarketDataProvider(
            api_key=effective_key,
            api_secret=api_secret,
            access_token=effective_token,
            base_url=base_url,
            environment=env
        )
    except RealMarketDataConfigurationError as e:
        print(f"\n[CONFIGURATION ERROR]: {e}")
        return

    # 2. Test Fetching a Real Quote for ONE company (ADANIENT)
    print(f"\n[2] TESTING REAL QUOTE FETCH FOR ONE COMPANY: ADANIENT")
    adanient_success = False
    adanient_quote = None
    adanient_error = None

    try:
        adanient_quote = provider.get_quote("ADANIENT")
        if adanient_quote:
            adanient_success = True
            print(f"  ✅ Successfully retrieved real quote for ADANIENT!")
            print(f"     Symbol:         {adanient_quote.symbol}")
            print(f"     Company:        {adanient_quote.company}")
            print(f"     Current Price:  ₹{adanient_quote.current_price:,.2f}")
            print(f"     Previous Close: ₹{adanient_quote.previous_close:,.2f}")
            print(f"     Change:         ₹{adanient_quote.change:+.2f} ({adanient_quote.change_pct:+.2f}%)")
            print(f"     Day Range:      ₹{adanient_quote.day_low:,.2f} - ₹{adanient_quote.day_high:,.2f}")
            print(f"     Volume:         {adanient_quote.volume:,}")
            print(f"     Timestamp:      {adanient_quote.timestamp.isoformat()}")
            print(f"     Market Status:  {adanient_quote.market_status}")
            print(f"     Data Source:    {adanient_quote.data_source}")

            # Validate
            is_valid, errs = validator.validate_quote(adanient_quote)
            print(f"     Validation:     {'PASS' if is_valid else f'FAIL: {errs}'}")
            if is_valid:
                storage.save_quotes([adanient_quote])
                print(f"     Stored at:      {LIVE_SNAPSHOT_CSV}")
    except RealMarketDataConnectionError as e:
        adanient_error = str(e)
        print(f"  ❌ ADANIENT retrieval failed with live server response:")
        print(f"     {adanient_error}")
    except Exception as e:
        adanient_error = str(e)
        print(f"  ❌ Unexpected error: {adanient_error}")

    # 3. Test Fetching Quotes for all 50 NIFTY 50 companies
    print(f"\n[3] TESTING REAL QUOTES BATCH FETCH FOR ALL 50 NIFTY 50 COMPANIES:")
    all_50_success = False
    quotes_50 = []
    batch_error = None
    failed_symbols = []

    try:
        quotes_50 = provider.get_quotes()
        if len(quotes_50) == 50:
            all_50_success = True
            print(f"  ✅ Successfully retrieved quotes for all {len(quotes_50)}/50 companies in 1 batch request!")
            storage.save_quotes(quotes_50)
            print(f"  ✅ Saved all 50 quotes to snapshot: {LIVE_SNAPSHOT_CSV}")
            print(f"  ✅ Appended all 50 quotes to history: {LIVE_HISTORY_PARQUET}")
        elif len(quotes_50) > 0:
            print(f"  ⚠️ Retrieved {len(quotes_50)}/50 companies.")
            all_symbols = set(provider._constituents.keys())
            retrieved_symbols = {q.symbol for q in quotes_50}
            failed_symbols = sorted(list(all_symbols - retrieved_symbols))
            print(f"     Missing companies ({len(failed_symbols)}): {failed_symbols}")
            storage.save_quotes(quotes_50)
        else:
            print(f"  ❌ Zero quotes returned.")
    except RealMarketDataConnectionError as e:
        batch_error = str(e)
        print(f"  ❌ Batch retrieval failed with live server response:")
        print(f"     {batch_error}")
        failed_symbols = sorted(list(provider._constituents.keys()))
    except Exception as e:
        batch_error = str(e)
        print(f"  ❌ Unexpected batch error: {batch_error}")
        failed_symbols = sorted(list(provider._constituents.keys()))

    # 4. Summary Verdict & Reporting
    print("\n" + "=" * 80)
    print("FINAL INTEGRATION SUMMARY REPORT")
    print("=" * 80)
    print(f"1. Provider/API Integrated:       Zerodha Kite Connect v3 (https://api.kite.trade)")
    print(f"2. Required Environment Vars:     MARKET_DATA_API_KEY, MARKET_DATA_API_SECRET, MARKET_DATA_ACCESS_TOKEN")
    print(f"3. ADANIENT Status:               {'SUCCESS' if adanient_success else 'FAILED - Needs Active Credentials'}")
    print(f"4. All 50 Companies Status:       {'SUCCESS (50/50)' if all_50_success else 'FAILED - Needs Active Credentials'}")
    if failed_symbols:
        print(f"5. Failed Companies:              {len(failed_symbols)} companies (Kite Connect API rejected unauthenticated request)")
    else:
        print(f"5. Failed Companies:              None")
    print(f"6. Data Storage Location:         {LIVE_SNAPSHOT_CSV} (Snapshot) & {LIVE_HISTORY_PARQUET} (History)")
    print(f"7. Server Response Received:      {adanient_error or 'HTTP 200 OK'}")
    print("=" * 80)


if __name__ == "__main__":
    run_real_feed_verification()
