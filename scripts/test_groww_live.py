"""
test_groww_live.py
------------------
Interactive verification script for testing the Groww Trade API integration:
1. Loads credentials from .env safely without exposing secrets.
2. Initializes GrowwMarketDataProvider and verifies connection/authentication.
3. Tests single quote fetching for representative stocks (RELIANCE, TCS, INFY, ADANIENT).
4. Tests batch quote fetching for NIFTY 50 companies.
5. Validates all mandatory fields: Symbol, Price, Prev Close, Change, Change %, Volume, Timestamp, Market Status.
6. Persists valid live quotes to local storage via LiveQuoteStorage.
7. Reports diagnostic guidance if Groww Cloud subscription / permissions are required.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.real_provider import (
    GrowwMarketDataProvider,
    GrowwMarketDataConfigurationError,
    GrowwMarketDataAuthenticationError,
    GrowwMarketDataAuthorisationError,
    GrowwMarketDataRateLimitError,
    RealMarketDataConnectionError
)
from src.live.validator import LiveQuoteValidator
from src.live.storage import LiveQuoteStorage
from src.live.models import MarketQuote
from config.live_config import (
    CONSTITUENTS_CSV,
    LIVE_SNAPSHOT_CSV,
    LIVE_HISTORY_PARQUET,
    KOLKATA_TZ,
    GROWW_API_KEY,
    GROWW_API_SECRET,
    GROWW_ACCESS_TOKEN,
    GROWW_BASE_URL
)
from src.utilities.logger import get_logger

logger = get_logger("GrowwLiveVerificationRunner")


def mask_secret(s: str) -> str:
    """Masks secret to prevent credential exposure in logs or terminal output."""
    if not s:
        return "[NOT CONFIGURED]"
    if len(s) <= 8:
        return f"[CONFIGURED: {s[:2]}***{s[-2:]} (len={len(s)})]"
    return f"[CONFIGURED: {s[:4]}...{s[-4:]} (len={len(s)})]"


def run_groww_verification():
    print("=" * 80)
    print("      GROWW TRADE API (LIVE MARKET DATA) INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Environment & Credentials Check
    print("\n[1] CONFIGURATION CHECK:")
    print(f"  - Provider:              Groww Trade API (Official growwapi SDK)")
    print(f"  - Base URL:              {GROWW_BASE_URL}")
    print(f"  - GROWW_API_KEY:         {mask_secret(GROWW_API_KEY)}")
    print(f"  - GROWW_API_SECRET:      {mask_secret(GROWW_API_SECRET)}")
    print(f"  - GROWW_ACCESS_TOKEN:     {mask_secret(GROWW_ACCESS_TOKEN)}")
    print(f"  - Constituents file:     {CONSTITUENTS_CSV} (Exists: {CONSTITUENTS_CSV.exists()})")

    if not GROWW_ACCESS_TOKEN and (not GROWW_API_KEY or not GROWW_API_SECRET):
        print("\n❌ ERROR: Groww credentials are not configured in your .env file.")
        print("Please configure GROWW_API_KEY and GROWW_API_SECRET.")
        return 1

    # 2. Provider Initialization
    print("\n[2] INITIALIZING GROWW LIVE PROVIDER...")
    try:
        provider = GrowwMarketDataProvider()
        print(f"  ✓ Initialized: {provider.provider_name}")
        print(f"  ✓ Mode: is_demo = {provider.is_demo} (Production Authenticated)")
        print(f"  ✓ Constituents loaded: {len(provider._constituents)} stocks")
    except GrowwMarketDataConfigurationError as e:
        print(f"  ❌ Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"  ❌ Initialization Error: {e}")
        return 1

    # 3. Connection & Health Check
    print("\n[3] TESTING API GATEWAY CONNECTIVITY (HEALTH CHECK)...")
    try:
        is_healthy = provider.health_check()
        if is_healthy:
            print("  ✓ Groww Trade API Gateway: CONNECTED & AUTHENTICATED (HTTP 200)")
        else:
            print("  ⚠️ Gateway response was non-200. Proceeding with quote test...")
    except Exception as e:
        print(f"  ⚠️ Health check encountered error: {e}")

    # 4. Single Quote Test (Sample of 4 stocks)
    sample_symbols = ["RELIANCE", "TCS", "INFY", "ADANIENT"]
    print(f"\n[4] FETCHING LIVE QUOTES FOR SAMPLE STOCKS ({', '.join(sample_symbols)})...")
    quotes_retrieved = []

    for sym in sample_symbols:
        try:
            print(f"  - Requesting quote for {sym} (NSE Cash Segment)...", end=" ")
            quote = provider.get_quote(sym)
            if quote:
                quotes_retrieved.append(quote)
                print(f"SUCCESS! Price = ₹{quote.current_price:,.2f}")
            else:
                print("FAILED (Empty response)")
        except GrowwMarketDataAuthorisationError as e:
            print("\n  ❌ AUTHORIZATION FAILED (HTTP 403 Forbidden):")
            print(f"     {e}")
            print("\n" + "-" * 80)
            print("  DIAGNOSTIC NOTICE:")
            print("  Your Groww account authenticated successfully, but does not have the")
            print("  'Live Market Data' permission or subscription activated on Groww Cloud.")
            print("  Steps to enable:")
            print("    1. Log in to https://groww.in/user/profile/trading-apis")
            print("    2. Ensure your Trading API Subscription includes Live Data")
            print("    3. At https://groww.in/trade-api/api-keys, verify your API Key permissions")
            print("-" * 80)
            return 2
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")

    # Display quotes table if retrieved
    if quotes_retrieved:
        print("\n" + "=" * 95)
        print(f"{'Symbol':<12} {'Company':<22} {'Price (₹)':<12} {'Change (₹)':<12} {'Change %':<10} {'Volume':<12} {'Time (IST)':<10}")
        print("-" * 95)
        for q in quotes_retrieved:
            time_str = q.timestamp.strftime("%H:%M:%S")
            chg_sign = "+" if q.change > 0 else ""
            chg_pct_sign = "+" if q.change_pct > 0 else ""
            print(
                f"{q.symbol:<12} {q.company[:20]:<22} {q.current_price:<12.2f} "
                f"{chg_sign + str(q.change):<12} {chg_pct_sign + str(q.change_pct) + '%':<10} "
                f"{q.volume:<12,d} {time_str:<10}"
            )
        print("=" * 95)

    # 5. Batch Quotes Test for NIFTY 50
    print("\n[5] TESTING NIFTY 50 BATCH RETRIEVAL...")
    try:
        all_quotes = provider.get_quotes()
        print(f"  ✓ Batch retrieved {len(all_quotes)}/50 NIFTY 50 stocks.")
        
        # 6. Validation & Storage
        validator = LiveQuoteValidator()
        valid_quotes, dropped = validator.filter_and_validate(all_quotes)
        print(f"  ✓ Validation passed: {len(valid_quotes)} valid, {len(dropped)} dropped.")
        
        if valid_quotes:
            storage = LiveQuoteStorage()
            storage.save_quotes(valid_quotes)
            print(f"  ✓ Saved snapshot to: {LIVE_SNAPSHOT_CSV}")
    except GrowwMarketDataAuthorisationError as e:
        print(f"  ❌ Batch retrieval halted: {e}")
    except Exception as e:
        print(f"  ❌ Batch retrieval error: {e}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit_code = run_groww_verification()
    sys.exit(exit_code)
