"""
validator.py
------------
Data validation and duplicate-checking component for Live Market Data.
Enforces strict physical and mathematical invariants across all real-time quotes,
guaranteeing data hygiene before storage or consumption.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Optional, Any
from zoneinfo import ZoneInfo

from src.live.models import MarketQuote
from config.live_config import KOLKATA_TZ
from src.utilities.logger import get_logger

logger = get_logger("LiveQuoteValidator")

class LiveQuoteValidationError(ValueError):
    """Raised when a market quote fails critical integrity validation."""
    pass


class LiveQuoteValidator:
    """
    Validates live quotes for schema completeness, price positivity,
    OHLC logical consistency, timezone integrity, and deduplication.
    """

    def __init__(self, max_duplicate_cache_size: int = 5000):
        # Ring-buffer of seen (symbol, timestamp_iso) to prevent duplicates
        self._seen_quote_keys: Set[str] = set()
        self._seen_keys_order: List[str] = []
        self._max_cache_size = max_duplicate_cache_size

    def validate_quote(self, quote: MarketQuote) -> Tuple[bool, List[str]]:
        """
        Validates an individual quote against institutional financial integrity rules.
        Returns (is_valid, list_of_error_reasons).
        """
        errors = []

        # 1. Symbol & Company
        if not quote.symbol or not isinstance(quote.symbol, str) or not quote.symbol.strip():
            errors.append("Symbol cannot be empty.")
        if not quote.company or not isinstance(quote.company, str) or not quote.company.strip():
            errors.append("Company cannot be empty.")

        # 2. Timezone handling (Must be Asia/Kolkata)
        if quote.timestamp.tzinfo is None:
            errors.append("Timestamp must be timezone-aware (Asia/Kolkata).")
        else:
            # Check timezone name
            tz_str = str(quote.timestamp.tzinfo)
            if "Kolkata" not in tz_str and "IST" not in tz_str and "+05:30" not in quote.timestamp.isoformat():
                errors.append(f"Timestamp timezone must be Asia/Kolkata, got: {tz_str}")

        # Check timestamp not in future (allow 60s clock skew)
        now_ist = datetime.now(KOLKATA_TZ)
        if quote.timestamp > now_ist + timedelta(seconds=60):
            errors.append(f"Timestamp is in the future: {quote.timestamp.isoformat()} > {now_ist.isoformat()}")

        # 3. Price positivity
        for price_field in ["open", "high", "low", "current_price", "previous_close", "day_high", "day_low"]:
            val = getattr(quote, price_field, None)
            if val is None or val <= 0.0 or not isinstance(val, (int, float)):
                errors.append(f"Field '{price_field}' must be strictly positive (> 0), got: {val}")

        # 4. Volume non-negativity
        if quote.volume is None or quote.volume < 0:
            errors.append(f"Volume must be non-negative (>= 0), got: {quote.volume}")

        # 5. OHLC and Day High/Low Logical consistency
        if quote.day_high < quote.day_low:
            errors.append(f"Logical violation: Day High ({quote.day_high}) < Day Low ({quote.day_low})")

        max_observed = max(quote.open, quote.current_price, quote.high)
        if quote.day_high < max_observed - 0.01:
            errors.append(f"Logical violation: Day High ({quote.day_high}) < max(Open, Current, High) ({max_observed})")

        min_observed = min(quote.open, quote.current_price, quote.low)
        if quote.day_low > min_observed + 0.01:
            errors.append(f"Logical violation: Day Low ({quote.day_low}) > min(Open, Current, Low) ({min_observed})")

        # 6. Change & Change % consistency (with rounding tolerance)
        if quote.previous_close > 0:
            expected_change = round(quote.current_price - quote.previous_close, 2)
            if abs(quote.change - expected_change) > 0.05:
                errors.append(f"Change mismatch: got {quote.change}, expected {expected_change}")

            expected_pct = round((expected_change / quote.previous_close) * 100.0, 2)
            if abs(quote.change_pct - expected_pct) > 0.1:
                errors.append(f"Change % mismatch: got {quote.change_pct}%, expected {expected_pct}%")

        # 7. Market Status
        if not quote.market_status or not isinstance(quote.market_status, str):
            errors.append("Market Status field cannot be empty.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Quote validation failed for {quote.symbol}: {errors}")
        return is_valid, errors

    def is_duplicate(self, quote: MarketQuote) -> bool:
        """
        Determines if a quote has already been registered in the recent sliding window.
        Uses composite key: (Symbol, Timestamp_To_Second).
        """
        ts_sec = quote.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        key = f"{quote.symbol}_{ts_sec}"
        return key in self._seen_quote_keys

    def register_quote(self, quote: MarketQuote) -> None:
        """Registers quote key in sliding deduplication cache."""
        ts_sec = quote.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        key = f"{quote.symbol}_{ts_sec}"
        if key not in self._seen_quote_keys:
            self._seen_quote_keys.add(key)
            self._seen_keys_order.append(key)

            # Evict oldest if exceeding capacity
            if len(self._seen_keys_order) > self._max_cache_size:
                evicted = self._seen_keys_order.pop(0)
                self._seen_quote_keys.discard(evicted)

    def filter_and_validate(self, quotes: List[MarketQuote]) -> Tuple[List[MarketQuote], List[Dict[str, Any]]]:
        """
        Filters out duplicates and invalid quotes from a batch.
        Returns:
            (valid_non_duplicate_quotes, dropped_records_audit)
        """
        valid_quotes = []
        dropped_records = []

        for q in quotes:
            is_valid, errors = self.validate_quote(q)
            if not is_valid:
                dropped_records.append({
                    "symbol": q.symbol,
                    "reason": "VALIDATION_FAILURE",
                    "details": errors,
                    "quote": q.to_dict()
                })
                continue

            if self.is_duplicate(q):
                dropped_records.append({
                    "symbol": q.symbol,
                    "reason": "DUPLICATE_TICK",
                    "details": [f"Duplicate quote detected for {q.symbol} at {q.timestamp.isoformat()}"],
                    "quote": q.to_dict()
                })
                continue

            self.register_quote(q)
            valid_quotes.append(q)

        return valid_quotes, dropped_records
