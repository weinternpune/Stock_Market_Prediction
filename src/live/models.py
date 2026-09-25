"""
models.py
---------
Standardized data structures for the Live Market Data subsystem.
Ensures uniform representation of real-time quotes across demo and production providers.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")

@dataclass
class MarketQuote:
    """
    Standardized live market quote data model.
    Encapsulates real-time pricing, volume, day boundaries, and market status.
    """
    symbol: str
    company: str
    timestamp: datetime
    open: float
    high: float
    low: float
    current_price: float
    previous_close: float
    change: float
    change_pct: float
    volume: int
    day_high: float
    day_low: float
    market_status: str
    is_demo: bool = False
    data_source: str = "LIVE"

    def __post_init__(self):
        # Enforce timezone awareness in Asia/Kolkata
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=KOLKATA_TZ)
        else:
            self.timestamp = self.timestamp.astimezone(KOLKATA_TZ)
            
        # Ensure numerical rounding
        self.open = round(float(self.open), 2)
        self.high = round(float(self.high), 2)
        self.low = round(float(self.low), 2)
        self.current_price = round(float(self.current_price), 2)
        self.previous_close = round(float(self.previous_close), 2)
        self.day_high = round(float(self.day_high), 2)
        self.day_low = round(float(self.day_low), 2)
        self.change = round(float(self.change), 2)
        self.change_pct = round(float(self.change_pct), 2)
        self.volume = int(self.volume)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes quote to dictionary format with ISO-8601 Asia/Kolkata timestamp."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        # Explicit user-facing key mapping as required by specification
        d["Symbol"] = self.symbol
        d["Company"] = self.company
        d["Timestamp"] = self.timestamp.isoformat()
        d["Open"] = self.open
        d["High"] = self.high
        d["Low"] = self.low
        d["Current Price"] = self.current_price
        d["Previous Close"] = self.previous_close
        d["Change"] = self.change
        d["Change %"] = self.change_pct
        d["Volume"] = self.volume
        d["Day High"] = self.day_high
        d["Day Low"] = self.day_low
        d["Market Status"] = self.market_status
        d["is_demo"] = self.is_demo
        d["data_source"] = self.data_source
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketQuote":
        """Instantiates MarketQuote from a dictionary representation."""
        # Support both lowercase and title-cased keys
        ts_val = data.get("timestamp") or data.get("Timestamp")
        if isinstance(ts_val, str):
            ts = datetime.fromisoformat(ts_val)
        elif isinstance(ts_val, datetime):
            ts = ts_val
        else:
            ts = datetime.now(KOLKATA_TZ)

        return cls(
            symbol=str(data.get("symbol") or data.get("Symbol")),
            company=str(data.get("company") or data.get("Company")),
            timestamp=ts,
            open=float(data.get("open") if data.get("open") is not None else data.get("Open", 0.0)),
            high=float(data.get("high") if data.get("high") is not None else data.get("High", 0.0)),
            low=float(data.get("low") if data.get("low") is not None else data.get("Low", 0.0)),
            current_price=float(data.get("current_price") if data.get("current_price") is not None else data.get("Current Price", 0.0)),
            previous_close=float(data.get("previous_close") if data.get("previous_close") is not None else data.get("Previous Close", 0.0)),
            change=float(data.get("change") if data.get("change") is not None else data.get("Change", 0.0)),
            change_pct=float(data.get("change_pct") if data.get("change_pct") is not None else data.get("Change %", 0.0)),
            volume=int(data.get("volume") if data.get("volume") is not None else data.get("Volume", 0)),
            day_high=float(data.get("day_high") if data.get("day_high") is not None else data.get("Day High", 0.0)),
            day_low=float(data.get("day_low") if data.get("day_low") is not None else data.get("Day Low", 0.0)),
            market_status=str(data.get("market_status") or data.get("Market Status", "UNKNOWN")),
            is_demo=bool(data.get("is_demo", False)),
            data_source=str(data.get("data_source", "LIVE"))
        )
