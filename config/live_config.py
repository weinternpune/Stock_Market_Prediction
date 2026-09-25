"""
live_config.py
--------------
Centralized configuration for the Live Market Data subsystem.
Reads environment variables for API credentials and defines market hours,
timezone standards (Asia/Kolkata), and storage paths.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Literal

# Base paths
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
LIVE_DATA_DIR = DATA_DIR / "live"

CONSTITUENTS_CSV = METADATA_DIR / "nifty50_constituents.csv"
LIVE_SNAPSHOT_CSV = LIVE_DATA_DIR / "latest_nifty50_quotes.csv"
LIVE_HISTORY_PARQUET = LIVE_DATA_DIR / "live_quotes_history.parquet"
LIVE_HISTORY_CSV = LIVE_DATA_DIR / "live_quotes_history.csv"

# Timezone standard
KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
TIMEZONE_NAME = "Asia/Kolkata"

# Auto-load .env from project root if present
def _load_env_file(root_path: Path) -> None:
    for env_name in [".env", ".env.local"]:
        env_file = root_path / env_name
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass

_load_env_file(PROJECT_ROOT)

# Provider selection from environment
# Options: 'demo' (default), 'real'
MARKET_DATA_PROVIDER: str = os.getenv("MARKET_DATA_PROVIDER", "demo").strip().lower()

# Real broker API credentials (Zerodha Kite Connect v3)
MARKET_DATA_API_KEY: str = os.getenv("MARKET_DATA_API_KEY", "").strip()
MARKET_DATA_API_SECRET: str = os.getenv("MARKET_DATA_API_SECRET", "").strip()
MARKET_DATA_ACCESS_TOKEN: str = os.getenv("MARKET_DATA_ACCESS_TOKEN", "").strip()
MARKET_DATA_BASE_URL: str = os.getenv("MARKET_DATA_BASE_URL", "https://api.kite.trade").strip()
if not MARKET_DATA_BASE_URL:
    MARKET_DATA_BASE_URL = "https://api.kite.trade"

MARKET_DATA_ENVIRONMENT: Literal["sandbox", "production"] = (
    "production" if os.getenv("MARKET_DATA_ENVIRONMENT", "sandbox").strip().lower() == "production" else "sandbox"
)

# Market Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
PRE_OPEN_START_HOUR = 9
PRE_OPEN_START_MINUTE = 0

# Demo provider update interval (seconds)
DEMO_UPDATE_INTERVAL_SECONDS = 5
