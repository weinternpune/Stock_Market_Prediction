"""
logger.py
---------
Structured logging utility for NIFTY 50 prediction pipeline.
Handles Windows console UTF-8 stream re-encoding and status tracking.
"""

import sys
import logging

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_logger(name: str = "Nifty50Pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class PipelineProgressTracker:
    """Tracks stock-by-stock processing progress, successes, and failures."""
    
    def __init__(self):
        self.successful_stocks = []
        self.failed_stocks = {}
        self.logger = get_logger("ProgressTracker")
        
    def record_success(self, symbol: str, best_model: str, expected_return: float):
        self.successful_stocks.append({
            "symbol": symbol,
            "best_model": best_model,
            "expected_return": expected_return
        })
        self.logger.info(f"✔ [{symbol}] Successfully completed pipeline. Best Model: {best_model} ({expected_return:+.2f}%)")
        
    def record_failure(self, symbol: str, stage: str, error_msg: str):
        self.failed_stocks[symbol] = {"stage": stage, "error": error_msg}
        self.logger.error(f"✖ [{symbol}] FAILED at stage '{stage}': {error_msg}")
        
    def summary(self):
        total = len(self.successful_stocks) + len(self.failed_stocks)
        self.logger.info("=" * 65)
        self.logger.info(f"PIPELINE SUMMARY: Total={total} | Success={len(self.successful_stocks)} | Failed={len(self.failed_stocks)}")
        if self.failed_stocks:
            for sym, info in self.failed_stocks.items():
                self.logger.warning(f"  - Failed Stock: {sym} (Stage: {info['stage']}, Error: {info['error']})")
        self.logger.info("=" * 65)
