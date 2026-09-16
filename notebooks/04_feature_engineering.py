"""
04_feature_engineering.py
-------------------------
Notebook runner for Phase 6 & 7: Feature Engineering & Target Construction.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.technical_indicators import run_feature_engineering

if __name__ == "__main__":
    print(">>> Executing Phase 6 & 7: Feature Engineering...")
    feat_df = run_feature_engineering()
    print(f"Engineered {len(feat_df.columns)} features across {len(feat_df)} rows.")
