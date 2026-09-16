"""
09_model_evaluation.py
----------------------
Notebook runner for Phase 13 & 14: Model Comparison & Master Evaluation.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config.project_config import MODEL_COMPARISON_CSV, PREDICTIONS_CSV

if __name__ == "__main__":
    print(">>> Executing Phase 13 & 14: Model Evaluation & Comparison Inspection...")
    if MODEL_COMPARISON_CSV.exists() and PREDICTIONS_CSV.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_CSV)
        pred_df = pd.read_csv(PREDICTIONS_CSV)
        
        print("\nModel Architecture Performance Summary:")
        print(comp_df.groupby('Model')[['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']].mean().round(2))
        
        print("\nModel Win Counts across NIFTY 50 Universe:")
        print(pred_df['Best_Model'].value_counts())
    else:
        print("Model comparison artifacts not found. Please run the forecasting pipeline first.")
