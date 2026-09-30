"""
pipeline.py
-----------
Master End-to-End Execution Pipeline for NIFTY 50 Stock Price Prediction.
Orchestrates Phases 1 through 17 in a single, reproducible command:
Master Raw Data -> Validation -> Cleaning -> Corporate Actions -> Features -> Models -> Evaluation -> Forecasts -> Ranking.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import validate_master_data
from src.data.cleaner import clean_data
from src.eda import run_eda
from src.features.technical_indicators import run_feature_engineering
from src.predictions.forecaster import run_predictions
from src.utilities.logger import get_logger

logger = get_logger("MasterPipeline")

def run_end_to_end_pipeline():
    """Executes the entire NIFTY 50 stock price prediction pipeline."""
    logger.info("=" * 75)
    logger.info(">>> NIFTY 50 STOCK MARKET PREDICTION: MASTER AUTOMATED PIPELINE <<<")
    logger.info("=" * 75)
    
    # Phase 2: Data Validation
    logger.info("\n>>> STEP 1: MASTER DATASET VALIDATION (Phase 2)")
    val_report, _ = validate_master_data()
    logger.info(f"Validation Status: {val_report['validation_status']} (Total rows: {val_report['total_rows']})")
    
    # Phase 3 & 4: Data Cleaning & Corporate Actions
    logger.info("\n>>> STEP 2: DATA CLEANING & CORPORATE SPLIT ADJUSTMENTS (Phase 3 & 4)")
    clean_df, clean_report = clean_data()
    logger.info(f"Cleaned dataset: {len(clean_df)} records, {len(clean_report['corporate_actions_detected'])} corporate actions adjusted.")
    
    # Phase 5: Exploratory Data Analysis
    logger.info("\n>>> STEP 3: EXPLORATORY DATA ANALYSIS (Phase 5)")
    eda_report = run_eda()
    logger.info(f"EDA completed across {eda_report['total_stocks']} stocks. Avg 5Y Return: {eda_report['average_cumulative_return_pct']}%.")
    
    # Phase 6 & 7: Feature Engineering
    logger.info("\n>>> STEP 4: TECHNICAL & STATISTICAL FEATURE ENGINEERING (Phase 6 & 7)")
    feat_df = run_feature_engineering()
    logger.info(f"Engineered {len(feat_df.columns)} features across {len(feat_df)} rows.")
    
    # Phase 8 - 17: Model Training, Evaluation, Future Forecasting & Ranking
    logger.info("\n>>> STEP 5: TIME-SERIES MODELING, EVALUATION & RANKING (Phase 8 - 17)")
    stock_results = run_predictions()
    logger.info(f"Modeling complete for {len(stock_results)} stocks.")
    
    logger.info("\n" + "=" * 75)
    logger.info(">>> PIPELINE EXECUTION SUCCESSFULLY COMPLETED! <<<")
    logger.info("Launch the dashboard with: streamlit run app/app.py")
    logger.info("=" * 75)

if __name__ == "__main__":
    run_end_to_end_pipeline()
