# NIFTY 50 Stock Price Prediction & Analytics System

An institutional-grade, end-to-end stock price prediction and financial analytics system for the **NIFTY 50** constituent companies. Built as a single reusable, automated pipeline that ingests historical market data, validates data integrity, applies corporate split adjustments, engineers 73 technical and momentum features, trains 4 diverse time-series model families, evaluates out-of-sample test accuracy with zero data leakage, and renders findings in a professional 10-page Streamlit multipage application.

---

## 📌 Project Objective

The system addresses the challenge of algorithmic forecasting in emerging markets by building a rigorous, non-leaking machine-learning framework across the entire NIFTY 50 universe. Rather than creating 50 disconnected scripts, it implements **one generalized pipeline** operating on a stock-by-stock basis that:
1. Loads and validates historical daily OHLCV records over a 5-year span (2021–2026).
2. Backward-adjusts unadjusted corporate splits and bonuses (|return| > 38%).
3. Generates statistical, momentum, volatility, and historical lag features.
4. Formulates a **21-trading-day (~1 month)** forward target price ($P_{t+21}$).
5. Trains and contrasts **Naive Baseline**, **ARIMA**, **XGBoost**, and **PyTorch LSTM**.
6. Selects the empirically best model for each constituent on unseen test data.
7. Computes empirical 90% confidence intervals and ranks all 50 constituents.
8. Presents findings in an interactive 10-page Streamlit research dashboard.

> **Disclaimer**: This is an educational and analytical research project. Financial market time-series exhibit high noise and non-stationarity. Model predictions and rankings do NOT constitute financial advice or guaranteed investment returns.

---

## 📂 Project Folder Structure

```
Stock_Market_Prediction/
│
├── config/
│   └── project_config.py            # Centralized settings (horizons, windows, paths)
│
├── data/
│   ├── raw/
│   │   └── master_merged_stock_data_all_50_stocks.csv  # 5-Year uploaded raw dataset
│   ├── metadata/
│   │   └── nifty50_constituents.csv # 50 constituents metadata with sector mapping
│   ├── cleaned/
│   │   ├── cleaned_nifty50_data.parquet
│   │   └── cleaned_nifty50_data.csv
│   ├── processed/
│   │   ├── feature_engineered_data.parquet
│   │   └── feature_engineered_data.csv
│   └── predictions/
│       ├── stock_predictions.csv    # Final 21-day projections & uncertainty bounds
│       ├── stock_ranking.csv        # 50-stock sorted ranking table
│       ├── model_comparison.csv     # MAE, RMSE, MAPE per model and stock
│       └── backtest_predictions.csv # Holdout test actual vs predicted series
│
├── models/
│   └── {SYMBOL}/                    # Stock-specific serialized models & scalers
│       ├── xgboost_model.joblib
│       ├── lstm_weights.pt
│       └── metrics.json
│
├── notebooks/                       # Modular phase runner scripts
│   ├── 01_data_validation.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_feature_engineering.py
│   ├── 05_baseline_model.py
│   ├── 06_arima_model.py
│   ├── 07_xgboost_model.py
│   ├── 08_lstm_model.py
│   └── 09_model_evaluation.py
│
├── src/
│   ├── data/
│   │   ├── validator.py             # Phase 2: Master data validation
│   │   └── cleaner.py               # Phase 3-4: Cleaning & corporate action adjustments
│   ├── features/
│   │   ├── technical_indicators.py  # Phase 6: 73 features (SMA, EMA, RSI, MACD, BB)
│   │   └── target_builder.py        # Phase 7-8: +21D target & strict chronological split
│   ├── models/
│   │   ├── baseline.py              # Phase 9: Naive persistence & moving average
│   │   ├── arima_model.py           # Phase 10: Statistical ARIMA benchmark
│   │   ├── xgboost_model.py         # Phase 11: Gradient-boosted decision trees
│   │   └── lstm_model.py            # Phase 12: PyTorch recurrent neural network
│   ├── evaluation/
│   │   └── evaluator.py             # Phase 13-14: MAE, RMSE, MAPE, Directional Accuracy
│   ├── predictions/
│   │   └── forecaster.py            # Phase 15-17: Training loop, forecasts & rankings
│   └── utilities/
│       ├── logger.py                # Console UTF-8 and stock status tracking
│       └── io_helpers.py            # Parquet, CSV, and JSON serialization
│
├── results/
│   ├── metrics/
│   │   ├── validation_report.json
│   │   ├── cleaning_summary.json
│   │   ├── corporate_actions.json
│   │   ├── eda_summary.json
│   │   ├── correlation_matrix.csv
│   │   └── master_model_metrics.json
│   └── reports/
│       └── final_findings_report.md # Comprehensive analytical report
│
├── app/
│   ├── app.py                       # Streamlit multipage application entry point
│   ├── components/
│   │   ├── cards.py                 # KPI card & performance chip widgets
│   │   ├── charts.py                # Plotly financial charts & heatmaps
│   │   └── styling.py               # Custom CSS design system
│   └── pages/                       # 10 Dedicated Streamlit pages
│       ├── 01_overview.py
│       ├── 02_stock_analysis.py
│       ├── 03_stock_ranking.py
│       ├── 04_model_performance.py
│       ├── 05_sector_analysis.py
│       ├── 06_correlation_analysis.py
│       ├── 07_backtesting.py
│       ├── 08_future_predictions.py
│       ├── 09_data_quality.py
│       └── 10_methodology.py
│
├── tests/                           # 27 Unit and integration tests
│   ├── test_data_validation.py
│   ├── test_data_cleaning.py
│   ├── test_features.py
│   ├── test_leakage_and_target.py   # Critical zero-leakage test suite
│   ├── test_models.py
│   ├── test_predictions.py
│   └── test_dashboard.py
│
├── requirements.txt
└── README.md
```

---

## ⚡ Master Data Pipeline Workflow

The architecture follows a strict sequential pipeline:

```
MASTER RAW CSV (5 Years, 50 Stocks)
       │
       ▼
1. DATA VALIDATION (validator.py) ──> OHLC consistency, duplicate checks, constituent coverage
       │
       ▼
2. DATA CLEANING (cleaner.py) ──────> Format cleaning, deduplication, 12 split/bonus adjustments
       │
       ▼
3. FEATURE ENGINEERING ─────────────> 73 Technical, momentum, volatility & lag features
       │
       ▼
4. TARGET CONSTRUCTION ─────────────> Future_Close_21D = Close.shift(-21)
       │
       ▼
5. CHRONOLOGICAL SPLIT ─────────────> Forward split: Train (80%), Val (10%), Holdout Test (10%)
       │
       ▼
6. MULTI-MODEL TRAINING ────────────> Baseline, ARIMA(1,1,1), XGBoost, PyTorch LSTM
       │
       ▼
7. OUT-OF-SAMPLE EVALUATION ────────> MAE, RMSE, MAPE, Directional Accuracy, Baseline Improvement
       │
       ▼
8. BEST MODEL SELECTION ────────────> Objective selection per stock based on lowest test RMSE
       │
       ▼
9. FUTURE 21-DAY FORECASTS ─────────> Forward projections + 90% confidence uncertainty intervals
       │
       ▼
10. LEAGUE TABLE & RANKING ─────────> 50-stock sorted ranking by expected return %
       │
       ▼
11. STREAMLIT MULTIPAGE DASHBOARD ──> 10 interactive analytical pages
```

---

## 🤖 The Four Model Architectures

1. **Naive Persistence Baseline**
   - Formula: $\hat{P}_{t+21} = P_t$
   - Serves as the fundamental random-walk null hypothesis. Any sophisticated model must outperform this benchmark to demonstrate genuine forecasting skill.

2. **Statistical Benchmark: ARIMA(1, 1, 1)**
   - Univariate time-series model fit on training closing prices.
   - Captures low-order autoregressive and moving-average dynamics with drift projection.

3. **Machine Learning: XGBoost Regressor**
   - Gradient-boosted decision trees trained on 66 engineered numeric indicators.
   - Hyperparameters: `n_estimators=120`, `max_depth=4`, `learning_rate=0.04`, `subsample=0.8`, `colsample_bytree=0.8`.
   - Feature importances logged per constituent.

4. **Deep Learning: PyTorch Stacked LSTM**
   - Recurrent Neural Network with a 20-session lookback window.
   - Architecture: Stacked LSTM (`hidden_dim=32`) $\rightarrow$ Dense projection head with ReLU.
   - Scalers fit strictly on the training partition to prevent lookahead contamination.

---

## 🖥️ Streamlit Multipage Dashboard (10 Pages)

The interactive dashboard uses Streamlit's modern `st.navigation` and `st.Page` system:

| # | Page | Description |
|---|---|---|
| 1 | **🏠 Overview** | High-level market KPIs, Top 5 gainers & losers, win count distribution, automated market summary. |
| 2 | **📈 Stock Analysis** | Single-stock deep dive: Candlestick chart with SMA 20/50/200, Volume, MACD, RSI, 21-day forecast KPI, and 4-model comparison table. |
| 3 | **🏆 Stock Ranking** | Master 50-stock league table sorted by expected return %, with industry, model, and return filters, and CSV export. |
| 4 | **🤖 Model Performance** | Cross-model win count bar chart, error boxplots (MAE, RMSE, MAPE), and stock-by-stock scorecard. |
| 5 | **🏭 Sector Analysis** | Industry breakdown of constituent count, average return, volatility, and sector champion rankings. |
| 6 | **🔗 Correlation Analysis** | Interactive Plotly heatmap of daily return correlations across all constituents with industry/custom filters. |
| 7 | **📊 Backtesting** | Out-of-sample actual vs predicted curves across held-out test sessions, prediction error distribution, and residual metrics. |
| 8 | **🔮 Future Predictions** | Dedicated forward 21-day forecast view with empirical 90% confidence intervals and one-click CSV export. |
| 9 | **🧹 Data Quality** | Institutional audit: OHLC validation (0 violations), missingness, deduplication, and corporate action log. |
| 10 | **ℹ️ Methodology & Disclaimer** | Full mathematical formulation, training pipeline, critical limitations, and non-investment advice notice. |

---

## 🔒 Zero Data Leakage Verification

Preventing data leakage is paramount in financial machine learning. The system strictly guarantees:
1. **Target Forward Shift**: $P_{t+21}$ is constructed using `shift(-21)`. The final 21 sessions of the dataset have NaN targets and are strictly reserved for future inference.
2. **Feature Isolation**: Target variables and forward dates are strictly excluded from the predictor feature matrix $X$.
3. **No Shuffling**: Train, validation, and test splits are divided chronologically forward in time.
4. **Scaler Encapsulation**: `StandardScaler` and `MinMaxScaler` are fitted strictly on training data and applied via transform on validation/test data.
5. **Automated Leakage Tests**: Dedicated tests in `tests/test_leakage_and_target.py` mathematically verify these conditions.

---

## 🚀 Installation & Execution

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd Stock_Market_Prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Automated Pipeline
To execute data validation, cleaning, corporate split adjustment, feature engineering, model training, evaluation, and ranking in one unified step:
```powershell
python src/pipeline.py
```

### 3. Run Automated Tests
```powershell
python -m unittest discover tests
```

### 4. Launch the Streamlit Dashboard
```powershell
streamlit run app/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Summary of Model Findings

- **Constituents Modeled**: 49 active continuous corporate entities (including corporate continuity mappings: `SRTRANSFIN` $\rightarrow$ `SHRIRAMFIN`, `ETERNAL` $\rightarrow$ `ZOMATO`, and `TMPV` $\rightarrow$ `TATAMOTORS`) + `JIOFIN`.
- **Total Historical Observations**: 60,271 daily sessions across 5 years.
- **Corporate Actions Adjusted**: 12 historical splits and bonus share events backward-adjusted.
- **Model Win Distribution**:
  - **PyTorch LSTM**: Best model for 18 constituents (particularly in volatile and momentum regimes).
  - **Baseline (Naive Persistence)**: Best model for 17 constituents (reflecting strong random-walk efficiency in high-cap stocks).
  - **ARIMA(1, 1, 1)**: Best model for 8 constituents (stable mean-reverting trends).
  - **XGBoost**: Best model for 6 constituents (complex non-linear interactions).

---

## ⚠️ Limitations & Analytical Caveats

1. **Market Unpredictability**: Financial markets are influenced by macroeconomic policy, interest rates, geopolitics, and corporate earnings that cannot be anticipated from historical price action alone.
2. **Survivorship Bias**: The dataset uses the current NIFTY 50 index constituent list. Companies excluded from the index during 2021–2026 are not captured.
3. **Corporate Actions**: While major splits and bonus issues have been backward-adjusted, unadjusted dividend distributions may introduce minor discrepancies.
4. **Forecast Uncertainty**: Projections represent central statistical expectations subject to wide confidence bounds.

---

## ⚖️ Disclaimer

*This application is an educational and analytical project. Predictions are generated by statistical and machine-learning models based on historical data. They are uncertain and should not be interpreted as investment advice, guaranteed returns, or recommendations to buy or sell any security.*
