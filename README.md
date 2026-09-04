# 📈 Nifty 500 Stock Market Prediction Pipeline & Analytics Dashboard
### Data Analytics Intern Project · Product Requirements Document (PRD v1.1)
**Author:** Data Analytics Intern | **Target:** Nifty 500 (NSE) | **Date:** September 2026

---

## 🚀 Project Overview

This repository provides an end-to-end quantitative analytics and predictive modeling system for the **Nifty 500 index**, the broad-market benchmark capturing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**.

The project strictly follows the **60-step roadmap across all 15 phases**:
1. **Authoritative Data Ingestion:** 5 years of daily market data (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions) from the official NSE index archive (`data/raw/nse_nifty500_raw.csv`).
2. **Cross-Exchange Proxy Reconciliation:** Sourced BSE 500 index (`data/raw/bse_500_raw.csv`, 1,240 common sessions; Price correlation: **0.99989**, Return correlation: **0.99930**) as a secondary reference proxy without price averaging or merging.
3. **Data Cleaning & Calendar Integrity:** 100% adherence to exchange trading calendars with **0.00% missing data** post-cleaning. Extreme macro return shocks (Russia-Ukraine War, 2024 Election Day) were verified and retained to prevent downside risk censorship bias.
4. **Feature Engineering:** 15+ technical indicators (SMA 10/20/50/200, EMA 12/26, 14-day RSI, MACD, Bollinger Bands, rolling volatility, return lags) across 1,040 clean modeling rows with zero lookahead bias.
5. **Multi-Family Predictive Modeling:**
   - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) & 5-Day SMA.
   - **Statistical Model:** Pure walk-forward ARIMA(1, 1, 1) rolling one step ahead with zero leakage.
   - **Classical ML:** Random Forest & XGBoost Regressors with 5-fold expanding-window cross-validation.
   - **Deep Learning:** PyTorch Stacked Long Short-Term Memory (LSTM) Neural Network.
6. **Transparent Performance Benchmark Scorecard:** Documented why the Naive Persistence baseline achieves the lowest level-price RMSE under the Martingale property of asset prices.
7. **Forward Forecasting:** Recursive forward projections from $T+1$ to $T+30$ trading sessions with expanding 95% volatility confidence bands.
8. **Interactive Streamlit Web Dashboard & Presentation:** Full 8-page web application (`app/app.py`), academic findings report (`reports/final_findings_report.md`), and an interactive 15-slide presentation deck (`reports/presentation_deck.html`).

---

## 🏆 Model Performance Benchmark Scorecard

Evaluated on the held-out out-of-sample test split (**October 28, 2025 to August 28, 2026**, 208 trading sessions):

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | Benchmark | **209.33** | **151.88** | **0.669%** | **N/A** | **+0.00% (Best)** |
| **Random Forest Regressor** | Classical ML | **229.66** | **167.47** | **0.735%** | 49.52% | -9.71% |
| **Moving Average (5-Day SMA)** | Benchmark | 288.87 | 218.16 | 0.957% | 51.92% | -37.99% |
| **ARIMA(1, 1, 1) Walk-Forward** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | -39.05% |
| **XGBoost Regressor** | Classical ML | 299.91 | 239.62 | 1.044% | 51.92% | -43.27% |
| **PyTorch LSTM Network** | Deep Learning | 560.27 | 453.11 | 1.974% | **52.88%** | -167.64% |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Anaconda Python recommended)
- Git

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Automated End-to-End Pipeline
Executes raw data validation, cleaning, EDA, feature engineering, model training, backtesting, and forward forecasting in a single command:
```bash
python src/pipeline.py
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📁 Repository Structure

```
Stock_Market_Prediction/
├── app/
│   └── app.py                      # 8-Page Interactive Streamlit Web Application
├── data/
│   ├── raw/                        # Untouched raw datasets (NSE Nifty 500 & BSE 500)
│   ├── processed/                  # Cleaned master series & test predictions
│   └── features/                   # 15+ engineered technical indicators
├── models/
│   ├── saved_models/               # Persisted model weights and scalers
│   ├── metrics_summary.json        # Master evaluation scorecard
│   └── outlier_investigation.csv   # Historical market shock audit
├── reports/
│   ├── final_findings_report.md    # 14-section academic findings report
│   ├── presentation_deck.html      # Interactive 15-slide executive presentation
│   └── PRD_v1.1.pdf                # Original Product Requirements Document
├── src/
│   ├── data_validation.py          # Phases 6-8: OHLC integrity & reconciliation
│   ├── data_cleaning.py            # Phases 9-11: Master cleaning & outlier audit
│   ├── eda.py                      # Phases 12-16: Volatility, normality, seasonality
│   ├── feature_engineering.py      # Phases 17-27: Technical features & target creation
│   ├── models/
│   │   ├── baseline.py             # Phases 28-29: Naive & 5-Day SMA baselines
│   │   ├── arima_model.py          # Phases 30-31: Walk-forward ARIMA(1,1,1)
│   │   ├── ml_models.py            # Phases 32-35: Random Forest & XGBoost
│   │   ├── lstm_model.py           # Phases 36-40: PyTorch LSTM Network
│   │   └── forecast_service.py     # Phase 47: Forward forecasting service
│   ├── evaluate.py                 # Phases 41-46: Backtesting & evaluation metrics
│   └── pipeline.py                 # Phase 57: Master pipeline orchestrator
├── requirements.txt                # Pinned dependencies
└── README.md                       # Complete documentation
```

---

## ⚠️ Disclaimer
*This project is conducted strictly for educational and analytical skill-building purposes under noisy financial time-series. Any predictions or findings generated should not be construed as investment advice or used for capital allocation.*
