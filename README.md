# 📈 Nifty 500 Stock Market Prediction Pipeline & Analytics Dashboard
### Data Analytics Intern Project · Product Requirements Document (PRD v1.1)
**Author:** Data Analytics Intern | **Status:** Complete & Scientifically Verified | **Date:** September 2026

---

## 🚀 Project Overview
This repository contains an end-to-end quantitative analytics and predictive modeling system for the **Nifty 500 index**, the broad-market benchmark representing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**. 

The project strictly follows the **Data Analytics Intern PRD v1.1**, covering the full quantitative lifecycle:
1. **Data Collection & Reconciliation:** Sourcing 5 years of daily market data (**September 2, 2021 to September 2, 2026**, 1,229 active trading sessions) from **NSE** (`^CRSLDX`) and cross-reconciled against the **BSE 500** (`BSE-500.BO`).
2. **Trading Calendar Integrity:** Strict adherence to the official exchange trading calendar (~250 trading days/year), ensuring **zero artificial weekend or holiday observations** are manufactured. Post-cleaning missing data: **0.00%** (PRD Target: $< 2\%$).
3. **Exploratory Data Analysis:** Price trends, moving average Golden/Death crosses, return leptokurtosis, and volatility clustering.
4. **Feature Engineering:** 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, rolling volatility, price/return lags).
5. **Multi-Family Predictive Modeling:**
   - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) & 5-Day Moving Average
   - **Statistical Model:** ARIMA(1, 1, 1) with Augmented Dickey-Fuller stationarity validation
   - **Classical ML:** Random Forest & XGBoost Regressors with feature importances
   - **Deep Learning:** PyTorch Long Short-Term Memory (LSTM) Neural Network
6. **Rigorous Statistical Significance Testing:** Directional hit rate validated via formal **Binomial Hypothesis Testing** ($H_0: p = 0.50$) with exact $p$-values and Wilson confidence intervals.
7. **Future Target Price Forecasting:** Genuine multi-step forward forecasting ($T+1$ to $T+30$ trading days) driven by **actual trained predictive models** (recursive ML rollouts and statistical ARIMA).
8. **Interactive Streamlit Application:** Full multi-page interactive web dashboard (`app/app.py`).

---

## 🏆 Model Performance Benchmark Scorecard (PRD FR7)

Evaluated strictly on the out-of-sample test split (**October 29, 2025 to September 1, 2026**, 206 trading sessions):

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate (%) | Binomial Test ($p$-val) | 95% Wilson CI | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **208.51** | **151.69** | **0.668%** | — | — | — | **+0.00%** |
| **Moving Average (5-day SMA)** | Benchmark | 288.97 | 219.52 | 0.963% | 50.97% | 0.4172 | [44.19%, 57.72%] | -38.59% |
| **ARIMA(1, 1, 1)** | Statistical | 287.52 | 216.00 | 0.948% | 50.97% | 0.4172 | [44.19%, 57.72%] | -37.89% |
| **Random Forest Regressor** | Classical ML | **223.29** | **167.39** | **0.735%** | 49.51% | 0.5828 | [42.76%, 56.29%] | -7.09% |
| **XGBoost Regressor** | Classical ML | **226.94** | **165.34** | **0.727%** | **57.28%** | **0.0215\*** | **[50.45%, 63.84%]** | -8.84% |
| **PyTorch LSTM Network** | Deep Learning | 434.21 | 357.37 | 1.559% | **54.85%** | 0.0927 | [48.03%, 61.50%] | -108.24% |

*\*Statistically significant at $\alpha = 0.05$ (one-sided exact Binomial test against 50% random walk).*

---

## 🔬 Scientific Findings & Key Insights

1. **Market Efficiency & Persistence Hurdle:** Financial index price levels approximate a martingale random walk ($\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$). Today's price is the minimum-variance quadratic estimator of tomorrow's price, explaining why no model beats the naive baseline on level-price RMSE.
2. **Statistical Significance of XGBoost Directional Edge:** XGBoost correctly predicted next-day market direction on **118 of 206 test days (57.28%)**. A formal one-sided Binomial hypothesis test yields $p = 0.0215$ (rejects random walk null at 5% significance level, 95% Wilson CI: $[50.45\%, 63.84\%]$). *Caveat:* With the lower confidence boundary near 50% and trading execution frictions unmodeled, this is an empirical statistical tilt rather than guaranteed trading alpha.
3. **Scientific Reality of Deep Learning (LSTM):** While the PyTorch LSTM achieved a 54.85% directional hit rate, its level-price RMSE of **434.21** was more than double the persistence baseline (208.51). Non-stationarity and error drift across sliding windows make raw price-level forecasting inherently difficult for deep neural networks, providing empirical validation for why quantitative finance targets stationary returns rather than raw price levels.
4. **Broad-Market Co-Movement:** Sourcing from both NSE and BSE revealed a price correlation of **0.9999** and return correlation of **0.9979**, demonstrating **strong common market dynamics** between the broad-market indices across both major Indian exchanges.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Clone & Navigate to the Repository
```bash
git clone https://github.com/Priyanka-38/Stock_Market_Prediction.git
cd Stock_Market_Prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the Automated Predictive Pipeline
To ingest data, engineer features, fit models, compute binomial hypothesis tests, and save predictions:
```bash
python src/pipeline.py
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Jupyter Notebook
The fully executed, publication-ready notebook is located at:
`notebooks/02_nifty500_prediction_pipeline.ipynb`
To launch JupyterLab or Notebook:
```bash
jupyter notebook notebooks/02_nifty500_prediction_pipeline.ipynb
```

---

## 📂 Repository Directory Structure
```
Stock_Market_Prediction/
├── app/
│   └── app.py                     # Streamlit 7-page interactive dashboard
├── data/
│   ├── raw/                       # Raw 5-year OHLCV CSVs (NSE & BSE)
│   ├── processed/                 # Cleaned dataset & backtest predictions
│   └── features/                  # 15+ engineered features dataset
├── models/
│   ├── saved_models/              # Saved model weights, scalers, and configs
│   ├── metrics_summary.json       # Benchmark metrics & binomial test stats
│   └── feature_importances.csv    # Tree feature importances
├── notebooks/
│   ├── 01_project_setup.ipynb     # Setup & environment verification
│   └── 02_nifty500_prediction_pipeline.ipynb  # Comprehensive executed notebook
├── reports/
│   └── final_findings_report.md   # Final intern findings report
├── src/
│   ├── data_collection.py         # 5Y data collection script
│   ├── data_preprocessing.py      # Cleaning & NSE/BSE reconciliation
│   ├── feature_engineering.py     # Technical indicator engineering
│   ├── evaluate.py                # Metrics & Binomial hypothesis testing
│   ├── pipeline.py                # End-to-end automated pipeline
│   └── models/
│       ├── baseline.py            # Naive & Moving Average baselines
│       ├── arima_model.py         # Statistical ARIMA(1,1,1) & ADF test
│       ├── ml_models.py           # Random Forest & XGBoost
│       ├── lstm_model.py          # PyTorch 2-layer LSTM sequence model
│       └── forecast_service.py    # Genuine model-driven multi-step forecasting
├── generate_notebook.py           # Programmatic notebook generator
├── requirements.txt               # Locked production dependencies
└── README.md                      # Comprehensive project documentation
```

---

## ⚖️ Academic & Non-Goals Disclaimer
This software was developed strictly as an educational Data Analytics Intern project. It is **not** financial advice, nor is it intended for live automated trading or real capital investment decisions.
