# 📈 Nifty 500 Stock Market Prediction Pipeline & Analytics Dashboard
### Data Analytics Intern Project · Product Requirements Document (PRD v1.1)
**Author:** Data Analytics Intern | **Status:** Complete & Scientifically Verified | **Date:** September 2026

---

## 🚀 Project Overview
This repository contains an end-to-end quantitative analytics and predictive modeling system for the **Nifty 500 index**, the broad-market benchmark representing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**. 

The project strictly follows the **Data Analytics Intern PRD v1.1**, covering the full quantitative lifecycle:
1. **Authoritative Data Sourcing:** Five years of daily market data (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions) sourced directly from the **official NSE historical index archive** (`data/NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv`) with the complete required schema: `Date, Open, High, Low, Close, Volume, Adjusted Close`.
2. **Cross-Exchange Proxy Validation:** Sourced the **BSE 500 index** (`BSE-500.BO`) as a cross-exchange broad-market proxy for reconciliation and co-movement validation across 1,229 common trading sessions (Price correlation: **0.9999**, Return correlation: **0.9969**).
3. **Trading Calendar Integrity:** Strict adherence to the official exchange trading calendar (~250 trading days/year), ensuring **zero artificial weekend or holiday observations** are manufactured. Post-cleaning missing data: **0.00%** (PRD Target: $< 2\%$).
4. **Feature Engineering:** 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, rolling volatility, price/return lags) across 1,040 feature rows with zero lookahead bias.
5. **Multi-Family Predictive Modeling:**
   - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) & 5-Day Moving Average
   - **Statistical Model:** ARIMA(1, 1, 1) with Augmented Dickey-Fuller stationarity validation
   - **Classical ML:** Random Forest & XGBoost Regressors with feature importances
   - **Deep Learning:** PyTorch Long Short-Term Memory (LSTM) Neural Network
6. **Rigorous Econometric & Statistical Evaluation:** Evaluated on RMSE, MAE, MAPE, and Directional Hit Rate, backed by formal **Binomial Hypothesis Testing**.
7. **Future Target Price Forecasting:** Genuine multi-step forward forecasting ($T+1$ to $T+30$ trading days) driven by **actual trained predictive models** (recursive ML rollouts and statistical ARIMA).
8. **Interactive Streamlit Application & Presentation Deck:** Full 8-page interactive web dashboard (`app/app.py`), 10-slide presentation deck (`reports/presentation_deck.md` & `reports/presentation_deck.html`), and executed Jupyter Notebook (`notebooks/02_nifty500_prediction_pipeline.ipynb`).

---

## 🏆 Model Performance Benchmark Scorecard (PRD FR7)

Evaluated strictly on the out-of-sample test split (**October 28, 2025 to August 28, 2026**, 208 trading sessions):

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate (%) | Binomial Test ($p$-val) | 95% Wilson CI | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **209.33** | **151.88** | **0.669%** | — | — | — | **+0.00% (Best RMSE)** |
| **Random Forest Regressor** | Classical ML | **228.48** | **169.05** | **0.742%** | 48.56% | 0.6862 | [41.85%, 55.31%] | -9.15% |
| **XGBoost Regressor** | Classical ML | **239.19** | **181.36** | **0.795%** | **51.92%** | 0.3138 | [45.16%, 58.62%] | -14.26% |
| **Moving Average (5-day SMA)** | Benchmark | 288.86 | 218.20 | 0.957% | 51.44% | 0.3645 | [44.69%, 58.15%] | -37.99% |
| **ARIMA(1, 1, 1)** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | 0.3138 | [45.16%, 58.62%] | -39.05% |
| **PyTorch LSTM Network** | Deep Learning | 678.33 | 610.56 | 2.627% | **54.81%** | 0.0938 | [48.02%, 61.42%] | -224.04% |

---

## 🔬 Scientific Findings & Key Insights

1. **Why the Naive Persistence Baseline Achieves the Best Level RMSE:** In daily financial time series, broad equity indices behave as **near-martingales**:
   $$\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$$
   Under the Efficient Market Hypothesis (EMH), today's closing price reflects available aggregate information, making it the minimum-variance quadratic estimator of tomorrow's price level. The Naive baseline achieves the lowest RMSE (**209.33**), and models predicting price changes incur variance penalty on sideways days.
2. **Directional Testing Realism:** Over the 208 out-of-sample trading days, XGBoost achieved 51.92% directional accuracy ($p = 0.3138$) and PyTorch LSTM achieved 54.81% ($p = 0.0938$). Neither model statistically beats a 50% random walk at $\alpha = 0.05$. Transparently reporting this reflects professional scientific rigor.
3. **Scientific Reality of Deep Learning (LSTM):** The PyTorch LSTM network yielded an RMSE of **678.33**—substantially worse than the persistence baseline (209.33). Non-stationarity and error drift across sliding windows make raw price-level forecasting inherently difficult for deep neural networks, providing empirical validation for why quantitative finance targets stationary returns rather than raw price levels.
4. **Broad-Market Co-Movement:** Sourcing from both NSE and BSE revealed a price correlation of **0.9999** and return correlation of **0.9969**, demonstrating **strong common market dynamics** between the broad-market indices across both major Indian exchanges.

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
```bash
python src/pipeline.py
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`. Navigate to:
- **🔮 Future Horizon Forecaster** to test real model rollouts ($T+1$ to $T+30$).
- **🖥️ Executive Presentation Deck** to view the interactive presentation slides directly in the browser!

### 5. Run the Jupyter Notebook
The fully executed, publication-ready notebook is located at:
`notebooks/02_nifty500_prediction_pipeline.ipynb`
```bash
jupyter notebook notebooks/02_nifty500_prediction_pipeline.ipynb
```

---

## 📂 Repository Directory Structure
```
Stock_Market_Prediction/
├── app/
│   └── app.py                     # Streamlit 8-page interactive dashboard & slide deck
├── data/
│   ├── NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv # Official NSE download
│   ├── raw/                       # Normalized 5-year OHLCV CSVs (NSE & BSE)
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
│   ├── final_findings_report.md   # Final findings report
│   ├── presentation_deck.md       # 10-slide presentation deck (Markdown)
│   └── presentation_deck.html     # Standalone HTML slide presentation
├── src/
│   ├── data_collection.py         # Authoritative official ingestion script
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
