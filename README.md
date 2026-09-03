# 📈 Nifty 500 Stock Market Prediction Pipeline & Analytics Dashboard
### Data Analytics Intern Project · Product Requirements Document (PRD v1.1)
**Author:** Data Analytics Intern | **Project Status:** PRD Functionality Implemented; Naive-Baseline Performance Target Not Achieved | **Date:** September 2026

---

## 🚀 Project Overview
This repository contains an end-to-end quantitative analytics and predictive modeling system for the **Nifty 500 index**, the broad-market benchmark representing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**. 

The project strictly follows the **Data Analytics Intern PRD v1.1**, covering the full quantitative lifecycle:
1. **Authoritative Data Sourcing:** Five years of daily market data (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions) sourced directly from the **official NSE historical index archive** (`data/NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv`) with the complete required schema: `Date, Open, High, Low, Close, Volume, Adjusted Close`.
2. **Cross-Exchange Proxy Validation:** Sourced the **BSE 500 index** (`BSE-500.BO`) as a cross-exchange broad-market proxy for reconciliation and co-movement validation across 1,229 common trading sessions (Price correlation: **0.9999**, Return correlation: **0.9969**). BSE-500 is documented as a secondary market proxy rather than an authoritative Nifty 500 duplicate.
3. **Trading Calendar Integrity & Outlier Handling:** Strict adherence to the official exchange trading calendar (~250 trading days/year), ensuring zero artificial weekend or holiday observations. Post-cleaning missing data: **0.00%** (PRD Target: $< 2\%$). Statistical return outliers ($|Z| > 5$) were investigated and verified against official exchange records as legitimate macroeconomic market shocks (Russia-Ukraine war outbreak and 2024 General Election Results day) and retained to prevent downside risk censorship bias.
4. **Feature Engineering:** 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, rolling volatility, price/return lags) across 1,040 feature rows with zero lookahead bias.
5. **Multi-Family Predictive Modeling:**
   - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) & 5-Day Moving Average
   - **Statistical Model:** ARIMA(1, 1, 1) with pure walk-forward rolling extend (`model.extend()`) and zero lookahead leakage
   - **Classical ML:** Random Forest & XGBoost Regressors with feature importances
   - **Deep Learning:** PyTorch Long Short-Term Memory (LSTM) Neural Network
6. **5-Fold Time-Series Cross-Validation:** Expanding-window cross-validation (`TimeSeriesSplit(n_splits=5)`) implemented to evaluate out-of-sample stability across rolling periods per PRD Section 12.
7. **Transparent Performance Assessment:** Honestly documented that **the PRD performance goal of outperforming the naive baseline was not achieved**, as the **Naive Persistence Baseline achieved the lowest level-price RMSE (209.33)** under the Martingale property of asset prices.
8. **Explicit Forecast Horizon:** Clarified that the historical backtesting target is 1-step ahead ($T+1$) daily close, while the forward forecasting horizon is user-selectable from $T+1$ to $T+30$ trading days driven by actual recursive model rollouts.
9. **Interactive Streamlit Dashboard & Presentation Deck:** Deployed a full 8-page interactive web dashboard (`app/app.py`), a 10-slide presentation deck (`reports/presentation_deck.md` & `reports/presentation_deck.html`), and an executed Jupyter Notebook (`notebooks/02_nifty500_prediction_pipeline.ipynb`).

---

## 🏆 Model Performance Benchmark Scorecard (PRD FR7)

Evaluated strictly on the out-of-sample test split (**October 28, 2025 to August 28, 2026**, 208 trading sessions):

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate | Binomial Test ($p$-val) | 95% Wilson CI | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **209.33** | **151.88** | **0.669%** | **N/A** | **N/A** | **N/A** | **+0.00% (Best RMSE)** |
| **Random Forest Regressor** | Classical ML | **228.48** | **169.05** | **0.742%** | 48.56% | 0.6862 | [41.85%, 55.31%] | -9.15% |
| **XGBoost Regressor** | Classical ML | **239.19** | **181.36** | **0.795%** | **51.92%** | 0.3138 | [45.16%, 58.62%] | -14.26% |
| **Moving Average (5-day SMA)** | Benchmark | 288.86 | 218.20 | 0.957% | 51.44% | 0.3645 | [44.69%, 58.15%] | -37.99% |
| **ARIMA(1, 1, 1) Walk-Forward** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | 0.3138 | [45.16%, 58.62%] | -39.05% |
| **PyTorch LSTM Network** | Deep Learning | 563.46 | 425.56 | 1.877% | **56.73%** | 0.0305 | [49.94%, 63.28%] | -169.17% |

---

## 🔄 5-Fold Time-Series Cross-Validation (PRD Section 12)

| Model Architecture | Folds | CV Mean RMSE (Points) | CV Std RMSE (Points) | CV Mean MAE (Points) | CV Std MAE (Points) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | 5 | **1094.54** | ±779.47 | 849.42 | ±626.72 |
| **XGBoost Regressor** | 5 | **1067.78** | ±828.08 | 832.33 | ±676.44 |

---

## 🔬 Scientific Findings & Key Insights

1. **Why the Naive Persistence Baseline Achieves the Best Level RMSE:** In daily financial time series, broad equity indices behave as **near-martingales**:
   $$\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$$
   Under the Efficient Market Hypothesis (EMH), today's closing price reflects available aggregate information, making it the minimum-variance quadratic estimator of tomorrow's price level. The Naive baseline achieves the lowest RMSE (**209.33**). Models predicting price changes incur variance penalties on sideways days, which is why beating persistence on RMSE is structurally difficult.
2. **Directional Accuracy for Naive Baseline:** The naive model predicts zero price movement ($P_{t+1} = P_t$), not a directional signal. Displaying its directional accuracy as **`N/A`** avoids the false implication that it predicted opposite market movements.
3. **Scientific Reality of Deep Learning (LSTM):** While the PyTorch LSTM captured sequential trend momentum (**56.73% directional hit rate**, $p = 0.0305$), its level-price RMSE of **563.46** was 2.7× worse than the naive baseline. Non-stationarity and error drift across sliding windows make raw price-level forecasting inherently difficult for neural networks, providing empirical validation for why quantitative finance targets stationary returns rather than raw price levels.
4. **Outlier Event Validation:** Extreme return events (e.g. 2022-02-24 Russia-Ukraine outbreak -5.04% and 2024-06-04 General Election counting day -6.76%) were verified against official exchange event records and preserved in the dataset to prevent censorship bias in downside risk estimation.

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
│   ├── outlier_investigation.csv  # Verified market shock event log
│   └── feature_importances.csv    # Tree feature importances
├── notebooks/
│   ├── 01_project_setup.ipynb     # Setup & environment verification
│   └── 02_nifty500_prediction_pipeline.ipynb  # Comprehensive executed notebook
├── presentation/
│   ├── nifty500_final_presentation.pptx # 11-Slide Executive PowerPoint Deck
│   ├── presentation_deck.html     # Standalone interactive HTML presentation
│   ├── presentation_deck.md       # Presentation slides markdown
│   └── final_findings_report.md   # Comprehensive executive findings report
├── reports/
│   ├── final_findings_report.md   # Final findings report
│   ├── presentation_deck.md       # Presentation deck (Markdown)
│   └── presentation_deck.html     # Standalone HTML slide presentation
├── src/
│   ├── data_collection.py         # Authoritative official ingestion script
│   ├── data_preprocessing.py      # Cleaning & outlier validation
│   ├── feature_engineering.py     # Technical indicator engineering
│   ├── evaluate.py                # Metrics, CV & Binomial hypothesis testing
│   ├── generate_pptx.py           # Programmatic PowerPoint presentation generator
│   ├── pipeline.py                # End-to-end automated pipeline
│   └── models/
│       ├── baseline.py            # Naive & Moving Average baselines
│       ├── arima_model.py         # Statistical ARIMA(1,1,1) & walk-forward extend
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
