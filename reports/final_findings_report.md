# 📈 Nifty 500 Stock Market Prediction Pipeline: Final Findings & Technical Report

### Data Analytics Intern Capstone Project · Product Requirements Document (PRD v1.1)
**Author:** Data Analytics Intern | **Target Index:** Nifty 500 (NSE) | **Date:** September 2026

---

## 1. Executive Summary

This project delivers an end-to-end quantitative analytics and predictive modeling system for the **Nifty 500 index**, the broad-market benchmark capturing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**.

Covering a strict five-year historical period (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions), the system integrates data validation, exchange trading calendar reconciliation, quantitative feature engineering, multi-family time-series modeling (Naive Baseline, 5-Day SMA, Walk-Forward ARIMA, Random Forest, XGBoost, and PyTorch LSTM), backtesting, and deployment via an 8-page interactive Streamlit dashboard.

### Key Finding:
- **The Naive Persistence Baseline achieved the lowest out-of-sample level-price RMSE (209.33 points, MAE 151.88, MAPE 0.669%)**, outperforming all feature-based and deep learning architectures on level RMSE.
- Under the **Martingale Property of Asset Prices** and the **Efficient Market Hypothesis (EMH)**:
  $$\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$$
  Today's closing price is the minimum-variance quadratic estimator of tomorrow's price level. Models forecasting directional deltas incur variance penalties on sideways days, demonstrating why empirical quantitative finance targets stationary returns rather than raw price levels.

---

## 2. Problem Statement

Forecasting financial equity index price levels is one of the most challenging problems in quantitative machine learning due to:
1. **High Noise-to-Signal Ratio:** Short-term fluctuations are predominantly driven by stochastic market noise and microstructural order flow.
2. **Non-Stationarity:** Macro trends, monetary regime shifts, and economic growth induce time-varying means and variances.
3. **Severe Lookahead Bias Risks:** Improper scaling, indicator leakage, or random data shuffling produce misleadingly optimistic backtests that fail out-of-sample.

---

## 3. Project Objective

To formulate, validate, and execute an empirical pipeline that predicts the **next trading day's closing price ($P_{t+1}$)** of the Nifty 500 index using 5 years of daily data, comparing:
- **Baseline Benchmarks:** Naive Persistence and 5-Day Simple Moving Average.
- **Statistical Econometrics:** Autoregressive Integrated Moving Average (ARIMA).
- **Classical Machine Learning:** Random Forest Regressor and XGBoost Regressor.
- **Deep Learning:** PyTorch Long Short-Term Memory (LSTM) Neural Network.

---

## 4. Data Sources & Architecture

Two distinct authoritative datasets covering identical trading dates (**1 September 2021 to 31 August 2026**) were utilized:

1. **Primary Dataset — NSE NIFTY 500:**
   - Source: Official National Stock Exchange historical index archive (`data/raw/nse_nifty500_raw.csv`).
   - Observations: 1,240 daily records.
   - Schema: `Date, Open, High, Low, Close`.
   - Role: Sole primary dataset for feature engineering, model training, and forecasting.
2. **Secondary Dataset — BSE 500 Proxy:**
   - Source: Bombay Stock Exchange broad-market index (`data/raw/bse_500_raw.csv`).
   - Observations: 1,240 daily records.
   - Schema: `Date, Open, High, Low, Close, Points Change, Change %, Volume, Turnover, P/E, P/B, Div Yield`.
   - Role: Cross-market reference proxy for calendar verification, macro reconciliation, and volume dynamics.
   - **Critical Rule:** The two index series were kept strictly separated and never merged or price-averaged.

---

## 5. Data Cleaning & Outlier Audit

- **Calendar Integrity:** Strict adherence to exchange trading sessions (~250 trading days/year). Zero synthetic holidays or weekend insertions. Post-cleaning missing OHLC values: **0.00%** (PRD target < 2%).
- **BSE Field Conversions:** Converted 23 string `'-'` volume placeholders to numeric floats and performed forward-fills on isolated non-trading special session entries.
- **Outlier Investigation ($|Z| > 3.5$):**
  - **2022-02-24 (-5.04%):** Russia-Ukraine war outbreak (verified cross-market shock).
  - **2024-06-04 (-6.76%):** Lok Sabha General Election counting day volatility.
  - **2024-06-05 (+3.59%):** Coalition clarity rebound rally.
  - **Verdict:** All extreme return movements were audited against official exchange records and retained to prevent downside risk censorship bias.

---

## 6. Exploratory Data Analysis (EDA)

- **Price Trajectory:** Sourced at 14,551.35 on Sep 1, 2021 and concluded at 23,450.35 on Aug 31, 2026 (+61.16% total gain).
- **Trading Sessions:** 682 Up days (55.0%), 557 Down days. Mean daily return: +0.043% (Annualized: ~11.26%).
- **Stylized Facts:**
  - **Skewness (-0.678) & Kurtosis (4.585):** Significant negative asymmetry and heavy tails.
  - **Jarque-Bera Test ($p < 10^{-250}$):** Strict rejection of normality.
  - **Volatility Clustering:** Mandelbrot volatility clustering observed with 20-day annualized volatility spanning from 4.31% to 32.14% (mean 13.40%).
- **Seasonality ANOVA:** Day-of-week return differences yielded an ANOVA $p$-value of **0.3120** (statistically insignificant, confirming weak-form market efficiency).
- **Cross-Market Co-Movement:** 5-year price correlation of **0.99989** and return correlation of **0.99930** between NSE Nifty 500 and BSE 500.

---

## 7. Feature Engineering

Thirty-four technical and statistical indicators were engineered without lookahead bias:
- **Trend Moving Averages:** SMA (10, 20, 50, 200), EMA (12, 26), and distance-to-mean ratios.
- **Momentum:** 14-day Wilder RSI, MACD line, 9-day signal line, MACD histogram.
- **Volatility:** Bollinger Bands (20-day, $\pm 2\sigma$), %B, Bandwidth, 10-day & 20-day rolling annualized volatility.
- **Lags & Ratios:** Return lags ($t-1, t-2, t-3, t-5$), Price lags ($t-1, t-2$), Intraday High/Low and Close/Open ratios.
- **Target Variable:** $\text{Target} = \text{Close}_{t+1}$.
- **Warm-Up Cutoff:** Dropped the first 200 trading days to allow 200-day SMA initialization, yielding 1,040 clean modeling rows.

---

## 8. Modeling Framework

- **Time-Based Split:** 832 training sessions (Jun 2022 – Oct 2025) and 208 held-out testing sessions (Oct 28, 2025 – Aug 28, 2026, 20.0% split). Zero random shuffling.
- **Statistical Model:** Walk-forward ARIMA(1, 1, 1) rolling one step at a time across the test set.
- **Classical ML:** Random Forest Regressor (150 trees, depth 8) and XGBoost Regressor (150 estimators, learning rate 0.03) with 5-fold expanding window cross-validation.
- **Deep Learning:** PyTorch stacked LSTM (2 layers, 64 hidden units, 20-day sequence lookback, dropout 0.2, Adam optimizer, early stopping). Scalers fitted strictly on training data.

---

## 9. Model Evaluation Scorecard

Evaluated on the out-of-sample test split (208 trading sessions):

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | Benchmark | **209.33** | **151.88** | **0.669%** | **N/A** | **+0.00% (Best)** |
| **Random Forest Regressor** | Classical ML | **229.66** | **167.47** | **0.735%** | 49.52% | -9.71% |
| **Moving Average (5-Day SMA)** | Benchmark | 288.87 | 218.16 | 0.957% | 51.92% | -37.99% |
| **ARIMA(1, 1, 1) Walk-Forward** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | -39.05% |
| **XGBoost Regressor** | Classical ML | 299.91 | 239.62 | 1.044% | 51.92% | -43.27% |
| **PyTorch LSTM Network** | Deep Learning | 560.27 | 453.11 | 1.974% | **52.88%** | -167.64% |

---

## 10. Backtesting Results

- **Tracking Dynamics:** Random Forest and ARIMA track broad multi-week market swings effectively but lag sharp inflection points by 1–2 sessions.
- **Persistence Superiority:** Naive persistence incurs minimal error because the daily standard deviation of Nifty 500 returns is ~0.89%. Sideways days penalize complex models that attempt to predict non-existent directional drifts.

---

## 11. Final Model Selection

- **Primary Benchmark:** **Naive Persistence Baseline** is selected as the primary reference model for price level estimation ($P_{t+1} = P_t$).
- **Primary Feature-Driven Model:** **Random Forest Regressor** is selected as the best machine learning model, achieving an RMSE of 229.66 points (only 9.7% behind the theoretical persistence limit) and demonstrating stable 5-fold CV performance ($1059.38 \pm 777.93$).

---

## 12. Future Forward Forecast (Phase 47)

Recursive forward projections from August 31, 2026 ($P_0 = 23,450.35$):
- **T+1 (2026-09-01):** Random Forest: **23,321.96** | 95% Confidence Interval: **[22,911.67, 23,732.24]**
- **T+5 (2026-09-07):** Random Forest: **23,322.47** | 95% Confidence Interval: **[22,405.04, 24,239.90]**
- **T+30 (2026-10-12):** Random Forest: **23,325.59** | 95% Confidence Interval: **[21,079.12, 25,572.06]**

---

## 13. Streamlit Application

An interactive 8-page dashboard is deployed locally at `http://localhost:8501`:
1. Executive Overview & Objectives
2. Historical Market Explorer (Interactive Candlesticks & Range Filter)
3. Quantitative EDA & Volatility Clustering
4. Technical Indicators & Feature Importances
5. Model Performance Benchmark Scorecard
6. Actual vs. Predicted Backtesting Visualizer
7. Future Horizon Forecaster ($T+1$ to $T+30$ slider)
8. Executive Presentation Deck (15 Interactive Slides)

---

## 14. Project Limitations & Next Steps

1. **Daily Granularity:** Daily OHLC data cannot capture intraday liquidity shocks or order book imbalances.
2. **Absence of Real-Time Sentiment:** News events and geopolitical headlines drive sudden regime breaks that technical indicators cannot anticipate.
3. **Price Levels vs. Returns:** Raw index levels are non-stationary. Future iterations should frame the problem around stationary log-return forecasting:
   $$r_{t+1} = \ln(P_{t+1} / P_t)$$
   to eliminate sequence drift in deep learning networks.
